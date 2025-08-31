"""
IntelliSupport RAG - Response Generation Service

This module implements the response generation functionality using
LLMs with advanced prompt engineering and safety checks.

Author: IntelliSupport Team
Created: 2025-08-31
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re
import asyncio

import openai
from openai import AsyncOpenAI

from src.config.settings import get_settings
from src.core.rag.retriever import RetrievalResult
from src.utils.exceptions import GenerationError, ExternalServiceError
from src.utils.text_processing import text_processor

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class GenerationResult:
    """
    Represents a generated response with metadata.
    
    Contains the generated response along with confidence scores,
    citations, and metadata for quality assessment.
    """
    response: str
    confidence_score: float
    citations: List[str]
    token_usage: Dict[str, int]
    model_used: str
    language: str
    metadata: Dict[str, Any]


class ResponseGenerator:
    """
    Advanced response generation service for IntelliSupport RAG.
    
    Provides context-aware response generation with safety checks,
    citation generation, and multi-language support.
    """
    
    def __init__(self):
        """Initialize response generator."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = settings.response_temperature
        self.max_tokens = settings.max_response_length
        self.max_context_length = settings.max_context_length
        self.include_citations = settings.include_citations
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
    
    async def generate_response(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
        user_context: Optional[Dict[str, Any]] = None,
        language: str = "en"
    ) -> GenerationResult:
        """
        Generate response using retrieved documents and user query.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved relevant documents
            user_context: Additional user context (product, tier, etc.)
            language: Response language (en, hi, ta)
            
        Returns:
            GenerationResult: Generated response with metadata
            
        Raises:
            GenerationError: If response generation fails
        """
        try:
            # Prepare context from retrieved documents
            context = self._prepare_context(retrieved_docs)
            
            # Select appropriate prompt template
            prompt_template = self._select_prompt_template(language, user_context)
            
            # Build the prompt
            system_prompt, user_prompt = self._build_prompt(
                query, context, prompt_template, user_context
            )
            
            # Generate response
            response = await self._call_llm(system_prompt, user_prompt)
            
            # Post-process response
            processed_response = self._post_process_response(response, language)
            
            # Generate citations if enabled
            citations = self._generate_citations(retrieved_docs) if self.include_citations else []
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                query, processed_response, retrieved_docs
            )
            
            result = GenerationResult(
                response=processed_response,
                confidence_score=confidence_score,
                citations=citations,
                token_usage=response.usage.model_dump(),
                model_used=response.model,
                language=language,
                metadata={
                    "num_retrieved_docs": len(retrieved_docs),
                    "context_length": len(context),
                    "query_length": len(query),
                    "user_context": user_context or {}
                }
            )
            
            logger.info(f"Generated response for query: {query[:50]}... (confidence: {confidence_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Response generation failed for query '{query}': {e}")
            raise GenerationError(f"Failed to generate response: {str(e)}")
    
    def _prepare_context(self, retrieved_docs: List[RetrievalResult]) -> str:
        """Prepare context from retrieved documents."""
        if not retrieved_docs:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(retrieved_docs):
            # Add document with source information
            doc_text = f"[Source {i+1}]: {doc.content}"
            
            # Check if adding this document would exceed context length
            if total_length + len(doc_text) > self.max_context_length:
                break
            
            context_parts.append(doc_text)
            total_length += len(doc_text)
        
        return "\n\n".join(context_parts)
    
    def _select_prompt_template(
        self, 
        language: str, 
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Select appropriate prompt template based on language and context."""
        # Default to English template
        template_key = "default_en"
        
        # Select based on language
        if language == "hi":
            template_key = "default_hi"
        elif language == "ta":
            template_key = "default_ta"
        
        # Override based on user context (e.g., premium users get different templates)
        if user_context and user_context.get("user_tier") == "premium":
            template_key = f"premium_{language}"
        
        return self.prompt_templates.get(template_key, self.prompt_templates["default_en"])
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        template: Dict[str, str],
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """Build system and user prompts from template."""
        # Format system prompt
        system_prompt = template["system"].format(
            company_name="IntelliSupport",
            product_name=user_context.get("product", "our platform") if user_context else "our platform"
        )
        
        # Format user prompt
        user_prompt = template["user"].format(
            context=context,
            query=query,
            user_tier=user_context.get("user_tier", "standard") if user_context else "standard"
        )
        
        return system_prompt, user_prompt
    
    async def _call_llm(self, system_prompt: str, user_prompt: str) -> Any:
        """Call the LLM with proper error handling."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            return response
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise GenerationError("Service temporarily unavailable due to high demand. Please try again later.")
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise ExternalServiceError("OpenAI", str(e), e)
            
        except Exception as e:
            logger.error(f"Unexpected error calling LLM: {e}")
            raise GenerationError(f"Failed to generate response: {str(e)}")
    
    def _post_process_response(self, response: Any, language: str) -> str:
        """Post-process the generated response."""
        content = response.choices[0].message.content.strip()
        
        # Basic safety checks
        if not content:
            return "I apologize, but I couldn't generate a helpful response. Please try rephrasing your question."
        
        # Remove any potential harmful content (basic implementation)
        # In production, use more sophisticated content filtering
        harmful_patterns = [
            r'(?i)(api[_\s]?key|password|secret|token)',
            r'(?i)(delete|remove|drop)\s+(database|table|user)',
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, content):
                logger.warning(f"Potentially harmful content detected and filtered")
                return "I cannot provide information that might compromise security. Please contact our support team for assistance."
        
        return content
    
    def _generate_citations(self, retrieved_docs: List[RetrievalResult]) -> List[str]:
        """Generate citations from retrieved documents."""
        citations = []
        
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source', 'Knowledge Base')
            citation = f"[{i+1}] {source}"
            
            # Add relevance score if available
            if doc.relevance_score > 0:
                citation += f" (Relevance: {doc.relevance_score:.2f})"
            
            citations.append(citation)
        
        return citations
    
    def _calculate_confidence_score(
        self,
        query: str,
        response: str,
        retrieved_docs: List[RetrievalResult]
    ) -> float:
        """Calculate confidence score for the generated response."""
        # Simple confidence calculation based on multiple factors
        factors = []
        
        # Factor 1: Average relevance of retrieved documents
        if retrieved_docs:
            avg_relevance = sum(doc.relevance_score for doc in retrieved_docs) / len(retrieved_docs)
            factors.append(avg_relevance)
        else:
            factors.append(0.0)
        
        # Factor 2: Response length (moderate length preferred)
        response_length_score = min(len(response) / 200, 1.0)
        factors.append(response_length_score)
        
        # Factor 3: Query-response keyword overlap
        query_keywords = set(query.lower().split())
        response_keywords = set(response.lower().split())
        if query_keywords:
            overlap_score = len(query_keywords.intersection(response_keywords)) / len(query_keywords)
            factors.append(overlap_score)
        else:
            factors.append(0.0)
        
        # Calculate weighted average
        weights = [0.5, 0.2, 0.3]  # Prioritize document relevance
        confidence = sum(f * w for f, w in zip(factors, weights))
        
        return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
    
    def _load_prompt_templates(self) -> Dict[str, Dict[str, str]]:
        """Load prompt templates for different languages and contexts."""
        return {
            "default_en": {
                "system": """You are IntelliSupport, a helpful and knowledgeable customer support assistant for {company_name}. 
Your role is to provide accurate, helpful, and friendly support to users of {product_name}.

Guidelines:
- Use the provided context to answer questions accurately
- Be concise but comprehensive in your responses
- If you're unsure about something, acknowledge it and suggest contacting human support
- Always maintain a professional and helpful tone
- Include relevant details from the context when applicable
- If the context doesn't contain relevant information, say so clearly""",
                
                "user": """Context from knowledge base:
{context}

User question: {query}

Please provide a helpful response based on the context above. If the context doesn't contain enough information to answer the question, please say so and suggest next steps."""
            },
            
            "default_hi": {
                "system": """आप IntelliSupport हैं, {company_name} के लिए एक सहायक और जानकार ग्राहक सहायता सहायक हैं।
आपका काम {product_name} के उपयोगकर्ताओं को सटीक, उपयोगी और मित्रवत सहायता प्रदान करना है।

दिशानिर्देश:
- प्रदान किए गए संदर्भ का उपयोग करके प्रश्नों का सटीक उत्तर दें
- अपने उत्तरों में संक्षिप्त लेकिन व्यापक रहें
- यदि आप किसी बात के बारे में अनिश्चित हैं, तो इसे स्वीकार करें और मानव सहायता से संपर्क करने का सुझाव दें""",
                
                "user": """ज्ञान आधार से संदर्भ:
{context}

उपयोगकर्ता प्रश्न: {query}

कृपया उपरोक्त संदर्भ के आधार पर एक सहायक उत्तर प्रदान करें।"""
            },
            
            "default_ta": {
                "system": """நீங்கள் IntelliSupport, {company_name} க்கான ஒரு உதவிகரமான மற்றும் அறிவுள்ள வாடிக்கையாளர் ஆதரவு உதவியாளர்.
உங்கள் பங்கு {product_name} பயனர்களுக்கு துல்லியமான, பயனுள்ள மற்றும் நட்பான ஆதரவை வழங்குவதாகும்.

வழிகாட்டுதல்கள்:
- வழங்கப்பட்ட சூழலைப் பயன்படுத்தி கேள்விகளுக்கு துல்லியமாக பதிலளிக்கவும்
- உங்கள் பதில்களில் சுருக்கமாக ஆனால் விரிவாக இருங்கள்
- ஏதேனும் விஷயத்தில் நீங்கள் உறுதியாக இல்லை என்றால், அதை ஒப்புக்கொண்டு மனித ஆதரவைத் தொடர்பு கொள்ள பரிந்துரைக்கவும்""",
                
                "user": """அறிவுத் தளத்திலிருந்து சூழல்:
{context}

பயனர் கேள்வி: {query}

மேலே உள்ள சூழலின் அடிப்படையில் ஒரு உதவிகரமான பதிலை வழங்கவும்."""
            }
        }
    
    async def generate_response(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult],
        user_context: Optional[Dict[str, Any]] = None,
        language: str = "en"
    ) -> GenerationResult:
        """
        Generate response using retrieved documents and user query.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved relevant documents
            user_context: Additional user context
            language: Response language
            
        Returns:
            GenerationResult: Generated response with metadata
        """
        try:
            # Detect language if not provided
            if language == "auto":
                language = text_processor.detect_language(query)
            
            # Prepare context
            context = self._prepare_context(retrieved_docs)
            
            # Build prompts
            system_prompt, user_prompt = self._build_prompts(
                query, context, language, user_context
            )
            
            # Generate response
            response = await self._generate_with_retry(system_prompt, user_prompt)
            
            # Post-process and validate
            processed_response = self._post_process_response(
                response.choices[0].message.content, language
            )
            
            # Generate citations
            citations = self._generate_citations(retrieved_docs)
            
            # Calculate confidence
            confidence = self._calculate_confidence_score(
                query, processed_response, retrieved_docs
            )
            
            result = GenerationResult(
                response=processed_response,
                confidence_score=confidence,
                citations=citations,
                token_usage=response.usage.model_dump(),
                model_used=response.model,
                language=language,
                metadata={
                    "context_docs": len(retrieved_docs),
                    "context_length": len(context),
                    "user_context": user_context or {}
                }
            )
            
            logger.info(f"Response generated successfully (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise GenerationError(f"Failed to generate response: {str(e)}")
    
    def _prepare_context(self, retrieved_docs: List[RetrievalResult]) -> str:
        """Prepare context string from retrieved documents."""
        if not retrieved_docs:
            return "No relevant information available."
        
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(retrieved_docs):
            # Format document with metadata
            doc_header = f"Document {i+1} (Relevance: {doc.relevance_score:.2f}):"
            doc_content = doc.content
            doc_text = f"{doc_header}\n{doc_content}"
            
            # Check context length limit
            if total_length + len(doc_text) > self.max_context_length:
                logger.warning(f"Context truncated at {total_length} characters")
                break
            
            context_parts.append(doc_text)
            total_length += len(doc_text)
        
        return "\n\n".join(context_parts)
    
    def _build_prompts(
        self,
        query: str,
        context: str,
        language: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """Build system and user prompts."""
        template = self.prompt_templates.get(f"default_{language}", self.prompt_templates["default_en"])
        
        # Format system prompt
        system_prompt = template["system"].format(
            company_name="IntelliSupport",
            product_name=user_context.get("product", "our platform") if user_context else "our platform"
        )
        
        # Format user prompt
        user_prompt = template["user"].format(
            context=context,
            query=query
        )
        
        return system_prompt, user_prompt
    
    async def _generate_with_retry(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> Any:
        """Generate response with retry logic."""
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=0.9,
                    frequency_penalty=0.1,
                    presence_penalty=0.1
                )
                
                return response
                
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limit hit, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(1)
                else:
                    raise
    
    def _post_process_response(self, response: str, language: str) -> str:
        """Post-process generated response."""
        if not response:
            return "I apologize, but I couldn't generate a helpful response. Please try rephrasing your question."
        
        # Clean up response
        response = response.strip()
        
        # Add language-specific post-processing
        if language == "hi":
            # Ensure proper Hindi formatting
            response = response.replace("।।", "।")  # Fix double periods
        elif language == "ta":
            # Ensure proper Tamil formatting
            response = response.replace("।।", "।")
        
        return response
    
    def _generate_citations(self, retrieved_docs: List[RetrievalResult]) -> List[str]:
        """Generate citations from retrieved documents."""
        citations = []
        
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source', 'Knowledge Base')
            doc_id = doc.document_id
            relevance = doc.relevance_score
            
            citation = f"[{i+1}] {source} (ID: {doc_id}, Relevance: {relevance:.2f})"
            citations.append(citation)
        
        return citations
    
    def _calculate_confidence_score(
        self,
        query: str,
        response: str,
        retrieved_docs: List[RetrievalResult]
    ) -> float:
        """Calculate confidence score for the response."""
        factors = []
        
        # Factor 1: Average document relevance
        if retrieved_docs:
            avg_relevance = sum(doc.relevance_score for doc in retrieved_docs) / len(retrieved_docs)
            factors.append(avg_relevance)
        else:
            factors.append(0.0)
        
        # Factor 2: Response completeness (length-based heuristic)
        response_completeness = min(len(response) / 100, 1.0)
        factors.append(response_completeness)
        
        # Factor 3: Query-response alignment
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        if query_words:
            alignment = len(query_words.intersection(response_words)) / len(query_words)
            factors.append(alignment)
        else:
            factors.append(0.0)
        
        # Weighted average
        weights = [0.5, 0.2, 0.3]
        confidence = sum(f * w for f, w in zip(factors, weights))
        
        return min(max(confidence, 0.0), 1.0)


# Global generator instance
generator = ResponseGenerator()
