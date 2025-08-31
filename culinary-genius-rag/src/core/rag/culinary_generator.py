"""
CulinaryGenius RAG - Specialized Culinary Response Generator

This module generates intelligent culinary responses with personalized
recipe recommendations, cooking guidance, and nutritional insights.

Author: CulinaryGenius Team
Created: 2025-08-31
"""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import openai
from openai import AsyncOpenAI

from src.config.settings import get_settings
from src.core.rag.culinary_retriever import CulinaryRetrievalResult, CulinaryQuery

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class CulinaryResponse:
    """Represents a generated culinary response with metadata."""
    response: str
    confidence_score: float
    response_type: str  # "recipe", "technique", "substitution", "nutrition", "general"
    recipes_included: int
    techniques_mentioned: List[str]
    ingredients_discussed: List[str]
    nutritional_highlights: List[str]
    cultural_context: Optional[str]
    cooking_tips: List[str]
    difficulty_assessment: Optional[str]
    estimated_time: Optional[Dict[str, int]]  # prep_time, cook_time
    token_usage: Dict[str, int]
    citations: List[str]


class CulinaryResponseGenerator:
    """
    Specialized response generator for culinary content.
    
    Creates personalized, informative, and actionable culinary responses
    with recipe recommendations, cooking guidance, and cultural insights.
    """
    
    def __init__(self):
        """Initialize the culinary response generator."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = settings.response_temperature
        self.max_tokens = settings.max_response_length
        
        # Load culinary prompt templates
        self.prompt_templates = self._load_culinary_prompts()
    
    async def generate_culinary_response(
        self,
        query: CulinaryQuery,
        retrieved_content: List[CulinaryRetrievalResult]
    ) -> CulinaryResponse:
        """
        Generate a comprehensive culinary response.
        
        Args:
            query: User's culinary query with preferences
            retrieved_content: Retrieved relevant culinary content
            
        Returns:
            CulinaryResponse: Generated response with culinary metadata
        """
        logger.info(f"🍳 Generating culinary response for: {query.text}")
        
        try:
            # Determine response type
            response_type = self._determine_response_type(query.text)
            
            # Select appropriate prompt template
            template = self._select_prompt_template(response_type, query)
            
            # Build context from retrieved content
            context = self._build_culinary_context(retrieved_content, query)
            
            # Generate response
            system_prompt, user_prompt = self._build_prompts(template, context, query)
            
            response = await self._call_llm(system_prompt, user_prompt)
            generated_text = response.choices[0].message.content
            
            # Extract culinary metadata from response
            metadata = self._extract_culinary_metadata(generated_text, retrieved_content)
            
            # Calculate confidence score
            confidence = self._calculate_culinary_confidence(query, generated_text, retrieved_content)
            
            # Generate citations
            citations = self._generate_culinary_citations(retrieved_content)
            
            result = CulinaryResponse(
                response=generated_text,
                confidence_score=confidence,
                response_type=response_type,
                recipes_included=metadata["recipes_count"],
                techniques_mentioned=metadata["techniques"],
                ingredients_discussed=metadata["ingredients"],
                nutritional_highlights=metadata["nutrition"],
                cultural_context=metadata["cultural_context"],
                cooking_tips=metadata["tips"],
                difficulty_assessment=metadata["difficulty"],
                estimated_time=metadata["time_estimate"],
                token_usage=response.usage.model_dump(),
                citations=citations
            )
            
            logger.info(f"✅ Generated {response_type} response (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to generate culinary response: {e}")
            raise
    
    def _determine_response_type(self, query_text: str) -> str:
        """Determine the type of culinary response needed."""
        query_lower = query_text.lower()
        
        if any(word in query_lower for word in ["recipe", "cook", "make", "prepare"]):
            return "recipe"
        elif any(word in query_lower for word in ["technique", "how to", "method"]):
            return "technique"
        elif any(word in query_lower for word in ["substitute", "replace", "instead of"]):
            return "substitution"
        elif any(word in query_lower for word in ["nutrition", "calories", "healthy", "diet"]):
            return "nutrition"
        else:
            return "general"
    
    def _build_culinary_context(
        self, 
        retrieved_content: List[CulinaryRetrievalResult], 
        query: CulinaryQuery
    ) -> str:
        """Build specialized culinary context from retrieved content."""
        if not retrieved_content:
            return "No specific culinary information found in the knowledge base."
        
        context_parts = []
        
        # Group content by type
        recipes = [c for c in retrieved_content if c.content_type == "recipe"]
        techniques = [c for c in retrieved_content if c.content_type == "technique"]
        ingredients = [c for c in retrieved_content if c.content_type == "ingredient"]
        
        # Add recipes
        if recipes:
            context_parts.append("🍽️ RELEVANT RECIPES:")
            for i, recipe in enumerate(recipes[:3]):  # Limit to top 3
                context_parts.append(f"Recipe {i+1} (Relevance: {recipe.relevance_score:.2f}):")
                context_parts.append(recipe.content)
                
                # Add metadata
                if recipe.cuisine:
                    context_parts.append(f"Cuisine: {recipe.cuisine}")
                if recipe.difficulty:
                    context_parts.append(f"Difficulty: {recipe.difficulty}")
                if recipe.dietary_tags:
                    context_parts.append(f"Dietary: {', '.join(recipe.dietary_tags)}")
                
                context_parts.append("")  # Empty line
        
        # Add techniques
        if techniques:
            context_parts.append("🔧 RELEVANT COOKING TECHNIQUES:")
            for technique in techniques[:2]:  # Limit to top 2
                context_parts.append(technique.content)
                context_parts.append("")
        
        # Add ingredient information
        if ingredients:
            context_parts.append("🥕 INGREDIENT INFORMATION:")
            for ingredient in ingredients[:2]:  # Limit to top 2
                context_parts.append(ingredient.content)
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _select_prompt_template(self, response_type: str, query: CulinaryQuery) -> Dict[str, str]:
        """Select appropriate prompt template."""
        template_key = f"{response_type}_template"
        
        # Add dietary restriction context to template selection
        if query.dietary_restrictions:
            template_key += "_dietary"
        
        return self.prompt_templates.get(template_key, self.prompt_templates["general_template"])
    
    def _build_prompts(
        self, 
        template: Dict[str, str], 
        context: str, 
        query: CulinaryQuery
    ) -> tuple[str, str]:
        """Build system and user prompts."""
        # Format system prompt
        system_prompt = template["system"].format(
            dietary_restrictions=", ".join(query.dietary_restrictions) if query.dietary_restrictions else "none",
            skill_level=query.skill_level,
            preferred_cuisines=", ".join(query.preferred_cuisines) if query.preferred_cuisines else "any"
        )
        
        # Format user prompt
        user_prompt = template["user"].format(
            context=context,
            query=query.text,
            available_ingredients=", ".join(query.available_ingredients) if query.available_ingredients else "not specified",
            exclude_ingredients=", ".join(query.exclude_ingredients) if query.exclude_ingredients else "none"
        )
        
        return system_prompt, user_prompt
    
    async def _call_llm(self, system_prompt: str, user_prompt: str) -> Any:
        """Call the LLM with culinary-optimized parameters."""
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
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _extract_culinary_metadata(
        self, 
        response_text: str, 
        retrieved_content: List[CulinaryRetrievalResult]
    ) -> Dict[str, Any]:
        """Extract culinary-specific metadata from generated response."""
        # Count recipes mentioned
        recipes_count = len([c for c in retrieved_content if c.content_type == "recipe"])
        
        # Extract techniques mentioned
        techniques = []
        for content in retrieved_content:
            if content.cooking_methods:
                techniques.extend(content.cooking_methods)
        
        # Extract ingredients discussed
        ingredients = []
        response_lower = response_text.lower()
        common_ingredients = [
            "chicken", "beef", "pork", "fish", "rice", "pasta", "tomato", "onion",
            "garlic", "oil", "butter", "salt", "pepper", "herbs", "spices"
        ]
        
        for ingredient in common_ingredients:
            if ingredient in response_lower:
                ingredients.append(ingredient)
        
        # Extract nutritional highlights
        nutrition_keywords = ["protein", "fiber", "vitamins", "minerals", "antioxidants", "healthy"]
        nutrition = [kw for kw in nutrition_keywords if kw in response_lower]
        
        # Extract cooking tips
        tips = []
        if "tip:" in response_lower or "pro tip:" in response_lower:
            tips.append("Professional cooking tips included")
        
        return {
            "recipes_count": recipes_count,
            "techniques": list(set(techniques)),
            "ingredients": list(set(ingredients)),
            "nutrition": nutrition,
            "cultural_context": self._extract_cultural_context(response_text),
            "tips": tips,
            "difficulty": self._extract_difficulty_mention(response_text),
            "time_estimate": self._extract_time_estimates(response_text)
        }
    
    def _extract_cultural_context(self, text: str) -> Optional[str]:
        """Extract cultural context from response."""
        cultural_keywords = ["traditional", "authentic", "originated", "culture", "history"]
        if any(keyword in text.lower() for keyword in cultural_keywords):
            return "Cultural context included"
        return None
    
    def _extract_difficulty_mention(self, text: str) -> Optional[str]:
        """Extract difficulty assessment from response."""
        for level in settings.skill_levels:
            if level in text.lower():
                return level
        return None
    
    def _extract_time_estimates(self, text: str) -> Optional[Dict[str, int]]:
        """Extract time estimates from response."""
        times = culinary_processor.extract_cooking_times(text)
        return times if any(times.values()) else None
    
    def _calculate_culinary_confidence(
        self,
        query: CulinaryQuery,
        response: str,
        retrieved_content: List[CulinaryRetrievalResult]
    ) -> float:
        """Calculate confidence score for culinary response."""
        factors = []
        
        # Factor 1: Average relevance of retrieved content
        if retrieved_content:
            avg_relevance = sum(c.relevance_score for c in retrieved_content) / len(retrieved_content)
            factors.append(avg_relevance)
        else:
            factors.append(0.0)
        
        # Factor 2: Dietary restriction compliance
        if query.dietary_restrictions:
            compliant_content = [c for c in retrieved_content 
                               if any(tag in c.dietary_tags for tag in query.dietary_restrictions)]
            compliance_score = len(compliant_content) / len(retrieved_content) if retrieved_content else 0
            factors.append(compliance_score)
        else:
            factors.append(1.0)
        
        # Factor 3: Response completeness
        response_completeness = min(len(response) / 200, 1.0)
        factors.append(response_completeness)
        
        # Factor 4: Skill level appropriateness
        if query.skill_level:
            appropriate_content = [c for c in retrieved_content 
                                 if not c.difficulty or c.difficulty == query.skill_level]
            skill_score = len(appropriate_content) / len(retrieved_content) if retrieved_content else 0
            factors.append(skill_score)
        else:
            factors.append(1.0)
        
        # Weighted average
        weights = [0.4, 0.2, 0.2, 0.2]
        confidence = sum(f * w for f, w in zip(factors, weights))
        
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_culinary_citations(self, retrieved_content: List[CulinaryRetrievalResult]) -> List[str]:
        """Generate culinary-specific citations."""
        citations = []
        
        for i, content in enumerate(retrieved_content):
            citation = f"[{i+1}] {content.content_type.title()}"
            
            if content.cuisine:
                citation += f" - {content.cuisine} Cuisine"
            
            if content.difficulty:
                citation += f" (Difficulty: {content.difficulty})"
            
            citation += f" (Relevance: {content.relevance_score:.2f})"
            
            citations.append(citation)
        
        return citations
    
    def _load_culinary_prompts(self) -> Dict[str, Dict[str, str]]:
        """Load specialized culinary prompt templates."""
        return {
            "recipe_template": {
                "system": """You are CulinaryGenius, an expert culinary assistant and chef with deep knowledge of global cuisines, cooking techniques, and nutrition. 

Your expertise includes:
- Recipe development and adaptation
- Cooking techniques from around the world
- Ingredient substitutions and dietary accommodations
- Nutritional guidance and meal planning
- Cultural food traditions and history

User preferences:
- Dietary restrictions: {dietary_restrictions}
- Skill level: {skill_level}
- Preferred cuisines: {preferred_cuisines}

Provide detailed, actionable culinary guidance with:
1. Clear step-by-step instructions
2. Ingredient alternatives when needed
3. Cooking tips and techniques
4. Cultural context when relevant
5. Nutritional insights
6. Difficulty assessment""",
                
                "user": """Based on the following culinary knowledge:

{context}

User Query: {query}
Available Ingredients: {available_ingredients}
Ingredients to Avoid: {exclude_ingredients}

Please provide a comprehensive culinary response that includes:
1. Direct answer to the query
2. Detailed recipe or cooking guidance
3. Alternative ingredients or methods if applicable
4. Cooking tips and techniques
5. Cultural or nutritional context
6. Difficulty level and time estimates"""
            },
            
            "technique_template": {
                "system": """You are CulinaryGenius, a master chef and culinary instructor specializing in cooking techniques and methods from around the world.

Your role is to provide expert guidance on:
- Cooking techniques and methods
- Equipment usage and tips
- Troubleshooting cooking problems
- Professional chef secrets
- Safety and best practices

User skill level: {skill_level}

Provide clear, educational responses with:
1. Step-by-step technique instructions
2. Common mistakes to avoid
3. Equipment recommendations
4. Visual cues and indicators
5. Practice tips for improvement""",
                
                "user": """Cooking technique knowledge:

{context}

User Question: {query}

Please provide expert guidance on this cooking technique, including:
1. Detailed step-by-step instructions
2. Key points for success
3. Common mistakes to avoid
4. Equipment needed
5. Practice recommendations
6. Variations or related techniques"""
            },
            
            "substitution_template": {
                "system": """You are CulinaryGenius, a culinary expert specializing in ingredient substitutions and dietary adaptations.

Your expertise covers:
- Ingredient substitutions for dietary restrictions
- Flavor profile matching
- Texture and cooking behavior analysis
- Nutritional equivalents
- Regional ingredient availability

User dietary restrictions: {dietary_restrictions}

Provide practical substitution advice with:
1. Direct substitution recommendations
2. Ratio and measurement adjustments
3. Flavor impact analysis
4. Cooking method modifications if needed
5. Multiple alternatives when possible""",
                
                "user": """Ingredient and substitution knowledge:

{context}

User Query: {query}
Dietary Restrictions: {dietary_restrictions}
Ingredients to Avoid: {exclude_ingredients}

Please provide detailed substitution guidance including:
1. Best substitute options
2. Measurement conversions
3. Flavor and texture differences
4. Cooking adjustments needed
5. Nutritional comparison
6. Where to find substitutes"""
            },
            
            "general_template": {
                "system": """You are CulinaryGenius, a knowledgeable culinary assistant with expertise in global cuisines, cooking techniques, nutrition, and food culture.

You help users with:
- Recipe recommendations and modifications
- Cooking techniques and troubleshooting
- Ingredient information and substitutions
- Nutritional guidance
- Cultural food knowledge
- Meal planning and preparation

User preferences:
- Dietary restrictions: {dietary_restrictions}
- Skill level: {skill_level}
- Preferred cuisines: {preferred_cuisines}

Always provide helpful, accurate, and actionable culinary advice.""",
                
                "user": """Culinary knowledge base:

{context}

User Question: {query}

Please provide a helpful culinary response based on the available knowledge."""
            }
        }


# Global generator instance
culinary_generator = CulinaryResponseGenerator()
