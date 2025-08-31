"""
IntelliSupport RAG - OpenAI Embedding Service

This module provides OpenAI embedding generation services with
batch processing, caching, and error handling capabilities.

Author: IntelliSupport Team
Created: 2025-08-31
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
import hashlib
import json

import openai
from openai import AsyncOpenAI
import numpy as np

from src.config.settings import get_settings
from src.utils.exceptions import EmbeddingError, ExternalServiceError

logger = logging.getLogger(__name__)
settings = get_settings()


class OpenAIEmbedder:
    """
    OpenAI embedding service with advanced features.
    
    Provides embedding generation with batch processing, caching,
    retry logic, and comprehensive error handling for production use.
    """
    
    def __init__(self):
        """Initialize OpenAI embedding service."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.batch_size = settings.embedding_batch_size
        self.cache: Dict[str, List[float]] = {}  # Simple in-memory cache
        
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot generate embedding for empty text")
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            logger.debug(f"Cache hit for embedding: {text[:50]}...")
            return self.cache[cache_key]
        
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Generate embedding
            response = await self.client.embeddings.create(
                model=self.model,
                input=cleaned_text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Cache the result
            self.cache[cache_key] = embedding
            
            logger.debug(f"Generated embedding for text: {text[:50]}... (dimension: {len(embedding)})")
            return embedding
            
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise EmbeddingError("Rate limit exceeded. Please try again later.")
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise ExternalServiceError("OpenAI", str(e), e)
            
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            EmbeddingError: If batch embedding generation fails
        """
        if not texts:
            return []
        
        embeddings = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                batch_embeddings = await self._process_batch(batch)
                embeddings.extend(batch_embeddings)
                
                # Add small delay between batches to respect rate limits
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Failed to process batch {i//self.batch_size + 1}: {e}")
                raise EmbeddingError(f"Batch embedding generation failed: {str(e)}")
        
        logger.info(f"Generated {len(embeddings)} embeddings in {len(texts)//self.batch_size + 1} batches")
        return embeddings
    
    async def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a single batch of texts."""
        # Check cache for each text
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(self._preprocess_text(text))
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=uncached_texts,
                    encoding_format="float"
                )
                
                # Fill in the embeddings and cache them
                for i, embedding_data in enumerate(response.data):
                    embedding = embedding_data.embedding
                    original_index = uncached_indices[i]
                    embeddings[original_index] = embedding
                    
                    # Cache the result
                    cache_key = self._get_cache_key(texts[original_index])
                    self.cache[cache_key] = embedding
                    
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
                raise
        
        return embeddings
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding generation."""
        # Clean text
        cleaned = text.strip()
        
        # Truncate if too long (OpenAI has token limits)
        max_chars = 8000  # Conservative limit
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + "..."
            logger.warning(f"Text truncated to {max_chars} characters for embedding")
        
        return cleaned
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Create hash of text + model for caching
        content = f"{self.model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get embedding service statistics.
        
        Returns:
            Dict[str, Any]: Service statistics including cache size and model info
        """
        return {
            "model": self.model,
            "cache_size": len(self.cache),
            "batch_size": self.batch_size,
            "supported_languages": ["en", "hi", "ta"],
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")


# Global embedder instance
embedder = OpenAIEmbedder()
