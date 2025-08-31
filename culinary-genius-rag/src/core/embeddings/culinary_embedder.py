"""
CulinaryGenius RAG - Specialized Culinary Embedder

This module provides embedding generation specifically optimized for
culinary content, including recipes, ingredients, and cooking techniques.

Author: CulinaryGenius Team
Created: 2025-08-31
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import openai
from openai import AsyncOpenAI
import numpy as np

from src.config.settings import get_settings
from src.utils.culinary_text_processor import culinary_processor, Recipe, Ingredient

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class CulinaryEmbedding:
    """Represents an embedding with culinary-specific metadata."""
    content: str
    embedding: List[float]
    content_type: str  # "recipe", "technique", "ingredient", "nutrition"
    cuisine: Optional[str] = None
    difficulty: Optional[str] = None
    dietary_tags: List[str] = None
    cooking_methods: List[str] = None
    prep_time: Optional[int] = None
    cook_time: Optional[int] = None


class CulinaryEmbedder:
    """
    Specialized embedding service for culinary content.
    
    Generates embeddings optimized for culinary queries with enhanced
    preprocessing for recipes, ingredients, and cooking techniques.
    """
    
    def __init__(self):
        """Initialize the culinary embedder."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.batch_size = settings.embedding_batch_size
        self.processor = culinary_processor
        
        # Embedding cache for performance
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    async def embed_recipe(self, recipe: Recipe) -> CulinaryEmbedding:
        """
        Generate embedding for a complete recipe.
        
        Args:
            recipe: Structured recipe object
            
        Returns:
            CulinaryEmbedding: Recipe embedding with metadata
        """
        # Create comprehensive text representation
        recipe_text = self._recipe_to_text(recipe)
        
        # Generate embedding
        embedding = await self._generate_embedding(recipe_text)
        
        # Extract cooking methods from instructions
        cooking_methods = []
        for instruction in recipe.instructions:
            methods = self.processor.extract_cooking_techniques(instruction)
            cooking_methods.extend(methods)
        
        return CulinaryEmbedding(
            content=recipe_text,
            embedding=embedding,
            content_type="recipe",
            cuisine=recipe.cuisine,
            difficulty=recipe.difficulty,
            dietary_tags=recipe.dietary_tags or [],
            cooking_methods=list(set(cooking_methods)),
            prep_time=recipe.prep_time,
            cook_time=recipe.cook_time
        )
    
    async def embed_cooking_technique(self, technique_name: str, description: str) -> CulinaryEmbedding:
        """
        Generate embedding for a cooking technique.
        
        Args:
            technique_name: Name of the cooking technique
            description: Detailed description of the technique
            
        Returns:
            CulinaryEmbedding: Technique embedding with metadata
        """
        # Create enhanced text for technique
        technique_text = f"Cooking Technique: {technique_name}\n\nDescription: {description}"
        
        # Add related keywords for better retrieval
        related_keywords = self._get_technique_keywords(technique_name)
        if related_keywords:
            technique_text += f"\n\nRelated: {', '.join(related_keywords)}"
        
        embedding = await self._generate_embedding(technique_text)
        
        return CulinaryEmbedding(
            content=technique_text,
            embedding=embedding,
            content_type="technique",
            cooking_methods=[technique_name.lower()]
        )
    
    async def embed_ingredient_info(self, ingredient_name: str, info: Dict[str, Any]) -> CulinaryEmbedding:
        """
        Generate embedding for ingredient information.
        
        Args:
            ingredient_name: Name of the ingredient
            info: Ingredient information (nutrition, substitutes, etc.)
            
        Returns:
            CulinaryEmbedding: Ingredient embedding with metadata
        """
        # Create comprehensive ingredient text
        ingredient_text = f"Ingredient: {ingredient_name}\n\n"
        
        if info.get('description'):
            ingredient_text += f"Description: {info['description']}\n\n"
        
        if info.get('substitutes'):
            ingredient_text += f"Substitutes: {', '.join(info['substitutes'])}\n\n"
        
        if info.get('storage'):
            ingredient_text += f"Storage: {info['storage']}\n\n"
        
        if info.get('season'):
            ingredient_text += f"Best Season: {info['season']}\n\n"
        
        if info.get('nutrition'):
            nutrition = info['nutrition']
            ingredient_text += f"Nutrition (per 100g): "
            ingredient_text += f"Calories: {nutrition.get('calories', 'N/A')}, "
            ingredient_text += f"Protein: {nutrition.get('protein', 'N/A')}g, "
            ingredient_text += f"Carbs: {nutrition.get('carbs', 'N/A')}g, "
            ingredient_text += f"Fat: {nutrition.get('fat', 'N/A')}g"
        
        embedding = await self._generate_embedding(ingredient_text)
        
        return CulinaryEmbedding(
            content=ingredient_text,
            embedding=embedding,
            content_type="ingredient",
            dietary_tags=info.get('dietary_tags', [])
        )
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embeddings
        """
        embeddings = []
        
        # Process in batches to respect rate limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                # Check cache first
                batch_embeddings = []
                uncached_texts = []
                uncached_indices = []
                
                for j, text in enumerate(batch):
                    if text in self._embedding_cache:
                        batch_embeddings.append(self._embedding_cache[text])
                        self._cache_hits += 1
                    else:
                        batch_embeddings.append(None)
                        uncached_texts.append(text)
                        uncached_indices.append(j)
                        self._cache_misses += 1
                
                # Generate embeddings for uncached texts
                if uncached_texts:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=uncached_texts
                    )
                    
                    # Cache and insert embeddings
                    for idx, embedding_data in enumerate(response.data):
                        text = uncached_texts[idx]
                        embedding = embedding_data.embedding
                        
                        self._embedding_cache[text] = embedding
                        batch_embeddings[uncached_indices[idx]] = embedding
                
                embeddings.extend(batch_embeddings)
                
                # Rate limiting
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i//self.batch_size + 1}: {e}")
                # Add empty embeddings for failed batch
                embeddings.extend([[] for _ in batch])
        
        return embeddings
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        # Check cache first
        if text in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[text]
        
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Cache the embedding
            self._embedding_cache[text] = embedding
            self._cache_misses += 1
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def _recipe_to_text(self, recipe: Recipe) -> str:
        """Convert recipe object to optimized text for embedding."""
        text_parts = []
        
        # Title and basic info
        text_parts.append(f"Recipe: {recipe.title}")
        
        if recipe.cuisine:
            text_parts.append(f"Cuisine: {recipe.cuisine}")
        
        if recipe.difficulty:
            text_parts.append(f"Difficulty: {recipe.difficulty}")
        
        if recipe.dietary_tags:
            text_parts.append(f"Dietary: {', '.join(recipe.dietary_tags)}")
        
        # Timing information
        if recipe.prep_time:
            text_parts.append(f"Prep Time: {recipe.prep_time} minutes")
        
        if recipe.cook_time:
            text_parts.append(f"Cook Time: {recipe.cook_time} minutes")
        
        if recipe.servings:
            text_parts.append(f"Serves: {recipe.servings}")
        
        # Ingredients
        if recipe.ingredients:
            ingredients_text = "Ingredients: " + ", ".join([
                f"{ing.quantity or ''} {ing.unit or ''} {ing.name} {ing.preparation or ''}".strip()
                for ing in recipe.ingredients
            ])
            text_parts.append(ingredients_text)
        
        # Instructions
        if recipe.instructions:
            instructions_text = "Instructions: " + " ".join([
                f"Step {i+1}: {instruction}"
                for i, instruction in enumerate(recipe.instructions)
            ])
            text_parts.append(instructions_text)
        
        return "\n\n".join(text_parts)
    
    def _get_technique_keywords(self, technique_name: str) -> List[str]:
        """Get related keywords for a cooking technique."""
        technique_keywords = {
            "sautéing": ["pan", "oil", "high heat", "quick cooking"],
            "braising": ["slow cooking", "liquid", "covered", "tender"],
            "grilling": ["barbecue", "char", "outdoor", "high heat"],
            "steaming": ["moist heat", "healthy", "vegetables", "gentle"],
            "roasting": ["oven", "dry heat", "browning", "caramelization"],
            "frying": ["oil", "crispy", "golden", "hot oil"],
            "baking": ["oven", "dry heat", "bread", "pastry"],
            "poaching": ["gentle", "liquid", "delicate", "low temperature"]
        }
        
        return technique_keywords.get(technique_name.lower(), [])
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics."""
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "model": self.model,
            "cache_size": len(self._embedding_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "batch_size": self.batch_size
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Embedding cache cleared")


# Global embedder instance
culinary_embedder = CulinaryEmbedder()
