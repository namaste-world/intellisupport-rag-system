"""
CulinaryGenius RAG - Specialized Culinary Retriever

This module implements intelligent retrieval for culinary content with
advanced filtering, ranking, and personalization capabilities.

Author: CulinaryGenius Team
Created: 2025-08-31
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import get_settings
from src.core.embeddings.culinary_embedder import culinary_embedder, CulinaryEmbedding
from src.utils.culinary_text_processor import culinary_processor

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class CulinaryRetrievalResult:
    """Represents a retrieved culinary content with relevance scoring."""
    content: str
    relevance_score: float
    content_type: str
    document_id: str
    metadata: Dict[str, Any]
    cuisine: Optional[str] = None
    difficulty: Optional[str] = None
    dietary_tags: List[str] = None
    cooking_methods: List[str] = None
    prep_time: Optional[int] = None
    cook_time: Optional[int] = None


@dataclass
class CulinaryQuery:
    """Represents a culinary query with user preferences."""
    text: str
    dietary_restrictions: List[str] = None
    preferred_cuisines: List[str] = None
    max_prep_time: Optional[int] = None
    max_cook_time: Optional[int] = None
    skill_level: str = "intermediate"
    available_ingredients: List[str] = None
    exclude_ingredients: List[str] = None
    meal_type: Optional[str] = None  # breakfast, lunch, dinner, snack, dessert


class CulinaryRetriever:
    """
    Advanced retrieval system for culinary content.
    
    Provides intelligent search and filtering capabilities specifically
    designed for recipes, cooking techniques, and food knowledge with
    personalization based on user preferences and constraints.
    """
    
    def __init__(self):
        """Initialize the culinary retriever."""
        self.embedder = culinary_embedder
        self.processor = culinary_processor
        
        # In-memory storage for embeddings (production would use vector DB)
        self.embeddings: List[CulinaryEmbedding] = []
        self.embedding_matrix: Optional[np.ndarray] = None
        
        # Retrieval statistics
        self.retrieval_stats = {
            "total_queries": 0,
            "avg_retrieval_time": 0.0,
            "cache_hits": 0
        }
    
    async def load_culinary_knowledge(self, data_path: str) -> None:
        """
        Load culinary knowledge base from processed data.
        
        Args:
            data_path: Path to processed culinary data directory
        """
        logger.info("🍳 Loading culinary knowledge base...")
        
        data_dir = Path(data_path)
        
        # Load recipes
        recipes_file = data_dir / "recipes.json"
        if recipes_file.exists():
            await self._load_recipes(recipes_file)
        
        # Load cooking techniques
        techniques_file = data_dir / "cooking_techniques.json"
        if techniques_file.exists():
            await self._load_techniques(techniques_file)
        
        # Load ingredient information
        ingredients_file = data_dir / "ingredients.json"
        if ingredients_file.exists():
            await self._load_ingredients(ingredients_file)
        
        # Build embedding matrix for efficient search
        if self.embeddings:
            self.embedding_matrix = np.array([emb.embedding for emb in self.embeddings])
            logger.info(f"✅ Loaded {len(self.embeddings)} culinary embeddings")
        else:
            logger.warning("⚠️ No embeddings loaded")
    
    async def retrieve(
        self,
        query: CulinaryQuery,
        top_k: int = 5,
        method: str = "hybrid"
    ) -> List[CulinaryRetrievalResult]:
        """
        Retrieve relevant culinary content for a query.
        
        Args:
            query: Culinary query with preferences
            top_k: Number of results to return
            method: Retrieval method ("semantic", "keyword", "hybrid")
            
        Returns:
            List[CulinaryRetrievalResult]: Ranked retrieval results
        """
        start_time = time.time()
        self.retrieval_stats["total_queries"] += 1
        
        logger.info(f"🔍 Retrieving culinary content for: {query.text}")
        
        if not self.embeddings:
            logger.warning("No culinary knowledge loaded")
            return []
        
        # Generate query embedding
        query_embedding = await self.embedder._generate_embedding(query.text)
        
        # Calculate semantic similarities
        similarities = cosine_similarity([query_embedding], self.embedding_matrix)[0]
        
        # Apply filters and ranking
        filtered_results = self._apply_culinary_filters(similarities, query)
        
        # Rank results
        ranked_results = self._rank_results(filtered_results, query, method)
        
        # Convert to retrieval results
        results = []
        for idx, score in ranked_results[:top_k]:
            embedding = self.embeddings[idx]
            
            result = CulinaryRetrievalResult(
                content=embedding.content,
                relevance_score=score,
                content_type=embedding.content_type,
                document_id=f"doc_{idx}",
                metadata={
                    "original_similarity": similarities[idx],
                    "boosted_score": score,
                    "rank": len(results) + 1
                },
                cuisine=embedding.cuisine,
                difficulty=embedding.difficulty,
                dietary_tags=embedding.dietary_tags or [],
                cooking_methods=embedding.cooking_methods or [],
                prep_time=embedding.prep_time,
                cook_time=embedding.cook_time
            )
            results.append(result)
        
        # Update statistics
        retrieval_time = time.time() - start_time
        self.retrieval_stats["avg_retrieval_time"] = (
            (self.retrieval_stats["avg_retrieval_time"] * (self.retrieval_stats["total_queries"] - 1) + retrieval_time) /
            self.retrieval_stats["total_queries"]
        )
        
        logger.info(f"✅ Retrieved {len(results)} culinary results in {retrieval_time:.3f}s")
        return results
    
    def _apply_culinary_filters(self, similarities: np.ndarray, query: CulinaryQuery) -> List[Tuple[int, float]]:
        """Apply culinary-specific filters to results."""
        filtered_results = []
        
        for idx, similarity in enumerate(similarities):
            embedding = self.embeddings[idx]
            
            # Skip if similarity too low
            if similarity < 0.1:
                continue
            
            # Apply dietary restriction filters
            if query.dietary_restrictions:
                if not self._matches_dietary_restrictions(embedding, query.dietary_restrictions):
                    continue
            
            # Apply cuisine filters
            if query.preferred_cuisines and embedding.cuisine:
                if embedding.cuisine not in query.preferred_cuisines:
                    similarity *= 0.7  # Reduce score but don't exclude
            
            # Apply time constraints
            if query.max_prep_time and embedding.prep_time:
                if embedding.prep_time > query.max_prep_time:
                    continue
            
            if query.max_cook_time and embedding.cook_time:
                if embedding.cook_time > query.max_cook_time:
                    continue
            
            # Apply skill level filter
            if query.skill_level and embedding.difficulty:
                skill_order = ["beginner", "intermediate", "advanced", "professional"]
                user_level = skill_order.index(query.skill_level)
                content_level = skill_order.index(embedding.difficulty)
                
                if content_level > user_level + 1:  # Allow one level above
                    similarity *= 0.5  # Reduce score for too difficult recipes
            
            # Apply ingredient filters
            if query.exclude_ingredients:
                content_lower = embedding.content.lower()
                if any(ing.lower() in content_lower for ing in query.exclude_ingredients):
                    continue
            
            filtered_results.append((idx, similarity))
        
        return filtered_results
    
    def _matches_dietary_restrictions(self, embedding: CulinaryEmbedding, restrictions: List[str]) -> bool:
        """Check if content matches dietary restrictions."""
        if not embedding.dietary_tags:
            return False
        
        # All restrictions must be satisfied
        for restriction in restrictions:
            if restriction not in embedding.dietary_tags:
                return False
        
        return True
    
    def _rank_results(
        self, 
        filtered_results: List[Tuple[int, float]], 
        query: CulinaryQuery, 
        method: str
    ) -> List[Tuple[int, float]]:
        """Rank filtered results using culinary-specific logic."""
        
        if method == "semantic":
            # Pure semantic similarity
            return sorted(filtered_results, key=lambda x: x[1], reverse=True)
        
        elif method == "hybrid":
            # Boost scores based on culinary factors
            boosted_results = []
            
            for idx, score in filtered_results:
                embedding = self.embeddings[idx]
                boosted_score = score
                
                # Boost for exact cuisine match
                if query.preferred_cuisines and embedding.cuisine in query.preferred_cuisines:
                    boosted_score *= 1.3
                
                # Boost for skill level match
                if query.skill_level and embedding.difficulty == query.skill_level:
                    boosted_score *= 1.2
                
                # Boost for content type preference
                if "recipe" in query.text.lower() and embedding.content_type == "recipe":
                    boosted_score *= 1.2
                elif "technique" in query.text.lower() and embedding.content_type == "technique":
                    boosted_score *= 1.2
                
                # Boost for ingredient availability
                if query.available_ingredients:
                    content_lower = embedding.content.lower()
                    available_count = sum(1 for ing in query.available_ingredients 
                                        if ing.lower() in content_lower)
                    if available_count > 0:
                        boost = 1 + (available_count * 0.1)
                        boosted_score *= boost
                
                boosted_results.append((idx, boosted_score))
            
            return sorted(boosted_results, key=lambda x: x[1], reverse=True)
        
        else:  # keyword method
            # Simple keyword-based ranking
            keyword_results = []
            query_words = set(query.text.lower().split())
            
            for idx, score in filtered_results:
                embedding = self.embeddings[idx]
                content_words = set(embedding.content.lower().split())
                
                # Calculate keyword overlap
                overlap = len(query_words.intersection(content_words))
                keyword_score = overlap / len(query_words) if query_words else 0
                
                # Combine with semantic score
                combined_score = (score * 0.7) + (keyword_score * 0.3)
                keyword_results.append((idx, combined_score))
            
            return sorted(keyword_results, key=lambda x: x[1], reverse=True)
    
    async def _load_recipes(self, recipes_file: Path) -> None:
        """Load recipe embeddings."""
        logger.info("📚 Loading recipe embeddings...")
        
        with open(recipes_file, 'r', encoding='utf-8') as f:
            recipes_data = json.load(f)
        
        for recipe_data in recipes_data:
            # Convert to Recipe object
            recipe = self._dict_to_recipe(recipe_data)
            
            # Generate embedding
            embedding = await self.embedder.embed_recipe(recipe)
            self.embeddings.append(embedding)
    
    async def _load_techniques(self, techniques_file: Path) -> None:
        """Load cooking technique embeddings."""
        logger.info("🔧 Loading cooking technique embeddings...")
        
        with open(techniques_file, 'r', encoding='utf-8') as f:
            techniques_data = json.load(f)
        
        for technique_data in techniques_data:
            embedding = await self.embedder.embed_cooking_technique(
                technique_data["name"],
                technique_data["description"]
            )
            self.embeddings.append(embedding)
    
    async def _load_ingredients(self, ingredients_file: Path) -> None:
        """Load ingredient information embeddings."""
        logger.info("🥕 Loading ingredient embeddings...")
        
        with open(ingredients_file, 'r', encoding='utf-8') as f:
            ingredients_data = json.load(f)
        
        for ingredient_data in ingredients_data:
            embedding = await self.embedder.embed_ingredient_info(
                ingredient_data["name"],
                ingredient_data
            )
            self.embeddings.append(embedding)
    
    def _dict_to_recipe(self, recipe_data: Dict[str, Any]) -> 'Recipe':
        """Convert dictionary to Recipe object."""
        from src.utils.culinary_text_processor import Recipe, Ingredient
        
        # Parse ingredients
        ingredients = []
        for ing_data in recipe_data.get("ingredients", []):
            if isinstance(ing_data, str):
                # Parse from string
                parsed_ing = self.processor._parse_ingredient_line(ing_data)
                if parsed_ing:
                    ingredients.append(parsed_ing)
            else:
                # Already structured
                ingredient = Ingredient(
                    name=ing_data.get("name", ""),
                    quantity=ing_data.get("quantity"),
                    unit=ing_data.get("unit"),
                    preparation=ing_data.get("preparation"),
                    optional=ing_data.get("optional", False),
                    raw_text=ing_data.get("raw_text", "")
                )
                ingredients.append(ingredient)
        
        return Recipe(
            title=recipe_data.get("title", ""),
            ingredients=ingredients,
            instructions=recipe_data.get("instructions", []),
            prep_time=recipe_data.get("prep_time"),
            cook_time=recipe_data.get("cook_time"),
            servings=recipe_data.get("servings"),
            difficulty=recipe_data.get("difficulty"),
            cuisine=recipe_data.get("cuisine"),
            dietary_tags=recipe_data.get("dietary_tags", []),
            nutrition_info=recipe_data.get("nutrition_info")
        )
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        return {
            "total_embeddings": len(self.embeddings),
            "content_types": self._get_content_type_distribution(),
            "cuisines": self._get_cuisine_distribution(),
            "difficulty_levels": self._get_difficulty_distribution(),
            "retrieval_stats": self.retrieval_stats,
            "embedding_stats": self.embedder.get_embedding_stats()
        }
    
    def _get_content_type_distribution(self) -> Dict[str, int]:
        """Get distribution of content types."""
        distribution = {}
        for embedding in self.embeddings:
            content_type = embedding.content_type
            distribution[content_type] = distribution.get(content_type, 0) + 1
        return distribution
    
    def _get_cuisine_distribution(self) -> Dict[str, int]:
        """Get distribution of cuisines."""
        distribution = {}
        for embedding in self.embeddings:
            if embedding.cuisine:
                cuisine = embedding.cuisine
                distribution[cuisine] = distribution.get(cuisine, 0) + 1
        return distribution
    
    def _get_difficulty_distribution(self) -> Dict[str, int]:
        """Get distribution of difficulty levels."""
        distribution = {}
        for embedding in self.embeddings:
            if embedding.difficulty:
                difficulty = embedding.difficulty
                distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution


# Global retriever instance
culinary_retriever = CulinaryRetriever()
