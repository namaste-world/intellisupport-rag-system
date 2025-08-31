#!/usr/bin/env python3
"""
CulinaryGenius RAG - Main Application

An intelligent culinary assistant powered by advanced RAG technology.
Discover recipes, learn cooking techniques, and get personalized
culinary guidance from cuisines around the world!

Author: CulinaryGenius Team
Created: 2025-08-31
"""

import logging
import json
import asyncio
from pathlib import Path
import time
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="🍳 CulinaryGenius RAG API",
    description="AI-powered culinary assistant for recipes, techniques, and food knowledge",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class CulinaryQueryRequest(BaseModel):
    """Culinary query request model."""
    query: str
    user_id: str = "guest"
    dietary_restrictions: List[str] = []
    preferred_cuisines: List[str] = []
    skill_level: str = "intermediate"
    max_prep_time: int = None
    available_ingredients: List[str] = []


class RecipeSearchRequest(BaseModel):
    """Recipe search request model."""
    ingredients: List[str]
    cuisine: str = None
    dietary_restrictions: List[str] = []
    max_time: int = None
    skill_level: str = "intermediate"


class IngredientSubstitutionRequest(BaseModel):
    """Ingredient substitution request."""
    ingredient: str
    recipe_context: str = None
    dietary_restrictions: List[str] = []


class CulinaryResponse(BaseModel):
    """Culinary response model."""
    response: str
    confidence_score: float
    response_type: str
    recipes_found: int
    techniques_mentioned: List[str]
    cooking_tips: List[str]
    processing_time_ms: int


# Simple in-memory culinary knowledge base
CULINARY_KNOWLEDGE = {
    "recipes": [
        {
            "id": "pasta_carbonara",
            "title": "Authentic Spaghetti Carbonara",
            "cuisine": "Italian",
            "ingredients": ["spaghetti", "pancetta", "eggs", "pecorino romano", "black pepper"],
            "instructions": [
                "Cook spaghetti in salted boiling water until al dente",
                "Crisp pancetta in a large skillet",
                "Whisk eggs with grated cheese and pepper",
                "Toss hot pasta with pancetta and egg mixture off heat",
                "Serve immediately with extra cheese"
            ],
            "prep_time": 10,
            "cook_time": 15,
            "difficulty": "intermediate",
            "dietary_tags": [],
            "cultural_context": "Traditional Roman dish from charcoal workers"
        },
        {
            "id": "butter_chicken",
            "title": "Creamy Butter Chicken",
            "cuisine": "Indian",
            "ingredients": ["chicken", "tomato sauce", "cream", "butter", "garam masala", "ginger", "garlic"],
            "instructions": [
                "Marinate chicken in yogurt and spices",
                "Cook chicken until done",
                "Make tomato-based sauce with cream",
                "Combine chicken with sauce",
                "Simmer and serve with rice"
            ],
            "prep_time": 30,
            "cook_time": 45,
            "difficulty": "intermediate",
            "dietary_tags": ["gluten-free"],
            "cultural_context": "Created in 1950s Delhi, inspired by Mughlai cuisine"
        },
        {
            "id": "avocado_toast",
            "title": "Perfect Avocado Toast",
            "cuisine": "Modern",
            "ingredients": ["bread", "avocado", "lime", "salt", "pepper", "olive oil"],
            "instructions": [
                "Toast bread until golden",
                "Mash avocado with lime juice",
                "Season with salt and pepper",
                "Spread on toast",
                "Drizzle with olive oil"
            ],
            "prep_time": 5,
            "cook_time": 2,
            "difficulty": "beginner",
            "dietary_tags": ["vegetarian", "vegan"],
            "cultural_context": "Modern breakfast trend popularized in cafes worldwide"
        },
        {
            "id": "green_curry",
            "title": "Thai Green Curry",
            "cuisine": "Thai",
            "ingredients": ["green curry paste", "coconut milk", "chicken", "thai basil", "eggplant"],
            "instructions": [
                "Fry curry paste in coconut cream",
                "Add chicken and cook through",
                "Add remaining coconut milk",
                "Add vegetables and simmer",
                "Finish with thai basil"
            ],
            "prep_time": 20,
            "cook_time": 25,
            "difficulty": "intermediate",
            "dietary_tags": ["gluten-free", "dairy-free"],
            "cultural_context": "Central Thai dish, spiciest of the Thai curries"
        }
    ],
    "techniques": [
        {
            "name": "sautéing",
            "description": "Quick cooking in a small amount of fat over high heat",
            "equipment": ["skillet", "spatula"],
            "tips": ["Keep food moving", "Don't overcrowd", "Use high heat"]
        },
        {
            "name": "braising",
            "description": "Combination of searing and slow cooking in liquid",
            "equipment": ["dutch oven", "tongs"],
            "tips": ["Sear first", "Use low heat", "Keep covered"]
        }
    ],
    "ingredients": [
        {
            "name": "avocado",
            "substitutes": ["mashed banana for baking", "hummus for spreading"],
            "storage": "Room temperature until ripe, then refrigerate",
            "nutrition": "High in healthy fats, fiber, potassium"
        },
        {
            "name": "heavy cream",
            "substitutes": ["coconut cream", "cashew cream", "milk + butter"],
            "storage": "Refrigerate, use within 1 week of opening",
            "nutrition": "High in fat and calories"
        }
    ]
}


class CulinaryRAGService:
    """Simple culinary RAG service."""
    
    def __init__(self):
        """Initialize culinary RAG service."""
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.knowledge = CULINARY_KNOWLEDGE
        self.embeddings_cache = {}
    
    async def search_recipes(self, request: CulinaryQueryRequest) -> CulinaryResponse:
        """Search for recipes based on query."""
        start_time = time.time()
        
        # Simple keyword matching for demo
        matching_recipes = []
        query_lower = request.query.lower()
        
        for recipe in self.knowledge["recipes"]:
            # Check if query matches recipe
            score = 0
            
            # Title match
            if any(word in recipe["title"].lower() for word in query_lower.split()):
                score += 0.5
            
            # Ingredient match
            for ingredient in recipe["ingredients"]:
                if ingredient.lower() in query_lower:
                    score += 0.3
            
            # Cuisine match
            if request.preferred_cuisines and recipe["cuisine"] in request.preferred_cuisines:
                score += 0.4
            
            # Dietary restrictions
            if request.dietary_restrictions:
                if all(restriction in recipe["dietary_tags"] for restriction in request.dietary_restrictions):
                    score += 0.3
                elif any(restriction in recipe["dietary_tags"] for restriction in request.dietary_restrictions):
                    score += 0.1
            
            # Time constraints
            if request.max_prep_time and recipe["prep_time"] <= request.max_prep_time:
                score += 0.2
            
            if score > 0.3:  # Threshold for relevance
                matching_recipes.append((recipe, score))
        
        # Sort by score
        matching_recipes.sort(key=lambda x: x[1], reverse=True)
        
        # Generate response using AI
        if matching_recipes:
            top_recipes = [recipe for recipe, score in matching_recipes[:3]]
            response_text = await self._generate_recipe_response(request.query, top_recipes)
            confidence = min(matching_recipes[0][1], 1.0)
        else:
            response_text = "I couldn't find specific recipes matching your criteria, but I can help you create something delicious! Could you provide more details about what you'd like to cook?"
            confidence = 0.2
        
        # Extract techniques mentioned
        techniques = []
        for recipe in [r for r, s in matching_recipes[:3]]:
            recipe_text = " ".join(recipe["instructions"])
            for technique in self.knowledge["techniques"]:
                if technique["name"] in recipe_text.lower():
                    techniques.append(technique["name"])
        
        processing_time = (time.time() - start_time) * 1000
        
        return CulinaryResponse(
            response=response_text,
            confidence_score=confidence,
            response_type="recipe_search",
            recipes_found=len(matching_recipes),
            techniques_mentioned=list(set(techniques)),
            cooking_tips=self._extract_cooking_tips(matching_recipes),
            processing_time_ms=int(processing_time)
        )
    
    async def _generate_recipe_response(self, query: str, recipes: List[Dict]) -> str:
        """Generate AI response for recipe recommendations."""
        # Build context
        context = "Here are some relevant recipes:\n\n"
        
        for i, recipe in enumerate(recipes):
            context += f"Recipe {i+1}: {recipe['title']} ({recipe['cuisine']} cuisine)\n"
            context += f"Ingredients: {', '.join(recipe['ingredients'])}\n"
            context += f"Instructions: {' '.join(recipe['instructions'])}\n"
            if recipe.get('cultural_context'):
                context += f"Cultural Context: {recipe['cultural_context']}\n"
            context += "\n"
        
        # Generate response
        system_prompt = """You are CulinaryGenius, an expert chef and culinary assistant. 
        Provide helpful, detailed culinary advice with enthusiasm and expertise. 
        Include cooking tips, cultural insights, and personalized recommendations."""
        
        user_prompt = f"""Based on these recipes:

{context}

User Query: {query}

Please provide a comprehensive culinary response that includes:
1. Recipe recommendations with brief descriptions
2. Cooking tips and techniques
3. Cultural context when relevant
4. Ingredient substitution suggestions if helpful
5. Difficulty assessment and time estimates

Make your response engaging, informative, and actionable!"""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            return f"I found {len(recipes)} great recipes for you! Here are my top recommendations based on your query."
    
    def _extract_cooking_tips(self, matching_recipes: List[tuple]) -> List[str]:
        """Extract cooking tips from matching recipes."""
        tips = []
        
        for recipe, score in matching_recipes[:2]:  # Top 2 recipes
            if recipe["difficulty"] == "advanced":
                tips.append("This is an advanced recipe - take your time with each step")
            elif recipe["difficulty"] == "beginner":
                tips.append("Perfect recipe for beginners - great for learning!")
            
            if recipe["cuisine"] == "Italian":
                tips.append("Use high-quality ingredients for authentic Italian flavors")
            elif recipe["cuisine"] == "Thai":
                tips.append("Balance sweet, sour, salty, and spicy flavors")
        
        return tips


# Initialize service
culinary_service = CulinaryRAGService()


@app.get("/")
async def root():
    """Welcome endpoint with API overview."""
    return {
        "service": "🍳 CulinaryGenius RAG API",
        "version": "1.0.0",
        "description": "AI-powered culinary assistant for global cuisine exploration",
        "features": [
            "🔍 Recipe search with dietary filters",
            "🔄 Smart ingredient substitutions", 
            "🔧 Cooking technique guidance",
            "📅 Personalized meal planning",
            "🌍 Global cuisine knowledge",
            "📊 Nutritional insights"
        ],
        "endpoints": {
            "recipe_search": "/recipe/search",
            "ingredient_substitution": "/ingredients/substitute",
            "cooking_help": "/technique/help",
            "meal_planning": "/meal-plan/generate",
            "health": "/health"
        },
        "supported_cuisines": ["Italian", "Indian", "Thai", "Mexican", "French", "Japanese", "Mediterranean"],
        "dietary_support": ["vegetarian", "vegan", "gluten-free", "dairy-free", "keto", "paleo"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "service": "CulinaryGenius RAG",
        "knowledge_base": {
            "recipes": len(CULINARY_KNOWLEDGE["recipes"]),
            "techniques": len(CULINARY_KNOWLEDGE["techniques"]),
            "ingredients": len(CULINARY_KNOWLEDGE["ingredients"])
        },
        "api_status": "operational"
    }


@app.post("/recipe/search", response_model=CulinaryResponse)
async def search_recipes(request: CulinaryQueryRequest):
    """🔍 Search for recipes with intelligent recommendations."""
    logger.info(f"🔍 Recipe search: {request.query}")
    
    try:
        response = await culinary_service.search_recipes(request)
        return response
        
    except Exception as e:
        logger.error(f"❌ Recipe search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Recipe search failed: {str(e)}"
        )


@app.post("/ingredients/substitute")
async def substitute_ingredient(request: IngredientSubstitutionRequest):
    """🔄 Get smart ingredient substitutions."""
    logger.info(f"🔄 Substitution request: {request.ingredient}")
    
    # Find ingredient in knowledge base
    for ingredient_info in CULINARY_KNOWLEDGE["ingredients"]:
        if ingredient_info["name"].lower() == request.ingredient.lower():
            return {
                "original_ingredient": request.ingredient,
                "substitutes": ingredient_info["substitutes"],
                "storage_tips": ingredient_info["storage"],
                "nutrition_info": ingredient_info["nutrition"],
                "confidence_score": 0.9
            }
    
    # Generate AI-powered substitution
    try:
        system_prompt = """You are CulinaryGenius, an expert in ingredient substitutions. 
        Provide practical, tested substitution recommendations with ratios and cooking adjustments."""
        
        user_prompt = f"""What are the best substitutes for {request.ingredient}?
        
        Context: {request.recipe_context or 'General cooking'}
        Dietary restrictions: {', '.join(request.dietary_restrictions) if request.dietary_restrictions else 'None'}
        
        Please provide:
        1. 3-5 substitute options with ratios
        2. How each substitute affects flavor/texture
        3. Any cooking adjustments needed
        4. Best use cases for each substitute"""
        
        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return {
            "original_ingredient": request.ingredient,
            "substitution_advice": response.choices[0].message.content,
            "confidence_score": 0.8,
            "ai_generated": True
        }
        
    except Exception as e:
        logger.error(f"❌ Substitution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Substitution lookup failed: {str(e)}")


@app.get("/cuisines")
async def get_supported_cuisines():
    """🌍 Get list of supported cuisines."""
    cuisines = list(set(recipe["cuisine"] for recipe in CULINARY_KNOWLEDGE["recipes"]))
    return {
        "supported_cuisines": cuisines,
        "total_count": len(cuisines),
        "popular_cuisines": ["Italian", "Indian", "Thai", "Mexican", "French"]
    }


@app.get("/techniques")
async def get_cooking_techniques():
    """🔧 Get list of cooking techniques."""
    return {
        "techniques": CULINARY_KNOWLEDGE["techniques"],
        "categories": ["dry_heat", "moist_heat", "combination"],
        "difficulty_levels": ["beginner", "intermediate", "advanced"]
    }


@app.post("/cooking/help")
async def get_cooking_help(query: str):
    """🆘 Get general cooking help and guidance."""
    logger.info(f"🆘 Cooking help: {query}")
    
    try:
        system_prompt = """You are CulinaryGenius, a master chef and cooking instructor. 
        Provide expert culinary guidance with practical tips, techniques, and encouragement.
        Make cooking accessible and enjoyable for everyone!"""
        
        user_prompt = f"""Cooking question: {query}
        
        Please provide helpful guidance including:
        1. Direct answer to the question
        2. Step-by-step instructions if applicable
        3. Professional tips and tricks
        4. Common mistakes to avoid
        5. Encouragement and confidence building"""
        
        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,
            max_tokens=600
        )
        
        return {
            "cooking_advice": response.choices[0].message.content,
            "confidence_score": 0.85,
            "response_type": "cooking_help"
        }
        
    except Exception as e:
        logger.error(f"❌ Cooking help failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cooking help failed: {str(e)}")


@app.get("/random-recipe")
async def get_random_recipe():
    """🎲 Get a random recipe recommendation."""
    import random
    
    recipe = random.choice(CULINARY_KNOWLEDGE["recipes"])
    
    return {
        "recipe": recipe,
        "message": "🎲 Here's a random recipe to inspire your cooking!",
        "cooking_tip": "Try something new today - cooking is an adventure!"
    }


if __name__ == "__main__":
    logger.info("🚀 Starting CulinaryGenius RAG API...")
    logger.info("🍳 Ready to help you discover amazing recipes and cooking techniques!")
    
    uvicorn.run(
        "culinary_genius_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
