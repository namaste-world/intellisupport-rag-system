"""
CulinaryGenius RAG - Specialized Culinary API Schemas

This module defines Pydantic models for culinary-specific API requests
and responses with comprehensive validation and documentation.

Author: CulinaryGenius Team
Created: 2025-08-31
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator


class CuisineType(str, Enum):
    """Supported cuisine types."""
    ITALIAN = "Italian"
    CHINESE = "Chinese"
    INDIAN = "Indian"
    MEXICAN = "Mexican"
    FRENCH = "French"
    JAPANESE = "Japanese"
    THAI = "Thai"
    MEDITERRANEAN = "Mediterranean"
    AMERICAN = "American"
    KOREAN = "Korean"


class DietaryRestriction(str, Enum):
    """Supported dietary restrictions."""
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten-free"
    DAIRY_FREE = "dairy-free"
    NUT_FREE = "nut-free"
    KETO = "keto"
    PALEO = "paleo"
    LOW_CARB = "low-carb"
    HALAL = "halal"
    KOSHER = "kosher"


class SkillLevel(str, Enum):
    """Cooking skill levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PROFESSIONAL = "professional"


class MealType(str, Enum):
    """Meal types."""
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    DESSERT = "dessert"
    APPETIZER = "appetizer"


class RecipeSearchRequest(BaseModel):
    """Request model for recipe search."""
    query: str = Field(..., min_length=1, max_length=500, description="Recipe search query")
    user_id: str = Field(..., min_length=1, max_length=100, description="User identifier")
    
    # Filters
    cuisines: Optional[List[CuisineType]] = Field(None, description="Preferred cuisine types")
    dietary_restrictions: Optional[List[DietaryRestriction]] = Field(None, description="Dietary restrictions")
    skill_level: SkillLevel = Field(SkillLevel.INTERMEDIATE, description="User's cooking skill level")
    meal_type: Optional[MealType] = Field(None, description="Type of meal")
    
    # Time constraints
    max_prep_time: Optional[int] = Field(None, ge=0, le=480, description="Maximum prep time in minutes")
    max_cook_time: Optional[int] = Field(None, ge=0, le=480, description="Maximum cook time in minutes")
    max_total_time: Optional[int] = Field(None, ge=0, le=600, description="Maximum total time in minutes")
    
    # Ingredients
    available_ingredients: Optional[List[str]] = Field(None, description="Ingredients user has available")
    exclude_ingredients: Optional[List[str]] = Field(None, description="Ingredients to avoid")
    
    # Preferences
    servings: Optional[int] = Field(None, ge=1, le=20, description="Desired number of servings")
    include_nutrition: bool = Field(True, description="Include nutritional information")
    include_cultural_context: bool = Field(True, description="Include cultural background")
    max_results: int = Field(5, ge=1, le=10, description="Maximum number of recipes to return")


class IngredientSubstitutionRequest(BaseModel):
    """Request model for ingredient substitution."""
    ingredient: str = Field(..., min_length=1, description="Ingredient to substitute")
    recipe_context: Optional[str] = Field(None, description="Recipe or cooking context")
    dietary_restrictions: Optional[List[DietaryRestriction]] = Field(None, description="Dietary restrictions")
    available_alternatives: Optional[List[str]] = Field(None, description="Available alternative ingredients")
    cooking_method: Optional[str] = Field(None, description="Cooking method being used")


class CookingTechniqueRequest(BaseModel):
    """Request model for cooking technique help."""
    technique: str = Field(..., min_length=1, description="Cooking technique to learn about")
    skill_level: SkillLevel = Field(SkillLevel.BEGINNER, description="User's skill level")
    equipment_available: Optional[List[str]] = Field(None, description="Available cooking equipment")
    specific_question: Optional[str] = Field(None, description="Specific question about the technique")


class MealPlanRequest(BaseModel):
    """Request model for meal planning."""
    days: int = Field(7, ge=1, le=14, description="Number of days to plan")
    meals_per_day: int = Field(3, ge=1, le=6, description="Number of meals per day")
    dietary_restrictions: Optional[List[DietaryRestriction]] = Field(None, description="Dietary restrictions")
    cuisines: Optional[List[CuisineType]] = Field(None, description="Preferred cuisines")
    skill_level: SkillLevel = Field(SkillLevel.INTERMEDIATE, description="Cooking skill level")
    budget_level: Optional[str] = Field("medium", description="Budget level (low, medium, high)")
    avoid_repetition: bool = Field(True, description="Avoid repeating ingredients/dishes")


class NutritionInfo(BaseModel):
    """Nutritional information model."""
    calories: Optional[int] = Field(None, description="Calories per serving")
    protein: Optional[float] = Field(None, description="Protein in grams")
    carbohydrates: Optional[float] = Field(None, description="Carbohydrates in grams")
    fat: Optional[float] = Field(None, description="Fat in grams")
    fiber: Optional[float] = Field(None, description="Fiber in grams")
    sugar: Optional[float] = Field(None, description="Sugar in grams")
    sodium: Optional[float] = Field(None, description="Sodium in milligrams")
    vitamins: Optional[Dict[str, str]] = Field(None, description="Vitamin content")
    minerals: Optional[Dict[str, str]] = Field(None, description="Mineral content")


class RecipeResult(BaseModel):
    """Individual recipe result."""
    title: str = Field(..., description="Recipe title")
    cuisine: Optional[str] = Field(None, description="Cuisine type")
    difficulty: Optional[str] = Field(None, description="Difficulty level")
    prep_time: Optional[int] = Field(None, description="Preparation time in minutes")
    cook_time: Optional[int] = Field(None, description="Cooking time in minutes")
    total_time: Optional[int] = Field(None, description="Total time in minutes")
    servings: Optional[int] = Field(None, description="Number of servings")
    
    ingredients: List[str] = Field(..., description="List of ingredients")
    instructions: List[str] = Field(..., description="Cooking instructions")
    
    dietary_tags: List[str] = Field(default_factory=list, description="Dietary restriction tags")
    cooking_methods: List[str] = Field(default_factory=list, description="Cooking methods used")
    
    nutrition_info: Optional[NutritionInfo] = Field(None, description="Nutritional information")
    cultural_context: Optional[str] = Field(None, description="Cultural background")
    chef_tips: List[str] = Field(default_factory=list, description="Professional cooking tips")
    
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to query")


class CulinaryResponse(BaseModel):
    """Main culinary response model."""
    response: str = Field(..., description="Generated culinary response")
    response_type: str = Field(..., description="Type of response (recipe, technique, etc.)")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Response confidence")
    
    # Culinary-specific metadata
    recipes: List[RecipeResult] = Field(default_factory=list, description="Recipe recommendations")
    techniques_mentioned: List[str] = Field(default_factory=list, description="Cooking techniques discussed")
    ingredients_discussed: List[str] = Field(default_factory=list, description="Ingredients mentioned")
    nutritional_highlights: List[str] = Field(default_factory=list, description="Nutritional benefits")
    cultural_insights: Optional[str] = Field(None, description="Cultural context and history")
    cooking_tips: List[str] = Field(default_factory=list, description="Professional cooking tips")
    
    # Time and difficulty
    estimated_prep_time: Optional[int] = Field(None, description="Estimated prep time in minutes")
    estimated_cook_time: Optional[int] = Field(None, description="Estimated cook time in minutes")
    difficulty_assessment: Optional[str] = Field(None, description="Overall difficulty level")
    
    # Metadata
    query_id: str = Field(..., description="Unique query identifier")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    sources_used: int = Field(..., description="Number of knowledge sources used")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "Here's a delicious Spaghetti Carbonara recipe perfect for your skill level...",
                "response_type": "recipe",
                "confidence_score": 0.92,
                "recipes": [
                    {
                        "title": "Authentic Spaghetti Carbonara",
                        "cuisine": "Italian",
                        "difficulty": "intermediate",
                        "prep_time": 10,
                        "cook_time": 15,
                        "servings": 4,
                        "ingredients": ["400g spaghetti", "150g pancetta", "3 eggs"],
                        "instructions": ["Boil pasta...", "Cook pancetta..."],
                        "relevance_score": 0.95
                    }
                ],
                "techniques_mentioned": ["boiling", "sautéing"],
                "cooking_tips": ["Keep egg mixture moving to prevent scrambling"],
                "query_id": "query_123",
                "processing_time_ms": 2500,
                "sources_used": 3
            }
        }


class SubstitutionResponse(BaseModel):
    """Response model for ingredient substitutions."""
    original_ingredient: str = Field(..., description="Original ingredient")
    substitutions: List[Dict[str, Any]] = Field(..., description="Substitution options")
    cooking_adjustments: Optional[str] = Field(None, description="Cooking method adjustments")
    flavor_impact: Optional[str] = Field(None, description="Expected flavor changes")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Substitution confidence")


class TechniqueResponse(BaseModel):
    """Response model for cooking technique guidance."""
    technique_name: str = Field(..., description="Cooking technique name")
    difficulty: str = Field(..., description="Technique difficulty level")
    instructions: List[str] = Field(..., description="Step-by-step instructions")
    equipment_needed: List[str] = Field(..., description="Required equipment")
    key_tips: List[str] = Field(..., description="Key success tips")
    common_mistakes: List[str] = Field(..., description="Common mistakes to avoid")
    practice_suggestions: List[str] = Field(..., description="How to practice and improve")


class MealPlanResponse(BaseModel):
    """Response model for meal planning."""
    meal_plan: Dict[str, Dict[str, RecipeResult]] = Field(..., description="Daily meal plan")
    shopping_list: List[Dict[str, Any]] = Field(..., description="Consolidated shopping list")
    nutrition_summary: Dict[str, Any] = Field(..., description="Weekly nutrition summary")
    preparation_tips: List[str] = Field(..., description="Meal prep suggestions")
    estimated_cost: Optional[str] = Field(None, description="Estimated cost range")


class CulinaryError(BaseModel):
    """Error response model for culinary API."""
    error: Dict[str, Any] = Field(..., description="Error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "code": "RECIPE_NOT_FOUND",
                    "message": "No recipes found matching your criteria",
                    "suggestions": [
                        "Try broader search terms",
                        "Remove some dietary restrictions",
                        "Increase maximum cooking time"
                    ],
                    "query_id": "query_xyz789"
                }
            }
        }
