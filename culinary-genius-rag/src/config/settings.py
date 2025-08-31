"""
CulinaryGenius RAG - Configuration Settings

This module manages all configuration settings for the CulinaryGenius RAG system,
including API keys, model parameters, and culinary-specific configurations.

Author: CulinaryGenius Team
Created: 2025-08-31
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseSettings, Field


class CulinarySettings(BaseSettings):
    """
    Configuration settings for CulinaryGenius RAG system.
    
    Manages all environment variables and configuration parameters
    for the culinary AI assistant with sensible defaults.
    """
    
    # API Configuration
    api_version: str = Field("1.0.0", description="API version")
    host: str = Field("0.0.0.0", description="API host")
    port: int = Field(8000, description="API port")
    environment: str = Field("development", description="Environment (development/staging/production)")
    debug: bool = Field(True, description="Debug mode")
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field("gpt-4o-mini", description="OpenAI model for generation")
    embedding_model: str = Field("text-embedding-3-small", description="OpenAI embedding model")
    
    # RAG Configuration
    max_context_length: int = Field(8000, description="Maximum context length for generation")
    max_response_length: int = Field(1000, description="Maximum response length")
    response_temperature: float = Field(0.3, description="Temperature for response generation")
    embedding_batch_size: int = Field(100, description="Batch size for embedding generation")
    
    # Culinary-Specific Configuration
    max_recipes_per_query: int = Field(5, description="Maximum recipes to return per query")
    include_nutritional_info: bool = Field(True, description="Include nutritional information")
    include_cultural_context: bool = Field(True, description="Include cultural context")
    default_serving_size: int = Field(4, description="Default serving size for recipes")
    
    # Supported dietary restrictions
    supported_diets: List[str] = Field(
        default=[
            "vegetarian", "vegan", "gluten-free", "dairy-free", 
            "nut-free", "keto", "paleo", "low-carb", "low-fat",
            "halal", "kosher", "pescatarian"
        ],
        description="Supported dietary restrictions"
    )
    
    # Supported cuisines
    supported_cuisines: List[str] = Field(
        default=[
            "Italian", "Chinese", "Indian", "Mexican", "French", "Japanese",
            "Thai", "Mediterranean", "American", "Korean", "Vietnamese",
            "Greek", "Spanish", "Turkish", "Lebanese", "Moroccan",
            "Brazilian", "Peruvian", "Ethiopian", "Russian"
        ],
        description="Supported cuisine types"
    )
    
    # Cooking skill levels
    skill_levels: List[str] = Field(
        default=["beginner", "intermediate", "advanced", "professional"],
        description="Cooking skill levels"
    )
    
    # Cooking methods
    cooking_methods: List[str] = Field(
        default=[
            "baking", "roasting", "grilling", "frying", "steaming",
            "boiling", "sautéing", "braising", "stewing", "smoking",
            "fermentation", "pickling", "curing", "sous-vide"
        ],
        description="Supported cooking methods"
    )
    
    # Data Sources Configuration
    recipe_sources: Dict[str, str] = Field(
        default={
            "allrecipes": "https://www.allrecipes.com",
            "food_network": "https://www.foodnetwork.com",
            "serious_eats": "https://www.seriouseats.com",
            "bon_appetit": "https://www.bonappetit.com",
            "epicurious": "https://www.epicurious.com"
        },
        description="Recipe data sources"
    )
    
    # Nutritional API Configuration
    nutrition_api_key: Optional[str] = Field(None, description="Nutritional API key (optional)")
    usda_api_key: Optional[str] = Field(None, description="USDA FoodData Central API key")
    
    # Cache Configuration
    enable_caching: bool = Field(True, description="Enable response caching")
    cache_ttl_seconds: int = Field(3600, description="Cache TTL in seconds")
    
    # Logging Configuration
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    
    # Security Configuration
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    allowed_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        description="Allowed hosts"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(100, description="Requests per minute limit")
    rate_limit_window: int = Field(60, description="Rate limit window in seconds")
    
    # Feature Flags
    enable_recipe_generation: bool = Field(True, description="Enable AI recipe generation")
    enable_meal_planning: bool = Field(True, description="Enable meal planning features")
    enable_shopping_lists: bool = Field(True, description="Enable shopping list generation")
    enable_cooking_timer: bool = Field(True, description="Enable cooking timer features")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
_settings: Optional[CulinarySettings] = None


def get_settings() -> CulinarySettings:
    """
    Get global settings instance.
    
    Returns:
        CulinarySettings: Global configuration settings
    """
    global _settings
    if _settings is None:
        _settings = CulinarySettings()
    return _settings


def reload_settings() -> CulinarySettings:
    """
    Reload settings from environment.
    
    Returns:
        CulinarySettings: Reloaded configuration settings
    """
    global _settings
    _settings = CulinarySettings()
    return _settings
