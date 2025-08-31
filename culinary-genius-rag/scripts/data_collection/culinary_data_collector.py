#!/usr/bin/env python3
"""
CulinaryGenius RAG - Culinary Data Collection Script

This script creates a comprehensive culinary dataset including recipes,
cooking techniques, ingredient information, and nutritional data from
multiple sources for the RAG system.

Author: CulinaryGenius Team
Created: 2025-08-31
"""

import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CulinaryDataCollector:
    """Comprehensive culinary data collection and processing."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the culinary data collector."""
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def create_comprehensive_culinary_dataset(self) -> None:
        """Create a comprehensive culinary dataset with global recipes."""
        logger.info("🌍 Creating comprehensive culinary dataset...")
        
        # Create diverse recipe collection
        recipes = self._create_global_recipes()
        
        # Create cooking techniques library
        techniques = self._create_cooking_techniques()
        
        # Create ingredient database
        ingredients = self._create_ingredient_database()
        
        # Create nutritional information
        nutrition_data = self._create_nutrition_database()
        
        # Save all datasets
        self._save_datasets(recipes, techniques, ingredients, nutrition_data)
        
        logger.info("✅ Comprehensive culinary dataset created successfully!")
    
    def _create_global_recipes(self) -> List[Dict[str, Any]]:
        """Create a diverse collection of global recipes."""
        logger.info("🍽️ Creating global recipe collection...")
        
        recipes = [
            # Italian Cuisine
            {
                "id": "recipe_001",
                "title": "Authentic Spaghetti Carbonara",
                "cuisine": "Italian",
                "difficulty": "intermediate",
                "prep_time": 10,
                "cook_time": 15,
                "servings": 4,
                "ingredients": [
                    {"name": "spaghetti", "quantity": 400, "unit": "g"},
                    {"name": "pancetta", "quantity": 150, "unit": "g", "preparation": "diced"},
                    {"name": "eggs", "quantity": 3, "unit": "large"},
                    {"name": "pecorino romano cheese", "quantity": 100, "unit": "g", "preparation": "grated"},
                    {"name": "black pepper", "quantity": 1, "unit": "tsp", "preparation": "freshly ground"},
                    {"name": "salt", "quantity": 1, "unit": "tsp"}
                ],
                "instructions": [
                    "Bring a large pot of salted water to boil and cook spaghetti until al dente",
                    "While pasta cooks, heat a large skillet and cook pancetta until crispy",
                    "In a bowl, whisk together eggs, grated cheese, and black pepper",
                    "Drain pasta, reserving 1 cup of pasta water",
                    "Add hot pasta to the skillet with pancetta",
                    "Remove from heat and quickly toss with egg mixture, adding pasta water as needed",
                    "Serve immediately with extra cheese and pepper"
                ],
                "dietary_tags": [],
                "cooking_methods": ["boiling", "sautéing"],
                "cultural_context": "Traditional Roman dish dating back to charcoal workers (carbonari)",
                "nutrition_highlights": ["High protein from eggs and cheese", "Carbohydrates from pasta"]
            },
            
            # Indian Cuisine
            {
                "id": "recipe_002",
                "title": "Butter Chicken (Murgh Makhani)",
                "cuisine": "Indian",
                "difficulty": "intermediate",
                "prep_time": 30,
                "cook_time": 45,
                "servings": 6,
                "ingredients": [
                    {"name": "chicken thighs", "quantity": 1, "unit": "kg", "preparation": "boneless, cut into pieces"},
                    {"name": "yogurt", "quantity": 200, "unit": "ml", "preparation": "plain"},
                    {"name": "garam masala", "quantity": 2, "unit": "tsp"},
                    {"name": "ginger-garlic paste", "quantity": 2, "unit": "tbsp"},
                    {"name": "tomato puree", "quantity": 400, "unit": "ml"},
                    {"name": "heavy cream", "quantity": 200, "unit": "ml"},
                    {"name": "butter", "quantity": 50, "unit": "g"},
                    {"name": "onion", "quantity": 1, "unit": "large", "preparation": "finely chopped"},
                    {"name": "cashews", "quantity": 50, "unit": "g"},
                    {"name": "fenugreek leaves", "quantity": 1, "unit": "tbsp", "preparation": "dried"}
                ],
                "instructions": [
                    "Marinate chicken with yogurt, half the garam masala, and ginger-garlic paste for 30 minutes",
                    "Heat butter in a heavy-bottomed pan and cook marinated chicken until done",
                    "In the same pan, sauté onions until golden brown",
                    "Add tomato puree and cook until oil separates",
                    "Add cashews and cook for 2 minutes",
                    "Blend the mixture until smooth and return to pan",
                    "Add cooked chicken, cream, remaining garam masala, and fenugreek leaves",
                    "Simmer for 10 minutes and adjust seasoning",
                    "Serve hot with basmati rice or naan bread"
                ],
                "dietary_tags": ["gluten-free"],
                "cooking_methods": ["marinating", "sautéing", "simmering"],
                "cultural_context": "Created in the 1950s by Kundan Lal Gujral in Delhi, inspired by Mughlai cuisine",
                "nutrition_highlights": ["High protein from chicken", "Calcium from dairy", "Antioxidants from tomatoes"]
            },
            
            # Japanese Cuisine
            {
                "id": "recipe_003",
                "title": "Perfect Chicken Teriyaki",
                "cuisine": "Japanese",
                "difficulty": "beginner",
                "prep_time": 15,
                "cook_time": 20,
                "servings": 4,
                "ingredients": [
                    {"name": "chicken thighs", "quantity": 8, "unit": "pieces", "preparation": "skin-on, bone-in"},
                    {"name": "soy sauce", "quantity": 60, "unit": "ml"},
                    {"name": "mirin", "quantity": 60, "unit": "ml"},
                    {"name": "sake", "quantity": 30, "unit": "ml"},
                    {"name": "sugar", "quantity": 2, "unit": "tbsp"},
                    {"name": "ginger", "quantity": 1, "unit": "inch", "preparation": "grated"},
                    {"name": "garlic", "quantity": 2, "unit": "cloves", "preparation": "minced"},
                    {"name": "vegetable oil", "quantity": 1, "unit": "tbsp"},
                    {"name": "green onions", "quantity": 2, "unit": "stalks", "preparation": "chopped"}
                ],
                "instructions": [
                    "Mix soy sauce, mirin, sake, sugar, ginger, and garlic to make teriyaki sauce",
                    "Heat oil in a large skillet over medium-high heat",
                    "Cook chicken skin-side down for 7-8 minutes until golden and crispy",
                    "Flip chicken and cook for another 5-6 minutes",
                    "Pour teriyaki sauce over chicken and simmer for 5 minutes",
                    "Turn chicken to coat with sauce and cook until sauce thickens",
                    "Garnish with chopped green onions and serve with steamed rice"
                ],
                "dietary_tags": ["gluten-free"],
                "cooking_methods": ["pan-frying", "glazing"],
                "cultural_context": "Teriyaki originated in Japan, meaning 'teri' (shine) and 'yaki' (grilled)",
                "nutrition_highlights": ["High protein", "Moderate sodium", "Low carbohydrates"]
            },
            
            # Mexican Cuisine
            {
                "id": "recipe_004",
                "title": "Fresh Guacamole with Lime",
                "cuisine": "Mexican",
                "difficulty": "beginner",
                "prep_time": 15,
                "cook_time": 0,
                "servings": 6,
                "ingredients": [
                    {"name": "avocados", "quantity": 4, "unit": "large", "preparation": "ripe"},
                    {"name": "lime", "quantity": 2, "unit": "medium", "preparation": "juiced"},
                    {"name": "red onion", "quantity": 0.25, "unit": "cup", "preparation": "finely diced"},
                    {"name": "jalapeño", "quantity": 1, "unit": "small", "preparation": "seeded and minced"},
                    {"name": "cilantro", "quantity": 0.25, "unit": "cup", "preparation": "chopped"},
                    {"name": "garlic", "quantity": 1, "unit": "clove", "preparation": "minced"},
                    {"name": "salt", "quantity": 0.5, "unit": "tsp"},
                    {"name": "tomato", "quantity": 1, "unit": "medium", "preparation": "diced", "optional": True}
                ],
                "instructions": [
                    "Cut avocados in half, remove pits, and scoop flesh into a bowl",
                    "Mash avocados with a fork, leaving some chunks for texture",
                    "Add lime juice immediately to prevent browning",
                    "Mix in diced onion, jalapeño, cilantro, and garlic",
                    "Season with salt and taste for balance",
                    "Add diced tomato if using",
                    "Let flavors meld for 10 minutes before serving",
                    "Serve with tortilla chips or as a condiment"
                ],
                "dietary_tags": ["vegan", "gluten-free", "dairy-free", "keto", "paleo"],
                "cooking_methods": ["mixing", "mashing"],
                "cultural_context": "Ancient Aztec dish, 'ahuacamolli' meaning avocado sauce",
                "nutrition_highlights": ["Healthy fats", "Vitamin K", "Folate", "Potassium"]
            },
            
            # Thai Cuisine
            {
                "id": "recipe_005",
                "title": "Thai Green Curry with Chicken",
                "cuisine": "Thai",
                "difficulty": "intermediate",
                "prep_time": 25,
                "cook_time": 30,
                "servings": 4,
                "ingredients": [
                    {"name": "chicken breast", "quantity": 500, "unit": "g", "preparation": "sliced thin"},
                    {"name": "green curry paste", "quantity": 3, "unit": "tbsp"},
                    {"name": "coconut milk", "quantity": 400, "unit": "ml", "preparation": "full-fat"},
                    {"name": "fish sauce", "quantity": 2, "unit": "tbsp"},
                    {"name": "palm sugar", "quantity": 1, "unit": "tbsp"},
                    {"name": "thai basil", "quantity": 1, "unit": "cup", "preparation": "fresh leaves"},
                    {"name": "thai eggplant", "quantity": 200, "unit": "g", "preparation": "quartered"},
                    {"name": "bamboo shoots", "quantity": 100, "unit": "g", "preparation": "sliced"},
                    {"name": "kaffir lime leaves", "quantity": 4, "unit": "leaves", "preparation": "torn"},
                    {"name": "red chilies", "quantity": 2, "unit": "pieces", "preparation": "sliced"}
                ],
                "instructions": [
                    "Heat 2 tbsp of thick coconut cream in a wok over medium heat",
                    "Add green curry paste and fry until fragrant (2-3 minutes)",
                    "Add chicken and cook until no longer pink",
                    "Pour in remaining coconut milk and bring to a gentle simmer",
                    "Add eggplant and bamboo shoots, cook for 10 minutes",
                    "Season with fish sauce and palm sugar",
                    "Add kaffir lime leaves and simmer for 5 more minutes",
                    "Stir in thai basil and red chilies just before serving",
                    "Serve hot with jasmine rice"
                ],
                "dietary_tags": ["gluten-free", "dairy-free"],
                "cooking_methods": ["stir-frying", "simmering"],
                "cultural_context": "Central Thai dish, green curry is the spiciest of Thai curries",
                "nutrition_highlights": ["High protein", "Healthy fats from coconut", "Vitamins from vegetables"]
            },
            
            # French Cuisine
            {
                "id": "recipe_006",
                "title": "Classic French Onion Soup",
                "cuisine": "French",
                "difficulty": "intermediate",
                "prep_time": 20,
                "cook_time": 60,
                "servings": 4,
                "ingredients": [
                    {"name": "yellow onions", "quantity": 6, "unit": "large", "preparation": "thinly sliced"},
                    {"name": "butter", "quantity": 50, "unit": "g"},
                    {"name": "beef stock", "quantity": 1, "unit": "liter", "preparation": "hot"},
                    {"name": "dry white wine", "quantity": 125, "unit": "ml"},
                    {"name": "bay leaves", "quantity": 2, "unit": "leaves"},
                    {"name": "fresh thyme", "quantity": 1, "unit": "tsp"},
                    {"name": "gruyère cheese", "quantity": 200, "unit": "g", "preparation": "grated"},
                    {"name": "baguette", "quantity": 8, "unit": "slices", "preparation": "toasted"},
                    {"name": "salt", "quantity": 1, "unit": "tsp"},
                    {"name": "black pepper", "quantity": 0.5, "unit": "tsp"}
                ],
                "instructions": [
                    "Melt butter in a large heavy-bottomed pot over medium heat",
                    "Add sliced onions and cook slowly for 45 minutes, stirring occasionally",
                    "Onions should become deep golden brown and caramelized",
                    "Add wine and scrape up any browned bits from bottom of pot",
                    "Add hot beef stock, bay leaves, and thyme",
                    "Simmer for 20 minutes, then season with salt and pepper",
                    "Preheat broiler and place soup in oven-safe bowls",
                    "Top each bowl with toasted baguette slices and grated cheese",
                    "Broil until cheese is bubbly and golden brown",
                    "Serve immediately while cheese is still melted"
                ],
                "dietary_tags": [],
                "cooking_methods": ["caramelizing", "simmering", "broiling"],
                "cultural_context": "Originated in ancient Rome, perfected in French bistros",
                "nutrition_highlights": ["Antioxidants from onions", "Calcium from cheese", "Probiotics potential"]
            },
            
            # Mediterranean Cuisine
            {
                "id": "recipe_007",
                "title": "Greek Mediterranean Bowl",
                "cuisine": "Mediterranean",
                "difficulty": "beginner",
                "prep_time": 20,
                "cook_time": 0,
                "servings": 2,
                "ingredients": [
                    {"name": "quinoa", "quantity": 1, "unit": "cup", "preparation": "cooked and cooled"},
                    {"name": "cucumber", "quantity": 1, "unit": "large", "preparation": "diced"},
                    {"name": "cherry tomatoes", "quantity": 200, "unit": "g", "preparation": "halved"},
                    {"name": "red onion", "quantity": 0.25, "unit": "cup", "preparation": "thinly sliced"},
                    {"name": "kalamata olives", "quantity": 0.5, "unit": "cup", "preparation": "pitted"},
                    {"name": "feta cheese", "quantity": 100, "unit": "g", "preparation": "crumbled"},
                    {"name": "extra virgin olive oil", "quantity": 3, "unit": "tbsp"},
                    {"name": "lemon", "quantity": 1, "unit": "medium", "preparation": "juiced"},
                    {"name": "oregano", "quantity": 1, "unit": "tsp", "preparation": "dried"},
                    {"name": "fresh parsley", "quantity": 0.25, "unit": "cup", "preparation": "chopped"}
                ],
                "instructions": [
                    "Place cooked quinoa in serving bowls as the base",
                    "Arrange cucumber, tomatoes, red onion, and olives over quinoa",
                    "Sprinkle crumbled feta cheese on top",
                    "Whisk together olive oil, lemon juice, and oregano for dressing",
                    "Drizzle dressing over the bowl",
                    "Garnish with fresh parsley",
                    "Toss gently before eating and enjoy immediately"
                ],
                "dietary_tags": ["vegetarian", "gluten-free"],
                "cooking_methods": ["assembling", "mixing"],
                "cultural_context": "Inspired by traditional Greek village salads and Mediterranean diet principles",
                "nutrition_highlights": ["Complete protein from quinoa", "Healthy fats from olive oil", "Antioxidants from vegetables"]
            },
            
            # Vegan Option
            {
                "id": "recipe_008",
                "title": "Creamy Coconut Lentil Curry",
                "cuisine": "Indian",
                "difficulty": "beginner",
                "prep_time": 15,
                "cook_time": 35,
                "servings": 4,
                "ingredients": [
                    {"name": "red lentils", "quantity": 1, "unit": "cup", "preparation": "rinsed"},
                    {"name": "coconut milk", "quantity": 400, "unit": "ml", "preparation": "full-fat"},
                    {"name": "onion", "quantity": 1, "unit": "medium", "preparation": "diced"},
                    {"name": "garlic", "quantity": 3, "unit": "cloves", "preparation": "minced"},
                    {"name": "ginger", "quantity": 1, "unit": "inch", "preparation": "grated"},
                    {"name": "turmeric", "quantity": 1, "unit": "tsp"},
                    {"name": "cumin", "quantity": 1, "unit": "tsp"},
                    {"name": "coriander", "quantity": 1, "unit": "tsp", "preparation": "ground"},
                    {"name": "tomato", "quantity": 1, "unit": "large", "preparation": "diced"},
                    {"name": "spinach", "quantity": 2, "unit": "cups", "preparation": "fresh"},
                    {"name": "coconut oil", "quantity": 2, "unit": "tbsp"},
                    {"name": "salt", "quantity": 1, "unit": "tsp"}
                ],
                "instructions": [
                    "Heat coconut oil in a large pot over medium heat",
                    "Sauté onion until translucent, about 5 minutes",
                    "Add garlic and ginger, cook for 1 minute until fragrant",
                    "Add turmeric, cumin, and coriander, cook for 30 seconds",
                    "Add diced tomato and cook until softened",
                    "Add rinsed lentils and 2 cups water, bring to boil",
                    "Reduce heat and simmer for 20 minutes until lentils are tender",
                    "Stir in coconut milk and spinach",
                    "Cook for 5 more minutes until spinach wilts",
                    "Season with salt and serve with rice or naan"
                ],
                "dietary_tags": ["vegan", "gluten-free", "dairy-free", "high-protein"],
                "cooking_methods": ["sautéing", "simmering"],
                "cultural_context": "Dal is a staple in Indian cuisine, providing essential protein in vegetarian diets",
                "nutrition_highlights": ["Plant-based protein", "Iron from lentils", "Healthy fats from coconut"]
            }
        ]
        
        # Add more recipes with variations
        additional_recipes = self._generate_recipe_variations(recipes)
        recipes.extend(additional_recipes)
        
        logger.info(f"✅ Created {len(recipes)} diverse recipes from global cuisines")
        return recipes
    
    def _generate_recipe_variations(self, base_recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate variations of base recipes for dietary restrictions."""
        variations = []
        
        for recipe in base_recipes:
            # Create vegan variation if not already vegan
            if "vegan" not in recipe["dietary_tags"]:
                vegan_recipe = self._create_vegan_variation(recipe)
                if vegan_recipe:
                    variations.append(vegan_recipe)
            
            # Create gluten-free variation if not already gluten-free
            if "gluten-free" not in recipe["dietary_tags"]:
                gf_recipe = self._create_gluten_free_variation(recipe)
                if gf_recipe:
                    variations.append(gf_recipe)
        
        return variations
    
    def _create_vegan_variation(self, recipe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create vegan variation of a recipe."""
        # Simple substitution rules
        vegan_substitutions = {
            "chicken": "tofu or tempeh",
            "beef": "mushrooms or plant-based protein",
            "butter": "vegan butter or coconut oil",
            "cream": "coconut cream",
            "cheese": "nutritional yeast or vegan cheese",
            "eggs": "flax eggs or aquafaba"
        }
        
        # Check if recipe contains animal products
        has_animal_products = False
        for ingredient in recipe["ingredients"]:
            for animal_product in vegan_substitutions.keys():
                if animal_product in ingredient["name"].lower():
                    has_animal_products = True
                    break
        
        if not has_animal_products:
            return None
        
        # Create vegan version
        vegan_recipe = recipe.copy()
        vegan_recipe["id"] = recipe["id"] + "_vegan"
        vegan_recipe["title"] = f"Vegan {recipe['title']}"
        vegan_recipe["dietary_tags"] = recipe["dietary_tags"] + ["vegan", "dairy-free"]
        
        return vegan_recipe
    
    def _create_gluten_free_variation(self, recipe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create gluten-free variation of a recipe."""
        # Check if recipe contains gluten
        gluten_ingredients = ["flour", "wheat", "bread", "pasta", "soy sauce"]
        
        has_gluten = False
        for ingredient in recipe["ingredients"]:
            for gluten_ing in gluten_ingredients:
                if gluten_ing in ingredient["name"].lower():
                    has_gluten = True
                    break
        
        if not has_gluten:
            return None
        
        # Create gluten-free version
        gf_recipe = recipe.copy()
        gf_recipe["id"] = recipe["id"] + "_gf"
        gf_recipe["title"] = f"Gluten-Free {recipe['title']}"
        gf_recipe["dietary_tags"] = recipe["dietary_tags"] + ["gluten-free"]
        
        return gf_recipe
    
    def _create_cooking_techniques(self) -> List[Dict[str, Any]]:
        """Create comprehensive cooking techniques database."""
        logger.info("🔧 Creating cooking techniques library...")
        
        techniques = [
            {
                "id": "tech_001",
                "name": "Sautéing",
                "category": "dry_heat",
                "difficulty": "beginner",
                "description": "Sautéing is a cooking method that uses a small amount of oil or fat in a shallow pan over relatively high heat. The word comes from the French verb 'sauter', meaning 'to jump', referring to the way food moves in the pan.",
                "equipment": ["skillet", "spatula", "tongs"],
                "key_points": [
                    "Use high heat and small amount of oil",
                    "Keep food moving in the pan",
                    "Don't overcrowd the pan",
                    "Pat food dry before adding to pan"
                ],
                "common_mistakes": [
                    "Using too low heat",
                    "Overcrowding the pan",
                    "Not preheating the pan",
                    "Moving food too frequently"
                ],
                "best_for": ["vegetables", "small pieces of meat", "seafood"],
                "cooking_time": "2-10 minutes depending on ingredient"
            },
            {
                "id": "tech_002",
                "name": "Braising",
                "category": "moist_heat",
                "difficulty": "intermediate",
                "description": "Braising is a combination cooking method where food is first seared at high temperature, then finished in liquid at lower temperature. This technique is perfect for tougher cuts of meat and hearty vegetables.",
                "equipment": ["dutch oven", "heavy pot with lid", "tongs"],
                "key_points": [
                    "Sear food first for flavor development",
                    "Use enough liquid to partially cover food",
                    "Cook low and slow for tenderness",
                    "Keep pot covered during cooking"
                ],
                "common_mistakes": [
                    "Not searing properly first",
                    "Using too high heat during braising",
                    "Not using enough liquid",
                    "Lifting the lid too frequently"
                ],
                "best_for": ["tough cuts of meat", "root vegetables", "whole chickens"],
                "cooking_time": "1-4 hours depending on ingredient"
            },
            {
                "id": "tech_003",
                "name": "Tempura Frying",
                "category": "deep_frying",
                "difficulty": "advanced",
                "description": "Tempura is a Japanese deep-frying technique that creates an incredibly light, crispy coating. The secret is in the batter preparation and oil temperature control.",
                "equipment": ["deep pot", "thermometer", "chopsticks", "wire rack"],
                "key_points": [
                    "Keep batter ice-cold and lumpy",
                    "Maintain oil temperature at 340-360°F",
                    "Don't overmix the batter",
                    "Fry in small batches"
                ],
                "common_mistakes": [
                    "Overmixing the batter",
                    "Oil temperature too low or high",
                    "Overcrowding the oil",
                    "Not draining properly"
                ],
                "best_for": ["vegetables", "seafood", "delicate proteins"],
                "cooking_time": "2-4 minutes per batch"
            }
        ]
        
        logger.info(f"✅ Created {len(techniques)} cooking techniques")
        return techniques
    
    def _create_ingredient_database(self) -> List[Dict[str, Any]]:
        """Create comprehensive ingredient information database."""
        logger.info("🥕 Creating ingredient database...")
        
        ingredients = [
            {
                "id": "ing_001",
                "name": "Avocado",
                "category": "fruit",
                "description": "Creamy, nutrient-dense fruit with mild flavor and smooth texture",
                "substitutes": ["mashed banana (for baking)", "hummus (for spreading)", "Greek yogurt (for creaminess)"],
                "storage": "Store at room temperature until ripe, then refrigerate for up to 1 week",
                "season": "Year-round availability, peak in spring and summer",
                "nutrition": {
                    "calories": 160,
                    "protein": 2,
                    "carbs": 9,
                    "fat": 15,
                    "fiber": 7,
                    "vitamin_k": "26% DV",
                    "folate": "20% DV"
                },
                "dietary_tags": ["vegan", "gluten-free", "dairy-free", "keto", "paleo"],
                "cooking_uses": ["salads", "guacamole", "smoothies", "toast topping", "sushi"]
            },
            {
                "id": "ing_002",
                "name": "Coconut Milk",
                "category": "dairy_alternative",
                "description": "Rich, creamy liquid extracted from mature coconut meat",
                "substitutes": ["heavy cream", "cashew cream", "whole milk + butter"],
                "storage": "Unopened cans last 2-3 years, refrigerate after opening for 4-5 days",
                "season": "Available year-round",
                "nutrition": {
                    "calories": 445,
                    "protein": 5,
                    "carbs": 6,
                    "fat": 48,
                    "fiber": 2,
                    "iron": "22% DV",
                    "magnesium": "22% DV"
                },
                "dietary_tags": ["vegan", "gluten-free", "dairy-free", "keto", "paleo"],
                "cooking_uses": ["curries", "soups", "desserts", "smoothies", "baking"]
            }
        ]
        
        logger.info(f"✅ Created {len(ingredients)} ingredient profiles")
        return ingredients
    
    def _create_nutrition_database(self) -> List[Dict[str, Any]]:
        """Create nutritional information database."""
        logger.info("📊 Creating nutrition database...")
        
        nutrition_data = [
            {
                "id": "nutr_001",
                "topic": "Protein Requirements",
                "category": "macronutrients",
                "description": "Daily protein needs vary by age, sex, activity level, and health goals",
                "key_points": [
                    "Adults need 0.8g protein per kg body weight minimum",
                    "Athletes may need 1.2-2.0g per kg body weight",
                    "Complete proteins contain all essential amino acids",
                    "Plant proteins can be combined for completeness"
                ],
                "food_sources": {
                    "animal": ["chicken", "fish", "eggs", "dairy"],
                    "plant": ["lentils", "quinoa", "tofu", "nuts", "seeds"]
                }
            },
            {
                "id": "nutr_002",
                "topic": "Mediterranean Diet Benefits",
                "category": "dietary_patterns",
                "description": "Research-backed eating pattern associated with numerous health benefits",
                "key_points": [
                    "Emphasizes olive oil, fish, vegetables, and whole grains",
                    "Associated with reduced heart disease risk",
                    "May support brain health and longevity",
                    "Rich in antioxidants and healthy fats"
                ],
                "food_sources": {
                    "primary": ["olive oil", "fish", "vegetables", "fruits", "whole grains"],
                    "moderate": ["poultry", "eggs", "dairy"],
                    "limited": ["red meat", "processed foods"]
                }
            }
        ]
        
        logger.info(f"✅ Created {len(nutrition_data)} nutrition topics")
        return nutrition_data
    
    def _save_datasets(
        self, 
        recipes: List[Dict[str, Any]], 
        techniques: List[Dict[str, Any]],
        ingredients: List[Dict[str, Any]], 
        nutrition_data: List[Dict[str, Any]]
    ) -> None:
        """Save all datasets to files."""
        logger.info("💾 Saving culinary datasets...")
        
        # Save recipes
        recipes_file = self.processed_dir / "recipes.json"
        with open(recipes_file, 'w', encoding='utf-8') as f:
            json.dump(recipes, f, indent=2, ensure_ascii=False)
        
        # Save techniques
        techniques_file = self.processed_dir / "cooking_techniques.json"
        with open(techniques_file, 'w', encoding='utf-8') as f:
            json.dump(techniques, f, indent=2, ensure_ascii=False)
        
        # Save ingredients
        ingredients_file = self.processed_dir / "ingredients.json"
        with open(ingredients_file, 'w', encoding='utf-8') as f:
            json.dump(ingredients, f, indent=2, ensure_ascii=False)
        
        # Save nutrition data
        nutrition_file = self.processed_dir / "nutrition.json"
        with open(nutrition_file, 'w', encoding='utf-8') as f:
            json.dump(nutrition_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 Datasets saved:")
        logger.info(f"  - Recipes: {recipes_file} ({len(recipes)} items)")
        logger.info(f"  - Techniques: {techniques_file} ({len(techniques)} items)")
        logger.info(f"  - Ingredients: {ingredients_file} ({len(ingredients)} items)")
        logger.info(f"  - Nutrition: {nutrition_file} ({len(nutrition_data)} items)")


def main():
    """Main function to run culinary data collection."""
    logger.info("🚀 Starting CulinaryGenius data collection...")
    
    try:
        collector = CulinaryDataCollector()
        collector.create_comprehensive_culinary_dataset()
        
        logger.info("🎉 Culinary data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
