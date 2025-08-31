"""
CulinaryGenius RAG - Specialized Culinary Text Processor

This module provides advanced text processing capabilities specifically
designed for culinary content, including recipe parsing, ingredient
extraction, and cooking technique recognition.

Author: CulinaryGenius Team
Created: 2025-08-31
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from fractions import Fraction

logger = logging.getLogger(__name__)


@dataclass
class Ingredient:
    """Represents a parsed ingredient with quantity and metadata."""
    name: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    preparation: Optional[str] = None  # e.g., "diced", "chopped", "minced"
    optional: bool = False
    raw_text: str = ""


@dataclass
class Recipe:
    """Represents a parsed recipe with structured data."""
    title: str
    ingredients: List[Ingredient]
    instructions: List[str]
    prep_time: Optional[int] = None  # minutes
    cook_time: Optional[int] = None  # minutes
    servings: Optional[int] = None
    difficulty: Optional[str] = None
    cuisine: Optional[str] = None
    dietary_tags: List[str] = None
    nutrition_info: Optional[Dict[str, Any]] = None


class CulinaryTextProcessor:
    """
    Advanced text processor for culinary content.
    
    Provides specialized functionality for processing recipes, ingredients,
    cooking techniques, and culinary knowledge with domain-specific
    understanding and extraction capabilities.
    """
    
    def __init__(self):
        """Initialize the culinary text processor."""
        self.units = self._load_cooking_units()
        self.cooking_verbs = self._load_cooking_verbs()
        self.ingredient_patterns = self._compile_ingredient_patterns()
        self.measurement_patterns = self._compile_measurement_patterns()
    
    def clean_recipe_text(self, text: str) -> str:
        """
        Clean and normalize recipe text.
        
        Args:
            text: Raw recipe text
            
        Returns:
            str: Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common recipe formatting issues
        text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1-\2', text)  # Fix ranges like "2 - 3" -> "2-3"
        text = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', text)  # Fix fractions like "1 / 2" -> "1/2"
        
        # Standardize temperature units
        text = re.sub(r'(\d+)\s*°?\s*F\b', r'\1°F', text)
        text = re.sub(r'(\d+)\s*°?\s*C\b', r'\1°C', text)
        
        return text.strip()
    
    def extract_ingredients(self, ingredient_text: str) -> List[Ingredient]:
        """
        Extract structured ingredients from text.
        
        Args:
            ingredient_text: Raw ingredient list text
            
        Returns:
            List[Ingredient]: Parsed ingredients with quantities and units
        """
        ingredients = []
        
        # Split by lines and clean
        lines = [line.strip() for line in ingredient_text.split('\n') if line.strip()]
        
        for line in lines:
            ingredient = self._parse_ingredient_line(line)
            if ingredient:
                ingredients.append(ingredient)
        
        return ingredients
    
    def _parse_ingredient_line(self, line: str) -> Optional[Ingredient]:
        """Parse a single ingredient line."""
        original_line = line
        line = line.lower().strip()
        
        # Skip empty lines or section headers
        if not line or line.endswith(':'):
            return None
        
        # Check if optional
        optional = 'optional' in line or line.startswith('(') and line.endswith(')')
        line = re.sub(r'\(.*optional.*\)', '', line).strip()
        
        # Extract quantity and unit
        quantity, unit, remaining = self._extract_quantity_unit(line)
        
        # Extract preparation method
        preparation = self._extract_preparation(remaining)
        
        # Clean ingredient name
        ingredient_name = self._clean_ingredient_name(remaining)
        
        if not ingredient_name:
            return None
        
        return Ingredient(
            name=ingredient_name,
            quantity=quantity,
            unit=unit,
            preparation=preparation,
            optional=optional,
            raw_text=original_line
        )
    
    def _extract_quantity_unit(self, text: str) -> Tuple[Optional[float], Optional[str], str]:
        """Extract quantity and unit from ingredient text."""
        # Pattern for quantity (including fractions)
        quantity_pattern = r'^(\d+(?:\.\d+)?(?:\s*[-/]\s*\d+(?:\.\d+)?)?|\d+\s+\d+/\d+|\d+/\d+)'
        
        match = re.match(quantity_pattern, text.strip())
        if not match:
            return None, None, text
        
        quantity_str = match.group(1)
        remaining = text[match.end():].strip()
        
        # Parse quantity (handle fractions)
        try:
            if '/' in quantity_str:
                # Handle mixed numbers like "1 1/2" or simple fractions like "1/2"
                if ' ' in quantity_str:
                    whole, frac = quantity_str.split(' ', 1)
                    quantity = float(whole) + float(Fraction(frac))
                else:
                    quantity = float(Fraction(quantity_str))
            elif '-' in quantity_str:
                # Handle ranges like "2-3", take the average
                parts = quantity_str.split('-')
                quantity = (float(parts[0]) + float(parts[1])) / 2
            else:
                quantity = float(quantity_str)
        except:
            quantity = None
        
        # Extract unit
        unit_pattern = r'^(' + '|'.join(self.units) + r')\b'
        unit_match = re.match(unit_pattern, remaining, re.IGNORECASE)
        
        if unit_match:
            unit = unit_match.group(1).lower()
            remaining = remaining[unit_match.end():].strip()
        else:
            unit = None
        
        return quantity, unit, remaining
    
    def _extract_preparation(self, text: str) -> Optional[str]:
        """Extract preparation method from ingredient text."""
        # Common preparation methods
        prep_methods = [
            'diced', 'chopped', 'minced', 'sliced', 'grated', 'shredded',
            'julienned', 'cubed', 'crushed', 'ground', 'whole', 'halved',
            'quartered', 'peeled', 'seeded', 'stemmed', 'trimmed'
        ]
        
        for prep in prep_methods:
            if prep in text.lower():
                return prep
        
        return None
    
    def _clean_ingredient_name(self, text: str) -> str:
        """Clean and extract the main ingredient name."""
        # Remove common words and phrases
        text = re.sub(r'\b(fresh|dried|frozen|canned|organic|raw|cooked)\b', '', text)
        text = re.sub(r'\b(for serving|to taste|as needed)\b', '', text)
        text = re.sub(r'[,\(\)]', '', text)
        
        return text.strip()
    
    def extract_cooking_techniques(self, text: str) -> List[str]:
        """
        Extract cooking techniques mentioned in text.
        
        Args:
            text: Recipe or cooking instruction text
            
        Returns:
            List[str]: Identified cooking techniques
        """
        techniques = []
        text_lower = text.lower()
        
        for verb in self.cooking_verbs:
            if verb in text_lower:
                techniques.append(verb)
        
        return list(set(techniques))  # Remove duplicates
    
    def extract_cooking_times(self, text: str) -> Dict[str, Optional[int]]:
        """
        Extract cooking times from recipe text.
        
        Args:
            text: Recipe text containing time information
            
        Returns:
            Dict[str, Optional[int]]: Prep time, cook time, total time in minutes
        """
        times = {"prep_time": None, "cook_time": None, "total_time": None}
        
        # Patterns for different time mentions
        time_patterns = {
            "prep_time": r'prep(?:aration)?\s*time:?\s*(\d+)\s*(min|minute|hour|hr)',
            "cook_time": r'cook(?:ing)?\s*time:?\s*(\d+)\s*(min|minute|hour|hr)',
            "total_time": r'total\s*time:?\s*(\d+)\s*(min|minute|hour|hr)'
        }
        
        for time_type, pattern in time_patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                value = int(match.group(1))
                unit = match.group(2)
                
                # Convert to minutes
                if unit in ['hour', 'hr']:
                    value *= 60
                
                times[time_type] = value
        
        return times
    
    def extract_servings(self, text: str) -> Optional[int]:
        """Extract serving size from recipe text."""
        patterns = [
            r'serves?\s*(\d+)',
            r'(\d+)\s*servings?',
            r'makes?\s*(\d+)\s*portions?',
            r'yield:?\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        
        return None
    
    def categorize_recipe_difficulty(self, recipe_text: str, num_ingredients: int, num_steps: int) -> str:
        """
        Categorize recipe difficulty based on various factors.
        
        Args:
            recipe_text: Full recipe text
            num_ingredients: Number of ingredients
            num_steps: Number of cooking steps
            
        Returns:
            str: Difficulty level (beginner, intermediate, advanced, professional)
        """
        difficulty_score = 0
        
        # Factor 1: Number of ingredients
        if num_ingredients > 15:
            difficulty_score += 2
        elif num_ingredients > 10:
            difficulty_score += 1
        
        # Factor 2: Number of steps
        if num_steps > 10:
            difficulty_score += 2
        elif num_steps > 6:
            difficulty_score += 1
        
        # Factor 3: Advanced techniques
        advanced_techniques = [
            'sous-vide', 'tempura', 'confit', 'flambé', 'julienne',
            'brunoise', 'chiffonade', 'emulsify', 'clarify', 'reduce'
        ]
        
        for technique in advanced_techniques:
            if technique in recipe_text.lower():
                difficulty_score += 1
        
        # Factor 4: Special equipment
        special_equipment = [
            'stand mixer', 'food processor', 'mandoline', 'thermometer',
            'pressure cooker', 'immersion blender', 'kitchen scale'
        ]
        
        for equipment in special_equipment:
            if equipment in recipe_text.lower():
                difficulty_score += 0.5
        
        # Determine difficulty level
        if difficulty_score >= 4:
            return "professional"
        elif difficulty_score >= 2.5:
            return "advanced"
        elif difficulty_score >= 1:
            return "intermediate"
        else:
            return "beginner"
    
    def detect_cuisine_type(self, recipe_text: str, ingredients: List[str]) -> Optional[str]:
        """
        Detect cuisine type from recipe content.
        
        Args:
            recipe_text: Full recipe text
            ingredients: List of ingredient names
            
        Returns:
            Optional[str]: Detected cuisine type
        """
        text_lower = recipe_text.lower()
        ingredients_lower = [ing.lower() for ing in ingredients]
        
        # Cuisine indicators
        cuisine_indicators = {
            "Italian": ["pasta", "parmesan", "basil", "oregano", "mozzarella", "prosciutto"],
            "Chinese": ["soy sauce", "ginger", "garlic", "sesame oil", "rice wine", "star anise"],
            "Indian": ["curry", "turmeric", "cumin", "coriander", "garam masala", "cardamom"],
            "Mexican": ["cilantro", "lime", "jalapeño", "cumin", "chili", "avocado"],
            "French": ["butter", "cream", "wine", "herbs de provence", "shallot", "cognac"],
            "Japanese": ["miso", "sake", "mirin", "nori", "wasabi", "dashi"],
            "Thai": ["fish sauce", "coconut milk", "lemongrass", "galangal", "thai basil"],
            "Mediterranean": ["olive oil", "feta", "olives", "tomato", "lemon", "rosemary"]
        }
        
        cuisine_scores = {}
        
        for cuisine, indicators in cuisine_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator in text_lower or any(indicator in ing for ing in ingredients_lower):
                    score += 1
            
            if score > 0:
                cuisine_scores[cuisine] = score
        
        # Return cuisine with highest score
        if cuisine_scores:
            return max(cuisine_scores, key=cuisine_scores.get)
        
        return None
    
    def extract_dietary_tags(self, recipe_text: str, ingredients: List[str]) -> List[str]:
        """
        Extract dietary restriction tags from recipe.
        
        Args:
            recipe_text: Full recipe text
            ingredients: List of ingredient names
            
        Returns:
            List[str]: Applicable dietary tags
        """
        tags = []
        text_lower = recipe_text.lower()
        ingredients_lower = [ing.lower() for ing in ingredients]
        
        # Check for meat/animal products
        meat_keywords = ['chicken', 'beef', 'pork', 'lamb', 'fish', 'salmon', 'tuna', 'shrimp']
        dairy_keywords = ['milk', 'cheese', 'butter', 'cream', 'yogurt']
        gluten_keywords = ['flour', 'wheat', 'bread', 'pasta', 'soy sauce']
        
        has_meat = any(keyword in text_lower or any(keyword in ing for ing in ingredients_lower) 
                      for keyword in meat_keywords)
        has_dairy = any(keyword in text_lower or any(keyword in ing for ing in ingredients_lower) 
                       for keyword in dairy_keywords)
        has_gluten = any(keyword in text_lower or any(keyword in ing for ing in ingredients_lower) 
                        for keyword in gluten_keywords)
        
        # Determine tags
        if not has_meat and not has_dairy:
            tags.append("vegan")
        elif not has_meat:
            tags.append("vegetarian")
        
        if not has_dairy:
            tags.append("dairy-free")
        
        if not has_gluten:
            tags.append("gluten-free")
        
        # Check for explicit mentions
        if "keto" in text_lower or "ketogenic" in text_lower:
            tags.append("keto")
        
        if "paleo" in text_lower or "paleolithic" in text_lower:
            tags.append("paleo")
        
        return tags
    
    def _load_cooking_units(self) -> List[str]:
        """Load common cooking units."""
        return [
            # Volume
            'cup', 'cups', 'tablespoon', 'tablespoons', 'tbsp', 'teaspoon', 'teaspoons', 'tsp',
            'fluid ounce', 'fluid ounces', 'fl oz', 'pint', 'pints', 'quart', 'quarts',
            'gallon', 'gallons', 'liter', 'liters', 'l', 'milliliter', 'milliliters', 'ml',
            
            # Weight
            'pound', 'pounds', 'lb', 'lbs', 'ounce', 'ounces', 'oz',
            'gram', 'grams', 'g', 'kilogram', 'kilograms', 'kg',
            
            # Count
            'piece', 'pieces', 'slice', 'slices', 'clove', 'cloves',
            'bunch', 'bunches', 'head', 'heads', 'can', 'cans',
            
            # Special
            'pinch', 'dash', 'splash', 'handful', 'to taste'
        ]
    
    def _load_cooking_verbs(self) -> List[str]:
        """Load cooking technique verbs."""
        return [
            'bake', 'roast', 'grill', 'fry', 'sauté', 'steam', 'boil', 'simmer',
            'braise', 'stew', 'poach', 'blanch', 'sear', 'caramelize', 'reduce',
            'whisk', 'fold', 'knead', 'marinate', 'season', 'garnish', 'serve',
            'chop', 'dice', 'mince', 'slice', 'grate', 'shred', 'julienne',
            'deglaze', 'emulsify', 'temper', 'proof', 'rest', 'chill'
        ]
    
    def _compile_ingredient_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for ingredient parsing."""
        return [
            re.compile(r'^(\d+(?:\.\d+)?(?:\s*[-/]\s*\d+(?:\.\d+)?)?)\s*(.+)'),
            re.compile(r'^(\d+\s+\d+/\d+)\s*(.+)'),
            re.compile(r'^(\d+/\d+)\s*(.+)')
        ]
    
    def _compile_measurement_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for measurement extraction."""
        units_pattern = '|'.join(self.units)
        return [
            re.compile(f r'(\d+(?:\.\d+)?)\s*({units_pattern})\b', re.IGNORECASE),
            re.compile(r'(\d+)\s*°\s*([CF])\b'),  # Temperature
            re.compile(r'(\d+)\s*(minutes?|mins?|hours?|hrs?)\b', re.IGNORECASE)  # Time
        ]


# Global processor instance
culinary_processor = CulinaryTextProcessor()
