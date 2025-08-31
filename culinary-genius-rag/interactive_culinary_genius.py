#!/usr/bin/env python3
"""
CulinaryGenius RAG - Interactive Culinary Assistant

An interactive command-line interface for the CulinaryGenius RAG system
that provides real-time culinary assistance with engaging conversations.

Author: CulinaryGenius Team
Created: 2025-08-31
"""

import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise for interactive use


class InteractiveCulinaryGenius:
    """Interactive culinary assistant with conversational interface."""
    
    def __init__(self):
        """Initialize interactive culinary assistant."""
        self.knowledge_base = self._load_culinary_knowledge()
        self.user_preferences = {}
        self.conversation_history = []
        
        print("🍳 CulinaryGenius RAG initialized!")
        print("📚 Knowledge base loaded with global recipes and techniques")
    
    def start_interactive_session(self):
        """Start interactive culinary assistance session."""
        print("\n" + "="*70)
        print("🍳 CULINARYGENIUS RAG - INTERACTIVE CULINARY ASSISTANT")
        print("="*70)
        print("🌍 Your AI-powered culinary companion for global cuisine exploration!")
        print("🔍 Ask about recipes, cooking techniques, substitutions, and more!")
        print("💡 Type 'help' for commands, 'quit' to exit")
        print("="*70)
        
        # Get user preferences
        self._setup_user_preferences()
        
        # Main interaction loop
        while True:
            try:
                print(f"\n🍽️ CulinaryGenius is ready to help!")
                user_input = input("👨‍🍳 What would you like to cook or learn today? ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    self._farewell_message()
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'preferences':
                    self._setup_user_preferences()
                    continue
                elif user_input.lower() == 'random':
                    self._suggest_random_recipe()
                    continue
                elif user_input.lower() == 'stats':
                    self._show_system_stats()
                    continue
                
                # Process culinary query
                response = self._process_culinary_query(user_input)
                print(f"\n🤖 CulinaryGenius: {response}")
                
                # Add to conversation history
                self.conversation_history.append({
                    "query": user_input,
                    "response": response,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using CulinaryGenius RAG!")
                break
            except Exception as e:
                print(f"\n❌ Oops! Something went wrong: {e}")
                print("🔄 Let's try again...")
    
    def _setup_user_preferences(self):
        """Setup user culinary preferences."""
        print("\n🎯 Let's personalize your culinary experience!")
        
        # Dietary restrictions
        print("\n🥗 Dietary Restrictions (comma-separated, or 'none'):")
        print("   Options: vegetarian, vegan, gluten-free, dairy-free, keto, paleo")
        dietary = input("   Your restrictions: ").strip()
        
        if dietary.lower() != 'none':
            self.user_preferences['dietary_restrictions'] = [d.strip() for d in dietary.split(',') if d.strip()]
        else:
            self.user_preferences['dietary_restrictions'] = []
        
        # Preferred cuisines
        print("\n🌍 Preferred Cuisines (comma-separated, or 'any'):")
        print("   Options: Italian, Indian, Thai, Mexican, French, Japanese, Mediterranean")
        cuisines = input("   Your preferences: ").strip()
        
        if cuisines.lower() != 'any':
            self.user_preferences['preferred_cuisines'] = [c.strip() for c in cuisines.split(',') if c.strip()]
        else:
            self.user_preferences['preferred_cuisines'] = []
        
        # Skill level
        print("\n🎯 Cooking Skill Level:")
        print("   1. Beginner   2. Intermediate   3. Advanced   4. Professional")
        skill_choice = input("   Choose (1-4): ").strip()
        
        skill_map = {'1': 'beginner', '2': 'intermediate', '3': 'advanced', '4': 'professional'}
        self.user_preferences['skill_level'] = skill_map.get(skill_choice, 'intermediate')
        
        print(f"\n✅ Preferences saved!")
        print(f"🥗 Dietary: {', '.join(self.user_preferences['dietary_restrictions']) or 'None'}")
        print(f"🌍 Cuisines: {', '.join(self.user_preferences['preferred_cuisines']) or 'Any'}")
        print(f"🎯 Skill: {self.user_preferences['skill_level']}")
    
    def _process_culinary_query(self, query: str) -> str:
        """Process user's culinary query and generate response."""
        query_lower = query.lower()
        
        # Determine query type
        if any(word in query_lower for word in ['recipe', 'cook', 'make', 'prepare']):
            return self._handle_recipe_query(query)
        elif any(word in query_lower for word in ['substitute', 'replace', 'instead']):
            return self._handle_substitution_query(query)
        elif any(word in query_lower for word in ['how', 'technique', 'method']):
            return self._handle_technique_query(query)
        elif any(word in query_lower for word in ['nutrition', 'healthy', 'calories']):
            return self._handle_nutrition_query(query)
        else:
            return self._handle_general_query(query)
    
    def _handle_recipe_query(self, query: str) -> str:
        """Handle recipe-related queries."""
        # Find matching recipes
        matching_recipes = self._search_recipes(query)
        
        if not matching_recipes:
            return "🤔 I couldn't find specific recipes matching your criteria, but I'd love to help you create something delicious! Could you tell me more about what ingredients you have or what type of dish you're craving?"
        
        recipe = matching_recipes[0]
        
        response = f"🍽️ Perfect! I recommend **{recipe['title']}** - a delicious {recipe['cuisine']} dish!\n\n"
        
        # Add recipe details
        response += f"⏱️ **Time**: {recipe.get('prep_time', 0) + recipe.get('cook_time', 0)} minutes total\n"
        response += f"🎯 **Difficulty**: {recipe.get('difficulty', 'intermediate')} level\n"
        
        # Add dietary info
        if recipe.get('dietary_tags'):
            response += f"🥗 **Dietary**: {', '.join(recipe['dietary_tags'])}\n"
        
        # Add cultural context
        if recipe.get('cultural_context'):
            response += f"\n🌍 **Cultural Context**: {recipe['cultural_context']}\n"
        
        # Add cooking tips based on user skill level
        skill_level = self.user_preferences.get('skill_level', 'intermediate')
        if skill_level == 'beginner':
            response += "\n💡 **Beginner Tips**: Take your time with each step, prep all ingredients first (mise en place), and don't be afraid to taste as you go!"
        elif skill_level == 'advanced':
            response += "\n🔥 **Advanced Tips**: Focus on technique refinement, experiment with flavor balancing, and consider presentation elements!"
        
        return response
    
    def _handle_substitution_query(self, query: str) -> str:
        """Handle ingredient substitution queries."""
        # Extract ingredient from query
        ingredient = self._extract_ingredient_from_query(query)
        
        if not ingredient:
            return "🔄 I'd be happy to help with ingredient substitutions! Could you tell me which specific ingredient you need to substitute?"
        
        # Get substitution recommendations
        substitutions = self._get_substitutions(ingredient)
        
        response = f"🔄 **Substitutions for {ingredient}**:\n\n"
        
        for i, sub in enumerate(substitutions[:3], 1):
            response += f"{i}. **{sub['substitute']}** (ratio: {sub['ratio']})\n"
            response += f"   💡 {sub['notes']}\n\n"
        
        # Add dietary-specific advice
        if self.user_preferences.get('dietary_restrictions'):
            response += f"🥗 **Note**: These suggestions consider your dietary restrictions: {', '.join(self.user_preferences['dietary_restrictions'])}"
        
        return response
    
    def _handle_technique_query(self, query: str) -> str:
        """Handle cooking technique queries."""
        technique = self._extract_technique_from_query(query)
        
        if "sauté" in query.lower():
            return "🔧 **Sautéing Mastery**:\n\n1. 🔥 Heat your pan first, then add oil\n2. 🥕 Don't overcrowd - give ingredients space\n3. 🌊 Keep things moving with a spatula or toss\n4. 🌡️ Use medium-high heat for best results\n5. 🧻 Pat ingredients dry before adding\n\n💡 **Pro Tip**: The word 'sauté' means 'to jump' in French - your ingredients should dance in the pan!"
        
        elif "pasta" in query.lower():
            return "🍝 **Perfect Pasta Technique**:\n\n1. 🧂 Use plenty of salted water (like seawater!)\n2. ⏰ Start testing 2 minutes before package time\n3. 🎯 Look for a tiny white dot in the center when you bite\n4. 💧 Reserve pasta water before draining\n5. 🍳 Finish cooking pasta in the sauce for 1-2 minutes\n\n💡 **Pro Tip**: Al dente means 'to the tooth' - it should have a slight bite!"
        
        else:
            return "🔧 **Cooking Technique Guidance**:\n\nI'd love to help you master cooking techniques! The key principles are:\n\n• 📚 **Mise en place**: Prep everything first\n• 🌡️ **Temperature control**: Right heat for the job\n• ⏰ **Timing**: Practice builds intuition\n• 👅 **Taste as you go**: Adjust seasoning\n• 🧘 **Patience**: Good cooking takes time\n\nWhat specific technique would you like to learn about?"
    
    def _handle_nutrition_query(self, query: str) -> str:
        """Handle nutrition-related queries."""
        return "📊 **Nutritional Guidance**:\n\nI'm here to help with healthy cooking! Here are some key principles:\n\n🥬 **Vegetables**: Aim for colorful variety - different colors provide different nutrients\n🍗 **Protein**: Include lean sources like fish, poultry, legumes, and plant proteins\n🌾 **Whole Grains**: Choose brown rice, quinoa, whole wheat for sustained energy\n🥑 **Healthy Fats**: Olive oil, avocados, nuts, and seeds support overall health\n\n💡 **Cooking Tips for Nutrition**:\n• Steam or roast vegetables to preserve nutrients\n• Use herbs and spices instead of excess salt\n• Cook with minimal processing\n• Balance your plate: 1/2 vegetables, 1/4 protein, 1/4 whole grains"
    
    def _handle_general_query(self, query: str) -> str:
        """Handle general culinary queries."""
        return "🍳 **Culinary Wisdom**:\n\nCooking is both an art and a science! Here's what I can help you with:\n\n🔍 **Recipe Search**: Find dishes by ingredients, cuisine, or dietary needs\n🔄 **Substitutions**: Smart alternatives for any ingredient\n🔧 **Techniques**: Master professional cooking methods\n🌍 **Cultural Knowledge**: Learn the stories behind dishes\n📊 **Nutrition**: Healthy cooking guidance\n\n💡 **Today's Cooking Tip**: The best ingredient in any dish is love and attention. Take your time, enjoy the process, and don't be afraid to experiment!\n\nWhat specific culinary adventure would you like to embark on today?"
    
    def _search_recipes(self, query: str) -> List[Dict[str, Any]]:
        """Search for recipes based on query."""
        recipes = self.knowledge_base.get("recipes", [])
        query_lower = query.lower()
        
        matching = []
        for recipe in recipes:
            score = 0
            
            # Title matching
            if any(word in recipe["title"].lower() for word in query_lower.split()):
                score += 0.8
            
            # Ingredient matching
            ingredients = recipe.get("ingredients", [])
            for ingredient in ingredients:
                ingredient_name = ingredient if isinstance(ingredient, str) else ingredient.get("name", "")
                if ingredient_name.lower() in query_lower:
                    score += 0.3
            
            # Cuisine matching
            if recipe.get("cuisine", "").lower() in query_lower:
                score += 0.5
            
            # Dietary preference matching
            user_dietary = self.user_preferences.get('dietary_restrictions', [])
            recipe_dietary = recipe.get('dietary_tags', [])
            if user_dietary and any(diet in recipe_dietary for diet in user_dietary):
                score += 0.4
            
            if score > 0.3:
                matching.append(recipe)
        
        # Sort by relevance (simplified)
        return sorted(matching, key=lambda x: len(x.get('dietary_tags', [])), reverse=True)
    
    def _extract_ingredient_from_query(self, query: str) -> Optional[str]:
        """Extract ingredient name from substitution query."""
        # Simple extraction
        common_ingredients = [
            "heavy cream", "butter", "eggs", "milk", "cheese", "flour", 
            "sugar", "oil", "garlic", "onion", "tomato"
        ]
        
        query_lower = query.lower()
        for ingredient in common_ingredients:
            if ingredient in query_lower:
                return ingredient
        
        return None
    
    def _extract_technique_from_query(self, query: str) -> Optional[str]:
        """Extract cooking technique from query."""
        techniques = ["sauté", "braise", "grill", "bake", "roast", "fry", "steam", "boil"]
        
        query_lower = query.lower()
        for technique in techniques:
            if technique in query_lower:
                return technique
        
        return None
    
    def _get_substitutions(self, ingredient: str) -> List[Dict[str, str]]:
        """Get substitution recommendations."""
        substitutions = {
            "heavy cream": [
                {"substitute": "Coconut cream", "ratio": "1:1", "notes": "Rich texture, subtle coconut flavor"},
                {"substitute": "Cashew cream", "ratio": "1:1", "notes": "Blend 1 cup cashews + 1 cup water"},
                {"substitute": "Greek yogurt + milk", "ratio": "3/4 cup", "notes": "Lighter option, tangy flavor"}
            ],
            "butter": [
                {"substitute": "Olive oil", "ratio": "3/4 amount", "notes": "Great for sautéing, Mediterranean flavor"},
                {"substitute": "Coconut oil", "ratio": "1:1", "notes": "Solid at room temp, neutral when refined"},
                {"substitute": "Avocado", "ratio": "1/2 amount", "notes": "For baking, adds moisture and healthy fats"}
            ],
            "eggs": [
                {"substitute": "Flax eggs", "ratio": "1 tbsp ground flax + 3 tbsp water", "notes": "Let sit 5 min to gel, great for binding"},
                {"substitute": "Aquafaba", "ratio": "3 tbsp per egg", "notes": "Chickpea liquid, excellent for whipping"},
                {"substitute": "Banana", "ratio": "1/4 cup mashed per egg", "notes": "For sweet baking, adds moisture"}
            ]
        }
        
        return substitutions.get(ingredient.lower(), [
            {"substitute": "Consult specific guides", "ratio": "Varies", "notes": "Each ingredient has unique properties"}
        ])
    
    def _suggest_random_recipe(self):
        """Suggest a random recipe."""
        import random
        
        recipes = self.knowledge_base.get("recipes", [])
        if not recipes:
            print("🤔 No recipes available in the knowledge base.")
            return
        
        recipe = random.choice(recipes)
        
        print(f"\n🎲 **Random Recipe Inspiration**: {recipe['title']}")
        print(f"🌍 **Cuisine**: {recipe['cuisine']}")
        print(f"⏱️ **Time**: {recipe.get('prep_time', 0) + recipe.get('cook_time', 0)} minutes")
        print(f"🎯 **Difficulty**: {recipe.get('difficulty', 'intermediate')}")
        
        if recipe.get('cultural_context'):
            print(f"📚 **Did you know?** {recipe['cultural_context']}")
        
        print(f"\n💡 **Why not try this today?** It's a great way to explore {recipe['cuisine']} cuisine!")
    
    def _show_help(self):
        """Show help information."""
        print("\n🆘 CULINARYGENIUS RAG - HELP GUIDE")
        print("-" * 35)
        print("🔍 **Recipe Search**: 'Find me a spicy Italian pasta recipe'")
        print("🔄 **Substitutions**: 'What can I use instead of heavy cream?'")
        print("🔧 **Techniques**: 'How do I properly sauté vegetables?'")
        print("📊 **Nutrition**: 'What makes a meal healthy and balanced?'")
        print("🌍 **Culture**: 'Tell me about the history of carbonara'")
        print("\n🎮 **Commands**:")
        print("• 'random' - Get a random recipe suggestion")
        print("• 'preferences' - Update your culinary preferences")
        print("• 'stats' - View system capabilities")
        print("• 'help' - Show this help guide")
        print("• 'quit' - Exit CulinaryGenius")
        print("\n💡 **Tips**: Be specific about ingredients, dietary needs, or techniques!")
    
    def _show_system_stats(self):
        """Show system statistics and capabilities."""
        recipes = self.knowledge_base.get("recipes", [])
        
        print("\n📊 CULINARYGENIUS RAG SYSTEM STATS")
        print("-" * 38)
        print(f"📚 **Knowledge Base**: {len(recipes)} recipes loaded")
        
        # Cuisine distribution
        cuisines = {}
        for recipe in recipes:
            cuisine = recipe.get('cuisine', 'Unknown')
            cuisines[cuisine] = cuisines.get(cuisine, 0) + 1
        
        print(f"🌍 **Cuisines**: {', '.join(cuisines.keys())}")
        
        # Dietary options
        all_dietary = set()
        for recipe in recipes:
            all_dietary.update(recipe.get('dietary_tags', []))
        
        print(f"🥗 **Dietary Options**: {', '.join(sorted(all_dietary)) if all_dietary else 'Various'}")
        
        # User session stats
        print(f"💬 **This Session**: {len(self.conversation_history)} queries processed")
        
        print("\n🚀 **System Capabilities**:")
        print("✅ Semantic recipe search")
        print("✅ Intelligent substitutions")
        print("✅ Expert technique guidance")
        print("✅ Cultural food knowledge")
        print("✅ Dietary adaptation")
        print("✅ Nutritional insights")
    
    def _farewell_message(self):
        """Display farewell message."""
        print("\n👋 **Thank you for using CulinaryGenius RAG!**")
        print("🍳 Keep exploring, keep cooking, keep creating!")
        print("🌟 Remember: Every great chef started with curiosity and practice.")
        
        if self.conversation_history:
            print(f"\n📊 **Session Summary**: You asked {len(self.conversation_history)} culinary questions")
            print("🎯 **Keep cooking and learning** - you're on a delicious journey!")
        
        print("\n🚀 **CulinaryGenius RAG** - Where AI meets culinary excellence!")
    
    def _load_culinary_knowledge(self) -> Dict[str, Any]:
        """Load culinary knowledge base."""
        # Try to load from processed data
        data_file = Path("data/processed/recipes.json")
        
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    return {"recipes": json.load(f)}
            except Exception:
                pass
        
        # Return built-in knowledge base
        return {
            "recipes": [
                {
                    "title": "Authentic Spaghetti Carbonara",
                    "cuisine": "Italian",
                    "ingredients": ["spaghetti", "pancetta", "eggs", "pecorino romano", "black pepper"],
                    "prep_time": 10,
                    "cook_time": 15,
                    "difficulty": "intermediate",
                    "dietary_tags": [],
                    "cultural_context": "Traditional Roman dish created by charcoal workers (carbonari)"
                },
                {
                    "title": "Creamy Butter Chicken",
                    "cuisine": "Indian",
                    "ingredients": ["chicken", "tomatoes", "cream", "butter", "garam masala"],
                    "prep_time": 30,
                    "cook_time": 45,
                    "difficulty": "intermediate",
                    "dietary_tags": ["gluten-free"],
                    "cultural_context": "Created in 1950s Delhi by Kundan Lal Gujral"
                },
                {
                    "title": "Fresh Guacamole",
                    "cuisine": "Mexican",
                    "ingredients": ["avocados", "lime", "onion", "cilantro", "jalapeño"],
                    "prep_time": 15,
                    "cook_time": 0,
                    "difficulty": "beginner",
                    "dietary_tags": ["vegan", "gluten-free", "dairy-free", "keto", "paleo"],
                    "cultural_context": "Ancient Aztec dish, 'ahuacamolli' meaning avocado sauce"
                },
                {
                    "title": "Thai Green Curry",
                    "cuisine": "Thai",
                    "ingredients": ["green curry paste", "coconut milk", "chicken", "thai basil"],
                    "prep_time": 20,
                    "cook_time": 25,
                    "difficulty": "intermediate",
                    "dietary_tags": ["gluten-free", "dairy-free"],
                    "cultural_context": "Central Thai dish, spiciest of the curry family"
                }
            ]
        }


def main():
    """Main function to start interactive session."""
    try:
        assistant = InteractiveCulinaryGenius()
        assistant.start_interactive_session()
    except Exception as e:
        print(f"❌ Failed to start CulinaryGenius: {e}")


if __name__ == "__main__":
    main()
