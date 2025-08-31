# 🍳 CulinaryGenius RAG - AI-Powered Culinary Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent culinary assistant powered by advanced RAG (Retrieval-Augmented Generation) technology. Discover recipes, learn cooking techniques, and get personalized culinary guidance from cuisines around the world! 🌍👨‍🍳

## ✨ Unique Features

- 🌍 **Global Recipe Database**: 1000+ recipes from 50+ countries with cultural context
- 🔄 **Smart Ingredient Substitution**: AI-powered alternatives based on availability and dietary needs
- 📚 **Cooking Technique Library**: Step-by-step guides for cooking methods worldwide
- 🥗 **Nutritional Intelligence**: Automatic nutritional analysis and health insights
- 🌱 **Dietary Adaptation**: Support for vegan, vegetarian, gluten-free, keto, and more
- 🎯 **Skill-Level Matching**: Recipes adapted to your cooking experience
- 🍂 **Seasonal Recommendations**: Recipes based on seasonal ingredient availability
- 🏷️ **Cultural Context**: Learn the history and traditions behind each dish

## 🚀 Quick Demo

```bash
# Ask for a recipe
curl -X POST "http://localhost:8000/recipe/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "spicy Indian curry with chicken", "dietary_restrictions": ["gluten-free"], "skill_level": "intermediate"}'

# Get cooking technique help
curl -X POST "http://localhost:8000/technique/help" \
  -H "Content-Type: application/json" \
  -d '{"technique": "tempura frying", "difficulty": "beginner"}'

# Smart ingredient substitution
curl -X POST "http://localhost:8000/ingredients/substitute" \
  -H "Content-Type: application/json" \
  -d '{"ingredient": "heavy cream", "dietary_restrictions": ["vegan"], "recipe_type": "pasta"}'
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│ Culinary RAG    │───▶│ Personalized    │
│  "Vegan pasta"  │    │    Pipeline     │    │    Recipe       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Multi-Source   │
                    │  Knowledge Base │
                    │                 │
                    │ • Recipes       │
                    │ • Techniques    │
                    │ • Nutrition     │
                    │ • Culture       │
                    └─────────────────┘
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/culinary-genius-rag.git
cd culinary-genius-rag

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your OpenAI API key to .env

# Initialize the system
python scripts/setup_system.py
```

## 🎮 Usage Examples

### Recipe Discovery
```python
from culinary_genius import CulinaryRAG

rag = CulinaryRAG()

# Find recipes based on ingredients you have
response = await rag.find_recipes(
    ingredients=["chicken", "tomatoes", "basil"],
    cuisine="Italian",
    dietary_restrictions=["gluten-free"],
    cooking_time="30 minutes"
)

print(response.recipe)
print(f"Difficulty: {response.difficulty}")
print(f"Nutrition: {response.nutrition_info}")
```

### Cooking Assistance
```python
# Get help with cooking techniques
help_response = await rag.get_cooking_help(
    query="How do I properly sear a steak?",
    skill_level="beginner"
)

print(help_response.instructions)
print(f"Tips: {help_response.pro_tips}")
```

### Meal Planning
```python
# Generate weekly meal plan
meal_plan = await rag.create_meal_plan(
    days=7,
    dietary_preferences=["vegetarian", "high-protein"],
    cuisine_variety=["Mediterranean", "Asian", "Mexican"]
)

for day, meals in meal_plan.items():
    print(f"{day}: {meals}")
```

## 📊 Data Sources

Our knowledge base includes:

- **Recipe Collections**: 1000+ recipes from authentic sources
- **Cooking Techniques**: Professional chef tutorials and guides  
- **Nutritional Database**: USDA nutrition data and health information
- **Cultural Context**: Food history and traditions from around the world
- **Ingredient Information**: Seasonal availability, substitutions, storage tips

## 🌟 Advanced Features

### Smart Substitutions
- Dietary restriction-aware replacements
- Flavor profile matching
- Texture and cooking behavior analysis
- Regional availability considerations

### Nutritional Intelligence
- Automatic macro/micronutrient calculation
- Health goal alignment (weight loss, muscle gain, etc.)
- Allergen detection and warnings
- Portion size optimization

### Cultural Learning
- Food history and origins
- Traditional cooking methods
- Regional variations and adaptations
- Festival and celebration foods

## 🚀 Getting Started

1. **Set up your environment**
2. **Run the data collection pipeline**
3. **Generate embeddings for the culinary knowledge base**
4. **Start the API server**
5. **Begin your culinary journey!**

Ready to revolutionize your cooking experience? Let's build something amazing! 🎉
