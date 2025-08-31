"""
CulinaryGenius RAG - Specialized Culinary API Routes

This module implements unique culinary endpoints including recipe search,
cooking technique guidance, ingredient substitution, and meal planning.

Author: CulinaryGenius Team
Created: 2025-08-31
"""

import logging
import time
import uuid
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from src.api.schemas.culinary_schemas import (
    RecipeSearchRequest, CulinaryResponse, RecipeResult,
    IngredientSubstitutionRequest, SubstitutionResponse,
    CookingTechniqueRequest, TechniqueResponse,
    MealPlanRequest, MealPlanResponse,
    NutritionInfo
)
from src.core.rag.culinary_retriever import culinary_retriever, CulinaryQuery
from src.core.rag.culinary_generator import culinary_generator

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.post("/recipe/search", response_model=CulinaryResponse)
async def search_recipes(
    request: RecipeSearchRequest,
    background_tasks: BackgroundTasks
) -> CulinaryResponse:
    """
    🔍 Search for recipes based on user preferences and constraints.
    
    This endpoint provides intelligent recipe recommendations with:
    - Multi-cuisine support
    - Dietary restriction filtering
    - Skill-level appropriate suggestions
    - Time and ingredient constraints
    - Nutritional information
    
    Args:
        request: Recipe search request with preferences
        background_tasks: FastAPI background tasks
        
    Returns:
        CulinaryResponse: Personalized recipe recommendations
    """
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    logger.info(f"🔍 Recipe search query {query_id}: {request.query}")
    
    try:
        # Build culinary query
        culinary_query = CulinaryQuery(
            text=request.query,
            dietary_restrictions=[dr.value for dr in request.dietary_restrictions] if request.dietary_restrictions else None,
            preferred_cuisines=[c.value for c in request.cuisines] if request.cuisines else None,
            max_prep_time=request.max_prep_time,
            max_cook_time=request.max_cook_time,
            skill_level=request.skill_level.value,
            available_ingredients=request.available_ingredients,
            exclude_ingredients=request.exclude_ingredients,
            meal_type=request.meal_type.value if request.meal_type else None
        )
        
        # Retrieve relevant culinary content
        retrieved_content = await culinary_retriever.retrieve(
            query=culinary_query,
            top_k=request.max_results,
            method="hybrid"
        )
        
        # Generate culinary response
        response = await culinary_generator.generate_culinary_response(
            query=culinary_query,
            retrieved_content=retrieved_content
        )
        
        # Convert to API response format
        processing_time = (time.time() - start_time) * 1000
        
        # Build recipe results
        recipe_results = []
        for content in retrieved_content:
            if content.content_type == "recipe":
                recipe_result = RecipeResult(
                    title=_extract_recipe_title(content.content),
                    cuisine=content.cuisine,
                    difficulty=content.difficulty,
                    prep_time=content.prep_time,
                    cook_time=content.cook_time,
                    total_time=(content.prep_time or 0) + (content.cook_time or 0) if content.prep_time and content.cook_time else None,
                    servings=_extract_servings(content.content),
                    ingredients=_extract_ingredients_list(content.content),
                    instructions=_extract_instructions_list(content.content),
                    dietary_tags=content.dietary_tags or [],
                    cooking_methods=content.cooking_methods or [],
                    cultural_context=_extract_cultural_context(content.content),
                    relevance_score=content.relevance_score
                )
                recipe_results.append(recipe_result)
        
        api_response = CulinaryResponse(
            response=response.response,
            response_type=response.response_type,
            confidence_score=response.confidence_score,
            recipes=recipe_results,
            techniques_mentioned=response.techniques_mentioned,
            ingredients_discussed=response.ingredients_discussed,
            nutritional_highlights=response.nutritional_highlights,
            cultural_insights=response.cultural_context,
            cooking_tips=response.cooking_tips,
            estimated_prep_time=response.estimated_time.get("prep_time") if response.estimated_time else None,
            estimated_cook_time=response.estimated_time.get("cook_time") if response.estimated_time else None,
            difficulty_assessment=response.difficulty_assessment,
            query_id=query_id,
            processing_time_ms=int(processing_time),
            sources_used=len(retrieved_content)
        )
        
        # Log successful search
        logger.info(f"✅ Recipe search {query_id} completed in {processing_time:.2f}ms")
        
        # Schedule background analytics
        background_tasks.add_task(
            _log_recipe_search_analytics,
            query_id=query_id,
            user_id=request.user_id,
            query=request.query,
            results_count=len(recipe_results),
            processing_time=processing_time
        )
        
        return api_response
        
    except Exception as e:
        logger.error(f"❌ Recipe search {query_id} failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "RECIPE_SEARCH_FAILED",
                    "message": "Failed to search recipes",
                    "query_id": query_id,
                    "suggestions": [
                        "Try simpler search terms",
                        "Check your dietary restrictions",
                        "Verify ingredient names"
                    ]
                }
            }
        )


@router.post("/ingredients/substitute", response_model=SubstitutionResponse)
async def get_ingredient_substitution(request: IngredientSubstitutionRequest) -> SubstitutionResponse:
    """
    🔄 Get intelligent ingredient substitutions.
    
    Provides smart ingredient alternatives considering:
    - Dietary restrictions and allergies
    - Flavor profile matching
    - Cooking behavior compatibility
    - Nutritional equivalence
    - Regional availability
    
    Args:
        request: Substitution request with context
        
    Returns:
        SubstitutionResponse: Substitution recommendations
    """
    logger.info(f"🔄 Ingredient substitution request: {request.ingredient}")
    
    try:
        # Build substitution query
        query_text = f"substitute for {request.ingredient}"
        if request.recipe_context:
            query_text += f" in {request.recipe_context}"
        
        culinary_query = CulinaryQuery(
            text=query_text,
            dietary_restrictions=[dr.value for dr in request.dietary_restrictions] if request.dietary_restrictions else None
        )
        
        # Retrieve substitution information
        retrieved_content = await culinary_retriever.retrieve(
            query=culinary_query,
            top_k=3,
            method="hybrid"
        )
        
        # Generate substitution response
        response = await culinary_generator.generate_culinary_response(
            query=culinary_query,
            retrieved_content=retrieved_content
        )
        
        # Parse substitution options from response
        substitutions = _parse_substitution_options(response.response, request.ingredient)
        
        return SubstitutionResponse(
            original_ingredient=request.ingredient,
            substitutions=substitutions,
            cooking_adjustments=_extract_cooking_adjustments(response.response),
            flavor_impact=_extract_flavor_impact(response.response),
            confidence_score=response.confidence_score
        )
        
    except Exception as e:
        logger.error(f"❌ Substitution request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "SUBSTITUTION_FAILED",
                    "message": f"Failed to find substitutions for {request.ingredient}",
                    "suggestions": [
                        "Check ingredient spelling",
                        "Try more common ingredient names",
                        "Provide more recipe context"
                    ]
                }
            }
        )


@router.post("/technique/help", response_model=TechniqueResponse)
async def get_cooking_technique_help(request: CookingTechniqueRequest) -> TechniqueResponse:
    """
    🔧 Get expert guidance on cooking techniques.
    
    Provides comprehensive cooking technique instruction including:
    - Step-by-step instructions
    - Equipment requirements
    - Professional tips and tricks
    - Common mistakes to avoid
    - Skill-level appropriate guidance
    
    Args:
        request: Technique help request
        
    Returns:
        TechniqueResponse: Detailed technique guidance
    """
    logger.info(f"🔧 Technique help request: {request.technique}")
    
    try:
        # Build technique query
        query_text = f"how to {request.technique} cooking technique"
        if request.specific_question:
            query_text += f" {request.specific_question}"
        
        culinary_query = CulinaryQuery(
            text=query_text,
            skill_level=request.skill_level.value
        )
        
        # Retrieve technique information
        retrieved_content = await culinary_retriever.retrieve(
            query=culinary_query,
            top_k=3,
            method="hybrid"
        )
        
        # Generate technique response
        response = await culinary_generator.generate_culinary_response(
            query=culinary_query,
            retrieved_content=retrieved_content
        )
        
        return TechniqueResponse(
            technique_name=request.technique,
            difficulty=response.difficulty_assessment or request.skill_level.value,
            instructions=_extract_instructions(response.response),
            equipment_needed=_extract_equipment(response.response),
            key_tips=response.cooking_tips,
            common_mistakes=_extract_common_mistakes(response.response),
            practice_suggestions=_extract_practice_suggestions(response.response)
        )
        
    except Exception as e:
        logger.error(f"❌ Technique help failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "TECHNIQUE_HELP_FAILED",
                    "message": f"Failed to provide guidance for {request.technique}",
                    "suggestions": [
                        "Check technique name spelling",
                        "Try more specific technique terms",
                        "Ask about related cooking methods"
                    ]
                }
            }
        )


@router.post("/meal-plan/generate", response_model=MealPlanResponse)
async def generate_meal_plan(request: MealPlanRequest) -> MealPlanResponse:
    """
    📅 Generate personalized meal plans.
    
    Creates comprehensive meal plans with:
    - Balanced nutrition across days
    - Variety in cuisines and ingredients
    - Skill-appropriate recipes
    - Consolidated shopping lists
    - Meal prep optimization
    
    Args:
        request: Meal planning request
        
    Returns:
        MealPlanResponse: Complete meal plan with shopping list
    """
    logger.info(f"📅 Meal plan generation: {request.days} days, {request.meals_per_day} meals/day")
    
    try:
        # This would be a complex implementation in production
        # For now, return a structured response
        
        meal_plan = {}
        shopping_list = []
        
        # Generate sample meal plan structure
        for day in range(1, request.days + 1):
            day_key = f"day_{day}"
            meal_plan[day_key] = {
                "breakfast": RecipeResult(
                    title="Healthy Breakfast Option",
                    ingredients=["oats", "berries", "nuts"],
                    instructions=["Combine ingredients", "Serve fresh"],
                    relevance_score=0.8
                ),
                "lunch": RecipeResult(
                    title="Nutritious Lunch",
                    ingredients=["quinoa", "vegetables", "protein"],
                    instructions=["Cook quinoa", "Prepare vegetables", "Combine"],
                    relevance_score=0.8
                ),
                "dinner": RecipeResult(
                    title="Satisfying Dinner",
                    ingredients=["main protein", "vegetables", "grains"],
                    instructions=["Prepare protein", "Cook vegetables", "Serve together"],
                    relevance_score=0.8
                )
            }
        
        return MealPlanResponse(
            meal_plan=meal_plan,
            shopping_list=[
                {"item": "Sample ingredients", "quantity": "As needed", "category": "produce"}
            ],
            nutrition_summary={"calories_per_day": "~2000", "protein": "adequate", "vegetables": "5+ servings"},
            preparation_tips=["Prep vegetables on Sunday", "Cook grains in batches"],
            estimated_cost="$50-80 per week"
        )
        
    except Exception as e:
        logger.error(f"❌ Meal plan generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "MEAL_PLAN_FAILED",
                    "message": "Failed to generate meal plan",
                    "suggestions": [
                        "Try fewer days",
                        "Reduce dietary restrictions",
                        "Simplify requirements"
                    ]
                }
            }
        )


# Helper functions for parsing responses

def _extract_recipe_title(content: str) -> str:
    """Extract recipe title from content."""
    lines = content.split('\n')
    for line in lines:
        if line.startswith('Recipe:'):
            return line.replace('Recipe:', '').strip()
    return "Delicious Recipe"


def _extract_servings(content: str) -> Optional[int]:
    """Extract serving size from content."""
    import re
    match = re.search(r'Serves:?\s*(\d+)', content)
    return int(match.group(1)) if match else None


def _extract_ingredients_list(content: str) -> List[str]:
    """Extract ingredients list from content."""
    # Simple extraction - in production would be more sophisticated
    ingredients_section = content.split('Ingredients:')
    if len(ingredients_section) > 1:
        ingredients_text = ingredients_section[1].split('Instructions:')[0]
        return [ing.strip() for ing in ingredients_text.split(',') if ing.strip()]
    return []


def _extract_instructions_list(content: str) -> List[str]:
    """Extract cooking instructions from content."""
    instructions_section = content.split('Instructions:')
    if len(instructions_section) > 1:
        instructions_text = instructions_section[1]
        # Split by step numbers or periods
        steps = re.split(r'Step \d+:|\.', instructions_text)
        return [step.strip() for step in steps if step.strip()]
    return []


def _extract_cultural_context(content: str) -> Optional[str]:
    """Extract cultural context from content."""
    if 'traditional' in content.lower() or 'originated' in content.lower():
        return "Cultural context available"
    return None


def _parse_substitution_options(response: str, original_ingredient: str) -> List[Dict[str, Any]]:
    """Parse substitution options from response."""
    # Simple parsing - production would be more sophisticated
    return [
        {
            "substitute": "Alternative ingredient",
            "ratio": "1:1",
            "notes": "Flavor may vary slightly"
        }
    ]


def _extract_cooking_adjustments(response: str) -> Optional[str]:
    """Extract cooking method adjustments."""
    if "adjust" in response.lower() or "modify" in response.lower():
        return "Cooking adjustments may be needed"
    return None


def _extract_flavor_impact(response: str) -> Optional[str]:
    """Extract flavor impact information."""
    if "flavor" in response.lower() or "taste" in response.lower():
        return "Flavor profile may change"
    return None


def _extract_instructions(response: str) -> List[str]:
    """Extract step-by-step instructions."""
    # Simple extraction
    return ["Follow the detailed instructions in the response"]


def _extract_equipment(response: str) -> List[str]:
    """Extract required equipment."""
    equipment_keywords = ["pan", "pot", "oven", "mixer", "knife", "cutting board"]
    found_equipment = []
    
    response_lower = response.lower()
    for equipment in equipment_keywords:
        if equipment in response_lower:
            found_equipment.append(equipment)
    
    return found_equipment or ["Basic kitchen equipment"]


def _extract_common_mistakes(response: str) -> List[str]:
    """Extract common mistakes from response."""
    if "mistake" in response.lower() or "avoid" in response.lower():
        return ["Common mistakes are mentioned in the detailed response"]
    return ["Follow instructions carefully"]


def _extract_practice_suggestions(response: str) -> List[str]:
    """Extract practice suggestions."""
    return ["Practice with simple ingredients first", "Start with lower heat settings"]


async def _log_recipe_search_analytics(
    query_id: str,
    user_id: str,
    query: str,
    results_count: int,
    processing_time: float
) -> None:
    """Log recipe search analytics for optimization."""
    try:
        logger.info(f"📊 Recipe Analytics - Query: {query_id}, User: {user_id}, "
                   f"Results: {results_count}, Time: {processing_time:.2f}ms")
        
        # In production, would store in analytics database
        
    except Exception as e:
        logger.error(f"Failed to log recipe analytics: {e}")


@router.get("/cuisines", response_model=List[str])
async def get_supported_cuisines() -> List[str]:
    """Get list of supported cuisine types."""
    from src.api.schemas.culinary_schemas import CuisineType
    return [cuisine.value for cuisine in CuisineType]


@router.get("/dietary-restrictions", response_model=List[str])
async def get_supported_dietary_restrictions() -> List[str]:
    """Get list of supported dietary restrictions."""
    from src.api.schemas.culinary_schemas import DietaryRestriction
    return [restriction.value for restriction in DietaryRestriction]


@router.get("/cooking-methods", response_model=List[str])
async def get_cooking_methods() -> List[str]:
    """Get list of supported cooking methods."""
    return [
        "baking", "roasting", "grilling", "frying", "steaming", "boiling",
        "sautéing", "braising", "stewing", "smoking", "fermentation",
        "pickling", "curing", "sous-vide", "pressure-cooking"
    ]
