"""
IntelliSupport RAG - Health Check API Routes

This module implements health check endpoints for monitoring
system status and component availability.

Author: IntelliSupport Team
Created: 2025-08-31
"""

import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config.settings import get_settings
from src.core.embeddings.openai_embedder import embedder
from src.core.rag.retriever import retriever
from src.core.rag.generator import generator

logger = logging.getLogger(__name__)
settings = get_settings()

# Create router
router = APIRouter()


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str
    timestamp: str
    version: str
    environment: str


class DetailedHealthStatus(BaseModel):
    """Detailed health status with component information."""
    status: str
    timestamp: str
    version: str
    environment: str
    components: Dict[str, Dict[str, Any]]
    performance: Dict[str, Any]


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Basic health check endpoint.
    
    Returns simple health status for load balancer health checks
    and basic monitoring. This endpoint should be fast and lightweight.
    
    Returns:
        HealthStatus: Basic health information
    """
    return HealthStatus(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        version=settings.api_version,
        environment=settings.environment
    )


@router.get("/health/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check() -> DetailedHealthStatus:
    """
    Detailed health check with component status.
    
    Provides comprehensive health information including the status
    of all system components, external services, and performance metrics.
    
    Returns:
        DetailedHealthStatus: Comprehensive health information
        
    Raises:
        HTTPException: If critical components are unhealthy
    """
    start_time = time.time()
    
    logger.debug("Starting detailed health check...")
    
    # Check all components
    components = {}
    overall_status = "healthy"
    
    # Check OpenAI Embedder
    try:
        embedder_stats = embedder.get_embedding_stats()
        components["embedder"] = {
            "status": "healthy",
            "model": embedder_stats["model"],
            "cache_size": embedder_stats["cache_size"],
            "details": embedder_stats
        }
    except Exception as e:
        logger.error(f"Embedder health check failed: {e}")
        components["embedder"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_status = "degraded"
    
    # Check Retriever
    try:
        retriever_stats = retriever.get_retrieval_stats()
        components["retriever"] = {
            "status": "healthy",
            "indexed_documents": retriever_stats["indexed_documents"],
            "vector_store_connected": retriever_stats["vector_store_connected"],
            "details": retriever_stats
        }
    except Exception as e:
        logger.error(f"Retriever health check failed: {e}")
        components["retriever"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_status = "degraded"
    
    # Check Generator (test with simple query)
    try:
        # Simple test to verify LLM connectivity
        test_docs = []  # Empty docs for basic test
        test_result = await generator.generate_response(
            query="test",
            retrieved_docs=test_docs,
            language="en"
        )
        
        components["generator"] = {
            "status": "healthy",
            "model": test_result.model_used,
            "test_confidence": test_result.confidence_score,
            "details": {
                "model": test_result.model_used,
                "language_support": ["en", "hi", "ta"]
            }
        }
    except Exception as e:
        logger.error(f"Generator health check failed: {e}")
        components["generator"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_status = "unhealthy"  # Generator failure is critical
    
    # Check external services (mock implementation)
    # In production, you would check actual service connectivity
    components["vector_store"] = {
        "status": "healthy" if retriever.vector_store else "not_configured",
        "type": "pinecone" if retriever.vector_store else "fallback"
    }
    
    components["cache"] = {
        "status": "healthy",  # Would check Redis connectivity
        "type": "redis"
    }
    
    # Performance metrics
    health_check_time = (time.time() - start_time) * 1000
    performance = {
        "health_check_time_ms": round(health_check_time, 2),
        "memory_usage": "N/A",  # Would implement actual memory monitoring
        "cpu_usage": "N/A"      # Would implement actual CPU monitoring
    }
    
    # Determine overall status
    unhealthy_components = [name for name, comp in components.items() if comp["status"] == "unhealthy"]
    if unhealthy_components:
        overall_status = "unhealthy"
        logger.warning(f"Unhealthy components detected: {unhealthy_components}")
    
    response = DetailedHealthStatus(
        status=overall_status,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        version=settings.api_version,
        environment=settings.environment,
        components=components,
        performance=performance
    )
    
    logger.info(f"Health check completed in {health_check_time:.2f}ms - Status: {overall_status}")
    
    # Return error status if unhealthy
    if overall_status == "unhealthy":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service unhealthy",
                "components": components,
                "timestamp": response.timestamp
            }
        )
    
    return response


@router.get("/health/readiness")
async def readiness_check() -> Dict[str, Any]:
    """
    Kubernetes readiness probe endpoint.
    
    Checks if the application is ready to receive traffic.
    This is used by Kubernetes to determine when to start
    routing traffic to the pod.
    
    Returns:
        Dict[str, Any]: Readiness status
    """
    try:
        # Check critical components
        critical_checks = []
        
        # Check if embedder is initialized
        try:
            embedder.get_embedding_stats()
            critical_checks.append(("embedder", True))
        except:
            critical_checks.append(("embedder", False))
        
        # Check if retriever is initialized
        try:
            retriever.get_retrieval_stats()
            critical_checks.append(("retriever", True))
        except:
            critical_checks.append(("retriever", False))
        
        # All critical components must be ready
        all_ready = all(status for _, status in critical_checks)
        
        if all_ready:
            return {
                "status": "ready",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "checks": dict(critical_checks)
            }
        else:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "checks": dict(critical_checks)
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        )


@router.get("/health/liveness")
async def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes liveness probe endpoint.
    
    Simple check to verify the application is still running.
    This is used by Kubernetes to determine if the pod should
    be restarted.
    
    Returns:
        Dict[str, Any]: Liveness status
    """
    return {
        "status": "alive",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": "N/A"  # Would include actual process ID in production
    }
