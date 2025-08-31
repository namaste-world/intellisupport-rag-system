"""
IntelliSupport RAG - Query Processing API Routes

This module implements the main query processing endpoints for the
IntelliSupport RAG system with comprehensive error handling and monitoring.

Author: IntelliSupport Team
Created: 2025-08-31
"""

import logging
import time
import uuid
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from src.api.schemas.query import (
    QueryRequest, 
    QueryResponse, 
    Citation,
    BulkQueryRequest,
    BulkQueryResponse
)
from src.core.rag.retriever import retriever
from src.core.rag.generator import generator
from src.utils.exceptions import IntelliSupportException, ValidationError
from src.utils.text_processing import text_processor

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


async def get_rag_dependencies():
    """
    Dependency to ensure RAG components are available.
    
    This function checks that all necessary RAG components
    are properly initialized before processing requests.
    """
    # In a real implementation, you would check if services are healthy
    return {
        "retriever": retriever,
        "generator": generator,
        "text_processor": text_processor
    }


@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    rag_deps: Dict[str, Any] = Depends(get_rag_dependencies)
) -> QueryResponse:
    """
    Process a single user query and return AI-generated response.
    
    This endpoint handles the complete RAG pipeline:
    1. Query preprocessing and validation
    2. Document retrieval from vector store
    3. Response generation using LLM
    4. Post-processing and citation generation
    
    Args:
        request: Query request with user query and context
        background_tasks: FastAPI background tasks for async operations
        rag_deps: RAG component dependencies
        
    Returns:
        QueryResponse: Generated response with metadata and citations
        
    Raises:
        HTTPException: For various error conditions with appropriate status codes
    """
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    logger.info(f"Processing query {query_id}: {request.query[:100]}...")
    
    try:
        # Validate and preprocess query
        if not request.query.strip():
            raise ValidationError("Query cannot be empty")
        
        # Detect language if set to auto
        language = request.language
        if language == "auto":
            language = text_processor.detect_language(request.query)
            logger.debug(f"Auto-detected language: {language}")
        
        # Step 1: Retrieve relevant documents
        logger.debug("Starting document retrieval...")
        retrieval_start = time.time()
        
        retrieved_docs = await retriever.retrieve(
            query=request.query,
            method="hybrid",  # Always use hybrid for best results
            top_k=request.max_results or 5,
            filters=_build_retrieval_filters(request.context)
        )
        
        retrieval_time = (time.time() - retrieval_start) * 1000
        logger.debug(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}ms")
        
        # Step 2: Generate response
        logger.debug("Starting response generation...")
        generation_start = time.time()
        
        generation_result = await generator.generate_response(
            query=request.query,
            retrieved_docs=retrieved_docs,
            user_context=request.context.dict() if request.context else None,
            language=language
        )
        
        generation_time = (time.time() - generation_start) * 1000
        logger.debug(f"Generated response in {generation_time:.2f}ms")
        
        # Step 3: Build response
        total_time = (time.time() - start_time) * 1000
        
        # Convert citations
        citations = []
        if request.include_citations or (request.include_citations is None and generation_result.citations):
            for i, citation_text in enumerate(generation_result.citations):
                if i < len(retrieved_docs):
                    doc = retrieved_docs[i]
                    citation = Citation(
                        source=doc.metadata.get('source', 'Knowledge Base'),
                        document_id=doc.document_id,
                        relevance_score=doc.relevance_score,
                        excerpt=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                    )
                    citations.append(citation)
        
        # Build retrieval statistics
        retrieval_stats = {
            "documents_retrieved": len(retrieved_docs),
            "average_relevance": sum(doc.relevance_score for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0.0,
            "retrieval_method": "hybrid",
            "retrieval_time_ms": int(retrieval_time),
            "generation_time_ms": int(generation_time)
        }
        
        response = QueryResponse(
            response=generation_result.response,
            confidence_score=generation_result.confidence_score,
            language=generation_result.language,
            citations=citations,
            query_id=query_id,
            processing_time_ms=int(total_time),
            token_usage=generation_result.token_usage,
            retrieval_stats=retrieval_stats
        )
        
        # Log successful processing
        logger.info(f"Query {query_id} processed successfully in {total_time:.2f}ms")
        
        # Schedule background tasks
        background_tasks.add_task(
            _log_query_metrics,
            query_id=query_id,
            user_id=request.user_id,
            processing_time=total_time,
            confidence_score=generation_result.confidence_score,
            num_retrieved_docs=len(retrieved_docs)
        )
        
        return response
        
    except IntelliSupportException as e:
        logger.error(f"Query {query_id} failed with IntelliSupport error: {e.message}")
        raise HTTPException(status_code=e.status_code, detail={
            "error": {
                "code": e.error_code,
                "message": e.message,
                "query_id": query_id,
                "details": e.details
            }
        })
        
    except Exception as e:
        logger.error(f"Query {query_id} failed with unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred while processing your query",
                "query_id": query_id
            }
        })


@router.post("/query/bulk", response_model=BulkQueryResponse)
async def process_bulk_queries(
    request: BulkQueryRequest,
    background_tasks: BackgroundTasks,
    rag_deps: Dict[str, Any] = Depends(get_rag_dependencies)
) -> BulkQueryResponse:
    """
    Process multiple queries in a single batch request.
    
    This endpoint is useful for testing, evaluation, and batch processing
    scenarios where multiple queries need to be processed efficiently.
    
    Args:
        request: Bulk query request with list of queries
        background_tasks: FastAPI background tasks
        rag_deps: RAG component dependencies
        
    Returns:
        BulkQueryResponse: Batch processing results with statistics
    """
    start_time = time.time()
    batch_id = request.batch_id or str(uuid.uuid4())
    
    logger.info(f"Processing bulk query batch {batch_id} with {len(request.queries)} queries")
    
    responses = []
    successful_count = 0
    failed_count = 0
    
    for i, query_request in enumerate(request.queries):
        try:
            # Process individual query
            response = await process_query(query_request, background_tasks, rag_deps)
            responses.append(response)
            successful_count += 1
            
        except HTTPException as e:
            # Create error response for failed query
            error_response = QueryResponse(
                response=f"Error processing query: {e.detail.get('error', {}).get('message', 'Unknown error')}",
                confidence_score=0.0,
                language="en",
                citations=[],
                query_id=str(uuid.uuid4()),
                processing_time_ms=0,
                token_usage={},
                retrieval_stats={}
            )
            responses.append(error_response)
            failed_count += 1
            
            logger.warning(f"Query {i+1} in batch {batch_id} failed: {e.detail}")
    
    total_time = (time.time() - start_time) * 1000
    
    bulk_response = BulkQueryResponse(
        responses=responses,
        batch_id=batch_id,
        total_queries=len(request.queries),
        successful_queries=successful_count,
        failed_queries=failed_count,
        total_processing_time_ms=int(total_time)
    )
    
    logger.info(f"Bulk batch {batch_id} completed: {successful_count} successful, {failed_count} failed")
    
    return bulk_response


def _build_retrieval_filters(context: Optional[UserContext]) -> Optional[Dict[str, Any]]:
    """
    Build retrieval filters from user context.
    
    Args:
        context: User context information
        
    Returns:
        Optional[Dict[str, Any]]: Filters for vector store retrieval
    """
    if not context:
        return None
    
    filters = {}
    
    # Add product filter if specified
    if context.product:
        filters["product"] = context.product
    
    # Add user tier filter for premium content
    if context.user_tier in ["premium", "enterprise"]:
        filters["access_level"] = {"$in": ["public", "premium", "enterprise"]}
    else:
        filters["access_level"] = "public"
    
    # Add platform filter if specified
    if context.platform:
        filters["platform"] = {"$in": ["all", context.platform]}
    
    return filters if filters else None


async def _log_query_metrics(
    query_id: str,
    user_id: str,
    processing_time: float,
    confidence_score: float,
    num_retrieved_docs: int
) -> None:
    """
    Log query metrics for monitoring and analytics.
    
    This background task logs important metrics about query processing
    for monitoring, analytics, and system optimization purposes.
    """
    try:
        # In a real implementation, you would:
        # 1. Store metrics in a time-series database
        # 2. Update Prometheus metrics
        # 3. Log to analytics platform
        
        logger.info(f"Query metrics - ID: {query_id}, User: {user_id}, "
                   f"Time: {processing_time:.2f}ms, Confidence: {confidence_score:.3f}, "
                   f"Docs: {num_retrieved_docs}")
        
        # TODO: Implement actual metrics storage
        # await metrics_client.record_query_metrics(...)
        
    except Exception as e:
        logger.error(f"Failed to log query metrics for {query_id}: {e}")
        # Don't raise exception as this is a background task
