"""
IntelliSupport RAG - Query API Schemas

This module defines Pydantic models for query-related API requests
and responses with proper validation and documentation.

Author: IntelliSupport Team
Created: 2025-08-31
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field, validator


class UserContext(BaseModel):
    """
    User context information for personalized responses.
    
    Contains additional information about the user and their environment
    to provide more relevant and personalized support responses.
    """
    product: Optional[str] = Field(None, description="Product or service being used")
    user_tier: Optional[str] = Field("standard", description="User tier (standard, premium, enterprise)")
    platform: Optional[str] = Field(None, description="Platform (web, mobile, desktop)")
    version: Optional[str] = Field(None, description="Application version")
    previous_queries: Optional[List[str]] = Field(None, description="Recent previous queries")


class QueryRequest(BaseModel):
    """
    Request model for query processing endpoint.
    
    Defines the structure for incoming query requests with validation
    and proper documentation for API consumers.
    """
    query: str = Field(..., min_length=1, max_length=1000, description="User query text")
    user_id: str = Field(..., min_length=1, max_length=100, description="Unique user identifier")
    language: str = Field("auto", description="Response language (en, hi, ta, auto)")
    context: Optional[UserContext] = Field(None, description="Additional user context")
    include_citations: Optional[bool] = Field(None, description="Include source citations in response")
    max_results: Optional[int] = Field(None, ge=1, le=10, description="Maximum number of results to return")
    
    @validator('language')
    def validate_language(cls, v):
        """Validate language code is supported."""
        supported_languages = ['en', 'hi', 'ta', 'auto']
        if v not in supported_languages:
            raise ValueError(f'Language must be one of: {supported_languages}')
        return v
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query content."""
        if not v.strip():
            raise ValueError('Query cannot be empty or only whitespace')
        return v.strip()


class Citation(BaseModel):
    """
    Citation information for source documents.
    
    Provides transparency about the sources used to generate responses
    for user verification and trust building.
    """
    source: str = Field(..., description="Source document or knowledge base section")
    document_id: str = Field(..., description="Unique document identifier")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    excerpt: Optional[str] = Field(None, description="Relevant excerpt from the document")


class QueryResponse(BaseModel):
    """
    Response model for query processing endpoint.
    
    Contains the generated response along with metadata, citations,
    and quality metrics for transparency and debugging.
    """
    response: str = Field(..., description="Generated response text")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Response confidence score (0-1)")
    language: str = Field(..., description="Response language")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    
    # Metadata
    query_id: str = Field(..., description="Unique query identifier for tracking")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="LLM token usage statistics")
    
    # Quality metrics
    retrieval_stats: Dict[str, Any] = Field(default_factory=dict, description="Retrieval performance statistics")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "To reset your password, please follow these steps: 1. Go to the login page 2. Click 'Forgot Password' 3. Enter your email address 4. Check your email for reset instructions",
                "confidence_score": 0.92,
                "language": "en",
                "citations": [
                    {
                        "source": "Password Reset Guide",
                        "document_id": "doc_123",
                        "relevance_score": 0.95,
                        "excerpt": "Password reset process involves..."
                    }
                ],
                "query_id": "query_abc123",
                "processing_time_ms": 1250,
                "token_usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 75,
                    "total_tokens": 225
                },
                "retrieval_stats": {
                    "documents_retrieved": 3,
                    "average_relevance": 0.87,
                    "retrieval_method": "hybrid"
                }
            }
        }


class QueryError(BaseModel):
    """
    Error response model for query processing failures.
    
    Provides structured error information for API consumers
    with proper error codes and debugging information.
    """
    error: Dict[str, Any] = Field(..., description="Error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "code": "RETRIEVAL_ERROR",
                    "message": "Failed to retrieve relevant documents",
                    "request_id": "req_xyz789",
                    "details": {
                        "query": "user query here",
                        "timestamp": "2025-08-31T10:30:00Z"
                    }
                }
            }
        }


class BulkQueryRequest(BaseModel):
    """
    Request model for bulk query processing.
    
    Allows processing multiple queries in a single request
    for batch operations and testing purposes.
    """
    queries: List[QueryRequest] = Field(..., min_items=1, max_items=50, description="List of queries to process")
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")
    
    @validator('queries')
    def validate_queries(cls, v):
        """Validate that all queries are unique."""
        query_texts = [q.query for q in v]
        if len(query_texts) != len(set(query_texts)):
            raise ValueError('Duplicate queries are not allowed in bulk requests')
        return v


class BulkQueryResponse(BaseModel):
    """
    Response model for bulk query processing.
    
    Contains responses for all processed queries along with
    batch-level statistics and error information.
    """
    responses: List[QueryResponse] = Field(..., description="Individual query responses")
    batch_id: str = Field(..., description="Batch identifier")
    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successfully processed queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    total_processing_time_ms: int = Field(..., description="Total batch processing time")
    
    class Config:
        schema_extra = {
            "example": {
                "responses": [
                    # ... individual QueryResponse objects
                ],
                "batch_id": "batch_abc123",
                "total_queries": 5,
                "successful_queries": 4,
                "failed_queries": 1,
                "total_processing_time_ms": 3500
            }
        }
