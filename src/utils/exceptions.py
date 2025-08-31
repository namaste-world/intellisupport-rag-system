"""
IntelliSupport RAG - Custom Exception Classes

This module defines custom exceptions for the IntelliSupport RAG system
to provide better error handling and user experience.

Author: IntelliSupport Team
Created: 2025-08-31
"""

from typing import Any, Dict, Optional


class IntelliSupportException(Exception):
    """
    Base exception class for all IntelliSupport RAG errors.
    
    Provides consistent error handling with proper status codes,
    error codes, and user-friendly messages.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize IntelliSupport exception.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for API responses
            status_code: HTTP status code for API responses
            details: Additional error details for debugging
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class ValidationError(IntelliSupportException):
    """Exception raised for input validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details={"field": field} if field else {}
        )


class AuthenticationError(IntelliSupportException):
    """Exception raised for authentication failures."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401
        )


class AuthorizationError(IntelliSupportException):
    """Exception raised for authorization failures."""
    
    def __init__(self, message: str = "Access denied"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403
        )


class RateLimitError(IntelliSupportException):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"retry_after": retry_after} if retry_after else {}
        )


class ExternalServiceError(IntelliSupportException):
    """Exception raised for external service failures."""
    
    def __init__(self, service: str, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message=f"{service} service error: {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details={
                "service": service,
                "original_error": str(original_error) if original_error else None
            }
        )


class RetrievalError(IntelliSupportException):
    """Exception raised for document retrieval failures."""
    
    def __init__(self, message: str = "Failed to retrieve relevant documents"):
        super().__init__(
            message=message,
            error_code="RETRIEVAL_ERROR",
            status_code=500
        )


class GenerationError(IntelliSupportException):
    """Exception raised for response generation failures."""
    
    def __init__(self, message: str = "Failed to generate response"):
        super().__init__(
            message=message,
            error_code="GENERATION_ERROR",
            status_code=500
        )


class EmbeddingError(IntelliSupportException):
    """Exception raised for embedding generation failures."""
    
    def __init__(self, message: str = "Failed to generate embeddings"):
        super().__init__(
            message=message,
            error_code="EMBEDDING_ERROR",
            status_code=500
        )


class VectorStoreError(IntelliSupportException):
    """Exception raised for vector store operations."""
    
    def __init__(self, operation: str, message: str):
        super().__init__(
            message=f"Vector store {operation} failed: {message}",
            error_code="VECTOR_STORE_ERROR",
            status_code=500,
            details={"operation": operation}
        )


class CacheError(IntelliSupportException):
    """Exception raised for cache operations."""
    
    def __init__(self, operation: str, message: str):
        super().__init__(
            message=f"Cache {operation} failed: {message}",
            error_code="CACHE_ERROR",
            status_code=500,
            details={"operation": operation}
        )


class ConfigurationError(IntelliSupportException):
    """Exception raised for configuration errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=f"Configuration error: {message}",
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details={"config_key": config_key} if config_key else {}
        )
