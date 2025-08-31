"""
IntelliSupport RAG - Document Retrieval Engine

This module implements the core document retrieval functionality with
hybrid search capabilities, re-ranking, and advanced filtering.

Author: IntelliSupport Team
Created: 2025-08-31
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.config.settings import get_settings
from src.core.embeddings.openai_embedder import embedder
from src.utils.exceptions import RetrievalError
from src.utils.text_processing import text_processor

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievalResult:
    """
    Represents a document retrieval result with metadata.
    
    Contains the retrieved document content along with relevance scores
    and metadata for ranking and filtering purposes.
    """
    content: str
    document_id: str
    relevance_score: float
    source: str
    metadata: Dict[str, Any]
    retrieval_method: str


class HybridRetriever:
    """
    Advanced hybrid retrieval engine for IntelliSupport RAG.
    
    Combines semantic search (embeddings) with keyword search (BM25/TF-IDF)
    and includes re-ranking capabilities for optimal relevance.
    """
    
    def __init__(self, vector_store_client=None):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store_client: Vector database client (Pinecone/Weaviate)
        """
        self.vector_store = vector_store_client
        self.embedder = embedder
        self.text_processor = text_processor
        
        # Initialize TF-IDF vectorizer for keyword search
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm='l2'
        )
        self.tfidf_matrix = None
        self.documents = []
        
        # Retrieval configuration
        self.top_k = settings.retrieval_top_k
        self.similarity_threshold = settings.retrieval_similarity_threshold
        self.rerank_top_k = settings.rerank_top_k
    
    async def retrieve(
        self, 
        query: str, 
        method: str = "hybrid",
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            method: Retrieval method ('semantic', 'keyword', 'hybrid')
            top_k: Number of documents to retrieve
            filters: Additional filters for retrieval
            
        Returns:
            List[RetrievalResult]: Ranked list of relevant documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty")
        
        top_k = top_k or self.top_k
        
        try:
            # Preprocess query
            processed_query = self.text_processor.preprocess_query(query)
            logger.debug(f"Processed query: {processed_query}")
            
            # Perform retrieval based on method
            if method == "semantic":
                results = await self._semantic_search(processed_query, top_k * 2, filters)
            elif method == "keyword":
                results = await self._keyword_search(processed_query, top_k * 2, filters)
            elif method == "hybrid":
                results = await self._hybrid_search(processed_query, top_k * 2, filters)
            else:
                raise RetrievalError(f"Unsupported retrieval method: {method}")
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.relevance_score >= self.similarity_threshold
            ]
            
            # Re-rank if we have enough results
            if len(filtered_results) > self.rerank_top_k:
                reranked_results = await self._rerank_results(query, filtered_results)
                final_results = reranked_results[:top_k]
            else:
                final_results = filtered_results[:top_k]
            
            logger.info(f"Retrieved {len(final_results)} documents for query: {query[:50]}...")
            return final_results
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            raise RetrievalError(f"Failed to retrieve documents: {str(e)}")
    
    async def _semantic_search(
        self, 
        query: str, 
        top_k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform semantic search using embeddings."""
        try:
            # Generate query embedding
            query_embedding = await self.embedder.generate_embedding(query)
            
            # Search vector store
            if self.vector_store:
                search_results = await self.vector_store.search(
                    vector=query_embedding,
                    top_k=top_k,
                    filter=filters
                )
                
                results = []
                for match in search_results.matches:
                    result = RetrievalResult(
                        content=match.metadata.get('content', ''),
                        document_id=match.id,
                        relevance_score=float(match.score),
                        source=match.metadata.get('source', 'unknown'),
                        metadata=match.metadata,
                        retrieval_method="semantic"
                    )
                    results.append(result)
                
                return results
            else:
                # Fallback to in-memory search if no vector store
                return await self._fallback_semantic_search(query_embedding, top_k)
                
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise RetrievalError(f"Semantic search failed: {str(e)}")
    
    async def _keyword_search(
        self, 
        query: str, 
        top_k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform keyword-based search using TF-IDF."""
        try:
            if self.tfidf_matrix is None or len(self.documents) == 0:
                logger.warning("No documents indexed for keyword search")
                return []
            
            # Transform query using fitted vectorizer
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include non-zero similarities
                    result = RetrievalResult(
                        content=self.documents[idx],
                        document_id=f"doc_{idx}",
                        relevance_score=float(similarities[idx]),
                        source="keyword_search",
                        metadata={"tfidf_score": float(similarities[idx])},
                        retrieval_method="keyword"
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise RetrievalError(f"Keyword search failed: {str(e)}")
    
    async def _hybrid_search(
        self, 
        query: str, 
        top_k: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Perform hybrid search combining semantic and keyword methods."""
        try:
            # Run both searches concurrently
            semantic_task = self._semantic_search(query, top_k, filters)
            keyword_task = self._keyword_search(query, top_k, filters)
            
            semantic_results, keyword_results = await asyncio.gather(
                semantic_task, keyword_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(semantic_results, Exception):
                logger.warning(f"Semantic search failed: {semantic_results}")
                semantic_results = []
            
            if isinstance(keyword_results, Exception):
                logger.warning(f"Keyword search failed: {keyword_results}")
                keyword_results = []
            
            # Combine and deduplicate results
            combined_results = self._combine_search_results(
                semantic_results, keyword_results, top_k
            )
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise RetrievalError(f"Hybrid search failed: {str(e)}")
    
    def _combine_search_results(
        self, 
        semantic_results: List[RetrievalResult], 
        keyword_results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """Combine semantic and keyword search results using RRF (Reciprocal Rank Fusion)."""
        # Create document score map
        doc_scores = {}
        
        # Add semantic results with RRF scoring
        for rank, result in enumerate(semantic_results):
            doc_id = result.document_id
            rrf_score = 1.0 / (rank + 60)  # RRF with k=60
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'result': result,
                    'semantic_score': result.relevance_score,
                    'keyword_score': 0.0,
                    'rrf_score': rrf_score
                }
            else:
                doc_scores[doc_id]['semantic_score'] = result.relevance_score
                doc_scores[doc_id]['rrf_score'] += rrf_score
        
        # Add keyword results with RRF scoring
        for rank, result in enumerate(keyword_results):
            doc_id = result.document_id
            rrf_score = 1.0 / (rank + 60)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    'result': result,
                    'semantic_score': 0.0,
                    'keyword_score': result.relevance_score,
                    'rrf_score': rrf_score
                }
            else:
                doc_scores[doc_id]['keyword_score'] = result.relevance_score
                doc_scores[doc_id]['rrf_score'] += rrf_score
        
        # Sort by combined RRF score
        sorted_docs = sorted(
            doc_scores.items(), 
            key=lambda x: x[1]['rrf_score'], 
            reverse=True
        )
        
        # Create final results with combined scores
        combined_results = []
        for doc_id, scores in sorted_docs[:top_k]:
            result = scores['result']
            # Update relevance score to combined score
            result.relevance_score = scores['rrf_score']
            result.retrieval_method = "hybrid"
            result.metadata.update({
                'semantic_score': scores['semantic_score'],
                'keyword_score': scores['keyword_score'],
                'rrf_score': scores['rrf_score']
            })
            combined_results.append(result)
        
        return combined_results
    
    async def _rerank_results(
        self, 
        query: str, 
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Re-rank results using cross-encoder or advanced scoring."""
        # For now, implement simple re-ranking based on content length and keyword overlap
        # In production, you would use a cross-encoder model here
        
        reranked_results = []
        
        for result in results:
            # Calculate additional relevance signals
            content_quality_score = self._calculate_content_quality(result.content)
            keyword_overlap_score = self._calculate_keyword_overlap(query, result.content)
            
            # Combine scores (weighted average)
            combined_score = (
                0.6 * result.relevance_score +
                0.2 * content_quality_score +
                0.2 * keyword_overlap_score
            )
            
            # Update relevance score
            result.relevance_score = combined_score
            result.metadata['content_quality'] = content_quality_score
            result.metadata['keyword_overlap'] = keyword_overlap_score
            
            reranked_results.append(result)
        
        # Sort by new combined score
        reranked_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.debug(f"Re-ranked {len(results)} results")
        return reranked_results
    
    def _calculate_content_quality(self, content: str) -> float:
        """Calculate content quality score based on length and structure."""
        if not content:
            return 0.0
        
        # Factors for quality scoring
        length_score = min(len(content) / 500, 1.0)  # Prefer moderate length
        sentence_count = len(content.split('.'))
        structure_score = min(sentence_count / 5, 1.0)  # Prefer well-structured content
        
        return (length_score + structure_score) / 2
    
    def _calculate_keyword_overlap(self, query: str, content: str) -> float:
        """Calculate keyword overlap between query and content."""
        query_keywords = set(query.lower().split())
        content_keywords = set(content.lower().split())
        
        if not query_keywords:
            return 0.0
        
        overlap = len(query_keywords.intersection(content_keywords))
        return overlap / len(query_keywords)
    
    async def _fallback_semantic_search(
        self, 
        query_embedding: List[float], 
        top_k: int
    ) -> List[RetrievalResult]:
        """Fallback semantic search using in-memory documents."""
        if not self.documents:
            return []
        
        # Generate embeddings for all documents (in production, these would be pre-computed)
        doc_embeddings = await self.embedder.generate_embeddings_batch(self.documents)
        
        # Calculate similarities
        similarities = []
        for doc_embedding in doc_embeddings:
            similarity = self.embedder.calculate_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                result = RetrievalResult(
                    content=self.documents[idx],
                    document_id=f"fallback_doc_{idx}",
                    relevance_score=similarities[idx],
                    source="fallback_semantic",
                    metadata={"similarity": similarities[idx]},
                    retrieval_method="semantic_fallback"
                )
                results.append(result)
        
        return results
    
    def index_documents(self, documents: List[str]) -> None:
        """
        Index documents for keyword search.
        
        Args:
            documents: List of document texts to index
        """
        try:
            self.documents = documents
            
            if documents:
                # Fit TF-IDF vectorizer
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
                logger.info(f"Indexed {len(documents)} documents for keyword search")
            else:
                logger.warning("No documents provided for indexing")
                
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise RetrievalError(f"Document indexing failed: {str(e)}")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get retrieval engine statistics.
        
        Returns:
            Dict[str, Any]: Statistics about the retrieval engine
        """
        return {
            "indexed_documents": len(self.documents),
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "rerank_top_k": self.rerank_top_k,
            "tfidf_features": self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
            "vector_store_connected": self.vector_store is not None,
        }


# Global retriever instance
retriever = HybridRetriever()
