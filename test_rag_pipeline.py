#!/usr/bin/env python3
"""
IntelliSupport RAG - Complete Pipeline Test

This script tests the complete RAG pipeline end-to-end to verify
all components are working correctly together.

Author: IntelliSupport Team
Created: 2025-08-31
"""

import logging
import json
import asyncio
from pathlib import Path
import time
from typing import List, Dict, Any

import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleRAGPipeline:
    """Simple RAG pipeline for testing."""
    
    def __init__(self):
        """Initialize RAG pipeline."""
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.documents = []
        self.embeddings = []
        self.document_metadata = []
    
    def load_data(self):
        """Load documents and embeddings."""
        logger.info("📚 Loading documents and embeddings...")
        
        # Load document corpus
        corpus_file = Path("data/processed/document_corpus.json")
        if corpus_file.exists():
            with open(corpus_file, 'r') as f:
                corpus_data = json.load(f)
            
            self.documents = [doc["content"] for doc in corpus_data]
            self.document_metadata = [doc["metadata"] for doc in corpus_data]
            logger.info(f"✅ Loaded {len(self.documents)} documents")
        
        # Load embeddings
        embeddings_file = Path("data/embeddings/test_embeddings.json")
        if embeddings_file.exists():
            with open(embeddings_file, 'r') as f:
                embeddings_data = json.load(f)
            
            self.embeddings = [entry["embedding"] for entry in embeddings_data]
            logger.info(f"✅ Loaded {len(self.embeddings)} embeddings")
        
        return len(self.documents) > 0 and len(self.embeddings) > 0
    
    async def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        logger.info(f"🔍 Retrieving documents for query: {query}")
        
        # Generate query embedding
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding
        
        # Calculate similarities
        similarities = []
        for doc_embedding in self.embeddings:
            similarity = cosine_similarity(
                [query_embedding], 
                [doc_embedding]
            )[0][0]
            similarities.append(similarity)
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            result = {
                "content": self.documents[idx],
                "similarity": similarities[idx],
                "metadata": self.document_metadata[idx]
            }
            results.append(result)
        
        logger.info(f"✅ Retrieved {len(results)} documents")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. Similarity: {result['similarity']:.3f} - Category: {result['metadata']['category']}")
        
        return results
    
    async def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Generate response using retrieved documents."""
        logger.info("🤖 Generating response...")
        
        # Prepare context
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"Document {i+1}: {doc['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        system_prompt = """You are IntelliSupport, a helpful customer support assistant. 
Use the provided context to answer user questions accurately and helpfully. 
If the context doesn't contain enough information, say so clearly."""
        
        user_prompt = f"""Context:
{context}

User Question: {query}

Please provide a helpful response based on the context above."""
        
        # Generate response
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        generated_response = response.choices[0].message.content
        
        logger.info("✅ Response generated successfully")
        return generated_response
    
    async def test_rag_pipeline(self, test_queries: List[str]) -> None:
        """Test the complete RAG pipeline."""
        logger.info("🧪 Testing complete RAG pipeline...")
        
        for i, query in enumerate(test_queries):
            logger.info(f"\n{'='*60}")
            logger.info(f"Test {i+1}/{len(test_queries)}: {query}")
            logger.info(f"{'='*60}")
            
            start_time = time.time()
            
            try:
                # Step 1: Retrieve documents
                retrieved_docs = await self.retrieve_documents(query, top_k=2)
                
                # Step 2: Generate response
                response = await self.generate_response(query, retrieved_docs)
                
                processing_time = (time.time() - start_time) * 1000
                
                # Display results
                logger.info(f"📝 Generated Response:")
                logger.info(f"   {response}")
                logger.info(f"⏱️  Processing Time: {processing_time:.2f}ms")
                
                print(f"\n🔍 Query: {query}")
                print(f"🤖 Response: {response}")
                print(f"⏱️  Time: {processing_time:.2f}ms")
                print(f"📊 Retrieved {len(retrieved_docs)} documents")
                
            except Exception as e:
                logger.error(f"❌ Test failed for query '{query}': {e}")


async def main():
    """Main test function."""
    logger.info("🚀 Starting RAG pipeline test...")
    
    # Initialize pipeline
    rag = SimpleRAGPipeline()
    
    # Load data
    if not rag.load_data():
        logger.error("❌ Failed to load data. Please run dataset and embedding scripts first.")
        return
    
    # Test queries
    test_queries = [
        "How do I reset my password?",
        "Can I update my billing information?",
        "Is there a mobile app?",
        "Why is the app running slowly?",
        "How do I export my data?"
    ]
    
    # Run tests
    await rag.test_rag_pipeline(test_queries)
    
    logger.info("🎉 RAG pipeline test completed!")


if __name__ == "__main__":
    asyncio.run(main())
EOF
