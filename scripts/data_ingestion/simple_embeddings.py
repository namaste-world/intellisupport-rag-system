#!/usr/bin/env python3
"""
IntelliSupport RAG - Simple Embedding Generation Script

This script generates embeddings for the customer support dataset.

Author: IntelliSupport Team
Created: 2025-08-31
"""

import logging
import json
import asyncio
from pathlib import Path
import time

import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def generate_embeddings():
    """Generate embeddings for the document corpus."""
    logger.info("🚀 Starting embedding generation...")
    
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Load document corpus
    corpus_file = Path("data/processed/document_corpus.json")
    if not corpus_file.exists():
        logger.error("❌ Document corpus not found. Please run download_dataset.py first.")
        return False
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    logger.info(f"📚 Loaded {len(documents)} documents for embedding generation")
    
    # Generate embeddings
    embeddings_data = []
    
    for i, doc in enumerate(documents):
        try:
            logger.info(f"🔄 Processing document {i+1}/{len(documents)}: {doc['id']}")
            
            # Generate embedding
            response = await client.embeddings.create(
                model="text-embedding-3-small",  # Using smaller model for testing
                input=doc["content"]
            )
            
            embedding = response.data[0].embedding
            
            # Store embedding with metadata
            embedding_entry = {
                "document_id": doc["id"],
                "content": doc["content"],
                "embedding": embedding,
                "embedding_dimension": len(embedding),
                "metadata": doc["metadata"],
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            
            embeddings_data.append(embedding_entry)
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding for document {doc['id']}: {e}")
            continue
    
    # Save embeddings
    embeddings_dir = Path("data/embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_file = embeddings_dir / "document_embeddings.json"
    with open(embeddings_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Embedding generation completed!")
    logger.info(f"  - Generated embeddings: {len(embeddings_data)}")
    logger.info(f"  - Embedding dimension: {embeddings_data[0]['embedding_dimension'] if embeddings_data else 'N/A'}")
    logger.info(f"  - Saved to: {embeddings_file}")
    
    return True


async def main():
    """Main function."""
    try:
        success = await generate_embeddings()
        if not success:
            exit(1)
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
