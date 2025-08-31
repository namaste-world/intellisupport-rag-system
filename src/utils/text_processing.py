"""
IntelliSupport RAG - Text Processing Utilities

This module provides text processing utilities for the RAG system including
cleaning, chunking, language detection, and preprocessing functions.

Author: IntelliSupport Team
Created: 2025-08-31
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import spacy
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

# Load spaCy models for different languages
try:
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("English spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp_en = None


@dataclass
class TextChunk:
    """
    Represents a chunk of text with metadata.
    
    This class stores processed text chunks along with their metadata
    for efficient retrieval and processing in the RAG pipeline.
    """
    content: str
    chunk_id: str
    source_document: str
    start_char: int
    end_char: int
    language: str
    metadata: Dict[str, Any]


class TextProcessor:
    """
    Advanced text processing utilities for RAG system.
    
    Provides comprehensive text processing capabilities including
    cleaning, chunking, language detection, and preprocessing.
    """
    
    def __init__(self):
        """Initialize text processor with language models."""
        self.nlp_en = nlp_en
        self.supported_languages = ['en', 'hi', 'ta']  # English, Hindi, Tamil
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            str: Language code (en, hi, ta) or 'en' as default
        """
        try:
            # Clean text for better detection
            cleaned_text = self.clean_text(text)
            if len(cleaned_text.strip()) < 10:
                return 'en'  # Default to English for short texts
            
            detected_lang = detect(cleaned_text)
            
            # Map detected languages to supported ones
            if detected_lang in self.supported_languages:
                return detected_lang
            elif detected_lang in ['hi-Latn', 'hi']:  # Hindi variants
                return 'hi'
            elif detected_lang in ['ta']:  # Tamil
                return 'ta'
            else:
                return 'en'  # Default to English
                
        except LangDetectException:
            logger.warning(f"Language detection failed for text: {text[:50]}...")
            return 'en'
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Args:
            text: Raw input text
            
        Returns:
            str: Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\?\!\-\:\;]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers (basic pattern)
        text = re.sub(r'\b\d{10,}\b', '', text)
        
        return text.strip()
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess user query for better retrieval.
        
        Args:
            query: Raw user query
            
        Returns:
            str: Preprocessed query
        """
        # Clean the query
        cleaned_query = self.clean_text(query)
        
        # Expand common abbreviations
        abbreviations = {
            'pwd': 'password',
            'acc': 'account',
            'info': 'information',
            'config': 'configuration',
            'auth': 'authentication',
            'api': 'application programming interface',
            'ui': 'user interface',
            'db': 'database'
        }
        
        words = cleaned_query.lower().split()
        expanded_words = []
        
        for word in words:
            if word in abbreviations:
                expanded_words.append(abbreviations[word])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 512, 
        overlap: int = 50,
        method: str = "semantic"
    ) -> List[TextChunk]:
        """
        Split text into chunks for vector storage.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum chunk size in tokens
            overlap: Overlap between chunks in tokens
            method: Chunking method ('fixed', 'semantic', 'sentence')
            
        Returns:
            List[TextChunk]: List of text chunks with metadata
        """
        if not text or len(text.strip()) == 0:
            return []
        
        chunks = []
        
        if method == "sentence" and self.nlp_en:
            chunks = self._chunk_by_sentences(text, chunk_size, overlap)
        elif method == "semantic" and self.nlp_en:
            chunks = self._chunk_semantically(text, chunk_size, overlap)
        else:
            chunks = self._chunk_fixed_size(text, chunk_size, overlap)
        
        return chunks
    
    def _chunk_fixed_size(self, text: str, chunk_size: int, overlap: int) -> List[TextChunk]:
        """Split text into fixed-size chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk = TextChunk(
                content=chunk_text,
                chunk_id=f"chunk_{i}",
                source_document="unknown",
                start_char=0,
                end_char=len(chunk_text),
                language=self.detect_language(chunk_text),
                metadata={"method": "fixed_size", "word_count": len(chunk_words)}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sentences(self, text: str, chunk_size: int, overlap: int) -> List[TextChunk]:
        """Split text by sentences with size constraints."""
        if not self.nlp_en:
            return self._chunk_fixed_size(text, chunk_size, overlap)
        
        doc = self.nlp_en(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        chunks = []
        
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                chunk = TextChunk(
                    content=chunk_text,
                    chunk_id=f"sent_chunk_{len(chunks)}",
                    source_document="unknown",
                    start_char=0,
                    end_char=len(chunk_text),
                    language=self.detect_language(chunk_text),
                    metadata={"method": "sentence", "sentence_count": len(current_chunk)}
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > 1:
                    current_chunk = current_chunk[-1:]  # Keep last sentence for overlap
                    current_size = len(current_chunk[0].split())
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add final chunk if exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = TextChunk(
                content=chunk_text,
                chunk_id=f"sent_chunk_{len(chunks)}",
                source_document="unknown",
                start_char=0,
                end_char=len(chunk_text),
                language=self.detect_language(chunk_text),
                metadata={"method": "sentence", "sentence_count": len(current_chunk)}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_semantically(self, text: str, chunk_size: int, overlap: int) -> List[TextChunk]:
        """Split text semantically using NLP analysis."""
        if not self.nlp_en:
            return self._chunk_by_sentences(text, chunk_size, overlap)
        
        doc = self.nlp_en(text)
        
        # Group sentences by semantic similarity
        sentences = [sent.text.strip() for sent in doc.sents]
        chunks = []
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # Check if adding this sentence would exceed chunk size
            if current_size + sentence_size > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk = TextChunk(
                    content=chunk_text,
                    chunk_id=f"sem_chunk_{len(chunks)}",
                    source_document="unknown",
                    start_char=0,
                    end_char=len(chunk_text),
                    language=self.detect_language(chunk_text),
                    metadata={"method": "semantic", "sentence_count": len(current_chunk)}
                )
                chunks.append(chunk)
                
                # Handle overlap
                if overlap > 0 and len(current_chunk) > 1:
                    overlap_sentences = current_chunk[-1:]
                    current_chunk = overlap_sentences
                    current_size = sum(len(s.split()) for s in overlap_sentences)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = TextChunk(
                content=chunk_text,
                chunk_id=f"sem_chunk_{len(chunks)}",
                source_document="unknown",
                start_char=0,
                end_char=len(chunk_text),
                language=self.detect_language(chunk_text),
                metadata={"method": "semantic", "sentence_count": len(current_chunk)}
            )
            chunks.append(chunk)
        
        return chunks
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List[str]: List of extracted keywords
        """
        if not self.nlp_en or not text:
            return []
        
        doc = self.nlp_en(text.lower())
        
        # Extract keywords based on POS tags and named entities
        keywords = []
        
        # Add named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'TECH']:
                keywords.append(ent.text.lower())
        
        # Add important nouns and adjectives
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                keywords.append(token.lemma_.lower())
        
        # Remove duplicates and return top keywords
        unique_keywords = list(dict.fromkeys(keywords))  # Preserve order
        return unique_keywords[:max_keywords]


# Global text processor instance
text_processor = TextProcessor()
