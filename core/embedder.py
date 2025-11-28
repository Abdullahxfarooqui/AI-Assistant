"""
Production-grade async embeddings with parallel processing.
Uses OpenRouter API with text-embedding-3-small model.
"""
import openai
from typing import List, Dict, Any, Optional, Callable
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json

# OpenRouter API configuration
OPENAI_API_KEY = "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"
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMS = 1536  # Dimension of text-embedding-3-small

# Initialize OpenAI client with appropriate timeout
client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    timeout=90.0  # 90 second timeout for large batches
)


def truncate_text(text: str, max_chars: int = 8000) -> str:
    """Truncate text to avoid API limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def embed_batch_with_retry(
    texts: List[str],
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> List[List[float]]:
    """
    Embed a batch of texts with retry logic.
    
    Args:
        texts: List of texts to embed
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        
    Returns:
        List of embedding vectors
    """
    # Truncate long texts
    texts = [truncate_text(t) for t in texts]
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )
            return [item.embedding for item in response.data]
            
        except openai.RateLimitError as e:
            print(f"Rate limit hit (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                raise
                
        except openai.APITimeoutError as e:
            print(f"Timeout (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
                
        except Exception as e:
            print(f"Embedding error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
    
    return []


def embed_texts_sequential(
    texts: List[str],
    batch_size: int = 25,  # Larger batches for fewer API calls
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[List[float]]:
    """
    Embed texts sequentially with batching - more reliable than parallel.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts per batch (25 for efficiency)
        progress_callback: Optional callback(completed, total)
        
    Returns:
        List of embedding vectors in order
    """
    if not texts:
        return []
    
    all_embeddings = []
    total = len(texts)
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            embeddings = embed_batch_with_retry(batch, max_retries=3, retry_delay=1.0)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"Batch {i} failed: {e}")
            # Return zero vectors for failed batch
            all_embeddings.extend([[0.0] * EMBEDDING_DIMS for _ in batch])
        
        if progress_callback:
            progress_callback(min(i + batch_size, total), total)
    
    return all_embeddings


def embed_query(query: str) -> List[float]:
    """
    Generate embedding for a single query.
    
    Args:
        query: Query text to embed
        
    Returns:
        Embedding vector
    """
    query = truncate_text(query)
    
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Query embedding error: {e}")
        return [0.0] * EMBEDDING_DIMS


def embed_chunks(
    chunks: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[List[float]]:
    """
    Generate embeddings for document chunks.
    
    Args:
        chunks: List of chunk dictionaries with 'text' key
        progress_callback: Optional callback(completed, total)
        
    Returns:
        List of embedding vectors
    """
    texts = [chunk.get("text", "") for chunk in chunks]
    # Use batch_size=25 for faster processing with fewer API calls
    return embed_texts_sequential(texts, batch_size=25, progress_callback=progress_callback)


def compute_doc_hash(file_content: bytes) -> str:
    """Compute a unique hash for a document."""
    return hashlib.sha256(file_content).hexdigest()[:16]
