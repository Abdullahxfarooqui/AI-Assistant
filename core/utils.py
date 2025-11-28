"""
Utility functions for hashing, file operations, and metadata management.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    for directory in [UPLOADS_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, FAISS_INDEX_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def compute_file_hash(file_bytes: bytes) -> str:
    """
    Compute SHA256 hash of file content for duplicate detection.
    
    Args:
        file_bytes: Raw bytes of the file
        
    Returns:
        SHA256 hash as hexadecimal string
    """
    return hashlib.sha256(file_bytes).hexdigest()


def save_json(data: Any, file_path: Path) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to serialize
        file_path: Path to save the JSON file
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file_path: Path) -> Optional[Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data or None if file doesn't exist
    """
    if not file_path.exists():
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_document_metadata_path(doc_hash: str) -> Path:
    """Get path to document metadata file."""
    return CHUNKS_DIR / f"{doc_hash}_metadata.json"


def get_chunks_path(doc_hash: str) -> Path:
    """Get path to chunks file for a document."""
    return CHUNKS_DIR / f"{doc_hash}_chunks.json"


def get_embeddings_path(doc_hash: str) -> Path:
    """Get path to embeddings file for a document."""
    return EMBEDDINGS_DIR / f"{doc_hash}.json"


def get_upload_path(filename: str, doc_hash: str) -> Path:
    """Get path to save uploaded file."""
    ext = Path(filename).suffix
    return UPLOADS_DIR / f"{doc_hash}{ext}"


def document_exists(doc_hash: str) -> bool:
    """
    Check if a document has already been processed.
    
    Args:
        doc_hash: SHA256 hash of the document
        
    Returns:
        True if document was already processed
    """
    metadata_path = get_document_metadata_path(doc_hash)
    chunks_path = get_chunks_path(doc_hash)
    embeddings_path = get_embeddings_path(doc_hash)
    
    return metadata_path.exists() and chunks_path.exists() and embeddings_path.exists()


def save_document_metadata(
    doc_hash: str,
    filename: str,
    file_type: str,
    num_chunks: int,
    num_pages: Optional[int] = None
) -> Dict[str, Any]:
    """
    Save document metadata to disk.
    
    Args:
        doc_hash: SHA256 hash of the document
        filename: Original filename
        file_type: Type of file (pdf, excel)
        num_chunks: Number of chunks created
        num_pages: Number of pages (for PDFs)
        
    Returns:
        The saved metadata dict
    """
    metadata = {
        "doc_hash": doc_hash,
        "filename": filename,
        "file_type": file_type,
        "num_chunks": num_chunks,
        "num_pages": num_pages
    }
    save_json(metadata, get_document_metadata_path(doc_hash))
    return metadata


def load_document_metadata(doc_hash: str) -> Optional[Dict[str, Any]]:
    """
    Load document metadata from disk.
    
    Args:
        doc_hash: SHA256 hash of the document
        
    Returns:
        Metadata dict or None
    """
    return load_json(get_document_metadata_path(doc_hash))


def save_chunks(doc_hash: str, chunks: List[Dict[str, Any]]) -> None:
    """
    Save chunks to disk.
    
    Args:
        doc_hash: SHA256 hash of the document
        chunks: List of chunk dictionaries
    """
    save_json(chunks, get_chunks_path(doc_hash))


def load_chunks(doc_hash: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load chunks from disk.
    
    Args:
        doc_hash: SHA256 hash of the document
        
    Returns:
        List of chunks or None
    """
    return load_json(get_chunks_path(doc_hash))


def get_all_processed_documents() -> List[Dict[str, Any]]:
    """
    Get list of all processed documents.
    
    Returns:
        List of document metadata dictionaries
    """
    documents = []
    if CHUNKS_DIR.exists():
        for file_path in CHUNKS_DIR.glob("*_metadata.json"):
            metadata = load_json(file_path)
            if metadata:
                documents.append(metadata)
    return documents


def get_faiss_metadata_path() -> Path:
    """Get path to FAISS metadata file."""
    return FAISS_INDEX_DIR / "metadata.json"


def get_faiss_index_path() -> Path:
    """Get path to FAISS index file."""
    return FAISS_INDEX_DIR / "index.faiss"
