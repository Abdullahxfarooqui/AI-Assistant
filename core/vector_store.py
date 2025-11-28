"""
Production FAISS vector store with intelligent retrieval.
Optimized for large documents with smart context merging.
"""
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import os


# Storage paths
DATA_DIR = Path("data")
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
METADATA_PATH = DATA_DIR / "chunks_metadata.json"


def ensure_data_dir():
    """Ensure data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


class VectorStore:
    """
    FAISS-based vector store optimized for RAG.
    Features:
    - Cosine similarity with normalized vectors
    - Smart context retrieval with adjacent chunks
    - Per-document search capability
    - Deduplication via chunk hashes
    """
    
    def __init__(self, dimension: int = 1536):
        """Initialize vector store."""
        self.dimension = dimension
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict[str, Any]] = []
        self._chunk_hashes: set = set()
        self._doc_chunks_cache: Dict[str, List[int]] = {}  # doc_hash -> list of indices
        
        ensure_data_dir()
        self._load()
    
    def _load(self) -> bool:
        """Load existing index from disk."""
        if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
            try:
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                
                # Build lookup structures
                self._chunk_hashes = set(c.get("hash", "") for c in self.chunks if c.get("hash"))
                self._rebuild_doc_cache()
                return True
            except Exception as e:
                print(f"Error loading index: {e}")
                self.index = None
                self.chunks = []
        return False
    
    def _save(self) -> None:
        """Save index to disk."""
        if self.index is None:
            return
        
        ensure_data_dir()
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f)
    
    def _rebuild_doc_cache(self) -> None:
        """Rebuild document-to-chunks cache."""
        self._doc_chunks_cache = {}
        for i, chunk in enumerate(self.chunks):
            doc_hash = chunk.get("doc_hash", "")
            if doc_hash:
                if doc_hash not in self._doc_chunks_cache:
                    self._doc_chunks_cache[doc_hash] = []
                self._doc_chunks_cache[doc_hash].append(i)
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        doc_hash: str
    ) -> int:
        """
        Add chunks with their embeddings.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: List of embedding vectors
            doc_hash: Document hash for grouping
            
        Returns:
            Number of chunks added
        """
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            return 0
        
        # Filter duplicates
        new_chunks = []
        new_embeddings = []
        
        for chunk, emb in zip(chunks, embeddings):
            chunk_hash = chunk.get("hash", "")
            if chunk_hash and chunk_hash in self._chunk_hashes:
                continue
            
            # Add doc_hash to chunk
            chunk["doc_hash"] = doc_hash
            new_chunks.append(chunk)
            new_embeddings.append(emb)
            
            if chunk_hash:
                self._chunk_hashes.add(chunk_hash)
        
        if not new_chunks:
            return 0
        
        # Convert to numpy
        vectors = np.array(new_embeddings, dtype=np.float32)
        vectors = self._normalize(vectors)
        
        # Initialize or extend index
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        start_idx = len(self.chunks)
        self.index.add(vectors)
        self.chunks.extend(new_chunks)
        
        # Update doc cache
        for i, chunk in enumerate(new_chunks):
            idx = start_idx + i
            if doc_hash not in self._doc_chunks_cache:
                self._doc_chunks_cache[doc_hash] = []
            self._doc_chunks_cache[doc_hash].append(idx)
        
        self._save()
        return len(new_chunks)
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        doc_hash: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            doc_hash: Optional - filter to specific document
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Normalize query
        query = np.array([query_embedding], dtype=np.float32)
        query = self._normalize(query)
        
        # Search more if filtering by document
        search_k = min(k * 5 if doc_hash else k * 2, self.index.ntotal)
        
        scores, indices = self.index.search(query, search_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            
            # Filter by document if specified
            if doc_hash and chunk.get("doc_hash") != doc_hash:
                continue
            
            results.append((chunk, float(scores[0][i])))
            
            if len(results) >= k:
                break
        
        return results
    
    def search_with_context(
        self,
        query_embedding: List[float],
        k: int = 10,
        context_window: int = 2,
        doc_hash: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search and include adjacent chunks for context.
        
        Returns merged results with surrounding context included.
        """
        base_results = self.search(query_embedding, k=k, doc_hash=doc_hash)
        
        if not base_results:
            return []
        
        # Group results by document
        doc_results: Dict[str, List[Tuple[int, float]]] = {}
        
        for chunk, score in base_results:
            d_hash = chunk.get("doc_hash", "")
            chunk_id = chunk.get("chunk_id", 0)
            
            if d_hash not in doc_results:
                doc_results[d_hash] = []
            doc_results[d_hash].append((chunk_id, score))
        
        # Build context-enriched results
        enriched_results = []
        
        for d_hash, matches in doc_results.items():
            doc_indices = self._doc_chunks_cache.get(d_hash, [])
            doc_chunks = [(self.chunks[i], i) for i in doc_indices]
            doc_chunks.sort(key=lambda x: x[0].get("chunk_id", 0))
            
            # Create mapping from chunk_id to position
            id_to_pos = {c[0].get("chunk_id"): pos for pos, c in enumerate(doc_chunks)}
            
            for chunk_id, score in matches:
                if chunk_id not in id_to_pos:
                    continue
                
                pos = id_to_pos[chunk_id]
                
                # Get context window
                start_pos = max(0, pos - context_window)
                end_pos = min(len(doc_chunks), pos + context_window + 1)
                
                context_chunks = [doc_chunks[i][0] for i in range(start_pos, end_pos)]
                
                # Merge texts
                merged_text = "\n\n---\n\n".join(c.get("text", "") for c in context_chunks)
                
                # Create result entry
                result = {
                    "text": merged_text,
                    "score": score,
                    "doc_hash": d_hash,
                    "filename": context_chunks[0].get("filename", ""),
                    "is_dataset": context_chunks[0].get("is_dataset", False),
                    "chunk_ids": [c.get("chunk_id") for c in context_chunks],
                    "num_chunks": len(context_chunks),
                    "type": context_chunks[pos - start_pos].get("type", "text") if context_chunks else "text"
                }
                
                enriched_results.append(result)
        
        # Sort by score
        enriched_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return enriched_results[:k]
    
    def get_document_chunks(self, doc_hash: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document, sorted by chunk_id."""
        indices = self._doc_chunks_cache.get(doc_hash, [])
        chunks = [self.chunks[i] for i in indices if i < len(self.chunks)]
        chunks.sort(key=lambda x: x.get("chunk_id", 0))
        return chunks
    
    def get_full_document_text(self, doc_hash: str) -> str:
        """Get complete text for a document."""
        chunks = self.get_document_chunks(doc_hash)
        return "\n\n".join(c.get("text", "") for c in chunks)
    
    def get_document_tables(self, doc_hash: str) -> List[Dict[str, Any]]:
        """Get all table chunks for a document."""
        chunks = self.get_document_chunks(doc_hash)
        return [c for c in chunks if c.get("type") == "table"]
    
    def has_document(self, doc_hash: str) -> bool:
        """Check if document is indexed."""
        return doc_hash in self._doc_chunks_cache
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get summary of all indexed documents."""
        docs = []
        for doc_hash in self._doc_chunks_cache.keys():
            chunks = self.get_document_chunks(doc_hash)
            if chunks:
                docs.append({
                    "doc_hash": doc_hash,
                    "filename": chunks[0].get("filename", "unknown"),
                    "num_chunks": len(chunks),
                    "is_dataset": chunks[0].get("is_dataset", False)
                })
        return docs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(self._doc_chunks_cache),
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension
        }
    
    def clear(self) -> None:
        """Clear entire index."""
        self.index = None
        self.chunks = []
        self._chunk_hashes = set()
        self._doc_chunks_cache = {}
        
        if FAISS_INDEX_PATH.exists():
            FAISS_INDEX_PATH.unlink()
        if METADATA_PATH.exists():
            METADATA_PATH.unlink()
    
    def delete_document(self, doc_hash: str) -> int:
        """Delete all chunks for a document (requires rebuild)."""
        if doc_hash not in self._doc_chunks_cache:
            return 0
        
        # Count chunks to delete
        count = len(self._doc_chunks_cache[doc_hash])
        
        # Filter out document chunks
        self.chunks = [c for c in self.chunks if c.get("doc_hash") != doc_hash]
        
        # Rebuild index from remaining chunks
        # This is expensive but necessary for FAISS
        if self.chunks:
            # Would need to re-embed or store embeddings
            # For now, just rebuild cache
            self._rebuild_doc_cache()
            self._chunk_hashes = set(c.get("hash", "") for c in self.chunks if c.get("hash"))
        else:
            self.clear()
        
        return count


# Singleton instance
_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get global vector store instance."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def reset_vector_store() -> None:
    """Reset global vector store instance."""
    global _store
    _store = None
