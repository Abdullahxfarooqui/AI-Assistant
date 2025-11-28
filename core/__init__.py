"""
Core module for RAG application.
Contains extraction, chunking, embedding, vector store, RAG engine, and data analysis components.

Key modules:
- data_engine: Python-based table extraction, statistics, trend/anomaly detection
- extractor: Unified document extraction
- chunker: Smart chunking with data preservation
- embedder: Embedding generation
- vector_store: FAISS-based vector storage
- rag_engine: Query processing and LLM integration
"""
from core.extractor import extract_document, get_extraction_summary
from core.chunker import chunk_document, get_chunk_summary
from core.embedder import embed_chunks, embed_query, compute_doc_hash
from core.vector_store import get_vector_store, VectorStore
from core.rag_engine import (
    query, 
    summarize_document, 
    list_documents,
    get_document_info,
    compare_documents,
    answer_question,
    analyze_query
)

# Import data engine functions
try:
    from core.data_engine import (
        extract_tables_from_pdf,
        extract_tables_from_excel,
        extract_tables_from_csv,
        merge_tables,
        compute_statistics,
        prepare_llm_chunks,
        format_full_table,
        format_sample_rows,
        format_statistics_summary
    )
    HAS_DATA_ENGINE = True
except ImportError:
    HAS_DATA_ENGINE = False

__all__ = [
    # Extraction
    "extract_document",
    "get_extraction_summary",
    # Chunking
    "chunk_document",
    "get_chunk_summary",
    # Embedding
    "embed_chunks",
    "embed_query",
    "compute_doc_hash",
    # Vector Store
    "get_vector_store",
    "VectorStore",
    # RAG Engine
    "query",
    "summarize_document",
    "list_documents",
    "get_document_info",
    "compare_documents",
    "answer_question",
    "analyze_query",
    # Data Engine (if available)
    "extract_tables_from_pdf",
    "extract_tables_from_excel", 
    "extract_tables_from_csv",
    "merge_tables",
    "compute_statistics",
    "prepare_llm_chunks",
    "format_full_table",
    "format_sample_rows",
    "format_statistics_summary"
]
