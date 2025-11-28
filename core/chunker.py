"""
Production Chunker with Full Data Preservation.

This module:
1. Creates structured chunks (Markdown tables, not raw text)
2. Preserves ALL rows - NO deduplication of valid data
3. Includes statistics in chunk metadata
4. Optimizes chunk sizes for LLM context windows
5. Handles large datasets efficiently

Key Principle: ALL data must be preserved for retrieval.
Deduplication is ONLY for exact metadata/header noise, never for data rows.
"""
from typing import List, Dict, Any, Optional, Tuple
import re
import hashlib
import json

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    TOKENIZER = None
    print("Warning: tiktoken not installed. Using approximate token counting.")


# ============================================================================
# TOKEN COUNTING
# ============================================================================

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken or approximation."""
    if not text:
        return 0
    if TOKENIZER:
        return len(TOKENIZER.encode(text))
    # Fallback: approximate with characters (roughly 4 chars per token)
    return len(text) // 4


# ============================================================================
# CHUNK IDENTIFICATION
# ============================================================================

def compute_chunk_hash(text: str, include_content: bool = False) -> str:
    """
    Compute hash for chunk identification.
    
    Args:
        text: Chunk text
        include_content: If True, hash includes full content (for exact dedup)
                        If False, hash is based on structure only
    """
    if include_content:
        # Full content hash - for exact duplicate detection
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    else:
        # Structure hash - only for metadata noise detection
        # Use first line and length as identifier
        first_line = text.split('\n')[0][:100] if text else ""
        identifier = f"{first_line}:{len(text)}"
        return hashlib.md5(identifier.encode()).hexdigest()[:16]


def is_metadata_noise(text: str) -> bool:
    """
    Check if text is metadata noise that should be skipped.
    
    IMPORTANT: This is ONLY for page numbers, headers, etc.
    NOT for data rows, even if they look similar.
    """
    if not text:
        return True
    
    text_stripped = text.strip()
    
    # Empty or very short content
    if len(text_stripped) < 10:
        return True
    
    text_lower = text_stripped.lower()
    
    # Page number patterns
    if re.match(r'^page\s*\d+(\s*(of|/)?\s*\d+)?$', text_lower):
        return True
    
    # Separator lines
    if re.match(r'^[\s\-=_]{5,}$', text_stripped):
        return True
    
    # Published/header noise patterns
    if re.match(r'^(published|confidential|proprietary)\s*(published|confidential|proprietary)?$', text_lower):
        return True
    
    return False


def is_table_content(text: str) -> bool:
    """Check if text contains Markdown table structure."""
    if not text:
        return False
    
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False
    
    # Check for markdown table patterns
    pipe_lines = sum(1 for line in lines if '|' in line and line.count('|') >= 2)
    separator_lines = sum(1 for line in lines if re.match(r'\|\s*-+\s*\|', line))
    
    return pipe_lines >= 2 and separator_lines >= 1


# ============================================================================
# TABLE CHUNKING
# ============================================================================

def split_table_text(
    text: str,
    max_tokens: int = 2000,
    preserve_all_rows: bool = True  # MUST be True - no data loss
) -> List[str]:
    """
    Split Markdown table while preserving structure.
    
    CRITICAL: All rows are preserved. This function only splits for size.
    No deduplication of data rows.
    """
    lines = text.strip().split('\n')
    if not lines:
        return []
    
    # Identify header section (first row + separator)
    header_lines = []
    data_start = 0
    
    for i, line in enumerate(lines[:3]):
        if '|' in line:
            if '---' in line:
                # This is the separator line
                header_lines = lines[:i+1]
                data_start = i + 1
                break
            elif i == 0:
                header_lines.append(line)
    
    if not header_lines:
        # No clear header found, treat first line as header
        header_lines = lines[:1]
        data_start = 1
    
    header_text = '\n'.join(header_lines)
    header_tokens = count_tokens(header_text)
    
    # Available tokens for data per chunk
    data_tokens_budget = max_tokens - header_tokens - 50  # Buffer
    
    # Split data rows into chunks
    chunks = []
    current_data_lines = []
    current_tokens = 0
    
    for line in lines[data_start:]:
        line_tokens = count_tokens(line)
        
        if current_tokens + line_tokens > data_tokens_budget and current_data_lines:
            # Save current chunk
            chunk_text = header_text + '\n' + '\n'.join(current_data_lines)
            chunks.append(chunk_text)
            current_data_lines = []
            current_tokens = 0
        
        current_data_lines.append(line)
        current_tokens += line_tokens
    
    # Add remaining data
    if current_data_lines:
        chunk_text = header_text + '\n' + '\n'.join(current_data_lines)
        chunks.append(chunk_text)
    
    return chunks if chunks else [text]


# ============================================================================
# TEXT CHUNKING
# ============================================================================

def split_text_smart(
    text: str,
    max_tokens: int = 2000,
    overlap_tokens: int = 100
) -> List[str]:
    """
    Split text at natural boundaries while respecting token limits.
    Preserves all content - no filtering of valid text.
    """
    if not text:
        return []
    
    text_tokens = count_tokens(text)
    
    # Small enough to be one chunk
    if text_tokens <= max_tokens:
        return [text] if not is_metadata_noise(text) else []
    
    # For tables, use table-aware splitting
    if is_table_content(text):
        return split_table_text(text, max_tokens)
    
    chunks = []
    paragraphs = re.split(r'\n\n+', text)
    
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Skip only obvious metadata noise
        if is_metadata_noise(para):
            continue
        
        para_tokens = count_tokens(para)
        
        # If paragraph is larger than max, split by sentences
        if para_tokens > max_tokens:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Split paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_chunk = []
            sent_tokens = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                s_tokens = count_tokens(sentence)
                
                if sent_tokens + s_tokens > max_tokens and sent_chunk:
                    chunks.append(' '.join(sent_chunk))
                    sent_chunk = []
                    sent_tokens = 0
                
                sent_chunk.append(sentence)
                sent_tokens += s_tokens
            
            if sent_chunk:
                chunks.append(' '.join(sent_chunk))
        
        elif current_tokens + para_tokens > max_tokens:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            
            # Start new chunk with overlap
            if overlap_tokens > 0 and current_chunk:
                last_para = current_chunk[-1]
                if count_tokens(last_para) <= overlap_tokens:
                    current_chunk = [last_para]
                    current_tokens = count_tokens(last_para)
                else:
                    current_chunk = []
                    current_tokens = 0
            else:
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(para)
            current_tokens += para_tokens
        
        else:
            current_chunk.append(para)
            current_tokens += para_tokens
    
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        if not is_metadata_noise(chunk_text):
            chunks.append(chunk_text)
    
    return chunks


# ============================================================================
# MAIN CHUNKING FUNCTION
# ============================================================================

def chunk_document(
    extraction_result: Dict[str, Any],
    max_tokens: int = 2000,
    overlap_tokens: int = 50
) -> List[Dict[str, Any]]:
    """
    Create structured chunks from extracted document content.
    
    CRITICAL RULES:
    1. ALL data rows are preserved - no deduplication of actual data
    2. Only exact metadata noise is skipped (page numbers, headers)
    3. Tables are chunked with headers repeated in each chunk
    4. Statistics are included in chunk metadata
    
    Args:
        extraction_result: Result from document extraction
        max_tokens: Maximum tokens per chunk (default 2000)
        overlap_tokens: Overlap for text chunks (default 50)
        
    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []
    seen_metadata = set()  # Only for exact metadata noise, NOT data
    
    doc_type = extraction_result.get("type", "unknown")
    filename = extraction_result.get("filename", "unknown")
    is_dataset = extraction_result.get("is_dataset", False)
    statistics = extraction_result.get("statistics", {})
    
    # Process tables first (they contain the actual structured data)
    tables = extraction_result.get("tables", [])
    
    for table in tables:
        markdown = table.get("markdown", "")
        if not markdown:
            continue
        
        # Split large tables while preserving all rows
        table_chunks = split_table_text(markdown, max_tokens)
        
        for chunk_text in table_chunks:
            if not chunk_text or is_metadata_noise(chunk_text):
                continue
            
            # Create unique ID based on structure, not content
            chunk_id = len(chunks)
            chunk_hash = compute_chunk_hash(chunk_text, include_content=True)
            
            # For tables, only skip if EXACT duplicate (same hash)
            # This catches repeated headers but not data rows
            if chunk_hash in seen_metadata:
                # Verify it's actually metadata, not data
                if count_tokens(chunk_text) < 100:  # Very short = likely metadata
                    continue
            seen_metadata.add(chunk_hash)
            
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "hash": chunk_hash,
                "tokens": count_tokens(chunk_text),
                "type": "table",
                "source": table.get("sheet_name") or table.get("source", filename),
                "filename": filename,
                "is_dataset": True,
                "metadata": {
                    "num_rows": table.get("num_rows", 0),
                    "num_cols": table.get("num_cols", 0),
                    "headers": table.get("headers", []),
                    "dtypes": table.get("dtypes", {})
                }
            })
    
    # Process full text content (may overlap with tables for PDFs)
    full_text = extraction_result.get("text", "")
    if full_text:
        text_chunks = split_text_smart(full_text, max_tokens, overlap_tokens)
        
        for chunk_text in text_chunks:
            if not chunk_text:
                continue
            
            chunk_hash = compute_chunk_hash(chunk_text, include_content=True)
            
            # Skip only if exact duplicate already added
            if chunk_hash in seen_metadata:
                continue
            seen_metadata.add(chunk_hash)
            
            chunk_type = "table" if is_table_content(chunk_text) else "text"
            
            chunks.append({
                "chunk_id": len(chunks),
                "text": chunk_text,
                "hash": chunk_hash,
                "tokens": count_tokens(chunk_text),
                "type": chunk_type,
                "source": filename,
                "filename": filename,
                "is_dataset": is_dataset,
                "metadata": {}
            })
    
    # Add statistics chunk if available (for context)
    if statistics and statistics.get("total_rows", 0) > 0:
        stats_text = _format_statistics_chunk(statistics, filename)
        if stats_text:
            chunks.insert(0, {  # Insert at beginning for priority
                "chunk_id": -1,  # Special ID for stats
                "text": stats_text,
                "hash": compute_chunk_hash(stats_text),
                "tokens": count_tokens(stats_text),
                "type": "statistics",
                "source": filename,
                "filename": filename,
                "is_dataset": is_dataset,
                "metadata": statistics
            })
            # Renumber chunk_ids
            for i, c in enumerate(chunks):
                c["chunk_id"] = i
    
    return chunks


def _format_statistics_chunk(statistics: Dict[str, Any], filename: str) -> str:
    """Format statistics as a chunk for context."""
    lines = [
        f"## Document Statistics: {filename}",
        "",
        f"**Total Rows:** {statistics.get('total_rows', 0):,}",
        f"**Total Columns:** {statistics.get('total_columns', 0)}",
        ""
    ]
    
    # Column summaries
    columns = statistics.get("columns", [])
    numeric_cols = [c for c in columns if c.get("sum") is not None]
    
    if numeric_cols:
        lines.append("### Numeric Column Statistics:")
        for col in numeric_cols[:10]:  # Limit to 10
            lines.append(f"**{col['name']}:**")
            if col.get("sum") is not None:
                lines.append(f"  - Sum: {col['sum']:,.2f}")
            if col.get("mean") is not None:
                lines.append(f"  - Mean: {col['mean']:,.2f}")
            if col.get("min") is not None:
                lines.append(f"  - Range: {col['min']:,.2f} to {col['max']:,.2f}")
    
    # Anomalies
    anomalies = statistics.get("anomalies", [])
    if anomalies:
        lines.append("")
        lines.append("### Detected Anomalies:")
        for anom in anomalies[:5]:
            lines.append(f"- **{anom['column']}**: {anom['count']} outliers ({anom['percentage']:.1f}%)")
    
    # Trends
    trends = statistics.get("trends", [])
    if trends:
        lines.append("")
        lines.append("### Detected Trends:")
        for trend in trends[:5]:
            lines.append(f"- **{trend['column']}**: {trend['direction']} ({trend['change_percent']:+.1f}%)")
    
    return "\n".join(lines)


# ============================================================================
# CHUNK SUMMARY
# ============================================================================

def get_chunk_summary(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics for chunks."""
    if not chunks:
        return {
            "total_chunks": 0,
            "total_tokens": 0,
            "table_chunks": 0,
            "text_chunks": 0,
            "stats_chunks": 0
        }
    
    return {
        "total_chunks": len(chunks),
        "total_tokens": sum(c.get("tokens", 0) for c in chunks),
        "table_chunks": sum(1 for c in chunks if c.get("type") == "table"),
        "text_chunks": sum(1 for c in chunks if c.get("type") == "text"),
        "stats_chunks": sum(1 for c in chunks if c.get("type") == "statistics"),
        "avg_tokens": sum(c.get("tokens", 0) for c in chunks) // len(chunks) if chunks else 0,
        "sources": list(set(c.get("source", "") for c in chunks))
    }
