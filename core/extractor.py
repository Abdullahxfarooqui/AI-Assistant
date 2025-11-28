"""
Production Document Extractor with Full Data Preservation.

This module:
1. Uses data_engine for structured table extraction (Pandas/pdfplumber)
2. Preserves ALL rows - no deduplication of valid data
3. Computes statistics automatically in Python
4. Handles large files (15-50MB) efficiently
5. Returns clean Markdown tables and statistics

Supports: PDF, Excel (xlsx/xls), CSV
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Callable
import io
import re
import json
import gc

# Import data engine for structured extraction
from core.data_engine import (
    extract_tables_from_pdf,
    extract_tables_from_excel,
    extract_tables_from_csv,
    merge_tables,
    compute_statistics,
    format_full_table,
    format_sample_rows,
    format_statistics_summary,
    ExtractedTable,
    DataStatistics
)

# Legacy imports for backward compatibility
try:
    from core.pdf_processor import extract_pdf_complete, get_page_count, clean_text
    HAS_PDF_PROCESSOR = True
except ImportError:
    HAS_PDF_PROCESSOR = False


# ============================================================================
# MAIN EXTRACTION FUNCTIONS
# ============================================================================

def extract_document(
    file_content: bytes,
    filename: str,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """
    Unified document extraction with full data preservation.
    
    Uses Python/Pandas for table extraction - NOT LLM.
    Preserves ALL rows including duplicates (no deduplication).
    Computes statistics automatically.
    
    Args:
        file_content: File bytes
        filename: Original filename
        progress_callback: Optional callback(current, total)
        
    Returns:
        Dict containing:
        - text: Full text/Markdown representation
        - tables: List of table dicts with DataFrame data
        - dataframes: List of actual pandas DataFrames (for analysis)
        - statistics: Computed statistics dict
        - type: File type (pdf/excel/csv)
        - filename: Original filename
        - is_dataset: Whether this is structured data
        - metadata: Additional metadata
    """
    ext = Path(filename).suffix.lower()
    
    if ext == '.pdf':
        return _extract_pdf(file_content, filename, progress_callback)
    elif ext in ['.xlsx', '.xls']:
        return _extract_excel(file_content, filename, progress_callback)
    elif ext == '.csv':
        return _extract_csv(file_content, filename, progress_callback)
    else:
        return _extract_text(file_content, filename)


def _extract_pdf(
    file_content: bytes,
    filename: str,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """
    Extract content from PDF with structured table extraction.
    Uses pdfplumber for accurate table detection.
    """
    try:
        # Use data_engine for structured extraction
        tables = extract_tables_from_pdf(file_content, progress_callback)
        
        # Build result
        all_tables_data = []
        all_dataframes = []
        full_text_parts = []
        total_rows = 0
        
        for table in tables:
            df = table.df
            all_dataframes.append(df)
            total_rows += len(df)
            
            # Create markdown representation
            markdown = _dataframe_to_markdown(df)
            
            table_data = {
                "source": table.source,
                "table_index": table.table_index,
                "headers": table.headers,
                "num_rows": table.num_rows,
                "num_cols": table.num_cols,
                "dtypes": table.dtypes,
                "markdown": markdown,
                "rows": df.head(100).values.tolist()  # Sample for JSON
            }
            all_tables_data.append(table_data)
            full_text_parts.append(f"## Table from {table.source}\n\n{markdown}")
        
        full_text = "\n\n---\n\n".join(full_text_parts)
        
        # Compute statistics from merged data
        if all_dataframes:
            merged_df = merge_tables(tables)
            stats = compute_statistics(merged_df)
            stats_dict = _statistics_to_dict(stats)
        else:
            merged_df = pd.DataFrame()
            stats_dict = {}
        
        return {
            "text": full_text,
            "tables": all_tables_data,
            "dataframes": all_dataframes,
            "statistics": stats_dict,
            "type": "pdf",
            "filename": filename,
            "num_pages": _get_pdf_page_count(file_content),
            "num_tables": len(tables),
            "total_table_rows": total_rows,
            "is_dataset": len(tables) > 0,
            "total_chars": len(full_text)
        }
        
    except Exception as e:
        print(f"PDF extraction error: {e}")
        # Fallback to legacy processor if available
        if HAS_PDF_PROCESSOR:
            return extract_pdf_complete(file_content, progress_callback)
        return {
            "text": f"Error extracting PDF: {str(e)}",
            "tables": [],
            "type": "pdf",
            "filename": filename,
            "error": str(e),
            "is_dataset": False
        }


def _extract_excel(
    file_content: bytes,
    filename: str,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """
    Extract content from Excel with full data preservation.
    Handles multiple sheets and large files.
    ALL rows preserved - no deduplication.
    """
    try:
        # Use data_engine for structured extraction
        tables = extract_tables_from_excel(file_content, progress_callback)
        
        # Build result
        all_tables_data = []
        all_dataframes = []
        full_text_parts = []
        total_rows = 0
        
        for table in tables:
            df = table.df
            all_dataframes.append(df)
            total_rows += len(df)
            
            # Create markdown representation
            markdown = _dataframe_to_markdown(df)
            
            table_data = {
                "sheet_name": table.source,
                "table_index": table.table_index,
                "headers": table.headers,
                "num_rows": table.num_rows,
                "num_cols": table.num_cols,
                "dtypes": table.dtypes,
                "markdown": markdown,
                "rows": df.head(100).values.tolist()
            }
            all_tables_data.append(table_data)
            full_text_parts.append(f"## Sheet: {table.source}\n\n{markdown}")
        
        full_text = "\n\n---\n\n".join(full_text_parts)
        
        # Compute statistics
        if all_dataframes:
            merged_df = merge_tables(tables)
            stats = compute_statistics(merged_df)
            stats_dict = _statistics_to_dict(stats)
        else:
            stats_dict = {}
        
        return {
            "text": full_text,
            "tables": all_tables_data,
            "dataframes": all_dataframes,
            "statistics": stats_dict,
            "type": "excel",
            "filename": filename,
            "num_sheets": len(tables),
            "num_tables": len(tables),
            "total_table_rows": total_rows,
            "is_dataset": True,
            "total_chars": len(full_text)
        }
        
    except Exception as e:
        print(f"Excel extraction error: {e}")
        return {
            "text": f"Error extracting Excel: {str(e)}",
            "tables": [],
            "type": "excel",
            "filename": filename,
            "error": str(e),
            "is_dataset": False
        }


def _extract_csv(
    file_content: bytes,
    filename: str,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """Extract content from CSV with full data preservation."""
    try:
        tables = extract_tables_from_csv(file_content, progress_callback)
        
        if not tables:
            return {
                "text": "Empty CSV file",
                "tables": [],
                "type": "csv",
                "filename": filename,
                "is_dataset": False
            }
        
        table = tables[0]
        df = table.df
        markdown = _dataframe_to_markdown(df)
        
        stats = compute_statistics(df)
        stats_dict = _statistics_to_dict(stats)
        
        table_data = {
            "sheet_name": "CSV",
            "table_index": 0,
            "headers": table.headers,
            "num_rows": table.num_rows,
            "num_cols": table.num_cols,
            "dtypes": table.dtypes,
            "markdown": markdown,
            "rows": df.head(100).values.tolist()
        }
        
        return {
            "text": markdown,
            "tables": [table_data],
            "dataframes": [df],
            "statistics": stats_dict,
            "type": "csv",
            "filename": filename,
            "num_tables": 1,
            "total_table_rows": len(df),
            "is_dataset": True,
            "total_chars": len(markdown)
        }
        
    except Exception as e:
        return {
            "text": f"Error extracting CSV: {str(e)}",
            "tables": [],
            "type": "csv",
            "filename": filename,
            "error": str(e),
            "is_dataset": False
        }


def _extract_text(
    file_content: bytes,
    filename: str
) -> Dict[str, Any]:
    """Extract plain text files."""
    try:
        text = file_content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = file_content.decode('latin-1')
        except:
            text = file_content.decode('utf-8', errors='ignore')
    
    return {
        "text": text,
        "tables": [],
        "type": "text",
        "filename": filename,
        "is_dataset": False,
        "total_chars": len(text)
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 200) -> str:
    """
    Convert DataFrame to clean Markdown table.
    Shows up to max_rows rows, with note if truncated.
    """
    if df.empty:
        return "*Empty table*"
    
    # Clean column names
    columns = [str(c).strip() if pd.notna(c) else f"Column_{i}" for i, c in enumerate(df.columns)]
    
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    
    display_df = df.head(max_rows)
    
    for _, row in display_df.iterrows():
        values = []
        for v in row:
            if pd.isna(v):
                values.append("NULL")
            else:
                s = str(v).replace("|", "\\|")[:100]  # Escape pipes, limit length
                values.append(s)
        lines.append("| " + " | ".join(values) + " |")
    
    if len(df) > max_rows:
        lines.append(f"\n*Showing {max_rows} of {len(df):,} rows*")
    
    return "\n".join(lines)


def _statistics_to_dict(stats: DataStatistics) -> Dict[str, Any]:
    """Convert DataStatistics object to serializable dict."""
    return {
        "total_rows": stats.total_rows,
        "total_columns": stats.total_columns,
        "memory_mb": stats.memory_mb,
        "numeric_columns": stats.numeric_columns,
        "date_columns": stats.date_columns,
        "categorical_columns": stats.categorical_columns,
        "null_summary": stats.null_summary,
        "columns": [
            {
                "name": cs.name,
                "dtype": cs.dtype,
                "non_null_count": cs.non_null_count,
                "null_count": cs.null_count,
                "unique_count": cs.unique_count,
                "min": cs.min_val,
                "max": cs.max_val,
                "mean": cs.mean_val,
                "median": cs.median_val,
                "sum": cs.sum_val,
                "top_values": cs.top_values,
                "detected_unit": cs.detected_unit
            }
            for cs in stats.column_stats
        ],
        "anomalies": stats.anomalies,
        "trends": stats.trends
    }


def _get_pdf_page_count(file_content: bytes) -> int:
    """Get PDF page count."""
    try:
        import fitz
        doc = fitz.open(stream=file_content, filetype="pdf")
        count = len(doc)
        doc.close()
        return count
    except:
        return 0


# ============================================================================
# EXTRACTION SUMMARY
# ============================================================================

def get_extraction_summary(extraction_result: Dict[str, Any]) -> str:
    """
    Generate human-readable summary of extracted content.
    Shows statistics and structure overview.
    """
    lines = []
    
    filename = extraction_result.get("filename", "Unknown")
    doc_type = extraction_result.get("type", "unknown").upper()
    
    lines.append(f"📄 **Document:** {filename}")
    lines.append(f"📁 **Type:** {doc_type}")
    
    if doc_type == "PDF":
        lines.append(f"📑 **Pages:** {extraction_result.get('num_pages', 0)}")
    
    num_tables = extraction_result.get("num_tables", 0)
    total_rows = extraction_result.get("total_table_rows", 0)
    
    if num_tables > 0:
        lines.append(f"📊 **Tables Found:** {num_tables}")
        lines.append(f"📈 **Total Data Rows:** {total_rows:,}")
    
    # Show table details
    for i, table in enumerate(extraction_result.get("tables", [])[:5]):
        headers = table.get("headers", [])
        rows = table.get("num_rows", 0)
        cols = table.get("num_cols", len(headers))
        source = table.get("sheet_name") or table.get("source", f"Table {i+1}")
        
        headers_preview = ", ".join(headers[:5])
        if len(headers) > 5:
            headers_preview += f"... (+{len(headers)-5} more)"
        
        lines.append(f"  - **{source}:** {rows:,} rows × {cols} cols")
        lines.append(f"    Columns: `{headers_preview}`")
    
    # Show statistics summary if available
    stats = extraction_result.get("statistics", {})
    if stats:
        lines.append("")
        lines.append("**📊 Statistics:**")
        
        if stats.get("numeric_columns"):
            lines.append(f"  - Numeric columns: {len(stats['numeric_columns'])}")
        if stats.get("date_columns"):
            lines.append(f"  - Date columns: {len(stats['date_columns'])}")
        if stats.get("anomalies"):
            lines.append(f"  - ⚠️ Anomalies detected: {len(stats['anomalies'])}")
        if stats.get("trends"):
            lines.append(f"  - 📈 Trends detected: {len(stats['trends'])}")
    
    is_dataset = extraction_result.get("is_dataset", False)
    lines.append("")
    lines.append(f"🔍 **Data Type:** {'Structured Dataset' if is_dataset else 'Text Document'}")
    
    return "\n".join(lines)


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

def extract_text(file_content: bytes, filename: str) -> Dict[str, Any]:
    """Legacy function - use extract_document instead."""
    return extract_document(file_content, filename)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Legacy function - public wrapper."""
    return _dataframe_to_markdown(df, max_rows)


def dataframe_to_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Legacy function - compute stats for a DataFrame."""
    stats = compute_statistics(df)
    return _statistics_to_dict(stats)
