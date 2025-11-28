"""
Production Data Engine for Large-Scale Document Analysis.

This module handles:
- Structured table extraction from PDFs and Excel files
- DataFrame merging with full data preservation (NO deduplication)
- Python-based statistics computation (totals, averages, min/max, trends, anomalies)
- Chunking data for LLM reasoning with structured JSON/Markdown
- All rows preserved - no data loss

Dependencies: pandas, pdfplumber, openpyxl, numpy
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import io
import re
import json
import warnings
from datetime import datetime
from dataclasses import dataclass, field

warnings.filterwarnings('ignore', category=UserWarning)

# Try to import pdfplumber (primary) and camelot (fallback)
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("Warning: pdfplumber not installed. Install with: pip install pdfplumber")

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExtractedTable:
    """Represents an extracted table with full data and metadata."""
    df: pd.DataFrame
    source: str  # filename, sheet name, or page number
    table_index: int
    num_rows: int
    num_cols: int
    headers: List[str]
    dtypes: Dict[str, str]
    is_complete: bool = True  # False if truncated
    extraction_method: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "table_index": self.table_index,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "headers": self.headers,
            "dtypes": self.dtypes,
            "is_complete": self.is_complete,
            "extraction_method": self.extraction_method,
            "sample_data": self.df.head(5).to_dict(orient='records')
        }


@dataclass
class ColumnStatistics:
    """Statistics for a single column."""
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    unique_count: int
    # Numeric fields
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    median_val: Optional[float] = None
    std_val: Optional[float] = None
    sum_val: Optional[float] = None
    # Categorical fields
    top_values: Optional[List[Tuple[str, int]]] = None
    # Date fields
    date_range: Optional[Tuple[str, str]] = None
    # Detected unit
    detected_unit: Optional[str] = None


@dataclass 
class DataStatistics:
    """Complete statistics for a DataFrame."""
    total_rows: int
    total_columns: int
    memory_mb: float
    column_stats: List[ColumnStatistics]
    date_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    null_summary: Dict[str, int]
    anomalies: List[Dict[str, Any]]
    trends: List[Dict[str, Any]]


# ============================================================================
# TABLE EXTRACTION FROM PDF
# ============================================================================

def extract_tables_from_pdf(
    pdf_path_or_bytes: Union[str, Path, bytes],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_pages: Optional[int] = None
) -> List[ExtractedTable]:
    """
    Extract ALL tables from a PDF using pdfplumber (primary) with PyMuPDF fallback.
    
    IMPORTANT: Preserves ALL rows - no deduplication or filtering of valid data.
    
    Args:
        pdf_path_or_bytes: Path to PDF file or bytes content
        progress_callback: Optional callback(current_page, total_pages)
        max_pages: Optional limit on pages to process
        
    Returns:
        List of ExtractedTable objects containing DataFrames
    """
    if not HAS_PDFPLUMBER:
        raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")
    
    tables = []
    table_index = 0
    
    # Open PDF
    if isinstance(pdf_path_or_bytes, bytes):
        pdf_file = io.BytesIO(pdf_path_or_bytes)
    else:
        pdf_file = str(pdf_path_or_bytes)
    
    with pdfplumber.open(pdf_file) as pdf:
        total_pages = len(pdf.pages)
        pages_to_process = total_pages if max_pages is None else min(max_pages, total_pages)
        
        for page_num, page in enumerate(pdf.pages[:pages_to_process]):
            if progress_callback:
                progress_callback(page_num + 1, total_pages)
            
            # Extract tables from page
            page_tables = page.extract_tables()
            
            for raw_table in page_tables:
                if not raw_table or len(raw_table) < 2:
                    continue
                
                # Convert to DataFrame - PRESERVE ALL ROWS
                df = _raw_table_to_dataframe(raw_table)
                
                if df.empty or len(df) < 1:
                    continue
                
                tables.append(ExtractedTable(
                    df=df,
                    source=f"Page {page_num + 1}",
                    table_index=table_index,
                    num_rows=len(df),
                    num_cols=len(df.columns),
                    headers=list(df.columns),
                    dtypes={col: str(df[col].dtype) for col in df.columns},
                    is_complete=True,
                    extraction_method="pdfplumber"
                ))
                table_index += 1
    
    # Try PyMuPDF fallback if no tables found
    if not tables and HAS_PYMUPDF:
        tables = _extract_tables_pymupdf(pdf_path_or_bytes, progress_callback, max_pages)
    
    return tables


def _raw_table_to_dataframe(raw_table: List[List]) -> pd.DataFrame:
    """
    Convert raw table data to clean DataFrame.
    PRESERVES ALL ROWS - no deduplication.
    """
    if not raw_table or len(raw_table) < 2:
        return pd.DataFrame()
    
    # First row as headers, rest as data
    headers = [_clean_header(h) for h in raw_table[0]]
    data_rows = raw_table[1:]
    
    # Handle duplicate/empty headers
    seen = {}
    clean_headers = []
    for i, h in enumerate(headers):
        if not h or h.isspace():
            h = f"Column_{i+1}"
        if h in seen:
            seen[h] += 1
            h = f"{h}_{seen[h]}"
        else:
            seen[h] = 0
        clean_headers.append(h)
    
    # Create DataFrame - ALL ROWS preserved
    df = pd.DataFrame(data_rows, columns=clean_headers)
    
    # Clean cell values but don't remove rows
    for col in df.columns:
        df[col] = df[col].apply(_clean_cell_value)
    
    # Attempt type inference
    df = _infer_column_types(df)
    
    return df


def _extract_tables_pymupdf(
    pdf_bytes: bytes,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_pages: Optional[int] = None
) -> List[ExtractedTable]:
    """Fallback table extraction using PyMuPDF."""
    if not HAS_PYMUPDF:
        return []
    
    tables = []
    table_index = 0
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    pages_to_process = total_pages if max_pages is None else min(max_pages, total_pages)
    
    for page_num in range(pages_to_process):
        if progress_callback:
            progress_callback(page_num + 1, total_pages)
        
        page = doc[page_num]
        
        try:
            page_tables = page.find_tables()
            
            for tab in page_tables:
                raw_table = tab.extract()
                if not raw_table or len(raw_table) < 2:
                    continue
                
                df = _raw_table_to_dataframe(raw_table)
                
                if df.empty:
                    continue
                
                tables.append(ExtractedTable(
                    df=df,
                    source=f"Page {page_num + 1}",
                    table_index=table_index,
                    num_rows=len(df),
                    num_cols=len(df.columns),
                    headers=list(df.columns),
                    dtypes={col: str(df[col].dtype) for col in df.columns},
                    is_complete=True,
                    extraction_method="pymupdf"
                ))
                table_index += 1
        except Exception as e:
            print(f"PyMuPDF table extraction error on page {page_num}: {e}")
    
    doc.close()
    return tables


# ============================================================================
# TABLE EXTRACTION FROM EXCEL
# ============================================================================

def extract_tables_from_excel(
    excel_path_or_bytes: Union[str, Path, bytes],
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[ExtractedTable]:
    """
    Extract ALL tables from Excel file with full data preservation.
    
    Handles:
    - Multiple sheets
    - Large files (reads in optimized mode)
    - All data types
    - NO deduplication - all rows preserved
    
    Args:
        excel_path_or_bytes: Path to Excel file or bytes content
        progress_callback: Optional callback(current_sheet, total_sheets)
        
    Returns:
        List of ExtractedTable objects containing DataFrames
    """
    tables = []
    table_index = 0
    
    # Handle bytes input
    if isinstance(excel_path_or_bytes, bytes):
        excel_file = io.BytesIO(excel_path_or_bytes)
    else:
        excel_file = excel_path_or_bytes
    
    # Get sheet names first
    xlsx = pd.ExcelFile(excel_file)
    sheet_names = xlsx.sheet_names
    
    for sheet_idx, sheet_name in enumerate(sheet_names):
        if progress_callback:
            progress_callback(sheet_idx + 1, len(sheet_names))
        
        try:
            # Read entire sheet - NO row limit, preserve ALL data
            df = pd.read_excel(
                xlsx, 
                sheet_name=sheet_name,
                dtype=str,  # Read as string first to preserve all values
                na_values=['', 'NA', 'N/A', 'null', 'NULL', 'None'],
                keep_default_na=True
            )
            
            if df.empty:
                continue
            
            # Drop completely empty rows and columns
            df = df.dropna(how='all', axis=0)
            df = df.dropna(how='all', axis=1)
            
            if df.empty:
                continue
            
            # Clean headers
            df.columns = [_clean_header(str(c)) for c in df.columns]
            
            # Infer proper types
            df = _infer_column_types(df)
            
            tables.append(ExtractedTable(
                df=df,
                source=sheet_name,
                table_index=table_index,
                num_rows=len(df),
                num_cols=len(df.columns),
                headers=list(df.columns),
                dtypes={col: str(df[col].dtype) for col in df.columns},
                is_complete=True,
                extraction_method="pandas"
            ))
            table_index += 1
            
        except Exception as e:
            print(f"Error reading sheet {sheet_name}: {e}")
    
    xlsx.close()
    return tables


def extract_tables_from_csv(
    csv_path_or_bytes: Union[str, Path, bytes],
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[ExtractedTable]:
    """Extract table from CSV file with full data preservation."""
    if isinstance(csv_path_or_bytes, bytes):
        csv_file = io.BytesIO(csv_path_or_bytes)
    else:
        csv_file = csv_path_or_bytes
    
    # Try multiple encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    df = None
    
    for encoding in encodings:
        try:
            if isinstance(csv_file, io.BytesIO):
                csv_file.seek(0)
            df = pd.read_csv(
                csv_file,
                encoding=encoding,
                dtype=str,
                na_values=['', 'NA', 'N/A', 'null', 'NULL', 'None'],
                keep_default_na=True,
                low_memory=False  # Better type inference
            )
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    
    if df is None or df.empty:
        return []
    
    if progress_callback:
        progress_callback(1, 1)
    
    # Clean
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)
    df.columns = [_clean_header(str(c)) for c in df.columns]
    df = _infer_column_types(df)
    
    return [ExtractedTable(
        df=df,
        source="CSV",
        table_index=0,
        num_rows=len(df),
        num_cols=len(df.columns),
        headers=list(df.columns),
        dtypes={col: str(df[col].dtype) for col in df.columns},
        is_complete=True,
        extraction_method="pandas"
    )]


# ============================================================================
# TABLE MERGING
# ============================================================================

def merge_tables(
    tables: List[ExtractedTable],
    merge_strategy: str = "auto"
) -> pd.DataFrame:
    """
    Merge multiple tables into a single DataFrame.
    
    Strategies:
    - "auto": Detect if tables have same schema, concat if yes, else keep separate
    - "concat": Concatenate all tables vertically
    - "union": Keep all columns, fill missing with NULL
    
    IMPORTANT: ALL rows are preserved, including duplicates.
    
    Args:
        tables: List of ExtractedTable objects
        merge_strategy: How to merge tables
        
    Returns:
        Merged DataFrame with all data preserved
    """
    if not tables:
        return pd.DataFrame()
    
    if len(tables) == 1:
        return tables[0].df.copy()
    
    dfs = [t.df for t in tables]
    
    if merge_strategy == "concat":
        # Simple vertical concatenation
        return pd.concat(dfs, ignore_index=True)
    
    elif merge_strategy == "union":
        # Keep all columns from all tables
        all_cols = set()
        for df in dfs:
            all_cols.update(df.columns)
        
        aligned_dfs = []
        for df in dfs:
            aligned = df.reindex(columns=list(all_cols))
            aligned_dfs.append(aligned)
        
        return pd.concat(aligned_dfs, ignore_index=True)
    
    else:  # "auto"
        # Check if all tables have same columns
        first_cols = set(dfs[0].columns)
        same_schema = all(set(df.columns) == first_cols for df in dfs)
        
        if same_schema:
            return pd.concat(dfs, ignore_index=True)
        else:
            # Use union strategy for different schemas
            return merge_tables(tables, merge_strategy="union")


# ============================================================================
# STATISTICS COMPUTATION
# ============================================================================

def compute_statistics(df: pd.DataFrame) -> DataStatistics:
    """
    Compute comprehensive statistics for a DataFrame.
    
    Includes:
    - Per-column: min, max, mean, median, std, sum (numeric)
    - Per-column: unique values, top values (categorical)
    - Date ranges for date columns
    - NULL counts per column
    - Anomaly detection (outliers, unexpected patterns)
    - Trend detection for time-series data
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        DataStatistics object with complete analysis
    """
    if df.empty:
        return DataStatistics(
            total_rows=0,
            total_columns=0,
            memory_mb=0,
            column_stats=[],
            date_columns=[],
            numeric_columns=[],
            categorical_columns=[],
            null_summary={},
            anomalies=[],
            trends=[]
        )
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Try to identify date columns that are strings
    for col in df.columns:
        if col not in date_cols and df[col].dtype == 'object':
            sample = df[col].dropna().head(100)
            if _looks_like_date_column(sample):
                try:
                    pd.to_datetime(sample, errors='coerce')
                    date_cols.append(col)
                except:
                    pass
    
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in date_cols]
    
    # Compute per-column statistics
    column_stats = []
    null_summary = {}
    
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        non_null_count = int(df[col].notna().sum())
        null_summary[col] = null_count
        
        col_stat = ColumnStatistics(
            name=col,
            dtype=str(df[col].dtype),
            non_null_count=non_null_count,
            null_count=null_count,
            unique_count=int(df[col].nunique())
        )
        
        # Numeric statistics
        if col in numeric_cols:
            col_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(col_data) > 0:
                col_stat.min_val = float(col_data.min())
                col_stat.max_val = float(col_data.max())
                col_stat.mean_val = float(col_data.mean())
                col_stat.median_val = float(col_data.median())
                col_stat.std_val = float(col_data.std()) if len(col_data) > 1 else 0.0
                col_stat.sum_val = float(col_data.sum())
                col_stat.detected_unit = _detect_unit(col, col_data)
        
        # Categorical top values
        elif col in categorical_cols:
            value_counts = df[col].value_counts().head(10)
            col_stat.top_values = [(str(k), int(v)) for k, v in value_counts.items()]
        
        # Date range
        if col in date_cols:
            try:
                dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(dates) > 0:
                    col_stat.date_range = (str(dates.min()), str(dates.max()))
            except:
                pass
        
        column_stats.append(col_stat)
    
    # Detect anomalies
    anomalies = _detect_anomalies(df, numeric_cols)
    
    # Detect trends
    trends = _detect_trends(df, numeric_cols, date_cols)
    
    return DataStatistics(
        total_rows=len(df),
        total_columns=len(df.columns),
        memory_mb=round(memory_mb, 2),
        column_stats=column_stats,
        date_columns=date_cols,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        null_summary=null_summary,
        anomalies=anomalies,
        trends=trends
    )


def _detect_anomalies(df: pd.DataFrame, numeric_cols: List[str]) -> List[Dict[str, Any]]:
    """Detect outliers and anomalies in numeric columns."""
    anomalies = []
    
    for col in numeric_cols:
        col_data = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(col_data) < 10:
            continue
        
        # IQR method for outliers
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = col_data[(col_data < lower) | (col_data > upper)]
        
        if len(outliers) > 0:
            anomalies.append({
                "column": col,
                "type": "outliers",
                "count": len(outliers),
                "percentage": round(len(outliers) / len(col_data) * 100, 2),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "sample_values": outliers.head(5).tolist()
            })
    
    return anomalies


def _detect_trends(
    df: pd.DataFrame, 
    numeric_cols: List[str],
    date_cols: List[str]
) -> List[Dict[str, Any]]:
    """Detect trends in time-series data."""
    trends = []
    
    if not date_cols or not numeric_cols:
        return trends
    
    # Use first date column as time index
    date_col = date_cols[0]
    
    try:
        df_sorted = df.copy()
        df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors='coerce')
        df_sorted = df_sorted.dropna(subset=[date_col]).sort_values(date_col)
        
        if len(df_sorted) < 5:
            return trends
        
        for num_col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            values = pd.to_numeric(df_sorted[num_col], errors='coerce').dropna()
            if len(values) < 5:
                continue
            
            # Simple trend detection via linear regression
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            slope = coeffs[0]
            
            # Determine trend direction
            if abs(slope) < values.std() * 0.01:
                direction = "stable"
            elif slope > 0:
                direction = "increasing"
            else:
                direction = "decreasing"
            
            trends.append({
                "column": num_col,
                "date_column": date_col,
                "direction": direction,
                "slope": float(slope),
                "start_value": float(values.iloc[0]),
                "end_value": float(values.iloc[-1]),
                "change_percent": float((values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100) if values.iloc[0] != 0 else 0
            })
    except Exception as e:
        print(f"Trend detection error: {e}")
    
    return trends


# ============================================================================
# LLM CHUNK PREPARATION
# ============================================================================

def prepare_llm_chunks(
    df: pd.DataFrame,
    chunk_size: int = 500,
    output_format: str = "markdown",
    include_stats: bool = True
) -> List[str]:
    """
    Prepare DataFrame chunks for LLM reasoning.
    
    Each chunk is structured data (Markdown or JSON), NOT raw text.
    LLM will receive clean, formatted data for analysis.
    
    Args:
        df: DataFrame to chunk
        chunk_size: Maximum rows per chunk (default 500)
        output_format: "markdown" or "json"
        include_stats: Include statistics in each chunk
        
    Returns:
        List of formatted strings ready for LLM
    """
    if df.empty:
        return []
    
    chunks = []
    total_rows = len(df)
    
    # Compute overall stats once
    stats = compute_statistics(df) if include_stats else None
    
    # Create header with schema information
    header = _create_schema_header(df, stats)
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk_df = df.iloc[start_idx:end_idx]
        
        if output_format == "json":
            chunk_content = _df_to_json_chunk(chunk_df, start_idx, end_idx, total_rows, header if start_idx == 0 else None)
        else:  # markdown
            chunk_content = _df_to_markdown_chunk(chunk_df, start_idx, end_idx, total_rows, header if start_idx == 0 else None)
        
        chunks.append(chunk_content)
    
    return chunks


def _create_schema_header(df: pd.DataFrame, stats: Optional[DataStatistics]) -> str:
    """Create schema description for LLM context."""
    lines = ["## Data Schema", ""]
    lines.append(f"**Total Rows:** {len(df):,}")
    lines.append(f"**Total Columns:** {len(df.columns)}")
    lines.append("")
    lines.append("### Columns:")
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isna().sum()
        lines.append(f"- **{col}** ({dtype}) - {null_count:,} nulls")
    
    if stats:
        # Add key statistics
        if stats.numeric_columns:
            lines.append("")
            lines.append("### Key Statistics:")
            for col_stat in stats.column_stats:
                if col_stat.name in stats.numeric_columns and col_stat.sum_val is not None:
                    lines.append(f"- **{col_stat.name}**: sum={col_stat.sum_val:,.2f}, avg={col_stat.mean_val:,.2f}, min={col_stat.min_val:,.2f}, max={col_stat.max_val:,.2f}")
        
        if stats.anomalies:
            lines.append("")
            lines.append("### Detected Anomalies:")
            for anom in stats.anomalies[:3]:
                lines.append(f"- **{anom['column']}**: {anom['count']} outliers ({anom['percentage']:.1f}%)")
        
        if stats.trends:
            lines.append("")
            lines.append("### Detected Trends:")
            for trend in stats.trends[:3]:
                lines.append(f"- **{trend['column']}**: {trend['direction']} ({trend['change_percent']:+.1f}%)")
    
    return "\n".join(lines)


def _df_to_markdown_chunk(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    total_rows: int,
    header: Optional[str] = None
) -> str:
    """Convert DataFrame chunk to Markdown table."""
    lines = []
    
    if header:
        lines.append(header)
        lines.append("")
    
    lines.append(f"### Data (rows {start_idx + 1:,} to {end_idx:,} of {total_rows:,})")
    lines.append("")
    
    # Build markdown table
    headers = list(df.columns)
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    for _, row in df.iterrows():
        values = [str(v) if pd.notna(v) else "NULL" for v in row]
        # Escape pipe characters
        values = [v.replace("|", "\\|") for v in values]
        lines.append("| " + " | ".join(values) + " |")
    
    return "\n".join(lines)


def _df_to_json_chunk(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    total_rows: int,
    header: Optional[str] = None
) -> str:
    """Convert DataFrame chunk to JSON format."""
    chunk_dict = {
        "chunk_info": {
            "start_row": start_idx + 1,
            "end_row": end_idx,
            "total_rows": total_rows
        },
        "columns": list(df.columns),
        "data": df.to_dict(orient='records')
    }
    
    if header:
        chunk_dict["schema"] = header
    
    return json.dumps(chunk_dict, default=str, indent=2)


# ============================================================================
# OUTPUT FORMATTING FOR LLM RESPONSES
# ============================================================================

def format_full_table(
    df: pd.DataFrame,
    max_rows: int = 1000,
    include_stats: bool = True
) -> str:
    """
    Format complete table for LLM output.
    Includes schema, stats, and full data (up to max_rows).
    """
    lines = []
    
    # Schema
    lines.append("## Complete Table Data")
    lines.append("")
    lines.append(f"**Total Rows:** {len(df):,}")
    lines.append(f"**Total Columns:** {len(df.columns)}")
    lines.append("")
    
    # Column info
    lines.append("### Column Details:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        lines.append(f"- **{col}** ({dtype})")
    
    lines.append("")
    
    if include_stats:
        stats = compute_statistics(df)
        
        # Numeric summaries
        if stats.numeric_columns:
            lines.append("### Numeric Column Statistics:")
            for col_stat in stats.column_stats:
                if col_stat.name in stats.numeric_columns and col_stat.sum_val is not None:
                    lines.append(f"- **{col_stat.name}**:")
                    lines.append(f"  - Sum: {col_stat.sum_val:,.2f}")
                    lines.append(f"  - Average: {col_stat.mean_val:,.2f}")
                    lines.append(f"  - Min: {col_stat.min_val:,.2f}")
                    lines.append(f"  - Max: {col_stat.max_val:,.2f}")
            lines.append("")
    
    # Full data table
    lines.append("### Data:")
    lines.append("")
    
    display_df = df.head(max_rows) if len(df) > max_rows else df
    
    headers = list(display_df.columns)
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    for _, row in display_df.iterrows():
        values = [str(v) if pd.notna(v) else "NULL" for v in row]
        values = [v.replace("|", "\\|")[:100] for v in values]  # Limit cell length
        lines.append("| " + " | ".join(values) + " |")
    
    if len(df) > max_rows:
        lines.append("")
        lines.append(f"*[Showing {max_rows:,} of {len(df):,} rows]*")
    
    return "\n".join(lines)


def format_sample_rows(df: pd.DataFrame, n: int = 10) -> str:
    """Format sample rows for LLM context."""
    if df.empty:
        return "No data available."
    
    sample = df.head(n)
    
    lines = []
    lines.append(f"### Sample Data ({n} of {len(df):,} rows)")
    lines.append("")
    
    headers = list(sample.columns)
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    for _, row in sample.iterrows():
        values = [str(v) if pd.notna(v) else "NULL" for v in row]
        values = [v.replace("|", "\\|")[:50] for v in values]
        lines.append("| " + " | ".join(values) + " |")
    
    return "\n".join(lines)


def format_statistics_summary(stats: DataStatistics) -> str:
    """Format statistics for LLM and display."""
    lines = []
    
    lines.append("## Data Statistics Summary")
    lines.append("")
    lines.append(f"- **Total Rows:** {stats.total_rows:,}")
    lines.append(f"- **Total Columns:** {stats.total_columns}")
    lines.append(f"- **Memory Usage:** {stats.memory_mb:.2f} MB")
    lines.append("")
    
    # Column breakdown
    lines.append("### Column Types:")
    lines.append(f"- Numeric: {len(stats.numeric_columns)} columns")
    lines.append(f"- Date/Time: {len(stats.date_columns)} columns")
    lines.append(f"- Categorical: {len(stats.categorical_columns)} columns")
    lines.append("")
    
    # Numeric stats
    if stats.numeric_columns:
        lines.append("### Numeric Column Statistics:")
        for col_stat in stats.column_stats:
            if col_stat.name in stats.numeric_columns and col_stat.sum_val is not None:
                lines.append(f"\n**{col_stat.name}**:")
                lines.append(f"- Sum: {col_stat.sum_val:,.2f}")
                lines.append(f"- Mean: {col_stat.mean_val:,.2f}")
                lines.append(f"- Min: {col_stat.min_val:,.2f}")
                lines.append(f"- Max: {col_stat.max_val:,.2f}")
                if col_stat.detected_unit:
                    lines.append(f"- Unit: {col_stat.detected_unit}")
    
    # NULL summary
    null_cols = {k: v for k, v in stats.null_summary.items() if v > 0}
    if null_cols:
        lines.append("")
        lines.append("### Missing Values:")
        for col, count in sorted(null_cols.items(), key=lambda x: -x[1]):
            pct = count / stats.total_rows * 100
            lines.append(f"- {col}: {count:,} ({pct:.1f}%)")
    
    # Anomalies
    if stats.anomalies:
        lines.append("")
        lines.append("### Anomalies Detected:")
        for anom in stats.anomalies:
            lines.append(f"- **{anom['column']}**: {anom['count']} outliers ({anom['percentage']:.1f}%)")
    
    # Trends
    if stats.trends:
        lines.append("")
        lines.append("### Trend Analysis:")
        for trend in stats.trends:
            lines.append(f"- **{trend['column']}**: {trend['direction']} ({trend['change_percent']:+.1f}% change)")
    
    return "\n".join(lines)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _clean_header(h: Any) -> str:
    """Clean and normalize a column header."""
    if h is None:
        return "Column"
    
    h = str(h).strip()
    
    # Remove common noise patterns
    h = re.sub(r'^Unnamed[:_\s]*\d*$', '', h, flags=re.IGNORECASE)
    h = re.sub(r'^\s*$', '', h)
    
    # Normalize whitespace
    h = re.sub(r'\s+', '_', h)
    h = re.sub(r'[^\w\-]', '', h)
    
    return h if h else "Column"


def _clean_cell_value(v: Any) -> Any:
    """Clean a cell value while preserving data."""
    if pd.isna(v) or v is None:
        return np.nan
    
    if isinstance(v, str):
        v = v.strip()
        if not v or v.lower() in ['nan', 'none', 'null', 'n/a', 'na', '-']:
            return np.nan
        return v
    
    return v


def _infer_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to infer proper data types for columns."""
    for col in df.columns:
        # Try numeric
        try:
            numeric = pd.to_numeric(df[col], errors='coerce')
            if numeric.notna().sum() > len(df) * 0.5:  # At least 50% valid
                df[col] = numeric
                continue
        except:
            pass
        
        # Try datetime
        try:
            dates = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            if dates.notna().sum() > len(df) * 0.5:
                df[col] = dates
                continue
        except:
            pass
    
    return df


def _looks_like_date_column(sample: pd.Series) -> bool:
    """Heuristic to detect date-like columns."""
    date_patterns = [
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
        r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2}',
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
    ]
    
    for val in sample.dropna().astype(str).head(10):
        for pattern in date_patterns:
            if re.search(pattern, val, re.IGNORECASE):
                return True
    return False


def _detect_unit(col_name: str, values: pd.Series) -> Optional[str]:
    """Attempt to detect unit from column name or values."""
    col_lower = col_name.lower()
    
    unit_patterns = {
        r'(price|cost|amount|revenue|salary|income|usd|eur|gbp|\$)': 'currency',
        r'(percent|pct|%)': 'percentage',
        r'(kg|kilogram|gram|weight|mass)': 'weight',
        r'(meter|metre|km|mile|distance|length|height|width)': 'distance',
        r'(hour|minute|second|day|week|month|year|duration|time)': 'time',
        r'(count|quantity|qty|number|num|total)': 'count',
        r'(rate|ratio)': 'rate',
        r'(temperature|temp|celsius|fahrenheit)': 'temperature',
    }
    
    for pattern, unit in unit_patterns.items():
        if re.search(pattern, col_lower):
            return unit
    
    return None


# ============================================================================
# HIGH-LEVEL EXTRACTION FUNCTION
# ============================================================================

def extract_all_tables(
    file_content: bytes,
    filename: str,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[List[ExtractedTable], DataStatistics]:
    """
    Universal table extraction from any supported file type.
    
    Returns all tables as DataFrames with full statistics.
    NO deduplication - all rows preserved.
    """
    ext = Path(filename).suffix.lower()
    
    if ext == '.pdf':
        tables = extract_tables_from_pdf(file_content, progress_callback)
    elif ext in ['.xlsx', '.xls']:
        tables = extract_tables_from_excel(file_content, progress_callback)
    elif ext == '.csv':
        tables = extract_tables_from_csv(file_content, progress_callback)
    else:
        return [], DataStatistics(0, 0, 0, [], [], [], [], {}, [], [])
    
    # Merge all tables for combined statistics
    if tables:
        merged_df = merge_tables(tables, merge_strategy="auto")
        stats = compute_statistics(merged_df)
    else:
        stats = DataStatistics(0, 0, 0, [], [], [], [], {}, [], [])
    
    return tables, stats
