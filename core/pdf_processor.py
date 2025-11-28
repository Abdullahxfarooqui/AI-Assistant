"""
Production-grade PDF processor for large files (15-50MB+).
Uses streaming, page-wise processing, and memory-efficient extraction.
"""
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import Dict, List, Any, Union, Generator, Optional, Callable
import io
import re
import gc
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


@dataclass
class ExtractedTable:
    """Represents an extracted table from PDF."""
    page_number: int
    table_index: int
    headers: List[str]
    rows: List[List[Any]]
    num_rows: int
    num_cols: int
    markdown: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "table_index": self.table_index,
            "headers": self.headers,
            "rows": self.rows[:100],  # Limit for JSON
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "markdown": self.markdown
        }


@dataclass
class ExtractedPage:
    """Represents an extracted page from PDF."""
    page_number: int
    text: str
    tables: List[ExtractedTable] = field(default_factory=list)
    has_tables: bool = False
    char_count: int = 0


# Noise patterns to filter out
NOISE_PATTERNS = [
    r'PUBLISHEDPublished',
    r'Published\s*Published',
    r'PUBLISHED\s*PUBLISHED',
    r'\bNaN\b',
    r'^[\s\-=_]{5,}$',
    r'^Page\s*\d+\s*(of\s*\d+)?$',
]

# Compiled patterns for speed
COMPILED_NOISE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]


def clean_text(text: str) -> str:
    """Clean extracted text, removing noise and normalizing."""
    if not text:
        return ""
    
    # Remove noise patterns
    for pattern in COMPILED_NOISE:
        text = pattern.sub('', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove lines that are just whitespace or very short
    lines = []
    for line in text.split('\n'):
        cleaned = line.strip()
        if cleaned and len(cleaned) > 2:
            lines.append(cleaned)
    
    return '\n'.join(lines)


def clean_cell(cell: Any) -> str:
    """Clean a table cell value."""
    if cell is None:
        return ""
    
    cell_str = str(cell).strip()
    
    # Check for noise patterns
    for pattern in COMPILED_NOISE:
        if pattern.match(cell_str):
            return ""
    
    # Normalize NULL-like values
    if cell_str.upper() in ['NULL', 'NONE', 'N/A', 'NA', 'NAN', '-', '']:
        return "NULL"
    
    return cell_str


def table_to_markdown(headers: List[str], rows: List[List[Any]], max_rows: int = 50) -> str:
    """Convert table to clean markdown format."""
    if not headers and not rows:
        return ""
    
    # Clean headers
    clean_headers = [clean_cell(h) or f"Col_{i+1}" for i, h in enumerate(headers)] if headers else [f"Col_{i+1}" for i in range(len(rows[0]))] if rows else []
    
    if not clean_headers:
        return ""
    
    parts = []
    
    # Header row
    parts.append("| " + " | ".join(clean_headers) + " |")
    parts.append("| " + " | ".join(["---"] * len(clean_headers)) + " |")
    
    # Data rows
    for row in rows[:max_rows]:
        clean_row = [clean_cell(cell) for cell in row]
        # Pad row if needed
        while len(clean_row) < len(clean_headers):
            clean_row.append("")
        clean_row = clean_row[:len(clean_headers)]
        parts.append("| " + " | ".join(clean_row) + " |")
    
    if len(rows) > max_rows:
        parts.append(f"\n*[Showing {max_rows} of {len(rows)} rows]*")
    
    return "\n".join(parts)


def extract_tables_from_page_pdfplumber(page, page_num: int) -> List[ExtractedTable]:
    """Extract tables from a pdfplumber page object."""
    tables = []
    
    try:
        extracted = page.extract_tables()
        
        for idx, table_data in enumerate(extracted):
            if not table_data or len(table_data) < 2:
                continue
            
            # First row as headers
            headers = [clean_cell(h) for h in table_data[0]]
            rows = [[clean_cell(cell) for cell in row] for row in table_data[1:]]
            
            # Skip tables with mostly empty cells
            non_empty_cells = sum(1 for row in rows for cell in row if cell and cell != "NULL")
            total_cells = sum(len(row) for row in rows)
            
            if total_cells > 0 and non_empty_cells / total_cells < 0.3:
                continue  # Skip mostly empty tables
            
            markdown = table_to_markdown(headers, rows)
            
            tables.append(ExtractedTable(
                page_number=page_num,
                table_index=idx,
                headers=headers,
                rows=rows,
                num_rows=len(rows),
                num_cols=len(headers),
                markdown=markdown
            ))
    except Exception as e:
        print(f"Table extraction error on page {page_num}: {e}")
    
    return tables


def extract_tables_from_page_pymupdf(page, page_num: int) -> List[ExtractedTable]:
    """Extract tables using PyMuPDF's find_tables."""
    tables = []
    
    try:
        tabs = page.find_tables()
        
        for idx, tab in enumerate(tabs):
            table_data = tab.extract()
            
            if not table_data or len(table_data) < 2:
                continue
            
            headers = [clean_cell(h) for h in table_data[0]]
            rows = [[clean_cell(cell) for cell in row] for row in table_data[1:]]
            
            # Skip mostly empty tables
            non_empty_cells = sum(1 for row in rows for cell in row if cell and cell != "NULL")
            total_cells = sum(len(row) for row in rows)
            
            if total_cells > 0 and non_empty_cells / total_cells < 0.3:
                continue
            
            markdown = table_to_markdown(headers, rows)
            
            tables.append(ExtractedTable(
                page_number=page_num,
                table_index=idx,
                headers=headers,
                rows=rows,
                num_rows=len(rows),
                num_cols=len(headers),
                markdown=markdown
            ))
    except Exception as e:
        print(f"PyMuPDF table extraction error on page {page_num}: {e}")
    
    return tables


def process_page_streaming(
    pdf_bytes: bytes,
    page_num: int,
    use_pdfplumber: bool = True
) -> ExtractedPage:
    """
    Process a single page - designed for parallel/streaming execution.
    Opens PDF fresh for each page to avoid memory issues with large files.
    """
    text = ""
    tables = []
    
    # Extract with PyMuPDF (faster for text)
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page_num < len(doc):
            page = doc[page_num]
            text = clean_text(page.get_text("text"))
            tables = extract_tables_from_page_pymupdf(page, page_num + 1)
        doc.close()
    except Exception as e:
        print(f"PyMuPDF error on page {page_num}: {e}")
    
    # Try pdfplumber for better table extraction if PyMuPDF found none
    if use_pdfplumber and len(tables) == 0:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                if page_num < len(pdf.pages):
                    plumber_page = pdf.pages[page_num]
                    plumber_tables = extract_tables_from_page_pdfplumber(plumber_page, page_num + 1)
                    if plumber_tables:
                        tables = plumber_tables
        except Exception as e:
            print(f"pdfplumber error on page {page_num}: {e}")
    
    return ExtractedPage(
        page_number=page_num + 1,
        text=text,
        tables=tables,
        has_tables=len(tables) > 0,
        char_count=len(text)
    )


def stream_pdf_pages(
    file_content: bytes,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_workers: int = 4
) -> Generator[ExtractedPage, None, None]:
    """
    Stream PDF pages using parallel processing.
    Memory-efficient: processes pages in batches.
    
    Args:
        file_content: PDF file bytes
        progress_callback: Optional callback(current_page, total_pages)
        max_workers: Number of parallel workers
        
    Yields:
        ExtractedPage objects one at a time
    """
    # Get page count first
    doc = fitz.open(stream=file_content, filetype="pdf")
    total_pages = len(doc)
    doc.close()
    
    # Process in batches to control memory
    batch_size = max_workers * 2
    
    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        batch_pages = list(range(batch_start, batch_end))
        
        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_page_streaming, file_content, page_num): page_num
                for page_num in batch_pages
            }
            
            # Collect results in order
            results = {}
            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    results[page_num] = future.result()
                except Exception as e:
                    print(f"Error processing page {page_num}: {e}")
                    results[page_num] = ExtractedPage(
                        page_number=page_num + 1,
                        text="",
                        tables=[],
                        has_tables=False,
                        char_count=0
                    )
            
            # Yield in order
            for page_num in batch_pages:
                if progress_callback:
                    progress_callback(page_num + 1, total_pages)
                yield results[page_num]
        
        # Force garbage collection between batches
        gc.collect()


def extract_pdf_complete(
    file_content: bytes,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """
    Complete PDF extraction with streaming and parallel processing.
    Handles large files (15-50MB+) efficiently.
    
    Returns:
        Dict with all extracted content, tables, and metadata
    """
    pages = []
    all_tables = []
    full_text_parts = []
    
    for page in stream_pdf_pages(file_content, progress_callback):
        pages.append({
            "page_number": page.page_number,
            "text": page.text,
            "has_tables": page.has_tables,
            "table_count": len(page.tables),
            "char_count": page.char_count
        })
        
        full_text_parts.append(page.text)
        
        for table in page.tables:
            all_tables.append(table.to_dict())
    
    # Create combined text
    full_text = "\n\n".join(full_text_parts)
    
    # Detect if this is primarily tabular data
    total_table_rows = sum(t.get("num_rows", 0) for t in all_tables)
    is_dataset = total_table_rows > 10 or len(all_tables) > 2
    
    return {
        "text": full_text,
        "pages": pages,
        "tables": all_tables,
        "type": "pdf",
        "num_pages": len(pages),
        "num_tables": len(all_tables),
        "total_table_rows": total_table_rows,
        "is_dataset": is_dataset,
        "total_chars": sum(p.get("char_count", 0) for p in pages)
    }


def get_page_count(file_content: bytes) -> int:
    """Get total page count without loading entire PDF."""
    doc = fitz.open(stream=file_content, filetype="pdf")
    count = len(doc)
    doc.close()
    return count
