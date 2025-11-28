"""
Production RAG Engine with Python-Based Data Analysis.

This engine:
1. Uses Python/Pandas for ALL table extraction and statistics (not LLM)
2. LLM is ONLY for reasoning, insights, interpretations, comparisons, and trends
3. Preserves ALL data rows - no deduplication of valid data
4. Handles large files (15-50MB+) efficiently
5. Returns structured Markdown tables, not raw text

Key Functions:
- query(): Unified query with auto intent detection
- analyze_query(): Analyze specific queries with context
- answer_question(): Answer questions about data
- summarize_document(): Generate comprehensive summary
- compare_documents(): Compare multiple documents
- get_document_info(): Full document metadata and structure
- list_all_documents(): List all indexed documents
"""
import requests
from typing import List, Dict, Any, Optional, Union
import re
import json
import pandas as pd
from pathlib import Path

from core.embedder import embed_query
from core.vector_store import get_vector_store

# Try to import data_engine functions
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
        format_statistics_summary,
        ExtractedTable,
        DataStatistics
    )
    HAS_DATA_ENGINE = True
except ImportError:
    HAS_DATA_ENGINE = False

# OpenRouter API configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-4o-mini"  # Fast model for quick responses

# Performance settings
MAX_CONTEXT_CHARS = 10000  # Increased for detailed analysis
DEFAULT_MAX_TOKENS = 4000  # Increased for comprehensive responses


# ============================================================================
# LLM SYSTEM PROMPTS - Focused on reasoning, NOT data extraction
# ============================================================================

SYSTEM_PROMPT_ANALYSIS = """You are an expert data analyst AI. Your role is to analyze STRUCTURED DATA and provide CLEAR, FORMATTED INSIGHTS.

CRITICAL FORMATTING RULES:
1. NEVER output tables with more than 8 columns - select the most important ones
2. NEVER output more than 15 rows in any table
3. NEVER include _UOM columns in data tables (mention units in column headers instead)
4. ALWAYS use proper Markdown table formatting
5. ALWAYS provide statistics as formatted lists, not raw dumps

DATA PRESENTATION GUIDELINES:
- For production data: Show ITEM_NAME, START_DATETIME, PROD_OIL_VOL, PROD_GAS_VOL, PROD_WAT_VOL as key columns
- Always include units in parentheses: "PROD_OIL_VOL (bbl)" instead of separate UOM column
- Round large numbers to 2 decimal places
- Use comma separators for thousands

RESPONSE STRUCTURE:
## Overview
(1-2 sentence summary)

## Key Findings
(Bullet points with specific values)

## Data Preview
(Clean table with 6-8 columns, max 10-15 rows)

## Statistics Summary
(Formatted statistics)

YOU MUST NOT:
- Dump raw data without formatting
- Show all 170 columns
- Output empty cells or NaN values without explanation
- Use technical jargon without explanation"""

SYSTEM_PROMPT_INSIGHT = """You are a data insights expert. Analyze the provided statistics and data to generate meaningful insights.

The data has been pre-processed with:
- Exact row counts and column definitions
- Computed statistics (sum, mean, min, max, std)
- Detected anomalies (outliers) with counts
- Detected trends (increasing/decreasing) with percentages

Generate insights about:
1. Key findings from the statistics
2. What anomalies suggest (data quality or real issues)
3. What trends indicate about the data
4. Recommendations based on the data
5. Questions that should be investigated further

Be specific with numbers. Reference actual values from the statistics."""

SYSTEM_PROMPT_COMPARE = """You are a comparative data analyst. Compare the provided datasets and identify:

1. Similarities in structure and content
2. Differences in values, ranges, and patterns
3. Which dataset is larger/more complete
4. Common columns and divergent columns
5. Trends that appear in one but not the other
6. Recommendations for consolidating or using together

Use specific values from both datasets in your comparison."""


# ============================================================================
# LLM API CALL
# ============================================================================

def call_llm(
    messages: List[Dict[str, str]],
    max_tokens: int = 2000,
    temperature: float = 0.1,
    timeout: int = 60
) -> str:
    """Call OpenRouter LLM API with error handling - optimized for speed."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "RAG Engine"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": min(max_tokens, DEFAULT_MAX_TOKENS),  # Cap tokens for speed
        "temperature": temperature
    }
    
    try:
        response = requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The document may be too large. Try a more specific query."
    except requests.exceptions.RequestException as e:
        return f"Error communicating with LLM: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# INTENT DETECTION
# ============================================================================

def detect_intent(query: str) -> str:
    """
    Detect query intent for optimized response strategy.
    
    Intents:
    - calculate: Totals, sums, averages (PRIORITY - fast Python computation)
    - explain: What is this, describe, overview
    - summarize: Brief summary
    - show: Display data, tables
    - search: Find specific records
    - compare: Compare documents or values
    - trend: Trend analysis
    - anomaly: Outlier detection
    - general: Default
    """
    q = query.lower()
    
    # PRIORITY: Check for calculations FIRST (these use fast Python, not LLM)
    if any(w in q for w in ['total', 'sum', 'average', 'avg', 'count', 'how many', 'how much', 
                             'calculate', 'mean', 'median', 'minimum', 'maximum', 'min', 'max',
                             'production of', 'sales of', 'volume of']):
        return "calculate"
    
    # Other intents (use LLM)
    if any(w in q for w in ['what is', 'what does', 'tell me about', 'describe', 'explain', 'overview', 'what are the columns']):
        return "explain"
    if any(w in q for w in ['summarize', 'summary', 'brief', 'quick overview', 'key points']):
        return "summarize"
    if any(w in q for w in ['show', 'list', 'display', 'all data', 'all rows', 'table', 'full data']):
        return "show"
    if any(w in q for w in ['find', 'search', 'where', 'filter', 'which', 'records with', 'entries where']):
        return "search"
    if any(w in q for w in ['compare', 'difference', 'vs', 'versus', 'between']):
        return "compare"
    if any(w in q for w in ['trend', 'over time', 'change', 'growth', 'decline', 'pattern']):
        return "trend"
    if any(w in q for w in ['anomaly', 'anomalies', 'outlier', 'outliers', 'unusual', 'abnormal']):
        return "anomaly"
    
    return "general"


# ============================================================================
# DATA RETRIEVAL FROM VECTOR STORE
# ============================================================================

def get_document_data(doc_hash: str) -> Dict[str, Any]:
    """
    Retrieve complete document data from vector store.
    Returns structured data including DataFrames, statistics, and metadata.
    """
    store = get_vector_store()
    
    chunks = store.get_document_chunks(doc_hash)
    if not chunks:
        return {"error": "Document not found"}
    
    # Reconstruct tables from chunks
    tables = store.get_document_tables(doc_hash)
    full_text = store.get_full_document_text(doc_hash)
    
    # Get document metadata
    docs = store.get_all_documents()
    doc_info = next((d for d in docs if d.get("doc_hash") == doc_hash), {})
    
    return {
        "doc_hash": doc_hash,
        "filename": doc_info.get("filename", "unknown"),
        "chunks": chunks,
        "tables": tables,
        "full_text": full_text,
        "num_chunks": len(chunks),
        "is_dataset": doc_info.get("is_dataset", False)
    }


def build_structured_context(
    results: List[Dict[str, Any]],
    doc_hash: Optional[str] = None,
    max_chars: int = 8000,
    include_stats: bool = True
) -> str:
    """
    Build structured context from search results.
    Formats data as clean Markdown tables with statistics.
    """
    if not results:
        return ""
    
    store = get_vector_store()
    
    # If specific document, get comprehensive data
    if doc_hash:
        full_text = store.get_full_document_text(doc_hash)
        tables = store.get_document_tables(doc_hash)
        
        if tables:
            # Prioritize table content
            context_parts = []
            for t in tables[:5]:  # Limit to 5 tables
                context_parts.append(t.get("text", ""))
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = full_text
        
        return context[:max_chars]
    
    # Combine search results
    texts = []
    seen = set()
    
    for r in results:
        text = r.get("text", "")
        text_key = text[:200]  # Use first 200 chars as key
        
        if text_key in seen:
            continue
        seen.add(text_key)
        
        source = r.get("filename", "unknown")
        is_table = r.get("type") == "table"
        
        if is_table:
            texts.append(f"## Table from {source}\n\n{text}")
        else:
            texts.append(f"## Content from {source}\n\n{text}")
    
    combined = "\n\n---\n\n".join(texts)
    return combined[:max_chars]


# ============================================================================
# PRE-COMPUTED STATISTICS (Python, not LLM)
# ============================================================================

def compute_stats_for_query(df: pd.DataFrame, query: str) -> str:
    """
    Compute statistics from DataFrame based on query intent.
    Returns formatted string with actual computed values.
    """
    if df is None or df.empty:
        return ""
    
    stats_lines = ["## PRE-COMPUTED STATISTICS (from Python):\n"]
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Basic counts
    stats_lines.append(f"**Total Rows:** {len(df):,}")
    stats_lines.append(f"**Total Columns:** {len(df.columns)}")
    
    # Check for key production columns
    prod_cols = [c for c in numeric_cols if any(x in c.upper() for x in ['PROD', 'SALES', 'VOL', 'OIL', 'GAS', 'WATER', 'WAT', 'ENERGY', 'INJECTION', 'INJ'])]
    
    if prod_cols:
        stats_lines.append("\n### Column Totals (SUM):")
        for col in prod_cols[:15]:  # Limit to 15 columns
            total = df[col].sum()
            avg = df[col].mean()
            min_val = df[col].min()
            max_val = df[col].max()
            # Get unit if available
            uom_col = col + "_UOM"
            unit = ""
            if uom_col in df.columns:
                uom_vals = df[uom_col].dropna()
                if len(uom_vals) > 0:
                    unit = f" ({uom_vals.iloc[0]})"
            stats_lines.append(f"- **{col}**{unit}: Total={total:,.2f}, Avg={avg:,.2f}, Min={min_val:,.2f}, Max={max_val:,.2f}")
    
    # Date range
    date_cols = [c for c in df.columns if any(x in c.upper() for x in ['DATE', 'TIME', 'DATETIME'])]
    for col in date_cols[:2]:
        try:
            dates = pd.to_datetime(df[col], errors='coerce').dropna()
            if len(dates) > 0:
                stats_lines.append(f"\n**{col} Range:** {dates.min()} to {dates.max()}")
        except:
            pass
    
    # Item counts if categorical
    cat_cols = [c for c in df.columns if 'ITEM' in c.upper() or 'NAME' in c.upper() or 'TYPE' in c.upper()]
    for col in cat_cols[:3]:
        unique_count = df[col].nunique()
        stats_lines.append(f"**Unique {col}:** {unique_count}")
    
    return "\n".join(stats_lines)


# Global DataFrame cache reference (set by app.py)
_dataframe_cache: Dict[str, pd.DataFrame] = {}

def set_dataframe_cache(cache: Dict[str, pd.DataFrame]):
    """Set the global DataFrame cache from app.py."""
    global _dataframe_cache
    _dataframe_cache = cache

def get_dataframe_for_doc(doc_hash: str, filename: str = None) -> Optional[pd.DataFrame]:
    """Get DataFrame from cache by doc_hash or filename."""
    global _dataframe_cache
    
    # Try by filename first
    if filename and filename in _dataframe_cache:
        return _dataframe_cache[filename]
    
    # Try to find by partial match
    for key, df in _dataframe_cache.items():
        if filename and filename in key:
            return df
    
    return None


# ============================================================================
# MAIN QUERY FUNCTION
# ============================================================================

def query(
    user_query: str,
    doc_hash: Optional[str] = None,
    k: int = 10,
    dataframe: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Main unified query function with intelligent intent detection.
    
    This function:
    1. Detects query intent (explain, calculate, show, search, etc.)
    2. For calculations with DataFrame: Uses Python directly (FAST, no LLM)
    3. For other queries: Retrieves context and uses LLM
    
    Args:
        user_query: User's natural language query
        doc_hash: Optional specific document to query
        k: Number of chunks to retrieve
        dataframe: Optional DataFrame for direct statistics computation
        
    Returns:
        Dict with answer, sources, intent, and metadata
    """
    # Detect intent FIRST
    intent = detect_intent(user_query)
    
    # FAST PATH: If calculate intent and we have DataFrame, skip vector search entirely
    if intent == "calculate" and dataframe is not None and not dataframe.empty:
        answer = _handle_calculate_intent(user_query, "", doc_hash, dataframe)
        return {
            "answer": answer,
            "sources": [],
            "intent": intent,
            "num_chunks": 0
        }
    
    # NORMAL PATH: Vector search for other intents
    store = get_vector_store()
    
    # Get relevant context
    query_embedding = embed_query(user_query)
    
    results = store.search_with_context(
        query_embedding,
        k=k,
        context_window=2,
        doc_hash=doc_hash
    )
    
    # Get filename from results
    filename = None
    if results:
        filename = results[0].get("filename")
    
    # Try to get DataFrame for pre-computed stats
    df = dataframe
    if df is None and filename:
        df = get_dataframe_for_doc(doc_hash, filename)
    
    # If specific document and no results, get full doc
    if doc_hash and not results:
        full_text = store.get_full_document_text(doc_hash)
        if full_text:
            results = [{
                "text": full_text,
                "filename": "document",
                "doc_hash": doc_hash,
                "score": 1.0
            }]
    
    if not results:
        return {
            "answer": "No relevant information found. Please upload a document first or try a different query.",
            "sources": [],
            "intent": "no_data"
        }
    
    # Detect intent
    intent = detect_intent(user_query)
    
    # Build context with pre-computed stats
    context = build_structured_context(results, doc_hash)
    
    # Add pre-computed statistics from Python (not LLM)
    if df is not None:
        stats = compute_stats_for_query(df, user_query)
        if stats:
            context = stats + "\n\n---\n\n" + context[:MAX_CONTEXT_CHARS - len(stats) - 100]
    
    # Handle different intents with appropriate strategies
    if intent == "show":
        answer = _handle_show_intent(user_query, context, doc_hash)
    elif intent == "calculate":
        answer = _handle_calculate_intent(user_query, context, doc_hash, df)
    elif intent == "explain":
        answer = _handle_explain_intent(user_query, context, df)
    elif intent == "summarize":
        answer = _handle_summarize_intent(user_query, context)
    elif intent == "search":
        answer = _handle_search_intent(user_query, context)
    elif intent == "trend":
        answer = _handle_trend_intent(user_query, context, doc_hash)
    elif intent == "anomaly":
        answer = _handle_anomaly_intent(user_query, context, doc_hash)
    else:
        answer = _handle_general_intent(user_query, context)
    
    # Format sources for return
    sources = []
    for r in results[:5]:
        sources.append({
            "filename": r.get("filename", "unknown"),
            "doc_hash": r.get("doc_hash", ""),
            "score": r.get("score", 0),
            "preview": r.get("text", "")[:300] + "..." if len(r.get("text", "")) > 300 else r.get("text", ""),
            "type": r.get("type", "text")
        })
    
    return {
        "answer": answer,
        "sources": sources,
        "intent": intent,
        "num_chunks": len(results)
    }


# ============================================================================
# INTENT HANDLERS
# ============================================================================

def _handle_show_intent(query: str, context: str, doc_hash: Optional[str]) -> str:
    """Handle 'show data' type queries - return formatted data summary."""
    # Truncate context for speed
    context = context[:MAX_CONTEXT_CHARS]
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_ANALYSIS},
        {"role": "user", "content": f"""Show data overview.

QUERY: {query}

DATA:
{context}

Provide:
1. Brief overview (2 sentences)
2. Sample table (5-6 key columns, 8 rows max)
3. Key stats (totals, averages)"""}
    ]
    
    return call_llm(messages, max_tokens=2000)


def _handle_calculate_intent(query: str, context: str, doc_hash: Optional[str], df: Optional[pd.DataFrame] = None) -> str:
    """Handle calculation queries - Python computes, LLM explains."""
    
    # If we have the DataFrame, compute the answer directly with Python
    if df is not None and not df.empty:
        result_lines = ["## Calculation Results (computed by Python):\n"]
        q_lower = query.lower()
        
        # Convert object columns that might be numeric strings
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Smart column matching - prioritize specific matches
        requested_cols = []
        
        # Check for specific resource mentions
        wants_oil = 'oil' in q_lower
        wants_gas = 'gas' in q_lower
        wants_water = 'water' in q_lower or 'wat' in q_lower
        wants_sales = 'sales' in q_lower or 'sold' in q_lower
        wants_production = 'production' in q_lower or 'produced' in q_lower or 'prod' in q_lower
        
        for col in numeric_cols:
            col_upper = col.upper()
            
            # Skip rate/density columns unless specifically asked
            if any(x in col_upper for x in ['RATE', 'DENSITY', 'DEC_', 'INV', '_R_MIN']) and 'rate' not in q_lower:
                continue
            
            # Match based on query specifics
            if wants_oil and wants_sales and 'SALES' in col_upper and 'OIL' in col_upper:
                requested_cols.append(col)
            elif wants_oil and wants_production and 'PROD' in col_upper and 'OIL' in col_upper and 'VOL' in col_upper:
                requested_cols.append(col)
            elif wants_oil and 'OIL' in col_upper and 'VOL' in col_upper and not wants_sales:
                # Default oil query = production volume
                if 'PROD' in col_upper:
                    requested_cols.append(col)
            elif wants_gas and 'GAS' in col_upper and 'VOL' in col_upper:
                if wants_sales and 'SALES' in col_upper:
                    requested_cols.append(col)
                elif 'PROD' in col_upper:
                    requested_cols.append(col)
            elif wants_water and ('WAT' in col_upper or 'WATER' in col_upper) and 'VOL' in col_upper:
                requested_cols.append(col)
        
        # If no specific match, try broader matching
        if not requested_cols:
            for col in numeric_cols:
                col_upper = col.upper()
                # Skip non-volume columns
                if any(x in col_upper for x in ['RATE', 'DENSITY', 'DEC_', 'INV', '_R_MIN']):
                    continue
                if wants_oil and 'OIL' in col_upper:
                    requested_cols.append(col)
                elif wants_gas and 'GAS' in col_upper:
                    requested_cols.append(col)
                elif wants_water and 'WAT' in col_upper:
                    requested_cols.append(col)
        
        # Still nothing? Use main production volumes
        if not requested_cols:
            main_cols = ['PROD_OIL_VOL', 'PROD_GAS_VOL', 'PROD_WAT_VOL', 'SALES_OIL_VOL', 'SALES_GAS_VOL']
            requested_cols = [c for c in main_cols if c in numeric_cols]
        
        # Remove duplicates
        requested_cols = list(dict.fromkeys(requested_cols))
        
        # Compute totals
        if any(w in q_lower for w in ['total', 'sum', 'all', 'production', 'how much']):
            result_lines.append("### TOTALS:")
            for col in requested_cols[:5]:  # Limit to 5 most relevant
                # Count non-null values
                non_null_count = df[col].notna().sum()
                null_count = df[col].isna().sum()
                
                # Sum only non-null values
                total = df[col].dropna().sum()
                
                uom = ""
                if col + "_UOM" in df.columns:
                    uom_vals = df[col + "_UOM"].dropna()
                    uom = f" {uom_vals.iloc[0]}" if len(uom_vals) > 0 else ""
                
                if total > 0:
                    result_lines.append(f"- **{col}**: {total:,.2f}{uom} ({non_null_count:,} records)")
                elif non_null_count == 0:
                    result_lines.append(f"- **{col}**: No data (all {null_count:,} values are NULL)")
                else:
                    result_lines.append(f"- **{col}**: {total:,.2f}{uom} ({non_null_count:,} non-null records)")
            
            result_lines.append(f"\n**Data Summary:** {len(df):,} total records")
        
        # Compute averages
        if any(w in q_lower for w in ['average', 'avg', 'mean']):
            result_lines.append("\n### AVERAGES:")
            for col in requested_cols:
                avg = df[col].mean()
                result_lines.append(f"- **{col}**: {avg:,.2f}")
        
        # Compute counts
        if any(w in q_lower for w in ['count', 'how many', 'number of']):
            result_lines.append(f"\n### COUNTS:")
            result_lines.append(f"- **Total Rows**: {len(df):,}")
            # Count by category if mentioned
            cat_cols = [c for c in df.columns if 'ITEM' in c.upper() or 'TYPE' in c.upper() or 'NAME' in c.upper()]
            for col in cat_cols[:3]:
                result_lines.append(f"- **Unique {col}**: {df[col].nunique():,}")
        
        # Compute min/max
        if any(w in q_lower for w in ['minimum', 'min', 'maximum', 'max', 'range']):
            result_lines.append("\n### MIN/MAX:")
            for col in requested_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                result_lines.append(f"- **{col}**: Min={min_val:,.2f}, Max={max_val:,.2f}")
        
        # If just asking for general stats, show all
        if len(result_lines) == 1:  # Only header added
            result_lines.append("### SUMMARY STATISTICS:")
            for col in requested_cols[:8]:
                total = df[col].sum()
                avg = df[col].mean()
                min_val = df[col].min()
                max_val = df[col].max()
                uom = ""
                if col + "_UOM" in df.columns:
                    uom_vals = df[col + "_UOM"].dropna()
                    uom = f" ({uom_vals.iloc[0]})" if len(uom_vals) > 0 else ""
                result_lines.append(f"\n**{col}**{uom}:")
                result_lines.append(f"  - Total: {total:,.2f}")
                result_lines.append(f"  - Average: {avg:,.2f}")
                result_lines.append(f"  - Min: {min_val:,.2f}")
                result_lines.append(f"  - Max: {max_val:,.2f}")
        
        return "\n".join(result_lines)
    
    # Fallback to LLM if no DataFrame
    context = context[:MAX_CONTEXT_CHARS]
    
    messages = [
        {"role": "system", "content": "You are a data analyst. Use the PRE-COMPUTED STATISTICS provided to answer. These values are already calculated by Python."},
        {"role": "user", "content": f"""Calculate: {query}

{context}

IMPORTANT: Use the exact values from PRE-COMPUTED STATISTICS above. Do not estimate or guess."""}
    ]
    
    return call_llm(messages, max_tokens=1500)


def _handle_explain_intent(query: str, context: str, df: Optional[pd.DataFrame] = None) -> str:
    """Handle 'explain/describe' queries - explain what the data contains."""
    
    # Build comprehensive stats from DataFrame
    stats_for_prompt = ""
    if df is not None and not df.empty:
        stats_lines = []
        stats_lines.append(f"DATASET SIZE: {len(df):,} records, {len(df.columns)} columns")
        
        # All columns by category
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        date_cols = [c for c in df.columns if 'DATE' in c.upper() or 'TIME' in c.upper()]
        
        # Production columns with full stats
        prod_cols = [c for c in numeric_cols if 'PROD' in c.upper() and 'VOL' in c.upper()]
        sales_cols = [c for c in numeric_cols if 'SALES' in c.upper() and 'VOL' in c.upper()]
        rate_cols = [c for c in numeric_cols if 'RATE' in c.upper()]
        energy_cols = [c for c in numeric_cols if 'ENERGY' in c.upper() or 'BTU' in c.upper()]
        
        if prod_cols:
            stats_lines.append("\n=== PRODUCTION VOLUMES ===")
            for col in prod_cols:
                total = df[col].dropna().sum()
                count = df[col].notna().sum()
                null_count = df[col].isna().sum()
                avg = df[col].dropna().mean() if count > 0 else 0
                min_val = df[col].dropna().min() if count > 0 else 0
                max_val = df[col].dropna().max() if count > 0 else 0
                std_val = df[col].dropna().std() if count > 0 else 0
                uom = ""
                if col + "_UOM" in df.columns:
                    uom_vals = df[col + "_UOM"].dropna()
                    uom = f" {uom_vals.iloc[0]}" if len(uom_vals) > 0 else ""
                stats_lines.append(f"\n{col}:")
                stats_lines.append(f"  Total: {total:,.2f}{uom}")
                stats_lines.append(f"  Average: {avg:,.2f}{uom}")
                stats_lines.append(f"  Min: {min_val:,.2f}{uom}, Max: {max_val:,.2f}{uom}")
                stats_lines.append(f"  Std Dev: {std_val:,.2f}")
                stats_lines.append(f"  Records with data: {count:,} ({100*count/len(df):.1f}%)")
                stats_lines.append(f"  NULL/Empty records: {null_count:,} ({100*null_count/len(df):.1f}%)")
        
        if sales_cols:
            stats_lines.append("\n=== SALES VOLUMES ===")
            for col in sales_cols:
                total = df[col].dropna().sum()
                count = df[col].notna().sum()
                null_count = df[col].isna().sum()
                avg = df[col].dropna().mean() if count > 0 else 0
                min_val = df[col].dropna().min() if count > 0 else 0
                max_val = df[col].dropna().max() if count > 0 else 0
                uom = ""
                if col + "_UOM" in df.columns:
                    uom_vals = df[col + "_UOM"].dropna()
                    uom = f" {uom_vals.iloc[0]}" if len(uom_vals) > 0 else ""
                stats_lines.append(f"\n{col}:")
                stats_lines.append(f"  Total: {total:,.2f}{uom}")
                stats_lines.append(f"  Average: {avg:,.2f}{uom}")
                stats_lines.append(f"  Min: {min_val:,.2f}{uom}, Max: {max_val:,.2f}{uom}")
                stats_lines.append(f"  Records with data: {count:,} ({100*count/len(df):.1f}%)")
        
        if rate_cols:
            stats_lines.append("\n=== FLOW RATES ===")
            for col in rate_cols[:4]:
                avg = df[col].dropna().mean()
                count = df[col].notna().sum()
                if count > 0 and avg > 0:
                    stats_lines.append(f"  {col}: Avg={avg:,.2f}, Records={count:,}")
        
        if energy_cols:
            stats_lines.append("\n=== ENERGY METRICS ===")
            for col in energy_cols[:4]:
                total = df[col].dropna().sum()
                count = df[col].notna().sum()
                if count > 0 and total > 0:
                    stats_lines.append(f"  {col}: Total={total:,.2f}, Records={count:,}")
        
        # Date range
        for col in date_cols[:1]:
            try:
                dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(dates) > 0:
                    stats_lines.append(f"\n=== TIME PERIOD ===")
                    stats_lines.append(f"  Start Date: {dates.min().strftime('%Y-%m-%d')}")
                    stats_lines.append(f"  End Date: {dates.max().strftime('%Y-%m-%d')}")
                    days = (dates.max() - dates.min()).days
                    stats_lines.append(f"  Duration: {days} days ({days//30} months)")
                    stats_lines.append(f"  Total date records: {len(dates):,}")
            except:
                pass
        
        # Categorical breakdown
        cat_cols = [c for c in df.columns if any(x in c.upper() for x in ['ITEM', 'NAME', 'TYPE', 'FIELD', 'STATUS', 'COMPLETION', 'WELL'])]
        if cat_cols:
            stats_lines.append("\n=== CATEGORICAL BREAKDOWN ===")
            for col in cat_cols[:8]:
                unique = df[col].nunique()
                top_vals = df[col].value_counts().head(5)
                stats_lines.append(f"\n{col}: {unique} unique values")
                for val, cnt in top_vals.items():
                    stats_lines.append(f"  - '{val}': {cnt:,} records ({100*cnt/len(df):.1f}%)")
        
        # Column list by type
        stats_lines.append(f"\n=== ALL COLUMNS ({len(df.columns)} total) ===")
        stats_lines.append(f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:15])}{'...' if len(numeric_cols) > 15 else ''}")
        stats_lines.append(f"Text columns ({len(text_cols)}): {', '.join(text_cols[:10])}{'...' if len(text_cols) > 10 else ''}")
        
        stats_for_prompt = "\n".join(stats_lines)
    
    context = context[:MAX_CONTEXT_CHARS]
    
    messages = [
        {"role": "system", "content": """You are an expert data analyst providing comprehensive document analysis. 

CREATE A DETAILED, PROFESSIONAL REPORT with these sections:

## 📋 Executive Summary
(3-4 sentences: what is this data, who would use it, what time period it covers, key purpose)

## 📊 Dataset Overview
| Metric | Value |
|--------|-------|
| Total Records | X |
| Total Columns | X |
| Date Range | X to Y |
| Data Granularity | daily/monthly |

## 🛢️ Production Analysis
(For EACH production metric - oil, gas, water, condensate - provide:)
- Total volume with units
- Average per record
- Min/Max range
- Data coverage (% of records with values)
- What this metric represents

## 💰 Sales Analysis
(Similar detailed breakdown of sales volumes)

## 📈 Key Performance Indicators
- Production vs Sales comparison (% sold)
- Water-to-Oil ratio
- Gas-to-Oil ratio
- Any efficiency metrics

## 🏭 Asset/Well Breakdown
(List all unique items/wells/fields with their record counts)

## 📝 Complete Column Dictionary
(Group columns by category and explain what each measures)

## 🔍 Data Quality Assessment
- Columns with high NULL rates
- Data completeness by category
- Any anomalies or concerns

## 💡 Key Insights & Recommendations
(5-7 actionable insights based on the data)

Use the ACTUAL STATISTICS provided - format numbers with commas and appropriate units."""},
        {"role": "user", "content": f"""Provide a comprehensive, detailed analysis of this document.

ACTUAL STATISTICS FROM THE DATA:
{stats_for_prompt}

SAMPLE DATA:
{context}

Create a thorough professional report using ALL the statistics above. Be specific with numbers."""}
    ]
    
    return call_llm(messages, max_tokens=4000)


def _handle_summarize_intent(query: str, context: str) -> str:
    """Handle summarization queries - concise overview with key stats."""
    context = context[:MAX_CONTEXT_CHARS]
    
    messages = [
        {"role": "system", "content": "Summarize data concisely."},
        {"role": "user", "content": f"""Summarize: {query}

DATA:
{context}

Provide: Overview, Key stats, 3 key findings."""}
    ]
    
    return call_llm(messages, max_tokens=1500)


def _handle_search_intent(query: str, context: str) -> str:
    """Handle search/filter queries - find specific records."""
    context = context[:MAX_CONTEXT_CHARS]
    
    messages = [
        {"role": "system", "content": "Find and show matching records."},
        {"role": "user", "content": f"""Find: {query}

DATA:
{context}

Show matching records as a table (max 10 rows)."""}
    ]
    
    return call_llm(messages, max_tokens=1500)


def _handle_trend_intent(query: str, context: str, doc_hash: Optional[str]) -> str:
    """Handle trend analysis queries."""
    context = context[:MAX_CONTEXT_CHARS]
    
    messages = [
        {"role": "system", "content": "Analyze trends concisely."},
        {"role": "user", "content": f"""Trend analysis: {query}

DATA:
{context}

Identify patterns and changes with specific values."""}
    ]
    
    return call_llm(messages, max_tokens=1500)


def _handle_anomaly_intent(query: str, context: str, doc_hash: Optional[str]) -> str:
    """Handle anomaly detection queries."""
    context = context[:MAX_CONTEXT_CHARS]
    
    messages = [
        {"role": "system", "content": "Find anomalies and outliers."},
        {"role": "user", "content": f"""Find anomalies: {query}

DATA:
{context}

List unusual values with explanations."""}
    ]
    
    return call_llm(messages, max_tokens=1500)


def _handle_general_intent(query: str, context: str) -> str:
    """Handle general queries with flexible response."""
    context = context[:MAX_CONTEXT_CHARS]
    
    messages = [
        {"role": "system", "content": "Answer data questions concisely with specific values."},
        {"role": "user", "content": f"""Question: {query}

DATA:
{context}

Answer directly with relevant data."""}
    ]
    
    return call_llm(messages, max_tokens=1500)


# ============================================================================
# DOCUMENT ANALYSIS FUNCTIONS
# ============================================================================

def summarize_document(doc_hash: str) -> Dict[str, Any]:
    """
    Generate comprehensive document summary with Python-computed statistics.
    
    Returns:
        Dict with summary, statistics, sample data, and metadata
    """
    store = get_vector_store()
    
    # Get all document content
    full_text = store.get_full_document_text(doc_hash)
    tables = store.get_document_tables(doc_hash)
    chunks = store.get_document_chunks(doc_hash)
    
    if not full_text and not tables:
        return {
            "status": "error",
            "message": "Document not found or empty"
        }
    
    # Get document info
    docs = store.get_all_documents()
    doc_info = next((d for d in docs if d.get("doc_hash") == doc_hash), {})
    
    # Build context prioritizing tables - limit size for speed
    if tables:
        context = "\n\n---\n\n".join(t.get("text", "")[:3000] for t in tables[:2])
    else:
        context = full_text[:8000]
    
    # Generate summary with LLM - optimized prompt
    messages = [
        {"role": "system", "content": "Summarize data documents concisely."},
        {"role": "user", "content": f"""Summarize: {doc_info.get('filename', 'Unknown')}

DATA:
{context}

Provide:
1. Document type
2. Key columns
3. Sample (5 rows)
4. Key stats
5. 3 key findings"""}
    ]
    
    summary = call_llm(messages, max_tokens=2000)
    
    return {
        "status": "success",
        "summary": summary,
        "filename": doc_info.get("filename", "unknown"),
        "num_chunks": len(chunks),
        "num_tables": len(tables),
        "is_dataset": doc_info.get("is_dataset", False)
    }


def get_document_info(doc_hash: str) -> Dict[str, Any]:
    """
    Get complete document information including structure and statistics.
    
    Returns:
        Dict with columns, data types, sample rows, and metadata
    """
    store = get_vector_store()
    
    chunks = store.get_document_chunks(doc_hash)
    tables = store.get_document_tables(doc_hash)
    
    if not chunks:
        return {"error": "Document not found"}
    
    docs = store.get_all_documents()
    doc_info = next((d for d in docs if d.get("doc_hash") == doc_hash), {})
    
    # Extract column information from table chunks
    columns = []
    sample_data = []
    
    for table in tables[:1]:  # First table
        text = table.get("text", "")
        # Parse markdown table to extract headers
        lines = text.strip().split("\n")
        if lines and "|" in lines[0]:
            headers = [h.strip() for h in lines[0].split("|") if h.strip()]
            columns = headers
            
            # Get sample rows
            for line in lines[2:12]:  # Skip header and separator
                if "|" in line:
                    row = [c.strip() for c in line.split("|") if c.strip()]
                    sample_data.append(row)
    
    return {
        "doc_hash": doc_hash,
        "filename": doc_info.get("filename", "unknown"),
        "num_chunks": len(chunks),
        "num_tables": len(tables),
        "is_dataset": doc_info.get("is_dataset", False),
        "columns": columns,
        "sample_data": sample_data[:10],
        "total_tokens": sum(c.get("tokens", 0) for c in chunks)
    }


def compare_documents(doc_hash1: str, doc_hash2: str) -> Dict[str, Any]:
    """
    Compare two documents and identify similarities/differences.
    """
    store = get_vector_store()
    
    # Get both documents
    doc1_text = store.get_full_document_text(doc_hash1)
    doc2_text = store.get_full_document_text(doc_hash2)
    
    docs = store.get_all_documents()
    doc1_info = next((d for d in docs if d.get("doc_hash") == doc_hash1), {})
    doc2_info = next((d for d in docs if d.get("doc_hash") == doc_hash2), {})
    
    if not doc1_text or not doc2_text:
        return {"error": "One or both documents not found"}
    
    # Truncate for comparison
    doc1_sample = doc1_text[:8000]
    doc2_sample = doc2_text[:8000]
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_COMPARE},
        {"role": "user", "content": f"""Compare these two documents:

## Document 1: {doc1_info.get('filename', 'Unknown')}
{doc1_sample}

## Document 2: {doc2_info.get('filename', 'Unknown')}
{doc2_sample}

Provide a detailed comparison:
1. Structural similarities and differences
2. Column/field comparison
3. Data overlap and unique values
4. Size and completeness comparison
5. Recommendations for using together"""}
    ]
    
    comparison = call_llm(messages, max_tokens=4000)
    
    return {
        "status": "success",
        "comparison": comparison,
        "doc1": doc1_info,
        "doc2": doc2_info
    }


def list_documents() -> List[Dict[str, Any]]:
    """List all indexed documents with metadata."""
    store = get_vector_store()
    return store.get_all_documents()


def list_all_documents() -> List[Dict[str, Any]]:
    """Alias for list_documents."""
    return list_documents()


def get_document_stats(doc_hash: str) -> Dict[str, Any]:
    """Get statistics about a document."""
    store = get_vector_store()
    
    chunks = store.get_document_chunks(doc_hash)
    tables = store.get_document_tables(doc_hash)
    
    if not chunks:
        return {"error": "Document not found"}
    
    return {
        "doc_hash": doc_hash,
        "filename": chunks[0].get("filename", "unknown") if chunks else "unknown",
        "num_chunks": len(chunks),
        "num_tables": len(tables),
        "is_dataset": chunks[0].get("is_dataset", False) if chunks else False,
        "total_tokens": sum(c.get("tokens", 0) for c in chunks)
    }


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

def answer_question(query_text: str, k: int = 10, doc_hash: Optional[str] = None) -> Dict[str, Any]:
    """Legacy function - use query() instead."""
    return query(query_text, doc_hash=doc_hash, k=k)


def analyze_query(query_text: str, k: int = 10, doc_hash: Optional[str] = None) -> Dict[str, Any]:
    """Legacy function - use query() instead."""
    return query(query_text, doc_hash=doc_hash, k=k)
