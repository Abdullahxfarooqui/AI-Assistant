"""
Production RAG Engine with Python-Based Data Analysis.

4-MODE INTELLIGENT ROUTING:
1. DATA_QUERY → query_data() + generate_visuals() → Charts + Stats + Answer
2. DOCUMENT_OVERVIEW → dataset_overview() → Full breakdown
3. FREEFORM_QUERY → conversational_response() → Refuse (only data questions)
4. SYSTEM_TASK → system_instruction() → Guidance message

Key Principles:
- Python/Pandas for ALL statistics computation (not LLM)
- LLM ONLY for reasoning, insights, interpretations
- Preserves ALL data rows - no deduplication
- ALWAYS shows visualizations for DATA_QUERY
- NEVER shows visualizations for other modes
"""
import requests
from typing import List, Dict, Any, Optional, Union, Tuple
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

# OpenRouter API configuration - Use environment variable or Streamlit secrets
import os

def get_api_key():
    """Get API key from Streamlit secrets or environment."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except:
        pass
    return os.environ.get("OPENROUTER_API_KEY", "")

OPENROUTER_API_KEY = get_api_key()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-4o-mini"  # Fast model for quick responses

# Performance settings
MAX_CONTEXT_CHARS = 6000
MAX_CONTEXT_CHARS_DETAILED = 12000
DEFAULT_MAX_TOKENS = 2000
DEFAULT_MAX_TOKENS_DETAILED = 4000


# ============================================================================
# LLM SYSTEM PROMPTS - FOCUSED & CONCISE
# ============================================================================

SYSTEM_PROMPT_CONCISE = """You are a precise RAG answering agent.

CRITICAL RULES:
1. Answer ONLY what the user asked - nothing more
2. Keep answers SHORT (2-5 sentences for simple questions)
3. If user asks about ONE metric, only answer about that metric
4. DO NOT output full reports unless explicitly asked
5. DO NOT dump entire dataset analysis
6. If context is huge, extract ONLY the portion relevant to the question

⛔ ANTI-HALLUCINATION GUARDRAILS (ABSOLUTE):
- NEVER make up data, values, or statistics
- NEVER estimate or approximate numbers
- NEVER assume data exists if not shown in context
- NEVER fill in missing values with guesses
- If data is unavailable, say: "This data is not available in the dataset"
- If uncertain, cite ONLY what appears in provided context
- If asked about something not in the data, DO NOT invent an answer

For metric questions (e.g., "What is oil production?"):
- State ONLY the metric value with units from the data
- Give a 1-sentence explanation based ONLY on provided data
- That's it. No more.

FORBIDDEN:
- Full dataset summaries when not asked
- All columns listing when not asked
- Multi-section reports for simple questions
- Statistics for unrelated metrics
- ANY invented, assumed, or estimated values
- Making up data points, averages, or trends"""

SYSTEM_PROMPT_DETAILED = """You are an expert data analyst providing comprehensive analysis.

The user has requested DETAILED analysis. Provide:
1. Complete overview of the requested topic
2. All relevant statistics with proper formatting
3. Trends and patterns
4. Data quality notes
5. Recommendations

Use proper Markdown formatting with tables where appropriate."""

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
# METRIC COLUMN MAPPING - STRICT FILTERING
# ============================================================================

# Define exact column mappings for each metric type
METRIC_COLUMNS = {
    "oil": ["PROD_OIL_VOL", "SALES_OIL_VOL", "INJ_OIL_VOL", "OIL_RATE", "OIL_DENSITY"],
    "gas": ["PROD_GAS_VOL", "SALES_GAS_VOL", "FUEL_GAS_VOL", "INJ_GAS_VOL", "GAS_RATE", "GAS_DENSITY", "FLARE_GAS_VOL"],
    "water": ["PROD_WAT_VOL", "SALES_WAT_VOL", "INJ_WAT_VOL", "WATER_RATE", "WATER_DENSITY"],
    "condensate": ["PROD_COND_VOL", "SALES_COND_VOL", "COND_RATE", "COND_DENSITY"],
    "lpg": ["PROD_LPG_VOL", "SALES_LPG_VOL", "LPG_RATE"],
    "ngl": ["PROD_NGL_VOL", "SALES_NGL_VOL", "NGL_RATE"],
    "heat": ["VOL_HEAT_SALES", "HEAT_RATE", "ENERGY_PROD", "ENERGY_SOLD"],
    "energy": ["ENERGY_PROD", "ENERGY_SOLD", "ENERGY_RATE", "BTU"],
    "injection": ["INJ_GAS_VOL", "INJ_WAT_VOL", "INJ_OIL_VOL", "INJ_RATE"],
    "production": ["PROD_OIL_VOL", "PROD_GAS_VOL", "PROD_WAT_VOL", "PROD_COND_VOL", "PROD_LPG_VOL"],
    "sales": ["SALES_OIL_VOL", "SALES_GAS_VOL", "SALES_WAT_VOL", "SALES_COND_VOL", "SALES_LPG_VOL"]
}

# Metric detection keywords
METRIC_KEYWORDS = {
    "oil": ["oil", "crude", "petroleum"],
    "gas": ["gas", "natural gas", "mmcf", "mcf", "fuel gas", "flare"],
    "water": ["water", "wat", "h2o", "brine", "produced water"],
    "condensate": ["condensate", "cond"],
    "lpg": ["lpg", "liquefied petroleum"],
    "ngl": ["ngl", "natural gas liquid"],
    "heat": ["heat", "thermal"],
    "energy": ["energy", "btu", "mmbtu"],
    "injection": ["injection", "inject", "injected"],
    "production": ["production", "produced", "prod"],
    "sales": ["sales", "sold", "sale"]
}


def get_target_columns(metrics: List[str], df_columns: List[str]) -> List[str]:
    """
    Get the EXACT columns to use based on detected metrics.
    
    STRICT FILTERING: Only returns columns that match the requested metric.
    If user asks about water production, ONLY water production columns are returned.
    
    Args:
        metrics: List of detected metric types (e.g., ['gas'], ['water', 'production'])
        df_columns: List of columns in the DataFrame
        
    Returns:
        List of column names that match the requested metrics
    """
    if not metrics:
        return []
    
    # Identify specific resources vs generic categories
    specific_resources = [m for m in metrics if m in ['oil', 'gas', 'water', 'condensate', 'lpg', 'ngl', 'heat', 'energy']]
    has_production = 'production' in metrics
    has_sales = 'sales' in metrics
    has_injection = 'injection' in metrics
    
    matched_columns = []
    
    for col in df_columns:
        col_upper = col.upper()
        
        # Skip UOM columns
        if col_upper.endswith('_UOM'):
            continue
        
        # Skip non-metric columns (no VOL, RATE, etc.)
        if not any(kw in col_upper for kw in ['VOL', 'RATE', 'ENERGY', 'BTU', 'HEAT']):
            continue
        
        # ====================================================================
        # CASE 1: Specific resource requested (e.g., water, oil, gas)
        # ====================================================================
        if specific_resources:
            # Column must contain the SPECIFIC resource keyword
            resource_keywords = {
                'oil': ['OIL'],
                'gas': ['GAS'],
                'water': ['WAT'],  # WAT catches both WAT and WATER
                'condensate': ['COND'],
                'lpg': ['LPG'],
                'ngl': ['NGL'],
                'heat': ['HEAT'],
                'energy': ['ENERGY', 'BTU']
            }
            
            col_matches_resource = False
            for resource in specific_resources:
                keywords = resource_keywords.get(resource, [resource.upper()])
                if any(kw in col_upper for kw in keywords):
                    col_matches_resource = True
                    break
            
            if not col_matches_resource:
                continue  # Skip this column - doesn't match requested resource
            
            # Additionally filter by production/sales/injection if specified
            if has_production and 'PROD' not in col_upper:
                continue
            if has_sales and 'SALES' not in col_upper:
                continue
            if has_injection and 'INJ' not in col_upper:
                continue
            
            matched_columns.append(col)
        
        # ====================================================================
        # CASE 2: Only generic category (production, sales, injection)
        # ====================================================================
        elif has_production or has_sales or has_injection:
            if has_production and 'PROD' in col_upper:
                matched_columns.append(col)
            elif has_sales and 'SALES' in col_upper:
                matched_columns.append(col)
            elif has_injection and 'INJ' in col_upper:
                matched_columns.append(col)
    
    return matched_columns


def filter_columns_for_metric(df_columns: List[str], query: str) -> Tuple[List[str], List[str]]:
    """
    Filter DataFrame columns to ONLY include columns relevant to the user's question.
    
    Returns:
        Tuple of (target_columns, detected_metrics)
    """
    metrics = detect_specific_metrics(query)
    
    if not metrics:
        # No specific metric detected - return empty (let handler decide)
        return [], metrics
    
    target_columns = get_target_columns(metrics, df_columns)
    
    return target_columns, metrics


# ============================================================================
# HALLUCINATION PREVENTION & VALIDATION
# ============================================================================

def validate_response_against_context(response: str, context: str) -> Dict[str, Any]:
    """
    Validate that the response does NOT introduce made-up data.
    
    Checks for:
    - Numerical claims not in context
    - Statistical claims not in context
    - Invented metrics or data points
    
    Args:
        response: LLM response text
        context: RAG context provided to LLM
        
    Returns:
        Dict with {'is_valid': bool, 'issues': List[str]}
    """
    issues = []
    
    # Check for common hallucination patterns
    hallucination_patterns = [
        ("approximately", "estimated", "roughly", "about", "around"),  # Estimations
        ("assumed", "likely", "probably", "probably indicates"),  # Assumptions without data
        ("would be", "could be", "should be"),  # Speculative
    ]
    
    response_lower = response.lower()
    context_lower = context.lower()
    
    # Flag speculative language in RAG mode
    for pattern_group in hallucination_patterns:
        for pattern in pattern_group:
            if pattern in response_lower and pattern not in context_lower:
                # Allow some speculation but log it
                if pattern in ("assumed", "estimated", "roughly", "approximately"):
                    issues.append(f"⚠️ Response uses '{pattern}' - ensure values are from data")
    
    return {
        "is_valid": len(issues) < 3,  # Allow some warnings but flag many
        "issues": issues,
        "needs_review": len(issues) > 0
    }


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
    # Get API key fresh each time (in case it wasn't available at module load)
    api_key = get_api_key() or OPENROUTER_API_KEY
    
    if not api_key:
        return "Error: No API key configured. Please set OPENROUTER_API_KEY in .streamlit/secrets.toml"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8503",
        "X-Title": "RAG Engine"
    }
    
    # Use appropriate token limit
    token_limit = min(max_tokens, DEFAULT_MAX_TOKENS)
    
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": token_limit,
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
# MODE-SPECIFIC HANDLERS (4-MODE SYSTEM) - REWRITTEN
# ============================================================================

def handle_freeform_query(user_query: str) -> Dict[str, Any]:
    """
    Handle FREEFORM_QUERY mode - general chat unrelated to dataset.
    
    STRICT MODE: This assistant ONLY answers questions about the uploaded dataset.
    For any non-dataset questions, it politely refuses and guides the user.
    
    Tool: conversational_response()
    - MUST NOT reference dataset at all
    - Returns refusal message (no LLM call needed - saves API cost)
    """
    refusal_message = """⚠️ **I can only answer questions about your uploaded dataset.**

I'm a specialized data analysis assistant focused on helping you explore and understand your oil & gas production data.

**Examples of questions I can help with:**
- "What is the total oil production?"
- "Show me gas production trends"
- "Compare water injection volumes"
- "What wells have the highest production?"
- "Create a chart of monthly production"
- "What's in this document?"

Please ask a question related to your uploaded data, and I'll be happy to help!"""
    
    return {
        "answer": refusal_message,
        "sources": [],
        "intent": "freeform_query",
        "query_mode": "freeform_query",
        "detail_mode": "normal",
        "specific_metrics": [],
        "target_columns": [],
        "show_visualizations": False,  # NEVER for freeform
        "num_chunks": 0,
        "query_type": "general"
    }


def handle_system_task(user_query: str) -> Dict[str, Any]:
    """
    Handle SYSTEM_TASK mode - requests to modify the RAG engine.
    
    Tool: system_instruction()
    - MUST return code fixes, architecture improvements, debugging recommendations
    """
    system_message = """🔧 **System Modification Request Detected**

I understand you want to modify or improve the system. Here's what I can offer:

---

## 📋 Current System Architecture

The RAG engine uses a 4-mode routing system:
1. **DATA_QUERY** → Queries about data (oil, gas, production, etc.)
2. **DOCUMENT_OVERVIEW** → Full document structure analysis
3. **FREEFORM_QUERY** → General questions (refused - data only)
4. **SYSTEM_TASK** → This mode (system modifications)

---

## 🛠️ For System Modifications

To modify this system, you'll need to edit these files:

```
rag-app/
├── app.py                    # Streamlit UI
├── core/
│   ├── rag_engine.py         # Main RAG logic
│   ├── query_router.py       # Intent classification
│   ├── data_engine.py        # Data processing
│   └── vector_store.py       # FAISS storage
```

### Common Modifications:
- **Add new intent**: Edit `query_router.py` → `classify_query_mode()`
- **Change visualizations**: Edit `app.py` → `render_query()` function
- **Modify LLM prompts**: Edit `rag_engine.py` → `SYSTEM_PROMPT_*` constants
- **Change statistics**: Edit `rag_engine.py` → `_handle_calculate_intent()`

---

## 💡 Recommendation

If you want to modify behavior, please specify exactly:
1. What current behavior needs fixing?
2. What is the expected behavior?
3. Which query type is affected?

I can then provide targeted code recommendations.

---

Would you like to ask a data question instead? Try: *"What is oil production?"*"""
    
    return {
        "answer": system_message,
        "sources": [],
        "intent": "system_task",
        "query_mode": "system_task",
        "detail_mode": "normal",
        "specific_metrics": [],
        "target_columns": [],
        "show_visualizations": False,  # NEVER for system tasks
        "num_chunks": 0,
        "query_type": "system"
    }


def handle_document_overview(dataframe: Optional[pd.DataFrame] = None, doc_hash: str = None, user_query: str = "") -> Dict[str, Any]:
    """
    Handle DOCUMENT_OVERVIEW mode - complete dataset breakdown or executive summary.
    
    Tool: dataset_overview()
    - MUST return: rows, columns, categories, numeric/text counts, date ranges,
      missing-value analysis, production/sales column groups
    - OR: Executive summary with business insights if requested
    """
    if dataframe is None or dataframe.empty:
        return {
            "answer": "📭 **No document loaded.** Please upload a document first to see its overview.",
            "sources": [],
            "intent": "document_overview",
            "query_mode": "document_overview",
            "detail_mode": "detailed",
            "specific_metrics": [],
            "target_columns": [],
            "show_visualizations": False,  # NEVER for overview
            "num_chunks": 0,
            "query_type": "data"
        }
    
    df = dataframe
    q_lower = user_query.lower()
    
    # ========================================================================
    # CHECK IF USER WANTS EXECUTIVE SUMMARY vs TECHNICAL OVERVIEW
    # ========================================================================
    is_executive_summary = any(phrase in q_lower for phrase in [
        "executive summary", "exec summary", "high level summary", "high-level summary",
        "brief summary", "quick summary", "business summary", "management summary",
        "summary of this", "give me a summary", "provide a summary", "summarize"
    ])
    
    if is_executive_summary:
        return _generate_executive_summary(df)
    
    # ========================================================================
    # BUILD COMPREHENSIVE DATASET OVERVIEW (Technical)
    # ========================================================================
    overview_parts = []
    
    # 📘 Header
    overview_parts.append("## 📘 Complete Dataset Overview\n")
    
    # ========================================================================
    # 1. BASIC SHAPE
    # ========================================================================
    overview_parts.append("### 📐 Dataset Shape")
    overview_parts.append(f"| Metric | Value |")
    overview_parts.append(f"|--------|-------|")
    overview_parts.append(f"| **Total Rows** | {len(df):,} |")
    overview_parts.append(f"| **Total Columns** | {len(df.columns)} |")
    overview_parts.append(f"| **Memory Usage** | {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB |")
    overview_parts.append("")
    
    # ========================================================================
    # 2. COLUMN TYPE BREAKDOWN
    # ========================================================================
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = []
    
    # Detect date columns
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                pd.to_datetime(df[col].dropna().head(10), errors='coerce')
                date_cols.append(col)
            except:
                pass
    
    overview_parts.append("### 📊 Column Categories")
    overview_parts.append(f"| Category | Count |")
    overview_parts.append(f"|----------|-------|")
    overview_parts.append(f"| Numeric Columns | {len(numeric_cols)} |")
    overview_parts.append(f"| Text/Categorical Columns | {len(text_cols)} |")
    overview_parts.append(f"| Date/Time Columns | {len(date_cols)} |")
    overview_parts.append("")
    
    # ========================================================================
    # 3. PRODUCTION & SALES COLUMN GROUPS
    # ========================================================================
    prod_cols = [c for c in df.columns if 'PROD' in c.upper() and any(x in c.upper() for x in ['VOL', 'VOLUME'])]
    sales_cols = [c for c in df.columns if 'SALES' in c.upper() and any(x in c.upper() for x in ['VOL', 'VOLUME'])]
    inj_cols = [c for c in df.columns if 'INJ' in c.upper() and any(x in c.upper() for x in ['VOL', 'VOLUME'])]
    
    overview_parts.append("### 🛢️ Metric Column Groups")
    
    if prod_cols:
        overview_parts.append(f"\n**Production Columns ({len(prod_cols)}):**")
        for col in prod_cols[:8]:
            try:
                total = pd.to_numeric(df[col], errors='coerce').dropna().sum()
                uom = ""
                if col + "_UOM" in df.columns:
                    uom_vals = df[col + "_UOM"].dropna()
                    uom = f" {uom_vals.iloc[0]}" if len(uom_vals) > 0 else ""
                overview_parts.append(f"- `{col}`: Total = {total:,.2f}{uom}")
            except:
                overview_parts.append(f"- `{col}`: (non-numeric)")
    
    if sales_cols:
        overview_parts.append(f"\n**Sales Columns ({len(sales_cols)}):**")
        for col in sales_cols[:8]:
            try:
                total = pd.to_numeric(df[col], errors='coerce').dropna().sum()
                uom = ""
                if col + "_UOM" in df.columns:
                    uom_vals = df[col + "_UOM"].dropna()
                    uom = f" {uom_vals.iloc[0]}" if len(uom_vals) > 0 else ""
                overview_parts.append(f"- `{col}`: Total = {total:,.2f}{uom}")
            except:
                overview_parts.append(f"- `{col}`: (non-numeric)")
    
    if inj_cols:
        overview_parts.append(f"\n**Injection Columns ({len(inj_cols)}):**")
        for col in inj_cols[:5]:
            try:
                total = pd.to_numeric(df[col], errors='coerce').dropna().sum()
                overview_parts.append(f"- `{col}`: Total = {total:,.2f}")
            except:
                overview_parts.append(f"- `{col}`: (non-numeric)")
    
    overview_parts.append("")
    
    # ========================================================================
    # 4. DATE RANGES
    # ========================================================================
    if date_cols:
        overview_parts.append("### 📅 Date Ranges")
        for col in date_cols[:3]:
            try:
                dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(dates) > 0:
                    overview_parts.append(f"- **{col}:** {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
                    days = (dates.max() - dates.min()).days
                    overview_parts.append(f"  - Duration: {days} days ({days//30} months)")
            except:
                pass
        overview_parts.append("")
    
    # ========================================================================
    # 5. MISSING VALUES ANALYSIS
    # ========================================================================
    overview_parts.append("### 📋 Missing Values Analysis")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    total_cells = df.size
    completeness = ((total_cells - total_missing) / total_cells * 100) if total_cells > 0 else 0
    
    overview_parts.append(f"- **Overall Completeness:** {completeness:.1f}%")
    overview_parts.append(f"- **Total Missing Cells:** {total_missing:,} / {total_cells:,}")
    
    # Top columns with missing values
    missing_cols = missing[missing > 0].sort_values(ascending=False).head(10)
    if len(missing_cols) > 0:
        overview_parts.append("\n**Top Columns with Missing Values:**")
        overview_parts.append("| Column | Missing | % Missing |")
        overview_parts.append("|--------|---------|-----------|")
        for col, count in missing_cols.items():
            pct = (count / len(df) * 100)
            overview_parts.append(f"| {col[:30]} | {count:,} | {pct:.1f}% |")
    overview_parts.append("")
    
    # ========================================================================
    # 6. KEY FIELDS (UNIQUE COUNTS)
    # ========================================================================
    overview_parts.append("### 🔍 Key Fields")
    key_candidates = [c for c in df.columns if any(kw in c.lower() for kw in ['name', 'id', 'well', 'field', 'item', 'type', 'status'])]
    
    if key_candidates:
        overview_parts.append("| Field | Unique Values | Sample Values |")
        overview_parts.append("|-------|---------------|---------------|")
        for col in key_candidates[:8]:
            unique_count = df[col].nunique()
            sample_vals = df[col].dropna().head(3).tolist()
            sample_str = ", ".join(str(v)[:20] for v in sample_vals)
            overview_parts.append(f"| {col[:25]} | {unique_count:,} | {sample_str} |")
    overview_parts.append("")
    
    # ========================================================================
    # 7. NUMERIC SUMMARY STATISTICS
    # ========================================================================
    if numeric_cols:
        overview_parts.append("### 📈 Numeric Column Summary")
        
        # Only show key production/sales columns
        key_numeric = [c for c in numeric_cols if any(x in c.upper() for x in ['PROD', 'SALES', 'INJ', 'VOL', 'ENERGY'])][:10]
        
        if key_numeric:
            overview_parts.append("| Column | Total | Average | Min | Max |")
            overview_parts.append("|--------|-------|---------|-----|-----|")
            for col in key_numeric:
                try:
                    data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(data) > 0:
                        total = data.sum()
                        avg = data.mean()
                        min_val = data.min()
                        max_val = data.max()
                        overview_parts.append(f"| {col[:25]} | {total:,.2f} | {avg:,.2f} | {min_val:,.2f} | {max_val:,.2f} |")
                except:
                    overview_parts.append(f"| {col[:25]} | N/A | N/A | N/A | N/A |")
    
    overview_parts.append("")
    
    # ========================================================================
    # 8. DATA QUALITY NOTES
    # ========================================================================
    overview_parts.append("### 💡 Data Quality Notes")
    
    # Completely empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        overview_parts.append(f"- ⚠️ **{len(empty_cols)} completely empty column(s):** {', '.join(empty_cols[:5])}")
    
    # Columns with >50% missing
    high_missing = [c for c in df.columns if df[c].isna().sum() / len(df) > 0.5]
    if high_missing:
        overview_parts.append(f"- ⚠️ **{len(high_missing)} column(s) with >50% missing data**")
    
    # Duplicate rows check
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        overview_parts.append(f"- ⚠️ **{dup_count:,} duplicate rows detected**")
    else:
        overview_parts.append("- ✅ No duplicate rows detected")
    
    overview_parts.append("")
    overview_parts.append("---")
    overview_parts.append("*Use specific questions like 'Show me oil production' to explore the data in detail.*")
    
    answer = "\n".join(overview_parts)
    
    return {
        "answer": answer,
        "sources": [],
        "intent": "document_overview",
        "query_mode": "document_overview",
        "detail_mode": "detailed",
        "specific_metrics": [],
        "target_columns": list(df.columns),
        "show_visualizations": False,  # NEVER for overview
        "num_chunks": 0,
        "query_type": "data"
    }


def _generate_executive_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a business-focused executive summary of the dataset.
    This is a high-level summary suitable for management/stakeholders.
    """
    summary_parts = []
    
    # ========================================================================
    # HEADER
    # ========================================================================
    summary_parts.append("## 📋 Executive Summary\n")
    
    # ========================================================================
    # 1. DOCUMENT OVERVIEW (2-3 sentences)
    # ========================================================================
    summary_parts.append("### 📄 Document Overview")
    
    # Detect date range
    date_range_str = ""
    date_cols = [c for c in df.columns if 'DATE' in c.upper() or 'TIME' in c.upper()]
    for col in date_cols[:1]:
        try:
            dates = pd.to_datetime(df[col], errors='coerce').dropna()
            if len(dates) > 0:
                start_date = dates.min().strftime('%B %d, %Y')
                end_date = dates.max().strftime('%B %d, %Y')
                date_range_str = f" covering the period from **{start_date}** to **{end_date}**"
                break
        except:
            pass
    
    summary_parts.append(
        f"This dataset contains **{len(df):,} records** across **{len(df.columns)} data fields**{date_range_str}. "
        f"The data represents operational metrics including production volumes, sales figures, and related measurements.\n"
    )
    
    # ========================================================================
    # 2. KEY METRICS SUMMARY
    # ========================================================================
    summary_parts.append("### 📊 Key Performance Metrics\n")
    
    # Find production columns
    prod_cols = [c for c in df.columns if 'PROD' in c.upper() and 'VOL' in c.upper()]
    sales_cols = [c for c in df.columns if 'SALES' in c.upper() and 'VOL' in c.upper()]
    
    # Build metrics table
    metrics_table = []
    metrics_table.append("| Metric | Total Volume | Average | Unit |")
    metrics_table.append("|--------|--------------|---------|------|")
    
    key_metrics_found = 0
    
    # Process production columns
    for col in prod_cols[:5]:
        try:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(data) > 0 and data.sum() > 0:
                total = data.sum()
                avg = data.mean()
                
                # Get unit
                uom = "units"
                if col + "_UOM" in df.columns:
                    uom_vals = df[col + "_UOM"].dropna()
                    uom = uom_vals.iloc[0] if len(uom_vals) > 0 else "units"
                
                # Clean column name for display
                display_name = col.replace("PROD_", "").replace("_VOL", "").replace("_", " ").title()
                metrics_table.append(f"| {display_name} Production | {total:,.2f} | {avg:,.2f} | {uom} |")
                key_metrics_found += 1
        except:
            pass
    
    # Process sales columns
    for col in sales_cols[:3]:
        try:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(data) > 0 and data.sum() > 0:
                total = data.sum()
                avg = data.mean()
                
                uom = "units"
                if col + "_UOM" in df.columns:
                    uom_vals = df[col + "_UOM"].dropna()
                    uom = uom_vals.iloc[0] if len(uom_vals) > 0 else "units"
                
                display_name = col.replace("SALES_", "").replace("_VOL", "").replace("_", " ").title()
                metrics_table.append(f"| {display_name} Sales | {total:,.2f} | {avg:,.2f} | {uom} |")
                key_metrics_found += 1
        except:
            pass
    
    if key_metrics_found > 0:
        summary_parts.extend(metrics_table)
        summary_parts.append("")
    else:
        # Fallback to any numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        key_numeric = [c for c in numeric_cols if not c.endswith('_UOM')][:5]
        if key_numeric:
            summary_parts.append("| Metric | Total | Average |")
            summary_parts.append("|--------|-------|---------|")
            for col in key_numeric:
                try:
                    data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(data) > 0:
                        summary_parts.append(f"| {col[:30]} | {data.sum():,.2f} | {data.mean():,.2f} |")
                except:
                    pass
            summary_parts.append("")
    
    # ========================================================================
    # 3. KEY HIGHLIGHTS (Business insights)
    # ========================================================================
    summary_parts.append("### 💡 Key Highlights\n")
    
    highlights = []
    
    # Find the largest production metric
    max_prod_col = None
    max_prod_val = 0
    for col in prod_cols[:5]:
        try:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(data) > 0 and data.sum() > max_prod_val:
                max_prod_val = data.sum()
                max_prod_col = col
        except:
            pass
    
    if max_prod_col:
        display_name = max_prod_col.replace("PROD_", "").replace("_VOL", "").replace("_", " ").title()
        uom = "units"
        if max_prod_col + "_UOM" in df.columns:
            uom_vals = df[max_prod_col + "_UOM"].dropna()
            uom = uom_vals.iloc[0] if len(uom_vals) > 0 else "units"
        highlights.append(f"- **Largest Production Volume:** {display_name} with **{max_prod_val:,.2f} {uom}** total")
    
    # Data completeness
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
    highlights.append(f"- **Data Completeness:** {completeness:.1f}% of all data fields are populated")
    
    # Number of unique items/wells if available
    key_id_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['item_name', 'well', 'facility', 'asset'])]
    if key_id_cols:
        unique_count = df[key_id_cols[0]].nunique()
        col_name = key_id_cols[0].replace("_", " ").title()
        highlights.append(f"- **Unique {col_name}s:** {unique_count:,} distinct entries in the dataset")
    
    # Date span
    if date_range_str:
        for col in date_cols[:1]:
            try:
                dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(dates) > 0:
                    days = (dates.max() - dates.min()).days
                    months = days // 30
                    highlights.append(f"- **Time Span:** {months} months ({days:,} days) of operational data")
                    break
            except:
                pass
    
    summary_parts.extend(highlights)
    summary_parts.append("")
    
    # ========================================================================
    # 4. DATA QUALITY NOTES (Brief)
    # ========================================================================
    summary_parts.append("### ⚠️ Data Quality Notes\n")
    
    quality_notes = []
    
    # Empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        quality_notes.append(f"- {len(empty_cols)} column(s) contain no data and may need review")
    
    # High missing columns
    high_missing = [c for c in df.columns if df[c].isna().sum() / len(df) > 0.5]
    if high_missing:
        quality_notes.append(f"- {len(high_missing)} column(s) have more than 50% missing values")
    
    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        quality_notes.append(f"- {dup_count:,} potential duplicate records identified")
    
    if quality_notes:
        summary_parts.extend(quality_notes)
    else:
        summary_parts.append("- ✅ No significant data quality issues detected")
    
    summary_parts.append("")
    
    # ========================================================================
    # 5. RECOMMENDATIONS
    # ========================================================================
    summary_parts.append("### 📌 Suggested Analysis\n")
    summary_parts.append("Based on this dataset, consider exploring:")
    summary_parts.append("- **Trend Analysis:** Production/sales patterns over time")
    summary_parts.append("- **Performance Comparison:** Metrics across different items/facilities")
    summary_parts.append("- **Anomaly Detection:** Unusual spikes or drops in key metrics")
    summary_parts.append("")
    summary_parts.append("---")
    summary_parts.append("*Ask specific questions like 'Show oil production trends' or 'Compare gas and water volumes' for detailed analysis.*")
    
    answer = "\n".join(summary_parts)
    
    return {
        "answer": answer,
        "sources": [],
        "intent": "document_overview",
        "query_mode": "document_overview",
        "detail_mode": "detailed",
        "specific_metrics": [],
        "target_columns": list(df.columns),
        "show_visualizations": False,  # No charts for executive summary
        "num_chunks": 0,
        "query_type": "data"
    }


def handle_data_query(
    user_query: str,
    dataframe: Optional[pd.DataFrame],
    context: str,
    doc_hash: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handle DATA_QUERY mode - the main data analysis path.
    
    Tools: query_data() + generate_visuals() (ALWAYS BOTH)
    
    MUST RETURN:
    - 📊 Summary block
    - 🔢 Statistics (total, avg, min, max, count)
    - 📈 Visuals (flag for app.py to render)
    - 💡 Interpretation
    """
    df = dataframe
    
    # Detect metrics and detail mode
    from core.query_router import get_requested_metrics
    specific_metrics = get_requested_metrics(user_query)
    detail_mode = detect_detail_mode(user_query)
    
    # Get target columns based on metrics
    target_columns = []
    if specific_metrics and df is not None:
        target_columns = get_target_columns(specific_metrics, df.columns.tolist())
    
    # ========================================================================
    # COMPUTE STATISTICS (ALWAYS - Python-based)
    # ========================================================================
    stats_block = ""
    if df is not None and not df.empty:
        stats_block = compute_data_statistics(df, specific_metrics, target_columns)
    
    # ========================================================================
    # GENERATE LLM ANSWER
    # ========================================================================
    # Build prompt with statistics
    is_detailed = (detail_mode == "detailed")
    
    if is_detailed:
        system_prompt = SYSTEM_PROMPT_DETAILED
        max_tokens = 3000
    else:
        system_prompt = SYSTEM_PROMPT_CONCISE
        max_tokens = 1500
    
    user_prompt = f"""Answer this data question:
{user_query}

PRE-COMPUTED STATISTICS (USE THESE - already calculated by Python):
{stats_block}

ADDITIONAL CONTEXT:
{context[:4000]}

RULES:
- Use the EXACT values from statistics above
- DO NOT make up numbers
- Include a brief interpretation
- If user asked about specific metric, focus on that metric only"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    answer = call_llm(messages, max_tokens=max_tokens)
    
    return {
        "answer": answer,
        "sources": [],
        "intent": "data_query",
        "query_mode": "data_query",
        "detail_mode": detail_mode,
        "specific_metrics": specific_metrics,
        "target_columns": target_columns,
        "show_visualizations": True,  # ALWAYS for data queries
        "num_chunks": 0,
        "query_type": "data",
        "stats_block": stats_block  # Pass stats for display
    }


def compute_data_statistics(
    df: pd.DataFrame, 
    specific_metrics: List[str],
    target_columns: List[str]
) -> str:
    """
    Compute ACTUAL statistics from DataFrame.
    Returns formatted string with total, avg, min, max, count.
    Includes comparison section when multiple metrics are requested.
    """
    if df is None or df.empty:
        return "No data available."
    
    stats_lines = []
    stats_lines.append("## 📊 Computed Statistics\n")
    stats_lines.append(f"**Dataset Size:** {len(df):,} rows × {len(df.columns)} columns\n")
    
    # Track metric totals for comparison
    metric_totals = {}
    
    # If we have target columns, compute their stats
    if target_columns:
        stats_lines.append("### Requested Metric Statistics:")
        stats_lines.append("| Column | Total | Average | Min | Max | Count |")
        stats_lines.append("|--------|-------|---------|-----|-----|-------|")
        
        for col in target_columns[:10]:  # Limit to 10 columns
            try:
                data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(data) > 0:
                    total = data.sum()
                    avg = data.mean()
                    min_val = data.min()
                    max_val = data.max()
                    count = len(data)
                    
                    # Get unit
                    uom = ""
                    if col + "_UOM" in df.columns:
                        uom_vals = df[col + "_UOM"].dropna()
                        uom = f" {uom_vals.iloc[0]}" if len(uom_vals) > 0 else ""
                    
                    stats_lines.append(
                        f"| {col[:30]} | {total:,.2f}{uom} | {avg:,.2f}{uom} | "
                        f"{min_val:,.2f} | {max_val:,.2f} | {count:,} |"
                    )
                    
                    # Track for comparison
                    metric_totals[col] = {"total": total, "avg": avg, "unit": uom}
            except Exception:
                pass
        stats_lines.append("")
        
        # Add comparison section if multiple metrics requested
        if len(specific_metrics) >= 2 and len(metric_totals) >= 2:
            stats_lines.append("### 📊 Metric Comparison:")
            stats_lines.append("| Metric | Total Volume | % of Combined |")
            stats_lines.append("|--------|--------------|---------------|")
            
            combined_total = sum(m["total"] for m in metric_totals.values())
            for col, data in metric_totals.items():
                pct = (data["total"] / combined_total * 100) if combined_total > 0 else 0
                stats_lines.append(f"| {col[:25]} | {data['total']:,.2f}{data['unit']} | {pct:.1f}% |")
            
            stats_lines.append(f"\n**Combined Total:** {combined_total:,.2f}")
            
            # Add which metric is larger
            if len(metric_totals) >= 2:
                sorted_metrics = sorted(metric_totals.items(), key=lambda x: x[1]["total"], reverse=True)
                largest = sorted_metrics[0]
                smallest = sorted_metrics[-1]
                if largest[1]["total"] > 0 and smallest[1]["total"] > 0:
                    ratio = largest[1]["total"] / smallest[1]["total"]
                    stats_lines.append(f"\n**Comparison:** {largest[0]} is {ratio:.1f}x larger than {smallest[0]}")
            stats_lines.append("")
    else:
        # Fall back to key production columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        key_cols = [c for c in numeric_cols if any(x in c.upper() for x in ['PROD', 'SALES', 'VOL', 'ENERGY'])][:8]
        
        if key_cols:
            stats_lines.append("### Key Metric Statistics:")
            stats_lines.append("| Column | Total | Average | Min | Max | Count |")
            stats_lines.append("|--------|-------|---------|-----|-----|-------|")
            
            for col in key_cols:
                try:
                    data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(data) > 0:
                        total = data.sum()
                        avg = data.mean()
                        min_val = data.min()
                        max_val = data.max()
                        count = len(data)
                        stats_lines.append(
                            f"| {col[:30]} | {total:,.2f} | {avg:,.2f} | "
                            f"{min_val:,.2f} | {max_val:,.2f} | {count:,} |"
                        )
                except Exception:
                    pass
            stats_lines.append("")
    
    # Date range if available
    date_cols = [c for c in df.columns if 'DATE' in c.upper() or 'TIME' in c.upper()]
    for col in date_cols[:1]:
        try:
            dates = pd.to_datetime(df[col], errors='coerce').dropna()
            if len(dates) > 0:
                stats_lines.append(f"**Date Range:** {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
        except:
            pass
    
    return "\n".join(stats_lines)


# ============================================================================
# QUESTION COMPLEXITY & METRIC DETECTION
# ============================================================================

def is_general_query(query: str) -> bool:
    """
    STRICT CHECK: Is this a general/overview query with NO specific resource?
    
    General queries should NOT trigger any oil/gas/water visualizations.
    They should only return dataset overview (rows, columns, date range).
    
    Returns:
        True if general query (no specific metric), False if resource-specific
    """
    q = query.lower()
    
    # First check: Does the query mention ANY specific resource?
    specific_resources = ['oil', 'gas', 'water', 'condensate', 'lpg', 'ngl', 
                          'heat', 'energy', 'injection', 'fuel', 'flare']
    
    has_specific_resource = any(resource in q for resource in specific_resources)
    
    # If a specific resource is mentioned, NOT a general query
    if has_specific_resource:
        return False
    
    # General query patterns - these should NOT trigger visualizations
    general_patterns = [
        'what is in', 'what\'s in', 'whats in',
        'tell me about', 'tell me what',
        'summarize', 'summary', 'overview',
        'describe', 'explain this', 'explain the',
        'what does this', 'what is this',
        'about this document', 'about this file', 'about the document',
        'what are the columns', 'list columns', 'show columns',
        'document info', 'file info', 'dataset info',
        'what data', 'what information', 'what fields',
        'give me a summary', 'provide summary', 'brief summary',
        'high level', 'high-level', 'general overview'
    ]
    
    if any(pattern in q for pattern in general_patterns):
        return True
    
    # Very short queries without specific resources are likely general
    words = q.split()
    if len(words) <= 3 and not has_specific_resource:
        return True
    
    return False


def detect_detail_mode(query: str) -> str:
    """
    Detect the detail level requested by the user.
    
    Returns:
        'detailed' - User wants full, comprehensive analysis (k=12-20)
        'normal' - Standard medium-level response (k=6-10)
        'brief' - Short, concise answer (k=3-5)
    """
    q = query.lower()
    
    # DETAILED mode triggers - user wants comprehensive analysis
    detailed_triggers = [
        'in detail', 'detailed', 'full', 'complete', 'comprehensive', 
        'everything', 'all', 'entire', 'whole', 'in-depth', 'thorough', 
        'elaborate', 'explain deeply', 'deep dive', 'full breakdown',
        'full report', 'complete analysis', 'detailed analysis',
        'give me everything', 'tell me everything', 'show me everything',
        'all statistics', 'all metrics', 'full summary', 'complete summary',
        'entire dataset', 'full dataset', 'deeply', 'extensively'
    ]
    
    if any(trigger in q for trigger in detailed_triggers):
        return 'detailed'
    
    # BRIEF mode triggers - user wants short answer
    brief_triggers = [
        'short', 'brief', 'quick', 'concise', 'simple', 
        'just tell me', 'one line', 'summary only', 'tldr',
        'in brief', 'briefly', 'quickly'
    ]
    
    if any(trigger in q for trigger in brief_triggers):
        return 'brief'
    
    # Default to normal mode
    return 'normal'


def get_retrieval_k(detail_mode: str, has_metrics: bool) -> int:
    """
    Get the number of chunks to retrieve based on detail mode.
    
    Args:
        detail_mode: 'detailed', 'normal', or 'brief'
        has_metrics: Whether specific metrics were detected
    
    Returns:
        k value for retrieval
    """
    if detail_mode == 'detailed':
        return 15  # 12-20 chunks for detailed analysis
    elif detail_mode == 'brief':
        return 4   # 3-5 chunks for brief answers
    else:
        return 8   # 6-10 chunks for normal queries


def detect_specific_metrics(query: str) -> List[str]:
    """
    Detect which specific metrics the user is asking about.
    Uses METRIC_KEYWORDS for comprehensive detection.
    
    IMPORTANT: When a specific resource (oil, gas, water) is detected,
    we DON'T include generic 'production' or 'sales' since those would
    add all resource types.
    
    Returns:
        List of metric categories: ['oil'], ['gas'], ['water'], etc.
        Empty list means no specific metric - could be general question
    """
    q = query.lower()
    metrics = []
    
    # Specific resources take priority
    specific_resources = ['oil', 'gas', 'water', 'condensate', 'lpg', 'ngl', 'heat', 'energy']
    
    # Check each metric type using METRIC_KEYWORDS
    for metric_type, keywords in METRIC_KEYWORDS.items():
        if any(keyword in q for keyword in keywords):
            metrics.append(metric_type)
    
    # CRITICAL FIX: If we have specific resources, remove generic production/sales
    # Because "gas production" should only show GAS, not all production
    has_specific_resource = any(m in specific_resources for m in metrics)
    
    if has_specific_resource:
        # Remove generic production/sales/injection since they would add other resources
        metrics = [m for m in metrics if m in specific_resources]
    
    return metrics


def is_simple_question(query: str) -> bool:
    """
    Determine if this is a simple, single-metric question.
    
    Simple questions:
    - "What is oil production?"
    - "Show me gas volume"
    - "What is the total water?"
    
    Complex questions:
    - "Give me complete analysis"
    - "Summarize the entire dataset"
    - "Compare all metrics"
    """
    q = query.lower()
    
    # If detailed mode is requested, not simple
    if detect_detail_mode(query):
        return False
    
    # Check for single metric questions
    metrics = detect_specific_metrics(query)
    if len(metrics) == 1 or len(metrics) == 2:  # e.g., "oil production" = 2 metrics
        return True
    
    # Short questions are usually simple
    word_count = len(q.split())
    if word_count <= 8:
        return True
    
    # Questions with "what is" about one thing
    simple_patterns = [
        r'^what is [\w\s]+\??$',
        r'^show [\w\s]+$',
        r'^tell me about [\w\s]+$',
        r'^how much [\w\s]+\??$',
        r'^what are the [\w\s]+\??$'
    ]
    
    for pattern in simple_patterns:
        if re.match(pattern, q):
            return True
    
    return False


def get_metric_specific_stats(df: pd.DataFrame, metrics: List[str]) -> str:
    """
    Get statistics ONLY for the requested metrics.
    
    Returns concise stats for just the asked metrics, not the whole dataset.
    """
    if df is None or df.empty:
        return ""
    
    result_lines = []
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Map metrics to column patterns
    metric_patterns = {
        'oil': ['OIL'],
        'gas': ['GAS'],
        'water': ['WAT', 'WATER'],
        'condensate': ['COND'],
        'energy': ['ENERGY', 'BTU'],
        'injection': ['INJ'],
        'sales': ['SALES'],
        'production': ['PROD']
    }
    
    # Find matching columns
    matched_cols = []
    for metric in metrics:
        patterns = metric_patterns.get(metric, [metric.upper()])
        for col in numeric_cols:
            col_upper = col.upper()
            if any(p in col_upper for p in patterns):
                if 'VOL' in col_upper or 'VOLUME' in col_upper or col not in matched_cols:
                    matched_cols.append(col)
    
    # Remove duplicates while preserving order
    matched_cols = list(dict.fromkeys(matched_cols))
    
    # Limit to most relevant columns (volumes first)
    volume_cols = [c for c in matched_cols if 'VOL' in c.upper()]
    other_cols = [c for c in matched_cols if 'VOL' not in c.upper()]
    matched_cols = (volume_cols + other_cols)[:5]  # Max 5 columns
    
    if not matched_cols:
        return ""
    
    for col in matched_cols:
        total = df[col].dropna().sum()
        count = df[col].notna().sum()
        avg = df[col].dropna().mean() if count > 0 else 0
        
        # Get unit
        uom = ""
        if col + "_UOM" in df.columns:
            uom_vals = df[col + "_UOM"].dropna()
            uom = f" {uom_vals.iloc[0]}" if len(uom_vals) > 0 else ""
        
        result_lines.append(f"**{col}**: Total = {total:,.2f}{uom}, Average = {avg:,.2f}{uom} ({count:,} records)")
    
    return "\n".join(result_lines)


def compress_answer(answer: str, max_words: int = 150) -> str:
    """
    Compress a long answer to be more concise.
    Used when detail_mode=False but LLM returned too much.
    """
    if not answer:
        return answer
    
    words = answer.split()
    if len(words) <= max_words:
        return answer
    
    # Try to find a natural break point
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    
    compressed = []
    word_count = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if word_count + sentence_words > max_words:
            break
        compressed.append(sentence)
        word_count += sentence_words
    
    if compressed:
        return " ".join(compressed)
    
    # Fallback: just truncate
    return " ".join(words[:max_words]) + "..."


def generate_dynamic_summary(df: pd.DataFrame, detail_mode: str, chunks: List[Dict] = None) -> str:
    """
    Generate a DYNAMIC summary that adapts to the requested detail level.
    Content is derived from actual data, not templates.
    
    Args:
        df: The DataFrame to analyze
        detail_mode: 'detailed', 'normal', or 'brief'
        chunks: Optional retrieved chunks for additional context
    
    Returns:
        Dynamic summary based on actual data content
    """
    if df is None or df.empty:
        return "No data available to summarize."
    
    # Analyze the actual data
    all_cols = df.columns.tolist()
    all_cols_upper = [c.upper() for c in all_cols]
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Detect timeframe
    timeframe = ""
    date_col = None
    for col in df.columns:
        if 'DATE' in col.upper() or 'TIME' in col.upper():
            try:
                dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(dates) > 0:
                    date_col = col
                    min_date = dates.min()
                    max_date = dates.max()
                    timeframe = f"{min_date.strftime('%B %Y')} to {max_date.strftime('%B %Y')}"
                    break
            except:
                pass
    
    # Detect entities
    entities = []
    entity_counts = {}
    for col in text_cols:
        col_upper = col.upper()
        if any(e in col_upper for e in ['WELL', 'FIELD', 'BATTERY', 'FACILITY', 'COMPLETION', 'ITEM']):
            unique_count = df[col].nunique()
            if unique_count > 0 and unique_count < 10000:
                entity_name = col.replace('_', ' ').title()
                entities.append(entity_name)
                entity_counts[col] = unique_count
    
    # Categorize numeric columns
    prod_cols = [c for c in numeric_cols if 'PROD' in c.upper() and 'VOL' in c.upper()]
    sales_cols = [c for c in numeric_cols if 'SALES' in c.upper() and 'VOL' in c.upper()]
    inj_cols = [c for c in numeric_cols if 'INJ' in c.upper() and 'VOL' in c.upper()]
    energy_cols = [c for c in numeric_cols if any(x in c.upper() for x in ['ENERGY', 'BTU', 'HEAT'])]
    
    # ========================================================================
    # BUILD SUMMARY BASED ON DETAIL LEVEL
    # ========================================================================
    
    if detail_mode == 'brief':
        # SHORT SUMMARY: 3-5 lines
        lines = []
        lines.append("This document contains oil and gas operational data.")
        if timeframe:
            lines.append(f"Time period: {timeframe}.")
        metrics_found = []
        if prod_cols: metrics_found.append("production")
        if sales_cols: metrics_found.append("sales")
        if inj_cols: metrics_found.append("injection")
        if metrics_found:
            lines.append(f"Includes {', '.join(metrics_found)} metrics.")
        return " ".join(lines)
    
    elif detail_mode == 'detailed':
        # DETAILED SUMMARY: 20-40 lines with full breakdown
        sections = []
        
        # Overview
        sections.append("**Document Overview**")
        sections.append(f"This dataset contains comprehensive oil and gas operational records covering {timeframe if timeframe else 'multiple time periods'}.")
        
        # Production Metrics with actual values
        if prod_cols:
            sections.append("")
            sections.append("**Production Metrics**")
            for col in prod_cols[:6]:
                data = df[col].dropna()
                if len(data) > 0:
                    total = data.sum()
                    avg = data.mean()
                    min_val = data.min()
                    max_val = data.max()
                    completeness = (len(data) / len(df)) * 100
                    # Get UOM if available
                    uom = ""
                    if col + "_UOM" in df.columns:
                        uom_vals = df[col + "_UOM"].dropna()
                        uom = f" {uom_vals.iloc[0]}" if len(uom_vals) > 0 else ""
                    sections.append(f"- **{col}**: Total={total:,.2f}{uom}, Avg={avg:,.2f}{uom}, Range=[{min_val:,.2f} - {max_val:,.2f}], Completeness={completeness:.1f}%")
        
        # Sales Metrics
        if sales_cols:
            sections.append("")
            sections.append("**Sales Metrics**")
            for col in sales_cols[:4]:
                data = df[col].dropna()
                if len(data) > 0:
                    total = data.sum()
                    avg = data.mean()
                    completeness = (len(data) / len(df)) * 100
                    uom = ""
                    if col + "_UOM" in df.columns:
                        uom_vals = df[col + "_UOM"].dropna()
                        uom = f" {uom_vals.iloc[0]}" if len(uom_vals) > 0 else ""
                    sections.append(f"- **{col}**: Total={total:,.2f}{uom}, Avg={avg:,.2f}{uom}, Completeness={completeness:.1f}%")
        
        # Injection Metrics
        if inj_cols:
            sections.append("")
            sections.append("**Injection Metrics**")
            for col in inj_cols[:3]:
                data = df[col].dropna()
                if len(data) > 0:
                    total = data.sum()
                    completeness = (len(data) / len(df)) * 100
                    sections.append(f"- **{col}**: Total={total:,.2f}, Completeness={completeness:.1f}%")
        
        # Asset Information
        if entity_counts:
            sections.append("")
            sections.append("**Asset Information**")
            for col, count in list(entity_counts.items())[:5]:
                sections.append(f"- {col}: {count} unique values")
                # Show top values if reasonable
                if count <= 10:
                    top_vals = df[col].value_counts().head(5)
                    top_str = ", ".join([f"{v}({c})" for v, c in top_vals.items()])
                    sections.append(f"  Top entries: {top_str}")
        
        # Data Quality
        sections.append("")
        sections.append("**Data Quality Analysis**")
        null_counts = df.isnull().sum()
        high_null_cols = null_counts[null_counts > len(df) * 0.5].head(5)
        if len(high_null_cols) > 0:
            sections.append(f"- Columns with >50% missing data: {', '.join(high_null_cols.index.tolist())}")
        else:
            sections.append("- Data completeness is generally good across all columns.")
        
        # Time patterns if date exists
        if date_col:
            sections.append("")
            sections.append("**Temporal Coverage**")
            sections.append(f"- Date range: {timeframe}")
            date_series = pd.to_datetime(df[date_col], errors='coerce').dropna()
            if len(date_series) > 1:
                date_diff = date_series.diff().dropna()
                avg_interval = date_diff.mean()
                sections.append(f"- Average data interval: {avg_interval.days} days")
        
        return "\n".join(sections)
    
    else:
        # NORMAL SUMMARY: Medium detail (5-10 lines)
        sections = []
        
        sections.append("**Document Summary**")
        
        # Domain description
        opening = "This document contains daily operational data from the oil and gas sector"
        if entities:
            entity_str = ", ".join(entities[:3])
            opening += f", tracking assets including {entity_str}"
        opening += "."
        sections.append(opening)
        
        if timeframe:
            sections.append(f"The dataset covers the period from {timeframe}.")
        
        # Production summary
        if prod_cols:
            sections.append("")
            sections.append("**Production Metrics**")
            prod_types = []
            if any('OIL' in c.upper() for c in prod_cols): prod_types.append("oil")
            if any('GAS' in c.upper() for c in prod_cols): prod_types.append("gas")
            if any('WAT' in c.upper() for c in prod_cols): prod_types.append("water")
            if any('COND' in c.upper() for c in prod_cols): prod_types.append("condensate")
            if prod_types:
                prod_str = ", ".join(prod_types[:-1]) + f", and {prod_types[-1]}" if len(prod_types) > 1 else prod_types[0]
                sections.append(f"Production volumes recorded for {prod_str}.")
        
        # Sales summary
        if sales_cols:
            sections.append("")
            sections.append("**Sales Metrics**")
            sections.append("Commercial sales volumes are tracked in this dataset.")
        
        # Other metrics
        other = []
        if inj_cols: other.append("injection volumes")
        if energy_cols: other.append("energy measurements")
        if other:
            sections.append("")
            sections.append(f"**Additional Metrics**: {', '.join(other)}.")
        
        sections.append("")
        sections.append("This data provides a comprehensive view of field performance and operational activities.")
        
        return "\n".join(sections)


def generate_dataset_overview(df: pd.DataFrame, filename: str = "") -> str:
    """Wrapper for backward compatibility - generates normal-level summary."""
    return generate_dynamic_summary(df, 'normal')


# ============================================================================
# INTENT DETECTION
# ============================================================================

def detect_intent(query: str) -> str:
    """
    Detect query intent for optimized response strategy.
    
    STRICT INTENT ROUTING:
    - general_overview: NO visualizations, just dataset metadata
    - resource_specific: Only show charts for the mentioned resource
    
    Intents:
    - general_overview: General queries with NO specific resource -> NO CHARTS
    - summarize: Brief summary 
    - explain: What is this, describe, overview
    - calculate: Totals, sums, averages (fast Python computation)
    - show: Display data, tables
    - search: Find specific records
    - compare: Compare documents or values
    - trend: Trend analysis
    - anomaly: Outlier detection
    - general: Default fallback
    """
    q = query.lower()
    
    # FIRST: Check if this is a GENERAL query with no specific resource
    # These should NEVER trigger oil/gas/water visualizations
    if is_general_query(query):
        return "general_overview"
    
    # SECOND: Check for explicit summary/explain requests
    if any(w in q for w in ['summarize', 'summary', 'give me summary', 'document summary', 'brief overview', 'key points', 'main points']):
        return "summarize"
    
    if any(w in q for w in ['what is this', 'what does this', 'describe this', 'explain this', 'overview of', 'what are the columns', 'about this document', 'about this file']):
        return "explain"
    
    # Check for calculations - but only if it's a specific calculation question
    calc_keywords = ['total', 'sum', 'average', 'avg', 'count', 'how many', 'how much', 
                     'calculate', 'mean', 'median', 'minimum', 'maximum', 'min of', 'max of']
    calc_patterns = ['what is the total', 'what is the sum', 'what is the average',
                     'give me total', 'give me the total', 'show total', 'show the total',
                     'calculate the', 'compute the', 'find the total', 'find the sum']
    
    if any(w in q for w in calc_keywords) or any(p in q for p in calc_patterns):
        return "calculate"
    
    # Resource-specific queries (oil, gas, water mentioned) -> these CAN have charts
    if any(w in q for w in ['what is', 'what does', 'describe', 'explain', 'overview']):
        return "explain"
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
# MAIN QUERY FUNCTION (REWRITTEN FOR 4-MODE SYSTEM)
# ============================================================================

def query(
    user_query: str,
    doc_hash: Optional[str] = None,
    k: int = 10,
    dataframe: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Main unified query function with 4-MODE intelligent routing.
    
    ROUTING LOGIC (STRICT PRIORITY ORDER):
    1. FREEFORM_QUERY → conversational_response() → Refuse
    2. SYSTEM_TASK → system_instruction() → Guidance
    3. DOCUMENT_OVERVIEW → dataset_overview() → Full breakdown
    4. DATA_QUERY → query_data() + generate_visuals() → Stats + Charts
    
    Args:
        user_query: User's natural language query
        doc_hash: Optional specific document to query
        k: Number of chunks to retrieve
        dataframe: Optional DataFrame for direct statistics computation
        
    Returns:
        Dict with answer, sources, intent, show_visualizations, and metadata
    """
    # ========================================================================
    # STEP 1: QUERY MODE CLASSIFICATION (4 MODES)
    # ========================================================================
    from core.query_router import classify_query_mode, QueryMode, get_requested_metrics
    
    # Pass DataFrame columns to help classification
    df_columns = list(dataframe.columns) if dataframe is not None else []
    mode, confidence, reason = classify_query_mode(user_query, df_columns)
    
    # ========================================================================
    # MODE 1: FREEFORM_QUERY - Refuse non-dataset questions
    # ========================================================================
    if mode == QueryMode.FREEFORM_QUERY:
        return handle_freeform_query(user_query)
    
    # ========================================================================
    # MODE 2: SYSTEM_TASK - Return system modification guidance
    # ========================================================================
    if mode == QueryMode.SYSTEM_TASK:
        return handle_system_task(user_query)
    
    # ========================================================================
    # MODE 3: DOCUMENT_OVERVIEW - Full dataset breakdown or Executive Summary
    # ========================================================================
    if mode == QueryMode.DOCUMENT_OVERVIEW:
        return handle_document_overview(dataframe, doc_hash, user_query)
    
    # ========================================================================
    # MODE 4: DATA_QUERY - Full RAG pipeline with visualizations
    # ========================================================================
    
    # Detect question parameters
    detail_mode = detect_detail_mode(user_query)
    specific_metrics = get_requested_metrics(user_query)
    
    # Get target columns for visualization filtering
    target_columns = []
    if specific_metrics and dataframe is not None:
        target_columns = get_target_columns(specific_metrics, dataframe.columns.tolist())
    
    # ========================================================================
    # STEP 2: RETRIEVE CONTEXT FROM VECTOR STORE
    # ========================================================================
    store = get_vector_store()
    
    # Adjust k based on detail mode
    if detail_mode == "detailed":
        k = max(k, 15)
    elif detail_mode == "brief":
        k = min(k, 5)
    
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
    
    # Try to get DataFrame if not provided
    df = dataframe
    if df is None and filename:
        df = get_dataframe_for_doc(doc_hash, filename)
    
    # Build context string
    context = build_structured_context(results, doc_hash, max_chars=MAX_CONTEXT_CHARS)
    
    # ========================================================================
    # STEP 3: COMPUTE STATISTICS (Python-based, NOT LLM)
    # ========================================================================
    stats_block = ""
    if df is not None and not df.empty:
        stats_block = compute_data_statistics(df, specific_metrics, target_columns)
        # Prepend stats to context
        context = stats_block + "\n\n---\n\n" + context
    
    # ========================================================================
    # STEP 4: GENERATE LLM ANSWER
    # ========================================================================
    is_detailed = (detail_mode == "detailed")
    
    if is_detailed:
        system_prompt = SYSTEM_PROMPT_DETAILED
        max_tokens = 3000
    else:
        system_prompt = SYSTEM_PROMPT_CONCISE
        max_tokens = 1500
    
    # Build user prompt
    metric_focus = ""
    if specific_metrics:
        metric_focus = f"\n\n**FOCUS ON:** {', '.join(specific_metrics).upper()} metrics only."
    
    # Check if we have actual statistics
    has_stats = stats_block and len(stats_block) > 100
    
    if has_stats:
        user_prompt = f"""Answer this data question using the COMPUTED STATISTICS below:
{user_query}
{metric_focus}

=== PRE-COMPUTED STATISTICS (USE THESE VALUES) ===
{stats_block}

=== ADDITIONAL CONTEXT ===
{context[:4000]}

IMPORTANT RULES:
1. The statistics above were computed from the actual dataset - USE THEM
2. Answer with the EXACT values shown in the statistics
3. For comparative analysis, show both metrics side by side
4. Include totals, averages, and any relevant comparisons
5. DO NOT say "data not available" if statistics are shown above"""
    else:
        user_prompt = f"""Answer this data question:
{user_query}
{metric_focus}

DATA CONTEXT:
{context[:6000]}

RULES:
1. Use EXACT values from the context above
2. DO NOT make up numbers or estimate
3. Include summary, key stats, and interpretation
4. If specific metric requested, focus ONLY on that metric"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    answer = call_llm(messages, max_tokens=max_tokens)
    
    # ========================================================================
    # STEP 5: FORMAT SOURCES
    # ========================================================================
    sources = []
    for r in results[:5]:
        sources.append({
            "filename": r.get("filename", "unknown"),
            "doc_hash": r.get("doc_hash", ""),
            "score": r.get("score", 0),
            "preview": r.get("text", "")[:300] + "..." if len(r.get("text", "")) > 300 else r.get("text", ""),
            "type": r.get("type", "text")
        })
    
    # ========================================================================
    # STEP 6: RETURN RESPONSE
    # ========================================================================
    return {
        "answer": answer,
        "sources": sources,
        "intent": "data_query",
        "query_mode": "data_query",
        "detail_mode": detail_mode,
        "specific_metrics": specific_metrics,
        "target_columns": target_columns,
        "show_visualizations": True,  # ALWAYS True for DATA_QUERY
        "num_chunks": len(results),
        "query_type": "data"
    }


# ============================================================================
# INTENT HANDLERS - REFACTORED FOR CONCISE ANSWERS
# ============================================================================

def _handle_simple_metric_question(query: str, df: pd.DataFrame, metrics: List[str]) -> Optional[str]:
    """
    Handle simple metric questions with direct Python computation.
    Returns a SHORT, FOCUSED answer about just the asked metric.
    
    Example: "What is oil production?" -> Returns only oil production stats
    """
    if df is None or df.empty or not metrics:
        return None
    
    result_lines = []
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Map metrics to column patterns
    metric_patterns = {
        'oil': ['OIL'],
        'gas': ['GAS'],
        'water': ['WAT', 'WATER'],
        'condensate': ['COND'],
        'energy': ['ENERGY', 'BTU'],
        'injection': ['INJ'],
        'sales': ['SALES'],
        'production': ['PROD']
    }
    
    # Find matching columns - prioritize volume columns
    matched_cols = []
    for metric in metrics:
        patterns = metric_patterns.get(metric, [metric.upper()])
        for col in numeric_cols:
            col_upper = col.upper()
            # Prioritize volume columns
            if any(p in col_upper for p in patterns) and 'VOL' in col_upper:
                if col not in matched_cols:
                    matched_cols.append(col)
    
    # If no volume columns, try any matching columns
    if not matched_cols:
        for metric in metrics:
            patterns = metric_patterns.get(metric, [metric.upper()])
            for col in numeric_cols:
                col_upper = col.upper()
                if any(p in col_upper for p in patterns):
                    if col not in matched_cols and len(matched_cols) < 3:
                        matched_cols.append(col)
    
    if not matched_cols:
        return None
    
    # Build concise answer
    metric_name = metrics[0].title() if metrics else "Data"
    result_lines.append(f"## {metric_name} Summary\n")
    
    for col in matched_cols[:3]:  # Max 3 columns
        total = df[col].dropna().sum()
        count = df[col].notna().sum()
        avg = df[col].dropna().mean() if count > 0 else 0
        
        # Get unit
        uom = ""
        if col + "_UOM" in df.columns:
            uom_vals = df[col + "_UOM"].dropna()
            uom = f" {uom_vals.iloc[0]}" if len(uom_vals) > 0 else ""
        
        result_lines.append(f"**{col}**:")
        result_lines.append(f"- Total: {total:,.2f}{uom}")
        result_lines.append(f"- Daily Average: {avg:,.2f}{uom}")
        result_lines.append(f"- Records: {count:,}\n")
    
    # Add brief explanation
    if 'oil' in metrics:
        result_lines.append("*Oil production represents crude oil extracted from wells, measured in barrels (bbl).*")
    elif 'gas' in metrics:
        result_lines.append("*Gas production represents natural gas volume, typically measured in MMcf (million cubic feet).*")
    elif 'water' in metrics:
        result_lines.append("*Water production represents produced water from wells, measured in barrels (bbl).*")
    
    return "\n".join(result_lines)


def _handle_show_intent(query: str, context: str, doc_hash: Optional[str], detail_mode: str = "normal") -> str:
    """Handle 'show data' type queries - return formatted data summary."""
    is_detailed = (detail_mode == "detailed")
    max_tokens = 2500 if is_detailed else 1000
    context = context[:MAX_CONTEXT_CHARS_DETAILED if is_detailed else MAX_CONTEXT_CHARS]
    
    if is_detailed:
        prompt = f"""Show comprehensive data overview.

QUERY: {query}

DATA:
{context}

Provide:
1. Full overview
2. Sample table (key columns, up to 15 rows)
3. Complete statistics
4. Data quality notes"""
    else:
        prompt = f"""Show data briefly.

QUERY: {query}

DATA:
{context}

Provide ONLY:
1. 2-sentence overview
2. Small sample table (5 columns, 5 rows max)
Keep it short."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_ANALYSIS if is_detailed else SYSTEM_PROMPT_CONCISE},
        {"role": "user", "content": prompt}
    ]
    
    return call_llm(messages, max_tokens=max_tokens)


def _handle_calculate_intent(
    query: str, 
    context: str, 
    doc_hash: Optional[str], 
    df: Optional[pd.DataFrame] = None,
    detail_mode: str = "normal",
    specific_metrics: List[str] = None
) -> str:
    """Handle calculation queries - Python computes, LLM explains."""
    
    # If we have the DataFrame, compute the answer directly with Python
    if df is not None and not df.empty:
        # Initialize result lines based on detail mode
        is_detailed = (detail_mode == "detailed")
        if is_detailed:
            result_lines = ["## Calculation Results (Detailed Analysis)\n"]
        else:
            result_lines = ["## Results\n"]
        
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
        
        # Use specific_metrics if provided, otherwise detect from query
        if specific_metrics:
            requested_cols = []
            metric_patterns = {
                'oil': ['OIL'],
                'gas': ['GAS'],
                'water': ['WAT', 'WATER'],
                'condensate': ['COND'],
                'energy': ['ENERGY', 'BTU'],
                'injection': ['INJ'],
                'sales': ['SALES'],
                'production': ['PROD']
            }
            for metric in specific_metrics:
                patterns = metric_patterns.get(metric, [metric.upper()])
                for col in numeric_cols:
                    col_upper = col.upper()
                    if any(p in col_upper for p in patterns) and 'VOL' in col_upper:
                        if col not in requested_cols:
                            requested_cols.append(col)
        else:
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
                    if 'PROD' in col_upper:
                        requested_cols.append(col)
                elif wants_gas and 'GAS' in col_upper and 'VOL' in col_upper:
                    if wants_sales and 'SALES' in col_upper:
                        requested_cols.append(col)
                    elif 'PROD' in col_upper:
                        requested_cols.append(col)
                elif wants_water and ('WAT' in col_upper or 'WATER' in col_upper) and 'VOL' in col_upper:
                    requested_cols.append(col)
            
            # If no specific match, use main production volumes
            if not requested_cols:
                main_cols = ['PROD_OIL_VOL', 'PROD_GAS_VOL', 'PROD_WAT_VOL', 'SALES_OIL_VOL', 'SALES_GAS_VOL']
                requested_cols = [c for c in main_cols if c in numeric_cols]
        
        # Remove duplicates and limit based on detail_mode
        requested_cols = list(dict.fromkeys(requested_cols))
        max_cols = 8 if detail_mode else 3
        requested_cols = requested_cols[:max_cols]
        
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


def _handle_explain_intent(
    query: str, 
    context: str, 
    df: Optional[pd.DataFrame] = None,
    detail_mode: str = "normal",
    specific_metrics: List[str] = None
) -> str:
    """Handle 'explain/describe' queries - explain what the data contains."""
    
    # For simple questions about specific metrics, give SHORT answers
    is_detailed = (detail_mode == "detailed")
    if not is_detailed and specific_metrics and len(specific_metrics) <= 2:
        if df is not None and not df.empty:
            # Use the simple metric handler
            simple_answer = _handle_simple_metric_question(query, df, specific_metrics)
            if simple_answer:
                return simple_answer
    
    # For detailed mode or complex questions, build comprehensive stats
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


def _handle_summarize_intent(
    query: str, 
    context: str, 
    df: Optional[pd.DataFrame] = None,
    detail_mode: str = "detailed"  # Summarize defaults to detailed
) -> str:
    """Handle summarization queries - comprehensive document summary."""
    
    # Build stats from DataFrame for better summary
    is_detailed = (detail_mode == "detailed")
    stats_section = ""
    if df is not None and not df.empty:
        stats_lines = []
        stats_lines.append(f"📊 **Dataset Overview**: {len(df):,} records, {len(df.columns)} columns")
        
        # Get date range if available
        date_cols = [c for c in df.columns if 'DATE' in c.upper() or 'TIME' in c.upper()]
        for col in date_cols[:1]:  # Just first date column
            try:
                dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(dates) > 0:
                    stats_lines.append(f"📅 **Date Range**: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
                    break
            except:
                pass
        
        # Key metrics - limit based on detail mode
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        prod_cols = [c for c in numeric_cols if 'PROD' in c.upper() and 'VOL' in c.upper()]
        sales_cols = [c for c in numeric_cols if 'SALES' in c.upper() and 'VOL' in c.upper()]
        
        max_metrics = 6 if detail_mode else 3
        if prod_cols or sales_cols:
            stats_lines.append("\n**Key Metrics:**")
            for col in (prod_cols + sales_cols)[:max_metrics]:
                total = df[col].dropna().sum()
                count = df[col].notna().sum()
                null_count = df[col].isna().sum()
                if count > 0:
                    uom = ""
                    if col + "_UOM" in df.columns:
                        uom_vals = df[col + "_UOM"].dropna()
                        uom = f" {uom_vals.iloc[0]}" if len(uom_vals) > 0 else ""
                    stats_lines.append(f"  - {col}: Total {total:,.2f}{uom} (data in {count:,} records, {null_count:,} empty)")
        
        # Categories - only in detail mode
        is_detailed = (detail_mode == "detailed")
        if is_detailed:
            cat_cols = ['ITEM_TYPE', 'FIELD_NAME', 'ITEM_NAME', 'SOURCE']
            for col in cat_cols:
                if col in df.columns:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) <= 5:
                        stats_lines.append(f"  - {col}: {', '.join(str(v) for v in unique_vals[:5])}")
                    else:
                        stats_lines.append(f"  - {col}: {len(unique_vals)} unique values")
        
        stats_section = "\n".join(stats_lines)
    
    max_context = MAX_CONTEXT_CHARS_DETAILED if is_detailed else MAX_CONTEXT_CHARS
    context = context[:max_context]
    max_tokens = 2500 if is_detailed else 1000
    
    if is_detailed:
        system_prompt = """You are a document analyst. Provide a comprehensive summary of the document/dataset.

Structure your response as:
## 📋 Document Summary
Brief overview of what this document contains.

## 📊 Key Statistics
Important numbers and metrics from the data.

## 🔍 Key Findings
3-5 important observations about the data.

## 📝 Notes
Any missing data, anomalies, or important context."""
    else:
        system_prompt = """You are a document analyst. Provide a BRIEF summary.

Keep it SHORT:
- 2-3 sentence overview
- 3-4 key metrics
- 2-3 key findings
No lengthy sections."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""Summarize this document/dataset.

USER REQUEST: {query}

PRE-COMPUTED STATISTICS:
{stats_section}

DATA CONTEXT:
{context}

{"Provide a comprehensive summary." if detail_mode else "Keep it brief - max 150 words."}"""}
    ]
    
    return call_llm(messages, max_tokens=max_tokens)


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


def _handle_general_intent(
    query: str, 
    context: str,
    detail_mode: str = "normal",
    specific_metrics: List[str] = None
) -> str:
    """Handle general queries with flexible response - CONCISE by default."""
    
    # Use smaller context for simple questions
    is_detailed = (detail_mode == "detailed")
    max_context = MAX_CONTEXT_CHARS_DETAILED if is_detailed else 3000
    context = context[:max_context]
    
    if is_detailed:
        system_prompt = SYSTEM_PROMPT_DETAILED
        user_prompt = f"""Provide a detailed answer to this question.

Question: {query}

DATA:
{context}

Give a comprehensive response with all relevant details."""
        max_tokens = 2500
    else:
        system_prompt = SYSTEM_PROMPT_CONCISE
        # For simple questions, be very direct
        if specific_metrics:
            metric_hint = f"Focus ONLY on: {', '.join(specific_metrics)}"
        else:
            metric_hint = "Answer briefly and directly"
        
        user_prompt = f"""Answer this question BRIEFLY (2-4 sentences max).

Question: {query}

{metric_hint}

DATA:
{context}

Give a SHORT, DIRECT answer. Do NOT provide full dataset analysis."""
        max_tokens = 500
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return call_llm(messages, max_tokens=max_tokens)


# ============================================================================
# DOCUMENT ANALYSIS FUNCTIONS
# ============================================================================

def summarize_document(doc_hash: str, dataframe: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Generate comprehensive document summary using Python-computed statistics.
    
    If a DataFrame is provided, generates a detailed executive summary.
    Otherwise falls back to LLM-based summary from chunks.
    
    Returns:
        Dict with summary, statistics, sample data, and metadata
    """
    store = get_vector_store()
    
    # Get document info first
    docs = store.get_all_documents()
    doc_info = next((d for d in docs if d.get("doc_hash") == doc_hash), {})
    filename = doc_info.get("filename", "unknown")
    
    # ========================================================================
    # IF DATAFRAME PROVIDED - Generate Executive Summary with Python
    # ========================================================================
    if dataframe is not None and not dataframe.empty:
        df = dataframe
        summary_parts = []
        
        # ====================================================================
        # HEADER
        # ====================================================================
        summary_parts.append(f"## 📋 Executive Summary: {filename}\n")
        
        # ====================================================================
        # 1. DOCUMENT OVERVIEW
        # ====================================================================
        summary_parts.append("### 📄 Document Overview\n")
        
        # Detect date range
        date_range_str = ""
        date_cols = [c for c in df.columns if 'DATE' in c.upper() or 'TIME' in c.upper()]
        for col in date_cols[:1]:
            try:
                dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(dates) > 0:
                    start_date = dates.min().strftime('%B %d, %Y')
                    end_date = dates.max().strftime('%B %d, %Y')
                    duration_days = (dates.max() - dates.min()).days
                    date_range_str = f"covering **{start_date}** to **{end_date}** ({duration_days} days)"
                    break
            except:
                pass
        
        # Document type detection
        doc_type = "Production Data" if any('PROD' in c.upper() for c in df.columns) else "Dataset"
        
        summary_parts.append(
            f"**Type:** {doc_type}  \n"
            f"**Size:** {len(df):,} records × {len(df.columns)} fields  \n"
            f"**Period:** {date_range_str if date_range_str else 'Not specified'}  \n"
            f"**Memory:** {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB\n"
        )
        
        # ====================================================================
        # 2. KEY METRICS SUMMARY
        # ====================================================================
        summary_parts.append("### 📊 Key Metrics Summary\n")
        
        # Find all production and sales columns
        prod_cols = [c for c in df.columns if 'PROD' in c.upper() and 'VOL' in c.upper()]
        sales_cols = [c for c in df.columns if 'SALES' in c.upper() and 'VOL' in c.upper()]
        
        if prod_cols or sales_cols:
            summary_parts.append("| Metric | Total | Average | Min | Max | Unit |")
            summary_parts.append("|--------|-------|---------|-----|-----|------|")
            
            all_metric_cols = prod_cols[:6] + sales_cols[:4]  # Limit columns
            for col in all_metric_cols:
                try:
                    data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(data) > 0 and data.sum() > 0:
                        total = data.sum()
                        avg = data.mean()
                        min_val = data.min()
                        max_val = data.max()
                        
                        # Get unit
                        uom = ""
                        if col + "_UOM" in df.columns:
                            uom_vals = df[col + "_UOM"].dropna()
                            uom = uom_vals.iloc[0] if len(uom_vals) > 0 else ""
                        
                        # Clean name
                        display_name = col.replace("_VOL", "").replace("_", " ").title()
                        summary_parts.append(
                            f"| {display_name} | {total:,.2f} | {avg:,.2f} | {min_val:,.2f} | {max_val:,.2f} | {uom} |"
                        )
                except:
                    pass
            summary_parts.append("")
        
        # ====================================================================
        # 3. TOP PERFORMERS (if ITEM_NAME exists)
        # ====================================================================
        if 'ITEM_NAME' in df.columns and prod_cols:
            summary_parts.append("### 🏆 Top 5 Performers\n")
            
            # Find main production column
            main_col = prod_cols[0]
            try:
                top_items = df.groupby('ITEM_NAME')[main_col].sum().nlargest(5)
                if len(top_items) > 0:
                    display_name = main_col.replace("_VOL", "").replace("_", " ").title()
                    summary_parts.append(f"**By {display_name}:**\n")
                    summary_parts.append("| Rank | Item | Total Volume |")
                    summary_parts.append("|------|------|--------------|")
                    for rank, (item, value) in enumerate(top_items.items(), 1):
                        summary_parts.append(f"| {rank} | {item} | {value:,.2f} |")
                    summary_parts.append("")
            except:
                pass
        
        # ====================================================================
        # 4. DATA QUALITY ASSESSMENT
        # ====================================================================
        summary_parts.append("### ✅ Data Quality Assessment\n")
        
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
        
        # Count empty columns
        empty_cols = len([c for c in df.columns if df[c].isna().all()])
        
        # Count duplicates
        dup_count = df.duplicated().sum()
        
        summary_parts.append(f"| Quality Metric | Status |")
        summary_parts.append(f"|----------------|--------|")
        summary_parts.append(f"| Data Completeness | {completeness:.1f}% |")
        summary_parts.append(f"| Empty Columns | {empty_cols} of {len(df.columns)} |")
        summary_parts.append(f"| Duplicate Records | {dup_count:,} |")
        summary_parts.append(f"| Total Missing Values | {missing_cells:,} |")
        summary_parts.append("")
        
        # ====================================================================
        # 5. KEY INSIGHTS
        # ====================================================================
        summary_parts.append("### 💡 Key Insights\n")
        
        insights = []
        
        # Insight 1: Largest metric
        if prod_cols:
            max_col = None
            max_val = 0
            for col in prod_cols:
                try:
                    val = pd.to_numeric(df[col], errors='coerce').sum()
                    if val > max_val:
                        max_val = val
                        max_col = col
                except:
                    pass
            if max_col:
                display_name = max_col.replace("PROD_", "").replace("_VOL", "").replace("_", " ").title()
                insights.append(f"- **Highest Volume:** {display_name} leads with {max_val:,.2f} total units")
        
        # Insight 2: Sales efficiency (if both prod and sales exist)
        if prod_cols and sales_cols:
            try:
                prod_total = pd.to_numeric(df[prod_cols[0]], errors='coerce').sum()
                sales_total = pd.to_numeric(df[sales_cols[0]], errors='coerce').sum()
                if prod_total > 0:
                    efficiency = (sales_total / prod_total) * 100
                    insights.append(f"- **Sales Efficiency:** {efficiency:.1f}% of production was sold")
            except:
                pass
        
        # Insight 3: Data coverage
        if date_range_str:
            insights.append(f"- **Coverage:** Data spans {date_range_str}")
        
        # Insight 4: Unique items
        if 'ITEM_NAME' in df.columns:
            unique_items = df['ITEM_NAME'].nunique()
            insights.append(f"- **Scope:** {unique_items:,} unique items/wells tracked")
        
        # Insight 5: Data freshness
        if 'LAST_UPDT_DATE' in df.columns:
            try:
                last_update = pd.to_datetime(df['LAST_UPDT_DATE'], errors='coerce').max()
                if pd.notna(last_update):
                    insights.append(f"- **Last Updated:** {last_update.strftime('%B %d, %Y')}")
            except:
                pass
        
        if insights:
            summary_parts.extend(insights)
        else:
            summary_parts.append("- Dataset loaded successfully with all metrics computed")
        
        summary_parts.append("")
        summary_parts.append("---")
        summary_parts.append("*For detailed analysis, use the Query tab to ask specific questions about this data.*")
        
        summary = "\n".join(summary_parts)
        
        return {
            "status": "success",
            "summary": summary,
            "filename": filename,
            "num_chunks": len(store.get_document_chunks(doc_hash)),
            "num_tables": len(store.get_document_tables(doc_hash)),
            "is_dataset": True
        }
    
    # ========================================================================
    # FALLBACK: LLM-based summary from chunks (when no DataFrame)
    # ========================================================================
    # Get all document content
    full_text = store.get_full_document_text(doc_hash)
    tables = store.get_document_tables(doc_hash)
    chunks = store.get_document_chunks(doc_hash)
    
    if not full_text and not tables:
        return {
            "status": "error",
            "message": "Document not found or empty"
        }
    
    # Build context prioritizing tables - limit size for speed
    if tables:
        context = "\n\n---\n\n".join(t.get("text", "")[:3000] for t in tables[:2])
    else:
        context = full_text[:8000]
    
    # Generate summary with LLM - improved prompt
    messages = [
        {"role": "system", "content": """You are a data analyst creating executive summaries.
Create a well-formatted executive summary with these sections:
1. Document Overview (type, size, date range)
2. Key Metrics (with actual numbers from the data)
3. Top Performers (if applicable)
4. Data Quality Notes
5. Key Insights (3-5 actionable findings)

Use markdown formatting with tables where appropriate."""},
        {"role": "user", "content": f"""Create an executive summary for: {filename}

DATA SAMPLE:
{context}

Generate a comprehensive executive summary with actual statistics from the data."""}
    ]
    
    summary = call_llm(messages, max_tokens=2500)
    
    return {
        "status": "success",
        "summary": summary,
        "filename": filename,
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
