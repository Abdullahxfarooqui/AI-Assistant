"""
Query Classification and Routing (4-MODE SYSTEM)

STRICT INTENT ROUTING ENGINE - Every user query MUST be classified into EXACTLY ONE mode:

1. DATA_QUERY - Any query about dataset columns, metrics, statistics, charts, visualizations
   - ALWAYS triggers: query_data() + generate_visuals()
   - MUST return: summary + stats + visuals + interpretation

2. DOCUMENT_OVERVIEW - "What's in this document?", "Describe the file", "Dataset overview"
   - ALWAYS triggers: dataset_overview()
   - MUST return: rows, columns, categories, missing values, date ranges, groups

3. FREEFORM_QUERY - General chat UNRELATED to dataset (ice cream, jokes, poems)
   - ALWAYS triggers: conversational_response()
   - MUST NOT reference dataset at all

4. SYSTEM_TASK - "Fix this", "Bug", "Error", "Improve the engine"
   - ALWAYS triggers: system_instruction()
   - MUST return: code fixes, architecture improvements

INTENT PRIORITY:
1. If ANY production/sales/gas/oil column appears → ALWAYS DATA_QUERY
2. If query looks like "what is in this document" → DOCUMENT_OVERVIEW
3. If query is general (ice cream, jokes) → FREEFORM_QUERY
4. If query mentions bugs/errors → SYSTEM_TASK
"""

from typing import Tuple, Dict, Any, List, Optional
import re


# ============================================================================
# QUERY MODE ENUM
# ============================================================================

class QueryMode:
    DATA_QUERY = "data_query"           # Questions about the dataset - ALWAYS show charts
    DOCUMENT_OVERVIEW = "document_overview"  # Full document breakdown - detailed overview
    FREEFORM_QUERY = "freeform_query"   # General chat - REFUSE to answer
    SYSTEM_TASK = "system_task"         # Modify the RAG engine - provide guidance


# ============================================================================
# KEYWORD SETS FOR CLASSIFICATION (PRIORITY ORDER)
# ============================================================================

# DATA_QUERY keywords - HIGHEST PRIORITY for dataset questions
# If ANY of these appear, it's a DATA_QUERY
DATA_KEYWORDS = {
    # Core Oil & Gas Metrics (ALWAYS DATA_QUERY)
    "oil", "gas", "water", "condensate", "lpg", "ngl", "crude",
    "production", "prod", "sales", "injection", "inject", "inj",
    "bbl", "barrel", "mcf", "mmbtu", "volume", "vol",
    "rate", "pressure", "temp", "temperature", "density",
    "flare", "fuel", "energy", "btu", "heat",
    
    # Wells, Fields, Assets
    "well", "wells", "field", "fields", "battery", "facility", "asset", 
    "completion", "completions", "reservoir",
    
    # Visualization Requests (ALWAYS DATA_QUERY)
    "plot", "chart", "visualize", "graph", "display", "show me", 
    "create chart", "draw", "render", "visualization",
    "trend", "trends", "compare", "comparison", "analyze", "analysis",
    "statistics", "stats", "metric", "metrics",
    
    # Data Exploration
    "column", "columns", "schema", "rows", "records", "table",
    "data quality", "missing values", "completeness",
    "total", "average", "avg", "sum", "max", "min", "count", "mean",
    "median", "std", "deviation", "variance",
    
    # Time References (in data context)
    "monthly", "yearly", "daily", "weekly", "quarterly",
    "by month", "by year", "by date", "over time", "time series",
}

# DOCUMENT_OVERVIEW keywords - Exact phrase matches for document overview
OVERVIEW_KEYWORDS = {
    # Primary overview phrases
    "what's in this document",
    "what is in this document",
    "what's in the document",
    "what is in the document",
    "what's in this file",
    "what is in this file",
    "what does this document contain",
    "what does the document contain",
    "what does this file contain",
    
    # Describe/Explain variants
    "describe this document",
    "describe the document",
    "describe this file",
    "describe the file",
    "describe this dataset",
    "describe the dataset",
    "explain this document",
    "explain the document",
    "explain this file",
    "explain this dataset",
    
    # Overview variants
    "document overview",
    "dataset overview",
    "file overview",
    "data overview",
    "overview of the data",
    "overview of this data",
    "overview of the document",
    "overview of the file",
    
    # Tell me about
    "tell me about this document",
    "tell me about the document",
    "tell me about this file",
    "tell me about this dataset",
    "tell me what this file contains",
    "tell me what the file contains",
    "tell me in detail what's inside",
    
    # Structure/Info
    "structure of this",
    "structure of the",
    "what data is in",
    "what information is in",
    "summarize the document",
    "summarize this document",
    
    # EXECUTIVE SUMMARY variants (NEW)
    "executive summary",
    "give me an executive summary",
    "give me executive summary",
    "provide executive summary",
    "provide an executive summary",
    "generate executive summary",
    "create executive summary",
    "show executive summary",
    "executive summary of this",
    "executive summary of the",
    "summary of this document",
    "summary of the document",
    "summary of this file",
    "summary of this dataset",
    "give me a summary",
    "provide a summary",
    "high level summary",
    "high-level summary",
    "brief summary",
    "quick summary",
    "full summary",
    "complete summary",
    "comprehensive summary",
}

# SYSTEM_TASK keywords - Bug fixes, engine modifications
SYSTEM_KEYWORDS = {
    # Fix/Bug keywords
    "fix this",
    "fix the bug",
    "fix this bug",
    "fix the error",
    "fix this error",
    "fix the issue",
    "fix this issue",
    "solve this issue",
    "solve the issue",
    "debug",
    "debugging",
    
    # Improvement keywords
    "improve the engine",
    "improve the system",
    "improve the rag",
    "make it detailed",
    "make it more detailed",
    "add more detail",
    "enhance the",
    
    # Modification keywords
    "modify the engine",
    "modify the rag",
    "modify the system",
    "change the architecture",
    "change the model",
    "change how you respond",
    "update the code",
    "update the prompt",
    "fix the prompt",
    "fix the intent",
    
    # Code/Implementation keywords
    "implement this",
    "add a feature",
    "refactor",
    "rewrite the",
}

# FREEFORM keywords - General chat NOT about data (STRICT MATCHING)
# If query matches these AND has NO data keywords, it's FREEFORM
FREEFORM_PATTERNS = [
    # Food/Recipes
    r"\bhow to eat\b", r"\bhow to cook\b", r"\brecipe\b", r"\bfood\b",
    r"\bice cream\b", r"\bcake\b", r"\bpizza\b", r"\bcooking\b",
    
    # Entertainment
    r"\bjoke\b", r"\bjokes\b", r"\btell me a joke\b", r"\bfunny\b",
    r"\bpoem\b", r"\bstory\b", r"\bsing\b", r"\bsong\b", r"\bmovie\b",
    r"\bgame\b", r"\bsports\b",
    
    # General Knowledge
    r"\bwho is\b", r"\bwho was\b", r"\bhistory of\b", r"\bgeography\b",
    r"\bdefine\b", r"\bmeaning of\b", r"\bwhat does .+ mean\b",
    r"\bscience\b", r"\bphysics\b", r"\bbiology\b", r"\bchemistry\b",
    
    # Personal/Casual
    r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgood morning\b", r"\bgood evening\b",
    r"\bthank you\b", r"\bthanks\b", r"\bbye\b", r"\bgoodbye\b",
    r"\bhow are you\b", r"\bwhat's your name\b", r"\bwho are you\b",
    
    # Life advice (not data)
    r"\badvice\b", r"\bhelp me with\b", r"\brelationship\b", r"\blove\b",
    r"\bweather\b", r"\btravel\b", r"\bvacation\b",
]


# ============================================================================
# CORE INTENT DETECTION FUNCTION (REWRITTEN)
# ============================================================================

def detect_intent(query: str, df_columns: List[str] = None) -> str:
    """
    STRICT Intent Detection - Routes query to exactly ONE of 4 modes.
    
    PRIORITY ORDER:
    1. FREEFORM_QUERY - Check first to filter out non-data questions
    2. SYSTEM_TASK - Check for bug/fix/modify requests
    3. DOCUMENT_OVERVIEW - Check for document structure questions
    4. DATA_QUERY - DEFAULT for anything data-related
    
    Args:
        query: User's query string
        df_columns: Optional list of DataFrame column names
        
    Returns:
        Intent string: "freeform", "system", "overview", or "data"
    """
    q = query.lower().strip()
    
    # ========================================================================
    # PRIORITY 1: FREEFORM queries (NOT about data)
    # ========================================================================
    # Check if query matches freeform patterns AND has NO data keywords
    has_data_keyword = any(kw in q for kw in DATA_KEYWORDS)
    
    if not has_data_keyword:
        # Check freeform patterns
        for pattern in FREEFORM_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                return "freeform"
    
    # ========================================================================
    # PRIORITY 2: SYSTEM_TASK (bug fixes, engine modifications)
    # ========================================================================
    for phrase in SYSTEM_KEYWORDS:
        if phrase in q:
            return "system"
    
    # ========================================================================
    # PRIORITY 3: DOCUMENT_OVERVIEW (what's in this document)
    # ========================================================================
    for phrase in OVERVIEW_KEYWORDS:
        if phrase in q:
            return "overview"
    
    # ========================================================================
    # PRIORITY 4: DATA_QUERY (DEFAULT for any data-related question)
    # ========================================================================
    # If ANY column name appears in query, it's DATA
    if df_columns:
        for col in df_columns:
            col_lower = col.lower()
            if col_lower in q or col_lower.replace("_", " ") in q:
                return "data"
    
    # If ANY data keyword appears, it's DATA
    if has_data_keyword:
        return "data"
    
    # ========================================================================
    # FALLBACK: Check if it looks like a question about documents
    # ========================================================================
    # If user is asking about something and we have data, assume DATA_QUERY
    question_starters = ["what", "how", "show", "tell", "give", "list", "calculate"]
    if any(q.startswith(word) for word in question_starters):
        return "data"
    
    # Final fallback - if we can't determine, treat as FREEFORM (safer)
    return "freeform"


# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_query_mode(query: str, df_columns: List[str] = None) -> Tuple[str, float, str]:
    """
    Classify a query into one of four modes with confidence scoring.
    
    Args:
        query: The user's query string
        df_columns: Optional list of DataFrame column names
        
    Returns:
        Tuple of (mode, confidence, reason)
        - mode: QueryMode constant
        - confidence: 0.0 to 1.0
        - reason: Human-readable explanation
    """
    q_lower = query.lower().strip()
    
    # Use the core detect_intent function
    intent = detect_intent(query, df_columns)
    
    # Map intent to QueryMode with confidence and reason
    if intent == "freeform":
        return (
            QueryMode.FREEFORM_QUERY,
            0.90,
            "Query is general chat unrelated to dataset"
        )
    
    elif intent == "system":
        return (
            QueryMode.SYSTEM_TASK,
            0.90,
            "User requested system modification or bug fix"
        )
    
    elif intent == "overview":
        return (
            QueryMode.DOCUMENT_OVERVIEW,
            0.95,
            "User requested full document/dataset overview"
        )
    
    else:  # "data"
        # Count how many data keywords match for confidence
        data_matches = sum(1 for kw in DATA_KEYWORDS if kw in q_lower)
        confidence = min(0.6 + (data_matches * 0.05), 0.98)
        return (
            QueryMode.DATA_QUERY,
            confidence,
            f"Query contains {data_matches} data-related keyword(s)"
        )


def get_query_mode(query: str, df_columns: List[str] = None) -> Dict[str, Any]:
    """
    Get detailed classification for a query with tool recommendations.
    
    Returns:
        Dict with:
        - mode: QueryMode constant
        - confidence: 0.0-1.0
        - reason: Human-readable explanation
        - tools: List of tools to use for this mode
        - should_use_rag: Boolean for backward compatibility
    """
    mode, confidence, reason = classify_query_mode(query, df_columns)
    
    # Define which tools to use for each mode (as per requirements)
    tools_map = {
        QueryMode.DATA_QUERY: ["query_data", "generate_visuals"],  # ALWAYS both
        QueryMode.DOCUMENT_OVERVIEW: ["dataset_overview"],
        QueryMode.FREEFORM_QUERY: ["conversational_response"],
        QueryMode.SYSTEM_TASK: ["system_instruction"],
    }
    
    return {
        "mode": mode,
        "confidence": confidence,
        "reason": reason,
        "tools": tools_map.get(mode, []),
        "should_use_rag": mode in [QueryMode.DATA_QUERY, QueryMode.DOCUMENT_OVERVIEW],
        "show_visualizations": mode == QueryMode.DATA_QUERY  # ONLY data queries show charts
    }


# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================

def classify_query(query: str) -> Tuple[str, float]:
    """
    Legacy function for backward compatibility.
    
    Returns:
        Tuple of (classification, confidence)
        - classification: 'data' or 'general'
        - confidence: 0.0 to 1.0
    """
    mode, confidence, _ = classify_query_mode(query)
    
    if mode in [QueryMode.DATA_QUERY, QueryMode.DOCUMENT_OVERVIEW]:
        return "data", confidence
    else:
        return "general", confidence


def should_use_rag(query: str) -> bool:
    """
    STRICT check: Should this query use the RAG pipeline?
    
    Returns True ONLY for DATA_QUERY and DOCUMENT_OVERVIEW.
    Returns False for FREEFORM_QUERY and SYSTEM_TASK.
    """
    mode, _, _ = classify_query_mode(query)
    return mode in [QueryMode.DATA_QUERY, QueryMode.DOCUMENT_OVERVIEW]


def get_query_classification(query: str) -> dict:
    """Legacy function - returns detailed classification."""
    return get_query_mode(query)


# ============================================================================
# METRIC DETECTION FUNCTIONS
# ============================================================================

def is_visualization_query(query: str) -> bool:
    """Check if the query explicitly requests visualization."""
    q_lower = query.lower()
    viz_keywords = {
        "plot", "chart", "visualize", "graph", "show me", 
        "create chart", "trend", "draw", "render", "display"
    }
    return any(kw in q_lower for kw in viz_keywords)


def get_requested_metrics(query: str) -> List[str]:
    """
    Extract which specific metrics the user is asking about.
    
    Returns list of metrics: ["oil", "gas", "water", etc.]
    If empty, the query is about general data (not specific resource).
    """
    q_lower = query.lower()
    metrics = []
    
    # Specific resource detection
    metric_map = {
        "oil": ["oil", "crude"],
        "gas": ["gas", "natural gas", "mcf", "mmcf", "fuel gas", "flare"],
        "water": ["water", "wat"],
        "condensate": ["condensate", "cond"],
        "lpg": ["lpg"],
        "ngl": ["ngl", "natural gas liquid"],
        "energy": ["energy", "btu", "mmbtu", "heat"],
        "injection": ["injection", "inject", "inj"],
    }
    
    for metric, keywords in metric_map.items():
        if any(kw in q_lower for kw in keywords):
            metrics.append(metric)
    
    # Also detect production/sales modifiers
    if "production" in q_lower or "prod" in q_lower or "produced" in q_lower:
        if "production" not in metrics:
            metrics.append("production")
    
    if "sales" in q_lower or "sold" in q_lower or "sale" in q_lower:
        if "sales" not in metrics:
            metrics.append("sales")
    
    return metrics


def has_specific_metric(query: str) -> bool:
    """Check if query mentions a specific resource (oil, gas, water, etc.)."""
    return len(get_requested_metrics(query)) > 0


# ============================================================================
# TOOL CALLING HELPERS
# ============================================================================

def get_tools_for_mode(mode: str) -> List[str]:
    """
    Get the list of tools that MUST be called for a given mode.
    
    DATA_QUERY → query_data() + generate_visuals() (ALWAYS both)
    DOCUMENT_OVERVIEW → dataset_overview()
    FREEFORM_QUERY → conversational_response()
    SYSTEM_TASK → system_instruction()
    """
    tools_map = {
        QueryMode.DATA_QUERY: ["query_data", "generate_visuals"],
        QueryMode.DOCUMENT_OVERVIEW: ["dataset_overview"],
        QueryMode.FREEFORM_QUERY: ["conversational_response"],
        QueryMode.SYSTEM_TASK: ["system_instruction"],
    }
    return tools_map.get(mode, [])


def should_show_visualizations(query: str) -> bool:
    """
    Determine if visualizations should be shown for this query.
    
    Returns True ONLY for DATA_QUERY mode.
    DOCUMENT_OVERVIEW, FREEFORM, and SYSTEM never show charts.
    """
    mode, _, _ = classify_query_mode(query)
    return mode == QueryMode.DATA_QUERY
