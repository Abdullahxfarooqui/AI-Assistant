## Query Classification & Routing Implementation

### Summary
Implemented a query classification system that separates **general LLM queries** from **document/RAG queries**. This prevents general knowledge questions from being unnecessarily routed through the RAG pipeline.

---

### Files Modified

#### 1. **core/query_router.py** (NEW)
- `RAG_KEYWORDS`: Set of keywords indicating document/dataset queries (oil, gas, data, statistics, production, etc.)
- `GENERAL_KEYWORDS`: Set of keywords indicating general LLM queries (how to, explain, define, math, cooking, sports, etc.)
- `classify_query(query)`: Returns classification ("rag" or "general") with confidence score
- `should_use_rag(query)`: Boolean check for routing decision
- `get_query_classification(query)`: Returns detailed classification info

**Classification Logic:**
- Counts keyword matches in both categories
- RAG > General → Route to RAG pipeline
- General > RAG → Use pure LLM
- No matches → Default to RAG (safe default)
- Confidence based on match ratio (capped at 0.95)

#### 2. **core/rag_engine.py**
- Added `handle_general_query(query)` function:
  - Calls LLM directly without any RAG pipeline
  - Returns `"query_type": "general"` flag
  - Returns `"intent": "general_llm"` for UI display
  - No embeddings, vector search, or dataframe operations
  - No visualizations
  - Uses relaxed temperature (0.7) for creative responses

- Updated `query()` function:
  - **STEP 1**: Calls `should_use_rag()` FIRST
  - If False → immediately returns `handle_general_query()` result
  - If True → proceeds with RAG pipeline as before
  - Added import for `query_router` module

#### 3. **app.py**
- Updated query processing logic:
  - Checks `result.get("query_type")` to detect general queries
  - Shows "📊 Data loaded" message ONLY for RAG queries
  - Removed auto-prefix for general questions
  - Added "general_llm" to intent_info dictionary
  - Spinner now says "Processing query..." (more generic)

---

### Expected Behavior After Fix

#### ✅ Query: "how to eat ice cream"
```
→ Classification: GENERAL
→ No RAG pipeline
→ No dataset loading
→ Pure LLM answer
→ No "Data loaded" message
→ No visualizations
```

#### ✅ Query: "what is in the document"
```
→ Classification: RAG
→ Loads dataset
→ Shows "📊 Data loaded" message
→ RAG pipeline executes
→ No charts (general overview)
```

#### ✅ Query: "show me gas production"
```
→ Classification: RAG (contains "gas" + "production")
→ Loads dataset
→ Shows "📊 Data loaded" message
→ Detects "gas" metric
→ Shows gas-only charts
→ Filters columns to gas columns only
```

#### ✅ Query: "give me detailed document analysis"
```
→ Classification: RAG (contains "document" + "analysis")
→ Loads dataset
→ Enters "detailed" mode
→ Retrieves more chunks (k=15)
→ Comprehensive RAG summary
→ No charts unless specific metric
```

---

### Test Results
All 10 tests pass:

**General Queries (NOT routing to RAG):**
- ✅ "how to eat ice cream" → general
- ✅ "what is 2 plus 2" → general  
- ✅ "how to cook pasta" → general

**Document/RAG Queries (routing to RAG):**
- ✅ "show me gas production" → rag
- ✅ "what is in the document" → rag
- ✅ "give me detailed document analysis" → rag
- ✅ "what is total oil production" → rag
- ✅ "show water injection statistics" → rag
- ✅ "analyze data quality and missing values" → rag
- ✅ "show me a chart of production trends" → rag

---

### Keyword Examples

**RAG Keywords** (triggers document routing):
- Data/Dataset related: data, dataset, document, table, column, rows, file, csv, excel, schema
- Statistics: statistics, stats, summary, summarize, overview, analyze, analysis
- Oil & Gas: oil, gas, water, condensate, production, sales, injection, energy, volume
- Visualization: visualize, chart, graph, plot, display, scatter, histogram, bar
- Quality: missing, null, empty, duplicate, clean, error

**General Keywords** (triggers pure LLM):
- How-to: how to, explain, define, tell me about
- Math: calculate, math, python, code, function
- General knowledge: recipe, cook, food, health, exercise, weather, sports, news, science, history

---

### Implementation Notes

1. **Safe Default**: If no keywords match, defaults to RAG (safe default assumption)
2. **Confidence Scores**: Helps identify uncertain classifications (future feature: use for hybrid approaches)
3. **No Breaking Changes**: Backward compatible - RAG queries work exactly as before
4. **Performance**: Query classification adds minimal overhead (string matching only)
5. **Extensible**: Easy to add more keywords or refine classification logic

---

### Next Steps (Optional Enhancements)

1. **User Feedback**: Track user corrections ("that was wrong classification") to improve keywords
2. **Confidence Thresholds**: For borderline cases (confidence < 0.6), use hybrid approach
3. **Context Awareness**: Remember previous query type to inform next query classification
4. **Telemetry**: Log classification decisions to identify patterns and misclassifications
