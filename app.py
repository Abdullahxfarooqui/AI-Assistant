"""
Production RAG Application - Streamlit UI with Visualizations
Handles large files (15-50MB+) with:
- Python-based table extraction (not LLM)
- Full data preservation (no deduplication)
- Statistics computation and display
- Trend and anomaly detection
- Auto-generated charts and visualizations
- Progress indicators for all operations
- METRIC-FOCUSED: Only shows charts for the metric user asked about
"""
import streamlit as st
import sys
from pathlib import Path
import time
import json
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.extractor import extract_document, get_extraction_summary
from core.chunker import chunk_document, get_chunk_summary
from core.embedder import embed_chunks, compute_doc_hash
from core.vector_store import get_vector_store
from core.rag_engine import (
    query, 
    summarize_document, 
    list_documents,
    get_document_info,
    compare_documents,
    set_dataframe_cache,
    get_target_columns,
    detect_specific_metrics,
    is_general_query,
    METRIC_COLUMNS
)

# Global DataFrame cache for visualizations
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}


# Page configuration
st.set_page_config(
    page_title="RAG Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .stat-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .source-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .processing-step {
        padding: 0.5rem 1rem;
        border-left: 3px solid #667eea;
        margin: 0.5rem 0;
        background: #f8f9ff;
    }
    .anomaly-warning {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .trend-info {
        background: #d1ecf1;
        border: 1px solid #17a2b8;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_excel_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load Excel/CSV data for visualization - CACHED."""
    try:
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        else:
            return pd.read_excel(filepath)
    except Exception as e:
        return None


def get_cached_dataframe(filename: str) -> Optional[pd.DataFrame]:
    """Get or load DataFrame for a document."""
    if filename in st.session_state.dataframes:
        return st.session_state.dataframes[filename]
    
    # Try to find the file
    data_dir = Path(__file__).parent / "data" / "uploads"
    file_path = data_dir / filename
    
    if not file_path.exists():
        # Check current directory
        file_path = Path(__file__).parent / filename
    
    if file_path.exists():
        df = load_excel_data(str(file_path))
        if df is not None:
            st.session_state.dataframes[filename] = df
        return df
    
    return None


def render_data_visualizations(df: pd.DataFrame, filename: str):
    """Render comprehensive visualizations for the dataset."""
    st.markdown("---")
    st.markdown("## 📊 Data Visualizations")
    
    # Get numeric and date columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Try to identify date column if not datetime type
    if not date_cols:
        for col in df.columns:
            if 'DATE' in col.upper() or 'TIME' in col.upper():
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    pass
    
    # Get categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Get key production columns
    prod_cols = [c for c in numeric_cols if any(x in c.upper() for x in ['PROD', 'SALES', 'VOL', 'RATE', 'ENERGY'])]
    
    # Create tabs for different visualizations
    viz_tabs = st.tabs(["📈 Time Series", "📊 Bar Charts", "🥧 Pie Charts", "📉 Histograms", "🔥 Heatmap", "📋 Data Table"])
    
    # Tab 1: Time Series
    with viz_tabs[0]:
        render_time_series(df, date_cols, prod_cols, numeric_cols)
    
    # Tab 2: Bar Charts
    with viz_tabs[1]:
        render_bar_charts(df, cat_cols, prod_cols, numeric_cols)
    
    # Tab 3: Pie Charts
    with viz_tabs[2]:
        render_pie_charts(df, cat_cols, prod_cols, numeric_cols)
    
    # Tab 4: Histograms
    with viz_tabs[3]:
        render_histograms(df, prod_cols, numeric_cols)
    
    # Tab 5: Heatmap
    with viz_tabs[4]:
        render_heatmap(df, numeric_cols)
    
    # Tab 6: Data Table
    with viz_tabs[5]:
        render_data_table(df, filename)


def render_time_series(df: pd.DataFrame, date_cols: List[str], prod_cols: List[str], numeric_cols: List[str]):
    """Render time series charts."""
    st.markdown("### 📈 Production Over Time")
    
    if not date_cols:
        st.info("No date columns detected for time series. Try selecting a date column.")
        return
    
    date_col = date_cols[0]
    
    # Column selection
    col1, col2 = st.columns(2)
    with col1:
        available_metrics = prod_cols if prod_cols else numeric_cols[:10]
        selected_metrics = st.multiselect(
            "Select metrics to plot:",
            available_metrics,
            default=available_metrics[:3] if len(available_metrics) >= 3 else available_metrics
        )
    
    with col2:
        group_col = st.selectbox(
            "Group by (optional):",
            ["None"] + [c for c in df.columns if df[c].nunique() <= 50 and df[c].dtype == 'object'],
            index=0
        )
    
    if not selected_metrics:
        st.warning("Select at least one metric to visualize.")
        return
    
    # Create time series chart
    df_sorted = df.sort_values(date_col)
    
    if group_col != "None":
        # Grouped time series
        fig = go.Figure()
        for metric in selected_metrics[:2]:  # Limit to 2 metrics for grouped
            for group in df[group_col].dropna().unique()[:10]:  # Limit groups
                group_data = df_sorted[df_sorted[group_col] == group]
                fig.add_trace(go.Scatter(
                    x=group_data[date_col],
                    y=group_data[metric],
                    mode='lines+markers',
                    name=f"{metric} - {group}",
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
        fig.update_layout(
            title=f"Production Over Time by {group_col}",
            xaxis_title="Date",
            yaxis_title="Value",
            height=500,
            hovermode='x unified'
        )
    else:
        # Simple time series
        fig = make_subplots(specs=[[{"secondary_y": len(selected_metrics) > 1}]])
        
        colors = px.colors.qualitative.Set2
        for i, metric in enumerate(selected_metrics):
            # Aggregate by date
            agg_data = df_sorted.groupby(date_col)[metric].sum().reset_index()
            
            fig.add_trace(go.Scatter(
                x=agg_data[date_col],
                y=agg_data[metric],
                mode='lines+markers',
                name=metric,
                line=dict(width=2, color=colors[i % len(colors)]),
                marker=dict(size=4)
            ), secondary_y=(i > 0))
        
        # Get unit if available
        unit = ""
        for metric in selected_metrics[:1]:
            uom_col = metric + "_UOM"
            if uom_col in df.columns:
                unit = df[uom_col].dropna().iloc[0] if len(df[uom_col].dropna()) > 0 else ""
                break
        
        fig.update_layout(
            title=f"Production Trends Over Time",
            xaxis_title="Date",
            yaxis_title=f"Value ({unit})" if unit else "Value",
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary stats below chart
    if selected_metrics:
        st.markdown("**Quick Stats:**")
        stat_cols = st.columns(len(selected_metrics[:4]))
        for i, metric in enumerate(selected_metrics[:4]):
            with stat_cols[i]:
                data = df[metric].dropna()
                st.metric(
                    metric[:20] + "..." if len(metric) > 20 else metric,
                    f"{data.sum():,.0f}",
                    f"Avg: {data.mean():,.1f}"
                )


def render_bar_charts(df: pd.DataFrame, cat_cols: List[str], prod_cols: List[str], numeric_cols: List[str]):
    """Render bar charts."""
    st.markdown("### 📊 Production by Category")
    
    col1, col2 = st.columns(2)
    
    with col1:
        category_col = st.selectbox(
            "Group by:",
            [c for c in cat_cols if df[c].nunique() <= 50] if cat_cols else df.columns[:5].tolist(),
            index=0
        )
    
    with col2:
        value_col = st.selectbox(
            "Metric:",
            prod_cols if prod_cols else numeric_cols[:10],
            index=0
        )
    
    if category_col and value_col:
        # Aggregate data
        agg_data = df.groupby(category_col)[value_col].agg(['sum', 'mean', 'count']).reset_index()
        agg_data.columns = [category_col, 'Total', 'Average', 'Count']
        agg_data = agg_data.sort_values('Total', ascending=False).head(20)
        
        # Get unit
        unit = ""
        uom_col = value_col + "_UOM"
        if uom_col in df.columns:
            unit = df[uom_col].dropna().iloc[0] if len(df[uom_col].dropna()) > 0 else ""
        
        # Create bar chart
        fig = px.bar(
            agg_data,
            x=category_col,
            y='Total',
            color='Average',
            color_continuous_scale='Viridis',
            title=f"Total {value_col} by {category_col}",
            labels={'Total': f'Total ({unit})' if unit else 'Total'},
            hover_data=['Average', 'Count']
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        with st.expander("📋 View Aggregated Data"):
            st.dataframe(agg_data, use_container_width=True)


def render_pie_charts(df: pd.DataFrame, cat_cols: List[str], prod_cols: List[str], numeric_cols: List[str]):
    """Render pie charts."""
    st.markdown("### 🥧 Distribution Charts")
    
    col1, col2 = st.columns(2)
    
    # Pie chart 1: Distribution by category
    with col1:
        if cat_cols:
            category_col = st.selectbox("Category:", cat_cols, key="pie_cat")
            value_col = st.selectbox("Value:", prod_cols if prod_cols else numeric_cols[:10], key="pie_val1")
            
            if category_col and value_col:
                agg_data = df.groupby(category_col)[value_col].sum().reset_index()
                agg_data = agg_data.nlargest(10, value_col)
                
                fig = px.pie(
                    agg_data,
                    values=value_col,
                    names=category_col,
                    title=f"{value_col} Distribution by {category_col}",
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
    
    # Pie chart 2: Compare multiple metrics
    with col2:
        st.markdown("**Compare Multiple Metrics**")
        compare_metrics = st.multiselect(
            "Select metrics to compare:",
            prod_cols if prod_cols else numeric_cols[:10],
            default=prod_cols[:3] if len(prod_cols) >= 3 else (prod_cols if prod_cols else numeric_cols[:3]),
            key="pie_compare"
        )
        
        if compare_metrics:
            totals = {col: df[col].sum() for col in compare_metrics}
            
            fig = px.pie(
                values=list(totals.values()),
                names=list(totals.keys()),
                title="Total Volume Comparison",
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)


def render_histograms(df: pd.DataFrame, prod_cols: List[str], numeric_cols: List[str]):
    """Render histograms."""
    st.markdown("### 📉 Distribution Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_col = st.selectbox(
            "Select column:",
            prod_cols if prod_cols else numeric_cols[:10],
            key="hist_col"
        )
    
    with col2:
        num_bins = st.slider("Number of bins:", 10, 100, 30, key="hist_bins")
    
    if selected_col:
        data = df[selected_col].dropna()
        
        # Get unit
        unit = ""
        uom_col = selected_col + "_UOM"
        if uom_col in df.columns:
            unit = df[uom_col].dropna().iloc[0] if len(df[uom_col].dropna()) > 0 else ""
        
        # Create histogram with box plot
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=data, nbinsx=num_bins, name="Distribution", marker_color='#667eea'),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(x=data, name="Box Plot", marker_color='#764ba2'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"Distribution of {selected_col}",
            xaxis2_title=f"Value ({unit})" if unit else "Value",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("**Distribution Statistics:**")
        stat_cols = st.columns(5)
        with stat_cols[0]:
            st.metric("Count", f"{len(data):,}")
        with stat_cols[1]:
            st.metric("Mean", f"{data.mean():,.2f}")
        with stat_cols[2]:
            st.metric("Median", f"{data.median():,.2f}")
        with stat_cols[3]:
            st.metric("Std Dev", f"{data.std():,.2f}")
        with stat_cols[4]:
            st.metric("Range", f"{data.max() - data.min():,.2f}")


def render_heatmap(df: pd.DataFrame, numeric_cols: List[str]):
    """Render correlation heatmap."""
    st.markdown("### 🔥 Correlation Heatmap")
    
    # Select columns for heatmap
    available_cols = numeric_cols[:20]  # Limit to 20 columns
    selected_cols = st.multiselect(
        "Select columns (max 15):",
        available_cols,
        default=available_cols[:min(10, len(available_cols))],
        key="heatmap_cols"
    )[:15]
    
    if len(selected_cols) >= 2:
        # Calculate correlation
        corr_matrix = df[selected_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=selected_cols,
            y=selected_cols,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title="Correlation Matrix"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Find strongest correlations
        st.markdown("**Strongest Correlations:**")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Column 1': corr_matrix.columns[i],
                    'Column 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
        corr_df = corr_df.nlargest(5, 'Abs Correlation')[['Column 1', 'Column 2', 'Correlation']]
        st.dataframe(corr_df, use_container_width=True)
    else:
        st.info("Select at least 2 columns to show correlation heatmap.")


def render_data_table(df: pd.DataFrame, filename: str):
    """Render interactive data table."""
    st.markdown("### 📋 Data Table")
    
    # Get key columns (exclude _UOM columns for cleaner display)
    all_cols = df.columns.tolist()
    key_cols = [c for c in all_cols if not c.endswith('_UOM')][:20]  # Limit default to 20 non-UOM columns
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_rows = st.selectbox("Show rows:", [10, 25, 50, 100, "All"], index=1)
    
    with col2:
        # Use a container with limited height for column selection
        columns_to_show = st.multiselect(
            "Select columns (max 20):",
            all_cols,
            default=key_cols[:10],  # Default to first 10 key columns
            max_selections=20  # Limit selections
        )
    
    with col3:
        sort_col = st.selectbox("Sort by:", ["None"] + all_cols[:30])  # Limit sort options
    
    # Apply filters
    display_df = df[columns_to_show] if columns_to_show else df[key_cols[:10]]
    
    if sort_col != "None" and sort_col in display_df.columns:
        display_df = display_df.sort_values(sort_col, ascending=False)
    
    if show_rows != "All":
        display_df = display_df.head(int(show_rows))
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Export option - simplified
    st.markdown("**📥 Export Options:**")
    csv = df.to_csv(index=False)
    st.download_button(
        "Download Full Dataset as CSV",
        csv,
        f"{Path(filename).stem}_export.csv",
        "text/csv",
        use_container_width=True
    )


def render_quick_stats(df: pd.DataFrame):
    """Render quick statistics summary."""
    st.markdown("### 📊 Quick Statistics")
    
    # Get key columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    prod_cols = [c for c in numeric_cols if any(x in c.upper() for x in ['PROD', 'SALES', 'VOL'])]
    
    # Display metrics
    if prod_cols:
        cols = st.columns(min(len(prod_cols), 4))
        for i, col in enumerate(prod_cols[:4]):
            with cols[i]:
                data = df[col].dropna()
                # Get unit
                unit = ""
                uom_col = col + "_UOM"
                if uom_col in df.columns:
                    unit = df[uom_col].dropna().iloc[0] if len(df[uom_col].dropna()) > 0 else ""
                
                st.metric(
                    col[:15] + "..." if len(col) > 15 else col,
                    f"{data.sum():,.0f} {unit}",
                    f"Mean: {data.mean():,.1f}"
                )
    
    # Summary table
    with st.expander("📋 Full Statistics Summary", expanded=False):
        stats_data = []
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                # Get unit
                unit = ""
                uom_col = col + "_UOM"
                if uom_col in df.columns:
                    unit_values = df[uom_col].dropna()
                    unit = unit_values.iloc[0] if len(unit_values) > 0 else ""
                
                stats_data.append({
                    'Column': col,
                    'Unit': unit,
                    'Count': len(data),
                    'Null %': f"{(df[col].isna().sum() / len(df)) * 100:.1f}%",
                    'Min': f"{data.min():,.2f}",
                    'Max': f"{data.max():,.2f}",
                    'Mean': f"{data.mean():,.2f}",
                    'Sum': f"{data.sum():,.0f}"
                })
        
        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)


def init_session():
    """Initialize session state."""
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_stats" not in st.session_state:
        st.session_state.last_stats = None


def process_file(file_bytes: bytes, filename: str, progress_placeholder) -> dict:
    """
    Process a file through the complete RAG pipeline.
    Uses Python for extraction and statistics (NOT LLM).
    """
    store = get_vector_store()
    doc_hash = compute_doc_hash(file_bytes)
    
    # Check if already indexed
    if store.has_document(doc_hash):
        docs = store.get_all_documents()
        doc_info = next((d for d in docs if d.get("doc_hash") == doc_hash), {})
        return {
            "status": "exists",
            "message": "Document already indexed",
            "doc_hash": doc_hash,
            "filename": doc_info.get("filename", filename),
            "num_chunks": doc_info.get("num_chunks", 0)
        }
    
    try:
        # Step 1: Extract (Python-based, not LLM)
        progress_placeholder.markdown("**Step 1/4:** 📄 Extracting tables with Python...")
        
        def extraction_progress(current, total):
            progress_placeholder.progress(
                current / total * 0.25,
                text=f"Processing page/sheet {current}/{total}..."
            )
        
        extraction = extract_document(file_bytes, filename, extraction_progress)
        
        if "error" in extraction:
            return {"status": "error", "message": extraction["error"]}
        
        # Store statistics for display
        statistics = extraction.get("statistics", {})
        
        # Show extraction summary
        summary = get_extraction_summary(extraction)
        progress_placeholder.markdown(f"**Extraction complete:**\n{summary}")
        time.sleep(0.5)
        
        # Step 2: Chunk (preserving all data)
        progress_placeholder.markdown("**Step 2/4:** ✂️ Creating structured chunks (all data preserved)...")
        
        chunks = chunk_document(extraction, max_tokens=2000)
        chunk_stats = get_chunk_summary(chunks)
        
        progress_placeholder.progress(0.5, text=f"Created {len(chunks)} chunks ({chunk_stats.get('table_chunks', 0)} tables)")
        time.sleep(0.3)
        
        if not chunks:
            return {"status": "error", "message": "No chunks created from document"}
        
        # Step 3: Embed
        progress_placeholder.markdown("**Step 3/4:** 🧠 Generating embeddings...")
        
        def embed_progress(current, total):
            progress = 0.5 + (current / total * 0.35)
            progress_placeholder.progress(
                progress,
                text=f"Embedding chunk {current}/{total}..."
            )
        
        embeddings = embed_chunks(chunks, embed_progress)
        
        # Step 4: Index
        progress_placeholder.markdown("**Step 4/4:** 📊 Building vector index...")
        progress_placeholder.progress(0.9, text="Adding to vector store...")
        
        num_added = store.add_chunks(chunks, embeddings, doc_hash)
        
        progress_placeholder.progress(1.0, text="✅ Complete!")
        
        return {
            "status": "success",
            "message": f"Successfully processed {filename}",
            "doc_hash": doc_hash,
            "filename": filename,
            "num_chunks": num_added,
            "num_tables": extraction.get("num_tables", 0),
            "is_dataset": extraction.get("is_dataset", False),
            "total_rows": extraction.get("total_table_rows", 0),
            "statistics": statistics
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def render_sidebar():
    """Render sidebar with document list and stats."""
    with st.sidebar:
        st.markdown("## 📊 Dashboard")
        
        store = get_vector_store()
        stats = store.get_stats()
        docs = store.get_all_documents()
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📁 Documents", stats["total_documents"])
        with col2:
            st.metric("📦 Chunks", stats["total_chunks"])
        
        st.markdown("---")
        
        # Document list
        st.markdown("## 📄 Indexed Documents")
        
        if not docs:
            st.info("No documents indexed yet. Upload a file to get started!")
        else:
            for doc in docs:
                icon = "📊" if doc.get("is_dataset") else "📄"
                name = doc.get('filename', 'Unknown')
                display_name = name[:25] + "..." if len(name) > 25 else name
                
                with st.expander(f"{icon} {display_name}"):
                    st.write(f"**Chunks:** {doc.get('num_chunks', 0)}")
                    st.write(f"**Type:** {'Dataset' if doc.get('is_dataset') else 'Document'}")
                    st.code(doc.get("doc_hash", "")[:16], language=None)
                    
                    # Quick summary button
                    if st.button("📋 Summary", key=f"sum_{doc.get('doc_hash', '')[:8]}"):
                        with st.spinner("Generating summary..."):
                            # Try to get DataFrame for better summary
                            df = get_cached_dataframe(name)
                            result = summarize_document(doc.get("doc_hash"), dataframe=df)
                            if result.get("status") == "success":
                                st.session_state.last_result = {
                                    "type": "summary",
                                    "content": result.get("summary", ""),
                                    "doc": name
                                }
        
        st.markdown("---")
        
        # Clear button
        if docs:
            if st.button("🗑️ Clear All Data", type="secondary"):
                store.clear()
                st.rerun()


def render_upload():
    """Render file upload section."""
    st.markdown("### 📤 Upload Documents")
    st.caption("Supports PDF, Excel (xlsx/xls), and CSV files • Large files up to 200MB supported")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "xlsx", "xls", "csv"],
        accept_multiple_files=True,
        help="Tables extracted with Python (pdfplumber/pandas) • All rows preserved • Statistics computed automatically"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            file_size_mb = len(file_bytes) / (1024 * 1024)
            
            st.markdown(f"**Processing:** {uploaded_file.name} ({file_size_mb:.1f} MB)")
            
            # Save file to uploads directory for later visualization access
            uploads_dir = Path(__file__).parent / "data" / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            saved_path = uploads_dir / uploaded_file.name
            with open(saved_path, 'wb') as f:
                f.write(file_bytes)
            
            # Cache the dataframe for visualizations
            try:
                import io
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(file_bytes))
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(io.BytesIO(file_bytes))
                else:
                    df = None
                
                if df is not None:
                    st.session_state.dataframes[uploaded_file.name] = df
            except Exception as e:
                st.warning(f"Could not cache dataframe for visualizations: {e}")
            
            # Progress placeholder
            progress_area = st.empty()
            
            # Process file
            result = process_file(file_bytes, uploaded_file.name, progress_area)
            
            # Show result
            if result["status"] == "success":
                st.success(f"✅ {result['message']}")
                
                # Metrics
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Chunks", result.get("num_chunks", 0))
                with cols[1]:
                    st.metric("Tables", result.get("num_tables", 0))
                with cols[2]:
                    st.metric("Data Rows", f"{result.get('total_rows', 0):,}")
                with cols[3]:
                    st.metric("Type", "Dataset" if result.get("is_dataset") else "Text")
                
                # Show statistics if available
                statistics = result.get("statistics", {})
                if statistics:
                    _render_statistics_panel(statistics)
                    
            elif result["status"] == "exists":
                st.info(f"ℹ️ {result['message']} ({result.get('num_chunks', 0)} chunks)")
            else:
                st.error(f"❌ Error: {result['message']}")
            
            progress_area.empty()


def _render_statistics_panel(statistics: Dict):
    """Render statistics panel with trends and anomalies."""
    with st.expander("📊 **Data Statistics** (computed with Python, not LLM)", expanded=True):
        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{statistics.get('total_rows', 0):,}")
        with col2:
            st.metric("Columns", statistics.get("total_columns", 0))
        with col3:
            st.metric("Memory", f"{statistics.get('memory_mb', 0):.1f} MB")
        
        # Column types
        st.markdown("**Column Types:**")
        col_types = []
        if statistics.get("numeric_columns"):
            col_types.append(f"📈 {len(statistics['numeric_columns'])} numeric")
        if statistics.get("date_columns"):
            col_types.append(f"📅 {len(statistics['date_columns'])} date")
        if statistics.get("categorical_columns"):
            col_types.append(f"🏷️ {len(statistics['categorical_columns'])} categorical")
        st.write(" • ".join(col_types) if col_types else "No columns detected")
        
        # Anomalies
        anomalies = statistics.get("anomalies", [])
        if anomalies:
            st.markdown("**⚠️ Detected Anomalies:**")
            for anom in anomalies[:3]:
                st.markdown(f"""
                <div class="anomaly-warning">
                    <strong>{anom['column']}</strong>: {anom['count']} outliers ({anom['percentage']:.1f}%)
                </div>
                """, unsafe_allow_html=True)
        
        # Trends
        trends = statistics.get("trends", [])
        if trends:
            st.markdown("**📈 Detected Trends:**")
            for trend in trends[:3]:
                direction_icon = "↗️" if trend['direction'] == "increasing" else "↘️" if trend['direction'] == "decreasing" else "➡️"
                st.markdown(f"""
                <div class="trend-info">
                    {direction_icon} <strong>{trend['column']}</strong>: {trend['direction']} ({trend['change_percent']:+.1f}% change)
                </div>
                """, unsafe_allow_html=True)


def render_query():
    """Render query interface with visualizations - 4-MODE SYSTEM."""
    st.markdown("### 🤖 Ask Questions")
    
    docs = list_documents()
    
    if not docs:
        st.info("📤 Upload a document first to start querying!")
        return
    
    # Document selector
    doc_options = {"All Documents": None}
    for doc in docs:
        doc_options[doc.get("filename", "Unknown")] = doc.get("doc_hash")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_doc = st.selectbox(
            "Select Document:",
            list(doc_options.keys()),
            index=0,
            key="query_doc_select"
        )
    with col2:
        show_viz = st.checkbox("📊 Charts", value=True, help="Auto-generate charts for data queries")
    
    # Query input
    user_query = st.text_area(
        "Your question:",
        placeholder="Examples: 'Show me oil production', 'What's in this document?', 'Gas trends over time'",
        height=100
    )
    
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a question")
            return
        
        # Get document hash and filename
        doc_hash = doc_options[selected_doc]
        selected_filename = selected_doc if selected_doc != "All Documents" else None
        
        # Load DataFrame if available
        df = None
        if selected_filename:
            df = get_cached_dataframe(selected_filename)
            
            # If not cached, try to load from uploads
            if df is None:
                data_dir = Path(__file__).parent / "data" / "uploads"
                for file in data_dir.glob("*"):
                    if file.name == selected_filename or selected_filename in file.name:
                        try:
                            if file.suffix.lower() in ['.xlsx', '.xls']:
                                df = pd.read_excel(file)
                            elif file.suffix.lower() == '.csv':
                                df = pd.read_csv(file)
                            if df is not None:
                                st.session_state.dataframes[selected_filename] = df
                                break
                        except:
                            pass
        
        # Update the cache in rag_engine
        set_dataframe_cache(st.session_state.dataframes)
        
        # Execute query
        with st.spinner("Processing..."):
            result = query(user_query, doc_hash=doc_hash, k=10, dataframe=df)
        
        # ===================================================================
        # 4-MODE RESPONSE HANDLING
        # ===================================================================
        query_mode = result.get("query_mode", "data_query")
        
        # ---------------------------------------------------------------
        # MODE 1: FREEFORM_QUERY - Show refusal message only
        # ---------------------------------------------------------------
        if query_mode == "freeform_query":
            st.markdown("### ⚠️ Non-Data Query Detected")
            st.markdown(result.get("answer", ""))
            return
        
        # ---------------------------------------------------------------
        # MODE 2: SYSTEM_TASK - Show system guidance only
        # ---------------------------------------------------------------
        if query_mode == "system_task":
            st.markdown("### 🔧 System Request")
            st.markdown(result.get("answer", ""))
            return
        
        # ---------------------------------------------------------------
        # MODE 3: DOCUMENT_OVERVIEW - Show detailed overview (no charts)
        # ---------------------------------------------------------------
        if query_mode == "document_overview":
            if df is not None:
                st.info(f"📊 Dataset: {len(df):,} rows × {len(df.columns)} columns")
            st.markdown(result.get("answer", ""))
            return
        
        # ---------------------------------------------------------------
        # MODE 4: DATA_QUERY - Full analysis with statistics and charts
        # ---------------------------------------------------------------
        
        # Show dataset info
        if df is not None:
            st.info(f"📊 Analyzed: {len(df):,} rows × {len(df.columns)} columns")
        
        # Show answer
        st.markdown("### 📋 Analysis Result")
        st.markdown(result.get("answer", "No answer generated"))
        
        # Get visualization parameters from result
        show_visualizations = result.get("show_visualizations", False)
        specific_metrics = result.get("specific_metrics", [])
        target_columns = result.get("target_columns", [])
        
        # ===================================================================
        # GENERATE VISUALIZATIONS (ONLY for DATA_QUERY with metrics)
        # ===================================================================
        if show_viz and show_visualizations and df is not None and (specific_metrics or target_columns):
            st.markdown("---")
            
            # Get columns to visualize
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # ================================================================
            # STRICT METRIC FILTERING - Only show columns for requested metric
            # ================================================================
            metric_column_map = {
                'oil': ['OIL'],
                'gas': ['GAS'],
                'water': ['WAT', 'WATER'],
                'condensate': ['COND'],
                'lpg': ['LPG'],
                'ngl': ['NGL'],
                'energy': ['ENERGY', 'BTU'],
                'heat': ['HEAT'],
                'injection': ['INJ'],
                'production': ['PROD'],
                'sales': ['SALES']
            }
            
            # Get the EXACT keywords for requested metrics only
            filter_keywords = []
            for m in specific_metrics:
                if m in metric_column_map:
                    filter_keywords.extend(metric_column_map[m])
            
            # Filter target_columns to ONLY include the requested metric
            if target_columns and filter_keywords:
                # Strict filter: column must contain at least one keyword
                prod_cols = [
                    c for c in target_columns 
                    if c in numeric_cols and any(kw in c.upper() for kw in filter_keywords)
                ][:4]  # Limit to 4 columns max
            elif target_columns:
                # No specific filter keywords, but have target columns
                prod_cols = [c for c in target_columns if c in numeric_cols][:4]
            else:
                # Fallback: find columns matching the metric keywords
                all_prod_cols = [c for c in numeric_cols if any(x in c.upper() for x in ['PROD', 'SALES', 'VOL', 'ENERGY', 'INJ'])]
                if filter_keywords:
                    prod_cols = [c for c in all_prod_cols if any(kw in c.upper() for kw in filter_keywords)][:4]
                else:
                    prod_cols = all_prod_cols[:4]
            
            if prod_cols:
                metric_title = ', '.join(specific_metrics).upper() if specific_metrics else "Production"
                st.markdown(f"### 📊 {metric_title} Visualizations")
                st.info(f"📌 Showing charts for: {', '.join(prod_cols[:4])}")
                
                # Quick stats cards
                stat_cols = st.columns(min(len(prod_cols), 4))
                for i, col in enumerate(prod_cols[:4]):
                    with stat_cols[i]:
                        data = df[col].dropna()
                        unit = ""
                        uom_col = col + "_UOM"
                        if uom_col in df.columns:
                            uom_vals = df[uom_col].dropna()
                            unit = uom_vals.iloc[0] if len(uom_vals) > 0 else ""
                        st.metric(
                            col[:15] + "..." if len(col) > 15 else col,
                            f"{data.sum():,.0f}",
                            f"Avg: {data.mean():,.1f} {unit}"
                        )
                
                # Chart tabs
                chart_tabs = st.tabs(["📈 Trend", "📊 Bar Chart", "🥧 Distribution"])
                
                with chart_tabs[0]:
                    # Time series - show ALL requested metrics
                    date_col = None
                    for col in df.columns:
                        if 'DATE' in col.upper() or 'TIME' in col.upper():
                            try:
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                                date_col = col
                                break
                            except:
                                pass
                    
                    if date_col and prod_cols:
                        df_sorted = df.sort_values(date_col)
                        
                        # Create multi-line chart for ALL metrics
                        fig = go.Figure()
                        colors = px.colors.qualitative.Set2
                        
                        for i, metric_col in enumerate(prod_cols[:4]):  # Show up to 4 metrics
                            agg_data = df_sorted.groupby(date_col)[metric_col].sum().reset_index()
                            fig.add_trace(go.Scatter(
                                x=agg_data[date_col],
                                y=agg_data[metric_col],
                                mode='lines+markers',
                                name=metric_col,
                                line=dict(width=2, color=colors[i % len(colors)]),
                                marker=dict(size=4)
                            ))
                        
                        fig.update_layout(
                            title=f"{metric_title} Trends Over Time",
                            xaxis_title="Date",
                            yaxis_title="Volume",
                            height=450,
                            hovermode='x unified',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No date column found for time series.")
                
                with chart_tabs[1]:
                    # Bar chart - show ALL metrics side by side
                    cat_cols = [c for c in df.columns if df[c].dtype == 'object' and df[c].nunique() <= 50]
                    if cat_cols and prod_cols:
                        cat_col = 'ITEM_NAME' if 'ITEM_NAME' in df.columns else cat_cols[0]
                        
                        # Aggregate ALL metrics by category
                        agg_dict = {col: 'sum' for col in prod_cols[:4]}
                        agg_data = df.groupby(cat_col).agg(agg_dict).reset_index()
                        
                        # Sort by total of all metrics and get top 10
                        agg_data['_total'] = agg_data[prod_cols[:4]].sum(axis=1)
                        agg_data = agg_data.nlargest(10, '_total').drop(columns=['_total'])
                        
                        # Create grouped bar chart
                        fig = go.Figure()
                        colors = px.colors.qualitative.Set2
                        
                        for i, metric_col in enumerate(prod_cols[:4]):
                            fig.add_trace(go.Bar(
                                x=agg_data[cat_col],
                                y=agg_data[metric_col],
                                name=metric_col,
                                marker_color=colors[i % len(colors)]
                            ))
                        
                        fig.update_layout(
                            title=f"Top 10 by {metric_title}",
                            xaxis_title=cat_col,
                            yaxis_title="Volume",
                            barmode='group',
                            height=450,
                            xaxis_tickangle=-45,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No categorical columns found for bar chart.")
                
                with chart_tabs[2]:
                    # Pie chart
                    if len(prod_cols) >= 2:
                        totals = {col: df[col].sum() for col in prod_cols[:5]}
                        title = f"{metric_title} Distribution"
                        fig = px.pie(
                            values=list(totals.values()),
                            names=list(totals.keys()),
                            title=title,
                            hole=0.4
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    elif len(prod_cols) == 1 and cat_cols:
                        cat_col = cat_cols[0]
                        agg_data = df.groupby(cat_col)[prod_cols[0]].sum().reset_index()
                        agg_data = agg_data.nlargest(8, prod_cols[0])
                        fig = px.pie(
                            agg_data, values=prod_cols[0], names=cat_col,
                            title=f"{prod_cols[0]} by {cat_col}",
                            hole=0.4
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"Single metric: {prod_cols[0]} (Total: {df[prod_cols[0]].sum():,.0f})")
            else:
                st.warning("No matching columns found for the requested metrics.")
        
        # Show sources (collapsed)
        sources = result.get("sources", [])
        if sources:
            with st.expander(f"📑 Sources ({len(sources)})", expanded=False):
                for i, src in enumerate(sources, 1):
                    score = src.get('score', 0)
                    score_pct = f"{score:.0%}" if isinstance(score, float) else str(score)
                    st.markdown(f"**{i}. {src.get('filename', 'Unknown')}** (relevance: {score_pct})")


def render_quick_actions():
    """Render quick action buttons."""
    docs = list_documents()
    
    if not docs:
        return
    
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Generate Summary**")
        doc_for_summary = st.selectbox(
            "Document:",
            [d.get("filename", "Unknown") for d in docs],
            key="summary_doc",
            label_visibility="collapsed"
        )
        
        if st.button("📋 Generate Summary", use_container_width=True):
            doc_hash = next(
                (d.get("doc_hash") for d in docs if d.get("filename") == doc_for_summary),
                None
            )
            if doc_hash:
                with st.spinner("Generating comprehensive summary..."):
                    # Get DataFrame for better summary
                    df = get_cached_dataframe(doc_for_summary)
                    result = summarize_document(doc_hash, dataframe=df)
                
                if result.get("status") == "success":
                    st.session_state.last_result = {
                        "type": "summary",
                        "content": result.get("summary", ""),
                        "doc": doc_for_summary
                    }
                else:
                    st.error(result.get("message", "Error generating summary"))
    
    with col2:
        st.markdown("**Document Info**")
        doc_for_info = st.selectbox(
            "Document:",
            [d.get("filename", "Unknown") for d in docs],
            key="info_doc",
            label_visibility="collapsed"
        )
        
        if st.button("ℹ️ Get Document Info", use_container_width=True):
            doc_hash = next(
                (d.get("doc_hash") for d in docs if d.get("filename") == doc_for_info),
                None
            )
            if doc_hash:
                with st.spinner("Retrieving document info..."):
                    info = get_document_info(doc_hash)
                
                if "error" not in info:
                    st.session_state.last_result = {
                        "type": "info",
                        "content": info,
                        "doc": doc_for_info
                    }
                else:
                    st.error(info.get("error", "Error getting document info"))
    
    # Show last result
    if st.session_state.last_result:
        result = st.session_state.last_result
        st.markdown("---")
        
        if result["type"] == "summary":
            st.markdown(f"### 📋 Summary: {result['doc']}")
            st.markdown(result["content"])
            
            # ================================================================
            # ADD VISUALIZATIONS TO SUMMARY
            # ================================================================
            df = get_cached_dataframe(result['doc'])
            if df is not None:
                st.markdown("---")
                st.markdown("### 📊 Key Visualizations")
                
                # Get production columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                prod_cols = [c for c in numeric_cols if any(x in c.upper() for x in ['PROD', 'SALES']) and 'VOL' in c.upper()][:5]
                
                if prod_cols:
                    # Create two columns for charts
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Pie chart - Production Distribution
                        totals = {col.replace('PROD_', '').replace('SALES_', '').replace('_VOL', ''): df[col].sum() 
                                  for col in prod_cols if df[col].sum() > 0}
                        if totals:
                            fig = px.pie(
                                values=list(totals.values()),
                                names=list(totals.keys()),
                                title="Production Volume Distribution",
                                hole=0.4
                            )
                            fig.update_layout(height=350)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_col2:
                        # Bar chart - Top metrics
                        if len(prod_cols) >= 2:
                            bar_data = {col.replace('PROD_', '').replace('SALES_', '').replace('_VOL', ''): df[col].sum() 
                                       for col in prod_cols[:5]}
                            fig = px.bar(
                                x=list(bar_data.keys()),
                                y=list(bar_data.values()),
                                title="Total Volume by Metric",
                                labels={'x': 'Metric', 'y': 'Total Volume'}
                            )
                            fig.update_layout(height=350)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Time series if date column exists
                    date_cols = [c for c in df.columns if 'DATE' in c.upper() or 'TIME' in c.upper()]
                    if date_cols and prod_cols:
                        try:
                            date_col = date_cols[0]
                            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                            df_sorted = df.sort_values(date_col)
                            
                            # Aggregate by date
                            fig = go.Figure()
                            colors = px.colors.qualitative.Set2
                            for i, col in enumerate(prod_cols[:3]):
                                agg_data = df_sorted.groupby(date_col)[col].sum().reset_index()
                                display_name = col.replace('PROD_', '').replace('SALES_', '').replace('_VOL', '')
                                fig.add_trace(go.Scatter(
                                    x=agg_data[date_col],
                                    y=agg_data[col],
                                    mode='lines',
                                    name=display_name,
                                    line=dict(width=2, color=colors[i % len(colors)])
                                ))
                            
                            fig.update_layout(
                                title="Production Trends Over Time",
                                xaxis_title="Date",
                                yaxis_title="Volume",
                                height=400,
                                hovermode='x unified',
                                legend=dict(orientation="h", yanchor="bottom", y=1.02)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            pass  # Skip if date parsing fails
            
        elif result["type"] == "info":
            st.markdown(f"### ℹ️ Document Info: {result['doc']}")
            info = result["content"]
            
            cols = st.columns(4)
            with cols[0]:
                st.metric("Chunks", info.get("num_chunks", 0))
            with cols[1]:
                st.metric("Tables", info.get("num_tables", 0))
            with cols[2]:
                st.metric("Columns", len(info.get("columns", [])))
            with cols[3]:
                st.metric("Tokens", f"{info.get('total_tokens', 0):,}")
            
            if info.get("columns"):
                st.markdown("**Columns:**")
                st.code(", ".join(info["columns"]))
            
            if info.get("sample_data"):
                st.markdown("**Sample Data:**")
                import pandas as pd
                sample_df = pd.DataFrame(
                    info["sample_data"],
                    columns=info.get("columns", [f"Col_{i}" for i in range(len(info["sample_data"][0]))])
                )
                st.dataframe(sample_df, use_container_width=True)
        
        if st.button("Clear Result"):
            st.session_state.last_result = None
            st.rerun()


def render_visualization_tab():
    """Render dedicated visualization tab."""
    st.markdown("### 📊 Data Visualization Dashboard")
    
    docs = list_documents()
    
    if not docs:
        st.info("📤 Upload a document first to access visualizations!")
        return
    
    # Document selection
    doc_names = [d.get("filename", "Unknown") for d in docs]
    selected_doc = st.selectbox("Select Document:", doc_names, key="viz_doc_select")
    
    if selected_doc:
        # Try to load the DataFrame
        df = get_cached_dataframe(selected_doc)
        
        if df is None:
            # Try loading from uploads directory first
            data_dir = Path(__file__).parent / "data" / "uploads"
            try:
                # First try exact match
                exact_path = data_dir / selected_doc
                if exact_path.exists():
                    if exact_path.suffix.lower() in ['.xlsx', '.xls']:
                        df = pd.read_excel(exact_path)
                    elif exact_path.suffix.lower() == '.csv':
                        df = pd.read_csv(exact_path)
                    if df is not None:
                        st.session_state.dataframes[selected_doc] = df
                
                # If not found, search for partial match
                if df is None:
                    for file in data_dir.glob("*"):
                        if selected_doc in file.name or file.name in selected_doc:
                            if file.suffix.lower() in ['.xlsx', '.xls']:
                                df = pd.read_excel(file)
                            elif file.suffix.lower() == '.csv':
                                df = pd.read_csv(file)
                            if df is not None:
                                st.session_state.dataframes[selected_doc] = df
                                break
            except Exception as e:
                st.warning(f"Could not load from uploads: {e}")
            
            # Fallback: try current directory
            if df is None:
                try:
                    filepath = Path(__file__).parent / selected_doc
                    if filepath.exists():
                        if selected_doc.endswith('.csv'):
                            df = pd.read_csv(filepath)
                        else:
                            df = pd.read_excel(filepath)
                        st.session_state.dataframes[selected_doc] = df
                except Exception as e:
                    pass  # Silent fail, will show info message below
        
        if df is not None:
            st.success(f"📊 Loaded: {len(df):,} rows × {len(df.columns)} columns")
            
            # Show quick stats
            render_quick_stats(df)
            
            # Show full visualizations
            render_data_visualizations(df, selected_doc)
        else:
            st.warning(f"📁 File '{selected_doc}' not found in uploads. Please re-upload the file.")
            
            # List available files
            data_dir = Path(__file__).parent / "data" / "uploads"
            if data_dir.exists():
                files = list(data_dir.glob("*"))
                if files:
                    st.info(f"Available files in uploads: {[f.name for f in files]}")
            st.markdown("""
            **Available Visualizations:**
            - 📈 Time Series Charts (production over time)
            - 📊 Bar Charts (by category/well)
            - 🥧 Pie Charts (distribution)
            - 📉 Histograms (distribution analysis)
            - 🔥 Correlation Heatmaps
            - 📋 Interactive Data Tables
            """)


def main():
    """Main application."""
    init_session()
    
    # Initialize sidebar state
    if "show_sidebar" not in st.session_state:
        st.session_state.show_sidebar = True
    
    # Header with sidebar toggle
    col_title, col_toggle = st.columns([6, 1])
    with col_title:
        st.markdown('<p class="main-header">📊 RAG Data Analyst</p>', unsafe_allow_html=True)
    with col_toggle:
        if st.button("☰" if st.session_state.show_sidebar else "◀", help="Toggle Sidebar"):
            st.session_state.show_sidebar = not st.session_state.show_sidebar
            st.rerun()
    
    st.markdown(
        '<p class="sub-header">Upload documents • Python extracts tables & computes stats • Visualizations & LLM insights</p>',
        unsafe_allow_html=True
    )
    
    # Key features info
    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown("""
        **This system uses Python for data processing and LLM for insights:**
        
        1. **Table Extraction**: Python (pdfplumber/pandas) extracts tables - not LLM
        2. **Statistics**: Totals, averages, min/max computed with Pandas
        3. **Visualizations**: Interactive charts with Plotly
        4. **Anomaly Detection**: Outliers detected using IQR method
        5. **Trend Detection**: Time-series patterns identified algorithmically
        6. **LLM Role**: Interprets statistics, provides insights, answers questions
        
        **All data rows are preserved** - no deduplication of valid data.
        """)
    
    # Sidebar (conditional)
    if st.session_state.show_sidebar:
        render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "🤖 Query", "📊 Visualizations", "⚡ Quick Actions"])
    
    with tab1:
        render_upload()
    
    with tab2:
        render_query()
    
    with tab3:
        render_visualization_tab()
    
    with tab4:
        render_quick_actions()
    
    # Footer
    st.markdown("---")
    st.caption("📊 Python (Pandas/pdfplumber) for extraction & stats | 📈 Plotly for visualizations | 🧠 GPT-4o-mini for insights | 🔍 FAISS for search")


if __name__ == "__main__":
    main()
