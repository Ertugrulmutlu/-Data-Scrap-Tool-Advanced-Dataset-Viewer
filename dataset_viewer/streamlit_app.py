import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration with dark theme
st.set_page_config(
    page_title="NPZ Dataset Viewer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS styles
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #fafafa;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: #262730;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #2d5aa0;
        margin: 0.5rem 0;
        color: #fafafa;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    .filter-section {
        background: #1e1e1e;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #333;
    }
    .chart-container {
        background: #262730;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #1e1e1e;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #262730;
        color: #fafafa;
    }
    .stSelectbox > div > div {
        background-color: #262730;
        color: #fafafa;
    }
    .stMultiSelect > div > div {
        background-color: #262730;
        color: #fafafa;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #fafafa;
    }
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #333;
        padding: 1rem;
        border-radius: 8px;
        color: #fafafa;
    }
    .element-container {
        color: #fafafa;
    }
    .stMarkdown {
        color: #fafafa;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
    }
    .stDataFrame {
        background-color: #262730;
    }
    .stExpander {
        background-color: #1e1e1e;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('''
<div class="main-header">
    <h1>📊 Advanced NPZ Dataset Analytics Platform</h1>
    <p>Comprehensive dataset analysis with advanced visualizations and statistical insights</p>
</div>
''', unsafe_allow_html=True)

# File loading section
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        query_params = st.query_params
        dataset_path = None
        uploaded = None
        
        if "dataset" in query_params and query_params["dataset"]:
            dataset_path = query_params["dataset"]
            st.info(f"📁 Query parameter dataset: `{dataset_path}`")
        else:
            uploaded = st.file_uploader("📤 Upload NPZ Dataset File", type=["npz"], 
                                      help="Select your NPZ dataset file for analysis")
    
    with col2:
        if dataset_path or uploaded:
            st.success("✅ Dataset loaded successfully!")

def safe_load_npz(path_or_file):
    """Safely load NPZ file with error handling"""
    try:
        with st.spinner("🔄 Loading dataset..."):
            if hasattr(path_or_file, "read"):
                npz = np.load(path_or_file, allow_pickle=True)
            else:
                npz = np.load(path_or_file, allow_pickle=True)
        return npz
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        return None

# Load dataset
npz = None
if dataset_path:
    if os.path.isfile(dataset_path):
        npz = safe_load_npz(dataset_path)
    else:
        st.error(f"❌ File not found: {dataset_path}")
elif uploaded:
    npz = safe_load_npz(uploaded)

if npz is None:
    st.stop()

# Extract and analyze data structure
data = list(npz["data"]) if "data" in npz else []
raw_keys = list(npz["keys"]) if "keys" in npz else []
key_list = [str(k) for k in raw_keys] if raw_keys else []

# Analyze data structure for additional insights
st.markdown("### 🔍 Dataset Structure Analysis")
data_structure_info = {}
if data:
    sample_entry = data[0]
    data_structure_info = {
        "Entry Length": len(sample_entry),
        "Image Shape": sample_entry[0].shape if len(sample_entry) > 0 else "N/A",
        "Image Data Type": str(sample_entry[0].dtype) if len(sample_entry) > 0 else "N/A",
        "Timestamp Available": len(sample_entry) > 1,
        "Keys Array Available": len(sample_entry) > 2,
        "Durations Available": len(sample_entry) > 3,
        "Phase Info Available": len(sample_entry) > 4,
    }

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    **📋 Dataset Overview:**
    - Total Entries: {len(data)}
    - Available Keys: {len(key_list)}
    - Entry Components: {data_structure_info.get('Entry Length', 0)}
    """)

with col2:
    st.markdown(f"""
    **🖼️ Image Data:**
    - Shape: {data_structure_info.get('Image Shape', 'N/A')}
    - Data Type: {data_structure_info.get('Image Data Type', 'N/A')}
    - Has Timestamps: {data_structure_info.get('Timestamp Available', False)}
    """)

with col3:
    st.markdown(f"""
    **🔧 Available Features:**
    - Keys Tracking: {data_structure_info.get('Keys Array Available', False)}
    - Duration Data: {data_structure_info.get('Durations Available', False)}
    - Phase Information: {data_structure_info.get('Phase Info Available', False)}
    """)

# Sidebar summary
with st.sidebar:
    st.markdown("### 📈 Quick Stats")
    st.markdown(f"""
    <div class="metric-card">
        <strong>Dataset Size:</strong> {len(data)} entries<br>
        <strong>Key Count:</strong> {len(key_list)}<br>
        <strong>Keys:</strong> {', '.join(key_list[:3])}{'...' if len(key_list) > 3 else ''}
    </div>
    """, unsafe_allow_html=True)

if not data:
    st.warning("⚠️ Dataset is empty or missing 'data' key.")
    st.stop()

def entry_to_dict(entry):
    """Convert entry to dictionary with comprehensive analysis"""
    keys_arr = np.array(entry[2]) if len(entry) > 2 else np.zeros(len(key_list))
    durations_arr = np.array(entry[3]) if len(entry) > 3 else np.zeros(len(key_list))
    phase = entry[4] if len(entry) > 4 else ""
    
    # Extract additional features
    active_keys = [key_list[i] for i, v in enumerate(keys_arr) if v == 1 and i < len(key_list)]
    num_active = len(active_keys)
    avg_hold = float(np.mean(durations_arr[durations_arr > 0])) if np.any(durations_arr > 0) else 0.0
    max_hold = float(np.max(durations_arr)) if len(durations_arr) > 0 else 0.0
    min_hold = float(np.min(durations_arr[durations_arr > 0])) if np.any(durations_arr > 0) else 0.0
    std_hold = float(np.std(durations_arr[durations_arr > 0])) if np.any(durations_arr > 0) else 0.0
    
    # Image statistics
    img_stats = {}
    if len(entry) > 0:
        img = entry[0]
        img_stats = {
            "mean_intensity": float(np.mean(img)),
            "std_intensity": float(np.std(img)),
            "min_intensity": float(np.min(img)),
            "max_intensity": float(np.max(img)),
            "entropy": float(-np.sum(img * np.log(img + 1e-10))) if np.all(img >= 0) else 0.0
        }
    
    return {
        "phase": phase,
        "active_keys": active_keys,
        "num_active": num_active,
        "avg_hold_duration": avg_hold,
        "max_hold_duration": max_hold,
        "min_hold_duration": min_hold,
        "std_hold_duration": std_hold,
        "durations": durations_arr,
        "timestamp": entry[1] if len(entry) > 1 else 0,
        **img_stats
    }

# Process all entries
with st.spinner("Processing dataset entries..."):
    rows = [entry_to_dict(e) for e in data]

# Create comprehensive DataFrame
df = pd.DataFrame({
    "phase": [r["phase"] for r in rows],
    "num_active": [r["num_active"] for r in rows],
    "avg_hold_duration": [r["avg_hold_duration"] for r in rows],
    "max_hold_duration": [r["max_hold_duration"] for r in rows],
    "min_hold_duration": [r["min_hold_duration"] for r in rows],
    "std_hold_duration": [r["std_hold_duration"] for r in rows],
    "timestamp": [r["timestamp"] for r in rows],
    "mean_intensity": [r.get("mean_intensity", 0) for r in rows],
    "std_intensity": [r.get("std_intensity", 0) for r in rows],
    "min_intensity": [r.get("min_intensity", 0) for r in rows],
    "max_intensity": [r.get("max_intensity", 0) for r in rows],
    "entropy": [r.get("entropy", 0) for r in rows],
})

# Add derived features
df["hold_duration_range"] = df["max_hold_duration"] - df["min_hold_duration"]
df["intensity_range"] = df["max_intensity"] - df["min_intensity"]
df["complexity_score"] = df["num_active"] * df["avg_hold_duration"] * df["entropy"]

# One-hot encode keys and keep active list
for k in key_list:
    df[f"key_{k}"] = [1 if k in r["active_keys"] else 0 for r in rows]
df["active_keys"] = [r["active_keys"] for r in rows]
df["no_active_key"] = df["active_keys"].apply(lambda lst: len(lst) == 0)

# Advanced sidebar filters
with st.sidebar:
    st.markdown("### 🔍 Advanced Filters")
    
    with st.expander("📊 Basic Filters", expanded=True):
        selected_phases = st.multiselect(
            "Select Phases", 
            options=sorted(df["phase"].unique()), 
            default=sorted(df["phase"].unique()),
            help="Filter entries by phase type"
        )
        
        selected_keys = st.multiselect(
            "Required Keys", 
            options=key_list, 
            default=key_list,
            help="Entries must contain at least one of these keys"
        )
        
        include_no_active = st.checkbox("Include entries with no active keys", value=True)
    
    with st.expander("⏱️ Duration Filters", expanded=True):
        duration_range = st.slider(
            "Average Hold Duration",
            min_value=float(df["avg_hold_duration"].min()),
            max_value=float(df["avg_hold_duration"].max()),
            value=(float(df["avg_hold_duration"].min()), float(df["avg_hold_duration"].max())),
            step=0.01,
            format="%.2f"
        )
        
        active_range = st.slider(
            "Number of Active Keys",
            min_value=int(df["num_active"].min()),
            max_value=int(df["num_active"].max()),
            value=(int(df["num_active"].min()), int(df["num_active"].max()))
        )
    
    with st.expander("🖼️ Image Filters", expanded=False):
        intensity_range = st.slider(
            "Mean Image Intensity",
            min_value=float(df["mean_intensity"].min()),
            max_value=float(df["mean_intensity"].max()),
            value=(float(df["mean_intensity"].min()), float(df["mean_intensity"].max())),
            format="%.3f"
        )
        
        entropy_range = st.slider(
            "Image Entropy (Complexity)",
            min_value=float(df["entropy"].min()),
            max_value=float(df["entropy"].max()),
            value=(float(df["entropy"].min()), float(df["entropy"].max())),
            format="%.3f"
        )

# Apply comprehensive filters
mask = pd.Series(True, index=df.index)
mask &= df["phase"].isin(selected_phases)
mask &= df["avg_hold_duration"].between(duration_range[0], duration_range[1])
mask &= df["num_active"].between(active_range[0], active_range[1])
mask &= df["mean_intensity"].between(intensity_range[0], intensity_range[1])
mask &= df["entropy"].between(entropy_range[0], entropy_range[1])

# Key filtering logic
if selected_keys:
    key_masks = [df[f"key_{k}"] == 1 for k in selected_keys if f"key_{k}" in df.columns]
    if key_masks:
        combined_key_mask = key_masks[0]
        for m in key_masks[1:]:
            combined_key_mask |= m
    else:
        combined_key_mask = pd.Series(False, index=df.index)
    if include_no_active:
        mask &= (combined_key_mask | df["no_active_key"])
    else:
        mask &= combined_key_mask
else:
    if include_no_active:
        mask &= df["no_active_key"]
    else:
        mask &= ~df["no_active_key"]

filtered = df[mask]

# Enhanced metrics dashboard
st.markdown("### 📊 Comprehensive Dashboard")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total Entries",
        value=len(df),
        help="Total number of entries in dataset"
    )

with col2:
    st.metric(
        label="Filtered Entries", 
        value=len(filtered),
        delta=len(filtered) - len(df),
        help="Entries after applying filters"
    )

with col3:
    avg_duration = filtered["avg_hold_duration"].mean() if len(filtered) > 0 else 0
    st.metric(
        label="Avg Hold Duration",
        value=f"{avg_duration:.2f}s",
        help="Average hold duration across filtered entries"
    )

with col4:
    avg_complexity = filtered["complexity_score"].mean() if len(filtered) > 0 else 0
    st.metric(
        label="Avg Complexity",
        value=f"{avg_complexity:.2f}",
        help="Average complexity score (keys × duration × entropy)"
    )

with col5:
    avg_entropy = filtered["entropy"].mean() if len(filtered) > 0 else 0
    st.metric(
        label="Avg Entropy",
        value=f"{avg_entropy:.2f}",
        help="Average image entropy (information content)"
    )

# Main content tabs with expanded analytics
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview", "🔬 Advanced Analytics", "📊 Statistical Analysis", 
    "🖼️ Image Analysis", "📋 Data Explorer", "🎯 Single Entry Inspector"
])

with tab1:
    st.markdown("### 📈 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced phase distribution
        if len(filtered) > 0:
            phase_counts = filtered["phase"].value_counts()
            fig_phase = px.pie(
                values=phase_counts.values, 
                names=phase_counts.index,
                title="Phase Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3,
                template="plotly_dark"
            )
            fig_phase.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_phase, use_container_width=True)
    
    with col2:
        # Key usage with better visualization
        if len(filtered) > 0:
            key_usage = {}
            for k in key_list:
                colname = f"key_{k}"
                if colname in filtered:
                    key_usage[k] = int(filtered[colname].sum())
            
            if key_usage:
                fig_keys = px.bar(
                    x=list(key_usage.keys()),
                    y=list(key_usage.values()),
                    title="Key Usage Distribution",
                    color=list(key_usage.values()),
                    color_continuous_scale="viridis",
                    template="plotly_dark"
                )
                fig_keys.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_keys, use_container_width=True)
    
    # Multi-metric timeline
    if len(filtered) > 0 and "timestamp" in filtered.columns:
        st.markdown("#### ⏰ Timeline Analysis")
        
        # Create subplot with multiple metrics
        fig_timeline = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Hold Duration Over Time", "Active Keys Over Time", 
                          "Image Intensity Over Time", "Complexity Score Over Time"),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # Sort by timestamp for proper timeline
        timeline_data = filtered.sort_values('timestamp') if filtered['timestamp'].nunique() > 1 else filtered
        
        fig_timeline.add_trace(
            go.Scatter(x=timeline_data.index, y=timeline_data["avg_hold_duration"],
                      name="Hold Duration", line=dict(color="cyan")),
            row=1, col=1
        )
        
        fig_timeline.add_trace(
            go.Scatter(x=timeline_data.index, y=timeline_data["num_active"],
                      name="Active Keys", line=dict(color="orange")),
            row=1, col=2
        )
        
        fig_timeline.add_trace(
            go.Scatter(x=timeline_data.index, y=timeline_data["mean_intensity"],
                      name="Intensity", line=dict(color="green")),
            row=2, col=1
        )
        
        fig_timeline.add_trace(
            go.Scatter(x=timeline_data.index, y=timeline_data["complexity_score"],
                      name="Complexity", line=dict(color="red")),
            row=2, col=2
        )
        
        fig_timeline.update_layout(
            height=600, 
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

with tab2:
    st.markdown("### 🔬 Advanced Analytics")
    
    # Chart filters
    with st.expander("📊 Chart Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            show_top_n = st.slider("Top N Items", 5, 25, 10)
        with col2:
            chart_phase_filter = st.multiselect(
                "Chart Phase Filter", 
                options=sorted(df["phase"].unique()), 
                default=sorted(df["phase"].unique()),
                key="chart_phase"
            )
        with col3:
            min_combo_count = st.slider("Min Combination Count", 1, 20, 2)
    
    chart_filtered = filtered[filtered["phase"].isin(chart_phase_filter)]
    
    # Key combination analysis with enhanced visualization
    st.markdown("#### 🔗 Advanced Key Combination Analysis")
    if len(chart_filtered) > 0:
        combo_counter = Counter(tuple(sorted(ks)) for ks in chart_filtered["active_keys"])
        common_combos = [item for item in combo_counter.most_common(show_top_n) if item[1] >= min_combo_count]
        
        if common_combos:
            combo_names = [', '.join(combo[0]) if combo[0] else 'None' for combo in common_combos]
            combo_counts = [combo[1] for combo in common_combos]
            
            # Enhanced horizontal bar chart
            fig_combo = px.bar(
                x=combo_counts,
                y=combo_names,
                orientation='h',
                title=f"Top {len(common_combos)} Key Combinations",
                color=combo_counts,
                color_continuous_scale="plasma",
                template="plotly_dark"
            )
            fig_combo.update_layout(
                height=max(400, len(common_combos) * 30),
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_combo, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Advanced boxplot with statistical annotations
        st.markdown("#### 📦 Statistical Distribution by Phase")
        if len(chart_filtered) > 0:
            fig_box = px.box(
                chart_filtered, 
                x="phase", 
                y="avg_hold_duration",
                title="Hold Duration Distribution by Phase",
                color="phase",
                template="plotly_dark"
            )
            
            # Add mean markers
            for phase in chart_filtered["phase"].unique():
                phase_data = chart_filtered[chart_filtered["phase"] == phase]
                mean_val = phase_data["avg_hold_duration"].mean()
                fig_box.add_hline(y=mean_val, line_dash="dash", line_color="red",
                                annotation_text=f"Mean: {mean_val:.2f}")
            
            fig_box.update_layout(height=400, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Correlation heatmap
        st.markdown("#### 🔥 Feature Correlation Matrix")
        numeric_cols = ["num_active", "avg_hold_duration", "mean_intensity", "entropy", "complexity_score"]
        correlation_data = chart_filtered[numeric_cols].corr()
        
        fig_corr = px.imshow(
            correlation_data,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            template="plotly_dark"
        )
        fig_corr.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # 3D scatter plot for multi-dimensional analysis
    if len(chart_filtered) > 0:
        st.markdown("#### 🌐 3D Feature Space Analysis")
        fig_3d = px.scatter_3d(
            chart_filtered,
            x="avg_hold_duration",
            y="mean_intensity", 
            z="entropy",
            color="phase",
            size="num_active",
            title="3D Feature Space: Duration vs Intensity vs Entropy",
            template="plotly_dark"
        )
        fig_3d.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.markdown("### 📊 Statistical Analysis")
    
    if len(filtered) > 0:
        # Statistical summary
        st.markdown("#### 📈 Descriptive Statistics")
        numeric_columns = ["num_active", "avg_hold_duration", "mean_intensity", "entropy", "complexity_score"]
        stats_df = filtered[numeric_columns].describe().round(3)
        st.dataframe(stats_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution analysis
            st.markdown("#### 📊 Distribution Analysis")
            selected_metric = st.selectbox(
                "Select Metric for Distribution Analysis",
                options=numeric_columns,
                index=0
            )
            
            if selected_metric in filtered.columns:
                # Histogram with statistical overlay
                fig_dist = px.histogram(
                    filtered, 
                    x=selected_metric,
                    nbins=30,
                    title=f"{selected_metric.replace('_', ' ').title()} Distribution",
                    template="plotly_dark",
                    marginal="box"
                )
                
                # Add statistical lines
                mean_val = filtered[selected_metric].mean()
                median_val = filtered[selected_metric].median()
                
                fig_dist.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                                 annotation_text=f"Mean: {mean_val:.3f}")
                fig_dist.add_vline(x=median_val, line_dash="dash", line_color="green",
                                 annotation_text=f"Median: {median_val:.3f}")
                
                fig_dist.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Phase comparison analysis
            st.markdown("#### 📈 Phase Comparison")
            comparison_metric = st.selectbox(
                "Select Metric for Phase Comparison",
                options=numeric_columns,
                index=1,
                key="phase_comparison"
            )
            
            # Violin plot for phase comparison
            fig_violin = px.violin(
                filtered,
                x="phase",
                y=comparison_metric,
                title=f"{comparison_metric.replace('_', ' ').title()} by Phase",
                template="plotly_dark",
                box=True
            )
            fig_violin.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_violin, use_container_width=True)
        
        # Statistical tests
        st.markdown("#### 🧪 Statistical Tests")
        if len(filtered["phase"].unique()) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                # ANOVA test for phases
                st.markdown("**ANOVA Test Results**")
                test_metric = st.selectbox("Test metric", numeric_columns, key="anova_metric")
                
                groups = [filtered[filtered["phase"] == phase][test_metric].values 
                         for phase in filtered["phase"].unique()]
                groups = [g for g in groups if len(g) > 0]  # Remove empty groups
                
                if len(groups) > 1:
                    f_stat, p_value = stats.f_oneway(*groups)
                    st.write(f"F-statistic: {f_stat:.4f}")
                    st.write(f"P-value: {p_value:.4f}")
                    st.write(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")
            
            with col2:
                # Correlation analysis
                st.markdown("**Correlation Analysis**")
                corr_x = st.selectbox("X variable", numeric_columns, key="corr_x")
                corr_y = st.selectbox("Y variable", numeric_columns, index=1, key="corr_y")
                
                if corr_x != corr_y:
                    correlation = filtered[corr_x].corr(filtered[corr_y])
                    st.write(f"Pearson correlation: {correlation:.4f}")
                    
                    # Significance test
                    stat, p_val = stats.pearsonr(filtered[corr_x], filtered[corr_y])
                    st.write(f"P-value: {p_val:.4f}")
                    st.write(f"Significant: {'Yes' if p_val < 0.05 else 'No'}")

with tab4:
    st.markdown("### 🖼️ Image Analysis & Computer Vision Insights")
    
    if len(filtered) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Image intensity analysis
            st.markdown("#### 🌈 Intensity Distribution Analysis")
            
            # Multi-dimensional intensity plot
            fig_intensity = px.scatter(
                filtered,
                x="mean_intensity",
                y="std_intensity",
                color="phase",
                size="entropy",
                title="Image Intensity vs Variability",
                hover_data=["num_active", "avg_hold_duration"],
                template="plotly_dark"
            )
            fig_intensity.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_intensity, use_container_width=True)
            
            # Intensity range analysis
            fig_range = px.histogram(
                filtered,
                x="intensity_range",
                color="phase",
                title="Image Intensity Range Distribution",
                template="plotly_dark",
                nbins=25
            )
            fig_range.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_range, use_container_width=True)
        
        with col2:
            # Entropy and complexity analysis
            st.markdown("#### 🧠 Complexity & Information Content")
            
            # Entropy vs complexity scatter
            fig_entropy = px.scatter(
                filtered,
                x="entropy",
                y="complexity_score",
                color="phase",
                size="num_active",
                title="Entropy vs Complexity Score",
                hover_data=["avg_hold_duration"],
                template="plotly_dark"
            )
            fig_entropy.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_entropy, use_container_width=True)
            
            # Image quality metrics
            st.markdown("#### 📊 Image Quality Metrics")
            quality_metrics = pd.DataFrame({
                'Metric': ['Mean Intensity', 'Std Intensity', 'Min Intensity', 'Max Intensity', 'Entropy'],
                'Value': [
                    filtered["mean_intensity"].mean(),
                    filtered["std_intensity"].mean(), 
                    filtered["min_intensity"].mean(),
                    filtered["max_intensity"].mean(),
                    filtered["entropy"].mean()
                ]
            })
            
            fig_quality = px.bar(
                quality_metrics,
                x="Metric",
                y="Value",
                title="Average Image Quality Metrics",
                color="Value",
                color_continuous_scale="viridis",
                template="plotly_dark"
            )
            fig_quality.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_quality, use_container_width=True)
        
        # Advanced image analysis
        st.markdown("#### 🔍 Advanced Image Pattern Analysis")
        
        # Clustering analysis based on image features
        if len(filtered) >= 10:  # Only show if enough data
            col1, col2 = st.columns(2)
            
            with col1:
                # Image features by phase (parallel coordinates)
                img_features = ["mean_intensity", "std_intensity", "entropy"]
                filtered['phase_num'] = filtered['phase'].astype('category').cat.codes

                fig_parallel = px.parallel_coordinates(
                    filtered.sample(min(100, len(filtered))),
                    dimensions=img_features,
                    color="phase_num",   # Sayısal!
                    title="Image Feature Patterns by Phase",
                    template="plotly_dark"
                )
                fig_parallel.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_parallel, use_container_width=True)
            
            with col2:
                # Complexity evolution
                if filtered.index.nunique() > 1:
                    # Create evolution plot
                    evolution_data = filtered.reset_index().sort_values('index')
                    
                    fig_evolution = go.Figure()
                    fig_evolution.add_trace(go.Scatter(
                        x=evolution_data.index,
                        y=evolution_data["complexity_score"],
                        mode='lines+markers',
                        name='Complexity Score',
                        line=dict(color='cyan', width=2)
                    ))
                    
                    fig_evolution.add_trace(go.Scatter(
                        x=evolution_data.index,
                        y=evolution_data["entropy"],
                        mode='lines+markers',
                        name='Entropy',
                        yaxis="y2",
                        line=dict(color='orange', width=2)
                    ))
                    
                    fig_evolution.update_layout(
                        title="Complexity & Entropy Evolution",
                        xaxis_title="Entry Index",
                        yaxis_title="Complexity Score",
                        yaxis2=dict(title="Entropy", overlaying="y", side="right"),
                        height=400,
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_evolution, use_container_width=True)

with tab5:
    st.markdown("### 📋 Advanced Data Explorer")
    
    # Enhanced table controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        search_term = st.text_input("🔍 Search", placeholder="Search phases, keys...")
    with col2:
        items_per_page = st.selectbox("Items per page", [10, 25, 50, 100], index=1)
    with col3:
        sort_by = st.selectbox("Sort by", ["original_index", "phase", "num_active", "avg_hold_duration", "complexity_score"])
    with col4:
        sort_order = st.selectbox("Sort order", ["Ascending", "Descending"])
    
    # Advanced filtering and sorting
    table_data = filtered.reset_index().rename(columns={"index": "original_index"})
    
    if search_term:
        search_mask = (
            table_data["phase"].astype(str).str.contains(search_term, case=False, na=False) |
            table_data["active_keys"].astype(str).str.contains(search_term, case=False, na=False)
        )
        table_data = table_data[search_mask]
    
    # Apply sorting
    ascending = sort_order == "Ascending"
    table_data = table_data.sort_values(sort_by, ascending=ascending)
    
    # Enhanced pagination with navigation
    total_items = len(table_data)
    total_pages = (total_items - 1) // items_per_page + 1 if total_items > 0 else 1
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, 
                              help=f"Navigate through {total_pages} pages") - 1
    
    start_idx = page * items_per_page
    end_idx = start_idx + items_per_page
    
    # Enhanced column configuration
    show_cols = ["original_index", "phase", "num_active", "avg_hold_duration", "complexity_score", 
                "mean_intensity", "entropy", "active_keys"]
    page_data = table_data[show_cols].iloc[start_idx:end_idx]
    
    st.dataframe(
        page_data,
        use_container_width=True,
        column_config={
            "original_index": st.column_config.NumberColumn("Index", width="small"),
            "phase": st.column_config.TextColumn("Phase", width="medium"),
            "num_active": st.column_config.NumberColumn("Active Keys", width="small"),
            "avg_hold_duration": st.column_config.NumberColumn("Avg Duration", format="%.3f", width="medium"),
            "complexity_score": st.column_config.NumberColumn("Complexity", format="%.2f", width="medium"),
            "mean_intensity": st.column_config.NumberColumn("Mean Intensity", format="%.3f", width="medium"),
            "entropy": st.column_config.NumberColumn("Entropy", format="%.3f", width="medium"),
            "active_keys": st.column_config.ListColumn("Active Keys", width="large"),
        }
    )
    
    # Enhanced export options
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"Showing {start_idx + 1}-{min(end_idx, total_items)} of {total_items} entries")
    with col2:
        if st.button("📥 Download Filtered CSV"):
            csv = page_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"npz_analysis_filtered_{len(filtered)}_entries.csv",
                mime="text/csv"
            )
    with col3:
        if st.button("📊 Download Statistics"):
            stats_csv = filtered[numeric_columns].describe().to_csv()
            st.download_button(
                label="Download Stats CSV",
                data=stats_csv,
                file_name="npz_statistics.csv",
                mime="text/csv"
            )


with tab6:
    st.markdown("### 🎯 Single Entry Inspector")

    # Session state initialization
    if "filtered_idx" not in st.session_state:
        st.session_state.filtered_idx = 0

    filtered_len = len(filtered)
    if filtered_len == 0:
        st.warning("No entries match the current filters.")
        st.stop()

    orig_idx_list = list(filtered.index)  # Original indices in data

    # SECTION 1: Navigation Controls
    st.markdown("#### 🧭 Navigation")
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([2, 1, 1, 2])
    
    with nav_col1:
        # Current position info
        current_orig_idx = orig_idx_list[st.session_state.filtered_idx]
        st.info(f"**Entry:** {st.session_state.filtered_idx + 1} of {filtered_len} (Original Index: {current_orig_idx})")
        
        # Direct navigation input
        new_idx = st.number_input(
            "Go to entry:",
            min_value=1,
            max_value=filtered_len,
            value=st.session_state.filtered_idx + 1,
            step=1
        ) - 1
        
        if new_idx != st.session_state.filtered_idx:
            st.session_state.filtered_idx = int(new_idx)
            st.rerun()
    
    with nav_col2:
        if st.button("⬅️ Previous", use_container_width=True, disabled=st.session_state.filtered_idx <= 0):
            st.session_state.filtered_idx -= 1
            st.rerun()
    
    with nav_col3:
        if st.button("➡️ Next", use_container_width=True, disabled=st.session_state.filtered_idx >= filtered_len - 1):
            st.session_state.filtered_idx += 1
            st.rerun()
    
    with nav_col4:
        nav_action_col1, nav_action_col2 = st.columns(2)
        with nav_action_col1:
            if st.button("🎲 Random", use_container_width=True):
                st.session_state.filtered_idx = np.random.randint(0, filtered_len)
                st.rerun()
        with nav_action_col2:
            if st.button("🔍 Similar", use_container_width=True):
                # Find similar entries
                current_entry = filtered.iloc[st.session_state.filtered_idx]
                similarity_scores = []
                for i, (idx, row) in enumerate(filtered.iterrows()):
                    if i != st.session_state.filtered_idx:
                        score = (
                            abs(row["avg_hold_duration"] - current_entry["avg_hold_duration"]) + 
                            abs(row["mean_intensity"] - current_entry["mean_intensity"]) + 
                            abs(row["entropy"] - current_entry["entropy"])
                        )
                        similarity_scores.append((i, score))
                
                if similarity_scores:
                    most_similar_filtered_idx = min(similarity_scores, key=lambda x: x[1])[0]
                    st.session_state.filtered_idx = most_similar_filtered_idx
                    st.rerun()

    st.divider()

    # SECTION 2: Current Entry Analysis
    current_orig_idx = orig_idx_list[st.session_state.filtered_idx]
    entry = data[int(current_orig_idx)]
    info = entry_to_dict(entry)
    
    st.markdown(f"#### 📊 Entry Analysis - #{st.session_state.filtered_idx + 1}")
    
    # Basic info in columns
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("""
        **📋 Basic Information**
        """)
        st.metric("Phase", info['phase'])
        st.metric("Active Keys Count", info['num_active'])
        if info['active_keys']:
            st.write("**Active Keys:**")
            for key in info['active_keys']:
                st.write(f"• {key}")
        else:
            st.write("**Active Keys:** None")
    
    with info_col2:
        st.markdown("""
        **⏱️ Duration Statistics**
        """)
        st.metric("Average Duration", f"{info['avg_hold_duration']:.3f}s")
        st.metric("Max Duration", f"{info['max_hold_duration']:.3f}s")
        st.metric("Min Duration", f"{info['min_hold_duration']:.3f}s")
        st.metric("Std Deviation", f"{info['std_hold_duration']:.3f}s")
    
    with info_col3:
        st.markdown("""
        **🖼️ Image Properties**
        """)
        st.metric("Mean Intensity", f"{info.get('mean_intensity', 0):.3f}")
        st.metric("Std Intensity", f"{info.get('std_intensity', 0):.3f}")
        st.metric("Entropy", f"{info.get('entropy', 0):.3f}")
        complexity = filtered.iloc[st.session_state.filtered_idx]['complexity_score']
        st.metric("Complexity Score", f"{complexity:.3f}")

    # Image visualization
    if len(entry) > 0:
        st.markdown("#### 🖼️ Image Visualization")
        img_col1, img_col2 = st.columns([3, 1])
        
        with img_col1:
            img_arr = (entry[0] * 255).astype(np.uint8)
            st.image(
                img_arr,
                caption=f"Entry {current_orig_idx} - Phase: {info['phase']} | Keys: {len(info['active_keys'])}",
                use_container_width=True
            )
        
        with img_col2:
            # Pixel intensity histogram
            fig_hist = px.histogram(
                x=img_arr.flatten(),
                nbins=50,
                title="Pixel Intensity",
                template="plotly_dark",
                labels={'x': 'Intensity', 'y': 'Count'}
            )
            fig_hist.update_layout(
                height=300, 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # Duration breakdown
    if len(entry) > 3 and len(entry[3]) > 0:
        st.markdown("#### ⏱️ Duration Breakdown")
        durations = entry[3]
        duration_df = pd.DataFrame({
            'Key': key_list[:len(durations)],
            'Duration': durations,
            'Active': ['Yes' if d > 0 else 'No' for d in durations]
        })
        
        dur_col1, dur_col2 = st.columns(2)
        
        with dur_col1:
            # All keys duration bar chart
            fig_dur_bar = px.bar(
                duration_df,
                x="Key",
                y="Duration",
                color="Active",
                title="Hold Durations by Key",
                template="plotly_dark",
                color_discrete_map={'Yes': '#00cc96', 'No': '#636363'}
            )
            fig_dur_bar.update_layout(
                height=350, 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_dur_bar, use_container_width=True)
        
        with dur_col2:
            # Active keys pie chart
            active_durations = duration_df[duration_df["Duration"] > 0]
            if len(active_durations) > 0:
                fig_dur_pie = px.pie(
                    active_durations,
                    values="Duration",
                    names="Key",
                    title="Active Keys Duration Share",
                    template="plotly_dark"
                )
                fig_dur_pie.update_layout(
                    height=350, 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_dur_pie, use_container_width=True)
            else:
                st.info("No active keys with duration > 0")

    st.divider()

    # SECTION 3: Entry Comparison
    st.markdown("#### ⚖️ Compare with Another Entry")
    
    comp_col1, comp_col2 = st.columns([1, 3])
    
    with comp_col1:
        compare_idx = st.number_input(
            "Select entry to compare:",
            min_value=1,
            max_value=filtered_len,
            value=min(st.session_state.filtered_idx + 2, filtered_len),
            step=1,
            help="Choose another entry for side-by-side comparison"
        ) - 1
        
        if compare_idx == st.session_state.filtered_idx:
            st.warning("⚠️ Select a different entry for comparison")
        else:
            compare_orig_idx = orig_idx_list[compare_idx]
            st.success(f"✅ Comparing with Entry #{compare_idx + 1} (Original: {compare_orig_idx})")
    
    with comp_col2:
        if compare_idx != st.session_state.filtered_idx and 0 <= compare_idx < filtered_len:
            # Get comparison entry data
            compare_orig_idx = orig_idx_list[compare_idx]
            compare_entry = data[int(compare_orig_idx)]
            compare_info = entry_to_dict(compare_entry)
            
            # Side-by-side comparison
            comp_display_col1, comp_display_col2 = st.columns(2)
            
            with comp_display_col1:
                st.markdown(f"**Current Entry #{st.session_state.filtered_idx + 1}**")
                if len(entry) > 0:
                    comp_img1 = (entry[0] * 255).astype(np.uint8)
                    st.image(comp_img1, caption=f"Entry {current_orig_idx}", use_container_width=True)
            
            with comp_display_col2:
                st.markdown(f"**Comparison Entry #{compare_idx + 1}**")
                if len(compare_entry) > 0:
                    comp_img2 = (compare_entry[0] * 255).astype(np.uint8)
                    st.image(comp_img2, caption=f"Entry {compare_orig_idx}", use_container_width=True)
            
            # Comparison metrics table
            st.markdown("**📊 Comparison Metrics**")
            comparison_metrics = pd.DataFrame({
                'Metric': [
                    'Phase', 
                    'Active Keys Count', 
                    'Average Duration (s)', 
                    'Mean Intensity', 
                    'Image Entropy', 
                    'Complexity Score'
                ],
                f'Entry #{st.session_state.filtered_idx + 1}': [
                    info['phase'],
                    info['num_active'],
                    f"{info['avg_hold_duration']:.3f}",
                    f"{info.get('mean_intensity', 0):.3f}",
                    f"{info.get('entropy', 0):.3f}",
                    f"{filtered.iloc[st.session_state.filtered_idx]['complexity_score']:.3f}"
                ],
                f'Entry #{compare_idx + 1}': [
                    compare_info['phase'],
                    compare_info['num_active'],
                    f"{compare_info['avg_hold_duration']:.3f}",
                    f"{compare_info.get('mean_intensity', 0):.3f}",
                    f"{compare_info.get('entropy', 0):.3f}",
                    f"{filtered.iloc[compare_idx]['complexity_score']:.3f}"
                ]
            })
            
            st.dataframe(comparison_metrics, use_container_width=True, hide_index=True)
            
            # Difference highlights
            st.markdown("**🔍 Key Differences**")
            diff_col1, diff_col2, diff_col3 = st.columns(3)
            
            with diff_col1:
                duration_diff = info['avg_hold_duration'] - compare_info['avg_hold_duration']
                duration_color = "🟢" if duration_diff > 0 else "🔴" if duration_diff < 0 else "🟡"
                st.write(f"{duration_color} Duration: {duration_diff:+.3f}s")
            
            with diff_col2:
                intensity_diff = info.get('mean_intensity', 0) - compare_info.get('mean_intensity', 0)
                intensity_color = "🟢" if intensity_diff > 0 else "🔴" if intensity_diff < 0 else "🟡"
                st.write(f"{intensity_color} Intensity: {intensity_diff:+.3f}")
            
            with diff_col3:
                key_diff = info['num_active'] - compare_info['num_active']
                key_color = "🟢" if key_diff > 0 else "🔴" if key_diff < 0 else "🟡"
                st.write(f"{key_color} Active Keys: {key_diff:+d}")

    # Quick navigation at bottom
    st.divider()
    st.markdown("#### 🚀 Quick Actions")
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    
    with quick_col1:
        if st.button("🏠 First Entry", use_container_width=True):
            st.session_state.filtered_idx = 0
            st.rerun()
    
    with quick_col2:
        if st.button("🎯 Middle Entry", use_container_width=True):
            st.session_state.filtered_idx = filtered_len // 2
            st.rerun()
    
    with quick_col3:
        if st.button("🏁 Last Entry", use_container_width=True):
            st.session_state.filtered_idx = filtered_len - 1
            st.rerun()
    
    with quick_col4:
        # Find most complex entry
        if st.button("🔥 Most Complex", use_container_width=True):
            most_complex_idx = filtered['complexity_score'].idxmax()
            # Find the position of this index in our filtered list
            st.session_state.filtered_idx = orig_idx_list.index(most_complex_idx)
            st.rerun()

# Advanced insights sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🧠 AI Insights")
    
    if len(filtered) > 0:
        # Automated insights
        insights = []
        
        # Phase with highest complexity
        if "complexity_score" in filtered.columns:
            max_complexity_phase = filtered.groupby("phase")["complexity_score"].mean().idxmax()
            insights.append(f"🏆 **{max_complexity_phase}** phase shows highest complexity")
        
        # Most active key
        key_activity = {}
        for k in key_list:
            if f"key_{k}" in filtered.columns:
                key_activity[k] = filtered[f"key_{k}"].sum()
        if key_activity:
            most_active_key = max(key_activity, key=key_activity.get)
            insights.append(f"🔑 **{most_active_key}** is the most frequently used key")
        
        # Duration patterns
        if filtered["avg_hold_duration"].std() > 0:
            cv = filtered["avg_hold_duration"].std() / filtered["avg_hold_duration"].mean()
            if cv > 0.5:
                insights.append("⚡ High variability in hold durations detected")
            else:
                insights.append("📊 Consistent hold duration patterns observed")
        
        # Display insights
        for insight in insights:
            st.markdown(insight)
    
    st.markdown("---")
    st.markdown("### 📊 Export Options")
    
    if st.button("📄 Generate Report", use_container_width=True):
        # Generate comprehensive analysis report
        report = f"""
        # NPZ Dataset Analysis Report
        
        ## Dataset Overview
        - Total Entries: {len(df)}
        - Filtered Entries: {len(filtered)}
        - Available Keys: {len(key_list)}
        - Phases: {', '.join(sorted(df['phase'].unique()))}
        
        ## Key Statistics
        - Average Hold Duration: {filtered['avg_hold_duration'].mean():.3f}s
        - Average Active Keys: {filtered['num_active'].mean():.1f}
        - Average Complexity Score: {filtered['complexity_score'].mean():.3f}
        - Average Image Entropy: {filtered['entropy'].mean():.3f}
        
        ## Phase Analysis
        {filtered.groupby('phase')[['avg_hold_duration', 'num_active', 'complexity_score']].mean().round(3).to_string()}
        
        ## Key Usage Statistics
        {pd.Series({k: filtered[f'key_{k}'].sum() for k in key_list if f'key_{k}' in filtered.columns}).to_string()}
        """
        
        st.download_button(
            label="📄 Download Analysis Report",
            data=report,
            file_name="npz_analysis_report.md",
            mime="text/markdown"
        )

# Footer with enhanced information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1.5rem; background: #1e1e1e; border-radius: 8px; margin-top: 2rem;">
    <h4>🚀 Advanced NPZ Dataset Analytics Platform</h4>
    <p>Professional-grade dataset analysis with machine learning insights and statistical visualization</p>
    <p><em>Powered by Streamlit, Plotly, and Advanced Analytics</em></p>
</div>
""", unsafe_allow_html=True)