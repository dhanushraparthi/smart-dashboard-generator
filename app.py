# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import json

st.set_page_config(
    page_title="Smart Dashboard Generator",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

st.title("📊 Smart Dashboard Generator")

# --- Sidebar ---
st.sidebar.header("Upload Your Data")
file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
st.sidebar.markdown("---")

# Filter placeholders
filter_values = {}

# --- Load Data ---
df = None
if file:
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        st.success(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Error loading file: {e}")

if df is not None:
    st.sidebar.header("Filters")
    for col in df.select_dtypes(include=["object", "category"]).columns:
        options = df[col].unique().tolist()
        selected = st.sidebar.multiselect(f"Filter {col}", options, default=options)
        filter_values[col] = selected

    # Apply filters
    filtered_df = df.copy()
    for col, vals in filter_values.items():
        filtered_df = filtered_df[filtered_df[col].isin(vals)]

    st.header("📈 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(filtered_df))
    col2.metric("Unique Categories", len(filtered_df.select_dtypes(include="object").columns))
    col3.metric("Columns Count", filtered_df.shape[1])

    st.header("💡 AI Insights")
    ai_text = ""
    if not filtered_df.empty:
        for col in filtered_df.select_dtypes(include=["number"]).columns:
            mean_val = filtered_df[col].mean()
            ai_text += f"{col}: Mean = {mean_val:.2f}\n"
    st.text(ai_text)

    st.header("📊 Visual Dashboards")
    neon_colors = px.colors.qualitative.Bold
    for col in filtered_df.select_dtypes(include=["object", "category"]).columns:
        counts = filtered_df[col].value_counts().reset_index()
        counts.columns = ["Category", "Count"]
        fig = px.bar(
            counts,
            x="Category",
            y="Count",
            color="Category",
            text_auto=True,
            title=f"{col} Distribution",
            color_discrete_sequence=neon_colors
        )
        fig.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Shareable Link ---
    st.sidebar.header("Share Dashboard")
    try:
        params_str = json.dumps(filter_values)
        st.sidebar.code(f"Your shareable link: ?filters={params_str}")
    except Exception as e:
        st.sidebar.error(f"Error creating shareable link: {e}")

else:
    st.info("Please upload a CSV or Excel file to see the dashboard.")
