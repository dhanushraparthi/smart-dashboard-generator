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

    # --- Key Metrics ---
    st.header("📈 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Records", len(filtered_df))

    numeric_cols = filtered_df.select_dtypes(include=["number"]).columns.tolist()
    if "Sales" in numeric_cols:
        total_sales = filtered_df["Sales"].sum()
        avg_sales = filtered_df["Sales"].mean()
        max_sales = filtered_df["Sales"].max()
        min_sales = filtered_df["Sales"].min()
        col2.metric("Total Sales", f"${total_sales:,.2f}")
        col3.metric("Average Sale", f"${avg_sales:,.2f}")
        col4.metric("Highest Sale", f"${max_sales:,.2f}")
        col5.metric("Lowest Sale", f"${min_sales:,.2f}")
    else:
        col2.metric("Numeric Col Count", len(numeric_cols))
        col3.metric("Average Value", "N/A")
        col4.metric("Max Value", "N/A")
        col5.metric("Min Value", "N/A")

    # --- AI Insights ---
    st.header("💡 AI Insights")
    ai_text = ""
    if not filtered_df.empty and numeric_cols:
        for col in numeric_cols:
            mean_val = filtered_df[col].mean()
            max_val = filtered_df[col].max()
            min_val = filtered_df[col].min()
            ai_text += f"{col}: Mean = {mean_val:.2f}, Max = {max_val:.2f}, Min = {min_val:.2f}\n"

        for col in filtered_df.select_dtypes(include=["object", "category"]).columns:
            top_vals = filtered_df[col].value_counts().head(3)
            ai_text += f"Top {col}s: {', '.join(top_vals.index)}\n"

    st.text(ai_text)

    # --- Visual Dashboards ---
    st.header("📊 Visual Dashboards")
    neon_colors = px.colors.qualitative.Bold

    # All category columns
    category_cols = filtered_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Create bar charts for each category
    for col in category_cols:
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

    # --- Top-selling Products & Top Regions ---
    if "Product" in category_cols:
        st.subheader("🏆 Top-selling Products")
        top_products = filtered_df.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_prod = px.bar(
            top_products,
            x="Product",
            y="Sales",
            text_auto=True,
            color="Product",
            title="Top 10 Products by Sales",
            color_discrete_sequence=neon_colors
        )
        fig_prod.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white'
        )
        st.plotly_chart(fig_prod, use_container_width=True)

    if "Region" in category_cols:
        st.subheader("🌍 Top Regions by Sales")
        top_regions = filtered_df.groupby("Region")["Sales"].sum().sort_values(ascending=False).reset_index()
        fig_reg = px.bar(
            top_regions,
            x="Region",
            y="Sales",
            text_auto=True,
            color="Region",
            title="Sales by Region",
            color_discrete_sequence=neon_colors
        )
        fig_reg.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white'
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    # --- Shareable Link ---
    st.sidebar.header("Share Dashboard")
    try:
        params_str = json.dumps(filter_values)
        st.sidebar.code(f"Your shareable link: ?filters={params_str}")
    except Exception as e:
        st.sidebar.error(f"Error creating shareable link: {e}")

else:
    st.info("Please upload a CSV or Excel file to see the dashboard.")
