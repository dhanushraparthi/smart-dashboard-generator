# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import json

st.set_page_config(page_title="Smart Dashboard Generator", layout="wide", page_icon="📊")
st.title("📊 Smart Dashboard Generator")

# ------------------- Sidebar -------------------
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "json"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file type!")
        st.stop()
    
    st.sidebar.header("Filters")
    filter_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    filters = {}
    for col in filter_cols:
        options = df[col].dropna().unique().tolist()
        selected = st.sidebar.multiselect(f"Filter {col}", options, default=options)
        filters[col] = selected
        df = df[df[col].isin(selected)]
    
    # ------------------- Key Metrics -------------------
    st.header("📈 Key Metrics")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    kpi_list = []
    if "Sales" in df.columns:
        total_sales = df["Sales"].sum()
        st.metric("Total Sales", f"${total_sales:,.2f}")
        kpi_list.append(f"Total Sales: ${total_sales:,.2f}")
    
    if "Quantity" in df.columns:
        total_qty = df["Quantity"].sum()
        st.metric("Total Quantity", total_qty)
        kpi_list.append(f"Total Quantity: {total_qty}")
    
    if "Profit" in df.columns:
        total_profit = df["Profit"].sum()
        st.metric("Total Profit", f"${total_profit:,.2f}")
        kpi_list.append(f"Total Profit: ${total_profit:,.2f}")
    
    # Highest sale
    if "Sales" in df.columns:
        max_sale_row = df.loc[df["Sales"].idxmax()]
        st.metric("Highest Sale", f"${max_sale_row['Sales']:,.2f} ({max_sale_row.get('Product Name', 'N/A')})")
        kpi_list.append(f"Highest Sale: ${max_sale_row['Sales']:,.2f} ({max_sale_row.get('Product Name', 'N/A')})")
    
    # ------------------- AI Insights -------------------
    st.header("🤖 AI Insights")
    ai_insights = []
    if "Profit" in df.columns and "Sales" in df.columns:
        profit_margin = total_profit / total_sales * 100 if total_sales != 0 else 0
        insight = f"Overall profit margin is {profit_margin:.2f}%."
        st.write(insight)
        ai_insights.append(insight)
    
    if "Quantity" in df.columns:
        top_selling_qty = df.groupby("Product Name")["Quantity"].sum().idxmax()
        insight = f"Top-selling product by quantity: {top_selling_qty}."
        st.write(insight)
        ai_insights.append(insight)
    
    # ------------------- Visual Dashboards -------------------
    st.header("📊 Visual Dashboards")
    neon_colors = px.colors.qualitative.Plotly
    
    # Categorical bar charts
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        if df[col].dropna().empty:
            continue
        df_counts = df[col].value_counts().reset_index()
        df_counts.columns = [col, 'count']
        fig = px.bar(
            df_counts,
            x=col,
            y='count',
            text_auto=True,
            title=f"{col.capitalize()} Distribution",
            color=col,
            color_discrete_sequence=neon_colors
        )
        fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    
    # Numeric histograms
    for col in numeric_cols:
        if df[col].dropna().empty:
            continue
        fig = px.histogram(
            df,
            x=col,
            nbins=20,
            title=f"{col.capitalize()} Distribution",
            color_discrete_sequence=neon_colors
        )
        fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')
        st.plotly_chart(fig, use_container_width=True)
    
    # ------------------- Shareable Link -------------------
    st.sidebar.header("Shareable Link")
    params = {col: filters[col] for col in filters}
    st.sidebar.code(f"Your shareable link: ?{json.dumps(params)}")
    
else:
    st.info("Upload a CSV, XLSX, or JSON file to generate dashboards.")
