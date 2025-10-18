import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Smart Data Dashboard", layout="wide")
st.title("📊 Smart Data Dashboard")
st.markdown("Upload CSV / Excel / JSON and explore insights. Share your filters via the URL!")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload dataset", type=["csv","xlsx","xls","json"])
if not uploaded_file:
    st.info("Please upload a dataset to begin.")
    st.stop()

def load_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    if name.endswith(".json"):
        return pd.read_json(uploaded)
    raise ValueError("Unsupported file type")

try:
    df = load_file(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
st.success("File loaded")
st.dataframe(df.head())

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filter Options")
filters = {}
for col in df.select_dtypes(include=['object', 'category']).columns:
    unique_vals = df[col].unique().tolist()
    # Read from query params if available
    query_params = st.experimental_get_query_params()
    default_vals = query_params.get(col, unique_vals)
    selected = st.sidebar.multiselect(f"Filter by {col}", unique_vals, default=default_vals)
    filters[col] = selected
    if selected:
        df = df[df[col].isin(selected)]

# Save filters in URL
params = {k:v for k,v in filters.items() if v}
st.experimental_set_query_params(**params)

# ---------------- KPIs ----------------
st.subheader("Key Metrics")
kpi_cols = st.columns(4)
if "sales" in df.columns:
    total_sales = df["sales"].sum()
    kpi_cols[0].metric("Total Sales", f"${total_sales:,.2f}")
else: total_sales = None
if "profit" in df.columns:
    total_profit = df["profit"].sum()
    kpi_cols[1].metric("Total Profit", f"${total_profit:,.2f}")
else: total_profit = None
if "quantity" in df.columns:
    total_quantity = df["quantity"].sum()
    kpi_cols[2].metric("Total Quantity", f"{int(total_quantity):,}")
else: total_quantity = None
if "discount" in df.columns:
    avg_discount = df["discount"].mean()
    kpi_cols[3].metric("Avg Discount", f"{avg_discount:.2%}")
else: avg_discount = None

# ---------------- Plots ----------------
st.subheader("Visual Insights")
plots = {}
for col in df.select_dtypes(include=['object', 'category']).columns:
    top_vals = df[col].value_counts().reset_index()
    top_vals.columns = [col, 'count']
    fig = px.bar(top_vals, x=col, y='count', color=col, text_auto=True,
                 title=f"{col.capitalize()} Distribution")
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# Numeric correlations
numeric_cols = df.select_dtypes(include=['float', 'int']).columns
if len(numeric_cols) > 1:
    st.subheader("Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis')
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------------- AI-style / Heuristic Insights ----------------
st.subheader("Automatic Insights")
insights = []
if "region" in df.columns and "sales" in df.columns:
    top_region = df.groupby("region")["sales"].sum().idxmax()
    insights.append(f"Highest sales region: {top_region}")
if "category" in df.columns and "profit" in df.columns:
    top_category = df.groupby("category")["profit"].sum().idxmax()
    insights.append(f"Most profitable category: {top_category}")
if "discount" in df.columns and "profit" in df.columns:
    corr = df["discount"].corr(df["profit"])
    insights.append(f"Discount vs Profit correlation: {corr:.2f}")
if "sales" in df.columns and "profit" in df.columns:
    margin = df["profit"].sum() / df["sales"].sum() if df["sales"].sum() != 0 else 0
    insights.append(f"Overall profit margin: {margin:.2%}")
if not insights:
    insights.append("Not enough data for detailed insights.")

for i in insights:
    st.markdown(f"- {i}")

st.info("Copy the URL from the browser to share your dashboard with the same filters applied.")
