# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import tempfile

# Optional OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Smart Data Dashboard", layout="wide", page_icon="📊")
st.title("📊 Smart Data Dashboard")
st.markdown("Upload your dataset (CSV/Excel/JSON) and explore KPIs, visual insights, and AI-style insights.")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload dataset", type=["csv","xlsx","xls","json"])
if not uploaded_file:
    st.info("Please upload a dataset to begin.")
    st.stop()

# ---------------- Load File ----------------
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

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
st.success("File loaded successfully!")
st.dataframe(df.head())

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")
filters = {}
for col in df.select_dtypes(include=['object','category']).columns:
    options = st.sidebar.multiselect(f"Filter {col.capitalize()}", df[col].unique(), default=df[col].unique())
    filters[col] = options

# Apply filters
filtered_df = df.copy()
for col, selected in filters.items():
    filtered_df = filtered_df[filtered_df[col].isin(selected)]

# ---------------- KPIs ----------------
st.subheader("Key Metrics")
kpi_cols = st.columns(4)
total_sales = filtered_df['sales'].sum() if 'sales' in filtered_df.columns else None
total_profit = filtered_df['profit'].sum() if 'profit' in filtered_df.columns else None
avg_discount = filtered_df['discount'].mean() if 'discount' in filtered_df.columns else None
total_quantity = filtered_df['quantity'].sum() if 'quantity' in filtered_df.columns else None

if total_sales is not None:
    kpi_cols[0].metric("Total Sales", f"${total_sales:,.2f}")
if total_profit is not None:
    kpi_cols[1].metric("Total Profit", f"${total_profit:,.2f}")
if avg_discount is not None:
    kpi_cols[2].metric("Avg Discount", f"{avg_discount:.2%}")
if total_quantity is not None:
    kpi_cols[3].metric("Total Quantity", f"{int(total_quantity):,}")

# ---------------- Visual Insights ----------------
st.subheader("Visual Insights")
plots = {}
numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = filtered_df.select_dtypes(include=['object','category']).columns.tolist()

for col in categorical_cols:
    st.markdown(f"### {col.capitalize()} Distribution")
    counts = filtered_df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(counts, x=col, y="count", color=col, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

for col in numeric_cols:
    st.markdown(f"### {col.capitalize()} Histogram")
    fig = px.histogram(filtered_df, x=col, nbins=20, color_discrete_sequence=["#00CC96"])
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# ---------------- AI Insights ----------------
st.subheader("AI / Heuristic Insights")
def heuristic_insights(df):
    insights = []
    if 'sales' in df.columns:
        insights.append(f"Total Sales: ${df['sales'].sum():,.2f}")
    if 'profit' in df.columns:
        insights.append(f"Total Profit: ${df['profit'].sum():,.2f}")
    if 'discount' in df.columns and 'profit' in df.columns:
        corr = df['discount'].corr(df['profit'])
        insights.append(f"Discount vs Profit correlation: {corr:.2f}")
    if 'sales' in df.columns and 'profit' in df.columns:
        margin = df['profit'].sum() / df['sales'].sum() if df['sales'].sum() != 0 else 0
        insights.append(f"Overall profit margin: {margin:.2%}")
    if not insights:
        insights.append("Not enough data for insights.")
    return insights

ai_text = "\n".join(heuristic_insights(filtered_df))

# Optional OpenAI summary
openai_key = os.environ.get("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None)
if openai_key and OPENAI_AVAILABLE:
    try:
        openai.api_key = openai_key
        prompt = "Summarize the following dataset insights in 3 short bullet points:\n" + ai_text
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=200,
            temperature=0.2
        )
        ai_text = response.choices[0].message.content.strip()
    except Exception:
        pass

for line in ai_text.split("\n"):
    st.markdown(f"- {line}")

# ---------------- Download Charts ----------------
st.subheader("Download Charts")
for name, fig in plots.items():
    st.markdown(f"**Download {name.capitalize()} chart:**")
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.write_image(tmp.name)
    tmp.close()
    with open(tmp.name, "rb") as f:
        st.download_button(f"📥 Download {name} chart", data=f, file_name=f"{name}.png", mime="image/png")
    os.remove(tmp.name)
