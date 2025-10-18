import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
import tempfile
import os

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Smart Data Dashboard", layout="wide")
st.title("📊 Smart Data Dashboard")
st.markdown("Upload CSV, Excel, JSON, or Parquet and get KPIs, visual insights, AI-style insights, and download PDF report.")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader(
    "Upload dataset (CSV, Excel, JSON, Parquet)", 
    type=["csv", "xls", "xlsx", "json", "parquet"]
)
if not uploaded_file:
    st.info("Please upload a dataset to begin.")
    st.stop()

# ---------------- Load File ----------------
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    elif uploaded_file.name.endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    else:
        st.error("Unsupported file type")
        st.stop()
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
st.success("File loaded successfully!")
st.dataframe(df.head())

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")
filtered_df = df.copy()
for col in df.select_dtypes(include=['object', 'category']).columns:
    selected = st.sidebar.multiselect(f"Filter {col}", options=df[col].unique(), default=df[col].unique())
    filtered_df = filtered_df[filtered_df[col].isin(selected)]

for col in df.select_dtypes(include=['int64', 'float64']).columns:
    min_val, max_val = float(df[col].min()), float(df[col].max())
    selected_range = st.sidebar.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
    filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])]

# ---------------- KPIs ----------------
st.subheader("Key Metrics")
kpi_cols = st.columns(4)

if "sales" in filtered_df.columns:
    total_sales = filtered_df["sales"].sum()
    kpi_cols[0].metric("Total Sales", f"${total_sales:,.2f}")
else: total_sales = None

if "profit" in filtered_df.columns:
    total_profit = filtered_df["profit"].sum()
    kpi_cols[1].metric("Total Profit", f"${total_profit:,.2f}")
else: total_profit = None

if "discount" in filtered_df.columns:
    avg_discount = filtered_df["discount"].mean()
    kpi_cols[2].metric("Avg Discount", f"{avg_discount:.2%}")
else: avg_discount = None

if "quantity" in filtered_df.columns:
    total_quantity = filtered_df["quantity"].sum()
    kpi_cols[3].metric("Total Quantity", f"{int(total_quantity):,}")
else: total_quantity = None

# ---------------- Visual Dashboards ----------------
st.subheader("Visual Insights")
plots = {}

for col in filtered_df.columns:
    if filtered_df[col].dtype in ['object', 'category']:
        top_vals = filtered_df[col].value_counts().reset_index()
        top_vals.columns = [col, "count"]
        fig = px.bar(top_vals, x=col, y="count", text_auto=True, title=f"{col.capitalize()} Distribution", color=col)
        st.plotly_chart(fig, use_container_width=True)
        plots[col] = fig

for col in filtered_df.select_dtypes(include=['int64', 'float64']).columns:
    fig = px.histogram(filtered_df, x=col, nbins=20, title=f"{col.capitalize()} Histogram", color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# ---------------- AI-style / Heuristic Insights ----------------
st.subheader("Automatic Insights")

def heuristic_insights(df):
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
    return insights

ai_insights = heuristic_insights(filtered_df)
st.markdown("\n".join([f"- {i}" for i in ai_insights]))

# ---------------- PDF Report ----------------
st.subheader("Download PDF Report")

def create_pdf(kpis, plots, insights_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Data Analysis Report", ln=True, align="C")
    pdf.ln(5)

    # KPIs
    pdf.set_font("Arial", size=12)
    for kpi in kpis:
        pdf.multi_cell(0, 6, kpi)

    pdf.ln(5)

    # Insights
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Insights:", ln=True)
    pdf.set_font("Arial", size=11)
    for line in insights_text:
        pdf.multi_cell(0, 6, line)

    pdf.ln(5)

    # Plots
    for name, fig in plots.items():
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.write_image(tmp_file.name)  # safe method
        pdf.image(tmp_file.name, w=170)
        tmp_file.close()
        os.unlink(tmp_file.name)

    return pdf.output(dest="S").encode("latin-1", errors="ignore")

kpis_list = []
if total_sales is not None: kpis_list.append(f"Total Sales: ${total_sales:,.2f}")
if total_profit is not None: kpis_list.append(f"Total Profit: ${total_profit:,.2f}")
if avg_discount is not None: kpis_list.append(f"Avg Discount: {avg_discount:.2%}")
if total_quantity is not None: kpis_list.append(f"Total Quantity: {int(total_quantity):,}")

if st.button("Download PDF Report"):
    pdf_bytes = create_pdf(kpis_list, plots, ai_insights)
    st.download_button("📥 Download PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")
