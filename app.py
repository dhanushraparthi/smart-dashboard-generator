import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
from pptx import Presentation
from pptx.util import Inches
import tempfile, os, textwrap

st.set_page_config(page_title="Smart Data Dashboard", layout="wide")
st.title("📊 Smart Data Dashboard")

# ---------------- Sidebar ----------------
st.sidebar.header("Upload & Filters")
uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv","xlsx","xls","json"])
if uploaded_file is None:
    st.info("Upload a CSV/Excel/JSON file to continue")
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

df = load_file(uploaded_file)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ---------------- Sidebar Filters ----------------
filter_cols = [c for c in df.select_dtypes(include=['object','category']).columns]
filters = {}
for col in filter_cols:
    options = st.sidebar.multiselect(f"Filter {col}", df[col].unique(), default=df[col].unique())
    filters[col] = options
    df = df[df[col].isin(options)]

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- KPIs ----------------
st.subheader("Key Metrics")
kpi_cols = st.columns(4)
total_sales = df["sales"].sum() if "sales" in df.columns else None
total_profit = df["profit"].sum() if "profit" in df.columns else None
avg_discount = df["discount"].mean() if "discount" in df.columns else None
total_quantity = df["quantity"].sum() if "quantity" in df.columns else None

if total_sales is not None: kpi_cols[0].metric("Total Sales", f"${total_sales:,.2f}")
if total_profit is not None: kpi_cols[1].metric("Total Profit", f"${total_profit:,.2f}")
if avg_discount is not None: kpi_cols[2].metric("Avg Discount", f"{avg_discount:.2%}")
if total_quantity is not None: kpi_cols[3].metric("Total Quantity", f"{int(total_quantity):,}")

kpis_list = []
if total_sales: kpis_list.append(f"Total Sales: ${total_sales:,.2f}")
if total_profit: kpis_list.append(f"Total Profit: ${total_profit:,.2f}")
if avg_discount: kpis_list.append(f"Avg Discount: {avg_discount:.2%}")
if total_quantity: kpis_list.append(f"Total Quantity: {int(total_quantity):,}")

# ---------------- Visual Dashboards ----------------
st.subheader("Visual Insights")
plots = {}

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()

# Bar charts for categorical columns
for col in categorical_cols:
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(counts, x=col, y="count", text_auto=True, color=col, title=f"{col.capitalize()} Distribution")
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# Line charts for numeric columns
for col in numeric_cols:
    fig = px.line(df, y=col, title=f"{col.capitalize()} Trend")
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# Scatter plot for first two numeric columns
if len(numeric_cols) >= 2:
    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=categorical_cols[0] if categorical_cols else None,
                     title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
    st.plotly_chart(fig, use_container_width=True)
    plots[f"{numeric_cols[0]}_vs_{numeric_cols[1]}"] = fig

# ---------------- AI / Heuristic Insights ----------------
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
    margin = df["profit"].sum() / df["sales"].sum() if df["sales"].sum()!=0 else 0
    insights.append(f"Overall profit margin: {margin:.2%}")
if not insights: insights.append("Not enough data for detailed insights.")

ai_insights = insights
for i in ai_insights: st.markdown(f"- {i}")

# ---------------- PDF Report ----------------
st.subheader("Download PDF Report")

def create_pdf(kpis, plots, insights_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Smart Data Dashboard Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    max_width = 190

    for kpi in kpis:
        for line in textwrap.wrap(kpi, width=90):
            pdf.multi_cell(max_width, 6, line)
        pdf.ln(1)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Insights:", ln=True)
    pdf.set_font("Arial", size=11)
    for insight in insights_text:
        for line in textwrap.wrap(insight, width=90):
            pdf.multi_cell(max_width, 6, line)
        pdf.ln(1)

    pdf.ln(5)
    for name, fig in plots.items():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            fig.write_image(tmp_file.name)  # safe
            pdf.image(tmp_file.name, w=170)
        os.remove(tmp_file.name)
    return pdf.output(dest="S").encode("utf-8")

if st.button("Download PDF"):
    pdf_bytes = create_pdf(kpis_list, plots, ai_insights)
    st.download_button("📥 Download PDF", data=pdf_bytes, file_name="dashboard_report.pdf", mime="application/pdf")

# ---------------- PPTX Export ----------------
st.subheader("Download PPTX Report")
def export_pptx(kpis, insights, plots):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    left, top = Inches(0.5), Inches(0.5)
    for kpi in kpis:
        txBox = slide.shapes.add_textbox(left, top, Inches(9), Inches(0.3))
        tf = txBox.text_frame
        tf.text = kpi
        top += Inches(0.4)

    for insight in insights:
        txBox = slide.shapes.add_textbox(left, top, Inches(9), Inches(0.3))
        tf = txBox.text_frame
        tf.text = insight
        top += Inches(0.4)

    for name, fig in plots.items():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            fig.write_image(tmp_file.name)
            slide.shapes.add_picture(tmp_file.name, Inches(0.5), top, width=Inches(9))
        os.remove(tmp_file.name)
        top += Inches(3)

    buf = tempfile.NamedTemporaryFile(delete=False, suffix=".pptx")
    prs.save(buf.name)
    buf.seek(0)
    ppt_data = buf.read()
    buf.close()
    os.remove(buf.name)
    return ppt_data

if st.button("Download PPTX"):
    ppt_bytes = export_pptx(kpis_list, ai_insights, plots)
    st.download_button("📥 Download PPTX", data=ppt_bytes, file_name="dashboard_report.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
