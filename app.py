# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from pptx import Presentation
from pptx.util import Inches
import tempfile
import os
import textwrap

# Optional OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Smart Data Dashboard", layout="wide", page_icon="📊")
st.title("📊 Smart Data Dashboard")

# ---------------- Sidebar ----------------
st.sidebar.header("Filters & Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel / JSON", type=["csv","xlsx","xls","json"])
if not uploaded_file:
    st.info("Please upload a dataset to start.")
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

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
st.success("File loaded successfully!")
st.dataframe(df.head())

# ---------------- Sidebar Filters ----------------
filter_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
filters = {}
for col in filter_cols:
    options = df[col].unique().tolist()
    selected = st.sidebar.multiselect(f"Filter {col}", options, default=options)
    filters[col] = selected
    df = df[df[col].isin(selected)]

# ---------------- KPIs ----------------
st.subheader("Key Metrics")
kpi_cols = st.columns(4)
kpi_list = []

if "sales" in df.columns:
    total_sales = df["sales"].sum()
    kpi_cols[0].metric("Total Sales", f"${total_sales:,.2f}")
    kpi_list.append(f"Total Sales: ${total_sales:,.2f}")
if "profit" in df.columns:
    total_profit = df["profit"].sum()
    kpi_cols[1].metric("Total Profit", f"${total_profit:,.2f}")
    kpi_list.append(f"Total Profit: ${total_profit:,.2f}")
if "quantity" in df.columns:
    total_qty = df["quantity"].sum()
    kpi_cols[2].metric("Total Quantity", f"{int(total_qty):,}")
    kpi_list.append(f"Total Quantity: {int(total_qty):,}")
if "discount" in df.columns:
    avg_discount = df["discount"].mean()
    kpi_cols[3].metric("Avg Discount", f"{avg_discount:.2%}")
    kpi_list.append(f"Avg Discount: {avg_discount:.2%}")

# ---------------- Visual Dashboards ----------------
st.subheader("Visual Insights")
plots = {}

# Bar charts for categorical columns
for col in filter_cols:
    top_vals = df[col].value_counts().reset_index()
    top_vals.columns = ['category', 'count']
    fig = px.bar(top_vals, x='category', y='count', text_auto=True, title=f"{col.capitalize()} Distribution", color='category')
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# Line chart for numerical trends
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
if len(num_cols) >= 2:
    fig_line = px.line(df[num_cols], title="Numerical Trends", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)
    plots["line_trends"] = fig_line

# Pie charts for categories
if filter_cols:
    fig_pie = px.pie(df, names=filter_cols[0], title=f"{filter_cols[0].capitalize()} Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)
    plots["pie"] = fig_pie

# ---------------- AI-style Insights ----------------
st.subheader("Automatic Insights")
def heuristic_insights(df):
    insights = []
    if "sales" in df.columns and "profit" in df.columns:
        top_profit_region = df.groupby("region")["profit"].sum().idxmax() if "region" in df.columns else "N/A"
        insights.append(f"Highest Profit Region: {top_profit_region}")
        margin = df["profit"].sum() / df["sales"].sum() if df["sales"].sum() != 0 else 0
        insights.append(f"Overall Profit Margin: {margin:.2%}")
    if "discount" in df.columns and "profit" in df.columns:
        corr = df["discount"].corr(df["profit"])
        insights.append(f"Discount vs Profit Correlation: {corr:.2f}")
    if not insights:
        insights.append("Not enough data for detailed insights.")
    return insights

ai_insights = heuristic_insights(df)

# Optional OpenAI summary
openai_key = os.environ.get("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None)
if openai_key and OPENAI_AVAILABLE:
    try:
        openai.api_key = openai_key
        prompt = "Summarize the following dataset insights in 3 short bullet points:\n" + "\n".join(ai_insights)
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=200,
            temperature=0.2
        )
        ai_text = response.choices[0].message.content.strip().split("\n")
    except Exception:
        ai_text = ai_insights
else:
    ai_text = ai_insights

st.markdown("\n".join([f"- {i}" for i in ai_text]))

# ---------------- PDF Download ----------------
st.subheader("📄 Download PDF Report")
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
        img_bytes = fig.to_image(format="png", engine="kaleido")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(img_bytes)
        tmp.flush()
        tmp.close()
        pdf.image(tmp.name, w=170)
        os.remove(tmp.name)

    return pdf.output(dest="S").encode("utf-8")

if st.button("Download PDF Report"):
    pdf_bytes = create_pdf(kpi_list, plots, ai_text)
    st.download_button("📥 Download PDF", data=pdf_bytes, file_name="dashboard_report.pdf", mime="application/pdf")

# ---------------- PPTX Download ----------------
st.subheader("📊 Download PPTX Presentation")
def export_pptx(kpis, insights, plots):
    prs = Presentation()
    slide_layout = prs.slide_layouts[5]  # blank slide

    # Add KPIs slide
    slide = prs.slides.add_slide(slide_layout)
    top = Inches(0.5)
    left = Inches(0.5)
    for kpi in kpis:
        slide.shapes.add_textbox(left, top, Inches(8), Inches(0.5)).text = kpi
        top += Inches(0.5)
    # Add Insights
    slide.shapes.add_textbox(left, top, Inches(8), Inches(1)).text = "\n".join(insights)

    # Add slides for charts
    for name, fig in plots.items():
        slide = prs.slides.add_slide(slide_layout)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(fig.to_image(format="png", engine="kaleido"))
        tmp.flush()
        tmp.close()
        slide.shapes.add_picture(tmp.name, Inches(0.5), Inches(1), width=Inches(8))
        os.remove(tmp.name)

    pptx_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pptx")
    prs.save(pptx_file.name)
    pptx_file.seek(0)
    data = pptx_file.read()
    pptx_file.close()
    os.remove(pptx_file.name)
    return data

if st.button("Download PPTX Presentation"):
    ppt_data = export_pptx(kpi_list, ai_text, plots)
    st.download_button("📥 Download PPTX", data=ppt_data, file_name="dashboard_presentation.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
