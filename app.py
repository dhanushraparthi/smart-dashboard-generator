import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
from pptx import Presentation
from pptx.util import Inches
import tempfile
import os

# ----------------- Page Setup -----------------
st.set_page_config(page_title="Smart Data Dashboard", layout="wide")
st.title("📊 Smart Data Dashboard")
st.markdown("Upload your dataset (CSV / Excel / JSON) to generate dashboards, AI insights, PDF and PPTX reports.")

# ----------------- Sidebar -----------------
st.sidebar.header("Filters & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv","xlsx","xls","json"])
if not uploaded_file:
    st.info("Please upload a dataset to begin.")
    st.stop()

def load_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    elif name.endswith(".json"):
        return pd.read_json(uploaded)
    else:
        raise ValueError("Unsupported file type")

try:
    df = load_file(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
st.success("Dataset loaded successfully!")
st.dataframe(df.head())

# ----------------- Sidebar Filters -----------------
numeric_cols = [c for c in df.columns if df[c].dtype in ['int64','float64']]
for col in numeric_cols:
    min_val, max_val = float(df[col].min()), float(df[col].max())
    val = st.sidebar.slider(f"{col} range", min_val, max_val, (min_val, max_val))
    df = df[(df[col] >= val[0]) & (df[col] <= val[1])]

all_columns = df.columns.tolist()
selected_visuals = st.sidebar.multiselect("Select columns to visualize", all_columns, default=all_columns[:3])
show_ai = st.sidebar.checkbox("Include AI Insights", value=True)
include_charts = st.sidebar.checkbox("Include charts in reports", value=True)

# ----------------- KPIs -----------------
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

# ----------------- Visual Dashboards -----------------
st.subheader("Visual Insights")
plots = {}
for col in selected_visuals:
    if df[col].dtype in ['object','category']:
        fig = px.bar(df[col].value_counts().reset_index(), x='index', y=col, text_auto=True,
                     title=f"{col.capitalize()} Distribution", color='index')
    else:
        fig = px.histogram(df, x=col, nbins=20, title=f"{col.capitalize()} Histogram", color_discrete_sequence=['#00cc96'])
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# ----------------- AI-style Insights -----------------
st.subheader("AI Insights")
def heuristic_insights(df):
    insights = []
    if "region" in df.columns and "sales" in df.columns:
        insights.append(f"Highest sales region: {df.groupby('region')['sales'].sum().idxmax()}")
    if "category" in df.columns and "profit" in df.columns:
        insights.append(f"Most profitable category: {df.groupby('category')['profit'].sum().idxmax()}")
    if "discount" in df.columns and "profit" in df.columns:
        insights.append(f"Discount vs Profit correlation: {df['discount'].corr(df['profit']):.2f}")
    if "sales" in df.columns and "profit" in df.columns:
        margin = df['profit'].sum()/df['sales'].sum() if df['sales'].sum()!=0 else 0
        insights.append(f"Overall profit margin: {margin:.2%}")
    return insights

ai_text = "\n".join(heuristic_insights(df)) if show_ai else "AI Insights Disabled."
st.markdown("\n".join([f"- {i}" for i in ai_text.split("\n") if i.strip()]))

# ----------------- PDF Export -----------------
st.subheader("Download PDF Report")
def create_pdf(kpis, plots, insights_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Data Analysis Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    for kpi in kpis:
        pdf.multi_cell(0, 6, kpi)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Insights:", ln=True)
    pdf.set_font("Arial", size=11)
    for line in insights_text.split("\n"):
        pdf.multi_cell(0, 6, line)
    pdf.ln(5)
    for name, fig in plots.items():
        try:
            img_bytes = fig.to_image(format="png", engine="kaleido")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(img_bytes)
            tmp.close()
            pdf.image(tmp.name, w=170)
            os.unlink(tmp.name)
        except Exception:
            continue
    return pdf.output(dest="S").encode("latin-1", errors="ignore")

kpis_list = []
if total_sales is not None: kpis_list.append(f"Total Sales: ${total_sales:,.2f}")
if total_profit is not None: kpis_list.append(f"Total Profit: ${total_profit:,.2f}")
if avg_discount is not None: kpis_list.append(f"Avg Discount: {avg_discount:.2%}")
if total_quantity is not None: kpis_list.append(f"Total Quantity: {int(total_quantity):,}")

if st.button("Download PDF"):
    pdf_bytes = create_pdf(kpis_list, plots, ai_text)
    st.download_button("Download PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")

# ----------------- PPTX Export -----------------
st.subheader("Download PPTX Report")
def export_pptx(kpis, insights_text, plots):
    prs = Presentation()
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    left = top = Inches(0.5)
    width = Inches(9)
    slide.shapes.add_textbox(left, top, width, Inches(1)).text = "\n".join(kpis)
    slide.shapes.add_textbox(left, top+Inches(1), width, Inches(1)).text = insights_text
    for name, fig in plots.items():
        try:
            img_bytes = fig.to_image(format="png", engine="kaleido")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(img_bytes)
            tmp.close()
            slide.shapes.add_picture(tmp.name, left, top+Inches(2), width=Inches(7))
            os.unlink(tmp.name)
        except Exception:
            continue
    buf = tempfile.NamedTemporaryFile(delete=False, suffix=".pptx")
    prs.save(buf.name)
    buf.seek(0)
    data = buf.read()
    buf.close()
    return data

if st.button("Download PPTX"):
    ppt_data = export_pptx(kpis_list, ai_text, plots)
    st.download_button("Download PPTX", data=ppt_data, file_name="report.pptx",
                       mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
