import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
from pptx import Presentation
from pptx.util import Inches
import tempfile, os
import textwrap

# ---------------- Page setup ----------------
st.set_page_config(page_title="Smart Data Dashboard", layout="wide")
st.title("📊 Smart Data Dashboard")
st.markdown("Upload CSV, Excel, or JSON to generate dashboards, AI insights, PDF, and PPTX.")

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv","xlsx","xls","json"])
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

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
st.success("File loaded")
st.dataframe(df.head())

# ---------------- Sidebar Dynamic Filters ----------------
filter_columns = st.sidebar.multiselect("Select columns to filter", df.columns)
filtered_df = df.copy()
for col in filter_columns:
    options = st.sidebar.multiselect(f"Filter {col}", df[col].unique(), default=df[col].unique())
    filtered_df = filtered_df[filtered_df[col].isin(options)]

# ---------------- KPIs ----------------
st.subheader("Key Metrics")
kpi_cols = st.columns(4)
total_sales = total_profit = avg_discount = total_quantity = None

if "sales" in filtered_df.columns:
    total_sales = filtered_df["sales"].sum()
    kpi_cols[0].metric("Total Sales", f"${total_sales:,.2f}")
if "profit" in filtered_df.columns:
    total_profit = filtered_df["profit"].sum()
    kpi_cols[1].metric("Total Profit", f"${total_profit:,.2f}")
if "discount" in filtered_df.columns:
    avg_discount = filtered_df["discount"].mean()
    kpi_cols[2].metric("Avg Discount", f"{avg_discount:.2%}")
if "quantity" in filtered_df.columns:
    total_quantity = filtered_df["quantity"].sum()
    kpi_cols[3].metric("Total Quantity", f"{int(total_quantity):,}")

kpis_list = []
if total_sales: kpis_list.append(f"Total Sales: ${total_sales:,.2f}")
if total_profit: kpis_list.append(f"Total Profit: ${total_profit:,.2f}")
if avg_discount: kpis_list.append(f"Avg Discount: {avg_discount:.2%}")
if total_quantity: kpis_list.append(f"Total Quantity: {int(total_quantity):,}")

# ---------------- Visual Dashboards ----------------
st.subheader("Visual Dashboards")
plots = {}
numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = filtered_df.select_dtypes(include="object").columns.tolist()

# Plot numeric distributions
for col in numeric_cols:
    fig = px.histogram(filtered_df, x=col, nbins=20, title=f"{col.capitalize()} Distribution")
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# Plot categorical distributions
for col in categorical_cols:
    counts = filtered_df[col].value_counts().reset_index()
    counts.columns = [col, 'count']
    fig = px.bar(counts, x=col, y='count', color=col, title=f"{col.capitalize()} Distribution")
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# ---------------- AI & Heuristic Insights ----------------
st.subheader("AI Insights / Summary")
insights = []

if "region" in filtered_df.columns and "sales" in filtered_df.columns:
    top_region = filtered_df.groupby("region")["sales"].sum().idxmax()
    insights.append(f"Highest sales region: {top_region}")
if "category" in filtered_df.columns and "profit" in filtered_df.columns:
    top_category = filtered_df.groupby("category")["profit"].sum().idxmax()
    insights.append(f"Most profitable category: {top_category}")
if "discount" in filtered_df.columns and "profit" in filtered_df.columns:
    corr = filtered_df["discount"].corr(filtered_df["profit"])
    insights.append(f"Discount vs Profit correlation: {corr:.2f}")
if "sales" in filtered_df.columns and "profit" in filtered_df.columns:
    margin = filtered_df["profit"].sum() / filtered_df["sales"].sum() if filtered_df["sales"].sum() != 0 else 0
    insights.append(f"Overall profit margin: {margin:.2%}")

if not insights:
    insights.append("Not enough data for insights.")

st.markdown("\n".join([f"- {i}" for i in insights]))
ai_text = "\n".join(insights)

# ---------------- PDF Report ----------------
st.subheader("Download PDF Report")
def create_pdf(kpis, plots, insights):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Smart Dashboard Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    for kpi in kpis:
        for line in textwrap.wrap(kpi, width=90):
            pdf.multi_cell(0, 6, line)
        pdf.ln(1)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Insights:", ln=True)
    pdf.set_font("Arial", size=11)
    for line in insights.split("\n"):
        for wline in textwrap.wrap(line, width=90):
            pdf.multi_cell(0, 6, wline)
        pdf.ln(1)

    for name, fig in plots.items():
        img_bytes = fig.to_image(format="png", engine="kaleido")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(img_bytes)
        tmp.close()
        pdf.image(tmp.name, w=170)
        os.remove(tmp.name)

    return pdf.output(dest="S").encode("latin-1", errors="ignore")

if st.button("Download PDF"):
    pdf_bytes = create_pdf(kpis_list, plots, ai_text)
    st.download_button("Download PDF", data=pdf_bytes, file_name="dashboard_report.pdf", mime="application/pdf")

# ---------------- PPTX Export ----------------
st.subheader("Download PPTX Report")
def create_pptx(kpis, plots, insights):
    prs = Presentation()
    slide_layout = prs.slide_layouts[5]

    # Cover slide
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Smart Dashboard Report"
    slide.placeholders[1].text = ""

    # KPI slide
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "KPIs"
    left = Inches(0.5)
    top = Inches(1.2)
    width = Inches(9)
    height = Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    tf = textbox.text_frame
    for kpi in kpis:
        for line in textwrap.wrap(kpi, width=90):
            p = tf.add_paragraph()
            p.text = line

    # Insights slide
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Insights"
    textbox = slide.shapes.add_textbox(left, top, width, height)
    tf = textbox.text_frame
    for line in insights.split("\n"):
        for wline in textwrap.wrap(line, width=90):
            p = tf.add_paragraph()
            p.text = wline

    # Plots slides
    for name, fig in plots.items():
        slide = prs.slides.add_slide(slide_layout)
        slide.shapes.title.text = name.capitalize()
        img_bytes = fig.to_image(format="png", engine="kaleido")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(img_bytes)
        tmp.close()
        slide.shapes.add_picture(tmp.name, Inches(1), Inches(1.5), width=Inches(8))
        os.remove(tmp.name)

    buf = tempfile.NamedTemporaryFile(delete=False, suffix=".pptx")
    prs.save(buf.name)
    buf.seek(0)
    data = buf.read()
    buf.close()
    os.remove(buf.name)
    return data

if st.button("Download PPTX"):
    pptx_data = create_pptx(kpis_list, plots, ai_text)
    st.download_button("Download PPTX", data=pptx_data, file_name="dashboard_report.pptx",
                       mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
