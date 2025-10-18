# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
from pptx import Presentation
from pptx.util import Inches
import os
import tempfile

# Optional OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Smart Data Dashboard", layout="wide")
st.title("📊 Smart Data Dashboard")
st.markdown("Upload CSV, Excel, or JSON to explore KPIs, dashboards, AI insights, and download reports.")

# ---------------- Sidebar ----------------
st.sidebar.header("Upload & Filters")
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv","xlsx","xls","json"])

if uploaded_file is None:
    st.info("Please upload a dataset to begin.")
    st.stop()

# ---------------- Load File ----------------
def load_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith((".xlsx",".xls")):
        return pd.read_excel(uploaded)
    if name.endswith(".json"):
        return pd.read_json(uploaded)
    raise ValueError("Unsupported file type")

try:
    df = load_file(uploaded_file)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
st.success("Dataset loaded successfully")
st.dataframe(df.head())

# ---------------- Sidebar Filters ----------------
filter_cols = [col for col in df.columns if df[col].dtype in ['int64','float64']]
selected_filters = {}
for col in filter_cols:
    min_val, max_val = float(df[col].min()), float(df[col].max())
    selected_filters[col] = st.sidebar.slider(f"{col.capitalize()} range", min_val, max_val, (min_val,max_val))
    df = df[(df[col] >= selected_filters[col][0]) & (df[col] <= selected_filters[col][1])]

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
if "discount" in df.columns:
    avg_discount = df["discount"].mean()
    kpi_cols[2].metric("Avg Discount", f"{avg_discount:.2%}")
    kpi_list.append(f"Average Discount: {avg_discount:.2%}")
if "quantity" in df.columns:
    total_quantity = df["quantity"].sum()
    kpi_cols[3].metric("Total Quantity", f"{int(total_quantity):,}")
    kpi_list.append(f"Total Quantity: {int(total_quantity):,}")

# ---------------- Visual Dashboards ----------------
st.subheader("Visual Insights")
plots = {}
for col in df.select_dtypes(include=['object','category']).columns:
    top_vals = df[col].value_counts().reset_index()
    top_vals.columns = [col, 'count']
    fig = px.bar(top_vals, x=col, y='count', text_auto=True, title=f"{col.capitalize()} Distribution",
                 color=col, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# Numeric plots
for col in df.select_dtypes(include=['int64','float64']).columns:
    fig = px.histogram(df, x=col, nbins=20, title=f"{col.capitalize()} Histogram", template="plotly_dark", color_discrete_sequence=['#00FFAA'])
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# ---------------- AI-style Insights ----------------
st.subheader("AI & Heuristic Insights")
def generate_insights(df):
    insights = []
    if "sales" in df.columns and "profit" in df.columns:
        margin = df["profit"].sum()/df["sales"].sum() if df["sales"].sum()!=0 else 0
        insights.append(f"Overall profit margin: {margin:.2%}")
    for col in df.select_dtypes(include=['object','category']).columns:
        top = df.groupby(col).sum().sort_values("sales" if "sales" in df.columns else df.columns[0], ascending=False).head(1).index[0]
        insights.append(f"Top {col}: {top}")
    return insights

heuristic_insights = generate_insights(df)
ai_text = "\n".join(heuristic_insights)

# Optional OpenAI summary
openai_key = os.environ.get("OPENAI_API_KEY")
if openai_key and OPENAI_AVAILABLE:
    try:
        openai.api_key = openai_key
        prompt = "Summarize the following dataset insights in 3 concise bullet points:\n" + ai_text
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=200,
            temperature=0.2
        )
        ai_text = response.choices[0].message.content.strip()
    except Exception:
        pass

st.markdown("\n".join([f"- {i}" for i in ai_text.split("\n") if i.strip()]))

# ---------------- PDF Export ----------------
st.subheader("Download PDF Report")
def create_pdf(kpis, plots, insights_text):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Smart Dashboard Report", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Arial", '', 12)
    for kpi in kpis:
        pdf.multi_cell(0, 8, kpi)
        pdf.ln(1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0,8,"Insights:", ln=True)
    pdf.set_font("Arial", '', 12)
    for line in insights_text.split("\n"):
        pdf.multi_cell(0,8,line)
    pdf.ln(5)
    for name, fig in plots.items():
        img_bytes = fig.to_image(format="png", engine="kaleido")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(img_bytes)
        tmp.flush()
        tmp.close()
        pdf.image(tmp.name, w=170)
    return pdf.output(dest="S").encode("latin-1", errors="ignore")

if st.button("Download PDF"):
    pdf_bytes = create_pdf(kpi_list, plots, ai_text)
    st.download_button("📥 Download PDF", data=pdf_bytes, file_name="dashboard_report.pdf", mime="application/pdf")

# ---------------- PPTX Export ----------------
st.subheader("Download PPTX Report")
def export_pptx(kpis, insights_text, plots):
    prs = Presentation()
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    left, top = Inches(0.5), Inches(0.5)
    width = Inches(9)
    txBox = slide.shapes.add_textbox(left, top, width, Inches(1))
    tf = txBox.text_frame
    tf.text = "KPIs:\n" + "\n".join(kpis) + "\n\nInsights:\n" + insights_text
    for name, fig in plots.items():
        img_bytes = fig.to_image(format="png", engine="kaleido")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(img_bytes)
        tmp.flush()
        tmp.close()
        slide = prs.slides.add_slide(slide_layout)
        slide.shapes.add_picture(tmp.name, Inches(1), Inches(1.5), width=Inches(7))
    tmp_ppt = tempfile.NamedTemporaryFile(delete=False, suffix=".pptx")
    prs.save(tmp_ppt.name)
    tmp_ppt.seek(0)
    return tmp_ppt

if st.button("Download PPTX"):
    ppt_file = export_pptx(kpi_list, ai_text, plots)
    st.download_button("📥 Download PPTX", data=open(ppt_file.name,"rb").read(),
                       file_name="dashboard_report.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
