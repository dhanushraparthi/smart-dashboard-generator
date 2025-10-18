import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import os
from fpdf import FPDF
from pptx import Presentation
from pptx.util import Inches
from sklearn.linear_model import LinearRegression

# Optional OpenAI AI insights
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------------- Streamlit Page ----------------
st.set_page_config(page_title="Smart Dashboard Generator", layout="wide")
st.title("📊 Smart Dashboard Generator")
st.markdown("Upload CSV, Excel, or JSON to generate visual dashboards, AI insights, and export PDF/PPTX reports.")

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters & Settings")
uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv","xlsx","xls","json"])
if not uploaded_file:
    st.info("Please upload a dataset to begin.")
    st.stop()

# ---------------- Load Data ----------------
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
    st.error(f"Failed to load file: {e}")
    st.stop()

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
st.success("Dataset loaded successfully!")
st.dataframe(df.head())

# ---------------- Sidebar Filters ----------------
filter_columns = st.sidebar.multiselect("Filter Columns", df.columns)
filter_values = {}
for col in filter_columns:
    values = st.sidebar.multiselect(f"Filter {col}", df[col].unique(), default=df[col].unique())
    filter_values[col] = values

# Apply filters
for col, vals in filter_values.items():
    df = df[df[col].isin(vals)]

# ---------------- KPI Metrics ----------------
st.subheader("Key Metrics")
kpi_cols = st.columns(4)
total_sales = df["sales"].sum() if "sales" in df.columns else None
total_profit = df["profit"].sum() if "profit" in df.columns else None
avg_discount = df["discount"].mean() if "discount" in df.columns else None
total_quantity = df["quantity"].sum() if "quantity" in df.columns else None

if total_sales is not None:
    kpi_cols[0].metric("Total Sales", f"${total_sales:,.2f}")
if total_profit is not None:
    kpi_cols[1].metric("Total Profit", f"${total_profit:,.2f}")
if avg_discount is not None:
    kpi_cols[2].metric("Avg Discount", f"{avg_discount:.2%}")
if total_quantity is not None:
    kpi_cols[3].metric("Total Quantity", f"{int(total_quantity):,}")

# ---------------- Visual Dashboards ----------------
st.subheader("Visual Dashboards")
plots = {}

for col in df.select_dtypes(include=["object","category"]).columns:
    try:
        top_vals = df[col].value_counts().reset_index()
        top_vals.columns = [col, "count"]
        fig = px.bar(top_vals, x=col, y="count", color=col, text_auto=True, title=f"{col.capitalize()} Distribution")
        st.plotly_chart(fig, use_container_width=True)
        plots[col] = fig
    except Exception as e:
        st.warning(f"Could not plot {col}: {e}")

# Numeric columns plots
for col in df.select_dtypes(include=np.number).columns:
    try:
        fig = px.histogram(df, x=col, nbins=20, title=f"{col.capitalize()} Histogram")
        st.plotly_chart(fig, use_container_width=True)
        plots[col] = fig
    except Exception as e:
        st.warning(f"Could not plot {col}: {e}")

# ---------------- AI / Heuristic Insights ----------------
st.subheader("AI Insights")

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
        insights.append("Not enough data for insights.")
    return insights

# Generate AI insights
ai_text = "\n".join(heuristic_insights(df))
openai_key = os.environ.get("OPENAI_API_KEY")
if openai_key and OPENAI_AVAILABLE:
    try:
        openai.api_key = openai_key
        prompt = "Summarize the dataset in 3 bullet points:\n" + ai_text
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=200
        )
        ai_text = response.choices[0].message.content.strip()
    except Exception:
        pass

st.markdown("\n".join([f"- {i}" for i in ai_text.split("\n") if i.strip()]))

# ---------------- PDF Export ----------------
st.subheader("Download PDF Report")

def create_pdf(kpis, plots, insights_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Smart Dashboard Report", ln=True, align="C")
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
            os.remove(tmp.name)
        except Exception:
            pass
    return pdf.output(dest="S").encode("latin-1", errors="ignore")

kpis_list = []
if total_sales: kpis_list.append(f"Total Sales: ${total_sales:,.2f}")
if total_profit: kpis_list.append(f"Total Profit: ${total_profit:,.2f}")
if avg_discount: kpis_list.append(f"Avg Discount: {avg_discount:.2%}")
if total_quantity: kpis_list.append(f"Total Quantity: {int(total_quantity):,}")

if st.button("Download PDF"):
    pdf_bytes = create_pdf(kpis_list, plots, ai_text)
    st.download_button("Download PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")

# ---------------- PPTX Export ----------------
st.subheader("Download PPTX Report")

def create_pptx(kpis, insights, plots):
    prs = Presentation()
    slide_layout = prs.slide_layouts[5]

    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Smart Dashboard Report"
    left, top = Inches(0.5), Inches(1.5)
    for kpi in kpis:
        slide.shapes.add_textbox(left, top, Inches(8), Inches(0.3)).text = kpi
        top += Inches(0.3)

    top += Inches(0.2)
    for line in insights.split("\n"):
        slide.shapes.add_textbox(left, top, Inches(8), Inches(0.3)).text = line
        top += Inches(0.3)

    for name, fig in plots.items():
        try:
            img_bytes = fig.to_image(format="png", engine="kaleido")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(img_bytes)
            tmp.close()
            slide.shapes.add_picture(tmp.name, Inches(0.5), top, width=Inches(8))
            os.remove(tmp.name)
            top += Inches(3)
        except Exception:
            pass
    output = tempfile.NamedTemporaryFile(delete=False, suffix=".pptx")
    prs.save(output.name)
    with open(output.name, "rb") as f:
        ppt_data = f.read()
    os.remove(output.name)
    return ppt_data

if st.button("Download PPTX"):
    ppt_data = create_pptx(kpis_list, ai_text, plots)
    st.download_button("Download PPTX", data=ppt_data, file_name="report.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
