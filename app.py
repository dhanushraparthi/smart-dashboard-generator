import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
from pptx import Presentation
from pptx.util import Inches
import tempfile
import os

# Optional: OpenAI for AI insights
try:
    import openai
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="Smart Data Dashboard", layout="wide")
st.title("📊 Smart Data Dashboard")
st.sidebar.header("Upload & Filters")

# ----------------- File Upload -----------------
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel / JSON", type=["csv","xlsx","xls","json"])
if not uploaded_file:
    st.info("Please upload a dataset to start.")
    st.stop()

def load_file(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith((".xlsx",".xls")):
        return pd.read_excel(file)
    elif name.endswith(".json"):
        return pd.read_json(file)
    else:
        raise ValueError("Unsupported file type")

try:
    df = load_file(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
st.sidebar.write("Columns detected:", df.columns.tolist())

# ----------------- Sidebar Filters -----------------
filter_cols = st.sidebar.multiselect("Select columns to visualize", df.columns.tolist(), default=df.columns.tolist())
st.sidebar.markdown("---")

# ----------------- KPIs -----------------
st.subheader("Key Metrics")
kpi_cols = st.columns(4)

def safe_sum(col):
    return df[col].sum() if col in df.columns else None

total_sales = safe_sum("sales")
total_profit = safe_sum("profit")
total_quantity = safe_sum("quantity")
avg_discount = df["discount"].mean() if "discount" in df.columns else None

if total_sales is not None:
    kpi_cols[0].metric("Total Sales", f"${total_sales:,.2f}")
if total_profit is not None:
    kpi_cols[1].metric("Total Profit", f"${total_profit:,.2f}")
if avg_discount is not None:
    kpi_cols[2].metric("Avg Discount", f"{avg_discount:.2%}")
if total_quantity is not None:
    kpi_cols[3].metric("Total Quantity", f"{int(total_quantity):,}")

# ----------------- Visualizations -----------------
st.subheader("Visual Insights")
plots = {}

for col in filter_cols:
    if df[col].dtype == 'object' or str(df[col].dtype).startswith('category'):
        top_vals = df[col].value_counts().reset_index()
        top_vals.columns = ['category', 'count']
        fig = px.bar(top_vals, x='category', y='count', text_auto=True,
                     title=f"{col.capitalize()} Distribution", color='category')
    else:
        fig = px.histogram(df, x=col, nbins=20, title=f"{col.capitalize()} Histogram", color_discrete_sequence=['#00cc96'])
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# ----------------- AI Insights -----------------
st.subheader("AI / Heuristic Insights")
def generate_insights(df):
    insights = []
    if "sales" in df.columns and "region" in df.columns:
        top_region = df.groupby("region")["sales"].sum().idxmax()
        insights.append(f"Highest sales region: {top_region}")
    if "profit" in df.columns and "category" in df.columns:
        top_cat = df.groupby("category")["profit"].sum().idxmax()
        insights.append(f"Most profitable category: {top_cat}")
    if "discount" in df.columns and "profit" in df.columns:
        corr = df["discount"].corr(df["profit"])
        insights.append(f"Discount vs Profit correlation: {corr:.2f}")
    return insights if insights else ["Not enough data for detailed insights."]

ai_insights = generate_insights(df)

# Optional OpenAI enhancement
openai_key = os.environ.get("OPENAI_API_KEY")
if openai_key and OPENAI_AVAILABLE:
    try:
        openai.api_key = openai_key
        prompt = "Summarize these dataset insights in 3 short bullet points:\n" + "\n".join(ai_insights)
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=200,
            temperature=0.2
        )
        ai_insights = [line.strip() for line in response.choices[0].message.content.split("\n") if line.strip()]
    except:
        pass

st.markdown("\n".join([f"- {i}" for i in ai_insights]))

# ----------------- PDF Report -----------------
st.subheader("Download PDF Report")
def create_pdf(kpis, plots, insights_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Smart Data Dashboard Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    for kpi in kpis:
        pdf.multi_cell(0, 6, kpi)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Insights:", ln=True)
    pdf.set_font("Arial", size=11)
    for line in insights_text:
        pdf.multi_cell(0, 6, line)
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

kpis_list = []
if total_sales is not None: kpis_list.append(f"Total Sales: ${total_sales:,.2f}")
if total_profit is not None: kpis_list.append(f"Total Profit: ${total_profit:,.2f}")
if avg_discount is not None: kpis_list.append(f"Avg Discount: {avg_discount:.2%}")
if total_quantity is not None: kpis_list.append(f"Total Quantity: {int(total_quantity):,}")

if st.button("Download PDF"):
    pdf_bytes = create_pdf(kpis_list, plots, ai_insights)
    st.download_button("Download PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")

# ----------------- PPTX Export -----------------
st.subheader("Download PPTX Report")
def export_pptx(kpis, insights, plots):
    prs = Presentation()
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    left, top = Inches(0.5), Inches(0.5)
    for i, kpi in enumerate(kpis):
        slide.shapes.add_textbox(left, top + i*0.3, Inches(9), Inches(0.3)).text = kpi
    slide.shapes.add_textbox(left, top + len(kpis)*0.3 + 0.1, Inches(9), Inches(1)).text = "\n".join(insights)
    for name, fig in plots.items():
        img_bytes = fig.to_image(format="png", engine="kaleido")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(img_bytes)
        tmp.flush()
        tmp.close()
        slide.shapes.add_picture(tmp.name, Inches(0.5), Inches(2), width=Inches(9))
        os.remove(tmp.name)
    buf = tempfile.NamedTemporaryFile(delete=False, suffix=".pptx")
    prs.save(buf.name)
    buf.seek(0)
    ppt_data = buf.read()
    buf.close()
    os.remove(buf.name)
    return ppt_data

if st.button("Download PPTX"):
    ppt_data = export_pptx(kpis_list, ai_insights, plots)
    st.download_button("Download PPTX", data=ppt_data, file_name="report.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
