import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
import tempfile
import os

# Optional OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Smart Dashboard Generator", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Smart Dashboard Generator")
st.markdown("Upload CSV / Excel / JSON files and get interactive dashboards, AI-style insights, and PDF reports.")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload your dataset", type=["csv","xlsx","xls","json"])
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
st.sidebar.header("Filters")
filtered_df = df.copy()
for col in df.select_dtypes(include=["object", "category"]).columns:
    unique_vals = df[col].unique()
    selected = st.sidebar.multiselect(f"Filter {col}", options=unique_vals, default=list(unique_vals))
    filtered_df = filtered_df[filtered_df[col].isin(selected)]

# ---------------- KPIs ----------------
st.subheader("Key Metrics")
kpi_cols = st.columns(4)
total_sales = filtered_df["sales"].sum() if "sales" in filtered_df.columns else None
total_profit = filtered_df["profit"].sum() if "profit" in filtered_df.columns else None
avg_discount = filtered_df["discount"].mean() if "discount" in filtered_df.columns else None
total_quantity = filtered_df["quantity"].sum() if "quantity" in filtered_df.columns else None

if total_sales is not None:
    kpi_cols[0].metric("Total Sales", f"${total_sales:,.2f}")
if total_profit is not None:
    kpi_cols[1].metric("Total Profit", f"${total_profit:,.2f}")
if avg_discount is not None:
    kpi_cols[2].metric("Avg Discount", f"{avg_discount:.2%}")
if total_quantity is not None:
    kpi_cols[3].metric("Total Quantity", f"{int(total_quantity):,}")

# Store for PDF export
st.session_state['total_sales'] = total_sales
st.session_state['total_profit'] = total_profit
st.session_state['avg_discount'] = avg_discount
st.session_state['total_quantity'] = total_quantity

# ---------------- Plots ----------------
st.subheader("Visual Dashboards")
plots = {}

# Dark theme template
dark_template = "plotly_dark"

for col in filtered_df.select_dtypes(include=["object", "category"]).columns:
    if col != "index":
        top_vals = filtered_df[col].value_counts().reset_index()
        top_vals.columns = ["category", "count"]
        fig = px.bar(
            top_vals, x="category", y="count",
            text_auto=True, color="category",
            title=f"Distribution of {col}",
            template=dark_template
        )
        st.plotly_chart(fig, use_container_width=True)
        plots[col] = fig

for col in filtered_df.select_dtypes(include=[np.number]).columns:
    fig = px.histogram(filtered_df, x=col, nbins=20, color_discrete_sequence=["#00ffff"], template=dark_template)
    fig.update_layout(title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

st.session_state['plots'] = plots

# ---------------- AI Insights ----------------
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

ai_text = "\n".join(heuristic_insights(filtered_df))
st.markdown("\n".join([f"- {i}" for i in ai_text.split("\n") if i.strip()]))
st.session_state['ai_insights'] = ai_text

# Optional OpenAI integration
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
        st.session_state['ai_insights'] = ai_text
    except Exception:
        pass

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
        clean_line = ''.join([c for c in line if ord(c) < 128])
        pdf.multi_cell(0, 6, clean_line)

    pdf.ln(5)
    for name, fig in plots.items():
        img_bytes = fig.to_image(format="png", engine="kaleido")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(img_bytes)
        tmp.flush()
        tmp.close()
        pdf.image(tmp.name, w=170)
    
    return pdf.output(dest="S").encode("utf-8")

if st.button("Generate PDF Report"):
    kpi_list = []
    if total_sales is not None: kpi_list.append(f"Total Sales: ${total_sales:,.2f}")
    if total_profit is not None: kpi_list.append(f"Total Profit: ${total_profit:,.2f}")
    if avg_discount is not None: kpi_list.append(f"Avg Discount: {avg_discount:.2%}")
    if total_quantity is not None: kpi_list.append(f"Total Quantity: {int(total_quantity):,}")

    pdf_bytes = create_pdf(kpi_list, plots, ai_text)
    st.download_button("📥 Download PDF", data=pdf_bytes, file_name="SmartDashboard_Report.pdf", mime="application/pdf")
