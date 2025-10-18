import streamlit as st
import pandas as pd
import plotly.express as px
import os
import tempfile
from fpdf import FPDF
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.shapes import MSO_SHAPE

# ----------------- Streamlit Page Setup -----------------
st.set_page_config(page_title="Smart Data Dashboard", layout="wide")
st.title("📊 Smart Data Dashboard")
st.sidebar.header("Filters & Settings")

# ----------------- File Upload -----------------
uploaded_file = st.sidebar.file_uploader("Upload CSV, Excel, or JSON", type=["csv", "xlsx", "xls", "json"])
if not uploaded_file:
    st.info("Please upload a dataset to proceed.")
    st.stop()

# ----------------- Load Data -----------------
def load_data(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    elif name.endswith(".json"):
        return pd.read_json(file)
    else:
        st.error("Unsupported file type")
        st.stop()

df = load_data(uploaded_file)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
st.write("### Dataset Preview")
st.dataframe(df.head())

# ----------------- Sidebar Filters -----------------
filter_cols = st.sidebar.multiselect("Filter Columns", df.columns.tolist())
filters = {}
for col in filter_cols:
    values = st.sidebar.multiselect(f"{col} values", df[col].unique(), default=df[col].unique())
    filters[col] = values

filtered_df = df.copy()
for col, vals in filters.items():
    filtered_df = filtered_df[filtered_df[col].isin(vals)]

# ----------------- KPIs -----------------
st.subheader("Key Metrics")
kpi_cols = st.columns(4)
total_sales = filtered_df["sales"].sum() if "sales" in filtered_df.columns else None
total_profit = filtered_df["profit"].sum() if "profit" in filtered_df.columns else None
avg_discount = filtered_df["discount"].mean() if "discount" in filtered_df.columns else None
total_quantity = filtered_df["quantity"].sum() if "quantity" in filtered_df.columns else None

if total_sales is not None: kpi_cols[0].metric("Total Sales", f"${total_sales:,.2f}")
if total_profit is not None: kpi_cols[1].metric("Total Profit", f"${total_profit:,.2f}")
if avg_discount is not None: kpi_cols[2].metric("Avg Discount", f"{avg_discount:.2%}")
if total_quantity is not None: kpi_cols[3].metric("Total Quantity", f"{int(total_quantity):,}")

kpis_list = []
if total_sales is not None: kpis_list.append(f"Total Sales: ${total_sales:,.2f}")
if total_profit is not None: kpis_list.append(f"Total Profit: ${total_profit:,.2f}")
if avg_discount is not None: kpis_list.append(f"Avg Discount: {avg_discount:.2%}")
if total_quantity is not None: kpis_list.append(f"Total Quantity: {int(total_quantity):,}")

# ----------------- Plots -----------------
st.subheader("Visual Insights")
plots = {}
for col in filtered_df.select_dtypes(include="object").columns:
    top_vals = filtered_df[col].value_counts().reset_index()
    top_vals.columns = [col, "count"]
    fig = px.bar(top_vals, x=col, y="count", color=col, text_auto=True,
                 title=f"{col.capitalize()} Distribution")
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

for col in filtered_df.select_dtypes(include="number").columns:
    fig = px.histogram(filtered_df, x=col, nbins=20, title=f"{col.capitalize()} Distribution", color_discrete_sequence=["#00FFAA"])
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# ----------------- AI / Heuristic Insights -----------------
st.subheader("Automatic Insights")
ai_insights = []
if "region" in filtered_df.columns and "sales" in filtered_df.columns:
    top_region = filtered_df.groupby("region")["sales"].sum().idxmax()
    ai_insights.append(f"Highest sales region: {top_region}")
if "category" in filtered_df.columns and "profit" in filtered_df.columns:
    top_category = filtered_df.groupby("category")["profit"].sum().idxmax()
    ai_insights.append(f"Most profitable category: {top_category}")
if "discount" in filtered_df.columns and "profit" in filtered_df.columns:
    corr = filtered_df["discount"].corr(filtered_df["profit"])
    ai_insights.append(f"Discount vs Profit correlation: {corr:.2f}")
if "sales" in filtered_df.columns and "profit" in filtered_df.columns:
    margin = filtered_df["profit"].sum()/filtered_df["sales"].sum() if filtered_df["sales"].sum()!=0 else 0
    ai_insights.append(f"Overall profit margin: {margin:.2%}")
if not ai_insights:
    ai_insights.append("Not enough data for detailed insights.")
st.write("\n".join([f"- {line}" for line in ai_insights]))

# ----------------- PDF Export -----------------
def create_pdf(kpis, plots, insights):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Smart Data Dashboard Report", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Arial", size=12)
    for kpi in kpis:
        for part in [kpi[i:i+80] for i in range(0, len(kpi), 80)]:
            pdf.multi_cell(0, 6, part)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Insights:", ln=True)
    pdf.set_font("Arial", size=11)
    for line in insights:
        for part in [line[i:i+80] for i in range(0, len(line), 80)]:
            pdf.multi_cell(0, 6, part)
    pdf.ln(5)
    
    for name, fig in plots.items():
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        try:
            fig.write_image(tmp_file.name, engine="kaleido")
            pdf.image(tmp_file.name, w=170)
        finally:
            tmp_file.close()
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)
    
    return pdf.output(dest="S").encode("latin-1", errors="ignore")

st.subheader("Download Reports")
if st.button("Download PDF"):
    pdf_bytes = create_pdf(kpis_list, plots, ai_insights)
    st.download_button("📥 Download PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")

# ----------------- PPTX Export -----------------
def export_pptx(kpis, insights, plots):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    left = Inches(0.5)
    top = Inches(0.5)
    
    slide.shapes.add_textbox(left, top, Inches(9), Inches(1)).text = "Smart Data Dashboard"
    
    top += Inches(1)
    for kpi in kpis:
        slide.shapes.add_textbox(left, top, Inches(9), Inches(0.3)).text = kpi
        top += Inches(0.3)
    
    slide.shapes.add_textbox(left, top, Inches(9), Inches(0.5)).text = "Insights:"
    top += Inches(0.5)
    for line in insights:
        slide.shapes.add_textbox(left, top, Inches(9), Inches(0.3)).text = line
        top += Inches(0.3)
    
    for name, fig in plots.items():
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        try:
            fig.write_image(tmp_file.name, engine="kaleido")
            slide.shapes.add_picture(tmp_file.name, left, top, width=Inches(9))
            top += Inches(3)
        finally:
            tmp_file.close()
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)
    
    buf = tempfile.NamedTemporaryFile(delete=False, suffix=".pptx")
    prs.save(buf.name)
    buf.seek(0)
    data = buf.read()
    buf.close()
    os.remove(buf.name)
    return data

if st.button("Download PPTX"):
    pptx_data = export_pptx(kpis_list, ai_insights, plots)
    st.download_button("📥 Download PPTX", data=pptx_data, file_name="report.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
