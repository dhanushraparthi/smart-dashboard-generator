import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
from pptx import Presentation
from pptx.util import Inches
import tempfile
import os

# ---------------- Page setup ----------------
st.set_page_config(page_title="📊 Smart Data Dashboard", layout="wide")
st.title("📊 Smart Data Dashboard")
st.markdown("Upload CSV / Excel / JSON to get KPIs, visual dashboards, AI insights, and download PDF/PPTX reports.")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls", "json"])
if not uploaded_file:
    st.info("Please upload a dataset to start.")
    st.stop()

# ---------------- Load File ----------------
def load_file(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    if name.endswith(".json"):
        return pd.read_json(file)
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
st.sidebar.header("Filter Data")
filter_col = st.sidebar.selectbox("Select column to filter", options=df.columns)
unique_vals = df[filter_col].unique()
selected_vals = st.sidebar.multiselect("Select values", options=unique_vals, default=unique_vals)
filtered_df = df[df[filter_col].isin(selected_vals)]

# ---------------- KPIs ----------------
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

# ---------------- Visual Dashboards ----------------
st.subheader("Visual Insights")
plots = {}

for col in filtered_df.select_dtypes(include=["object", "category"]).columns:
    top_vals = filtered_df[col].value_counts().reset_index()
    top_vals.columns = [col, "count"]
    try:
        fig = px.bar(top_vals, x=col, y="count", color=col, text_auto=True,
                     title=f"{col.capitalize()} Distribution", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        plots[col] = fig
    except Exception as e:
        st.warning(f"Could not render chart for {col}: {e}")

# ---------------- AI / Heuristic Insights ----------------
st.subheader("AI Insights")
def heuristic_insights(df):
    insights = []
    if "region" in df.columns and "sales" in df.columns:
        insights.append(f"Highest sales region: {df.groupby('region')['sales'].sum().idxmax()}")
    if "category" in df.columns and "profit" in df.columns:
        insights.append(f"Most profitable category: {df.groupby('category')['profit'].sum().idxmax()}")
    if "discount" in df.columns and "profit" in df.columns:
        corr = df["discount"].corr(df["profit"])
        insights.append(f"Discount vs Profit correlation: {corr:.2f}")
    if "sales" in df.columns and "profit" in df.columns:
        margin = df["profit"].sum() / df["sales"].sum() if df["sales"].sum() != 0 else 0
        insights.append(f"Overall profit margin: {margin:.2%}")
    if not insights:
        insights.append("Not enough data for insights.")
    return insights

ai_insights = heuristic_insights(filtered_df)
for i in ai_insights:
    st.write(f"- {i}")

# ---------------- PDF Export ----------------
st.subheader("Download PDF Report")
def create_pdf(kpis, plots, insights):
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
    for line in insights:
        pdf.multi_cell(0, 6, line)
    pdf.ln(5)
    for name, fig in plots.items():
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        try:
            fig.write_image(tmp_file.name, engine="kaleido")
            pdf.image(tmp_file.name, w=170)
        except Exception:
            continue
        finally:
            tmp_file.close()
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)
    return pdf.output(dest="S").encode("latin-1", errors="ignore")

kpis_list = []
if total_sales is not None: kpis_list.append(f"Total Sales: ${total_sales:,.2f}")
if total_profit is not None: kpis_list.append(f"Total Profit: ${total_profit:,.2f}")
if avg_discount is not None: kpis_list.append(f"Avg Discount: {avg_discount:.2%}")
if total_quantity is not None: kpis_list.append(f"Total Quantity: {int(total_quantity):,}")

if st.button("Download PDF"):
    pdf_bytes = create_pdf(kpis_list, plots, ai_insights)
    st.download_button("📥 Download PDF", data=pdf_bytes, file_name="dashboard_report.pdf", mime="application/pdf")

# ---------------- PPTX Export ----------------
st.subheader("Download PPTX Report")
def export_pptx(kpis, insights, plots):
    prs = Presentation()
    slide_layout = prs.slide_layouts[5]  # blank slide

    slide = prs.slides.add_slide(slide_layout)
    top = Inches(0.5)
    for kpi in kpis:
        txBox = slide.shapes.add_textbox(Inches(0.5), top, Inches(9), Inches(0.3))
        tf = txBox.text_frame
        tf.text = kpi
        top += Inches(0.3)

    for insight in insights:
        txBox = slide.shapes.add_textbox(Inches(0.5), top, Inches(9), Inches(0.3))
        tf = txBox.text_frame
        tf.text = insight
        top += Inches(0.3)

    for name, fig in plots.items():
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        try:
            fig.write_image(tmp_file.name, engine="kaleido")
            slide = prs.slides.add_slide(slide_layout)
            slide.shapes.add_picture(tmp_file.name, Inches(1), Inches(1), width=Inches(8))
        except Exception:
            continue
        finally:
            tmp_file.close()
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)

    pptx_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pptx")
    prs.save(pptx_file.name)
    pptx_file.close()
    return pptx_file.name

if st.button("Download PPTX"):
    pptx_path = export_pptx(kpis_list, ai_insights, plots)
    with open(pptx_path, "rb") as f:
        st.download_button("📥 Download PPTX", data=f, file_name="dashboard_report.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
    if os.path.exists(pptx_path):
        os.remove(pptx_path)
