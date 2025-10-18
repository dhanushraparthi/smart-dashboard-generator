import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import tempfile
import os

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Smart Data Dashboard", layout="wide")
st.title("📊 Smart Data Dashboard")
st.markdown("Upload CSV / Excel / JSON to get KPIs, visualizations, AI-style insights, and PDF report.")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader(
    "Upload your dataset", type=["csv", "xlsx", "xls", "json"]
)
if not uploaded_file:
    st.info("Please upload a dataset to begin.")
    st.stop()

def load_file(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    elif name.endswith(".json"):
        return pd.read_json(file)
    else:
        st.error("Unsupported file format")
        st.stop()

df = load_file(uploaded_file)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
st.success("File loaded successfully!")
st.dataframe(df.head())

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")
filters = {}
for col in df.select_dtypes(include=['object', 'category']).columns:
    options = df[col].unique().tolist()
    selected = st.sidebar.multiselect(f"Filter {col}", options, default=options)
    filters[col] = selected

filtered_df = df.copy()
for col, selected in filters.items():
    filtered_df = filtered_df[filtered_df[col].isin(selected)]

# ---------------- KPIs ----------------
st.subheader("Key Metrics")
kpi_cols = st.columns(4)
if "sales" in filtered_df.columns:
    total_sales = filtered_df["sales"].sum()
    kpi_cols[0].metric("Total Sales", f"${total_sales:,.2f}")
else:
    total_sales = None
if "profit" in filtered_df.columns:
    total_profit = filtered_df["profit"].sum()
    kpi_cols[1].metric("Total Profit", f"${total_profit:,.2f}")
else:
    total_profit = None
if "quantity" in filtered_df.columns:
    total_qty = filtered_df["quantity"].sum()
    kpi_cols[2].metric("Total Quantity", f"{int(total_qty):,}")
else:
    total_qty = None
if "discount" in filtered_df.columns:
    avg_discount = filtered_df["discount"].mean()
    kpi_cols[3].metric("Avg Discount", f"{avg_discount:.2%}")
else:
    avg_discount = None

# ---------------- Visual Dashboards ----------------
st.subheader("Visual Insights")
plots = {}

for col in filtered_df.select_dtypes(include=['object', 'category']).columns:
    counts = filtered_df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(counts, x=col, y="count", text_auto=True, title=f"{col.capitalize()} Distribution")
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# Example numeric visualizations
numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    fig = px.histogram(filtered_df, x=col, nbins=20, title=f"{col.capitalize()} Histogram")
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# ---------------- AI-style Insights ----------------
st.subheader("AI-style Insights")
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

ai_insights = heuristic_insights(filtered_df)
for ins in ai_insights:
    st.write(f"- {ins}")

# ---------------- PDF Report ----------------
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            fig.write_image(tmp_file.name)
            pdf.image(tmp_file.name, w=170)
            os.remove(tmp_file.name)
    return pdf.output(dest="S").encode("latin-1", errors="ignore")

kpis_list = []
if total_sales is not None:
    kpis_list.append(f"Total Sales: ${total_sales:,.2f}")
if total_profit is not None:
    kpis_list.append(f"Total Profit: ${total_profit:,.2f}")
if avg_discount is not None:
    kpis_list.append(f"Average Discount: {avg_discount:.2%}")
if total_qty is not None:
    kpis_list.append(f"Total Quantity: {int(total_qty):,}")

if st.button("Download PDF"):
    pdf_bytes = create_pdf(kpis_list, plots, ai_insights)
    st.download_button("📥 Download PDF", data=pdf_bytes, file_name="dashboard_report.pdf", mime="application/pdf")
