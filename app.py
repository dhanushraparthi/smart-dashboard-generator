import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import tempfile

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Smart Dashboard", layout="wide")
st.title("📊 Smart Data Dashboard")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV dataset.")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
st.success("File loaded!")
st.dataframe(df.head())

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filter Options")
filters = {}
for col in df.select_dtypes(include=['object', 'category']).columns:
    unique_vals = df[col].unique().tolist()
    selected = st.sidebar.multiselect(f"Filter by {col}", unique_vals, default=unique_vals)
    filters[col] = selected
    if selected:
        df = df[df[col].isin(selected)]

# ---------------- KPIs ----------------
st.subheader("Key Metrics")
kpi_cols = st.columns(4)
kpi_cols[0].metric("Total Sales", f"${df['sales'].sum():,.2f}" if "sales" in df else "N/A")
kpi_cols[1].metric("Total Profit", f"${df['profit'].sum():,.2f}" if "profit" in df else "N/A")
kpi_cols[2].metric("Total Quantity", f"{int(df['quantity'].sum())}" if "quantity" in df else "N/A")
kpi_cols[3].metric("Avg Discount", f"{df['discount'].mean():.2%}" if "discount" in df else "N/A")

# ---------------- Plots ----------------
st.subheader("Visual Insights")
plots = {}
for col in df.select_dtypes(include=['object', 'category']).columns:
    top_vals = df[col].value_counts().reset_index()
    top_vals.columns = [col, "count"]
    fig = px.bar(top_vals, x=col, y="count", color=col, text_auto=True,
                 title=f"{col.capitalize()} Distribution")
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# ---------------- AI-style Insights ----------------
st.subheader("AI-style Insights")
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
for i in insights:
    st.markdown(f"- {i}")

# ---------------- PDF Generation ----------------
def create_pdf(df, plots, insights):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Smart Dashboard Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)

    # Add KPIs
    pdf.cell(0, 10, "Key Insights:", ln=True)
    for insight in insights:
        # Split long insights safely
        for part in [insight[i:i+90] for i in range(0, len(insight), 90)]:
            pdf.multi_cell(0, 6, part)
        pdf.ln(1)

    # Add plots
    for col, fig in plots.items():
        pdf.cell(0, 8, f"{col.capitalize()} Distribution", ln=True)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            fig.write_image(tmp_file.name)
            pdf.image(tmp_file.name, w=180)
    return pdf.output(dest="S").encode("latin-1", errors="ignore")

# ---------------- Download PDF ----------------
if st.button("📥 Download PDF"):
    pdf_bytes = create_pdf(df, plots, insights)
    st.download_button("Download PDF", pdf_bytes, file_name="dashboard.pdf", mime="application/pdf")
