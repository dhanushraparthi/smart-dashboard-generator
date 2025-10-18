import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import textwrap
import os

st.set_page_config(page_title="Smart Dashboard Generator", layout="wide")
st.title("📊 Smart Dashboard Generator")

# --- Sidebar for file upload and filters ---
st.sidebar.header("Upload your data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV, Excel, or JSON file", type=["csv", "xlsx", "json"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_json(uploaded_file)

    st.sidebar.header("Filter Data")
    filter_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    filter_dict = {}
    for col in filter_cols:
        options = st.sidebar.multiselect(f"Filter {col}", df[col].unique(), default=df[col].unique())
        filter_dict[col] = options
        df = df[df[col].isin(options)]

    st.header("Data Preview")
    st.dataframe(df.head())

    # --- KPIs ---
    st.header("Key Performance Indicators")
    kpi_list = []
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    for col in numeric_cols:
        mean_val = df[col].mean()
        st.metric(label=f"Average {col}", value=f"{mean_val:.2f}")
        kpi_list.append(f"Average {col}: {mean_val:.2f}")

    # --- Visual Dashboards ---
    st.header("Visual Dashboards")
    plots = []
    for col in filter_cols:
        counts = df[col].value_counts().reset_index()
        counts.columns = ['category', 'count']
        fig = px.bar(counts, x='category', y='count', color='category', text_auto=True, title=f"{col.capitalize()} Distribution")
        st.plotly_chart(fig, use_container_width=True)
        plots.append(fig)

    # --- AI Insights (basic heuristics) ---
    st.header("AI Insights")
    ai_insights = []
    for col in numeric_cols:
        insight = f"Column '{col}' has mean={df[col].mean():.2f} and max={df[col].max():.2f}."
        st.write(insight)
        ai_insights.append(insight)

    # --- PDF Download ---
    st.header("Download PDF Report")
    def create_pdf(kpis, plots, insights):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Smart Dashboard Report", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", '', 12)
        import tempfile

        # Add KPIs
        pdf.cell(0, 8, "Key Performance Indicators:", ln=True)
        for kpi in kpis:
            safe_text = kpi.encode('latin-1', errors='replace').decode('latin-1')
            pdf.multi_cell(0, 6, '\n'.join(textwrap.wrap(safe_text, width=90)))
        pdf.ln(5)

        # Add AI Insights
        pdf.cell(0, 8, "AI Insights:", ln=True)
        for insight in insights:
            safe_text = insight.encode('latin-1', errors='replace').decode('latin-1')
            pdf.multi_cell(0, 6, '\n'.join(textwrap.wrap(safe_text, width=90)))
        pdf.ln(5)

        # Add plots as images
        for fig in plots:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                fig.write_image(tmp_file.name)
                pdf.image(tmp_file.name, w=180)
                os.remove(tmp_file.name)

        return pdf.output(dest='S').encode('latin-1', errors='replace')

    pdf_bytes = create_pdf(kpi_list, plots, ai_insights)

    st.download_button("📥 Download PDF", data=pdf_bytes, file_name="dashboard_report.pdf", mime='application/pdf')
