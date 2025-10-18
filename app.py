import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from pptx import Presentation
from pptx.util import Inches
import io
import tempfile

# ---------------- Streamlit Page Setup ----------------
st.set_page_config(page_title="Smart Dashboard Generator 🌙", layout="wide")

st.markdown("""
    <style>
        body { background-color: #0E1117; color: #FAFAFA; }
        .stApp { background-color: #0E1117; }
        h1, h2, h3 { color: #00FFFF !important; }
        .stButton>button {
            background-color: #00FFFF; color: black; border-radius: 10px; font-weight: bold;
        }
        .stSelectbox, .stMultiSelect, .stTextInput {
            background-color: #1E1E1E !important; color: #FAFAFA !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌙 Smart Dashboard Generator")
st.caption("Upload your dataset → Generate detailed dashboards → Download as PDF or PowerPoint")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader(
    "Upload a dataset (CSV, Excel, JSON)", 
    type=["csv", "xlsx", "xls", "json"]
)

# ---------------- Helper Functions ----------------
def load_file(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    elif uploaded.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    elif uploaded.name.lower().endswith(".json"):
        return pd.read_json(uploaded)
    else:
        st.error("Unsupported file type")
        return None

def generate_ai_insights(df, numeric_cols):
    if not numeric_cols:
        return "No numeric columns available for analysis."

    insights = ""
    for col in numeric_cols:
        mean_val = df[col].mean()
        median_val = df[col].median()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        skewness = df[col].skew()

        insights += f"**{col}:** Avg={mean_val:.2f}, Median={median_val:.2f}, Min={min_val:.2f}, Max={max_val:.2f}, Std={std_val:.2f}. "
        if skewness > 1:
            insights += "Right-skewed distribution detected. "
        elif skewness < -1:
            insights += "Left-skewed distribution detected. "
        else:
            insights += "Distribution is roughly symmetric. "

    # Strong correlations
    corr = df[numeric_cols].corr()
    strong_corrs = []
    for i in numeric_cols:
        for j in numeric_cols:
            if i != j and abs(corr.loc[i,j]) > 0.7:
                strong_corrs.append((i,j,corr.loc[i,j]))
    if strong_corrs:
        insights += "\n\n**Strong correlations:** "
        for i,j,val in strong_corrs:
            insights += f"{i} & {j} (r={val:.2f}); "
    else:
        insights += "\n\nNo strong correlations detected."

    return insights

def create_pdf(dataframe, insights, plots):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(200, 10, "Smart Dashboard Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, insights)
    pdf.ln(5)
    pdf.cell(0, 10, "Data Preview:", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 6, dataframe.head().to_string(index=False))

    for plot in plots:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            plot.savefig(tmp.name, format="png", bbox_inches="tight")
            pdf.image(tmp.name, x=10, y=None, w=180)
    return pdf.output(dest="S").encode("latin-1", errors="ignore")

def export_pptx(kpis, insights, plots):
    prs = Presentation()
    # Title slide
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Smart Dashboard Report"
    slide.placeholders[1].text = "AI Insights and Visuals"

    # KPI slide
    kpi_slide = prs.slides.add_slide(prs.slide_layouts[1])
    kpi_slide.shapes.title.text = "Key Metrics"
    kpi_text = "\n".join([f"{k}: Mean={v['mean']}, Max={v['max']}, Min={v['min']}" for k,v in kpis.items()])
    kpi_slide.placeholders[1].text = kpi_text

    # Insights slide
    insights_slide = prs.slides.add_slide(prs.slide_layouts[1])
    insights_slide.shapes.title.text = "AI Insights"
    insights_slide.placeholders[1].text = insights

    # Chart slides
    for plot in plots:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            plot.savefig(tmp.name, format="png", bbox_inches="tight")
            slide.shapes.add_picture(tmp.name, Inches(1), Inches(1.5), Inches(8), Inches(4.5))
    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf.getvalue()

# ---------------- Main Logic ----------------
if uploaded_file:
    df = load_file(uploaded_file)
    if df is None:
        st.stop()
    st.success("✅ Data loaded successfully!")
    st.dataframe(df.head())

    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    filters = {}
    for col in cat_cols:
        vals = df[col].dropna().unique()
        chosen = st.sidebar.multiselect(f"Filter by {col}", vals)
        if chosen:
            filters[col] = chosen

    for col, vals in filters.items():
        df = df[df[col].isin(vals)]

    st.sidebar.write(f"📦 Rows after filtering: {len(df)}")

    # KPIs
    st.subheader("📊 Key Metrics")
    kpis = {}
    if numeric_cols:
        for col in numeric_cols[:5]:
            kpis[col] = {
                "mean": round(df[col].mean(), 2),
                "max": round(df[col].max(), 2),
                "min": round(df[col].min(), 2)
            }
        kpi_cols = st.columns(len(kpis))
        for i,(col,vals) in enumerate(kpis.items()):
            kpi_cols[i].metric(col, vals["mean"], f"Max: {vals['max']} | Min: {vals['min']}")

    # Multi-Chart Dashboard
    st.subheader("📈 Detailed Visual Dashboard")
    plt.style.use("dark_background")
    plots = []

    # Numeric per category (hist + box)
    for col in numeric_cols:
        for cat_col in cat_cols:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(
                data=df,
                x=col,
                hue=cat_col,
                multiple="stack",
                palette="tab10",
                alpha=0.7,
                ax=ax
            )
            ax.set_title(f"{col} distribution by {cat_col}", color="#00FFFF")
            st.pyplot(fig)
            plots.append(fig)

            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.boxplot(
                x=cat_col,
                y=col,
                data=df,
                palette="tab10",
                ax=ax2
            )
            ax2.set_title(f"{col} boxplot by {cat_col}", color="#00FFFF")
            st.pyplot(fig2)
            plots.append(fig2)

    # Count plots
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(
            x=col,
            data=df,
            palette="tab20",
            order=df[col].value_counts().index
        )
        ax.set_title(f"Counts of {col}", color="#00FFFF")
        st.pyplot(fig)
        plots.append(fig)

    # Numeric per category barplots
    for cat_col in cat_cols:
        for num_col in numeric_cols:
            fig, ax = plt.subplots(figsize=(6,4))
            mean_vals = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
            sns.barplot(
                x=mean_vals.index,
                y=mean_vals.values,
                palette="tab10",
                ax=ax
            )
            ax.set_title(f"Mean {num_col} by {cat_col}", color="#00FFFF")
            st.pyplot(fig)
            plots.append(fig)

    # Correlation Heatmap
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap", color="#00FFFF")
        st.pyplot(fig)
        plots.append(fig)

    # AI Insights
    st.subheader("🧠 AI Insights")
    ai_insights = generate_ai_insights(df, numeric_cols)
    st.markdown(ai_insights)

    # Downloads
    st.subheader("📤 Export Your Dashboard")
    pdf_bytes = create_pdf(df, ai_insights, plots)
    pptx_bytes = export_pptx(kpis, ai_insights, plots)

    c1, c2 = st.columns(2)
    c1.download_button("📄 Download PDF Report", pdf_bytes, "SmartDashboard.pdf", "application/pdf")
    c2.download_button("📊 Download PowerPoint Report", pptx_bytes, "SmartDashboard.pptx",
                       "application/vnd.openxmlformats-officedocument.presentationml.presentation")
else:
    st.info("👆 Please upload a CSV, Excel, or JSON file to begin.")
