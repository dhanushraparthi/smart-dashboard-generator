import streamlit as st
import pandas as pd
import plotly.express as px
import io

st.set_page_config(page_title="Smart Dashboard Generator", layout="wide", initial_sidebar_state="expanded")

st.title("🌟 Smart Dashboard Generator")

# ---------------- Sidebar ----------------
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV, Excel, or JSON", type=["csv","xlsx","json"])

st.sidebar.header("Filters")
filter_columns = {}
if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith("xlsx"):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith("json"):
        df = pd.read_json(uploaded_file)
    else:
        st.sidebar.error("Unsupported file type!")
        st.stop()

    for col in df.select_dtypes(include=['object', 'category']).columns:
        options = st.sidebar.multiselect(f"Filter {col}", df[col].unique(), default=df[col].unique())
        filter_columns[col] = options

    # Apply filters
    for col, opts in filter_columns.items():
        df = df[df[col].isin(opts)]

# ---------------- KPIs ----------------
st.subheader("Key Metrics")
if uploaded_file:
    kpi_list = []
    for col in df.select_dtypes(include=['number']).columns:
        total = df[col].sum()
        avg = df[col].mean()
        kpi_list.append(f"**{col}** → Total: {total}, Average: {round(avg,2)}")

    for kpi in kpi_list:
        st.markdown(kpi)

# ---------------- AI Insights ----------------
st.subheader("AI Insights")
ai_insights = []
if uploaded_file:
    for col in df.select_dtypes(include=['number']).columns:
        top = df[col].idxmax()
        bottom = df[col].idxmin()
        ai_insights.append(f"Column **{col}** → Max: {df[col].max()} (Row {top}), Min: {df[col].min()} (Row {bottom})")

    for insight in ai_insights:
        st.markdown(insight)

# ---------------- Dashboards ----------------
st.subheader("Interactive Dashboards")
plots = {}
if uploaded_file:
    for col in df.select_dtypes(include=['object', 'category']).columns:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(counts, x=col, y="count", color=col, text_auto=True,
                     title=f"{col} Distribution", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        plots[col] = fig

# ---------------- Download Charts ----------------
st.subheader("Download Charts as Images")
for name, fig in plots.items():
    st.markdown(f"**{name.capitalize()} Chart**")

    # Create PNG bytes
    buf = io.BytesIO()
    fig.write_image(buf, format="png")
    buf.seek(0)
    
    st.download_button(
        label=f"📥 Download {name} chart (PNG)",
        data=buf,
        file_name=f"{name}.png",
        mime="image/png"
    )

st.markdown("---")
st.info("⚡ Upload your dataset, filter columns, explore KPIs & AI insights, visualize data, and download charts as PNGs.")
