import streamlit as st
import pandas as pd
import plotly.express as px
import io
import plotly.io as pio

st.set_page_config(page_title="Smart Dashboard Generator", layout="wide")

st.title("🌟 Smart Dashboard Generator")

# ---------------- Sidebar ----------------
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV, Excel, or JSON", type=["csv", "xlsx", "json"])

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
st.subheader("Download Charts")
for name, fig in plots.items():
    st.markdown(f"**{name.capitalize()} Chart**")

    # Save as HTML
    buf_html = io.StringIO()
    fig.write_html(buf_html, include_plotlyjs='cdn')
    buf_html.seek(0)
    st.download_button(
        label=f"📥 Download {name} chart (HTML)",
        data=buf_html.getvalue(),
        file_name=f"{name}.html",
        mime="text/html"
    )

    # Save as PNG using orca fallback
    try:
        buf_png = io.BytesIO()
        img_bytes = pio.to_image(fig, format="png", engine="kaleido")  # Only works if Kaleido works
        buf_png.write(img_bytes)
        buf_png.seek(0)
        st.download_button(
            label=f"📥 Download {name} chart (PNG)",
            data=buf_png,
            file_name=f"{name}.png",
            mime="image/png"
        )
    except Exception:
        st.warning(f"⚠️ PNG download for {name} chart failed. Please use HTML download.")

st.markdown("---")
st.info("⚡ Upload your dataset, filter columns, explore KPIs & AI insights, visualize data, and download charts as HTML or PNG.")
