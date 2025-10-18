# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import io

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Smart Dashboard Generator",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Smart Dashboard Generator")

# ----------------- SIDEBAR -----------------
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV, Excel or JSON file",
    type=["csv", "xlsx", "json"]
)

st.sidebar.header("Filters")
filters = {}

# ----------------- LOAD DATA -----------------
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        st.success("Data loaded successfully!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    st.warning("Please upload a CSV, Excel, or JSON file.")
    st.stop()

# ----------------- CATEGORICAL & NUMERICAL -----------------
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

# ----------------- FILTERS -----------------
for col in cat_cols:
    options = df[col].unique()
    selected = st.sidebar.multiselect(f"Filter by {col}", options, default=options)
    filters[col] = selected

# Apply filters
filtered_df = df.copy()
for col, selected in filters.items():
    filtered_df = filtered_df[filtered_df[col].isin(selected)]

# ----------------- DASHBOARD -----------------
st.header("📈 Interactive Dashboards")
neon_colors = px.colors.qualitative.Dark24  # neon/dark colors

plots = {}

# Categorical charts
for col in cat_cols:
    counts = filtered_df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(
        counts,
        x=col,
        y="count",
        text_auto=True,
        title=f"{col} Distribution",
        color=col,
        color_discrete_sequence=neon_colors,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# Numerical charts
for col in num_cols:
    fig = px.histogram(
        filtered_df,
        x=col,
        nbins=20,
        title=f"{col} Distribution",
        color_discrete_sequence=neon_colors,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
    plots[col] = fig

# ----------------- AI INSIGHTS -----------------
st.header("🤖 AI Insights (Summary)")
ai_insights = []

# Categorical insights
for col in cat_cols:
    if not filtered_df[col].empty:
        top = filtered_df[col].value_counts().idxmax()
        insight = f"Most common {col}: '{top}' ({filtered_df[col].value_counts().max()} occurrences)"
        st.info(insight)
        ai_insights.append(insight)

# Numerical insights
for col in num_cols:
    if not filtered_df[col].empty:
        mean = filtered_df[col].mean()
        median = filtered_df[col].median()
        insight = f"{col}: mean={mean:.2f}, median={median:.2f}"
        st.info(insight)
        ai_insights.append(insight)

# ----------------- DOWNLOAD CHARTS -----------------
st.header("📥 Download Dashboard Charts")
for col, fig in plots.items():
    buf = io.BytesIO()
    try:
        fig.write_image(buf, format="png", engine="kaleido")
        buf.seek(0)
        st.download_button(
            label=f"Download {col} Chart as PNG",
            data=buf,
            file_name=f"{col}_chart.png",
            mime="image/png"
        )
    except Exception as e:
        st.warning(f"⚠️ PNG download for {col} failed. Using HTML instead.")
        html_buf = io.StringIO()
        fig.write_html(html_buf)
        html_buf.seek(0)
        st.download_button(
            label=f"Download {col} Chart as HTML",
            data=html_buf,
            file_name=f"{col}_chart.html",
            mime="text/html"
        )

st.success("Dashboard ready! Apply filters and download charts.")
