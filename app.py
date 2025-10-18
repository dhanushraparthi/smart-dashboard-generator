import streamlit as st
import pandas as pd
import plotly.express as px
import urllib.parse

st.set_page_config(page_title="Smart Dashboard Generator", layout="wide", page_icon="📊")
st.title("📊 Smart Dashboard Generator")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV, Excel, JSON, Parquet)", 
    type=["csv", "xlsx", "xls", "json", "parquet"]
)

# --- Read URL Query Parameters ---
query_params = st.experimental_get_query_params()

df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        else:
            st.error("Unsupported file type!")
    except Exception as e:
        st.error(f"Error loading file: {e}")

if df is not None:
    st.sidebar.header("Filter Options")
    filter_values = {}

    # Sidebar filters or apply query params
    for col in df.select_dtypes(include=["object", "category"]).columns:
        options = df[col].dropna().unique().tolist()
        default_selection = options
        # If URL has pre-defined filters
        if col in query_params:
            default_selection = query_params[col][0].split(",")
            # Make sure the values exist in column
            default_selection = [v for v in default_selection if v in options]
        selected = st.sidebar.multiselect(f"{col}", options, default=default_selection)
        filter_values[col] = selected

    # Apply filters
    filtered_df = df.copy()
    for col, selected in filter_values.items():
        filtered_df = filtered_df[filtered_df[col].isin(selected)]

    st.subheader("Filtered Dataset")
    st.dataframe(filtered_df, use_container_width=True)

    # --- Dashboard Charts ---
    st.subheader("Visual Dashboards")
    neon_colors = px.colors.qualitative.Bold
    plots = []

    for col in filtered_df.select_dtypes(include=["object", "category"]).columns:
        fig = px.bar(
            filtered_df[col].value_counts().reset_index(),
            x='index',
            y=col,
            text_auto=True,
            title=f"{col} Distribution",
            color='index',
            color_discrete_sequence=neon_colors
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        plots.append(fig)

    # --- AI Insights ---
    st.subheader("🤖 AI Insights")
    ai_insights = []
    for col in filtered_df.select_dtypes(include=["number"]).columns:
        insight = f"{col}: Mean = {filtered_df[col].mean():.2f}, Max = {filtered_df[col].max():.2f}, Min = {filtered_df[col].min():.2f}"
        st.markdown(f"- {insight}")
        ai_insights.append(insight)

    # --- Shareable Link ---
    st.subheader("🔗 Shareable Link")
    params = {}
    for col, selected in filter_values.items():
        params[col] = ",".join(map(str, selected))

    base_url = st.secrets.get("BASE_URL", "https://share.streamlit.io/your-username/your-repo/main/app.py")
    query_str = urllib.parse.urlencode(params)
    shareable_link = f"{base_url}?{query_str}"
    st.text_input("Share this link", shareable_link, key="share_link")
    st.info("Anyone with this link will see the same filtered dashboard!")

else:
    st.info("Please upload a dataset to generate dashboards.")
