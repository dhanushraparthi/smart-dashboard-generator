import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import io

# Page setup
st.set_page_config(
    page_title="Smart Dashboard Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center; color: #00ffff;'>🌌 Smart Dashboard Generator</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Upload & Filters")
uploaded_file = st.sidebar.file_uploader("Upload your CSV/Excel file", type=["csv", "xlsx"])
ai_insights_input = st.sidebar.text_area("AI Insights / Notes", value="Add your insights here...")

if uploaded_file:
    # Read CSV/Excel
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.sidebar.write("Columns detected:")
    columns = df.columns.tolist()

    # Filters
    filter_cols = st.sidebar.multiselect("Select columns to filter", options=columns)
    filters = {}
    for col in filter_cols:
        unique_vals = df[col].unique().tolist()
        selected = st.sidebar.multiselect(f"Filter {col}", options=unique_vals, default=unique_vals)
        filters[col] = selected

    filtered_df = df.copy()
    for col, selected in filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(selected)]

    st.subheader("Filtered Data")
    st.dataframe(filtered_df)

    # KPI Cards
    st.subheader("Key Metrics")
    kpi_cols = st.multiselect("Select numeric columns for KPIs", options=df.select_dtypes(include="number").columns.tolist())
    kpis = {}
    for col in kpi_cols:
        kpis[col] = {
            "Mean": round(filtered_df[col].mean(), 2),
            "Sum": round(filtered_df[col].sum(), 2),
            "Max": round(filtered_df[col].max(), 2),
            "Min": round(filtered_df[col].min(), 2)
        }
        st.metric(label=f"{col} (Mean)", value=kpis[col]["Mean"])

    # Multi-Chart Dashboard
    st.subheader("Dashboards")
    plots = {}
    neon_colors = px.colors.qualitative.Plotly  # bright neon-like colors

    for col in columns:
        if filtered_df[col].dtype == "object" or filtered_df[col].nunique() < 30:
            fig = px.bar(
                filtered_df[col].value_counts().reset_index(),
                x="index", y=col, text_auto=True,
                title=f"{col.capitalize()} Distribution",
                color="index",
                color_discrete_sequence=neon_colors
            )
        else:
            fig = px.histogram(
                filtered_df, x=col, nbins=20,
                title=f"{col.capitalize()} Distribution",
                color_discrete_sequence=neon_colors
            )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            font=dict(color="cyan")
        )
        st.plotly_chart(fig, use_container_width=True)
        plots[col] = fig

    # AI Insights
    st.subheader("🤖 AI Insights")
    st.markdown(f"<p style='color:#00ff00'>{ai_insights_input}</p>", unsafe_allow_html=True)

    # Download Charts
    st.subheader("Download Charts")
    for name, fig in plots.items():
        # HTML download
        buf_html = io.StringIO()
        fig.write_html(buf_html, include_plotlyjs='cdn')
        buf_html.seek(0)
        st.download_button(
            label=f"📥 Download {name} (HTML)",
            data=buf_html.getvalue(),
            file_name=f"{name}.html",
            mime="text/html"
        )

        # PNG download
        try:
            img_bytes = pio.to_image(fig, format="png")
            buf_png = io.BytesIO(img_bytes)
            st.download_button(
                label=f"📥 Download {name} (PNG)",
                data=buf_png,
                file_name=f"{name}.png",
                mime="image/png"
            )
        except Exception:
            st.warning(f"⚠️ PNG download for {name} failed. Please use HTML download.")

else:
    st.info("Please upload a CSV or Excel file to start generating dashboards.")
