import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Smart Dashboard Generator", layout="wide")

st.title("📊 Smart Dashboard Generator")
st.markdown("Upload your dataset to generate interactive dashboards with AI insights and key metrics.")

# ------------------ Load Data ------------------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("File loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    st.info("Please upload a CSV or Excel file to continue.")
    st.stop()

# ------------------ Sidebar Filters ------------------
st.sidebar.header("Filters")
filter_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
filters = {}
for col in filter_cols:
    options = df[col].unique().tolist()
    selected = st.sidebar.multiselect(f"Filter by {col}", options, default=options)
    filters[col] = selected

filtered_df = df.copy()
for col, selected in filters.items():
    filtered_df = filtered_df[filtered_df[col].isin(selected)]

# ------------------ Key Metrics ------------------
st.subheader("Key Metrics")
kpi_list = []
if "Sales" in df.columns:
    total_sales = filtered_df["Sales"].sum()
    avg_sales = filtered_df["Sales"].mean()
    max_sales = filtered_df["Sales"].max()
    top_sales_row = filtered_df.loc[filtered_df["Sales"].idxmax()]
    kpi_list.extend([
        f"Total Sales: ${total_sales:,.2f}",
        f"Average Sales: ${avg_sales:,.2f}",
        f"Highest Sale: ${max_sales:,.2f} (Category: {top_sales_row.get('Category','N/A')}, Region: {top_sales_row.get('Region','N/A')})"
    ])

for kpi in kpi_list:
    st.metric(label=kpi.split(":")[0], value=kpi.split(":")[1].strip())

# ------------------ AI Insights ------------------
st.subheader("AI Insights")
ai_insights = []
if "Sales" in df.columns:
    top_category = filtered_df.groupby("Category")["Sales"].sum().idxmax()
    top_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
    ai_insights.append(f"📈 Highest revenue is from the '{top_category}' category.")
    ai_insights.append(f"🌍 Most sales come from the '{top_region}' region.")

for insight in ai_insights:
    st.write(insight)

# ------------------ Visual Dashboards ------------------
st.subheader("Visual Dashboards")
numeric_cols = filtered_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = filtered_df.select_dtypes(include=["object", "category"]).columns.tolist()

neon_colors = px.colors.qualitative.Plotly

for col in categorical_cols:
    fig = px.bar(
        filtered_df[col].value_counts().reset_index(),
        x='index', y=col,
        text_auto=True,
        title=f"{col.capitalize()} Distribution",
        color='index',
        color_discrete_sequence=neon_colors
    )
    st.plotly_chart(fig, use_container_width=True)

for col in numeric_cols:
    fig = px.histogram(
        filtered_df, x=col,
        nbins=20,
        title=f"{col.capitalize()} Distribution",
        color_discrete_sequence=neon_colors
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Shareable Link ------------------
st.subheader("Shareable Dashboard Link")
params = {col: filters[col] for col in filter_cols}
st.write(f"Your shareable link: ?{pd.io.json.dumps(params)}")

# ------------------ Dashboard Conclusion ------------------
st.subheader("Dashboard Conclusion")
def generate_conclusion(kpi_list, ai_insights):
    conclusion_lines = []
    if kpi_list:
        conclusion_lines.append("**Key Metrics Summary:**")
        for kpi in kpi_list:
            conclusion_lines.append(f"- {kpi}")
    if ai_insights:
        conclusion_lines.append("\n**AI Insights:**")
        for insight in ai_insights:
            conclusion_lines.append(f"- {insight}")
    conclusion_lines.append("\n**Overall Conclusion:**")
    conclusion_lines.append(
        "This dashboard provides an interactive overview of the dataset, highlighting top-performing categories, regions, and key trends. "
        "Filters allow dynamic exploration and insights help make data-driven decisions."
    )
    return "\n".join(conclusion_lines)

st.markdown(generate_conclusion(kpi_list, ai_insights))
