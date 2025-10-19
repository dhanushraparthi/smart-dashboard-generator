import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="🧠 Smart Universal Dashboard Generator", layout="wide")
st.title("🌍 Smart Universal Dashboard Generator")

# ------------------- File Upload -------------------
file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls", "json"])

if file:
    file_ext = file.name.split(".")[-1].lower()
    if file_ext == "csv":
        df = pd.read_csv(file)
    elif file_ext in ["xlsx", "xls"]:
        df = pd.read_excel(file)
    elif file_ext == "json":
        df = pd.read_json(file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    st.success("✅ Data loaded successfully!")
    st.dataframe(df.head())

    # ------------------- Data Type Detection -------------------
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]

    # ------------------- Filters -------------------
    st.sidebar.header("🔍 Filters")
    filters = {}
    for col in cat_cols:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) < 50:
            selected = st.sidebar.multiselect(f"Filter by {col}", options=unique_vals)
            if selected:
                filters[col] = selected

    filtered_df = df.copy()
    for col, vals in filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(vals)]

    # ------------------- Tabs -------------------
    tabs = st.tabs([
        "📈 Overview Dashboard",
        "📊 Categorical Analysis",
        "📉 Correlation Insights",
        "🧠 AI Summary & Conclusion"
    ])

    # ------------------- Overview Dashboard -------------------
    with tabs[0]:
        st.subheader("📈 Overview Dashboard")
        if len(num_cols) > 0:
            st.write("### Key Numerical Insights")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", f"{len(filtered_df):,}")
            col2.metric("Numeric Columns", f"{len(num_cols)}")
            col3.metric("Categorical Columns", f"{len(cat_cols)}")

            # Plot top numerical distributions
            for col in num_cols[:3]:
                fig = px.histogram(filtered_df, x=col, nbins=30, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numerical columns found in dataset.")

    # ------------------- Categorical Dashboard -------------------
    with tabs[1]:
        st.subheader("📊 Categorical Analysis")
        if len(cat_cols) > 0 and len(num_cols) > 0:
            num_col = st.selectbox("Select numeric column for analysis", num_cols)
            for col in cat_cols[:2]:
                fig = px.bar(
                    filtered_df.groupby(col)[num_col].mean().reset_index(),
                    x=col, y=num_col, color=num_col,
                    title=f"Average {num_col} by {col}"
                )
                st.plotly_chart(fig, use_container_width=True)
        elif len(cat_cols) > 0:
            for col in cat_cols[:3]:
                fig = px.histogram(filtered_df, x=col, title=f"Count of {col}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns found.")

    # ------------------- Correlation Dashboard -------------------
    with tabs[2]:
        st.subheader("📉 Correlation Insights")
        if len(num_cols) > 1:
            corr = filtered_df[num_cols].corr()
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)

            # Top correlation pairs
            corr_unstack = corr.unstack().sort_values(ascending=False)
            corr_unstack = corr_unstack[corr_unstack < 1]
            st.write("**Top 5 Strongest Relationships:**")
            st.dataframe(corr_unstack.head(5))
        else:
            st.info("Not enough numerical columns to compute correlations.")

    # ------------------- AI Insights -------------------
    with tabs[3]:
        st.subheader("🧠 AI Summary & Conclusion")

        insights = []

        # General dataset insights
        insights.append(f"Dataset contains **{len(df):,} records** and **{len(df.columns)} columns**.")
        if len(num_cols) > 0:
            insights.append(f"There are **{len(num_cols)} numerical columns**, suitable for trend and correlation analysis.")
        if len(cat_cols) > 0:
            insights.append(f"There are **{len(cat_cols)} categorical columns**, ideal for segmentation or grouping.")
        if len(date_cols) > 0:
            insights.append(f"Date columns detected: {', '.join(date_cols)} — possible for time series visualizations.")

        # Statistical observations
        for col in num_cols[:3]:
            try:
                mean = filtered_df[col].mean()
                std = filtered_df[col].std()
                insights.append(f"Column **{col}** → mean: {mean:.2f}, std: {std:.2f}.")
            except:
                pass

        # Dynamic conclusion
        if len(num_cols) > 1:
            insights.append("Dataset shows multiple numerical relationships — correlation analysis may uncover key drivers.")
        elif len(cat_cols) > 1:
            insights.append("Dataset is mostly categorical — frequency and proportion charts are most useful.")
        else:
            insights.append("Dataset appears simple — not enough variation for complex dashboards.")

        for i in insights:
            st.write("- " + i)

        st.info("This conclusion adapts based on your uploaded dataset — no manual configuration needed!")

    # ------------------- Shareable Link -------------------
    st.sidebar.markdown("### 🔗 Shareable Dashboard Link")
    filters_str = "&".join([f"{k}={','.join(v)}" for k, v in filters.items()])
    share_url = f"?{filters_str}" if filters else "?"
    st.sidebar.code(f"Share this dashboard: {share_url}")

else:
    st.info("👆 Upload a dataset to generate dashboards.")
