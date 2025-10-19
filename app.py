import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Smart Dashboard Generator", layout="wide")
st.title("📊 Smart Dashboard Generator with AI Insights")

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

    st.write("### Preview of Data")
    st.dataframe(df.head())

    # ------------------- Sidebar Filters -------------------
    st.sidebar.header("🔍 Filters")
    filters = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) < 50:
            selected = st.sidebar.multiselect(f"Filter by {col}", options=unique_vals)
            if selected:
                filters[col] = selected

    filtered_df = df.copy()
    for col, vals in filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(vals)]

    # ------------------- Helper: Smart Column Finder -------------------
    def find_similar_column(possible_names):
        for col in df.columns:
            for name in possible_names:
                if name.lower() in col.lower():
                    return col
        return None

    product_col = find_similar_column(["product", "item", "name"])
    sales_col = find_similar_column(["sales", "amount", "revenue"])
    profit_col = find_similar_column(["profit", "margin"])
    qty_col = find_similar_column(["quantity", "qty", "count"])

    # ------------------- Key Metrics -------------------
    st.subheader("📈 Key Metrics")
    kpi_texts = []

    if sales_col:
        total_sales = filtered_df[sales_col].sum()
        st.metric("Total Sales", f"${total_sales:,.2f}")
        kpi_texts.append(f"Total Sales: ${total_sales:,.2f}")

    if profit_col:
        total_profit = filtered_df[profit_col].sum()
        st.metric("Total Profit", f"${total_profit:,.2f}")
        kpi_texts.append(f"Total Profit: ${total_profit:,.2f}")

    if qty_col:
        total_qty = filtered_df[qty_col].sum()
        st.metric("Total Quantity", f"{total_qty:,}")
        kpi_texts.append(f"Total Quantity: {total_qty:,}")

    if product_col and sales_col:
        top_selling = (
            filtered_df.groupby(product_col)[sales_col]
            .sum()
            .sort_values(ascending=False)
            .head(1)
        )
        top_product = top_selling.index[0]
        top_value = top_selling.iloc[0]
        st.metric("Top Selling Product", f"{top_product} (${top_value:,.2f})")
        kpi_texts.append(f"Top Selling Product: {top_product} (${top_value:,.2f})")

    # ------------------- AI Insights -------------------
    st.subheader("🤖 AI Insights")

    insights = []
    if sales_col and profit_col:
        profit_margin = (filtered_df[profit_col].sum() / filtered_df[sales_col].sum()) * 100
        insights.append(f"Overall profit margin is {profit_margin:.2f}%.")

    if sales_col and qty_col:
        avg_sales = filtered_df[sales_col].mean()
        insights.append(f"Average sale per record is ${avg_sales:,.2f}.")

    if len(insights) == 0:
        insights.append("Not enough numerical data to generate insights.")

    for i in insights:
        st.write(f"- {i}")

    # ------------------- Visual Dashboards -------------------
    st.subheader("📊 Visual Dashboards")

    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = filtered_df.select_dtypes(exclude=[np.number]).columns.tolist()

    if sales_col and product_col:
        fig1 = px.bar(
            filtered_df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(10).reset_index(),
            x=product_col, y=sales_col,
            title="Top 10 Products by Sales", color=sales_col
        )
        st.plotly_chart(fig1, use_container_width=True)

    elif len(cat_cols) >= 1 and len(numeric_cols) >= 1:
        fig1 = px.bar(filtered_df, x=cat_cols[0], y=numeric_cols[0],
                      title=f"{numeric_cols[0]} by {cat_cols[0]}")
        st.plotly_chart(fig1, use_container_width=True)

    else:
        st.info("Upload a dataset with both numeric and categorical columns to generate visual dashboards.")

    # ------------------- Shareable Link -------------------
    st.subheader("🔗 Shareable Dashboard Link")
    filters_str = "&".join([f"{k}={','.join(v)}" for k, v in filters.items()])
    share_url = f"?{filters_str}" if filters else "?"
    st.code(f"Share this dashboard link: {share_url}")

    # ------------------- Conclusion -------------------
    st.subheader("🧠 Conclusion")
    if sales_col and profit_col:
        if profit_margin > 20:
            st.success("The business is performing strongly with healthy profit margins.")
        elif profit_margin > 10:
            st.warning("Profit margins are moderate — there is room for improvement.")
        else:
            st.error("Profit margins are low — cost optimization may be needed.")
    else:
        st.info("Upload a dataset with Sales and Profit columns to generate a detailed conclusion.")
else:
    st.info("👆 Upload a dataset to begin.")
