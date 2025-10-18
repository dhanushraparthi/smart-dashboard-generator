import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import openai

st.set_page_config(page_title="Smart Dashboard Generator", layout="wide", page_icon="📊")
st.title("📊 Smart Dashboard Generator")

# --- Sidebar: Upload Data ---
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Choose CSV or Excel file", type=['csv', 'xlsx'])
ai_key_input = st.sidebar.text_input("OpenAI API Key (optional, for AI insights)", type="password")

# --- Load Data ---
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    st.info("Upload a dataset to get started.")
    st.stop()

# --- Sidebar: Filters ---
st.sidebar.header("Filter data")
filter_values = {}
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        options = df[col].unique().tolist()
        selected = st.sidebar.multiselect(f"{col}", options, default=options)
        filter_values[col] = selected
    elif pd.api.types.is_numeric_dtype(df[col]):
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        selected = st.sidebar.slider(f"{col}", min_val, max_val, (min_val, max_val))
        filter_values[col] = selected

# --- Apply Filters ---
filtered_df = df.copy()
for col, val in filter_values.items():
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        filtered_df = filtered_df[filtered_df[col].isin(val)]
    else:
        filtered_df = filtered_df[(filtered_df[col] >= val[0]) & (filtered_df[col] <= val[1])]

if filtered_df.empty:
    st.warning("No data available after applying filters.")
    st.stop()

# --- Key Metrics ---
st.header("📈 Key Metrics")
numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns
kpi_list = []
for col in numeric_cols:
    mean_val = filtered_df[col].mean()
    max_val = filtered_df[col].max()
    min_val = filtered_df[col].min()
    kpi_list.append(f"{col}: Mean={mean_val:.2f}, Max={max_val:.2f}, Min={min_val:.2f}")

st.write("\n".join(kpi_list))

# --- AI Insights ---
ai_insights = []
if ai_key_input:
    openai.api_key = ai_key_input
    st.header("🤖 AI Insights")
    try:
        prompt = f"Provide insights for this dataset:\n{filtered_df.head(20).to_dict()}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200
        )
        ai_text = response['choices'][0]['text'].strip()
        ai_insights.append(ai_text)
        st.info(ai_text)
    except Exception as e:
        st.error(f"Error fetching AI insights: {e}")

# --- Dashboard Charts ---
st.header("📊 Visual Dashboards")
neon_colors = px.colors.qualitative.Bold
plots = []

for col in filtered_df.columns:
    if filtered_df[col].dtype == "object" or filtered_df[col].dtype.name == "category":
        top_vals = filtered_df[col].value_counts().reset_index()
        top_vals.columns = ['category', 'count']
        fig = px.bar(
            top_vals,
            x='category',
            y='count',
            text_auto=True,
            title=f"{col} Distribution",
            color='category',
            color_discrete_sequence=neon_colors
        )
    else:
        fig = px.histogram(
            filtered_df,
            x=col,
            nbins=20,
            title=f"{col} Distribution",
            color_discrete_sequence=neon_colors
        )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    plots.append(fig)

# --- Shareable Link ---
st.sidebar.header("Share Dashboard")
params = {col: filter_values[col] for col in filter_values}
st.sidebar.code(f"Your shareable link: ?{pd.io.json.dumps(params)}")

st.success("Dashboard ready! Use the shareable link to share filters.")

