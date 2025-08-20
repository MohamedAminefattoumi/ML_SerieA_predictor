import streamlit as st
import pandas as pd

st.title("Dataset Viewer & Feature Selector")

# 1. Upload CSV file
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file:
    # Read the uploaded CSV into a DataFrame
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", data.head())

    # 2. Show all columns
    st.write("### Columns in dataset")
    st.write(list(data.columns))

    # 3. Select columns to keep
    selected_columns = st.multiselect("Select columns to keep", data.columns, default=list(data.columns))
    if selected_columns:
        st.write("### Dataset with selected columns")
        st.dataframe(data[selected_columns])

    # 4. Option to remove columns
    remove_columns = st.multiselect("Or select columns to remove", data.columns)
    if remove_columns:
        remaining_columns = [col for col in data.columns if col not in remove_columns]
        st.write("### Dataset after removing selected columns")
        st.dataframe(data[remaining_columns])
