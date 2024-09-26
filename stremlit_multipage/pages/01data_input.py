import streamlit as st
import pandas as pd

st.title('Data Input')

uploaded_file = st.file_uploader("Choose a file", type=['csv', 'tab'])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension == 'tab':
        df = pd.read_csv(uploaded_file, delimiter='\t')
    st.write("Data Preview:")
    st.write(df.head())
    st.session_state.df = df
else:
    st.info("Please upload a CSV or TAB file to start.")
