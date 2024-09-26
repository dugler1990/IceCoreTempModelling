import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import io

# Streamlit app title
st.title('Time Series Analysis and ARCH Model')

# Upload dataset
st.sidebar.header('Upload Your Dataset')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv',"tab"])

if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1]

    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension == 'tab':
        df = pd.read_csv(uploaded_file, delimiter='\t')

    st.write("Data Preview:")
    st.write(df.head())

    # Select columns for x and y axes
    st.sidebar.header('Select Columns')
    columns = df.columns.tolist()
    x_axis = st.sidebar.selectbox('Select X Axis', columns)
    y_axes = st.sidebar.multiselect('Select Y Axes', columns)

    # Draw line graphs
    if len(y_axes) > 0:
        st.header('Line Graphs')
        fig, ax = plt.subplots(figsize=(10, 6))
        for y in y_axes:
            ax.plot(df[x_axis], df[y], label=y)
        ax.set_xlabel(x_axis)
        ax.set_ylabel('Values')
        ax.legend()
        st.pyplot(fig)
    
    # ARCH model parameters
    st.sidebar.header('ARCH Model Parameters')
    p = st.sidebar.slider('ARCH Term (p)', 0, 10, 1)
    q = st.sidebar.slider('GARCH Term (q)', 0, 10, 1)
    mean = st.sidebar.selectbox('Mean Model', ['Constant', 'Zero'])
    volatility = st.sidebar.selectbox('Volatility Model', ['Constant', 'GARCH'])

    # Select column for ARCH model
    st.sidebar.header('Select Column for ARCH Model')
    arch_column = st.sidebar.selectbox('Select Column for ARCH Model', columns)

    if arch_column:
        st.header(f'ARCH Model for {arch_column}')

        # Fit the ARCH model
        if mean == 'Constant':
            mean_model = 'constant'
        else:
            mean_model = 'zero'

        if volatility == 'Constant':
            vol_model = 'constant'
        else:
            vol_model = 'garch'
        
        model = arch_model(df[arch_column], vol=vol_model, p=p, q=q, mean=mean_model)
        model_fit = model.fit(disp="off")

        # Show summary
        st.subheader('Model Summary')
        st.text(model_fit.summary().as_text())

        # Plot ACF and PACF
        st.subheader('ACF and PACF')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_acf(df[arch_column].dropna(), ax=axes[0])
        plot_pacf(df[arch_column].dropna(), ax=axes[1])
        axes[0].set_title('ACF')
        axes[1].set_title('PACF')
        st.pyplot(fig)

        # Ljung-Box Test
        st.subheader('Ljung-Box Test')
        lb_test = acorr_ljungbox(df[arch_column].dropna(), lags=[10])
        st.write(pd.DataFrame(lb_test, columns=['Ljung-Box Statistic', 'p-value']))

else:
    st.info("Please upload a CSV file to start.")
