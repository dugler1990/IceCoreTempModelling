import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Streamlit app title
st.title('Time Series Analysis and ARCH Model')

# Upload dataset
st.sidebar.header('Upload Your Dataset')
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'tab'])

if uploaded_file is not None:
    # Read the file based on extension
    file_extension = uploaded_file.name.split('.')[-1]

    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension == 'tab':
        df = pd.read_csv(uploaded_file, delimiter='\t')

    st.write("Data Preview:")
    st.write(df.head())

    # Number of line plots
    num_plots = st.sidebar.slider('Number of Line Plots', 1, 8, 1)

    # Create tabs for each line plot
    plot_tabs = st.sidebar.tabs([f'Plot {i+1}' for i in range(num_plots)])

    # Storage for user inputs
    plot_params = []

    for i, tab in enumerate(plot_tabs):
        with tab:
            st.header(f'Configuration for Plot {i+1}')
            x_axis = st.selectbox(f'Select X Axis for Plot {i+1}', df.columns.tolist(), key=f'x_axis_{i}')
            y_axis = st.selectbox(f'Select Y Axis for Plot {i+1}', df.columns.tolist(), key=f'y_axis_{i}')
            plot_params.append((x_axis, y_axis))

    # Draw line graphs
    st.header('Line Graphs')
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
    
    if num_plots == 1:
        axes = [axes]

    for i, (x_axis, y_axis) in enumerate(plot_params):
        axes[i].plot(df[x_axis], df[y_axis], label=y_axis)
        axes[i].set_ylabel(y_axis)
        axes[i].legend()
        axes[i].set_title(f'Plot {i+1}')
    
    axes[-1].set_xlabel(x_axis)  # Set x-label on the last subplot
    st.pyplot(fig)
    
    # ARCH model parameters
    st.sidebar.header('ARCH Model Parameters')
    p = st.sidebar.slider('ARCH Term (p)', 0, 10, 1)
    q = st.sidebar.slider('GARCH Term (q)', 0, 10, 1)
    mean = st.sidebar.selectbox('Mean Model', ['Constant', 'Zero'])
    volatility = st.sidebar.selectbox('Volatility Model', ['Constant', 'GARCH'])

    # Select column for ARCH model
    st.sidebar.header('Select Column for ARCH Model')
    arch_column = st.sidebar.selectbox('Select Column for ARCH Model', df.columns.tolist())

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
    st.info("Please upload a CSV or TAB file to start.")
