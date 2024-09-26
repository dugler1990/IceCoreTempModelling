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
    x_axis_set = set()

    for i, tab in enumerate(plot_tabs):
        with tab:
            st.header(f'Configuration for Plot {i+1}')
            x_axis = st.selectbox(f'Select X Axis for Plot {i+1}', df.columns.tolist(), key=f'x_axis_{i}')
            y_axis = st.selectbox(f'Select Y Axis for Plot {i+1}', df.columns.tolist(), key=f'y_axis_{i}')
            plot_params.append((x_axis, y_axis))
            x_axis_set.add(x_axis)

    # Ensure that all plots have the same x-axis
    if len(x_axis_set) > 1:
        st.error("All plots must have the same X axis.")
    else:
        common_x_axis = x_axis_set.pop()

        # Draw line graphs
        st.header('Line Graphs')
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)

        if num_plots == 1:
            axes = [axes]

        y_columns = [y_axis for _, y_axis in plot_params]

        # Compute correlations for the common x-axis
        correlation_dict = {}
        for i, (x_axis, y_axis) in enumerate(plot_params):
            ax = axes[i]
            ax.plot(df[x_axis], df[y_axis], label=y_axis)
            ax.set_ylabel(y_axis)
            ax.legend()
            ax.set_title(f'Plot {i+1}')

            if x_axis == common_x_axis:
                # Calculate correlations
                correlations = df[y_columns].corr()[y_axis]
                correlation_dict[y_axis] = correlations

                # Add correlation annotations
                for j, other_y in enumerate(y_columns):
                    if other_y != y_axis:
                        corr_value = correlations[other_y]
                        ax.annotate(f'{other_y}: {corr_value:.2f}', 
                                    xy=(0.05, 0.95 - 0.05 * j), 
                                    xycoords='axes fraction',
                                    fontsize=10,
                                    bbox=dict(facecolor='white', alpha=0.5))

        axes[-1].set_xlabel(common_x_axis)  # Set x-label on the last subplot
        st.pyplot(fig)

        # Compute and display correlation matrix for the selected y-axis columns
        st.subheader('Correlation Matrix')
        correlation_matrix = df[y_columns].corr()
        st.write(correlation_matrix)

        # Display correlation matrix as a heatmap
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax_corr, fmt='.2f')
        ax_corr.set_title('Correlation Heatmap')
        st.pyplot(fig_corr)

        # ARCH model parameters
        st.sidebar.header('ARCH Model Parameters')
        p = st.sidebar.slider('ARCH Term (p)', 0, 10, 1)
        q = st.sidebar.slider('GARCH Term (q)', 0, 10, 1)
        mean = st.sidebar.selectbox('Mean Model', ['constant', 'AR', 'ARX', 'lin', 'zero'])
        volatility = st.sidebar.selectbox('Volatility Model', ['constant', 'GARCH', 'EGARCH', 'HARCH', 'APARCH'])

        # Exogenous variables configuration
        if mean == 'ARX':
            st.sidebar.header('ARX Model Configuration')
            exog_columns = st.sidebar.multiselect('Select Exogenous Variables', df.columns.tolist())
            lags = st.sidebar.slider('Lag for Exogenous Variables', 1, 10, 1)
            shift_exog = st.sidebar.slider('Shift Exogenous Variables', 0, 10, 0)
            
            if len(exog_columns) > 0:
                # Shift exogenous variables
                shifted_exog = pd.DataFrame()
                for col in exog_columns:
                    shifted_exog[col] = df[col].shift(shift_exog)
                
                df = df.join(shifted_exog)

        # Select column for ARCH model
        st.sidebar.header('Select Column for ARCH Model')
        arch_column = st.sidebar.selectbox('Select Column for ARCH Model', df.columns.tolist())

        if arch_column:
            st.header(f'ARCH Model for {arch_column}')

            # Create ARX model with exogenous variables
            exog_data = df[shifted_exog.columns] if mean == 'ARX' else None
            model = arch_model(df[arch_column], vol=volatility, p=p, q=q, mean=mean, x=exog_data)
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
