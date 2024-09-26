import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm

# Define tabs
tabs = st.tabs(["Data Input", "Plots", "ARCH Model", "Regression"])

# Manage tab selection
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 'Data Input'

selected_tab = st.selectbox('Select Tab', ["Data Input", "Plots", "ARCH Model", "Regression"])
st.session_state.selected_tab = selected_tab

# Data Input Tab
if st.session_state.selected_tab == 'Data Input':
    with tabs[0]:
        st.header('Data Input')
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

# Plots Tab
if st.session_state.selected_tab == 'Plots':
    with tabs[1]:
        st.header('Plots')
        if 'df' not in st.session_state:
            st.error("Please upload data in the 'Data Input' tab first.")
        else:
            df = st.session_state.df
            num_plots = st.sidebar.slider('Number of Line Plots', 1, 8, 1)
            plot_params = []
            x_axis_set = set()
            for i in range(num_plots):
                st.sidebar.subheader(f'Configuration for Plot {i+1}')
                x_axis = st.sidebar.selectbox(f'Select X Axis for Plot {i}', df.columns.tolist(), key=f'x_axis_{i}')
                y_axis = st.sidebar.selectbox(f'Select Y Axis for Plot {i}', df.columns.tolist(), key=f'y_axis_{i}')
                plot_params.append((x_axis, y_axis))
                x_axis_set.add(x_axis)
            if len(x_axis_set) > 1:
                st.error("All plots must have the same X axis.")
            else:
                common_x_axis = x_axis_set.pop()
                fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
                if num_plots == 1:
                    axes = [axes]
                y_columns = [y_axis for _, y_axis in plot_params]
                for i, (x_axis, y_axis) in enumerate(plot_params):
                    ax = axes[i]
                    ax.plot(df[x_axis], df[y_axis], label=y_axis)
                    ax.set_ylabel(y_axis)
                    ax.legend()
                    ax.set_title(f'Plot {i+1}')
                    if x_axis == common_x_axis:
                        correlations = df[y_columns].corr()[y_axis]
                        for j, other_y in enumerate(y_columns):
                            if other_y != y_axis:
                                corr_value = correlations[other_y]
                                ax.annotate(f'{other_y}: {corr_value:.2f}', 
                                            xy=(0.05, 0.95 - 0.05 * j), 
                                            xycoords='axes fraction',
                                            fontsize=10,
                                            bbox=dict(facecolor='white', alpha=0.5))
                axes[-1].set_xlabel(common_x_axis)
                st.pyplot(fig)
                st.subheader('Correlation Matrix')
                correlation_matrix = df[y_columns].corr()
                st.write(correlation_matrix)
                fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax_corr, fmt='.2f')
                ax_corr.set_title('Correlation Heatmap')
                st.pyplot(fig_corr)

# ARCH Model Tab
if st.session_state.selected_tab == 'ARCH Model':
    with tabs[2]:
        st.header('ARCH Model')
        if 'df' not in st.session_state:
            st.error("Please upload data in the 'Data Input' tab first.")
        else:
            df = st.session_state.df
            p = st.sidebar.slider('ARCH Term (p)', 0, 10, 1)
            q = st.sidebar.slider('GARCH Term (q)', 0, 10, 1)
            mean = st.sidebar.selectbox('Mean Model', ['constant', 'AR', 'ARX', 'lin', 'zero'])
            volatility = st.sidebar.selectbox('Volatility Model', ['constant', 'GARCH', 'EGARCH', 'HARCH', 'APARCH'])
            arch_column = st.sidebar.selectbox('Select Column for ARCH Model', df.columns.tolist())
            if arch_column:
                st.header(f'ARCH Model for {arch_column}')
                if mean == 'ARX':
                    st.sidebar.header('ARX Model Configuration')
                    exog_columns = st.sidebar.multiselect('Select Exogenous Variables', df.columns.tolist())
                    lags = st.sidebar.slider('Lag for Exogenous Variables', 1, 10, 1)
                    include_current = st.sidebar.checkbox('Include Current Value of Exogenous Variables')
                    if len(exog_columns) > 0:
                        temp_df = df.copy()
                        lagged_exog = pd.DataFrame(index=df.index)
                        for col in exog_columns:
                            if include_current:
                                lagged_exog[col] = df[col].shift(-1)
                            for lag in range(1, lags + 1):
                                lagged_exog[f'{col}_lagged_{lag-1}'] = df[col].shift(lag)
                            lagged_exog[col] = df[col]
                        temp_df = pd.concat([temp_df, lagged_exog], axis=1)
                        relevant_columns = [arch_column] + list(lagged_exog.columns)
                        temp_df = temp_df.dropna(subset=relevant_columns)
                    else:
                        temp_df = df.copy()
                else:
                    temp_df = df.copy()
                exog_data = temp_df.filter(like='_lagged') if mean == 'ARX' else None
                model = arch_model(temp_df[arch_column], vol=volatility, p=p, q=q, mean=mean, x=exog_data)
                try:
                    model_fit = model.fit(disp="off")
                    st.subheader('Model Summary')
                    st.text(model_fit.summary().as_text())
                except np.linalg.LinAlgError as e:
                    st.error(f"Model fitting error: {e}")
                st.subheader('ACF and PACF')
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                plot_acf(temp_df[arch_column].dropna(), ax=axes[0])
                plot_pacf(temp_df[arch_column].dropna(), ax=axes[1])
                axes[0].set_title('ACF')
                axes[1].set_title('PACF')
                st.pyplot(fig)
                st.subheader('Ljung-Box Test')
                lb_test = acorr_ljungbox(temp_df[arch_column].dropna(), lags=[10])
                st.write(pd.DataFrame(lb_test, columns=['Ljung-Box Statistic', 'p-value']))

# Regression Tab
if st.session_state.selected_tab == 'Regression':
    with tabs[3]:
        st.header('Regression')
        if 'df' not in st.session_state:
            st.error("Please upload data in the 'Data Input' tab first.")
        else:
            df = st.session_state.df
            x_axis = st.sidebar.selectbox('Select X Axis for Regression', df.columns.tolist())
            y_axis = st.sidebar.selectbox('Select Y Axis for Regression', df.columns.tolist())
            transformation_options = st.sidebar.multiselect('Select Transformations for X Axis', ['None', 'Squared', 'Cubic'])
            X = df[[x_axis]].copy()
            if 'Squared' in transformation_options:
                X[f"{x_axis}_squared"] = X[x_axis] ** 2
            if 'Cubic' in transformation_options:
                X[f"{x_axis}_cubic"] = X[x_axis] ** 3
            X = sm.add_constant(X)
            y = df[y_axis]
            model = sm.OLS(y, X).fit()
            st.subheader('Regression Results')
            st.write(model.summary())
            st.subheader('Regression Plot')
            plt.figure(figsize=(10, 6))
            plt.scatter(df[x_axis], df[y_axis], label='Data', color='blue')
            plt.plot(df[x_axis], model.predict(X), label='Fitted Line', color='red')
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.legend()
            st.pyplot()
