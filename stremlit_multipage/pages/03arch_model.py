import streamlit as st
import numpy as np
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import pandas as pd

st.title('ARCH Model')

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
            print(f"include current : {include_current}")
            if len(exog_columns) > 0:
                temp_df = df.copy()
                lagged_exog = pd.DataFrame(index=df.index)
                for col in exog_columns:
                    if include_current:
                        lagged_exog[col] = df[col].shift(-1)
                    else:
                        lagged_exog[col] = df[col]
                    for lag in range(1, lags + 1):
                        lagged_exog[f'{col}_lagged_{lag-1}'] = lagged_exog[col].shift(lag)
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
