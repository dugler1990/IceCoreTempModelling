import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

st.title('Flexible Regression Setup')

# Check if dataframe is available
if 'df' not in st.session_state:
    st.error("Please upload data in the 'Data Input' tab first.")
else:
    df = st.session_state.df

    # Initialize session state variables
    if 'filter_count' not in st.session_state:
        st.session_state.filter_count = 0
    if 'filter_thresholds' not in st.session_state:
        st.session_state.filter_thresholds = []
    if 'filtered_features' not in st.session_state:
        st.session_state.filtered_features = []

    # Step 1: Variable Selection
    all_columns = df.columns.tolist()
    selected_vars = st.sidebar.multiselect('Select Variables for Regression', all_columns)

    if not selected_vars:
        st.warning("Please select at least one variable.")
    else:
        # Step 2: Lagged Variables
        lagged_vars = st.sidebar.multiselect('Select Variables for Lagging', selected_vars)
        lags = {var: st.sidebar.slider(f'Number of Lags for {var}', 0, 5, 1) for var in lagged_vars}

        # Step 3: Transformations
        transformation_options = st.sidebar.multiselect('Select Transformations', ['Squared', 'Cubic'], default=['Squared', 'Cubic'])

        # Step 4: Change Calculation
        change_vars = st.sidebar.multiselect('Select Variables for Change Calculation', selected_vars)

        # Create feature matrix
        X = pd.DataFrame(index=df.index)

        for var in selected_vars:
            # Original variable
            X[var] = df[var]

            # Apply lagged variables
            if var in lagged_vars:
                for lag in range(1, lags[var] + 1):
                    X[f'{var}_lagged_{lag}'] = df[var].shift(lag)

                    # Apply transformations to lagged variables
                    if 'Squared' in transformation_options:
                        X[f'{var}_lagged_{lag}_squared'] = X[f'{var}_lagged_{lag}'] ** 2
                    if 'Cubic' in transformation_options:
                        X[f'{var}_lagged_{lag}_cubic'] = X[f'{var}_lagged_{lag}'] ** 3

            # Apply transformations to original variables
            if 'Squared' in transformation_options:
                X[f'{var}_squared'] = df[var] ** 2
            if 'Cubic' in transformation_options:
                X[f'{var}_cubic'] = df[var] ** 3

            # Calculate changes
            if var in change_vars:
                X[f'{var}_change'] = df[var].diff().shift(-1)

                # Apply transformations to changes
                if 'Squared' in transformation_options:
                    X[f'{var}_change_squared'] = X[f'{var}_change'] ** 2
                if 'Cubic' in transformation_options:
                    X[f'{var}_change_cubic'] = X[f'{var}_change'] ** 3

                # Apply lagged changes
                if var in lagged_vars:
                    for lag in range(1, lags[var] + 1):
                        X[f'{var}_change_lagged_{lag}'] = df[var].diff().shift(lag)

                        # Apply transformations to lagged changes
                        if 'Squared' in transformation_options:
                            X[f'{var}_change_lagged_{lag}_squared'] = X[f'{var}_change_lagged_{lag}'] ** 2
                        if 'Cubic' in transformation_options:
                            X[f'{var}_change_lagged_{lag}_cubic'] = X[f'{var}_change_lagged_{lag}'] ** 3

        # Drop rows with NaN values (from lagging and differencing)
        X = X.dropna()

        # Step 5: P-Value Filtering
        y_axis = st.sidebar.selectbox('Select Y Axis for Regression', all_columns)
        y = df[y_axis].loc[X.index]

        # Track button click and filtering process
        p_value_threshold = st.sidebar.number_input('Select p-value Threshold', min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        apply_filter_button = st.sidebar.button('Apply P-Value Filter')

        if apply_filter_button:
            st.session_state.filter_count += 1
            st.session_state.filter_thresholds.append(p_value_threshold)

        # Run the iterative filtering
        current_X = X.copy()
        removed_features = []
        for _ in range(st.session_state.filter_count):
            X_with_const = sm.add_constant(current_X)
            model = sm.OLS(y, X_with_const).fit()
            
            p_values = model.pvalues[1:]  # Exclude constant
            threshold = st.session_state.filter_thresholds[_]
            to_remove = p_values[p_values >= threshold].index.tolist()
            removed_features.extend(to_remove)
            
            # Remove features from current_X
            current_X = current_X.drop(columns=to_remove, errors='ignore')
        
        st.session_state.filtered_features = list(set(removed_features))  # Unique features to remove

        # Step 6: Feature Removal Dropdown
        features = X.columns.tolist()
        valid_filtered_features = [f for f in st.session_state.filtered_features if f in features]
        features_to_remove = st.sidebar.multiselect('Select Features to Remove', valid_filtered_features, default=valid_filtered_features)

        # Remove selected features from the DataFrame
        if features_to_remove:
            X = X.drop(columns=features_to_remove, errors='ignore')

        # Display the created feature matrix and summary statistics
        st.subheader('Independent Variables Created')
        st.write(X.head())
        st.write("Summary Statistics:")
        st.write(X.describe().T[['mean', 'min', 'max']])

        # Rebuild the model after feature removal
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()

        # Display updated regression results
        st.subheader('Updated Regression Results After Feature Removal')
        st.write(model.summary())

        # Step 7: Regression Plot
        st.subheader('Regression Plot')
        plot_type = st.sidebar.selectbox('Choose Plot Type', ['Line', 'Scatter'])

        fig, ax = plt.subplots(figsize=(10, 6))
        if plot_type == 'Scatter':
            ax.scatter(df.index, df[y_axis], label='Data', color='blue')
        elif plot_type == 'Line':
            ax.plot(df.index, df[y_axis], label='Data', color='blue')
        ax.plot(X.index, model.predict(X_with_const), label='Fitted Line', color='red')
        ax.set_xlabel('Index')
        ax.set_ylabel(y_axis)
        ax.legend()
        st.pyplot(fig)
