import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import plot_tree

st.title('Flexible XGBoost Setup')

if 'df' not in st.session_state:
    st.error("Please upload data in the 'Data Input' tab first.")
else:
    df = st.session_state.df

    # Initialize model variables
    model = None
    model_fitted = False  # Flag to indicate if model has been fitted

    # Step 1: Variable Selection
    all_columns = df.columns.tolist()
    selected_vars = st.sidebar.multiselect('Select Variables for xgboost', all_columns)

    if not selected_vars:
        st.warning("Please select at least one variable.")
    else:
        # Step 2: Lagged Variables
        lagged_vars = st.sidebar.multiselect('Select Variables for Lagging', selected_vars)
        lags = {var: st.sidebar.slider(f'Number of Lags for {var}', 0, 50, 1) for var in lagged_vars}

        # Step 3: Transformations
        transformation_options = st.sidebar.multiselect('Select Transformations', ['Squared', 'Cubic',"Sqrt"], default=['Squared', 'Cubic',"Sqrt"])

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
                    if'Sqrt' in transformation_options:
                        X[f'{var}_lagged_{lag}_sqrt'] = X[f'{var}_lagged_{lag}'] ** 0.5
                    if 'Squared' in transformation_options:
                        X[f'{var}_lagged_{lag}_squared'] = X[f'{var}_lagged_{lag}'] ** 2
                    if 'Cubic' in transformation_options:
                        X[f'{var}_lagged_{lag}_cubic'] = X[f'{var}_lagged_{lag}'] ** 3

            # Apply transformations to original variables
            if'Sqrt' in transformation_options:
                X[f'{var}_sqrt'] = X[f'{var}'] ** 0.5

            if 'Squared' in transformation_options:
                X[f'{var}_squared'] = df[var] ** 2
            if 'Cubic' in transformation_options:
                X[f'{var}_cubic'] = df[var] ** 3

            # Calculate changes
            if var in change_vars:
                X[f'{var}_change'] = df[var].diff().shift(-1)

                # Apply transformations to changes
                if'Sqrt' in transformation_options:
                    X[f'{var}_change_sqrt'] = X[f'{var}'] ** 0.5
                if 'Squared' in transformation_options:
                    X[f'{var}_change_squared'] = X[f'{var}_change'] ** 2
                if 'Cubic' in transformation_options:
                    X[f'{var}_change_cubic'] = X[f'{var}_change'] ** 3

                # Apply lagged changes
                if var in lagged_vars:
                    for lag in range(1, lags[var] + 1):
                        X[f'{var}_change_lagged_{lag}'] = df[var].diff().shift(lag)

                        # Apply transformations to lagged changes
                        if'Sqrt' in transformation_options:
                            X[f'{var}_change_lagged_{lag}_sqrt'] = X[f'{var}_change_lagged_{lag}'] ** 0.5
                        if 'Squared' in transformation_options:
                            X[f'{var}_change_lagged_{lag}_squared'] = X[f'{var}_change_lagged_{lag}'] ** 2
                        if 'Cubic' in transformation_options:
                            X[f'{var}_change_lagged_{lag}_cubic'] = X[f'{var}_change_lagged_{lag}'] ** 3

        # Drop rows with NaN values (from lagging and differencing)
        X = X.dropna()

        # Step 5: XGBoost Model Parameters
        y_axis = st.sidebar.selectbox('Select Y Axis for xgboost', all_columns)
        y = df[y_axis].loc[X.index]

        # Select proportion of training data and split method
        test_size = st.sidebar.slider('Test Size Proportion', 0.1, 0.9, 0.2)
        split_method = st.sidebar.selectbox('Select Data Split Method', ['Random', 'End', 'Middle'])

        # Hyperparameters
        max_depth = st.sidebar.slider('Max Depth', 1, 10, 6)
        learning_rate = st.sidebar.slider('Learning Rate', 0.01, 0.3, 0.1)
        n_estimators = st.sidebar.slider('Number of Estimators', 10, 500, 100)

        # Split data into training and test sets
        if split_method == 'Random':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        elif split_method == 'End':
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        elif split_method == 'Middle':
            split_idx = len(X) // 2
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

        # Initialize XGBoost model
        model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse',
                                 max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)

        # Initialize placeholders
        placeholder_matrix = st.container()
        placeholder_results = st.container()
        placeholder_plot = st.container()

        # Run XGBoost function
        def run_xgboost(update=False):
            global model, model_fitted

            if update or not model_fitted:
                # Re-train the model if 'Run Full' is clicked or if the model has not been created yet
                model.fit(X_train, y_train)
                model_fitted = True
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
                rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
                mae_train = mean_absolute_error(y_train, y_train_pred)
                mae_test = mean_absolute_error(y_test, y_test_pred)
                r2_train = r2_score(y_train, y_train_pred)
                r2_test = r2_score(y_test, y_test_pred)
                
                # Cross-validation
                cv_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse',
                                            max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(cv_model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
            else:
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
                rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
                mae_train = mean_absolute_error(y_train, y_train_pred)
                mae_test = mean_absolute_error(y_test, y_test_pred)
                r2_train = r2_score(y_train, y_train_pred)
                r2_test = r2_score(y_test, y_test_pred)

            if model is not None:
                # Only update the placeholders with new content
                with placeholder_matrix:
                    st.subheader('Independent Variables Created')
                    st.write(X.head())
                    st.write("Summary Statistics:")
                    st.write(X.describe().T[['mean', 'min', 'max']])

                with placeholder_results:
                    st.subheader('XGBoost Model Results')
                    st.write(f"Training RMSE: {rmse_train:.2f}")
                    st.write(f"Test RMSE: {rmse_test:.2f}")
                    st.write(f"Training MAE: {mae_train:.2f}")
                    st.write(f"Test MAE: {mae_test:.2f}")
                    st.write(f"Training R²: {r2_train:.2f}")
                    st.write(f"Test R²: {r2_test:.2f}")
                    st.write(f"Cross-Validation RMSE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

                with placeholder_plot:
                    st.subheader('Model Predictions and Performance')
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.scatter(y_test.index, y_test, label='True Values', color='blue', alpha=0.6)
                    ax.scatter(y_test.index, y_test_pred, label='Predicted Values', color='red', alpha=0.6)
                    ax.set_xlabel('Index')
                    ax.set_ylabel(y_axis)
                    ax.legend()
                    st.pyplot(fig)

        # Button to run the model and update metrics
        if st.sidebar.button('Run Full'):
            run_xgboost(update=True)

        # Initial run of the model
        if not model_fitted:
            run_xgboost()

        # Step 6: Feature Importance and Tree Visualization
        filter_button = st.sidebar.button('Show Feature Importance')
        tree_button = st.sidebar.button('Visualize Tree')

        if filter_button and model_fitted:
            with placeholder_matrix:
                st.subheader('Independent Variables Created')
                st.write(X.head())
                st.write("Summary Statistics:")
                st.write(X.describe().T[['mean', 'min', 'max']])

            with placeholder_results:
                st.subheader('Feature Importance')
                feature_importances = model.feature_importances_
                importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
                importance_df = importance_df.sort_values(by='Importance', ascending=False)
                st.write(importance_df)

            with placeholder_plot:
                st.subheader('Feature Importance Plot')
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(importance_df['Feature'], importance_df['Importance'])
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                st.pyplot(fig)

        if tree_button and model_fitted:
            with placeholder_matrix:
                st.subheader('Independent Variables Created')
                st.write(X.head())
                st.write("Summary Statistics:")
                st.write(X.describe().T[['mean', 'min', 'max']])

            with placeholder_results:
                st.subheader('Tree Visualization')
                booster = model.get_booster()
                
                # Visualize the first tree
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(booster, rankdir='LR', ax=ax, num_trees=0, with_legend=True)
                st.pyplot(fig)
                
                # Display detailed node information
                st.subheader('Tree Details')
                tree_info = booster.get_dump()[0]
                st.write("Tree Structure:")
                st.text(tree_info)

        # Step 7: Predict New Data
        st.sidebar.subheader('Predict New Data')
        if model_fitted:
            st.subheader('Enter New Data for Prediction')
    
            # Use the first row of the dataset as default values
            default_values = X.iloc[0].to_dict()
            
            # Create input fields for the new data with default values
            input_data = {}
            for feature in X.columns:
                default_value = default_values[feature]
                input_data[feature] = st.number_input(f'Enter value for {feature}', value=default_value, format="%.2f")
    
            # Convert input data to DataFrame
            new_data = pd.DataFrame([input_data])
    
            # Make prediction if model is fitted
            if st.button('Predict'):
                prediction = model.predict(new_data)
                st.write(f'Predicted value: {prediction[0]:.2f}')
        else:
            st.warning("Please train the model in the 'Model Training' section before making predictions.")
