import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


st.title('Flexible MLP Setup')

if 'df' not in st.session_state:
    st.error("Please upload data in the 'Data Input' tab first.")
else:
    df = st.session_state.df

    # Initialize model variables
    model = None
    model_fitted = False  # Flag to indicate if model has been fitted

    # Step 1: Variable Selection
    all_columns = df.columns.tolist()
    selected_vars = st.sidebar.multiselect('Select Variables for MLP', all_columns)

    if not selected_vars:
        st.warning("Please select at least one variable.")
    else:
        # Step 2: Lagged Variables
        lagged_vars = st.sidebar.multiselect('Select Variables for Lagging', selected_vars)
        lags = {var: st.sidebar.slider(f'Number of Lags for {var}', 0, 50, 1) for var in lagged_vars}

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

        # Step 5: MLP Model Parameters
        y_axis = st.sidebar.selectbox('Select Y Axis for MLP', all_columns)
        y = df[y_axis].loc[X.index]

                # Scale features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)

        # Select proportion of training data and split method
        test_size = st.sidebar.slider('Test Size Proportion', 0.1, 0.9, 0.2)
        split_method = st.sidebar.selectbox('Select Data Split Method', ['Random', 'End', 'Middle'])

        # Hyperparameters
        hidden_layer_sizes = st.sidebar.slider('Number of Hidden Layers', 1, 100, 2)
        neurons_per_layer = st.sidebar.slider('neurons_per_layer', 1, 100, 2)
        activation = st.sidebar.selectbox('Activation Function', ['identity', 'logistic', 'tanh', 'relu'])
        max_iter = st.sidebar.slider('Max Iterations', 100, 10000, 200)

        # Split data into training and test sets
        if split_method == 'Random':
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        elif split_method == 'End':
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        elif split_method == 'Middle':
            split_idx = len(X) // 2
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

        # Initialize MLP model
        model = MLPRegressor(hidden_layer_sizes=(neurons_per_layer,) * hidden_layer_sizes, 
                             activation=activation, max_iter=max_iter, random_state=0)

        # Initialize placeholders
        placeholder_matrix = st.container()
        placeholder_results = st.container()
        placeholder_plot = st.container()

        # Run MLP function
        def run_mlp(update=False):
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
                cv_model = MLPRegressor(hidden_layer_sizes=(hidden_layer_sizes,) * hidden_layer_sizes, 
                                        activation=activation, max_iter=max_iter, random_state=0)
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
                    st.subheader('MLP Model Results')
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
            run_mlp(update=True)

        # Initial run of the model
        if not model_fitted:
            run_mlp()

        # Step 7: Predict New Data
        st.sidebar.subheader('Predict New Data')
        if model_fitted:
            st.subheader('Enter New Data for Prediction')
    
            # Use the first row of the dataset as default values
            default_values = X.iloc[0].to_dict()
            
            # Create input fields for the new data with default values
            input_data = {}
            
            print(f"xscaled {X_scaled}")
            for feature in X:
                default_value = default_values[feature]
                input_data[feature] = st.number_input(f'Enter value for {feature}', value=default_value, format="%.2f")
    
            # Convert input data to DataFrame
            new_data = pd.DataFrame([input_data])
            new_data = scaler_X.transform(new_data)
            # Make prediction if model is fitted
            if st.button('Predict'):
                prediction = model.predict(new_data)
                st.write(f'Predicted value: {prediction[0]:.2f}')
        else:
            st.warning("Please train the model in the 'Model Training' section before making predictions.")
