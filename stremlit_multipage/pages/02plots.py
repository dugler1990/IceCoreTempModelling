import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Plots')

if 'df' not in st.session_state:
    st.error("Please upload data in the 'Data Input' tab first.")
else:
    df = st.session_state.df

    # Initialize or update the number of plots in session state
    if 'num_plots' not in st.session_state:
        st.session_state.num_plots = st.sidebar.slider('Number of Line Plots', 1, 8, 1)
    else:
        st.session_state.num_plots = st.sidebar.slider('Number of Line Plots', 1, 8, st.session_state.num_plots)
    
    num_plots = st.session_state.num_plots

    # Initialize plot parameters in session state if not already set
    if 'plot_params' not in st.session_state:
        st.session_state.plot_params = [(df.columns[0], df.columns[1])] * num_plots
    else:
        # Ensure plot_params length matches num_plots
        if len(st.session_state.plot_params) != num_plots:
            st.session_state.plot_params = [(df.columns[0], df.columns[1])] * num_plots

    # Print session state for debugging
    print(f"Session state before widget setup: {list(st.session_state.keys())}")
    print(f"Initial plot_params: {st.session_state.plot_params}")

    # Display selectboxes for each plot configuration
    for i in range(num_plots):
        x_axis_key = f'x_axis_{i}'
        y_axis_key = f'y_axis_{i}'

        # Print session state before widget creation
        print(f"Session state before selectbox {i}: {list(st.session_state.keys())}")

        # Ensure plot_params has the necessary keys
        if len(st.session_state.plot_params) <= i:
            st.session_state.plot_params.extend([(df.columns[0], df.columns[1])] * (i + 1 - len(st.session_state.plot_params)))

        # Get current x_axis and y_axis from plot_params
        current_x_axis, current_y_axis = st.session_state.plot_params[i]

        # Create selectboxes with the saved state values
        x_axis = st.sidebar.selectbox(
            f'Select X Axis for Plot {i+1}', 
            df.columns.tolist(), 
            key=x_axis_key,
            index=df.columns.tolist().index(current_x_axis)  # Set index based on the saved value
        )
        y_axis = st.sidebar.selectbox(
            f'Select Y Axis for Plot {i+1}', 
            df.columns.tolist(), 
            key=y_axis_key,
            index=df.columns.tolist().index(current_y_axis)  # Set index based on the saved value
        )

        # Update plot_params in session state
        st.session_state.plot_params[i] = (x_axis, y_axis)

    # Print the updated plot parameters and session state
    print(f"Updated plot_params: {st.session_state.plot_params}")
    print(f"Session state after widget setup: {list(st.session_state.keys())}")

    # Ensure that the X axis is the same for all plots
    x_axis_set = {x_axis for x_axis, _ in st.session_state.plot_params if x_axis is not None}
    
    if len(x_axis_set) > 1:
        st.error("All plots must have the same X axis.")
    else:
        common_x_axis = next(iter(x_axis_set), None)
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]
        y_columns = [y_axis for _, y_axis in st.session_state.plot_params if y_axis is not None]
        
        for i, (x_axis, y_axis) in enumerate(st.session_state.plot_params):
            if x_axis is not None and y_axis is not None:
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
        if common_x_axis:
            axes[-1].set_xlabel(common_x_axis)
        st.pyplot(fig)

        st.subheader('Correlation Matrix')
        if y_columns:
            correlation_matrix = df[y_columns].corr()
            st.write(correlation_matrix)
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax_corr, fmt='.2f')
            ax_corr.set_title('Correlation Heatmap')
            st.pyplot(fig_corr)

    # Print final state for debugging
    print(f"Final session state: {list(st.session_state.keys())}")
    print(f"Final plot_params: {st.session_state.plot_params}")
