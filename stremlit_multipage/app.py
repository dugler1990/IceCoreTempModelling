

import streamlit as st

st.set_page_config(page_title="Multipage App", layout="wide")

# Define pages
pages = {
    "Data Input": "pages/01data_input.py",
    "Plots": "pages/02plots.py",
    "ARCH Model": "pages/03arch_model.py",
    "Regression": "pages/04regression.py"
}
# Introduction Section
st.title("Ice Core Data Analysis!")
st.write("""
         \n\n\n
    This application allows you to perform various analyses on your data. Here is a brief overview of what you can do:
    
    1. **Data Input**: Upload your CSV or TAB file here. This is where you can preview and prepare your data for analysis.
    
    2. **Plots**: Visualize your data through line plots. You can customize which columns to plot and view correlation matrices.
    
    3. **ARCH Model**: Fit an ARCH model to your data. This section allows you to configure and view the results of the model.
    
    4. **Regression**: Perform regression analysis. You can select variables, apply transformations, and view the regression results and plot.
    

    """)


