import streamlit as st
import pandas as pd
import plotly.express as px

def render():
    st.title("Data Exploration")
    st.markdown("Explore the EMI prediction dataset with interactive visualizations.")
    
    # Placeholder: load actual dataset when available
    st.info("This page will display dataset statistics, distributions, and correlations.")
    
    # Example visualization placeholder
    with st.expander("Example: Sample data preview"):
        sample_data = pd.DataFrame({
            'income': [50000, 60000, 70000],
            'credit_score': [700, 750, 800],
            'requested_amount': [200000, 250000, 300000]
        })
        st.dataframe(sample_data)
        
        fig = px.scatter(sample_data, x='income', y='requested_amount', 
                        title='Income vs Requested Amount')
        st.plotly_chart(fig, use_container_width=True)
