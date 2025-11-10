import streamlit as st
import sys
from pathlib import Path

# Add current directory to path for page imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from pages import (
    home,
    classification_prediction,
    regression_prediction,
    data_exploration,
    model_monitoring,
    admin_panel
)

# Page configuration
st.set_page_config(
    page_title="EMI Predict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, clean UI
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    /* Sidebar navigation buttons */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.25rem 0;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    /* Header styling */
    h1 {
        color: #1e3c72;
        font-weight: 700;
        padding-bottom: 1rem;
        border-bottom: 3px solid #2a5298;
    }
    
    h2, h3 {
        color: #2a5298;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stMetric"] label {
        color: white !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: white !important;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2a5298;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("💰 EMI Predict AI")
st.sidebar.markdown("---")

st.sidebar.markdown("### About")
st.sidebar.info(
    "EMI Predict AI is an intelligent financial risk assessment platform "
    "that predicts loan eligibility and EMI amounts using machine learning."
)

st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 EMI Predict AI")

# Render the home page content
home.render()
