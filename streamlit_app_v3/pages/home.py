import streamlit as st
import os
from pathlib import Path

def render():
    # Hero Section
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='color: #1e3c72; font-size: 3rem;'>💰 EMI Predict AI</h1>
            <p style='font-size: 1.5rem; color: #2a5298;'>Intelligent Financial Risk Assessment Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project Introduction
    st.header("📋 About This Project")
    st.markdown("""
    **EMI Predict AI** is a comprehensive machine learning platform designed to assist financial institutions 
    in making data-driven loan approval decisions and EMI (Equated Monthly Installment) predictions.
    
    ### 🎯 Key Features:
    - **Classification Model**: Predicts loan eligibility (Approved/Rejected) based on customer profile
    - **Regression Model**: Estimates optimal EMI amount based on financial parameters
    - **Real-time Predictions**: Instant predictions with user-friendly interface
    - **Data Exploration**: Interactive visualizations and statistical analysis
    - **MLflow Integration**: Complete experiment tracking and model versioning
    - **Admin Dashboard**: Dataset management and system configuration
    
    ### 🔬 Technology Stack:
    - **Machine Learning**: scikit-learn, XGBoost, Random Forest
    - **Data Processing**: pandas, numpy
    - **Visualization**: Plotly, matplotlib
    - **Experiment Tracking**: MLflow
    - **Web Framework**: Streamlit
    """)
    
    st.markdown("---")
    
    # Available Models Section
    st.header("🤖 Available Models")
    
    # Check for models in the models/ directory
    models_dir = Path(os.getcwd()) / "models"
    
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
        
        if model_files:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 1.5rem; border-radius: 10px; color: white;'>
                        <h3 style='color: white; margin: 0;'>📊 Regression Models</h3>
                        <p style='margin: 0.5rem 0 0 0;'>EMI Amount Prediction</p>
                    </div>
                """, unsafe_allow_html=True)
                
                regression_models = [f for f in model_files if 'regress' in f.name.lower()]
                if regression_models:
                    for model in regression_models:
                        st.success(f"✅ {model.name}")
                        st.caption(f"Size: {model.stat().st_size / 1024:.2f} KB")
                else:
                    st.info("🔄 Training in progress...")
            
            with col2:
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                padding: 1.5rem; border-radius: 10px; color: white;'>
                        <h3 style='color: white; margin: 0;'>🔵 Classification Models</h3>
                        <p style='margin: 0.5rem 0 0 0;'>Loan Eligibility Prediction</p>
                    </div>
                """, unsafe_allow_html=True)
                
                classification_models = [f for f in model_files if 'classif' in f.name.lower()]
                if classification_models:
                    for model in classification_models:
                        st.success(f"✅ {model.name}")
                        st.caption(f"Size: {model.stat().st_size / 1024:.2f} KB")
                else:
                    st.info("🔄 Training in progress...")
            
            st.info(f"📁 **Total Models Found**: {len(model_files)} in `models/` directory")
        else:
            st.warning("⚠️ No trained models found in the `models/` directory. Please train models first.")
    else:
        st.error("❌ Models directory not found. Please create a `models/` folder in the project root.")
    
    st.markdown("---")
    
    # Navigation Section
    st.header("🧭 Quick Navigation")
    st.markdown("Choose a section to get started:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background: #f0f2f6; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                <h2 style='margin: 0;'>🔵</h2>
                <h4>Classification</h4>
                <p>Predict loan eligibility</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Classification", use_container_width=True, key="nav_class"):
            st.switch_page("pages/classification_prediction.py")
    
    with col2:
        st.markdown("""
            <div style='background: #f0f2f6; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                <h2 style='margin: 0;'>📊</h2>
                <h4>Regression</h4>
                <p>Calculate EMI amount</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Regression", use_container_width=True, key="nav_reg"):
            st.switch_page("pages/regression_prediction.py")
    
    with col3:
        st.markdown("""
            <div style='background: #f0f2f6; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                <h2 style='margin: 0;'>📈</h2>
                <h4>Explore Data</h4>
                <p>Visualize insights</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Exploration", use_container_width=True, key="nav_explore"):
            st.switch_page("pages/data_exploration.py")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
            <div style='background: #f0f2f6; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                <h2 style='margin: 0;'>🔍</h2>
                <h4>Model Monitor</h4>
                <p>Track experiments</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Monitoring", use_container_width=True, key="nav_monitor"):
            st.switch_page("pages/model_monitoring.py")
    
    with col5:
        st.markdown("""
            <div style='background: #f0f2f6; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                <h2 style='margin: 0;'>⚙️</h2>
                <h4>Admin Panel</h4>
                <p>Manage system</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Admin", use_container_width=True, key="nav_admin"):
            st.switch_page("pages/admin_panel.py")
    
    with col6:
        st.markdown("""
            <div style='background: #f0f2f6; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                <h2 style='margin: 0;'>📚</h2>
                <h4>Documentation</h4>
                <p>Learn more</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("View Docs", use_container_width=True, key="nav_docs"):
            st.info("📖 Documentation available in README.md")
    
    st.markdown("---")
    
    # Statistics Dashboard
    st.header("📊 System Overview")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(
            label="🤖 Models Available",
            value=len(model_files) if models_dir.exists() and model_files else 0,
            delta="Active"
        )
    
    with metric_col2:
        st.metric(
            label="📁 Dataset Status",
            value="Ready" if Path(os.getcwd(), "DataSet", "emi_prediction_dataset.csv").exists() else "Missing",
            delta="Check Admin Panel"
        )
    
    with metric_col3:
        st.metric(
            label="🔬 MLflow Status",
            value="Connected" if Path(os.getcwd(), "mlruns").exists() else "Not Found",
            delta="Tracking enabled"
        )
    
    with metric_col4:
        st.metric(
            label="🚀 System Status",
            value="Online",
            delta="Running"
        )
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem 0;'>
            <p><strong>EMI Predict AI v3.0</strong> | Built with Streamlit & MLflow</p>
            <p>© 2025 Financial Risk Assessment Platform</p>
        </div>
    """, unsafe_allow_html=True)
