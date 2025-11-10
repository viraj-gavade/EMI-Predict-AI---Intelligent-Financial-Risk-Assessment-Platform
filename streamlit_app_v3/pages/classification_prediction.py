import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import os

# Expected feature list for the classification model
FEATURE_LIST = [
    'age', 'gender', 'income', 'employment_type', 'education', 'marital_status',
    'dependents', 'house_type', 'credit_score', 'current_emi_amount', 
    'loan_history', 'existing_loans', 'monthly_expenses', 'bank_balance',
    'emergency_fund', 'requested_amount', 'requested_tenure'
]

def load_classification_models():
    """Load all classification models from models/ directory"""
    models = {}
    models_dir = Path(os.getcwd()) / "models"
    
    if models_dir.exists():
        model_files = list(models_dir.glob("*classif*.pkl")) + list(models_dir.glob("*classif*.joblib"))
        
        for model_file in model_files:
            try:
                model = joblib.load(model_file)
                models[model_file.stem] = model
                st.sidebar.success(f"✅ Loaded: {model_file.name}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load {model_file.name}: {str(e)}")
    
    return models

def prepare_input_data(input_dict, reference_df=None):
    """Prepare and align input data with model requirements"""
    df = pd.DataFrame([input_dict])
    
    # Get model's expected features (if available)
    expected_features = FEATURE_LIST
    
    # Add missing features with default values
    for feature in expected_features:
        if feature not in df.columns:
            # Fill with 0 or median from reference
            if reference_df is not None and feature in reference_df.columns:
                df[feature] = reference_df[feature].median()
            else:
                df[feature] = 0
    
    # Ensure correct column order
    df = df[expected_features]
    return df

def render():
    st.title("🔵 Classification Prediction")
    st.markdown("Predict loan eligibility using trained classification models")
    
    # Load models
    with st.spinner("Loading classification models..."):
        models = load_classification_models()
    
    if not models:
        st.warning("⚠️ No classification models found in `models/` directory.")
        st.info("💡 Tip: Train a classification model and save it with 'classif' in the filename.")
        return
    
    st.success(f"✅ Loaded {len(models)} classification model(s)")
    
    # Model selection
    st.subheader("🤖 Select Model")
    selected_model_name = st.selectbox(
        "Choose a classification model:",
        list(models.keys()),
        help="Select which model to use for predictions"
    )
    selected_model = models[selected_model_name]
    
    st.markdown("---")
    
    # Input method selection
    st.subheader("📥 Input Method")
    input_method = st.radio(
        "Choose how to provide input data:",
        ["Manual Input", "Upload CSV"],
        horizontal=True
    )
    
    predictions = None
    input_data = None
    probabilities = None
    
    if input_method == "Manual Input":
        st.markdown("### 📝 Enter Customer Details")
        
        with st.form("manual_input_form"):
            col1, col2, col3 = st.columns(3)
            
            input_dict = {}
            
            with col1:
                st.markdown("**Personal Information**")
                input_dict['age'] = st.number_input("Age", min_value=18, max_value=100, value=30)
                input_dict['gender'] = st.selectbox("Gender", ["Male", "Female", "Other"])
                input_dict['marital_status'] = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                input_dict['dependents'] = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
                input_dict['education'] = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
            
            with col2:
                st.markdown("**Financial Information**")
                input_dict['income'] = st.number_input("Monthly Income ($)", min_value=0, value=50000, step=1000)
                input_dict['monthly_expenses'] = st.number_input("Monthly Expenses ($)", min_value=0, value=20000, step=1000)
                input_dict['bank_balance'] = st.number_input("Bank Balance ($)", min_value=0, value=100000, step=5000)
                input_dict['emergency_fund'] = st.number_input("Emergency Fund ($)", min_value=0, value=50000, step=5000)
                input_dict['credit_score'] = st.slider("Credit Score", min_value=300, max_value=850, value=700)
            
            with col3:
                st.markdown("**Employment & Loan Details**")
                input_dict['employment_type'] = st.selectbox("Employment Type", ["Private", "Government", "Self-Employed", "Unemployed"])
                input_dict['house_type'] = st.selectbox("House Type", ["Own", "Rented", "Mortgage"])
                input_dict['existing_loans'] = st.selectbox("Existing Loans", ["Yes", "No"])
                input_dict['current_emi_amount'] = st.number_input("Current EMI ($)", min_value=0, value=0, step=100)
                input_dict['loan_history'] = st.selectbox("Loan History", ["Good", "Average", "Poor"])
                input_dict['requested_amount'] = st.number_input("Requested Loan Amount ($)", min_value=0, value=200000, step=10000)
                input_dict['requested_tenure'] = st.number_input("Requested Tenure (months)", min_value=12, max_value=360, value=60, step=12)
            
            submit_button = st.form_submit_button("🔮 Predict Eligibility", use_container_width=True)
        
        if submit_button:
            with st.spinner("Making prediction..."):
                try:
                    # Prepare input data
                    input_data = prepare_input_data(input_dict)
                    
                    # Make prediction
                    predictions = selected_model.predict(input_data)
                    
                    # Get probability if available
                    if hasattr(selected_model, 'predict_proba'):
                        probabilities = selected_model.predict_proba(input_data)
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.exception(e)
    
    else:  # Upload CSV
        st.markdown("### 📤 Upload CSV File")
        st.info("💡 CSV should contain columns matching the model's expected features")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with customer data for batch predictions"
        )
        
        if uploaded_file is not None:
            try:
                input_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(input_data)} records")
                
                with st.expander("📊 View Uploaded Data"):
                    st.dataframe(input_data.head(10))
                
                if st.button("🔮 Run Batch Predictions", use_container_width=True):
                    with st.spinner("Making predictions..."):
                        try:
                            # Prepare data
                            prepared_data = prepare_input_data(input_data.iloc[0].to_dict())
                            
                            # For batch, process all rows
                            all_predictions = []
                            all_probabilities = []
                            
                            for idx, row in input_data.iterrows():
                                row_data = prepare_input_data(row.to_dict())
                                pred = selected_model.predict(row_data)
                                all_predictions.append(pred[0])
                                
                                if hasattr(selected_model, 'predict_proba'):
                                    prob = selected_model.predict_proba(row_data)
                                    all_probabilities.append(prob[0])
                            
                            predictions = np.array(all_predictions)
                            if all_probabilities:
                                probabilities = np.array(all_probabilities)
                            
                        except Exception as e:
                            st.error(f"❌ Batch prediction failed: {str(e)}")
                            st.exception(e)
            
            except Exception as e:
                st.error(f"❌ Failed to read CSV file: {str(e)}")
    
    # Display Results
    if predictions is not None:
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        # Single prediction result
        if len(predictions) == 1:
            result = predictions[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if result == 1 or result == "Approved" or result == "Eligible":
                    st.success("### ✅ APPROVED")
                    st.markdown("**The loan application is likely to be approved!**")
                else:
                    st.error("### ❌ REJECTED")
                    st.markdown("**The loan application is likely to be rejected.**")
            
            with col2:
                if probabilities is not None:
                    st.markdown("### 📈 Confidence Scores")
                    prob_df = pd.DataFrame({
                        'Class': ['Rejected', 'Approved'],
                        'Probability': probabilities[0]
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x='Class',
                        y='Probability',
                        color='Probability',
                        color_continuous_scale=['red', 'green'],
                        text='Probability',
                        title='Prediction Probabilities'
                    )
                    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display probability values
                    st.metric("Rejection Probability", f"{probabilities[0][0]:.2%}")
                    st.metric("Approval Probability", f"{probabilities[0][1]:.2%}")
        
        else:  # Batch predictions
            st.markdown(f"### 📦 Batch Results ({len(predictions)} records)")
            
            results_df = input_data.copy()
            results_df['Prediction'] = predictions
            
            if probabilities is not None:
                results_df['Approval_Probability'] = probabilities[:, 1]
                results_df['Rejection_Probability'] = probabilities[:, 0]
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            approved_count = sum(predictions == 1)
            rejected_count = len(predictions) - approved_count
            
            with col1:
                st.metric("Total Applications", len(predictions))
            with col2:
                st.metric("✅ Approved", approved_count, delta=f"{approved_count/len(predictions)*100:.1f}%")
            with col3:
                st.metric("❌ Rejected", rejected_count, delta=f"{rejected_count/len(predictions)*100:.1f}%")
            
            # Display results table
            st.dataframe(results_df)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="classification_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Visualization
            fig = px.pie(
                values=[approved_count, rejected_count],
                names=['Approved', 'Rejected'],
                title='Prediction Distribution',
                color_discrete_sequence=['#00cc96', '#ef553b']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Compare Models Section
    if len(models) > 1 and input_method == "Manual Input":
        st.markdown("---")
        st.header("🔄 Compare All Models")
        st.markdown("See how different models predict on the same input")
        
        if st.button("🔍 Compare Models", use_container_width=True):
            if 'input_dict' in locals():
                comparison_results = []
                
                with st.spinner("Running predictions across all models..."):
                    for model_name, model in models.items():
                        try:
                            input_data = prepare_input_data(input_dict)
                            pred = model.predict(input_data)[0]
                            
                            result_dict = {
                                'Model': model_name,
                                'Prediction': 'Approved' if pred == 1 else 'Rejected'
                            }
                            
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(input_data)[0]
                                result_dict['Approval_Probability'] = f"{proba[1]:.2%}"
                                result_dict['Rejection_Probability'] = f"{proba[0]:.2%}"
                            
                            comparison_results.append(result_dict)
                        
                        except Exception as e:
                            st.warning(f"⚠️ {model_name} failed: {str(e)}")
                
                if comparison_results:
                    st.markdown("### 📊 Model Comparison Results")
                    comparison_df = pd.DataFrame(comparison_results)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Visualize comparison
                    if 'Approval_Probability' in comparison_df.columns:
                        comparison_df['Approval_Prob_Numeric'] = comparison_df['Approval_Probability'].str.rstrip('%').astype(float) / 100
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=comparison_df['Model'],
                            y=comparison_df['Approval_Prob_Numeric'],
                            text=comparison_df['Approval_Probability'],
                            textposition='auto',
                            marker_color='lightblue'
                        ))
                        fig.update_layout(
                            title='Approval Probability Comparison Across Models',
                            xaxis_title='Model',
                            yaxis_title='Approval Probability',
                            yaxis_tickformat='.0%'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Please submit the manual input form first!")
    
    st.markdown("---")
    st.info("💡 **Tip**: For best results, ensure all input values are accurate and within expected ranges.")

# Run the page
if __name__ == "__main__":
    render()
