import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import os

# Expected feature list for the classification model - updated to match actual model
EXPECTED_FEATURES = [
    'age', 'education', 'monthly_salary', 'years_of_employment', 'family_size', 'dependents',
    'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund', 'requested_amount',
    'requested_tenure', 'max_monthly_emi', 'gender_Male', 'marital_status_Single',
    'employment_type_Private', 'employment_type_Self-Employed', 'company_type_MNC',
    'company_type_Mid-size', 'company_type_Small', 'company_type_Startup', 'house_type_Own',
    'house_type_Rented', 'existing_loans_Yes', 'emi_scenario_Education EMI',
    'emi_scenario_Home Appliances EMI', 'emi_scenario_Personal Loan EMI',
    'emi_scenario_Vehicle EMI', 'total_monthly_expenses', 'savings_ratio',
    'expense_to_income_ratio', 'income_per_member', 'dependents_ratio',
    'employment_stability', 'loan_to_income_ratio', 'emergency_coverage'
]

# Class labels mapping
CLASS_LABELS = {
    0: "Not Eligible",
    1: "Eligible", 
    2: "High Risk"
}

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

def prepare_input_data(input_dict, model=None):
    """Prepare and align input data with model requirements"""
    df = pd.DataFrame([input_dict])
    
    # Get model's expected features
    if model and hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
    else:
        expected_features = EXPECTED_FEATURES
    
    # Feature Engineering and One-Hot Encoding
    # Handle categorical variables with one-hot encoding
    
    # Gender encoding
    df['gender_Male'] = 1 if input_dict.get('gender') == 'Male' else 0
    
    # Marital status encoding
    df['marital_status_Single'] = 1 if input_dict.get('marital_status') == 'Single' else 0
    
    # Employment type encoding
    df['employment_type_Private'] = 1 if input_dict.get('employment_type') == 'Private' else 0
    df['employment_type_Self-Employed'] = 1 if input_dict.get('employment_type') == 'Self-Employed' else 0
    
    # Company type encoding
    df['company_type_MNC'] = 1 if input_dict.get('company_type') == 'MNC' else 0
    df['company_type_Mid-size'] = 1 if input_dict.get('company_type') == 'Mid-size' else 0
    df['company_type_Small'] = 1 if input_dict.get('company_type') == 'Small' else 0
    df['company_type_Startup'] = 1 if input_dict.get('company_type') == 'Startup' else 0
    
    # House type encoding
    df['house_type_Own'] = 1 if input_dict.get('house_type') == 'Own' else 0
    df['house_type_Rented'] = 1 if input_dict.get('house_type') == 'Rented' else 0
    
    # Existing loans encoding
    df['existing_loans_Yes'] = 1 if input_dict.get('existing_loans') == 'Yes' else 0
    
    # EMI scenario encoding
    emi_scenario = input_dict.get('emi_scenario', 'Home Loan EMI')
    df['emi_scenario_Education EMI'] = 1 if emi_scenario == 'Education EMI' else 0
    df['emi_scenario_Home Appliances EMI'] = 1 if emi_scenario == 'Home Appliances EMI' else 0
    df['emi_scenario_Personal Loan EMI'] = 1 if emi_scenario == 'Personal Loan EMI' else 0
    df['emi_scenario_Vehicle EMI'] = 1 if emi_scenario == 'Vehicle EMI' else 0
    
    # Map field names to expected names
    field_mapping = {
        'income': 'monthly_salary',
        'monthly_income': 'monthly_salary',
        'salary': 'monthly_salary',
        'monthly_expenses': 'total_monthly_expenses',
        'expenses': 'total_monthly_expenses'
    }
    
    for old_name, new_name in field_mapping.items():
        if old_name in input_dict and new_name not in df.columns:
            df[new_name] = input_dict[old_name]
    
    # Calculate derived features
    monthly_salary = df.get('monthly_salary', pd.Series([input_dict.get('monthly_salary', input_dict.get('income', 50000))])).iloc[0]
    current_emi = df.get('current_emi_amount', pd.Series([input_dict.get('current_emi_amount', 0)])).iloc[0]
    total_expenses = df.get('total_monthly_expenses', pd.Series([input_dict.get('monthly_expenses', 25000)])).iloc[0]
    dependents = input_dict.get('dependents', 0)
    family_size = input_dict.get('family_size', dependents + 2)
    emergency_fund = input_dict.get('emergency_fund', 50000)
    requested_amount = input_dict.get('requested_amount', 200000)
    
    # Derived calculations
    df['total_monthly_expenses'] = total_expenses + current_emi
    df['savings_ratio'] = (monthly_salary - df['total_monthly_expenses'].iloc[0]) / monthly_salary if monthly_salary > 0 else 0
    df['expense_to_income_ratio'] = df['total_monthly_expenses'].iloc[0] / monthly_salary if monthly_salary > 0 else 0
    df['income_per_member'] = monthly_salary / family_size if family_size > 0 else monthly_salary
    df['dependents_ratio'] = dependents / (dependents + 1)
    
    # Employment stability score
    employment_type = input_dict.get('employment_type', 'Private')
    if employment_type == 'Government':
        df['employment_stability'] = 1.0
    elif employment_type == 'Private':
        df['employment_stability'] = 0.8
    else:  # Self-Employed
        df['employment_stability'] = 0.6
    
    df['loan_to_income_ratio'] = requested_amount / (monthly_salary * 12) if monthly_salary > 0 else 0
    df['emergency_coverage'] = emergency_fund / df['total_monthly_expenses'].iloc[0] if df['total_monthly_expenses'].iloc[0] > 0 else 0
    
    # Handle ordinal encoding for education
    if 'education' in df.columns:
        education_mapping = {
            'High School': 0,
            'Graduate': 1,
            'Post Graduate': 2,
            'Professional': 3
        }
        edu_value = df['education'].iloc[0] if 'education' in input_dict else 'Graduate'
        df['education'] = education_mapping.get(edu_value, 1)
    else:
        education_mapping = {
            'High School': 0,
            'Graduate': 1,
            'Post Graduate': 2,
            'Professional': 3
        }
        edu_value = input_dict.get('education', 'Graduate')
        df['education'] = education_mapping.get(edu_value, 1)
    
    # Add missing features with default values
    for feature in expected_features:
        if feature not in df.columns:
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
                input_dict['education'] = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
            
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
                input_dict['requested_amount'] = st.number_input("Requested Loan Amount ($)", min_value=0, value=200000, step=10000)
                input_dict['requested_tenure'] = st.number_input("Requested Tenure (months)", min_value=12, max_value=360, value=60, step=12)
            
            # Additional required fields
            col4, col5 = st.columns(2)
            with col4:
                input_dict['years_of_employment'] = st.number_input("Years of Employment", min_value=0, max_value=50, value=5)
                input_dict['family_size'] = st.number_input("Family Size", min_value=1, max_value=15, value=input_dict['dependents'] + 2)
                input_dict['company_type'] = st.selectbox("Company Type", ["Large", "MNC", "Mid-size", "Small", "Startup"])
            
            with col5:
                input_dict['emi_scenario'] = st.selectbox("EMI Scenario", ["Home Loan EMI", "Education EMI", "Home Appliances EMI", "Personal Loan EMI", "Vehicle EMI"])
                input_dict['max_monthly_emi'] = st.number_input("Max Monthly EMI ($)", min_value=0, value=15000, step=500)
            
            submit_button = st.form_submit_button("🔮 Predict Eligibility", use_container_width=True)
        
        if submit_button:
            with st.spinner("Making prediction..."):
                try:
                    # Prepare input data
                    input_data = prepare_input_data(input_dict, selected_model)
                    
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
                            # For batch, process all rows
                            all_predictions = []
                            all_probabilities = []
                            
                            for idx, row in input_data.iterrows():
                                row_data = prepare_input_data(row.to_dict(), selected_model)
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
            predicted_class = CLASS_LABELS.get(result, "Unknown")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### 🎯 Loan Eligibility")
                if result == 1:  # Eligible
                    st.success(f"### ✅ {predicted_class}")
                    st.markdown("**The loan application is likely to be approved!**")
                elif result == 2:  # High Risk
                    st.warning(f"### ⚠️ {predicted_class}")
                    st.markdown("**The application requires careful review due to high risk.**")
                else:  # Not Eligible
                    st.error(f"### ❌ {predicted_class}")
                    st.markdown("**The loan application is likely to be rejected.**")
            
            with col2:
                if probabilities is not None:
                    st.markdown("### 📈 Confidence Scores")
                    
                    # Create probability dataframe with class labels
                    class_names = [CLASS_LABELS[i] for i in range(len(probabilities[0]))]
                    prob_df = pd.DataFrame({
                        'Class': class_names,
                        'Probability': probabilities[0]
                    })
                    
                    # Color mapping for different classes
                    colors = ['#ef553b', '#00cc96', '#ff7f0e']  # Red, Green, Orange
                    
                    fig = px.bar(
                        prob_df,
                        x='Class',
                        y='Probability',
                        color='Class',
                        color_discrete_sequence=colors,
                        text='Probability',
                        title='Prediction Probabilities'
                    )
                    fig.update_traces(texttemplate='%{y:.2%}', textposition='outside')
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display probability values as metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Not Eligible", f"{probabilities[0][0]:.2%}")
                    with col_b:
                        st.metric("Eligible", f"{probabilities[0][1]:.2%}")
                    with col_c:
                        st.metric("High Risk", f"{probabilities[0][2]:.2%}")
        
        else:  # Batch predictions
            st.markdown(f"### 📦 Batch Results ({len(predictions)} records)")
            
            results_df = input_data.copy()
            results_df['Prediction_Code'] = predictions
            results_df['Prediction'] = [CLASS_LABELS.get(p, "Unknown") for p in predictions]
            
            if probabilities is not None and probabilities.shape[1] >= 3:
                results_df['Not_Eligible_Prob'] = probabilities[:, 0]
                results_df['Eligible_Prob'] = probabilities[:, 1]
                results_df['High_Risk_Prob'] = probabilities[:, 2]
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            not_eligible_count = sum(predictions == 0)
            eligible_count = sum(predictions == 1)
            high_risk_count = sum(predictions == 2)
            total = len(predictions)
            
            with col1:
                st.metric("Total Applications", total)
            with col2:
                st.metric("✅ Eligible", eligible_count, delta=f"{eligible_count/total*100:.1f}%")
            with col3:
                st.metric("⚠️ High Risk", high_risk_count, delta=f"{high_risk_count/total*100:.1f}%")
            with col4:
                st.metric("❌ Not Eligible", not_eligible_count, delta=f"{not_eligible_count/total*100:.1f}%")
            
            # Display results table
            display_columns = [col for col in results_df.columns if not col.endswith('_Code')]
            st.dataframe(results_df[display_columns], use_container_width=True)
            
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
                values=[not_eligible_count, eligible_count, high_risk_count],
                names=['Not Eligible', 'Eligible', 'High Risk'],
                title='Prediction Distribution',
                color_discrete_sequence=['#ef553b', '#00cc96', '#ff7f0e']
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
                            input_data = prepare_input_data(input_dict, model)
                            pred = model.predict(input_data)[0]
                            
                            result_dict = {
                                'Model': model_name,
                                'Prediction': CLASS_LABELS.get(pred, "Unknown"),
                                'Prediction_Code': pred
                            }
                            
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(input_data)[0]
                                if len(proba) >= 3:
                                    result_dict['Not_Eligible'] = f"{proba[0]:.2%}"
                                    result_dict['Eligible'] = f"{proba[1]:.2%}"
                                    result_dict['High_Risk'] = f"{proba[2]:.2%}"
                                    result_dict['Max_Prob'] = f"{max(proba):.2%}"
                            
                            comparison_results.append(result_dict)
                        
                        except Exception as e:
                            st.warning(f"⚠️ {model_name} failed: {str(e)}")
                
                if comparison_results:
                    st.markdown("### 📊 Model Comparison Results")
                    comparison_df = pd.DataFrame(comparison_results)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Visualize comparison
                    if 'Eligible' in comparison_df.columns:
                        # Create grouped bar chart for 3-class comparison
                        fig = go.Figure()
                        
                        # Convert percentage strings to numeric
                        not_eligible_vals = [float(x.rstrip('%'))/100 for x in comparison_df['Not_Eligible']]
                        eligible_vals = [float(x.rstrip('%'))/100 for x in comparison_df['Eligible']]
                        high_risk_vals = [float(x.rstrip('%'))/100 for x in comparison_df['High_Risk']]
                        
                        fig.add_trace(go.Bar(
                            name='Not Eligible',
                            x=comparison_df['Model'],
                            y=not_eligible_vals,
                            text=comparison_df['Not_Eligible'],
                            textposition='auto',
                            marker_color='#ef553b'
                        ))
                        
                        fig.add_trace(go.Bar(
                            name='Eligible',
                            x=comparison_df['Model'],
                            y=eligible_vals,
                            text=comparison_df['Eligible'],
                            textposition='auto',
                            marker_color='#00cc96'
                        ))
                        
                        fig.add_trace(go.Bar(
                            name='High Risk',
                            x=comparison_df['Model'],
                            y=high_risk_vals,
                            text=comparison_df['High_Risk'],
                            textposition='auto',
                            marker_color='#ff7f0e'
                        ))
                        
                        fig.update_layout(
                            title='Prediction Probability Comparison Across Models',
                            xaxis_title='Model',
                            yaxis_title='Probability',
                            yaxis_tickformat='.0%',
                            barmode='group',
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show prediction consensus
                    if len(comparison_results) > 1:
                        st.markdown("### 🤝 Model Consensus")
                        prediction_counts = comparison_df['Prediction'].value_counts()
                        consensus_fig = px.pie(
                            values=prediction_counts.values,
                            names=prediction_counts.index,
                            title='Prediction Agreement Across Models',
                            color_discrete_sequence=['#ef553b', '#00cc96', '#ff7f0e']
                        )
                        st.plotly_chart(consensus_fig, use_container_width=True)
            else:
                st.warning("⚠️ Please submit the manual input form first!")
    
    st.markdown("---")
    st.info("💡 **Tip**: For best results, ensure all input values are accurate and within expected ranges.")

# Run the page
if __name__ == "__main__":
    render()
