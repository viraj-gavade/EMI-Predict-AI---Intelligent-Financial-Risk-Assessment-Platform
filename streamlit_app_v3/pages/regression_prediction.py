import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import os
import json

def standardize_column_names(df):
    """
    Standardize column names to match expected schema.
    Handles case-insensitive matching and common variations.
    """
    column_mapping = {
        'income': 'monthly_salary',
        'salary': 'monthly_salary',
        'monthly_income': 'monthly_salary',
        'age': 'age',
        'gender': 'gender',
        'marital_status': 'marital_status',
        'dependents': 'dependents',
        'education': 'education',
        'employment_type': 'employment_type',
        'company_type': 'company_type',
        'years_of_employment': 'years_of_employment',
        'family_size': 'family_size',
        'monthly_expenses': 'monthly_expenses',
        'expenses': 'monthly_expenses',
        'bank_balance': 'bank_balance',
        'emergency_fund': 'emergency_fund',
        'credit_score': 'credit_score',
        'current_emi_amount': 'current_emi_amount',
        'current_emi': 'current_emi_amount',
        'house_type': 'house_type',
        'existing_loans': 'existing_loans',
        'emi_scenario': 'emi_scenario',
        'requested_amount': 'requested_amount',
        'loan_amount': 'requested_amount',
        'requested_tenure': 'requested_tenure',
        'tenure': 'requested_tenure',
        'loan_tenure': 'requested_tenure',
        'emi_eligibility': 'emi_eligibility'
    }
    
    # Create a case-insensitive mapping
    df_renamed = df.copy()
    for col in df.columns:
        col_lower = col.lower().strip().replace(' ', '_')
        if col_lower in column_mapping:
            df_renamed.rename(columns={col: column_mapping[col_lower]}, inplace=True)
    
    return df_renamed

def safe_get_value(input_dict, keys, default=0):
    """
    Safely get value from dict, trying multiple possible keys.
    Returns default if none found.
    """
    if isinstance(keys, str):
        keys = [keys]
    
    for key in keys:
        if key in input_dict:
            value = input_dict[key]
            if value is not None and value != '':
                return value
    
    return default

def load_regression_models():
    """Load all regression models from models/ directory"""
    models = {}
    models_dir = Path(os.getcwd()) / "models"
    
    if models_dir.exists():
        # Look for regression models
        model_files = list(models_dir.glob("*regress*.pkl")) + list(models_dir.glob("*regress*.joblib"))
        model_files += list(models_dir.glob("*regression*.pkl")) + list(models_dir.glob("*regression*.joblib"))
        
        for model_file in model_files:
            try:
                model = joblib.load(model_file)
                models[model_file.stem] = {
                    'model': model,
                    'path': model_file
                }
                st.sidebar.success(f"✅ Loaded: {model_file.name}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load {model_file.name}: {str(e)}")
    
    return models

def load_feature_names(model_name):
    """Load feature names for a specific model"""
    models_dir = Path(os.getcwd()) / "models"
    
    # Try to load from JSON file first
    features_file = models_dir / f"{model_name}_features.json"
    if features_file.exists():
        with open(features_file, 'r') as f:
            data = json.load(f)
            return data.get('feature_names', [])
    
    # Try generic features file
    generic_file = models_dir / "regression_features.json"
    if generic_file.exists():
        with open(generic_file, 'r') as f:
            data = json.load(f)
            return data.get('feature_names', [])
    
    return None

def preprocess_input(input_dict, model, model_name):
    """
    Preprocess input data to match model's expected features.
    Handles one-hot encoding, feature engineering, and alignment.
    """
    # Standardize field names
    input_dict_std = {}
    for key, value in input_dict.items():
        key_lower = key.lower().strip().replace(' ', '_')
        # Map common variations
        if key_lower in ['income', 'salary', 'monthly_income']:
            input_dict_std['monthly_salary'] = value
        elif key_lower in ['expenses', 'monthly_expense']:
            input_dict_std['monthly_expenses'] = value
        elif key_lower in ['current_emi', 'emi_amount']:
            input_dict_std['current_emi_amount'] = value
        elif key_lower in ['loan_amount']:
            input_dict_std['requested_amount'] = value
        elif key_lower in ['tenure', 'loan_tenure']:
            input_dict_std['requested_tenure'] = value
        else:
            input_dict_std[key] = value
    
    # Debug: Show what fields we have
    if st.session_state.get('debug_mode', False):
        with st.expander("🔍 Debug: Input Fields"):
            st.write("**Standardized Input Keys:**", list(input_dict_std.keys()))
    
    # Get expected feature names from model or file
    expected_features = None
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
    else:
        expected_features = load_feature_names(model_name)
    
    if expected_features is None:
        st.error("❌ Could not determine model's expected features!")
        return None
    
    # Create initial DataFrame
    df = pd.DataFrame([input_dict_std])
    
    # Feature engineering - calculate derived features with safe value retrieval
    monthly_salary = safe_get_value(input_dict_std, ['monthly_salary', 'income', 'salary'], 0)
    monthly_expenses = safe_get_value(input_dict_std, ['monthly_expenses', 'expenses'], 0)
    current_emi = safe_get_value(input_dict_std, ['current_emi_amount', 'current_emi'], 0)
    
    if monthly_salary > 0 or current_emi > 0 or monthly_expenses > 0:
        df['total_monthly_expenses'] = monthly_expenses + current_emi
    else:
        df['total_monthly_expenses'] = 0
    
    if monthly_salary > 0:
        total_exp = df.get('total_monthly_expenses', pd.Series([0])).iloc[0]
        df['savings_ratio'] = (monthly_salary - total_exp) / monthly_salary
        df['expense_to_income_ratio'] = total_exp / monthly_salary
    else:
        df['savings_ratio'] = 0
        df['expense_to_income_ratio'] = 0
    
    family_size = safe_get_value(input_dict_std, 'family_size', 1)
    if monthly_salary > 0 and family_size > 0:
        df['income_per_member'] = monthly_salary / family_size
    else:
        df['income_per_member'] = monthly_salary
    
    dependents = safe_get_value(input_dict_std, 'dependents', 0)
    if dependents >= 0:
        df['dependents_ratio'] = dependents / (dependents + 1)
    else:
        df['dependents_ratio'] = 0
    
    # Employment stability score
    employment_type = safe_get_value(input_dict_std, 'employment_type', 'Private')
    if employment_type == 'Government':
        df['employment_stability'] = 1.0
    elif employment_type == 'Private':
        df['employment_stability'] = 0.8
    else:  # Self-Employed
        df['employment_stability'] = 0.6
    
    requested_amount = safe_get_value(input_dict_std, ['requested_amount', 'loan_amount'], 0)
    if requested_amount > 0 and monthly_salary > 0:
        df['loan_to_income_ratio'] = requested_amount / (monthly_salary * 12)
    else:
        df['loan_to_income_ratio'] = 0
    
    emergency_fund = safe_get_value(input_dict_std, 'emergency_fund', 0)
    total_exp = df.get('total_monthly_expenses', pd.Series([0])).iloc[0]
    if emergency_fund > 0 and total_exp > 0:
        df['emergency_coverage'] = emergency_fund / total_exp
    else:
        df['emergency_coverage'] = 0
    
    # Handle categorical encoding - One-Hot Encoding
    categorical_mappings = {
        'gender': ['Male'],  # Female is reference (dropped)
        'marital_status': ['Single'],  # Married is reference
        'employment_type': ['Private', 'Self-Employed'],  # Government is reference
        'company_type': ['MNC', 'Mid-size', 'Small', 'Startup'],  # Large is reference
        'house_type': ['Own', 'Rented'],  # Mortgage is reference
        'existing_loans': ['Yes'],  # No is reference
        'emi_scenario': ['Education EMI', 'Home Appliances EMI', 'Personal Loan EMI', 'Vehicle EMI']  # Home Loan is reference
    }
    
    # Create one-hot encoded features
    for cat_col, categories in categorical_mappings.items():
        if cat_col in df.columns:
            value = df[cat_col].iloc[0]
            for cat in categories:
                col_name = f"{cat_col}_{cat}"
                if col_name in expected_features:
                    df[col_name] = 1 if value == cat else 0
    
    # Handle ordinal encoding for education
    if 'education' in df.columns:
        education_mapping = {
            'High School': 0,
            'Graduate': 1,
            'Post Graduate': 2,
            'Professional': 3
        }
        edu_value = df['education'].iloc[0]
        df['education'] = education_mapping.get(edu_value, 1)
    
    # Handle ordinal encoding for emi_eligibility
    if 'emi_eligibility' in df.columns:
        eligibility_mapping = {
            'Not_Eligible': 0,
            'Eligible': 1,
            'High_Risk': 2
        }
        elig_value = df['emi_eligibility'].iloc[0]
        # If it's a percentage, convert to category
        if isinstance(elig_value, (int, float)):
            if elig_value < 30:
                elig_value = 'Not_Eligible'
            elif elig_value > 50:
                elig_value = 'High_Risk'
            else:
                elig_value = 'Eligible'
        df['emi_eligibility'] = eligibility_mapping.get(elig_value, 1)
    
    # Align with expected features
    missing_features = []
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0  # Fill missing features with 0
            missing_features.append(feature)
    
    # Remove extra features not in expected list
    extra_features = [col for col in df.columns if col not in expected_features]
    if extra_features:
        df = df.drop(columns=extra_features)
    
    # Reorder columns to match expected order
    df = df[expected_features]
    
    # Show warnings about missing features
    if missing_features and len(missing_features) > 0:
        if len(missing_features) <= 5:
            st.warning(f"⚠️ {len(missing_features)} feature(s) were missing and filled with zeros: {', '.join(missing_features)}")
        else:
            st.warning(f"⚠️ {len(missing_features)} feature(s) were missing and filled with zeros (showing first 5): {', '.join(missing_features[:5])}...")
    
    return df

def render():
    st.title("📊 Regression Prediction")
    st.markdown("Predict EMI amount using trained regression models")
    
    # Debug mode toggle (in sidebar)
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    with st.sidebar:
        st.session_state.debug_mode = st.checkbox("🔍 Debug Mode", value=st.session_state.debug_mode, help="Show detailed debugging information")
    
    # Load models
    with st.spinner("Loading regression models..."):
        models_dict = load_regression_models()
    
    if not models_dict:
        st.warning("⚠️ No regression models found in `models/` directory.")
        st.info("💡 Tip: Train a regression model and save it with 'regress' or 'regression' in the filename.")
        return
    
    st.success(f"✅ Loaded {len(models_dict)} regression model(s)")
    
    # Model selection dropdown
    st.subheader("🤖 Select Model")
    selected_model_name = st.selectbox(
        "Choose a regression model:",
        list(models_dict.keys()),
        help="Select which model to use for EMI predictions"
    )
    selected_model = models_dict[selected_model_name]['model']
    
    # Display model info
    with st.expander("ℹ️ Model Information"):
        st.write(f"**Model Type:** {type(selected_model).__name__}")
        if hasattr(selected_model, 'n_features_in_'):
            st.write(f"**Expected Features:** {selected_model.n_features_in_}")
        if hasattr(selected_model, 'feature_names_in_'):
            st.write(f"**Feature Names ({len(selected_model.feature_names_in_)}):**")
            # Show first 10 features
            for i, name in enumerate(list(selected_model.feature_names_in_)[:10]):
                st.write(f"  {i+1}. {name}")
            if len(selected_model.feature_names_in_) > 10:
                st.write(f"  ... and {len(selected_model.feature_names_in_) - 10} more")
    
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
    actual_values = None
    
    if input_method == "Manual Input":
        st.markdown("### 📝 Enter Customer Details")
        
        with st.form("manual_input_form"):
            col1, col2, col3 = st.columns(3)
            
            input_dict = {}
            
            with col1:
                st.markdown("**👤 Personal Information**")
                input_dict['age'] = st.number_input("Age", min_value=18, max_value=100, value=35)
                input_dict['gender'] = st.selectbox("Gender", ["Male", "Female"], index=0)
                input_dict['marital_status'] = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=1)
                input_dict['dependents'] = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
                input_dict['family_size'] = st.number_input("Family Size", min_value=1, max_value=15, value=input_dict['dependents'] + 2)
                input_dict['education'] = st.selectbox("Education Level", ["High School", "Graduate", "Post Graduate", "Professional"], index=1)
            
            with col2:
                st.markdown("**💰 Financial Information**")
                input_dict['monthly_salary'] = st.number_input("Monthly Salary ($)", min_value=0, value=75000, step=5000)
                input_dict['monthly_expenses'] = st.number_input("Monthly Expenses ($)", min_value=0, value=30000, step=1000)
                input_dict['bank_balance'] = st.number_input("Bank Balance ($)", min_value=0, value=200000, step=10000)
                input_dict['emergency_fund'] = st.number_input("Emergency Fund ($)", min_value=0, value=100000, step=5000)
                input_dict['credit_score'] = st.slider("Credit Score", min_value=300, max_value=850, value=720)
                input_dict['current_emi_amount'] = st.number_input("Current EMI Amount ($)", min_value=0, value=5000, step=500)
            
            with col3:
                st.markdown("**� Employment & Loan Details**")
                input_dict['employment_type'] = st.selectbox("Employment Type", ["Government", "Private", "Self-Employed"], index=1)
                input_dict['company_type'] = st.selectbox("Company Type", ["Large", "MNC", "Mid-size", "Small", "Startup"], index=1)
                input_dict['years_of_employment'] = st.number_input("Years of Employment", min_value=0, max_value=50, value=10)
                input_dict['house_type'] = st.selectbox("House Type", ["Own", "Rented", "Mortgage"], index=0)
                input_dict['existing_loans'] = st.selectbox("Existing Loans", ["Yes", "No"], index=1)
                input_dict['emi_scenario'] = st.selectbox("EMI Scenario", ["Home Loan EMI", "Education EMI", "Home Appliances EMI", "Personal Loan EMI", "Vehicle EMI"], index=0)
            
            col4, col5 = st.columns(2)
            with col4:
                input_dict['requested_amount'] = st.number_input("Requested Loan Amount ($)", min_value=0, value=500000, step=50000)
            with col5:
                input_dict['requested_tenure'] = st.number_input("Loan Tenure (months)", min_value=12, max_value=360, value=120, step=12)
            
            # EMI Eligibility as category
            input_dict['emi_eligibility'] = st.selectbox("EMI Eligibility", ["Not_Eligible", "Eligible", "High_Risk"], index=1)
            
            submit_button = st.form_submit_button("🔮 Predict EMI Amount", use_container_width=True)
        
        if submit_button:
            # Debug: Show input dict
            if st.session_state.get('debug_mode', False):
                with st.expander("🔍 Debug: Raw Input Dictionary"):
                    st.write("**Keys in input_dict:**", list(input_dict.keys()))
                    st.json(input_dict)
            
            with st.spinner("Calculating EMI prediction..."):
                try:
                    # Preprocess input data
                    processed_data = preprocess_input(input_dict, selected_model, selected_model_name)
                    
                    if processed_data is not None:
                        # Make prediction
                        predictions = selected_model.predict(processed_data)
                        input_data = processed_data
                        
                        if st.session_state.get('debug_mode', False):
                            st.success(f"✓ Prediction successful: ${predictions[0]:,.2f}")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.exception(e)
                    
                    # Show debug info
                    with st.expander("🔍 Debug Information"):
                        st.write("**Input Dictionary:**")
                        st.json(input_dict)
                        if hasattr(selected_model, 'feature_names_in_'):
                            st.write("**Expected Features:**")
                            st.write(list(selected_model.feature_names_in_))
    
    else:  # Upload CSV
        st.markdown("### 📤 Upload CSV File")
        st.info("💡 CSV should contain feature columns. Optionally include 'actual_emi', 'target', or 'emi_amount' column for comparison.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with customer data for batch predictions"
        )
        
        if uploaded_file is not None:
            try:
                input_data_raw = pd.read_csv(uploaded_file)
                
                # Standardize column names
                input_data = standardize_column_names(input_data_raw)
                
                st.success(f"✅ Loaded {len(input_data)} records")
                
                # Show column mapping info
                renamed_cols = [col for col in input_data_raw.columns if col not in input_data.columns]
                if renamed_cols:
                    with st.expander("📋 Column Name Standardization"):
                        st.info(f"✓ Standardized {len(renamed_cols)} column name(s) to match model schema")
                        mapping_display = []
                        for old_col in renamed_cols:
                            # Find the new name
                            for new_col in input_data.columns:
                                if new_col not in input_data_raw.columns:
                                    mapping_display.append(f"  • '{old_col}' → '{new_col}'")
                                    break
                        if mapping_display:
                            st.text("\n".join(mapping_display[:10]))
                
                # Check for actual values column
                actual_col = None
                for col in ['actual_emi', 'target', 'emi_amount', 'true_emi', 'approved_emi_amount']:
                    if col in input_data.columns:
                        actual_col = col
                        actual_values = input_data[col].values
                        st.info(f"📊 Found actual values in column: '{col}' - Will show comparison plots")
                        break
                
                with st.expander("📊 View Uploaded Data (Standardized)"):
                    st.dataframe(input_data.head(10))
                
                if st.button("🔮 Run Batch Predictions", use_container_width=True):
                    with st.spinner("Making predictions..."):
                        try:
                            # Prepare data for all rows
                            all_predictions = []
                            failed_rows = []
                            
                            progress_bar = st.progress(0)
                            for idx, row in input_data.iterrows():
                                try:
                                    row_data = preprocess_input(row.to_dict(), selected_model, selected_model_name)
                                    if row_data is not None:
                                        pred = selected_model.predict(row_data)
                                        all_predictions.append(pred[0])
                                    else:
                                        all_predictions.append(np.nan)
                                        failed_rows.append(idx)
                                except Exception as e:
                                    all_predictions.append(np.nan)
                                    failed_rows.append(idx)
                                
                                # Update progress
                                progress_bar.progress((idx + 1) / len(input_data))
                            
                            predictions = np.array(all_predictions)
                            
                            # Remove NaN values for display
                            valid_predictions = predictions[~np.isnan(predictions)]
                            
                            if failed_rows:
                                st.warning(f"⚠️ Failed to predict for {len(failed_rows)} row(s). Check data completeness.")
                                with st.expander("View Failed Rows"):
                                    st.write(f"Row indices: {failed_rows[:20]}{'...' if len(failed_rows) > 20 else ''}")
                            
                            if len(valid_predictions) > 0:
                                st.success(f"✅ Successfully predicted {len(valid_predictions)} out of {len(input_data)} records")
                            else:
                                st.error("❌ No valid predictions could be made. Check your data format.")
                            
                        except Exception as e:
                            st.error(f"❌ Batch prediction failed: {str(e)}")
                            st.exception(e)
            
            except Exception as e:
                st.error(f"❌ Failed to read CSV file: {str(e)}")
                with st.expander("💡 Troubleshooting Tips"):
                    st.write("- Ensure the CSV file is properly formatted")
                    st.write("- Check that column names match expected features")
                    st.write("- Verify there are no special characters in column names")
                    st.write("- Make sure numeric columns contain valid numbers")
    
    # Display Results
    if predictions is not None:
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        # Single prediction result
        if len(predictions) == 1:
            predicted_emi = predictions[0]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 💵 Predicted EMI Amount")
                st.markdown(f"<h1 style='text-align: center; color: #1e3c72;'>${predicted_emi:,.2f}</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 1.2rem;'>per month</p>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📈 Loan Summary")
                if 'input_dict' in locals():
                    requested_tenure = safe_get_value(input_dict, ['requested_tenure', 'tenure', 'loan_tenure'], 120)
                    requested_amount = safe_get_value(input_dict, ['requested_amount', 'loan_amount'], 0)
                    
                    if requested_tenure > 0 and requested_amount > 0:
                        total_payable = predicted_emi * requested_tenure
                        interest_amount = total_payable - requested_amount
                        interest_rate_approx = (interest_amount / requested_amount) / (requested_tenure / 12) * 100
                        
                        st.metric("Total Amount Payable", f"${total_payable:,.2f}")
                        st.metric("Total Interest", f"${interest_amount:,.2f}")
                        st.metric("Approx. Interest Rate", f"{interest_rate_approx:.2f}% p.a.")
                    else:
                        st.warning("⚠️ Loan details incomplete. Cannot calculate summary.")
            
            # Affordability check
            if 'input_dict' in locals():
                st.markdown("### 💡 Affordability Analysis")
                
                # Safely get income value (could be 'income' or 'monthly_salary')
                income = safe_get_value(input_dict, ['monthly_salary', 'income', 'salary'], None)
                monthly_expenses = safe_get_value(input_dict, ['monthly_expenses', 'expenses'], 0)
                
                if income is None or income == 0:
                    st.warning("⚠️ Income value is missing or zero — EMI-to-Income ratio cannot be calculated.")
                    st.info("💡 Please provide a valid monthly income to see affordability analysis.")
                else:
                    emi_to_income_ratio = (predicted_emi / income) * 100
                    
                    col3, col4, col5 = st.columns(3)
                    with col3:
                        st.metric("EMI to Income Ratio", f"{emi_to_income_ratio:.1f}%")
                    with col4:
                        remaining_income = income - predicted_emi - monthly_expenses
                        st.metric("Remaining Monthly Income", f"${remaining_income:,.2f}")
                    with col5:
                        if emi_to_income_ratio <= 40:
                            st.success("✅ Affordable")
                        elif emi_to_income_ratio <= 50:
                            st.warning("⚠️ Moderate")
                        else:
                            st.error("❌ High Risk")
                    
                    # Recommendation
                    if emi_to_income_ratio > 50:
                        st.warning("⚠️ **Recommendation:** EMI exceeds 50% of monthly income. Consider reducing loan amount or extending tenure.")
                    elif remaining_income < 10000:
                        st.info("💡 **Tip:** Limited remaining income. Ensure you have adequate emergency funds.")
                    else:
                        st.success("✅ **Good to go!** The EMI appears affordable based on your income.")
        
        else:  # Batch predictions
            st.markdown(f"### 📦 Batch Results ({len(predictions)} records)")
            
            results_df = input_data.copy()
            results_df['Predicted_EMI'] = predictions
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average EMI", f"${np.mean(predictions):,.2f}")
            with col2:
                st.metric("Min EMI", f"${np.min(predictions):,.2f}")
            with col3:
                st.metric("Max EMI", f"${np.max(predictions):,.2f}")
            with col4:
                st.metric("Std Deviation", f"${np.std(predictions):,.2f}")
            
            # Display results table
            st.dataframe(results_df.head(20))
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="regression_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Interactive Plots
            st.markdown("---")
            st.header("📈 Visualization")
            
            # Distribution plot
            fig_dist = px.histogram(
                predictions,
                nbins=30,
                title='Distribution of Predicted EMI Values',
                labels={'value': 'Predicted EMI', 'count': 'Frequency'},
                color_discrete_sequence=['#667eea']
            )
            fig_dist.update_layout(showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # If actual values available, show comparison plots
            if actual_values is not None:
                st.markdown("### 🎯 Actual vs Predicted Comparison")
                
                tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Residual Plot", "Error Metrics"])
                
                with tab1:
                    # Scatter plot: Actual vs Predicted
                    fig_scatter = go.Figure()
                    
                    # Add scatter points
                    fig_scatter.add_trace(go.Scatter(
                        x=actual_values,
                        y=predictions,
                        mode='markers',
                        name='Predictions',
                        marker=dict(size=8, color='blue', opacity=0.6)
                    ))
                    
                    # Add perfect prediction line
                    min_val = min(min(actual_values), min(predictions))
                    max_val = max(max(actual_values), max(predictions))
                    fig_scatter.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_scatter.update_layout(
                        title='Actual vs Predicted EMI',
                        xaxis_title='Actual EMI',
                        yaxis_title='Predicted EMI',
                        hovermode='closest'
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with tab2:
                    # Residual plot
                    residuals = predictions - actual_values
                    
                    fig_residual = go.Figure()
                    fig_residual.add_trace(go.Scatter(
                        x=predictions,
                        y=residuals,
                        mode='markers',
                        marker=dict(size=8, color='green', opacity=0.6)
                    ))
                    
                    # Add zero line
                    fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                    
                    fig_residual.update_layout(
                        title='Residual Plot',
                        xaxis_title='Predicted EMI',
                        yaxis_title='Residuals (Predicted - Actual)',
                        hovermode='closest'
                    )
                    st.plotly_chart(fig_residual, use_container_width=True)
                
                with tab3:
                    # Calculate error metrics
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    
                    mae = mean_absolute_error(actual_values, predictions)
                    mse = mean_squared_error(actual_values, predictions)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(actual_values, predictions)
                    mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
                        st.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")
                    with col2:
                        st.metric("R² Score", f"{r2:.4f}")
                        st.metric("Mean Absolute % Error", f"{mape:.2f}%")
                    with col3:
                        accuracy = 100 - mape
                        st.metric("Model Accuracy", f"{accuracy:.2f}%")
                        
                        if r2 > 0.8:
                            st.success("✅ Excellent Fit")
                        elif r2 > 0.6:
                            st.info("✓ Good Fit")
                        else:
                            st.warning("⚠️ Moderate Fit")
    
    # Model comparison (if multiple models loaded)
    if len(models_dict) > 1 and input_method == "Manual Input":
        st.markdown("---")
        st.header("🔄 Compare All Models")
        st.markdown("See EMI predictions from different regression models")
        
        if st.button("🔍 Compare Models", use_container_width=True):
            if 'input_dict' in locals():
                comparison_results = []
                
                with st.spinner("Running predictions across all models..."):
                    for model_name, model_data in models_dict.items():
                        try:
                            model = model_data['model']
                            processed_data = preprocess_input(input_dict, model, model_name)
                            
                            if processed_data is not None:
                                pred = model.predict(processed_data)[0]
                                
                                comparison_results.append({
                                    'Model': model_name,
                                    'Predicted_EMI': f"${pred:,.2f}",
                                    'EMI_Numeric': pred
                                })
                        
                        except Exception as e:
                            st.warning(f"⚠️ {model_name} failed: {str(e)}")
                
                if comparison_results:
                    st.markdown("### 📊 Model Comparison Results")
                    comparison_df = pd.DataFrame(comparison_results)
                    st.dataframe(comparison_df[['Model', 'Predicted_EMI']], use_container_width=True)
                    
                    # Visualize comparison
                    fig = px.bar(
                        comparison_df,
                        x='Model',
                        y='EMI_Numeric',
                        title='EMI Predictions Across Different Models',
                        labels={'EMI_Numeric': 'Predicted EMI ($)'},
                        color='EMI_Numeric',
                        color_continuous_scale='Blues'
                    )
                    fig.update_traces(texttemplate='$%{y:,.2f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    avg_prediction = comparison_df['EMI_Numeric'].mean()
                    std_prediction = comparison_df['EMI_Numeric'].std()
                    st.info(f"📊 Average across models: ${avg_prediction:,.2f} ± ${std_prediction:,.2f}")
            else:
                st.warning("⚠️ Please submit the manual input form first!")
    
    st.markdown("---")
    st.info("💡 **Tip**: EMI predictions are based on multiple financial factors. Always verify with your financial advisor.")
