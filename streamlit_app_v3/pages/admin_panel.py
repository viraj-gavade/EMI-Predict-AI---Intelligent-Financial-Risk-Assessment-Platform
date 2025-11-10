import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path
import shutil
import pickle
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Define paths
BASE_DIR = Path(os.getcwd())
DATA_DIR = BASE_DIR / "data"
CLASSIFICATION_MODEL_DIR = BASE_DIR / "ml" / "classification"
REGRESSION_MODEL_DIR = BASE_DIR / "ml" / "regression"
DATASET_DIR = BASE_DIR / "DataSet"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CLASSIFICATION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
REGRESSION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)

# File size limits (in MB)
MAX_DATASET_SIZE_MB = 100
MAX_MODEL_SIZE_MB = 500

# ==================== HELPER FUNCTIONS ====================

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def login(username, password):
    """Authenticate user"""
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        st.session_state['authenticated'] = True
        st.session_state['username'] = username
        return True
    return False

def logout():
    """Logout user"""
    st.session_state['authenticated'] = False
    if 'username' in st.session_state:
        del st.session_state['username']

def get_file_size(file_path):
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    except:
        return "N/A"

def get_file_modified_time(file_path):
    """Get file last modified time"""
    try:
        timestamp = os.path.getmtime(file_path)
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return "N/A"

def get_total_storage(directory):
    """Calculate total storage used in directory"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # Convert to MB
    except:
        return 0

def list_datasets(directory):
    """List all datasets in directory"""
    datasets = []
    try:
        for file in os.listdir(directory):
            if file.endswith(('.csv', '.xlsx', '.xls')):
                file_path = os.path.join(directory, file)
                datasets.append({
                    'Name': file,
                    'Size': get_file_size(file_path),
                    'Uploaded On': get_file_modified_time(file_path),
                    'Path': file_path
                })
    except Exception as e:
        st.error(f"Error listing datasets: {str(e)}")
    
    return pd.DataFrame(datasets) if datasets else pd.DataFrame()

def list_models(directory, model_type):
    """List all models in directory"""
    models = []
    try:
        for file in os.listdir(directory):
            if file.endswith(('.pkl', '.pickle', '.joblib', '.h5', '.pt', '.pth')):
                file_path = os.path.join(directory, file)
                
                # Try to load metadata if exists
                metadata_path = os.path.join(directory, f"{file}.json")
                metadata = {}
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                models.append({
                    'Name': file,
                    'Type': model_type,
                    'Size': get_file_size(file_path),
                    'Uploaded On': get_file_modified_time(file_path),
                    'Path': file_path,
                    'Metadata': metadata
                })
    except Exception as e:
        st.error(f"Error listing models: {str(e)}")
    
    return pd.DataFrame(models) if models else pd.DataFrame()

def save_uploaded_file(uploaded_file, directory):
    """Save uploaded file to directory"""
    try:
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True, file_path
    except Exception as e:
        return False, str(e)

def delete_file(file_path):
    """Delete a file"""
    try:
        os.remove(file_path)
        # Also remove metadata if exists
        metadata_path = f"{file_path}.json"
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        return True, "File deleted successfully"
    except Exception as e:
        return False, str(e)

def validate_file_upload(uploaded_file, max_size_mb, allowed_extensions):
    """Validate uploaded file"""
    # Check file extension
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    if file_ext not in allowed_extensions:
        return False, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
    
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File too large. Max size: {max_size_mb}MB, Your file: {file_size_mb:.2f}MB"
    
    return True, "Valid"

def rename_file(old_path, new_name):
    """Rename a file"""
    try:
        directory = os.path.dirname(old_path)
        new_path = os.path.join(directory, new_name)
        
        if os.path.exists(new_path):
            return False, "File with this name already exists"
        
        os.rename(old_path, new_path)
        
        # Rename metadata file if exists
        old_metadata = f"{old_path}.json"
        new_metadata = f"{new_path}.json"
        if os.path.exists(old_metadata):
            os.rename(old_metadata, new_metadata)
        
        return True, "File renamed successfully"
    except Exception as e:
        return False, str(e)

def create_model_metadata(model_name, model_type, accuracy=None, rmse=None, notes=""):
    """Create metadata JSON for a model"""
    metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "upload_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "uploaded_by": st.session_state.get('username', 'admin'),
        "notes": notes
    }
    
    if accuracy is not None:
        metadata["accuracy"] = accuracy
    if rmse is not None:
        metadata["rmse"] = rmse
    
    return metadata

# ==================== LOGIN UI ====================

def render_login():
    """Render login page"""
    st.title("🔐 Admin Login")
    st.markdown("Please authenticate to access the admin dashboard.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit = st.form_submit_button("🔓 Login", use_container_width=True)
            
            if submit:
                if login(username, password):
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Try again.")
        
        st.markdown("---")
        st.info("💡 **Default Credentials:**\n- Username: `admin`\n- Password: `admin123`")

# ==================== ADMIN DASHBOARD ====================

def render_admin_dashboard():
    """Render main admin dashboard"""
    
    # Header with logout
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🛠️ Admin Dashboard — Data & Model Management")
        st.markdown("Upload, view, and manage your datasets and trained models here.")
    
    with col2:
        st.markdown("")  # Spacing
        st.markdown("")  # Spacing
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()
    
    # Display current user
    st.caption(f"👤 Logged in as: **{st.session_state.get('username', 'admin')}**")
    st.markdown("---")
    
    # Storage overview
    st.subheader("💾 Storage Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        data_storage = get_total_storage(DATA_DIR) + get_total_storage(DATASET_DIR)
        st.metric("Dataset Storage", f"{data_storage:.2f} MB")
    
    with col2:
        classification_storage = get_total_storage(CLASSIFICATION_MODEL_DIR)
        st.metric("Classification Models", f"{classification_storage:.2f} MB")
    
    with col3:
        regression_storage = get_total_storage(REGRESSION_MODEL_DIR)
        st.metric("Regression Models", f"{regression_storage:.2f} MB")
    
    with col4:
        total_storage = data_storage + classification_storage + regression_storage
        st.metric("Total Storage", f"{total_storage:.2f} MB")
    
    st.markdown("---")
    
    # Create tabs for Datasets and Models
    tab1, tab2 = st.tabs(["📂 Dataset Management", "🤖 Model Management"])
    
    # ==================== DATASET MANAGEMENT TAB ====================
    with tab1:
        st.header("Dataset Management")
        
        # Upload new dataset
        st.subheader("📤 Upload New Dataset")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_dataset = st.file_uploader(
                "Choose a CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                help=f"Max file size: {MAX_DATASET_SIZE_MB}MB",
                key="dataset_uploader"
            )
        
        with col2:
            save_location = st.selectbox(
                "Save Location",
                ["data/", "DataSet/"],
                help="Choose where to save the dataset"
            )
        
        if uploaded_dataset is not None:
            # Validate file
            is_valid, message = validate_file_upload(
                uploaded_dataset,
                MAX_DATASET_SIZE_MB,
                ['.csv', '.xlsx', '.xls']
            )
            
            if is_valid:
                st.success(f"✅ File validated: {uploaded_dataset.name} ({uploaded_dataset.size / (1024*1024):.2f} MB)")
                
                # Preview data
                with st.expander("👁️ Preview Data"):
                    try:
                        if uploaded_dataset.name.endswith('.csv'):
                            df_preview = pd.read_csv(uploaded_dataset)
                        else:
                            df_preview = pd.read_excel(uploaded_dataset)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", len(df_preview))
                        with col2:
                            st.metric("Columns", len(df_preview.columns))
                        with col3:
                            st.metric("Memory", f"{df_preview.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
                        
                        st.dataframe(df_preview.head(10), use_container_width=True)
                        
                        # Reset file pointer
                        uploaded_dataset.seek(0)
                    except Exception as e:
                        st.error(f"Error previewing file: {str(e)}")
                
                # Save button
                if st.button("💾 Save Dataset", type="primary", use_container_width=True):
                    target_dir = DATA_DIR if save_location == "data/" else DATASET_DIR
                    success, result = save_uploaded_file(uploaded_dataset, target_dir)
                    
                    if success:
                        st.success(f"✅ Dataset saved successfully to {save_location}")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ Error saving dataset: {result}")
            else:
                st.error(f"❌ {message}")
        
        st.markdown("---")
        
        # List existing datasets
        st.subheader("📊 Existing Datasets")
        
        # Combine datasets from both directories
        datasets_data = list_datasets(DATA_DIR)
        datasets_dataset = list_datasets(DATASET_DIR)
        
        if not datasets_data.empty:
            datasets_data['Location'] = 'data/'
        if not datasets_dataset.empty:
            datasets_dataset['Location'] = 'DataSet/'
        
        all_datasets = pd.concat([datasets_data, datasets_dataset], ignore_index=True) if not datasets_data.empty or not datasets_dataset.empty else pd.DataFrame()
        
        if not all_datasets.empty:
            st.info(f"📁 Found {len(all_datasets)} dataset(s)")
            
            # Display datasets with actions
            for idx, row in all_datasets.iterrows():
                with st.expander(f"📄 {row['Name']} — {row['Size']} — {row['Location']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"**Size:** {row['Size']}")
                        st.markdown(f"**Uploaded:** {row['Uploaded On']}")
                        st.markdown(f"**Location:** {row['Location']}")
                    
                    with col2:
                        # Preview button
                        if st.button(f"👁️ Preview", key=f"preview_dataset_{idx}"):
                            try:
                                if row['Name'].endswith('.csv'):
                                    df = pd.read_csv(row['Path'])
                                else:
                                    df = pd.read_excel(row['Path'])
                                
                                st.dataframe(df.head(20), use_container_width=True)
                                
                                # Show basic stats
                                st.markdown("**Dataset Info:**")
                                st.write(f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                                st.write(f"- Memory: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
                            except Exception as e:
                                st.error(f"Error loading dataset: {str(e)}")
                        
                        # Download button
                        try:
                            with open(row['Path'], 'rb') as f:
                                st.download_button(
                                    label="📥 Download",
                                    data=f,
                                    file_name=row['Name'],
                                    mime='application/octet-stream',
                                    key=f"download_dataset_{idx}"
                                )
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                    
                    with col3:
                        # Rename functionality
                        with st.form(f"rename_dataset_{idx}"):
                            new_name = st.text_input("New name", value=row['Name'], key=f"rename_input_{idx}")
                            if st.form_submit_button("✏️ Rename"):
                                success, message = rename_file(row['Path'], new_name)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                        
                        # Delete button with confirmation
                        if st.button(f"🗑️ Delete", key=f"delete_dataset_{idx}", type="secondary"):
                            st.session_state[f'confirm_delete_dataset_{idx}'] = True
                        
                        if st.session_state.get(f'confirm_delete_dataset_{idx}', False):
                            st.warning("⚠️ Are you sure?")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("✅ Yes", key=f"confirm_yes_dataset_{idx}"):
                                    success, message = delete_file(row['Path'])
                                    if success:
                                        st.success("✅ Deleted!")
                                        st.session_state[f'confirm_delete_dataset_{idx}'] = False
                                        st.rerun()
                                    else:
                                        st.error(f"Error: {message}")
                            with col_b:
                                if st.button("❌ No", key=f"confirm_no_dataset_{idx}"):
                                    st.session_state[f'confirm_delete_dataset_{idx}'] = False
                                    st.rerun()
        else:
            st.info("📭 No datasets found. Upload your first dataset above!")
    
    # ==================== MODEL MANAGEMENT TAB ====================
    with tab2:
        st.header("Model Management")
        
        # Upload new model
        st.subheader("📤 Upload New Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_model = st.file_uploader(
                "Choose a model file",
                type=['pkl', 'pickle', 'joblib', 'h5', 'pt', 'pth'],
                help=f"Max file size: {MAX_MODEL_SIZE_MB}MB",
                key="model_uploader"
            )
        
        with col2:
            model_type = st.selectbox(
                "Model Type",
                ["Classification", "Regression"],
                help="Select the type of model"
            )
        
        if uploaded_model is not None:
            # Validate file
            is_valid, message = validate_file_upload(
                uploaded_model,
                MAX_MODEL_SIZE_MB,
                ['.pkl', '.pickle', '.joblib', '.h5', '.pt', '.pth']
            )
            
            if is_valid:
                st.success(f"✅ File validated: {uploaded_model.name} ({uploaded_model.size / (1024*1024):.2f} MB)")
                
                # Metadata form
                with st.expander("📝 Add Model Metadata (Optional)"):
                    accuracy = st.number_input("Accuracy", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
                    rmse = st.number_input("RMSE", min_value=0.0, value=0.0, step=0.01)
                    notes = st.text_area("Notes", placeholder="Add any notes about this model...")
                
                # Save button
                if st.button("💾 Save Model", type="primary", use_container_width=True):
                    target_dir = CLASSIFICATION_MODEL_DIR if model_type == "Classification" else REGRESSION_MODEL_DIR
                    success, result = save_uploaded_file(uploaded_model, target_dir)
                    
                    if success:
                        # Save metadata
                        metadata = create_model_metadata(
                            uploaded_model.name,
                            model_type,
                            accuracy if accuracy > 0 else None,
                            rmse if rmse > 0 else None,
                            notes
                        )
                        
                        metadata_path = f"{result}.json"
                        try:
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=4)
                        except:
                            pass
                        
                        st.success(f"✅ Model saved successfully to ml/{model_type.lower()}/")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ Error saving model: {result}")
            else:
                st.error(f"❌ {message}")
        
        st.markdown("---")
        
        # List existing models
        st.subheader("🤖 Existing Models")
        
        # Get models from both directories
        classification_models = list_models(CLASSIFICATION_MODEL_DIR, "Classification")
        regression_models = list_models(REGRESSION_MODEL_DIR, "Regression")
        
        all_models = pd.concat([classification_models, regression_models], ignore_index=True) if not classification_models.empty or not regression_models.empty else pd.DataFrame()
        
        if not all_models.empty:
            st.info(f"🤖 Found {len(all_models)} model(s)")
            
            # Filter by type
            filter_type = st.radio("Filter by type:", ["All", "Classification", "Regression"], horizontal=True)
            
            if filter_type != "All":
                all_models = all_models[all_models['Type'] == filter_type]
            
            # Display models with actions
            for idx, row in all_models.iterrows():
                with st.expander(f"🤖 {row['Name']} — {row['Type']} — {row['Size']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"**Type:** {row['Type']}")
                        st.markdown(f"**Size:** {row['Size']}")
                        st.markdown(f"**Uploaded:** {row['Uploaded On']}")
                        
                        # Show metadata if available
                        if row['Metadata']:
                            st.markdown("---")
                            st.markdown("**📊 Metadata:**")
                            metadata = row['Metadata']
                            if 'accuracy' in metadata:
                                st.metric("Accuracy", f"{metadata['accuracy']:.4f}")
                            if 'rmse' in metadata:
                                st.metric("RMSE", f"{metadata['rmse']:.4f}")
                            if 'uploaded_by' in metadata:
                                st.markdown(f"*Uploaded by: {metadata['uploaded_by']}*")
                    
                    with col2:
                        # View full metadata
                        if row['Metadata']:
                            if st.button(f"📋 View Full Metadata", key=f"view_metadata_{idx}"):
                                st.json(row['Metadata'])
                        
                        # Download button
                        try:
                            with open(row['Path'], 'rb') as f:
                                st.download_button(
                                    label="📥 Download Model",
                                    data=f,
                                    file_name=row['Name'],
                                    mime='application/octet-stream',
                                    key=f"download_model_{idx}"
                                )
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                        
                        # Model info
                        if st.button(f"ℹ️ View Info", key=f"info_model_{idx}"):
                            st.markdown("**File Details:**")
                            st.write(f"- Name: {row['Name']}")
                            st.write(f"- Type: {row['Type']}")
                            st.write(f"- Size: {row['Size']}")
                            st.write(f"- Path: `{row['Path']}`")
                            st.write(f"- Modified: {row['Uploaded On']}")
                    
                    with col3:
                        # Rename functionality
                        with st.form(f"rename_model_{idx}"):
                            new_name = st.text_input("New name", value=row['Name'], key=f"rename_model_input_{idx}")
                            if st.form_submit_button("✏️ Rename"):
                                success, message = rename_file(row['Path'], new_name)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                        
                        # Delete button with confirmation
                        if st.button(f"🗑️ Delete", key=f"delete_model_{idx}", type="secondary"):
                            st.session_state[f'confirm_delete_model_{idx}'] = True
                        
                        if st.session_state.get(f'confirm_delete_model_{idx}', False):
                            st.warning("⚠️ Are you sure?")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("✅ Yes", key=f"confirm_yes_model_{idx}"):
                                    success, message = delete_file(row['Path'])
                                    if success:
                                        st.success("✅ Deleted!")
                                        st.session_state[f'confirm_delete_model_{idx}'] = False
                                        st.rerun()
                                    else:
                                        st.error(f"Error: {message}")
                            with col_b:
                                if st.button("❌ No", key=f"confirm_no_model_{idx}"):
                                    st.session_state[f'confirm_delete_model_{idx}'] = False
                                    st.rerun()
        else:
            st.info("📭 No models found. Upload your first model above!")
    
    # Footer
    st.markdown("---")
    st.info("💡 **Tip**: Always backup your models before deleting them. Use the download button to save copies locally.")

# ==================== MAIN RENDER FUNCTION ====================

def render():
    """Main render function"""
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    # Check authentication
    if not check_authentication():
        render_login()
    else:
        render_admin_dashboard()

# Run the page
if __name__ == "__main__":
    render()
