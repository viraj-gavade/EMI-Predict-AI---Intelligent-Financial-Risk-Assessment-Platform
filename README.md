# 💰 EMI Predict AI - Intelligent Financial Risk Assessment Platform

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![MLflow](https://img.shields.io/badge/mlflow-v2.5+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

> **A comprehensive machine learning platform for intelligent loan eligibility prediction and EMI (Equated Monthly Installment) calculation using advanced ML algorithms and interactive web interface.**

## 🎯 Project Overview

EMI Predict AI is a sophisticated financial risk assessment platform that leverages machine learning to help financial institutions make data-driven loan approval decisions. The platform combines classification models for loan eligibility prediction with regression models for EMI amount estimation, all wrapped in an intuitive Streamlit web interface.

### 🚀 Key Features

- **🔵 Loan Classification**: Multi-class prediction (Eligible/Not Eligible/High Risk)
- **📊 EMI Regression**: Intelligent EMI amount prediction based on financial profile
- **📈 Interactive Dashboard**: Real-time predictions with user-friendly interface
- **🔍 Data Exploration**: Comprehensive EDA tools with interactive visualizations
- **🎛️ MLflow Integration**: Complete experiment tracking and model versioning
- **⚙️ Admin Panel**: Dataset management and system configuration
- **📱 Responsive Design**: Modern UI with gradient themes and mobile-friendly layout

## 🏗️ Architecture

```
EMI Predict AI/
├── streamlit_app_v3/          # Main Streamlit Application
│   ├── app.py                 # Entry point and navigation
│   └── pages/                 # Modular page components
│       ├── home.py           # Dashboard and navigation
│       ├── classification_prediction.py
│       ├── regression_prediction.py
│       ├── data_exploration.py
│       ├── model_monitoring.py
│       └── admin_panel.py
├── models/                    # Trained ML Models
│   ├── emi_classifier_lite.joblib    # Lightweight classification model
│   ├── emi_regressor_lite.joblib     # Lightweight regression model
│   └── regression_features.json     # Feature metadata
├── DataSet/                   # Training Data
│   └── emi_prediction_dataset.csv   # Main dataset (50MB+)
├── mlruns/                    # MLflow Experiment Tracking
├── data/                      # Processed data files
├── ml/                        # Model development scripts
├── docs/                      # Documentation
├── .streamlit/               # Streamlit configuration
│   └── config.toml
├── requirements.txt          # Python dependencies
└── deployment.md            # Deployment guide
```

## 🔬 Technology Stack

### Machine Learning & Data Science
- **scikit-learn** - Core ML algorithms (RandomForest, XGBoost)
- **pandas & numpy** - Data manipulation and numerical computations
- **MLflow** - Experiment tracking and model registry

### Web Framework & Visualization
- **Streamlit** - Interactive web application framework
- **Plotly** - Interactive charts and visualizations
- **matplotlib & seaborn** - Statistical plotting

### Data Processing
- **joblib** - Model serialization and parallel processing
- **openpyxl** - Excel file handling

## 📊 Data Schema

The platform works with comprehensive financial datasets containing:

### Core Features (28 columns)
```python
# Personal Information
'age', 'gender', 'marital_status', 'education', 'family_size', 'dependents'

# Employment & Income
'monthly_salary', 'employment_type', 'years_of_employment', 'company_type'

# Housing & Expenses
'house_type', 'monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 
'groceries_utilities', 'other_monthly_expenses'

# Financial Profile
'existing_loans', 'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund'

# Loan Request
'emi_scenario', 'requested_amount', 'requested_tenure', 'max_monthly_emi'

# Target Variables
'emi_eligibility'  # Classification target
# EMI amount (regression target - calculated)
```

### Engineered Features (36 total)
The platform automatically generates additional features:
- **One-hot encoded categoricals**: `gender_Male`, `marital_status_Single`, etc.
- **Calculated ratios**: `savings_ratio`, `expense_to_income_ratio`, `loan_to_income_ratio`
- **Derived metrics**: `employment_stability`, `emergency_coverage`, `dependents_ratio`

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- 4GB+ RAM (for full dataset processing)

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/viraj-gavade/EMI-Predict-AI---Intelligent-Financial-Risk-Assessment-Platform.git
cd "EMI Predict AI - Intelligent Financial Risk Assessment Platform"
```

2. **Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Application**
```bash
streamlit run streamlit_app_v3/app.py
```

5. **Access the Platform**
Open your browser and navigate to: `http://localhost:8501`

## 💡 Usage Guide

### 🏠 Home Dashboard
- **System Overview**: View model status, dataset availability, and system health
- **Quick Navigation**: Direct links to all platform features
- **Model Metrics**: Real-time statistics about loaded models

### 🔵 Classification Prediction
1. **Select Model**: Choose from available classification models
2. **Input Customer Data**: Fill out the comprehensive form with customer details
3. **Get Prediction**: Receive loan eligibility classification with confidence scores
4. **Risk Assessment**: View detailed risk analysis and recommendations

**Prediction Classes:**
- `0: Not Eligible` - High risk, loan should be rejected
- `1: Eligible` - Low risk, loan can be approved  
- `2: High Risk` - Moderate risk, requires additional review

### 📊 Regression Prediction
1. **Model Selection**: Choose EMI regression model
2. **Customer Profile**: Input detailed financial information
3. **EMI Calculation**: Get predicted EMI amount with confidence intervals
4. **Financial Analysis**: View affordability metrics and recommendations

### 📈 Data Exploration
- **Dataset Upload**: Support for CSV and Excel files (up to 100MB)
- **Automatic EDA**: Statistical summaries, distribution plots, correlation analysis
- **Interactive Visualizations**: Plotly-based charts with zoom, filter, and export features
- **Missing Data Analysis**: Identify and visualize data quality issues

### 🔍 Model Monitoring (MLflow Integration)
- **Experiment Tracking**: View all ML experiments and runs
- **Model Comparison**: Compare metrics across different model versions
- **Performance Metrics**: Accuracy, precision, recall, F1-score tracking
- **Model Registry**: Centralized model versioning and metadata

### ⚙️ Admin Panel
**Authentication Required** (`admin` / `admin123`)
- **Dataset Management**: Upload, view, and manage training datasets
- **Model Upload**: Deploy new trained models
- **System Configuration**: Modify platform settings
- **File Management**: Organize data and model files

## 🤖 Model Information

### Classification Model (`emi_classifier_lite.joblib`)
- **Algorithm**: Random Forest Classifier
- **Features**: 36 engineered features
- **Classes**: 3-class prediction (Not Eligible/Eligible/High Risk)
- **Size**: 30KB (optimized for deployment)
- **Performance**: Trained on comprehensive financial dataset

### Regression Model (`emi_regressor_lite.joblib`)
- **Algorithm**: Random Forest Regressor
- **Features**: 36 engineered features
- **Target**: EMI amount prediction (₹5,000 - ₹55,000 range)
- **Size**: 43KB (optimized for deployment)
- **Metrics**: MAE, RMSE, R² tracking via MLflow

## 🛠️ Development

### Project Structure
```python
# Main application entry point
streamlit_app_v3/app.py

# Page modules (modular architecture)
pages/
├── home.py                    # Dashboard
├── classification_prediction.py  # Loan eligibility
├── regression_prediction.py     # EMI calculation  
├── data_exploration.py          # EDA tools
├── model_monitoring.py          # MLflow integration
└── admin_panel.py              # System management
```

### Key Functions

**Model Loading & Prediction**
```python
def load_classification_models():
    """Dynamically load all classification models from models/ directory"""

def prepare_input_data(input_dict, model=None):
    """Prepare and align input data with model requirements"""

def make_prediction(model, input_data):
    """Generate predictions with confidence scores"""
```

**Data Processing**
```python
def standardize_column_names(df):
    """Handle various column naming conventions"""

def engineer_features(df):
    """Create derived features for improved model performance"""
```

### Adding New Models
1. Save trained model to `models/` directory
2. Use naming convention: `*classif*.joblib` or `*regress*.joblib`
3. Models are automatically detected and loaded
4. Ensure feature compatibility with existing schema

## 🌐 Deployment

### Streamlit Cloud (Recommended)
1. **Repository Setup**: Code is optimized for Streamlit Cloud
2. **Model Size**: Lightweight models (< 100MB) for GitHub compatibility
3. **Dependencies**: All requirements specified in `requirements.txt`
4. **Configuration**: `.streamlit/config.toml` included

**Deployment Steps:**
1. Fork/clone repository to your GitHub
2. Connect GitHub to Streamlit Cloud
3. Select repository and set main file: `streamlit_app_v3/app.py`
4. Deploy automatically!

### Local Production
```bash
# Install production dependencies
pip install -r requirements.txt

# Run with production settings
streamlit run streamlit_app_v3/app.py --server.port 8501 --server.headless true
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app_v3/app.py"]
```

## 📈 Performance & Scalability

### Model Performance
- **Classification Accuracy**: Optimized for financial risk assessment
- **Regression Performance**: Low MAE on EMI predictions
- **Prediction Speed**: < 100ms per prediction
- **Memory Usage**: < 500MB total footprint

### Scalability Features
- **Lightweight Models**: Deployment-optimized (30-43KB)
- **Efficient Processing**: Vectorized operations with pandas/numpy
- **Caching**: Streamlit built-in caching for model loading
- **Modular Architecture**: Easy to scale individual components

## 🔒 Security & Authentication

### Admin Panel Security
- Username/password authentication
- Session management
- File upload validation
- Path traversal protection

### Data Security
- No sensitive data stored in repository
- Model files use joblib serialization
- Input validation and sanitization
- Error handling without data exposure

## 🛠️ Troubleshooting

### Common Issues

**1. Model Loading Errors**
```python
# Ensure models are in correct directory
models/
├── emi_classifier_lite.joblib
└── emi_regressor_lite.joblib
```

**2. Feature Mismatch**
- Check `regression_features.json` for expected features
- Verify input data column names match expected schema

**3. MLflow Connection Issues**
```bash
# Check MLflow tracking directory
ls mlruns/
# Should contain experiment directories
```

**4. Large File Issues**
- Use lightweight models for deployment
- Dataset files > 50MB may need external hosting

### Performance Optimization
```python
# Enable Streamlit caching
@st.cache_data
def load_data():
    return pd.read_csv('dataset.csv')

@st.cache_resource  
def load_model():
    return joblib.load('model.joblib')
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Install development dependencies
4. Make your changes
5. Add tests if applicable
6. Commit changes (`git commit -m 'Add AmazingFeature'`)
7. Push to branch (`git push origin feature/AmazingFeature`)
8. Open Pull Request

### Code Style
- Follow PEP 8 Python style guide
- Use meaningful variable names
- Add docstrings for functions
- Comment complex logic
- Keep functions focused and small

## 📝 Changelog

### v3.0 (Current)
- ✅ Complete Streamlit web interface
- ✅ Multi-model support (classification + regression)
- ✅ MLflow integration for experiment tracking
- ✅ Interactive data exploration tools
- ✅ Admin panel with authentication
- ✅ Lightweight models for cloud deployment
- ✅ Responsive UI with modern design

### v2.0 (Previous)
- Basic Streamlit interface
- Single model prediction
- Manual model loading

### v1.0 (Initial)
- Command-line interface
- Basic ML model training
- CSV data processing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team & Contributors

**Lead Developer**: [Viraj Gavade](https://github.com/viraj-gavade)
- Full-stack development
- ML model architecture
- Streamlit interface design
- Deployment optimization

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/viraj-gavade/EMI-Predict-AI---Intelligent-Financial-Risk-Assessment-Platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/viraj-gavade/EMI-Predict-AI---Intelligent-Financial-Risk-Assessment-Platform/discussions)
- **Email**: Support available through GitHub

## 🎯 Future Roadmap

### Planned Features
- [ ] **Advanced Models**: XGBoost, Neural Networks, Ensemble methods
- [ ] **Real-time API**: REST API for external integration
- [ ] **Database Integration**: PostgreSQL/MySQL support
- [ ] **User Management**: Multi-user authentication system
- [ ] **Advanced Analytics**: Business intelligence dashboard
- [ ] **Mobile App**: React Native companion app
- [ ] **A/B Testing**: Model comparison framework
- [ ] **Automated Retraining**: Continuous learning pipeline

### Technical Improvements
- [ ] **Performance**: GPU acceleration for large datasets
- [ ] **Monitoring**: Application performance monitoring (APM)
- [ ] **Testing**: Comprehensive unit and integration tests
- [ ] **CI/CD**: Automated testing and deployment pipeline
- [ ] **Documentation**: Interactive API documentation

---

## ⭐ Star History

If you found this project helpful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=viraj-gavade/EMI-Predict-AI---Intelligent-Financial-Risk-Assessment-Platform&type=Date)](https://star-history.com/#viraj-gavade/EMI-Predict-AI---Intelligent-Financial-Risk-Assessment-Platform&Date)

---

<div align="center">

**Built with ❤️ using Python, Streamlit & MLflow**

[🚀 Live Demo](https://your-streamlit-app.streamlit.app) | [📖 Documentation](./docs/) | [🐛 Report Bug](https://github.com/viraj-gavade/EMI-Predict-AI---Intelligent-Financial-Risk-Assessment-Platform/issues) | [💡 Request Feature](https://github.com/viraj-gavade/EMI-Predict-AI---Intelligent-Financial-Risk-Assessment-Platform/issues)

</div>