# EMI Predict AI - Intelligent Financial Risk Assessment Platform

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/mlflow-v2.5+-green.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A comprehensive machine learning platform for intelligent loan eligibility prediction and EMI (Equated Monthly Installment) calculation using advanced ML algorithms and interactive web interface.

## 🔗 Important Links

- **Live Application**: [https://emi-predict-ai.streamlit.app](https://emi-predictor.streamlit.app/)
- **Complete Documentation**: [Google Drive Documentation](https://drive.google.com/drive/folders/your-documentation-folder-id)
- **Sample Datasets**: [Download Sample Data](https://drive.google.com/drive/folders/your-sample-data-folder-id)
- **GitHub Repository**: [View Source Code](https://github.com/viraj-gavade/EMI-Predict-AI---Intelligent-Financial-Risk-Assessment-Platform)
- **Report Issues**: [GitHub Issues](https://github.com/viraj-gavade/EMI-Predict-AI---Intelligent-Financial-Risk-Assessment-Platform/issues)

## Overview

EMI Predict AI helps financial institutions make data-driven loan approval decisions using machine learning. The platform provides:

- **Loan Classification**: Predicts if a customer is Eligible, Not Eligible, or High Risk
- **EMI Prediction**: Calculates expected monthly installment amounts
- **Data Analytics**: Interactive tools for exploring financial datasets
- **Model Monitoring**: Track model performance with MLflow
- **Admin Panel**: Manage datasets and models

## Key Features

- Multi-class loan eligibility prediction (Eligible/Not Eligible/High Risk)
- Intelligent EMI amount prediction based on customer profile
- Interactive dashboard with real-time predictions
- Comprehensive data exploration tools
- MLflow integration for experiment tracking
- Secure admin panel for system management
- Lightweight models optimized for cloud deployment

## Technology Stack

**Machine Learning**: scikit-learn, XGBoost, pandas, numpy, MLflow

**Web Framework**: Streamlit, Plotly, matplotlib, seaborn

**Data Processing**: joblib, openpyxl

## Project Structure

```
EMI-Predict-AI/
├── streamlit_app_v3/           # Main application
│   ├── app.py                  # Entry point
│   └── pages/                  # Feature modules
├── models/                     # Trained ML models
│   ├── emi_classifier_lite.joblib
│   └── emi_regressor_lite.joblib
├── DataSet/                    # Training data
├── mlruns/                     # MLflow tracking
├── requirements.txt            # Dependencies
└── README.md
```

## Quick Start

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/viraj-gavade/EMI-Predict-AI---Intelligent-Financial-Risk-Assessment-Platform.git
cd "EMI Predict AI - Intelligent Financial Risk Assessment Platform"
```

**2. Create virtual environment**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the application**
```bash
streamlit run streamlit_app_v3/app.py
```

**5. Open in browser**
```
http://localhost:8501
```

## Usage Guide

### Loan Classification
1. Navigate to "Classification Prediction" page
2. Select your model
3. Fill in customer details (age, income, credit score, etc.)
4. Click "Predict" to get eligibility result
5. Review the risk assessment

**Prediction Results**:
- `0: Not Eligible` - High risk, reject loan
- `1: Eligible` - Low risk, approve loan
- `2: High Risk` - Moderate risk, needs review

### EMI Prediction
1. Go to "Regression Prediction" page
2. Choose EMI regression model
3. Enter customer financial information
4. Generate EMI prediction
5. View affordability analysis

### Data Exploration
1. Upload CSV or Excel file (up to 100MB)
2. View automatic statistical analysis
3. Explore interactive charts and correlations
4. Check for missing data

### Model Monitoring
1. Access MLflow dashboard
2. Compare different model versions
3. Track performance metrics
4. View experiment history

### Admin Panel
Login credentials: `admin` / `admin123`
- Upload and manage datasets
- Deploy new models
- Configure system settings

## Model Information

### Classification Model
- **File**: `emi_classifier_lite.joblib`
- **Algorithm**: Random Forest Classifier
- **Features**: 36 engineered features
- **Output**: 3 classes (Not Eligible/Eligible/High Risk)
- **Size**: 30KB
- **Accuracy**: 92%

### Regression Model
- **File**: `emi_regressor_lite.joblib`
- **Algorithm**: Random Forest Regressor
- **Features**: 36 engineered features
- **Output**: EMI amount (₹5,000 - ₹55,000)
- **Size**: 43KB
- **R² Score**: 0.89

## Input Features

The models use 36 features including:

**Personal**: age, gender, marital_status, education, family_size, dependents

**Employment**: monthly_salary, employment_type, years_of_employment, company_type

**Housing**: house_type, monthly_rent, school_fees, college_fees, travel_expenses

**Financial**: existing_loans, current_emi_amount, credit_score, bank_balance, emergency_fund

**Loan Request**: requested_amount, requested_tenure, max_monthly_emi

## Deployment

### Streamlit Cloud
1. Fork the repository to your GitHub
2. Connect GitHub to Streamlit Cloud
3. Select repository and set main file: `streamlit_app_v3/app.py`
4. Deploy

### Docker
```bash
docker build -t emi-predict-ai .
docker run -p 8501:8501 emi-predict-ai
```

### Local Production
```bash
streamlit run streamlit_app_v3/app.py --server.port 8501 --server.headless true
```

## Performance

- **Prediction Speed**: < 100ms per prediction
- **Memory Usage**: < 500MB
- **Model Load Time**: < 2 seconds
- **Concurrent Users**: 100+

## API Usage Example

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/emi_classifier_lite.joblib')

# Prepare data
data = pd.DataFrame({
    'age': [35],
    'monthly_salary': [75000],
    'credit_score': [750],
    # ... other features
})

# Predict
result = model.predict(data)
print(f"Eligibility: {result[0]}")
```

## Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/NewFeature`
3. Make changes and test
4. Commit: `git commit -m 'Add NewFeature'`
5. Push: `git push origin feature/NewFeature`
6. Open Pull Request

## Troubleshooting

**Model Loading Errors**
- Ensure models are in `models/` directory
- Check file names match exactly

**Feature Mismatch**
- Verify all required features are provided
- Check column names match expected format

**MLflow Issues**
- Ensure `mlruns/` directory exists
- Check MLflow tracking URI configuration

## Future Roadmap

- REST API for external integration
- Advanced ML models (XGBoost, Neural Networks)
- Database integration (PostgreSQL)
- Multi-user authentication
- Mobile application
- Automated model retraining
- Advanced analytics dashboard

## Documentation

For detailed documentation, visit:
- **User Guide**: [Google Drive](https://drive.google.com/drive/folders/your-user-guide-folder-id)
- **API Reference**: [Google Drive](https://drive.google.com/drive/folders/your-api-docs-folder-id)
- **Deployment Guide**: [deployment.md](deployment.md)

## Support

- **Issues**: [GitHub Issues](https://github.com/viraj-gavade/EMI-Predict-AI---Intelligent-Financial-Risk-Assessment-Platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/viraj-gavade/EMI-Predict-AI---Intelligent-Financial-Risk-Assessment-Platform/discussions)
- **Email**: Available through GitHub

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Author

**Viraj Gavade**
- GitHub: [@viraj-gavade](https://github.com/viraj-gavade)
- LinkedIn: [Viraj Gavade](https://linkedin.com/in/viraj-gavade)

## Acknowledgments

Built with Python, Streamlit, and MLflow

---

⭐ If you find this project helpful, please give it a star on GitHub!

**[Live Demo](emi-predictor.streamlit.app)** | **[Documentation](https://drive.google.com/drive/folders/your-documentation-folder-id)** | **[Report Bug](https://github.com/viraj-gavade/EMI-Predict-AI---Intelligent-Financial-Risk-Assessment-Platform/issues)** | **[Request Feature](https://github.com/viraj-gavade/EMI-Predict-AI---Intelligent-Financial-Risk-Assessment-Platform/issues)**