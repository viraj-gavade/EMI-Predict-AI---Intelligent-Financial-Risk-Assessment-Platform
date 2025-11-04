# EMI Predict AI — Streamlit App

This Streamlit multi-page app provides:
- Real-time prediction (manual & batch) using MLflow models or uploaded pickles
- Interactive data exploration and visualization
- Lightweight MLflow monitoring by scanning local `mlruns/`
- Administrative dataset operations (view, backup, append, export)

How to run (Windows PowerShell):

1. Create a Python environment and install requirements:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r "streamlit_app/requirements.txt"
```

2. Start the app from repository root:

```powershell
streamlit run "streamlit_app/app.py"
```

Notes:
- Place your dataset at `DataSet/emi_prediction_dataset.csv` or use the Admin page to upload one.
- To use an MLflow model, provide a model URI (e.g. `runs:/<run_id>/model` or a local path) on the Predict page.
- The Monitor page reads `mlruns/` folder and offers a simple tabular view of metric files.

Security/backup: when appending dataset rows the original CSV is copied into `DataSet/backups/`.
