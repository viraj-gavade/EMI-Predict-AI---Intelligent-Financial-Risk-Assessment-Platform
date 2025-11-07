import os
import shutil
import pandas as pd
import mlflow
import tempfile
from mlflow.tracking import MlflowClient
import pandas as pd


def load_dataset():
    path = os.path.join(os.getcwd(), "DataSet", "emi_prediction_dataset.csv")
    if not os.path.exists(path):
        return None
    try:
        # use low_memory=False to avoid dtype guessing for large files
        df = pd.read_csv(path, low_memory=False)
        # normalize mixed-type object columns to avoid pyarrow conversion errors in Streamlit
        for col in df.columns:
            if df[col].dtype == object:
                coerced = pd.to_numeric(df[col], errors="coerce")
                # if most values coerce to numeric, keep as numeric
                if coerced.notna().mean() > 0.9:
                    df[col] = coerced
                else:
                    # otherwise convert to string to ensure consistent dtype
                    df[col] = df[col].astype(str)
        return df
    except Exception:
        return None


def backup_dataset(path):
    if not os.path.exists(path):
        return
    bak_dir = os.path.join(os.path.dirname(path), "backups")
    os.makedirs(bak_dir, exist_ok=True)
    base = os.path.basename(path)
    dst = os.path.join(bak_dir, f"{base}.bak")
    shutil.copy2(path, dst)


def safe_load_mlflow_model(model_uri: str):
    """
    Try to load an MLflow model. Accepts local paths or model URIs.
    """
    # allow local path or model:/name/versions/X
    return mlflow.pyfunc.load_model(model_uri)


def list_experiments():
    """Return list of experiments accessible in local mlruns or remote tracking server."""
    client = MlflowClient()
    exps = client.list_experiments()
    # return list of tuples (experiment_id, name)
    return [(e.experiment_id, e.name) for e in exps]


def list_runs(experiment_id: str, max_results: int = 50):
    """Return a list of runs for a given experiment id using MlflowClient.search_runs."""
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id], order_by=["attributes.start_time DESC"], max_results=max_results)
    out = []
    for r in runs:
        d = {
            "run_id": r.info.run_id,
            "status": r.info.status,
            "start_time": getattr(r.info, "start_time", None),
            "end_time": getattr(r.info, "end_time", None),
            "metrics": r.data.metrics,
            "params": r.data.params,
            "tags": r.data.tags,
        }
        out.append(d)
    return out


def get_metric_history(run_id: str, metric_key: str):
    """Return metric history as a pandas DataFrame with columns [timestamp, value]."""
    client = MlflowClient()
    history = client.get_metric_history(run_id, metric_key)
    rows = []
    for m in history:
        rows.append({"timestamp": m.timestamp, "value": m.value})
    if rows:
        return pd.DataFrame(rows).sort_values("timestamp")
    return pd.DataFrame(columns=["timestamp", "value"])


def list_artifacts(run_id: str, path: str = ""):
    """List artifacts for a run. Returns list of dicts {path, is_dir, file_size}.
    If path is empty, lists top-level artifacts."""
    client = MlflowClient()
    try:
        arts = client.list_artifacts(run_id, path)
        out = []
        for a in arts:
            out.append({"path": a.path, "is_dir": a.is_dir, "file_size": getattr(a, "file_size", None)})
        return out
    except Exception:
        return []


def download_artifact(run_id: str, artifact_path: str):
    """Download an artifact file from a run to a temporary directory and return the local file path.
    Note: artifact_path should be the full path relative to the run's artifact root.
    """
    client = MlflowClient()
    tmpdir = tempfile.mkdtemp()
    try:
        local_path = client.download_artifacts(run_id, artifact_path, dst_path=tmpdir)
        return local_path
    except Exception:
        return None


def make_prediction_df(sample: dict):
    import pandas as pd

    return pd.DataFrame([sample])
