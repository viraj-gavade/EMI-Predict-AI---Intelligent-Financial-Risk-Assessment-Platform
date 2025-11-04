import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import mlflow
import joblib
import sys
from pathlib import Path

# Ensure repository root is on sys.path so `streamlit_app` package imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit_app.utils import (
    load_dataset,
    safe_load_mlflow_model,
    make_prediction_df,
    backup_dataset,
    list_experiments,
    list_runs,
    get_metric_history,
)


APP_TITLE = "EMI Predict AI — Intelligent Financial Risk Assessment"


def main():
    st.set_page_config(APP_TITLE, layout="wide")
    st.sidebar.title(APP_TITLE)
    page = st.sidebar.selectbox("Choose page", ["Home", "Predict", "Explore", "Monitor", "Admin"])

    if page == "Home":
        show_home()
    elif page == "Predict":
        show_predict()
    elif page == "Explore":
        show_explore()
    elif page == "Monitor":
        show_monitor()
    elif page == "Admin":
        show_admin()


def show_home():
    st.title("Welcome")
    st.markdown(
        "This multi-page app provides real-time prediction, interactive exploration, MLflow monitoring, and basic administrative dataset operations for the EMI prediction project. Use the sidebar to navigate."
    )


def show_predict():
    st.header("Real-time prediction")
    st.markdown("Provide input manually or upload a CSV. Select a model (MLflow model URI or local pickle) and run predictions.")

    # dataset preview to infer columns
    df = load_dataset()
    if df is None:
        st.warning("Dataset not found in `DataSet/emi_prediction_dataset.csv`. Upload a dataset in Admin or place it in that path.")

    st.subheader("Model selection")
    model_uri = st.text_input("MLflow model URI (or local model path)", value="")
    upload_local = st.file_uploader("Or upload a sklearn joblib/pickle model file", type=["pkl", "joblib"], help="Optional fallback")

    # allow Monitor page to pre-load a model into session state
    model = st.session_state.get("loaded_model") if "loaded_model" in st.session_state else None
    loaded_model_uri = st.session_state.get("loaded_model_uri") if "loaded_model_uri" in st.session_state else None

    if loaded_model_uri and model is not None:
        st.info(f"Model loaded from Monitor: {loaded_model_uri}")
        if st.button("Clear loaded model"):
            st.session_state.pop("loaded_model", None)
            st.session_state.pop("loaded_model_uri", None)
            model = None

    if model is None:
        if model_uri:
            try:
                model = safe_load_mlflow_model(model_uri)
                st.success("Loaded MLflow model")
            except Exception as e:
                st.error(f"Failed to load MLflow model: {e}")

        if model is None and upload_local is not None:
            try:
                model = joblib.load(upload_local)
                st.success("Loaded uploaded model")
            except Exception as e:
                st.error(f"Failed to load uploaded model: {e}")

    st.subheader("Manual input")
    col1, col2 = st.columns(2)
    sample = {}
    if df is not None:
        # show first 6 numeric columns as sample inputs
        sample_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
    else:
        sample_cols = []

    with st.form("manual_form"):
        for c in sample_cols:
            sample[c] = st.number_input(c, value=float(df[c].median()) if df is not None and c in df else 0.0)
        submitted = st.form_submit_button("Predict single row")

    if submitted:
        inp = make_prediction_df(sample)
        if model is not None:
            try:
                preds = model.predict(inp)
                st.write("Prediction:")
                st.write(preds)
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
        else:
            st.info("No model loaded — showing input DataFrame")
            st.write(inp)

    st.subheader("Batch prediction via CSV")
    upload = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
    if upload is not None:
        uploaded_df = pd.read_csv(upload)
        st.write("Uploaded preview:")
        st.dataframe(uploaded_df.head())

        if st.button("Run batch prediction"):
            if model is None:
                st.error("No model loaded")
            else:
                try:
                    preds = model.predict(uploaded_df)
                    out = uploaded_df.copy()
                    out["prediction"] = preds
                    st.success("Batch predictions complete")
                    st.dataframe(out.head(100))
                    buf = io.StringIO()
                    out.to_csv(buf, index=False)
                    st.download_button("Download predictions CSV", buf.getvalue(), file_name="predictions.csv")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")


def show_explore():
    st.header("Interactive Data Exploration")
    df = load_dataset()
    if df is None:
        st.warning("Dataset not found in `DataSet/emi_prediction_dataset.csv`.")
        return

    st.subheader("Dataset preview")
    st.dataframe(df.sample(min(200, len(df))))

    st.subheader("Column statistics")
    st.write(df.describe())

    st.subheader("Quick plots")
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric:
        x = st.selectbox("X (numeric)", numeric, index=0)
        y = st.selectbox("Y (numeric)", numeric, index=min(1, len(numeric)-1))
        fig = px.scatter(df, x=x, y=y, color=df.columns[-1] if df.shape[1] > 1 else None)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation heatmap")
    corr = df[numeric].corr()
    fig2 = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig2, use_container_width=True)


def show_monitor():
    st.header("MLflow monitoring")
    st.markdown("Browse experiments and runs from the configured MLflow tracking server or local `mlruns/`.")

    try:
        exps = list_experiments()
    except Exception as e:
        st.error(f"Failed to list MLflow experiments: {e}")
        return

    if not exps:
        st.info("No experiments found in MLflow tracking store.")
        return

    exp_map = {f"{name} ({eid})": eid for eid, name in exps}
    sel = st.selectbox("Experiment", list(exp_map.keys()))
    exp_id = exp_map[sel]

    runs = list_runs(exp_id, max_results=100)
    if not runs:
        st.info("No runs found for this experiment.")
        return

    # show runs in a table
    runs_df = pd.DataFrame([
        {
            "run_id": r["run_id"],
            "status": r["status"],
            "start_time": r["start_time"],
            **{f"metric_{k}": v for k, v in r["metrics"].items()},
        }
        for r in runs
    ])
    st.subheader("Runs")
    st.dataframe(runs_df)

    # allow selecting a run for details
    run_ids = [r["run_id"] for r in runs]
    sel_run = st.selectbox("Select run", run_ids)
    chosen = next((r for r in runs if r["run_id"] == sel_run), None)
    if chosen:
        st.markdown(f"**Run ID:** {chosen['run_id']}")
        st.write("Params:")
        st.json(chosen["params"])
        st.write("Metrics (latest):")
        st.json(chosen["metrics"]) 

        # plot metric history for available metrics
        metric_keys = list(chosen["metrics"].keys())
        if metric_keys:
            key = st.selectbox("Metric to plot", metric_keys)
            hist_df = get_metric_history(chosen["run_id"], key)
            if not hist_df.empty:
                fig = px.line(hist_df, x="timestamp", y="value", title=f"{key} over time")
                st.plotly_chart(fig, use_container_width=True)

        if st.button("Load model from this run (runs:/<id>/model)"):
            run_uri = f"runs:/{chosen['run_id']}/model"
            try:
                model = safe_load_mlflow_model(run_uri)
                st.session_state["loaded_model"] = model
                st.session_state["loaded_model_uri"] = run_uri
                st.success(f"Loaded model from run and available in Predict page: {run_uri}")
            except Exception as e:
                st.error(f"Failed to load model from run: {e}")


def show_admin():
    st.header("Administrative dataset operations")
    path = os.path.join(os.getcwd(), "DataSet", "emi_prediction_dataset.csv")
    st.write("Dataset path:", path)

    df = load_dataset()
    if df is None:
        st.warning("No dataset present. Upload a CSV to create one.")

    st.subheader("View and export")
    if df is not None:
        st.dataframe(df.head(200))
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("Download dataset CSV", buf.getvalue(), file_name="emi_prediction_dataset.csv")

    st.subheader("Append rows (simple)")
    uploaded = st.file_uploader("Upload CSV to append to dataset", type=["csv"])
    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        if st.button("Append to dataset"):
            if df is None:
                # create new dataset
                new_df.to_csv(path, index=False)
                st.success("Created new dataset from uploaded CSV")
            else:
                backup_dataset(path)
                out = pd.concat([df, new_df], ignore_index=True)
                out.to_csv(path, index=False)
                st.success(f"Appended {len(new_df)} rows and backed up previous file")


if __name__ == "__main__":
    main()
