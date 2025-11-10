import streamlit as st
import mlflow
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set MLflow tracking URI to local mlruns folder
MLRUNS_PATH = "file:///D:/Labmentix/EMI Predict AI - Intelligent Financial Risk Assessment Platform/mlruns"

# ==================== HELPER FUNCTIONS ====================

def set_mlflow_tracking():
    """Set MLflow tracking URI to local folder"""
    try:
        mlflow.set_tracking_uri(MLRUNS_PATH)
        return True
    except Exception as e:
        st.error(f"Error setting MLflow tracking URI: {str(e)}")
        return False

def get_experiments():
    """Get all MLflow experiments"""
    try:
        experiments = mlflow.search_experiments()
        if not experiments:
            return []
        
        exp_data = []
        for exp in experiments:
            exp_data.append({
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'artifact_location': exp.artifact_location,
                'lifecycle_stage': exp.lifecycle_stage,
                'creation_time': datetime.fromtimestamp(exp.creation_time / 1000.0) if exp.creation_time else None,
                'tags': exp.tags
            })
        
        return exp_data
    except Exception as e:
        st.error(f"Error fetching experiments: {str(e)}")
        return []

def get_runs(experiment_id, filter_string=None):
    """Get all runs for a specific experiment"""
    try:
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            order_by=["start_time DESC"]
        )
        return runs
    except Exception as e:
        st.error(f"Error fetching runs: {str(e)}")
        return pd.DataFrame()

def get_available_metrics(runs_df):
    """Extract all unique metrics from runs"""
    if runs_df.empty:
        return []
    
    metric_cols = [col for col in runs_df.columns if col.startswith('metrics.')]
    metrics = [col.replace('metrics.', '') for col in metric_cols]
    return sorted(metrics)

def get_available_params(runs_df):
    """Extract all unique parameters from runs"""
    if runs_df.empty:
        return []
    
    param_cols = [col for col in runs_df.columns if col.startswith('params.')]
    params = [col.replace('params.', '') for col in param_cols]
    return sorted(params)

def format_duration(start_time, end_time):
    """Calculate duration between start and end time"""
    if pd.isna(start_time) or pd.isna(end_time):
        return "N/A"
    
    duration = end_time - start_time
    hours = int(duration.total_seconds() // 3600)
    minutes = int((duration.total_seconds() % 3600) // 60)
    seconds = int(duration.total_seconds() % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def find_best_run(runs_df, metric, maximize=True):
    """Find the best run based on a metric"""
    if runs_df.empty or f'metrics.{metric}' not in runs_df.columns:
        return None
    
    runs_with_metric = runs_df.dropna(subset=[f'metrics.{metric}'])
    if runs_with_metric.empty:
        return None
    
    if maximize:
        best_idx = runs_with_metric[f'metrics.{metric}'].idxmax()
    else:
        best_idx = runs_with_metric[f'metrics.{metric}'].idxmin()
    
    return runs_with_metric.loc[best_idx]

def plot_metric_progression(runs_df, metric):
    """Create line chart showing metric progression across runs"""
    if runs_df.empty or f'metrics.{metric}' not in runs_df.columns:
        return None
    
    # Sort by start time
    plot_df = runs_df.dropna(subset=[f'metrics.{metric}']).sort_values('start_time')
    
    if plot_df.empty:
        return None
    
    fig = px.line(
        plot_df,
        x='start_time',
        y=f'metrics.{metric}',
        title=f'{metric.upper()} Progression Over Time',
        labels={'start_time': 'Run Start Time', f'metrics.{metric}': metric.upper()},
        markers=True
    )
    
    fig.update_traces(line_color='#667eea', marker=dict(size=10))
    fig.update_layout(
        xaxis_title="Run Start Time",
        yaxis_title=metric.upper(),
        hovermode='x unified'
    )
    
    return fig

def plot_model_comparison(runs_df, metric):
    """Create bar chart comparing model performances"""
    if runs_df.empty or f'metrics.{metric}' not in runs_df.columns:
        return None
    
    plot_df = runs_df.dropna(subset=[f'metrics.{metric}']).copy()
    
    if plot_df.empty:
        return None
    
    # Create short run names
    plot_df['run_name'] = plot_df['run_id'].str[:8] + '...'
    
    # Sort by metric value
    plot_df = plot_df.sort_values(f'metrics.{metric}', ascending=False)
    
    fig = px.bar(
        plot_df.head(20),
        x='run_name',
        y=f'metrics.{metric}',
        title=f'Top 20 Models by {metric.upper()}',
        labels={'run_name': 'Run ID', f'metrics.{metric}': metric.upper()},
        color=f'metrics.{metric}',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title="Run ID",
        yaxis_title=metric.upper(),
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    return fig

def plot_param_metric_relationship(runs_df, param, metric):
    """Create scatter plot showing relationship between parameter and metric"""
    if runs_df.empty:
        return None
    
    param_col = f'params.{param}'
    metric_col = f'metrics.{metric}'
    
    if param_col not in runs_df.columns or metric_col not in runs_df.columns:
        return None
    
    plot_df = runs_df.dropna(subset=[param_col, metric_col]).copy()
    
    if plot_df.empty:
        return None
    
    # Try to convert param to numeric
    try:
        plot_df[param_col] = pd.to_numeric(plot_df[param_col])
        numeric_param = True
    except:
        numeric_param = False
    
    if numeric_param:
        fig = px.scatter(
            plot_df,
            x=param_col,
            y=metric_col,
            title=f'{metric.upper()} vs {param}',
            labels={param_col: param, metric_col: metric.upper()},
            trendline="ols",
            opacity=0.7
        )
    else:
        fig = px.box(
            plot_df,
            x=param_col,
            y=metric_col,
            title=f'{metric.upper()} by {param}',
            labels={param_col: param, metric_col: metric.upper()},
            color=param_col
        )
    
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(
        xaxis_title=param,
        yaxis_title=metric.upper()
    )
    
    return fig

def plot_parallel_coordinates(runs_df, params, metric):
    """Create parallel coordinates plot for multi-parameter visualization"""
    if runs_df.empty or len(params) < 2:
        return None
    
    metric_col = f'metrics.{metric}'
    param_cols = [f'params.{p}' for p in params]
    
    # Check if all columns exist
    required_cols = param_cols + [metric_col]
    missing_cols = [col for col in required_cols if col not in runs_df.columns]
    if missing_cols:
        return None
    
    plot_df = runs_df[required_cols].dropna().copy()
    
    if plot_df.empty or len(plot_df) < 2:
        return None
    
    # Convert params to numeric where possible
    for col in param_cols:
        try:
            plot_df[col] = pd.to_numeric(plot_df[col])
        except:
            # If conversion fails, encode categorically
            plot_df[col] = pd.Categorical(plot_df[col]).codes
    
    # Prepare dimensions for parallel coordinates
    dimensions = []
    
    for param in params:
        param_col = f'params.{param}'
        dimensions.append(dict(
            label=param,
            values=plot_df[param_col]
        ))
    
    dimensions.append(dict(
        label=metric.upper(),
        values=plot_df[metric_col]
    ))
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=plot_df[metric_col],
                colorscale='Viridis',
                showscale=True,
                cmin=plot_df[metric_col].min(),
                cmax=plot_df[metric_col].max()
            ),
            dimensions=dimensions
        )
    )
    
    fig.update_layout(
        title=f'Parallel Coordinates: Parameters vs {metric.upper()}',
        height=500
    )
    
    return fig

def create_runs_summary_table(runs_df):
    """Create a formatted summary table of runs"""
    if runs_df.empty:
        return pd.DataFrame()
    
    # Select relevant columns
    summary_cols = ['run_id', 'start_time', 'end_time', 'status']
    
    # Add all metrics and params
    metric_cols = [col for col in runs_df.columns if col.startswith('metrics.')]
    param_cols = [col for col in runs_df.columns if col.startswith('params.')]
    
    available_cols = summary_cols + metric_cols + param_cols
    available_cols = [col for col in available_cols if col in runs_df.columns]
    
    summary_df = runs_df[available_cols].copy()
    
    # Calculate duration
    if 'start_time' in summary_df.columns and 'end_time' in summary_df.columns:
        summary_df['duration'] = summary_df.apply(
            lambda row: format_duration(row['start_time'], row['end_time']),
            axis=1
        )
    
    # Format timestamps
    if 'start_time' in summary_df.columns:
        summary_df['start_time'] = summary_df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    if 'end_time' in summary_df.columns:
        summary_df['end_time'] = summary_df['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Rename columns for display
    summary_df.columns = [col.replace('metrics.', '').replace('params.', '') for col in summary_df.columns]
    
    return summary_df

def export_runs_to_csv(runs_df):
    """Convert runs dataframe to CSV format"""
    if runs_df.empty:
        return None
    
    # Create export dataframe
    export_df = runs_df.copy()
    
    # Format timestamps
    if 'start_time' in export_df.columns:
        export_df['start_time'] = export_df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    if 'end_time' in export_df.columns:
        export_df['end_time'] = export_df['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Rename columns
    export_df.columns = [col.replace('metrics.', 'metric_').replace('params.', 'param_') for col in export_df.columns]
    
    return export_df.to_csv(index=False)

# ==================== MAIN RENDER FUNCTION ====================

def render():
    st.title("🔍 Model Monitoring & MLflow Tracking")
    st.markdown("""
    Monitor your machine learning experiments, track model performance, and analyze training runs.
    This dashboard integrates with your local MLflow tracking server.
    """)
    
    # Set MLflow tracking URI
    if not set_mlflow_tracking():
        st.error("❌ Failed to connect to MLflow. Please check your mlruns folder path.")
        return
    
    # Add refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        refresh = st.button("🔄 Refresh", help="Reload experiments and runs")
    
    # Get experiments
    with st.spinner("Loading experiments..."):
        experiments = get_experiments()
    
    if not experiments:
        st.warning("⚠️ No experiments found in your MLflow tracking directory.")
        st.info(f"📁 MLflow Tracking URI: `{MLRUNS_PATH}`")
        st.markdown("""
        **To get started:**
        1. Run your ML experiments with MLflow tracking enabled
        2. Make sure experiments are saved to the mlruns folder
        3. Refresh this page to see your experiments
        """)
        return
    
    st.success(f"✅ Found {len(experiments)} experiment(s)")
    
    # Experiment selection
    st.markdown("---")
    st.subheader("🎯 Select Experiment")
    
    exp_names = [exp['name'] for exp in experiments]
    selected_exp_name = st.selectbox(
        "Choose an experiment to analyze",
        exp_names,
        help="Select the experiment you want to monitor"
    )
    
    # Get selected experiment details
    selected_exp = next(exp for exp in experiments if exp['name'] == selected_exp_name)
    experiment_id = selected_exp['experiment_id']
    
    # Show experiment metadata
    with st.expander("📊 Experiment Metadata"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Experiment ID", experiment_id)
        with col2:
            st.metric("Lifecycle Stage", selected_exp['lifecycle_stage'])
        with col3:
            if selected_exp['creation_time']:
                st.metric("Created", selected_exp['creation_time'].strftime('%Y-%m-%d'))
        
        st.markdown(f"**Artifact Location:** `{selected_exp['artifact_location']}`")
        
        if selected_exp['tags']:
            st.markdown("**Tags:**")
            st.json(selected_exp['tags'])
    
    # Get runs for selected experiment
    st.markdown("---")
    with st.spinner("Loading runs..."):
        runs_df = get_runs(experiment_id)
    
    if runs_df.empty:
        st.warning(f"⚠️ No runs found for experiment '{selected_exp_name}'")
        return
    
    st.success(f"✅ Found {len(runs_df)} run(s)")
    
    # Get available metrics and params
    available_metrics = get_available_metrics(runs_df)
    available_params = get_available_params(runs_df)
    
    if not available_metrics:
        st.warning("⚠️ No metrics found in runs")
        return
    
    # Filters
    st.markdown("---")
    st.subheader("🔧 Filters & Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_metric = st.selectbox(
            "Select Metric to Visualize",
            available_metrics,
            help="Choose the primary metric for analysis"
        )
    
    with col2:
        # Metric optimization direction
        metric_lower_is_better = ['rmse', 'mae', 'mse', 'loss', 'error']
        is_lower_better = any(m in selected_metric.lower() for m in metric_lower_is_better)
        maximize = not is_lower_better
        
        optimization = "Minimize" if is_lower_better else "Maximize"
        st.info(f"📊 Optimizing: **{optimization}** {selected_metric}")
    
    with col3:
        # Date filter
        date_filter = st.selectbox(
            "Filter by Date",
            ["All Time", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
            help="Filter runs by creation date"
        )
    
    # Apply date filter
    filtered_runs = runs_df.copy()
    
    if date_filter != "All Time":
        now = datetime.now()
        if date_filter == "Last 24 Hours":
            cutoff = now - timedelta(days=1)
        elif date_filter == "Last 7 Days":
            cutoff = now - timedelta(days=7)
        else:  # Last 30 Days
            cutoff = now - timedelta(days=30)
        
        filtered_runs = filtered_runs[filtered_runs['start_time'] >= cutoff]
        st.info(f"Showing {len(filtered_runs)} run(s) from {date_filter.lower()}")
    
    # Create tabs for organized content
    tab1, tab2, tab3 = st.tabs(["📋 Overview", "📈 Visualizations", "🖥️ MLflow UI"])
    
    # ==================== TAB 1: OVERVIEW ====================
    with tab1:
        st.header("Runs Overview")
        
        # Find best run
        best_run = find_best_run(filtered_runs, selected_metric, maximize=maximize)
        
        if best_run is not None:
            st.markdown("---")
            st.subheader("🏆 Best Performing Run")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Run ID", best_run['run_id'][:12] + "...")
            with col2:
                metric_value = best_run[f'metrics.{selected_metric}']
                st.metric(selected_metric.upper(), f"{metric_value:.6f}")
            with col3:
                if 'start_time' in best_run and 'end_time' in best_run:
                    duration = format_duration(best_run['start_time'], best_run['end_time'])
                    st.metric("Duration", duration)
                else:
                    st.metric("Duration", "N/A")
            with col4:
                st.metric("Status", best_run.get('status', 'N/A'))
            
            # Show best run parameters
            with st.expander("🔍 Best Run Parameters"):
                param_cols = [col for col in best_run.index if col.startswith('params.')]
                if param_cols:
                    params_dict = {col.replace('params.', ''): best_run[col] for col in param_cols}
                    st.json(params_dict)
                else:
                    st.info("No parameters logged")
            
            # Show all metrics for best run
            with st.expander("📊 Best Run All Metrics"):
                metric_cols = [col for col in best_run.index if col.startswith('metrics.')]
                if metric_cols:
                    metrics_dict = {col.replace('metrics.', ''): best_run[col] for col in metric_cols}
                    
                    # Display as formatted dataframe
                    metrics_df = pd.DataFrame([metrics_dict])
                    st.dataframe(metrics_df, use_container_width=True)
                else:
                    st.info("No metrics logged")
        
        # Summary statistics
        st.markdown("---")
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Runs", len(filtered_runs))
        
        with col2:
            if f'metrics.{selected_metric}' in filtered_runs.columns:
                avg_metric = filtered_runs[f'metrics.{selected_metric}'].mean()
                st.metric(f"Avg {selected_metric}", f"{avg_metric:.6f}")
            else:
                st.metric(f"Avg {selected_metric}", "N/A")
        
        with col3:
            successful_runs = len(filtered_runs[filtered_runs['status'] == 'FINISHED'])
            st.metric("Successful Runs", successful_runs)
        
        with col4:
            failed_runs = len(filtered_runs[filtered_runs['status'] == 'FAILED'])
            st.metric("Failed Runs", failed_runs)
        
        # Insights section
        st.markdown("---")
        st.subheader("💡 Automatic Insights")
        
        # Most used hyperparameters
        if available_params:
            st.markdown("**Most Common Hyperparameters:**")
            
            param_insights = []
            for param in available_params[:5]:  # Show top 5 params
                param_col = f'params.{param}'
                if param_col in filtered_runs.columns:
                    most_common = filtered_runs[param_col].mode()
                    if len(most_common) > 0:
                        count = (filtered_runs[param_col] == most_common[0]).sum()
                        param_insights.append({
                            'Parameter': param,
                            'Most Common Value': most_common[0],
                            'Frequency': f"{count}/{len(filtered_runs)}"
                        })
            
            if param_insights:
                insights_df = pd.DataFrame(param_insights)
                st.dataframe(insights_df, use_container_width=True)
        
        # Performance insights
        if f'metrics.{selected_metric}' in filtered_runs.columns:
            metric_values = filtered_runs[f'metrics.{selected_metric}'].dropna()
            
            if len(metric_values) > 0:
                std_dev = metric_values.std()
                mean_val = metric_values.mean()
                
                if std_dev / mean_val > 0.2:  # High variance
                    st.warning(f"⚠️ High variance detected in {selected_metric} ({std_dev:.4f}). Consider reviewing hyperparameters.")
                else:
                    st.success(f"✅ Consistent performance across runs (std: {std_dev:.4f})")
                
                # Check for improvement trend
                if len(metric_values) >= 3:
                    recent_runs = filtered_runs.nlargest(3, 'start_time')
                    recent_metric = recent_runs[f'metrics.{selected_metric}'].mean()
                    
                    older_runs = filtered_runs.nsmallest(3, 'start_time')
                    older_metric = older_runs[f'metrics.{selected_metric}'].mean()
                    
                    if maximize:
                        improving = recent_metric > older_metric
                    else:
                        improving = recent_metric < older_metric
                    
                    if improving:
                        improvement = abs(recent_metric - older_metric) / abs(older_metric) * 100
                        st.success(f"📈 Model performance improving! Recent runs show {improvement:.2f}% improvement.")
                    else:
                        st.info("ℹ️ Model performance has stabilized or needs tuning.")
        
        # Runs summary table
        st.markdown("---")
        st.subheader("📑 All Runs Summary")
        
        summary_df = create_runs_summary_table(filtered_runs)
        
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True, height=400)
            
            # Export button
            csv_data = export_runs_to_csv(filtered_runs)
            if csv_data:
                st.download_button(
                    label="📥 Download All Metrics as CSV",
                    data=csv_data,
                    file_name=f"mlflow_runs_{selected_exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("No runs to display")
    
    # ==================== TAB 2: VISUALIZATIONS ====================
    with tab2:
        st.header("Performance Visualizations")
        
        # Metric progression
        st.subheader(f"📈 {selected_metric.upper()} Progression Over Time")
        fig1 = plot_metric_progression(filtered_runs, selected_metric)
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning(f"No data available to plot {selected_metric} progression")
        
        st.markdown("---")
        
        # Model comparison
        st.subheader(f"📊 Model Performance Comparison")
        fig2 = plot_model_comparison(filtered_runs, selected_metric)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning(f"No data available to compare models")
        
        st.markdown("---")
        
        # Parameter vs Metric relationship
        st.subheader("🔬 Parameter Impact Analysis")
        
        if available_params:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_param = st.selectbox(
                    "Select Parameter",
                    available_params,
                    help="Choose a parameter to analyze its impact on the metric"
                )
            
            with col2:
                st.info(f"Analyzing: **{selected_param}** vs **{selected_metric}**")
            
            fig3 = plot_param_metric_relationship(filtered_runs, selected_param, selected_metric)
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning(f"Insufficient data to plot relationship")
        else:
            st.info("No parameters logged in runs")
        
        st.markdown("---")
        
        # Parallel coordinates for multi-parameter analysis
        st.subheader("🎯 Multi-Parameter Tuning Visualization")
        
        if len(available_params) >= 2:
            selected_params_parallel = st.multiselect(
                "Select Parameters (2-6 recommended)",
                available_params,
                default=available_params[:min(4, len(available_params))],
                help="Choose multiple parameters to visualize their combined effect"
            )
            
            if len(selected_params_parallel) >= 2:
                fig4 = plot_parallel_coordinates(filtered_runs, selected_params_parallel, selected_metric)
                if fig4:
                    st.plotly_chart(fig4, use_container_width=True)
                    st.info("💡 Each line represents a run. Color indicates the metric value. Look for patterns!")
                else:
                    st.warning("Insufficient data for parallel coordinates plot")
            else:
                st.info("Select at least 2 parameters to create parallel coordinates plot")
        else:
            st.info("Need at least 2 parameters for multi-parameter visualization")
        
        st.markdown("---")
        
        # Additional metrics comparison
        st.subheader("📊 Compare Multiple Metrics")
        
        if len(available_metrics) > 1:
            compare_metrics = st.multiselect(
                "Select Metrics to Compare",
                available_metrics,
                default=available_metrics[:min(3, len(available_metrics))],
                help="Compare multiple metrics side by side"
            )
            
            if compare_metrics:
                # Create subplot for multiple metrics
                metric_data = []
                for metric in compare_metrics:
                    metric_col = f'metrics.{metric}'
                    if metric_col in filtered_runs.columns:
                        values = filtered_runs[metric_col].dropna()
                        if len(values) > 0:
                            metric_data.append({
                                'Metric': metric,
                                'Mean': values.mean(),
                                'Std': values.std(),
                                'Min': values.min(),
                                'Max': values.max()
                            })
                
                if metric_data:
                    compare_df = pd.DataFrame(metric_data)
                    st.dataframe(compare_df, use_container_width=True)
                    
                    # Create comparison chart
                    fig5 = go.Figure()
                    
                    for metric in compare_metrics:
                        metric_col = f'metrics.{metric}'
                        if metric_col in filtered_runs.columns:
                            plot_df = filtered_runs.dropna(subset=[metric_col]).sort_values('start_time')
                            fig5.add_trace(go.Scatter(
                                x=plot_df['start_time'],
                                y=plot_df[metric_col],
                                name=metric,
                                mode='lines+markers'
                            ))
                    
                    fig5.update_layout(
                        title="Multi-Metric Comparison Over Time",
                        xaxis_title="Run Start Time",
                        yaxis_title="Metric Value",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig5, use_container_width=True)
                else:
                    st.warning("No data available for selected metrics")
        else:
            st.info("Only one metric available")
    
    # ==================== TAB 3: MLFLOW UI ====================
    with tab3:
        st.header("🖥️ MLflow UI Integration")
        
        st.markdown("""
        The MLflow UI provides a comprehensive interface for exploring experiments, comparing runs, and managing models.
        
        **To access the full MLflow UI:**
        1. Open a terminal in your project directory
        2. Run: `mlflow ui --backend-store-uri file:///D:/Labmentix/EMI Predict AI - Intelligent Financial Risk Assessment Platform/mlruns`
        3. Open your browser to: `http://localhost:5000`
        """)
        
        st.markdown("---")
        
        # Try to embed MLflow UI
        st.subheader("Embedded MLflow UI")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("💡 If the iframe below is empty, start the MLflow UI server first using the command above.")
        
        with col2:
            if st.button("🔗 Open in New Tab"):
                st.markdown("[Open MLflow UI](http://localhost:5000)", unsafe_allow_html=True)
        
        try:
            st.components.v1.iframe("http://localhost:5000", height=800, scrolling=True)
        except Exception as e:
            st.warning("⚠️ Unable to embed MLflow UI. The server may not be running.")
            st.markdown("""
            **Start MLflow UI with:**
            ```bash
            mlflow ui --backend-store-uri file:///D:/Labmentix/EMI Predict AI - Intelligent Financial Risk Assessment Platform/mlruns
            ```
            """)
            st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        
        # Quick links
        st.subheader("📚 Useful MLflow Resources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **MLflow Documentation:**
            - [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
            - [MLflow Models](https://mlflow.org/docs/latest/models.html)
            - [MLflow Projects](https://mlflow.org/docs/latest/projects.html)
            """)
        
        with col2:
            st.markdown("""
            **Quick Commands:**
            - `mlflow ui` - Start UI server
            - `mlflow runs list` - List runs
            - `mlflow experiments list` - List experiments
            """)
    
    # Footer
    st.markdown("---")
    st.info("💡 **Tip**: Use filters and visualizations to identify the best hyperparameters and track model improvements over time!")

# Run the page
if __name__ == "__main__":
    render()
