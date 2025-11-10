import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

def render():
    st.title("📊 Data Exploration and Visualization")
    st.markdown("""
    Upload your dataset to explore trends, visualize relationships, and perform quick Exploratory Data Analysis (EDA).
    This interactive tool helps you understand your data without writing any code.
    """)
    
    # File uploader
    st.markdown("---")
    st.subheader("📁 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset for analysis (max 200MB recommended)"
    )
    
    # Load sample data option
    col1, col2 = st.columns([3, 1])
    with col2:
        use_sample = st.button("📊 Use Sample Dataset", help="Load the EMI prediction dataset")
    
    df = None
    
    # Load data
    if uploaded_file is not None:
        try:
            # Check file size
            file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
            if file_size > 100:
                st.warning(f"⚠️ Large file detected ({file_size:.2f} MB). Processing may be slow.")
            
            with st.spinner("Loading dataset..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Dataset loaded successfully! ({len(df):,} rows, {len(df.columns)} columns)")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return
    
    elif use_sample:
        try:
            from pathlib import Path
            import os
            sample_path = Path(os.getcwd()) / "DataSet" / "emi_prediction_dataset.csv"
            if sample_path.exists():
                with st.spinner("Loading sample dataset..."):
                    df = pd.read_csv(sample_path)
                st.success(f"✅ Sample dataset loaded! ({len(df):,} rows, {len(df.columns)} columns)")
            else:
                st.error("❌ Sample dataset not found in DataSet/ folder")
                return
        except Exception as e:
            st.error(f"❌ Error loading sample data: {str(e)}")
            return
    
    if df is None:
        st.info("👆 Please upload a dataset or use the sample data to begin exploration")
        
        # Show features preview
        with st.expander("💡 What you can do with this tool"):
            st.markdown("""
            - **Overview**: View dataset shape, column types, and basic statistics
            - **Visualizations**: Create interactive plots (histograms, scatter, box, bar, heatmaps)
            - **Insights**: Get automatic insights about correlations, missing values, and distributions
            - **Filtering**: Apply filters to explore specific data segments
            - **Export**: Download cleaned or filtered datasets
            """)
        return
    
    # Store in session state for persistence
    st.session_state['exploration_df'] = df
    
    # Create tabs for organized content
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📈 Visualizations", "🔍 Insights", "⚙️ Data Operations"])
    
    # ==================== TAB 1: OVERVIEW ====================
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
        with col4:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            st.metric("Categorical Columns", len(categorical_cols))
        
        st.markdown("---")
        
        # Data Preview
        st.subheader("📊 Data Preview")
        preview_rows = st.slider("Number of rows to display", 5, 100, 10, key="preview_slider")
        st.dataframe(df.head(preview_rows), use_container_width=True)
        
        # Column Information
        st.markdown("---")
        st.subheader("📝 Column Information")
        
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(2),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
        
        # Summary Statistics
        st.markdown("---")
        st.subheader("📊 Summary Statistics")
        
        stat_option = st.radio("Select columns:", ["Numeric Only", "All Columns"], horizontal=True)
        
        if stat_option == "Numeric Only":
            st.dataframe(df.describe(), use_container_width=True)
        else:
            st.dataframe(df.describe(include='all'), use_container_width=True)
    
    # ==================== TAB 2: VISUALIZATIONS ====================
    with tab2:
        st.header("Interactive Visualizations")
        
        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not numeric_cols and not categorical_cols:
            st.warning("No suitable columns found for visualization")
            return
        
        # Visualization type selector
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart", "Correlation Heatmap", "Pair Plot"],
            help="Choose the type of chart you want to create"
        )
        
        st.markdown("---")
        
        # HISTOGRAM
        if viz_type == "Histogram":
            st.subheader("📊 Distribution Analysis")
            
            if numeric_cols:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    selected_col = st.selectbox("Select column", numeric_cols, key="hist_col")
                
                with col2:
                    bins = st.slider("Number of bins", 10, 100, 30, key="hist_bins")
                
                # Create histogram
                fig = px.histogram(
                    df, 
                    x=selected_col, 
                    nbins=bins,
                    title=f'Distribution of {selected_col}',
                    labels={selected_col: selected_col},
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis_title=selected_col,
                    yaxis_title="Count"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{df[selected_col].mean():.2f}")
                with col2:
                    st.metric("Median", f"{df[selected_col].median():.2f}")
                with col3:
                    st.metric("Std Dev", f"{df[selected_col].std():.2f}")
                with col4:
                    st.metric("Range", f"{df[selected_col].max() - df[selected_col].min():.2f}")
            else:
                st.warning("No numeric columns available for histogram")
        
        # BOX PLOT
        elif viz_type == "Box Plot":
            st.subheader("📦 Outlier Detection")
            
            if numeric_cols:
                selected_cols = st.multiselect(
                    "Select columns to compare",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))],
                    key="box_cols"
                )
                
                if selected_cols:
                    # Create box plot
                    fig = go.Figure()
                    for col in selected_cols:
                        fig.add_trace(go.Box(
                            y=df[col],
                            name=col,
                            boxmean='sd'
                        ))
                    
                    fig.update_layout(
                        title="Box Plot - Outlier Analysis",
                        yaxis_title="Value",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show outlier counts
                    st.markdown("**Outlier Count (using IQR method):**")
                    outlier_info = {}
                    for col in selected_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                        outlier_info[col] = outliers
                    
                    outlier_df = pd.DataFrame(list(outlier_info.items()), columns=['Column', 'Outlier Count'])
                    st.dataframe(outlier_df, use_container_width=True)
                else:
                    st.info("Please select at least one column")
            else:
                st.warning("No numeric columns available for box plot")
        
        # SCATTER PLOT
        elif viz_type == "Scatter Plot":
            st.subheader("🔵 Relationship Analysis")
            
            if len(numeric_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                with col2:
                    y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col], key="scatter_y")
                with col3:
                    color_col = st.selectbox("Color by (optional)", ['None'] + categorical_cols, key="scatter_color")
                
                # Create scatter plot
                if color_col == 'None':
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f'{y_col} vs {x_col}',
                        trendline="ols",
                        opacity=0.6
                    )
                else:
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        title=f'{y_col} vs {x_col} (colored by {color_col})',
                        opacity=0.6
                    )
                
                fig.update_traces(marker=dict(size=8))
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation
                corr = df[x_col].corr(df[y_col])
                st.metric("Correlation Coefficient", f"{corr:.4f}")
                
                if abs(corr) > 0.7:
                    st.success("✅ Strong correlation detected!")
                elif abs(corr) > 0.4:
                    st.info("ℹ️ Moderate correlation")
                else:
                    st.warning("⚠️ Weak or no correlation")
            else:
                st.warning("Need at least 2 numeric columns for scatter plot")
        
        # BAR CHART
        elif viz_type == "Bar Chart":
            st.subheader("📊 Categorical Analysis")
            
            if categorical_cols:
                selected_col = st.selectbox("Select categorical column", categorical_cols, key="bar_col")
                
                # Count values
                value_counts = df[selected_col].value_counts().reset_index()
                value_counts.columns = [selected_col, 'Count']
                
                # Limit to top N
                top_n = st.slider("Show top N categories", 5, 50, 10, key="bar_top_n")
                value_counts = value_counts.head(top_n)
                
                # Create bar chart
                fig = px.bar(
                    value_counts,
                    x=selected_col,
                    y='Count',
                    title=f'Distribution of {selected_col} (Top {top_n})',
                    color='Count',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Unique Values", df[selected_col].nunique())
                with col2:
                    st.metric("Most Common", df[selected_col].mode()[0])
                with col3:
                    st.metric("Most Common Count", df[selected_col].value_counts().iloc[0])
            else:
                st.warning("No categorical columns available for bar chart")
        
        # LINE CHART
        elif viz_type == "Line Chart":
            st.subheader("📈 Trend Analysis")
            
            if numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    y_cols = st.multiselect(
                        "Select columns to plot",
                        numeric_cols,
                        default=numeric_cols[:min(3, len(numeric_cols))],
                        key="line_cols"
                    )
                
                with col2:
                    x_col = st.selectbox("X-axis (optional)", ['Index'] + numeric_cols, key="line_x")
                
                if y_cols:
                    # Create line chart
                    if x_col == 'Index':
                        fig = px.line(
                            df.reset_index(),
                            x='index',
                            y=y_cols,
                            title='Line Chart',
                            labels={'index': 'Index', 'value': 'Value'}
                        )
                    else:
                        fig = px.line(
                            df,
                            x=x_col,
                            y=y_cols,
                            title=f'Line Chart: {", ".join(y_cols)} vs {x_col}'
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least one column to plot")
            else:
                st.warning("No numeric columns available for line chart")
        
        # CORRELATION HEATMAP
        elif viz_type == "Correlation Heatmap":
            st.subheader("🔥 Correlation Matrix")
            
            if len(numeric_cols) >= 2:
                # Calculate correlation matrix
                corr_matrix = df[numeric_cols].corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    zmin=-1,
                    zmax=1
                )
                fig.update_layout(
                    title="Correlation Heatmap",
                    width=800,
                    height=800
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top correlations
                st.markdown("**Top Positive Correlations:**")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
                
                corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
                st.dataframe(corr_df.head(10), use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis")
        
        # PAIR PLOT
        elif viz_type == "Pair Plot":
            st.subheader("🔗 Pair Plot Analysis")
            
            if len(numeric_cols) >= 2:
                selected_cols = st.multiselect(
                    "Select columns (2-5 recommended)",
                    numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))],
                    key="pair_cols"
                )
                
                if len(selected_cols) >= 2:
                    if len(selected_cols) > 6:
                        st.warning("⚠️ Too many columns selected. This may be slow. Consider selecting 2-5 columns.")
                    
                    color_col = st.selectbox("Color by (optional)", ['None'] + categorical_cols, key="pair_color")
                    
                    with st.spinner("Generating pair plot..."):
                        if color_col == 'None':
                            fig = px.scatter_matrix(
                                df[selected_cols],
                                dimensions=selected_cols,
                                title="Pair Plot"
                            )
                        else:
                            fig = px.scatter_matrix(
                                df,
                                dimensions=selected_cols,
                                color=color_col,
                                title=f"Pair Plot (colored by {color_col})"
                            )
                        
                        fig.update_traces(diagonal_visible=False, marker=dict(size=3))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least 2 columns")
            else:
                st.warning("Need at least 2 numeric columns for pair plot")
    
    # ==================== TAB 3: INSIGHTS ====================
    with tab3:
        st.header("🔍 Automatic Insights")
        
        # Missing Values Analysis
        st.subheader("📊 Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': (missing_data.values / len(df) * 100).round(2)
            })
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.bar(
                    missing_df,
                    x='Column',
                    y='Missing %',
                    title='Missing Values by Column',
                    color='Missing %',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")
        
        st.markdown("---")
        
        # Top Correlations
        st.subheader("🔗 Top Feature Correlations")
        
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            
            # Extract top correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j],
                        'Abs Correlation': abs(corr_matrix.iloc[i, j])
                    })
            
            corr_df = pd.DataFrame(corr_pairs).sort_values('Abs Correlation', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top 5 Positive Correlations:**")
                top_positive = corr_df.nlargest(5, 'Correlation')[['Feature 1', 'Feature 2', 'Correlation']]
                st.dataframe(top_positive, use_container_width=True)
            
            with col2:
                st.markdown("**Top 5 Negative Correlations:**")
                top_negative = corr_df.nsmallest(5, 'Correlation')[['Feature 1', 'Feature 2', 'Correlation']]
                st.dataframe(top_negative, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")
        
        st.markdown("---")
        
        # Data Distribution Summary
        st.subheader("📈 Data Distribution Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numeric Features:**")
            if numeric_cols:
                numeric_summary = pd.DataFrame({
                    'Column': numeric_cols,
                    'Mean': [df[col].mean() for col in numeric_cols],
                    'Median': [df[col].median() for col in numeric_cols],
                    'Std': [df[col].std() for col in numeric_cols],
                    'Skewness': [df[col].skew() for col in numeric_cols]
                }).round(2)
                st.dataframe(numeric_summary, use_container_width=True)
            else:
                st.info("No numeric columns found")
        
        with col2:
            st.markdown("**Categorical Features:**")
            if categorical_cols:
                cat_summary = pd.DataFrame({
                    'Column': categorical_cols,
                    'Unique Values': [df[col].nunique() for col in categorical_cols],
                    'Most Common': [df[col].mode()[0] if not df[col].mode().empty else 'N/A' for col in categorical_cols],
                    'Most Common %': [(df[col].value_counts().iloc[0] / len(df) * 100).round(2) if len(df[col].value_counts()) > 0 else 0 for col in categorical_cols]
                })
                st.dataframe(cat_summary, use_container_width=True)
            else:
                st.info("No categorical columns found")
        
        st.markdown("---")
        
        # Feature Type Summary
        st.subheader("📊 Dataset Composition")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Numeric Features", len(numeric_cols))
            st.metric("Categorical Features", len(categorical_cols))
        
        with col2:
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            st.metric("Total Data Points", f"{total_cells:,}")
            st.metric("Missing Data Points", f"{missing_cells:,}")
        
        with col3:
            completeness = ((total_cells - missing_cells) / total_cells * 100).round(2)
            st.metric("Data Completeness", f"{completeness}%")
            
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("Memory Usage", f"{memory_usage:.2f} MB")
    
    # ==================== TAB 4: DATA OPERATIONS ====================
    with tab4:
        st.header("⚙️ Data Operations")
        
        # Filtering Section
        st.subheader("🔍 Filter Data")
        
        filter_col = st.selectbox("Select column to filter", df.columns, key="filter_col")
        
        if df[filter_col].dtype in [np.int64, np.float64]:
            # Numeric filter
            min_val = float(df[filter_col].min())
            max_val = float(df[filter_col].max())
            
            filter_range = st.slider(
                f"Select range for {filter_col}",
                min_val,
                max_val,
                (min_val, max_val),
                key="filter_range"
            )
            
            filtered_df = df[(df[filter_col] >= filter_range[0]) & (df[filter_col] <= filter_range[1])]
        else:
            # Categorical filter
            unique_values = df[filter_col].unique().tolist()
            selected_values = st.multiselect(
                f"Select values for {filter_col}",
                unique_values,
                default=unique_values[:min(5, len(unique_values))],
                key="filter_values"
            )
            
            if selected_values:
                filtered_df = df[df[filter_col].isin(selected_values)]
            else:
                filtered_df = df
        
        st.info(f"Filtered dataset: {len(filtered_df):,} rows (from {len(df):,} original rows)")
        
        if st.checkbox("Show filtered data", key="show_filtered"):
            st.dataframe(filtered_df.head(50), use_container_width=True)
        
        st.markdown("---")
        
        # Data Cleaning
        st.subheader("🧹 Data Cleaning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Remove Duplicates", key="remove_dupes"):
                original_len = len(filtered_df)
                filtered_df = filtered_df.drop_duplicates()
                st.success(f"✅ Removed {original_len - len(filtered_df)} duplicate rows")
        
        with col2:
            if st.button("Drop Rows with Missing Values", key="drop_missing"):
                original_len = len(filtered_df)
                filtered_df = filtered_df.dropna()
                st.success(f"✅ Removed {original_len - len(filtered_df)} rows with missing values")
        
        st.markdown("---")
        
        # Export Section
        st.subheader("💾 Export Data")
        
        export_format = st.radio("Select format", ["CSV", "Excel"], horizontal=True, key="export_format")
        
        if export_format == "CSV":
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Data')
            
            st.download_button(
                label="📥 Download Filtered Data as Excel",
                data=buffer.getvalue(),
                file_name="filtered_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Advanced: Automated EDA
        st.subheader("🤖 Automated EDA Report")
        st.info("💡 **Pro Tip**: Install `ydata-profiling` for comprehensive automated reports")
        
        if st.button("Generate Profiling Report (Beta)", key="profile_btn"):
            try:
                from ydata_profiling import ProfileReport
                
                with st.spinner("Generating comprehensive report... This may take a few minutes."):
                    profile = ProfileReport(df.sample(min(1000, len(df))), title="Data Profiling Report", minimal=True)
                    profile_html = profile.to_html()
                
                st.components.v1.html(profile_html, height=800, scrolling=True)
                
            except ImportError:
                st.warning("⚠️ `ydata-profiling` not installed. Install it with: `pip install ydata-profiling`")
                st.code("pip install ydata-profiling", language="bash")
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.info("💡 **Tip**: Use the tabs above to explore different aspects of your data. Start with Overview, then dive into Visualizations and Insights!")

# Run the page
if __name__ == "__main__":
    render()
