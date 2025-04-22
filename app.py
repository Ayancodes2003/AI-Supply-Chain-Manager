import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np
from pathlib import Path

# Import utility functions
from utils.data_loader import load_data, get_regions, get_categories, filter_data
from utils.data_processor import (
    calculate_kpis,
    prepare_monthly_trend_data,
    prepare_category_distribution,
    prepare_regional_performance,
    prepare_forecast_vs_actual
)

# Import enhanced modules
from utils.advanced_analytics import (
    forecast_demand,
    detect_anomalies,
    segment_warehouses,
    create_supply_chain_network,
    create_heatmap,
    generate_optimization_recommendations
)
from utils.auth import setup_auth, check_auth, get_user_role, save_user_preferences, load_user_preferences
from utils.data_upload import upload_data, list_available_datasets, load_selected_dataset
from utils.customization import customize_dashboard, apply_theme, get_chart_template, create_chart
from utils.time_series import generate_time_series_data, create_time_series_chart, perform_seasonal_decomposition

# Set page configuration
st.set_page_config(
    page_title="FMCG Sales & Supply Chain Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .kpi-label {
        font-size: 1rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def get_cached_data():
    return load_data()

# Create necessary directories
def create_directories():
    # Create config directory if it doesn't exist
    config_dir = Path("./config")
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create uploads directory if it doesn't exist
    uploads_dir = Path("./data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

# Main function
def main():
    # Create necessary directories
    create_directories()

    # Set up authentication
    authenticator, name, authentication_status = setup_auth()

    # Check authentication status
    if authentication_status == False:
        st.error("Username/password is incorrect")
        return
    elif authentication_status == None:
        st.warning("Please enter your username and password")
        return

    # Store authentication status in session state
    st.session_state['authentication_status'] = authentication_status
    st.session_state['username'] = name

    # Get user role
    user_role = get_user_role()

    # Load user preferences
    user_prefs = load_user_preferences(name)

    # Apply theme
    apply_theme(user_prefs.get('theme', 'Light'))

    # Header with logout button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown(f'<div class="main-header">FMCG Sales & Supply Chain Dashboard</div>', unsafe_allow_html=True)
    with col2:
        authenticator.logout("Logout", "main")

    st.markdown(f"<div style='text-align: right; color: #666;'>Welcome, {name}</div>", unsafe_allow_html=True)

    # Dashboard customization
    customization = customize_dashboard()

    # Save preferences if they've changed
    if customization != user_prefs:
        save_user_preferences(name, customization)
        user_prefs = customization

    # Navigation
    st.sidebar.markdown("## Navigation")

    # Define available pages
    available_pages = ["Dashboard", "Advanced Analytics", "Time Series Analysis", "Data Upload"]

    # Get default view from user preferences, fallback to 'Dashboard' if not found
    default_view = user_prefs.get('default_view', 'Dashboard')

    # Make sure the default view is in the available pages
    if default_view not in available_pages:
        default_view = 'Dashboard'

    # Create the navigation radio buttons
    page = st.sidebar.radio(
        "Go to",
        available_pages,
        index=available_pages.index(default_view)
    )

    # Load data
    df = get_cached_data()

    # Sidebar filters
    st.sidebar.markdown("## Filters")

    # Region filter
    regions = get_regions(df)
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=regions,
        default=regions
    )

    # Category filter
    categories = get_categories(df)
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        options=categories,
        default=categories
    )

    # Apply filters
    filtered_df = filter_data(df, regions=selected_regions, categories=selected_categories)

    # Main Dashboard Page
    if page == "Dashboard":
        display_dashboard(filtered_df, user_prefs)

    # Advanced Analytics Page
    elif page == "Advanced Analytics":
        display_advanced_analytics(filtered_df, user_prefs)

    # Time Series Analysis Page
    elif page == "Time Series Analysis":
        display_time_series_analysis(filtered_df, user_prefs)

    # Data Upload Page
    elif page == "Data Upload":
        display_data_upload_page(user_role)

# Dashboard page
def display_dashboard(filtered_df, user_prefs):
    # Calculate KPIs
    kpis = calculate_kpis(filtered_df)

    # Display KPIs
    st.markdown('<div class="sub-header">Key Performance Indicators</div>', unsafe_allow_html=True)

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get('warehouse_count', 0):,}</div>
            <div class="kpi-label">Warehouses</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get('avg_product_weight', 0):,.1f}</div>
            <div class="kpi-label">Avg. Product Weight (tons)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get('transport_issues', 0):,}</div>
            <div class="kpi-label">Transport Issues (Last Year)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get('storage_issues', 0):,}</div>
            <div class="kpi-label">Storage Issues (Last 3 Months)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Visualizations
    st.markdown('<div class="sub-header">Performance Analysis</div>', unsafe_allow_html=True)

    # Get chart template based on theme
    template = get_chart_template(user_prefs.get('theme', 'Light'))

    # Row 1: Regional Performance and Category Distribution
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Regional Performance")
        regional_data = prepare_regional_performance(filtered_df)
        if not regional_data.empty:
            fig = create_chart(
                regional_data,
                user_prefs.get('chart_type', 'Bar'),
                'Region',
                'Total Weight',
                color='Region',
                title='Total Product Weight by Region (tons)',
                template=template
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for regional performance visualization.")

    with row1_col2:
        st.subheader("Category Distribution")
        category_data = prepare_category_distribution(filtered_df)
        if not category_data.empty:
            fig = px.pie(
                category_data,
                values='Count',
                names='Category',
                title='Distribution by Location Type',
                hole=0.4,
                template=template
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for category distribution visualization.")

    # Row 2: Forecast vs Actual and Warehouse Issues
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("Forecast vs Actual")
        forecast_data = prepare_forecast_vs_actual(filtered_df)
        if not forecast_data.empty:
            fig = px.bar(
                forecast_data,
                x='Region',
                y=['Actual', 'Forecast'],
                barmode='group',
                title='Actual vs Forecasted Values by Region',
                template=template
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for forecast vs actual visualization.")

    with row2_col2:
        st.subheader("Warehouse Issues by Region")
        if 'zone' in filtered_df.columns and 'wh_breakdown_l3m' in filtered_df.columns:
            wh_issues = filtered_df.groupby('zone')['wh_breakdown_l3m'].sum().reset_index()
            wh_issues.columns = ['Region', 'Issues']

            fig = px.bar(
                wh_issues,
                x='Region',
                y='Issues',
                color='Region',
                title='Warehouse Breakdowns by Region (Last 3 Months)',
                template=template
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for warehouse issues visualization.")

    # Supply Chain Network Visualization
    st.markdown("---")
    st.markdown('<div class="sub-header">Supply Chain Network</div>', unsafe_allow_html=True)

    network_fig = create_supply_chain_network(filtered_df)
    if network_fig:
        st.plotly_chart(network_fig, use_container_width=True)
    else:
        st.info("No data available for supply chain network visualization.")

    # Data Export
    st.markdown("---")
    st.markdown('<div class="sub-header">Data Export</div>', unsafe_allow_html=True)

    export_col1, export_col2 = st.columns([3, 1])

    with export_col1:
        st.markdown("Export the filtered data to CSV for further analysis.")

    with export_col2:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="fmcg_data_export.csv",
            mime="text/csv"
        )

    # Data Preview
    if user_prefs.get('show_data_preview', True):
        st.markdown("---")
        st.markdown('<div class="sub-header">Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(filtered_df.head(10), use_container_width=True)

# Advanced Analytics page
def display_advanced_analytics(filtered_df, user_prefs):
    st.markdown('<div class="sub-header">Advanced Analytics</div>', unsafe_allow_html=True)

    # Get chart template based on theme
    template = get_chart_template(user_prefs.get('theme', 'Light'))

    # Tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["Anomaly Detection", "Warehouse Segmentation", "Regional Heatmap", "Optimization Recommendations"])

    with tab1:
        st.subheader("Anomaly Detection")
        st.markdown("Identify unusual patterns in warehouse data that may indicate issues.")

        # Parameters
        contamination = st.slider("Expected Anomaly Percentage", 0.01, 0.2, 0.05, 0.01)

        # Detect anomalies
        anomaly_df = detect_anomalies(filtered_df, contamination=contamination)

        # Display results
        if 'is_anomaly' in anomaly_df.columns:
            # Count anomalies by region
            anomaly_count = anomaly_df.groupby(['zone', 'is_anomaly']).size().reset_index(name='count')
            anomaly_count = anomaly_count[anomaly_count['is_anomaly'] == 'Yes']

            if not anomaly_count.empty:
                fig = px.bar(
                    anomaly_count,
                    x='zone',
                    y='count',
                    color='zone',
                    title='Anomalies Detected by Region',
                    labels={'zone': 'Region', 'count': 'Number of Anomalies'},
                    template=template
                )
                st.plotly_chart(fig, use_container_width=True)

                # Display anomalous warehouses
                st.subheader("Anomalous Warehouses")
                anomalous_wh = anomaly_df[anomaly_df['is_anomaly'] == 'Yes']
                if not anomalous_wh.empty:
                    st.dataframe(anomalous_wh[['Ware_house_ID', 'zone', 'WH_capacity_size', 'product_wg_ton']], use_container_width=True)
                else:
                    st.info("No anomalies detected.")
            else:
                st.info("No anomalies detected in the selected regions.")
        else:
            st.error("Could not perform anomaly detection on the current dataset.")

    with tab2:
        st.subheader("Warehouse Segmentation")
        st.markdown("Segment warehouses into performance groups based on key metrics.")

        # Parameters
        n_clusters = st.slider("Number of Segments", 2, 5, 3, 1)

        # Segment warehouses
        segmented_df = segment_warehouses(filtered_df, n_clusters=n_clusters)

        # Display results
        if 'performance_segment' in segmented_df.columns:
            # Count warehouses by segment and region
            segment_count = segmented_df.groupby(['zone', 'performance_segment']).size().reset_index(name='count')

            if not segment_count.empty:
                fig = px.bar(
                    segment_count,
                    x='zone',
                    y='count',
                    color='performance_segment',
                    title='Warehouse Segments by Region',
                    labels={'zone': 'Region', 'count': 'Number of Warehouses', 'performance_segment': 'Performance Segment'},
                    template=template,
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Display segment characteristics
                st.subheader("Segment Characteristics")
                segment_stats = segmented_df.groupby('performance_segment').agg({
                    'product_wg_ton': 'mean',
                    'transport_issue_l1y': 'mean' if 'transport_issue_l1y' in segmented_df.columns else 'count',
                    'storage_issue_reported_l3m': 'mean' if 'storage_issue_reported_l3m' in segmented_df.columns else 'count'
                }).reset_index()

                st.dataframe(segment_stats, use_container_width=True)
            else:
                st.info("No segments could be created with the current data.")
        else:
            st.error("Could not perform segmentation on the current dataset.")

    with tab3:
        st.subheader("Regional Performance Heatmap")
        st.markdown("Visualize performance variations across regions and zones.")

        # Create heatmap
        heatmap_fig = create_heatmap(filtered_df)

        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.info("Not enough data to create a heatmap.")

    with tab4:
        st.subheader("Optimization Recommendations")
        st.markdown("Get data-driven recommendations to optimize your supply chain.")

        # Generate recommendations
        recommendations = generate_optimization_recommendations(filtered_df)

        # Display recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}. {rec}**")
        else:
            st.info("No specific recommendations available for the current data.")

# Time Series Analysis page
def display_time_series_analysis(filtered_df, user_prefs):
    st.markdown('<div class="sub-header">Time Series Analysis</div>', unsafe_allow_html=True)

    # Get chart template based on theme
    template = get_chart_template(user_prefs.get('theme', 'Light'))

    # Generate time series data
    time_series_df = generate_time_series_data(filtered_df)

    if time_series_df.empty:
        st.error("Could not generate time series data from the current dataset.")
        return

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        # Zone selection
        zones = time_series_df['Zone'].unique()
        selected_zone = st.selectbox("Select Zone", options=zones)

    with col2:
        # Aggregation level
        aggregation = st.selectbox(
            "Aggregation Level",
            options=["month", "quarter"],
            index=0
        )

    # Create time series chart
    st.subheader("Sales Trend Analysis")
    ts_chart = create_time_series_chart(time_series_df, zone=selected_zone, aggregation=aggregation)

    if ts_chart:
        st.plotly_chart(ts_chart, use_container_width=True)
    else:
        st.info("Not enough data to create a time series chart.")

    # Seasonal Decomposition
    st.subheader("Seasonal Decomposition")
    st.markdown("Break down the time series into trend, seasonal, and residual components.")

    decomp_chart = perform_seasonal_decomposition(time_series_df, selected_zone)

    if decomp_chart:
        st.plotly_chart(decomp_chart, use_container_width=True)
    else:
        st.info("Not enough data for seasonal decomposition.")

    # Forecast
    st.subheader("Future Forecast")
    st.markdown("Predict future values based on historical patterns.")

    # Forecast parameters
    forecast_periods = st.slider("Forecast Periods", 1, 12, 3)

    # Generate forecast
    forecast_df = forecast_demand(filtered_df, periods=forecast_periods)

    if not forecast_df.empty:
        fig = px.bar(
            forecast_df,
            x='Period',
            y='Forecasted_Weight',
            color='Zone',
            title=f'Forecasted Product Weight for Next {forecast_periods} Periods',
            template=template,
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to generate a forecast.")

# Data Upload page
def display_data_upload_page(user_role):
    st.markdown('<div class="sub-header">Data Management</div>', unsafe_allow_html=True)

    # Only allow admin users to upload data
    if user_role == 'admin':
        # Data upload section
        st.subheader("Upload New Data")
        uploaded_df = upload_data()

        if uploaded_df is not None:
            st.success("Data uploaded successfully! You can now use it in the dashboard.")
    else:
        st.warning("You need admin privileges to upload new data.")

    # List available datasets
    st.subheader("Available Datasets")
    datasets = list_available_datasets()

    if datasets:
        for dataset in datasets:
            st.markdown(f"- {dataset}")
    else:
        st.info("No datasets available.")

if __name__ == "__main__":
    main()
