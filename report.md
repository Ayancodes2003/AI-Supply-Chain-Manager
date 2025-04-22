# FMCG Sales & Supply Chain Dashboard - Comprehensive Project Report

## Executive Summary

The FMCG Sales & Supply Chain Dashboard is a comprehensive data analytics platform designed to help Fast-Moving Consumer Goods (FMCG) companies in India analyze their sales and supply chain data. The dashboard addresses critical business challenges such as stock mismanagement, supply-demand mismatches, and regional performance disparities by providing interactive visualizations, advanced analytics, and machine learning capabilities.

Built using Python and Streamlit, the dashboard offers a user-friendly interface with multiple analytical modules, including basic performance metrics, advanced analytics, time series analysis, and machine learning models. The platform enables users to identify bottlenecks, monitor key performance indicators, and make data-driven decisions to optimize their supply chain operations.

## Business Context

FMCG companies in India face several challenges in managing their supply chains effectively:

1. **Stock Management Issues**: Frequent stock-outs or overstocking leading to lost sales or increased holding costs
2. **Supply-Demand Mismatches**: Difficulty in aligning production with market demand
3. **Regional Performance Disparities**: Uneven performance across different geographical regions
4. **Limited Data Visibility**: Lack of comprehensive insights into sales data, inventory turnover, and distribution delays

This dashboard addresses these challenges by providing a centralized platform for data analysis and visualization, enabling stakeholders to make informed decisions based on real-time data.

## Technical Architecture

### Technology Stack

- **Frontend**: Streamlit (Python-based web application framework)
- **Backend**: Python 3.10+
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Machine Learning**: Scikit-learn
- **Authentication**: Custom authentication system

### Project Structure

```
FMCG-sales-and-supply-chain/
├── app.py                  # Main Streamlit application
├── utils/                  # Helper functions and modules
│   ├── __init__.py
│   ├── data_loader.py      # Data loading and filtering functions
│   ├── data_processor.py   # KPI calculation and data preparation
│   ├── advanced_analytics.py # Anomaly detection, segmentation, and optimization
│   ├── time_series.py      # Time series analysis and forecasting
│   ├── ml_analytics.py     # Machine learning models and what-if analysis
│   ├── auth.py             # User authentication and session management
│   ├── data_upload.py      # Data upload and management
│   └── customization.py    # Dashboard customization options
├── assets/                 # Images and other static assets
├── data/                   # CSV data files
│   └── uploads/            # User-uploaded data files
├── config/                 # Configuration files
│   ├── auth.yaml           # User credentials and authentication settings
│   └── preferences/        # User preference files
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Detailed Feature Description

### 1. Authentication System

The dashboard implements a user authentication system with role-based access control:

- **Login Screen**: Users must authenticate with username and password
- **User Roles**: Two roles are supported - Admin and Regular User
- **Role-Based Access**: Admins have additional privileges such as data upload capabilities
- **Session Management**: User sessions are maintained throughout the application
- **Logout Functionality**: Users can securely log out of the application

**Implementation Details**:
- The authentication system is implemented in `utils/auth.py`
- User credentials are stored in `config/auth.yaml`
- Default credentials:
  - Admin: username: `admin`, password: `admin`
  - Regular User: username: `user`, password: `password`

### 2. Dashboard Customization

Users can personalize their dashboard experience through various customization options:

- **Themes**: Choose from Light, Dark, Blue, or Green color themes
- **Chart Types**: Select default chart type (Bar, Line, Scatter, Pie)
- **Layout Options**: Choose between Standard, Compact, or Expanded layouts
- **Data Preview Toggle**: Show or hide the data preview section
- **Default View**: Set the default page to display on login

**Implementation Details**:
- Customization options are implemented in `utils/customization.py`
- User preferences are stored in `config/preferences/` directory
- Theme settings affect both the UI elements and chart appearances

### 3. Main Dashboard

The main dashboard provides an overview of key performance indicators and visualizations:

- **KPI Cards**: Display critical metrics including:
  - Total number of warehouses
  - Average product weight in tons
  - Transport issues in the last year
  - Storage issues in the last 3 months

- **Performance Analysis Visualizations**:
  - Regional Performance: Bar chart showing total product weight by region
  - Category Distribution: Pie chart showing distribution by location type
  - Forecast vs Actual: Comparison of actual vs forecasted values by region
  - Warehouse Issues: Bar chart showing warehouse breakdowns by region

- **Supply Chain Network**: Network visualization showing connections between zones and regional zones

- **Data Export**: Functionality to download filtered data as CSV for further analysis

- **Data Preview**: Table showing a sample of the filtered data

**Implementation Details**:
- The main dashboard is implemented in the `display_dashboard()` function in `app.py`
- Visualizations are created using Plotly
- KPI calculations are performed in `utils/data_processor.py`

### 4. Advanced Analytics

The Advanced Analytics section provides deeper insights into the data through specialized analytical techniques:

- **Anomaly Detection**:
  - Identifies unusual patterns in warehouse data using Isolation Forest algorithm
  - Visualizes anomalies by region
  - Displays detailed information about anomalous warehouses

- **Warehouse Segmentation**:
  - Segments warehouses into performance groups (High, Medium, Low)
  - Visualizes segments by region
  - Provides segment characteristics and statistics

- **Regional Heatmap**:
  - Visualizes performance variations across regions and zones
  - Uses color intensity to represent metric values

- **Optimization Recommendations**:
  - Generates data-driven recommendations to optimize supply chain
  - Identifies specific areas for improvement based on data analysis

**Implementation Details**:
- Advanced analytics features are implemented in `utils/advanced_analytics.py`
- The interface is created in the `display_advanced_analytics()` function in `app.py`
- Segmentation uses K-means clustering from scikit-learn
- Anomaly detection uses Isolation Forest from scikit-learn

### 5. Time Series Analysis

The Time Series Analysis section focuses on temporal patterns and forecasting:

- **Sales Trend Analysis**:
  - Visualizes sales trends over time
  - Supports different aggregation levels (month, quarter)
  - Allows filtering by zone

- **Seasonal Decomposition**:
  - Breaks down time series into trend, seasonal, and residual components
  - Helps identify underlying patterns in the data

- **Future Forecast**:
  - Predicts future values based on historical patterns
  - Allows adjusting the forecast period
  - Visualizes forecasted values by zone

**Implementation Details**:
- Time series features are implemented in `utils/time_series.py`
- The interface is created in the `display_time_series_analysis()` function in `app.py`
- Since the sample data doesn't have real time series, the system simulates time series data with seasonal patterns

### 6. Machine Learning Analytics

The ML Analytics section provides advanced machine learning capabilities for deeper data analysis:

- **Regression Analysis**:
  - Builds linear regression models to analyze relationships between variables
  - Displays regression metrics (R², MSE) for model evaluation
  - Visualizes actual vs. predicted values
  - Shows regression equation and coefficients

- **Decision Tree Analysis**:
  - Creates decision tree models for interpretable predictions
  - Visualizes feature importance
  - Displays the decision tree structure
  - Shows decision rules in text format

- **Clustering Analysis**:
  - Performs K-means clustering to segment data points
  - Visualizes clusters in 2D space
  - Shows cluster centers and statistics
  - Allows adjusting the number of clusters

- **What-If Analysis**:
  - Enables interactive scenario planning with trained models
  - Provides sliders to adjust feature values
  - Shows real-time prediction updates
  - Compares with original data distribution

**Implementation Details**:
- ML analytics features are implemented in `utils/ml_analytics.py`
- The interface is created in the `display_ml_analytics()` function in `app.py`
- Models are built using scikit-learn
- Decision tree visualization uses matplotlib
- Interactive elements use Streamlit widgets

### 7. Data Upload and Management

The Data Upload section allows administrators to manage the data used in the dashboard:

- **Data Upload**: Admins can upload new CSV or Excel files
- **File Preview**: Shows a preview of the uploaded data
- **Confirmation**: Requires confirmation before saving the file
- **Available Datasets**: Lists all available datasets in the system

**Implementation Details**:
- Data upload features are implemented in `utils/data_upload.py`
- The interface is created in the `display_data_upload_page()` function in `app.py`
- Uploaded files are stored in the `data/uploads/` directory
- Only users with admin role can upload new data

## Data Processing Pipeline

The data processing pipeline consists of several stages:

1. **Data Loading**: Raw data is loaded from CSV files using pandas
2. **Data Filtering**: Users can filter data by region, category, and other parameters
3. **Data Transformation**: Raw data is transformed into formats suitable for visualization and analysis
4. **KPI Calculation**: Key performance indicators are calculated from the filtered data
5. **Visualization Preparation**: Data is prepared for various visualization types
6. **Machine Learning Processing**: Data is processed for machine learning models

**Implementation Details**:
- Data loading and filtering are implemented in `utils/data_loader.py`
- KPI calculation and visualization preparation are implemented in `utils/data_processor.py`
- The pipeline is designed to be modular and extensible

## User Workflow

### 1. Authentication

1. User navigates to the dashboard URL
2. User enters username and password
3. System validates credentials and assigns appropriate role
4. User is redirected to the main dashboard

### 2. Dashboard Navigation

1. User selects a page from the navigation menu:
   - Dashboard
   - Advanced Analytics
   - Time Series Analysis
   - ML Analytics
   - Data Upload (admin only)

2. User applies filters:
   - Selects regions from the multiselect dropdown
   - Selects categories from the multiselect dropdown

3. User customizes dashboard appearance:
   - Selects theme
   - Chooses default chart type
   - Adjusts layout options
   - Toggles data preview

### 3. Data Analysis

1. **Basic Analysis**:
   - User views KPIs and visualizations on the main dashboard
   - User exports filtered data for further analysis

2. **Advanced Analysis**:
   - User explores anomalies, segments, and recommendations
   - User analyzes regional performance through heatmaps

3. **Time Series Analysis**:
   - User selects zone and aggregation level
   - User views sales trends and seasonal patterns
   - User generates forecasts for future periods

4. **Machine Learning Analysis**:
   - User selects variables for regression or decision tree analysis
   - User runs models and interprets results
   - User performs clustering to identify segments
   - User conducts what-if analysis to explore scenarios

### 4. Data Management (Admin Only)

1. Admin navigates to the Data Upload page
2. Admin uploads new data files
3. Admin previews and confirms the upload
4. Admin views available datasets in the system

## Technical Implementation Details

### Data Handling

- **Data Format**: The system works with CSV and Excel files
- **Data Structure**: The data should contain columns for:
  - Warehouse information (ID, capacity, location)
  - Product information (weight, category)
  - Regional information (zone, regional zone)
  - Performance metrics (transport issues, storage issues, breakdowns)

- **Data Processing**: The system uses pandas for data manipulation:
  - Filtering based on user selections
  - Aggregation for visualizations
  - Feature engineering for machine learning models

### Visualization Techniques

- **Interactive Charts**: All visualizations are interactive, allowing users to:
  - Hover over data points for more information
  - Zoom in/out of charts
  - Pan across charts
  - Download chart images

- **Chart Types**:
  - Bar charts for comparing values across categories
  - Pie charts for showing proportions
  - Line charts for time series data
  - Scatter plots for relationship analysis
  - Heatmaps for regional performance
  - Network diagrams for supply chain visualization

### Machine Learning Models

- **Regression Models**:
  - Linear Regression for relationship analysis
  - Model evaluation using R² and MSE metrics
  - Train/test split for validation

- **Decision Trees**:
  - Decision Tree Regressor for interpretable predictions
  - Feature importance analysis
  - Tree visualization and rule extraction

- **Clustering**:
  - K-means clustering for segmentation
  - Standardization of features before clustering
  - Cluster statistics and visualization

- **Anomaly Detection**:
  - Isolation Forest for identifying outliers
  - Configurable contamination parameter

### User Interface Design

- **Responsive Layout**: The dashboard adapts to different screen sizes
- **Modular Design**: The interface is organized into tabs and sections
- **Consistent Styling**: Consistent color schemes and design elements
- **Interactive Elements**: Sliders, dropdowns, and buttons for user interaction
- **Feedback Mechanisms**: Success/error messages for user actions

## Deployment and Usage

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

Run the Streamlit app:
```
streamlit run app.py
```

### User Access

- **Admin Access**: Username: `admin`, Password: `admin`
- **Regular User Access**: Username: `user`, Password: `password`

## Future Enhancements

### Short-term Improvements

1. **Enhanced Data Import**:
   - Support for more file formats (JSON, SQL databases)
   - Automated data cleaning and validation

2. **Additional Visualizations**:
   - Geospatial maps for regional analysis
   - Sankey diagrams for flow analysis
   - 3D visualizations for complex relationships

3. **Model Persistence**:
   - Save trained models for future use
   - Load pre-trained models for quick analysis

### Long-term Roadmap

1. **Advanced ML Models**:
   - Support for more algorithms (Random Forest, XGBoost, etc.)
   - Hyperparameter tuning capabilities
   - Automated feature selection

2. **Predictive Maintenance**:
   - Predict warehouse breakdowns before they occur
   - Schedule preventive maintenance

3. **Inventory Optimization**:
   - Recommend optimal inventory levels
   - Suggest reorder points and quantities

4. **Route Optimization**:
   - Optimize delivery routes
   - Reduce transportation costs and time

5. **Integration Capabilities**:
   - API for connecting with other systems
   - Real-time data integration

## Conclusion

The FMCG Sales & Supply Chain Dashboard provides a comprehensive solution for analyzing and optimizing supply chain operations in the Fast-Moving Consumer Goods sector. By combining interactive visualizations, advanced analytics, and machine learning capabilities, the dashboard enables users to gain valuable insights from their data and make informed decisions.

The modular architecture and extensible design allow for future enhancements and customizations, making the dashboard a valuable tool for FMCG companies looking to improve their supply chain efficiency and performance.

Through its user-friendly interface and powerful analytical capabilities, the dashboard addresses the key challenges faced by FMCG companies in India, helping them to better manage their stock, align supply with demand, understand regional performance variations, and gain comprehensive visibility into their supply chain operations.
