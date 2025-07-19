# FMCG Sales & Supply Chain Dashboard
## Hosted on https://fmcgsupply.streamlit.app/
(User: admin, pass: admin)

A Streamlit dashboard for analyzing Fast-Moving Consumer Goods (FMCG) sales and supply chain data in India.

## Problem Statement

FMCG companies in India often struggle with stock mismanagement, supply-demand mismatches, and regional performance disparities. Poor visibility into sales data, inventory turnover, and distribution delays leads to stock-outs, overstocking, and lost revenue.

This dashboard helps businesses:
- Identify key bottlenecks in sales and supply across regions
- Monitor KPIs like stock turnover, sales vs demand, returns, and regional profitability

## Features

### Basic Features
- Region and category filters
- Key performance indicators
- Interactive visualizations
- Data export functionality

### Enhanced Features
- **Advanced Analytics**
  - Anomaly detection to identify unusual patterns
  - Warehouse segmentation for performance analysis
  - Regional performance heatmap
  - Data-driven optimization recommendations

- **Time Series Analysis**
  - Sales trend visualization
  - Seasonal decomposition
  - Future demand forecasting

- **Machine Learning Analytics**
  - Regression analysis with metrics and visualizations
  - Decision tree analysis with feature importance
  - Clustering analysis for data segmentation
  - What-if analysis for scenario planning

- **User Experience**
  - User authentication with admin/user roles
  - Dashboard customization (themes, chart types, layout)
  - Data upload functionality for admins

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```
streamlit run app.py
```

## Data

The dashboard uses FMCG sales and warehouse data stored in CSV format in the `data` directory.

## Project Structure

- `app.py`: Main Streamlit application
- `utils/`: Helper functions and modules
  - `data_loader.py`: Data loading and filtering functions
  - `data_processor.py`: KPI calculation and data preparation
  - `advanced_analytics.py`: Anomaly detection, segmentation, and optimization
  - `time_series.py`: Time series analysis and forecasting
  - `ml_analytics.py`: Machine learning models and what-if analysis
  - `auth.py`: User authentication and session management
  - `data_upload.py`: Data upload and management
  - `customization.py`: Dashboard customization options
- `assets/`: Images and other static assets
- `data/`: CSV data files
  - `uploads/`: User-uploaded data files
- `config/`: Configuration files
  - `auth.yaml`: User credentials and authentication settings
  - `preferences/`: User preference files
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## Extended Workflow (Human-in-the-Loop + AI Extraction)

This project now supports a full pipeline for document-based analytics:

1. **Document Extraction (PDF to CSV using Gemini AI)**
   - Upload PDF invoices.
   - OCR and AI (Gemini) extract structured data into CSV.
   - Download or proceed to verification.

2. **Human-in-the-Loop (HITL) Verification**
   - Upload PDFs and the extracted CSV.
   - Visually verify and correct extracted data (bounding boxes, highlights).
   - Download the verified CSV.

3. **Dashboard Analytics**
   - Upload or select any CSV (including verified ones) for full analytics and insights.

### AI Model
- Uses Google Gemini API for document understanding and extraction (replaceable with other LLMs).

### How to Use
- Run the app and use the sidebar to navigate:
  1. Document Extraction (PDF to CSV)
  2. HITL Verification (PDF + CSV)
  3. Dashboard (analytics)

All steps are integrated into a single Streamlit app for seamless workflow.
