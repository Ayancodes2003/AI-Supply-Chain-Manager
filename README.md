# FMCG Sales & Supply Chain Dashboard

A Streamlit dashboard for analyzing Fast-Moving Consumer Goods (FMCG) sales and supply chain data in India.

## Problem Statement

FMCG companies in India often struggle with stock mismanagement, supply-demand mismatches, and regional performance disparities. Poor visibility into sales data, inventory turnover, and distribution delays leads to stock-outs, overstocking, and lost revenue.

This dashboard helps businesses:
- Identify key bottlenecks in sales and supply across regions
- Monitor KPIs like stock turnover, sales vs demand, returns, and regional profitability

## Features

- Region and category filters
- Date range selection
- Key performance indicators
- Interactive visualizations
- Data export functionality

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
- `utils/`: Helper functions for data processing
- `assets/`: Images and other static assets
- `data/`: CSV data files
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation
