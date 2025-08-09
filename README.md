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

## New Features (2025 Update)

- Unified single-app navigation with three pages:
  - Document Extraction (PDF to CSV via Gemini)
  - HITL Verification (PDF + CSV visual inspection with highlights)
  - Dashboard (analytics on any selected CSV)
- Gemini integration for structured data extraction from invoices
- Tesseract OCR replaces PaddleOCR for simpler cross-platform setup
- Secure secrets via `.env` (no keys in code)
- Improved dataset management via `data/uploads/`
- Performance and UX improvements; warnings silenced for cleaner UI

## Environment Setup (.env)

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-gemini-api-key-here
```

Notes:
- `.env` is ignored by git (see `.gitignore`).
- The app will raise a clear error if `GEMINI_API_KEY` is not set.

## OCR Requirements (Tesseract)

This project uses Tesseract via `pytesseract`:
- Install Tesseract binary (Windows): download from the official wiki and add install path (e.g., `C:\Program Files\Tesseract-OCR`) to PATH
- Python packages are installed via `requirements.txt` (includes `pytesseract` and `Pillow`)

## PDF Processing

- Uses `PyMuPDF` (module name `fitz`) for high-quality PDF page rendering.
- Ensure `PyMuPDF` is installed (provided in `requirements.txt`).

## Running Locally

1) Install dependencies
```
pip install -r requirements.txt
```

2) Set your `.env` as described above

3) Run the app
```
streamlit run app.py
```

## Deployment

### Streamlit Cloud
- Add `runtime.txt` with a compatible Python version (we use `python-3.10`)
- Pin compatible scientific stack in `requirements.txt`:
  - `scipy==1.11.4`
  - `statsmodels==0.14.1`
- Push to GitHub and deploy in Streamlit Cloud

### Share Your Local App (no deployment)
- Cloudflare Tunnel (no signup needed):
```
cloudflared tunnel --url http://localhost:8501
```
Copy the `https://*.trycloudflare.com` URL and share it.

## Troubleshooting

- statsmodels/scipy import error on cloud:
  - Ensure `runtime.txt` = `python-3.10`
  - Use `scipy==1.11.4` and `statsmodels==0.14.1`
- OpenCV issues on some hosts:
  - Prefer `opencv-python-headless` if a host cannot install GUI deps, or remove OpenCV usage
- Streamlit set_page_config warning:
  - Only call `st.set_page_config` once in `app.py` as the first Streamlit command
- `fitz` import error:
  - Make sure `PyMuPDF` is installed (do not install the `fitz` package)
- fuzzywuzzy speed warning:
  - `pip install python-Levenshtein`
- Gemini 404 or auth errors:
  - Ensure endpoint is `https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent`
  - Verify `GEMINI_API_KEY` is set in `.env`

## Data Flow

- PDFs are uploaded on the Document Extraction page
- Extracted CSV can be downloaded or sent to HITL Verification
- Verified CSVs can be saved to `data/uploads/` and used by the Dashboard

## Tech Stack

- Streamlit, Python
- Pandas, NumPy, Plotly, Matplotlib, Seaborn
- Scikit-learn, Statsmodels, Prophet
- PyMuPDF (fitz), Tesseract (pytesseract)
- Google Gemini (via REST API), python-dotenv
