# Customer Churn & Prescriptive Retention Analytics Platform

An end-to-end, enterprise-grade telecommunications customer retention platform built on a synthetic panel of 6,000 customers (Jan 2021 – Dec 2025) calibrated to real Indian Telecom Regulatory Authority of India (TRAI) benchmarks. 

The platform integrates longitudinal data engineering, hazard-based machine learning, game-theoretic SHAP explainability, and an interactive "What-If" financial ROI simulator to transform raw customer risk scores into actionable, profit-maximizing retention strategies.

---

## Key Features

- **Longitudinal Telecom Data Engine:** Synthetic 5-year customer panel tracking monthly usage, support calls, billing changes, and churn status calibrated against TRAI industry ARPU and retention metrics.
- **Hazard-Based Churn Predictor:** Machine learning pipeline (HistGradientBoosting) trained on chronological splits with class-imbalance mitigation, predicting month-by-month churn probabilities.
- **Diagnostic Root-Cause Analysis (SHAP):** Local feature attribution using SHAP waterfall plots to isolate why individual customers are at risk of churning.
- **Prescriptive ROI Simulator:** Interactive decision support tool allowing retention teams to simulate targeted discount offers, recalculating churn probability and projected Lifetime Value (LTV) to ensure positive net profitability.
- **Production-Ready Dashboard:** Streamlit UI featuring interactive filters, cohort retention curves, data quality diagnostics, and per-customer prescriptive analytics.

---

## System Architecture & Technical Scope

The project is structured into three integrated core pillars:

### 1. Data Engineering & Telecom Benchmarking
- Engineered synthetic panel data generating realistic customer life cycles across Consumer, Enterprise, and SMB segments.
- Calibrated pricing trends, ARPU figures, and baseline churn rates to match real-world TRAI telecom benchmarks.
- Implemented robust dataset cleaning and storage pipelines (`data/processed/customer_month_clean.csv`).

### 2. Predictive Machine Learning Engine
- Developed custom feature engineering pipelines (`src/features.py`) computing rolling indicators: usage delta percentage (`usage_change_pct`), support call frequency (`support_calls_6m_avg`), and pricing variations.
- Addressed severe class imbalance (~0.9% monthly churn) using chronological train/validation/test splits (2025 hold-out test set).
- Serialized the optimized pipeline and feature names into `models/churn_model.joblib`.

### 3. Prescriptive Analytics & Cloud Infrastructure
- Architected the SHAP diagnostic explainer enabling real-time root-cause analysis for any selected customer ID.
- Designed the interactive "What-If" financial retention simulator computing net revenue saved based on LTV math and offer costs.
- Managed full UI integration across 6 dynamic dashboard tabs in Streamlit.
- Packaged the entire platform into a lightweight Docker container for containerized cloud deployment on Google Cloud Run.

---

## Tech Stack

- **Language:** Python 3.11
- **Dashboard UI:** Streamlit
- **Machine Learning & Analytics:** Scikit-learn, SHAP, NumPy, Pandas
- **Data Visualization:** Matplotlib, Seaborn
- **Containerization & Cloud:** Docker, Google Cloud Run

---

## Repository Structure

```text
customer-churn-prescriptive-analytics/
├── app.py                      # Main Streamlit dashboard application
├── Dockerfile                  # Container definition for GCP Cloud Run
├── requirements.txt            # Python dependencies
├── data/
│   ├── raw/                    # Raw generated synthetic data
│   └── processed/              # Cleaned & ML-ready panel data
├── src/
│   ├── generate_synthetic_data.py # Synthetic data generation script
│   ├── industry_benchmarks.py     # TRAI benchmark integration
│   ├── clean_data.py              # Data cleaning and pipeline processing
│   ├── features.py                # Rolling feature engineering pipeline
│   └── churn_model.py             # Machine learning model training script
├── models/
│   └── churn_model.joblib         # Serialized ML model artifact
└── docs/
    ├── data_dictionary.md         # Data schema specifications
    └── business_insights.md       # Strategic domain recommendations

Quickstart & Local Setup
1. Clone the Repository
Bash
git clone [https://github.com/AbhitiRachel/customer-churn-prescriptive-analytics.git](https://github.com/AbhitiRachel/customer-churn-prescriptive-analytics.git)
cd customer-churn-prescriptive-analytics
2. Install Dependencies
It is recommended to use a Python 3.11 virtual environment:

Bash
pip install -r requirements.txt
3. Execute Data & ML Pipeline
Generate the dataset and train the machine learning model:

Bash
python src/generate_synthetic_data.py
python src/industry_benchmarks.py
python src/clean_data.py
python src/churn_model.py
4. Launch the Streamlit Dashboard
Bash
streamlit run app.py
Cloud Deployment (Google Cloud Run)
To deploy this application to Google Cloud Run as a containerized web service:

Bash
# 1. Build container image in Google Artifact Registry
gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT_ID/churn-app

# 2. Deploy to Cloud Run
gcloud run deploy churn-app \
  --image gcr.io/YOUR_GCP_PROJECT_ID/churn-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
Authors & Attribution
Abhiti Rachel Kannapogu — Prescriptive Analytics Lead & Cloud Architect (SHAP Explainability, Financial ROI Simulator, Dashboard Integration & GCP Cloud Run Deployment)

Data Engineering & ML Modeling Contributors — Data Engineering, Feature Engineering & Hazard Model Training


---

### How to use it on GitHub:
1. Open your repository in your browser: [customer-churn-prescriptive-analytics](https://github.com/AbhitiRachel/customer-churn-prescriptive-analytics)
2. Click **Add a README** (or **Create new file** and name it `README.md`).
3. Hover over the top-right corner of the code block above, click **Copy**, and paste it into the GitHub text editor.
4. Scroll down and click **Commit changes**.
