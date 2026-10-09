"""
Customer Churn & Retention Dashboard
====================================
Streamlit dashboard for exploring a synthetic telecom customer panel,
predicting churn risk, explaining predictions with SHAP, and simulating
the ROI of retention discounts.

Reads directly from the processed CSV (data/processed/customer_month_clean.csv)
and the trained model (models/churn_model.joblib); no database is required.

Run from the project root:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit as st

from src.features import build_features

sns.set_style("whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parent
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "customer_month_clean.csv"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "reference" / "industry_benchmarks.csv"
CLEANING_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "cleaning_log.md"

st.set_page_config(page_title="Customer Churn & Retention Dashboard", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_PATH, parse_dates=["activity_month", "signup_month"])


@st.cache_data
def load_benchmarks():
    if BENCHMARK_PATH.exists():
        return pd.read_csv(BENCHMARK_PATH)
    return None


@st.cache_resource
def load_ml_assets():
    """Load the saved model bundle -> (pipeline, feature_names)."""
    path = PROJECT_ROOT / "models" / "churn_model.joblib"
    if not path.exists():
        return None, None
    assets = joblib.load(path)
    return assets["model"], assets["features"]


@st.cache_data
def get_ml_data():
    """Run the full panel through the feature engineering step once."""
    return build_features(load_data().copy())


def churn_rate_by(frame, col, **kwargs):
    """Monthly churn rate (%) per group — simple mean of the 0/1 flag."""
    return 100 * frame.groupby(col, **kwargs)["churn_this_month"].mean()


def show(fig):
    """Render a matplotlib figure and free it (avoids figure build-up on reruns)."""
    st.pyplot(fig)
    plt.close(fig)


st.title("Customer Churn & Retention Dashboard")
st.caption(
    "Exploratory analytics, churn prediction, SHAP explanations and retention ROI simulation "
    "on a synthetic telecom panel (Jan 2021 - Dec 2025) calibrated to TRAI industry benchmarks."
)

if not CLEAN_PATH.exists():
    st.error(
        "Cleaned dataset not found at `data/processed/customer_month_clean.csv`.\n\n"
        "Run the pipeline first, from the project root:\n\n"
        "```\npython3 src/generate_synthetic_data.py\n"
        "python3 src/industry_benchmarks.py\n"
        "python3 src/clean_data.py\n```"
    )
    st.stop()

df = load_data()
benchmarks = load_benchmarks()

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("About this dataset")
    st.markdown(
        "- **Synthetic data** — not real customers\n"
        "- 6,000 customers, monthly panel, Jan 2021 - Dec 2025\n"
        "- Churn modeled month-by-month via a hazard model (not a static label)\n"
        "- Pricing trend calibrated in shape/magnitude to real TRAI industry ARPU data\n"
        "- Full details: `docs/data_dictionary.md`, `docs/business_insights.md`"
    )
    st.divider()
    st.header("Filters")
    min_month, max_month = df["activity_month"].min(), df["activity_month"].max()
    date_range = st.slider(
        "Activity month range",
        min_value=min_month.to_pydatetime(),
        max_value=max_month.to_pydatetime(),
        value=(min_month.to_pydatetime(), max_month.to_pydatetime()),
        format="MMM YYYY",
    )
    segment_filter = st.multiselect(
        "Customer segment", options=sorted(df["customer_segment"].dropna().unique()), default=[]
    )

mask = (df["activity_month"] >= pd.Timestamp(date_range[0])) & (
    df["activity_month"] <= pd.Timestamp(date_range[1])
)
if segment_filter:
    mask &= df["customer_segment"].isin(segment_filter)
fdf = df[mask]

if fdf.empty:
    st.warning("No rows match the current filters. Widen the date range or clear the segment filter.")
    st.stop()

# ---------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------
st.subheader("Key metrics (for the selected filter)")

latest_month = fdf["activity_month"].max()
latest = fdf[fdf["activity_month"] == latest_month]
ever_churn = fdf.groupby("customer_id")["churn_this_month"].max()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Active customers (latest month)", f"{len(latest):,}")
col2.metric("Monthly churn rate", f"{100 * fdf['churn_this_month'].mean():.2f}%")
col3.metric("% customers ever churned", f"{100 * ever_churn.mean():.1f}%")
col4.metric("Monthly revenue run-rate", f"${latest['monthly_charges'].sum():,.0f}")
col5.metric(
    "Revenue lost, latest month",
    f"${latest.loc[latest['churn_this_month'] == 1, 'monthly_charges'].sum():,.0f}",
)

st.divider()

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Trends over time",
        "Churn drivers",
        "Cohort retention",
        "Industry calibration",
        "Data quality",
        "Customer explainer",
    ]
)

with tab1:
    st.markdown("#### Monthly churn rate")
    monthly_churn = churn_rate_by(fdf, "activity_month")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    monthly_churn.plot(ax=ax, color="#C44E52")
    ax.set_ylabel("Monthly churn rate (%)")
    ax.axvline(pd.Timestamp("2024-07-01"), color="gray", linestyle="--", alpha=0.6)
    show(fig)

    st.markdown("#### Monthly revenue")
    monthly_rev = fdf.groupby("activity_month")["monthly_charges"].sum()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    monthly_rev.plot(ax=ax, color="#4C72B0")
    ax.set_ylabel("Total monthly revenue")
    show(fig)

    st.markdown("#### Active customers over time")
    active_trend = fdf.groupby("activity_month").size()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    active_trend.plot(ax=ax, color="#55A868")
    ax.set_ylabel("Active customers")
    show(fig)

with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Churn rate by contract type")
        churn_contract = churn_rate_by(fdf, "contract_type").sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        churn_contract.plot(kind="bar", ax=ax, color="#55A868")
        ax.set_ylabel("Monthly churn rate (%)")
        show(fig)

        st.markdown("#### Churn rate by payment method")
        churn_payment = churn_rate_by(fdf, "payment_method").sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        churn_payment.plot(kind="bar", ax=ax, color="#8172B2")
        ax.set_ylabel("Monthly churn rate (%)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        show(fig)

    with col_b:
        st.markdown("#### Churn rate by tenure group")
        order = ["0-6mo", "7-12mo", "1-2yr", "2-4yr", "4yr+"]
        churn_tenure = churn_rate_by(fdf, "tenure_group", observed=True).reindex(order)
        fig, ax = plt.subplots(figsize=(5, 4))
        churn_tenure.plot(kind="bar", ax=ax, color="#DD8452")
        ax.set_ylabel("Monthly churn rate (%)")
        show(fig)

        st.markdown("#### High-value vs. regular customers")
        churn_value = churn_rate_by(fdf, "is_high_value")
        churn_value.index = churn_value.index.map({0: "Regular value", 1: "High value"})
        fig, ax = plt.subplots(figsize=(5, 4))
        churn_value.plot(kind="bar", ax=ax, color="#C44E52")
        ax.set_ylabel("Monthly churn rate (%)")
        show(fig)

with tab3:
    st.markdown("#### Acquisition cohort retention curves")
    st.warning(
        "Cohorts close to the end of the observation window (2025, and late-2024) "
        "have not had enough calendar time to reach longer tenure milestones. Their "
        "apparent drop-off late in the curve is right-censoring, not necessarily real "
        "elevated churn. Compare cohorts only up to a tenure horizon all of them have reached."
    )
    cohort_start = df[df["tenure_months"] == 0][["customer_id", "cohort_year"]]
    merged = df.drop(columns=["cohort_year"], errors="ignore").merge(cohort_start, on="customer_id")
    fig, ax = plt.subplots(figsize=(9, 5))
    for year, grp in merged.groupby("cohort_year"):
        if year == 2025:
            continue
        curve = grp[grp["tenure_months"] <= 24].groupby("tenure_months")["customer_id"].nunique()
        curve_pct = 100 * curve / curve.iloc[0]
        curve_pct.plot(ax=ax, label=str(year), marker="o", markersize=3)
    ax.set_xlabel("Months since acquisition")
    ax.set_ylabel("Retention (%)")
    ax.legend(title="Cohort year")
    show(fig)

    st.markdown("#### Cohort sizes by acquisition year")
    cohort_sizes = df[df["tenure_months"] == 0].groupby("cohort_year")["customer_id"].nunique()
    st.bar_chart(cohort_sizes)

with tab4:
    st.markdown("#### Synthetic ARPU vs. real industry ARPU (TRAI)")
    st.caption(
        "Trend-shape calibration only — synthetic figures are not forced to match "
        "real rupee values, and no synthetic customer represents a real company's customer."
    )
    if benchmarks is not None:
        synth_arpu = df.groupby(df["activity_month"].dt.year)["monthly_charges"].mean()
        bench = benchmarks[benchmarks["metric"] == "blended_wireless_arpu"].set_index("year")["value"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax2 = ax.twinx()
        synth_arpu.plot(ax=ax, marker="o", color="#4C72B0", label="Synthetic ARPU (project units)")
        bench.plot(ax=ax2, marker="s", color="#DD8452", label="Real industry ARPU (INR, TRAI)")
        ax.set_ylabel("Synthetic ARPU")
        ax2.set_ylabel("Real industry ARPU (INR)")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        show(fig)

        st.markdown("#### Benchmark source data")
        st.dataframe(benchmarks, use_container_width=True)
    else:
        st.info("industry_benchmarks.csv not found — run `python3 src/industry_benchmarks.py` first.")

with tab5:
    st.markdown("#### Cleaning log")
    if CLEANING_LOG_PATH.exists():
        st.markdown(CLEANING_LOG_PATH.read_text())
    else:
        st.info("cleaning_log.md not found — run `python3 src/clean_data.py` first.")

    st.markdown("#### Quick data checks (on current cleaned file)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows", f"{len(df):,}")
    c2.metric("Unique customers", f"{df['customer_id'].nunique():,}")
    c3.metric("Duplicate rows", f"{df.duplicated().sum():,}")


def render_tab6():
    st.markdown("#### Per-customer explanation (SHAP + what-if)")

    pipeline, feature_names = load_ml_assets()
    if pipeline is None:
        st.warning("Model not found. Run `python3 src/churn_model.py` first to train and save it.")
        return

    ml_df = get_ml_data()
    selected_id = st.selectbox("Select a Customer ID to analyze", ml_df["customer_id"].unique())

    hist_raw = df[df["customer_id"] == selected_id].sort_values("activity_month")
    customer_data = (
        ml_df[ml_df["customer_id"] == selected_id].sort_values("activity_month").tail(1)
    )
    st.dataframe(customer_data, use_container_width=True)

    X_cust = customer_data[feature_names]
    preprocessor = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    def dense(x):
        return x.toarray() if hasattr(x, "toarray") else np.asarray(x)

    col_diag, col_presc = st.columns(2)

    # ---------------- Diagnostic: SHAP ----------------
    with col_diag:
        st.markdown("#### Diagnostic: Root-Cause Analysis")
        try:
            names = list(preprocessor.get_feature_names_out())
        except Exception:
            names = None

        background = dense(preprocessor.transform(ml_df[feature_names].sample(100, random_state=0)))
        # HistGradientBoosting isn't supported by TreeExplainer, so explain
        # predict_proba against a small background sample instead.
        explainer = shap.Explainer(
            lambda x: clf.predict_proba(x)[:, 1], background, feature_names=names
        )
        sv = explainer(dense(preprocessor.transform(X_cust)))

        plt.figure(figsize=(8, 4))
        shap.plots.waterfall(sv[0], show=False)
        show(plt.gcf())

    # ---------------- Prescriptive: profit simulator ----------------
    with col_presc:
        st.markdown("#### Prescriptive: Profit Simulator")

        baseline_prob = pipeline.predict_proba(X_cust)[:, 1][0]
        proposed_discount = st.slider(
            "Select Targeted Discount (%)", min_value=0, max_value=30, value=10, step=5
        )

        # Discount the RAW latest row, then rebuild features so anything derived
        # from monthly_charges stays consistent.
        sim_raw = hist_raw.copy()
        sim_raw.loc[sim_raw.index[-1], "monthly_charges"] *= (100 - proposed_discount) / 100
        sim_feats = build_features(sim_raw)
        sim_row = sim_feats.sort_values("activity_month").tail(1)
        new_prob = pipeline.predict_proba(sim_row[feature_names])[:, 1][0]

        base_ltv = 1200  # assumed average lifetime value
        risk_reduction = baseline_prob - new_prob
        cost_of_discount = base_ltv * (proposed_discount / 100)
        net_profit = base_ltv * risk_reduction - cost_of_discount

        c1, c2, c3 = st.columns(3)
        c1.metric("Original Risk", f"{baseline_prob:.1%}")
        c2.metric("Simulated Risk", f"{new_prob:.1%}", delta=f"{-risk_reduction:.1%}", delta_color="inverse")

        if net_profit > 0:
            c3.metric("Net Profit Saved", f"${net_profit:.2f}", delta="Profitable")
            st.success(f"APPROVE: {proposed_discount}% discount is net-positive.")
        else:
            c3.metric("Net Profit Saved", f"${net_profit:.2f}", delta="Loss", delta_color="inverse")
            st.error("REJECT: Discount cost outweighs retention value.")


with tab6:
    render_tab6()

st.divider()
st.caption(
    "Synthetic data for demonstration purposes. See `docs/data_dictionary.md` for "
    "field definitions and leakage guidance."
)