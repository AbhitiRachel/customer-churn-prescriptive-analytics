"""
Shared Feature Engineering — Modelling Scope
=============================================
Builds the model-ready feature table from the cleaned customer-month panel
(data/processed/customer_month_clean.csv). Used by churn_model.py,
customer_segmentation.py and explainability.py so all three see exactly the
same features.

LEAKAGE GUARD: every feature here uses only that row's month or EARLIER
months for the same customer (groupby + shift / rolling over past rows).
Nothing is joined back from a future month — see docs/data_dictionary.md.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "customer_month_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUT = PROJECT_ROOT / "outputs"

TARGET = "churn_this_month"

CATEGORICAL_FEATURES = ["region", "customer_segment", "contract_type", "payment_method", "plan"]
NUMERIC_FEATURES = [
    "tenure_months", "age", "monthly_charges", "usage_gb", "num_addon_services",
    "support_calls_this_month", "is_high_value", "complaint_flag",
    # engineered (past-only)
    "is_new_customer", "price_per_gb", "is_autopay", "month_of_year",
    "support_calls_3m", "support_calls_6m_avg", "usage_change_pct",
    "usage_vs_own_avg", "charges_change_pct", "months_observed",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_panel():
    df = pd.read_csv(CLEAN_PATH, parse_dates=["activity_month", "signup_month"])
    dupes = df.duplicated(["customer_id", "activity_month"]).sum()
    assert dupes == 0, f"{dupes} duplicate customer-month keys in clean panel"
    return df.sort_values(["customer_id", "activity_month"]).reset_index(drop=True)


def build_features(df):
    df = df.sort_values(["customer_id", "activity_month"]).copy()
    g = df.groupby("customer_id", sort=False)

    df["is_new_customer"] = (df["tenure_months"] < 6).astype(int)
    df["price_per_gb"] = df["monthly_charges"] / (df["usage_gb"] + 1)
    df["is_autopay"] = df["payment_method"].str.contains("Auto", na=False).astype(int)
    df["month_of_year"] = df["activity_month"].dt.month

    # rolling windows over the current + previous rows only
    df["support_calls_3m"] = g["support_calls_this_month"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["support_calls_6m_avg"] = g["support_calls_this_month"].transform(lambda s: s.rolling(6, min_periods=1).mean())

    prev_usage = g["usage_gb"].shift(1)
    prev_charges = g["monthly_charges"].shift(1)
    df["usage_change_pct"] = ((df["usage_gb"] - prev_usage) / prev_usage).fillna(0.0)
    df["charges_change_pct"] = ((df["monthly_charges"] - prev_charges) / prev_charges).fillna(0.0)

    # usage relative to this customer's own history up to and including this month (expanding mean)
    own_avg = g["usage_gb"].transform(lambda s: s.expanding().mean())
    df["usage_vs_own_avg"] = df["usage_gb"] / own_avg
    df["months_observed"] = g.cumcount() + 1

    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df
