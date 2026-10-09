"""
Data Preparation Pipeline — Longitudinal Version
=================================================
Cleans the raw customer-month panel. Same philosophy as v1: minimal
transformation of the master data, derived features added alongside,
every decision logged.

LEAKAGE GUARD (critical for this version): `churn_hazard_latent` is a
generator-internal column that encodes the exact probability used to draw
that month's churn outcome — trivially predictive, and would leak the
target if handed to Member 2. It is dropped here, same as v1's
`churn_probability_latent`.

No other column in this cleaned table is calculated from a future month —
each row's fields describe only that customer's state in that month.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "customer_month_raw.csv"
OUT_MASTER = PROJECT_ROOT / "data" / "processed" / "customer_month_clean.csv"
OUT_LOG = PROJECT_ROOT / "data" / "processed" / "cleaning_log.md"


log_lines = ["# Data Cleaning Log — Longitudinal Panel\n"]


def log(msg):
    log_lines.append(f"- {msg}")
    print(msg)


def main():
    OUT_MASTER.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_PATH, parse_dates=["activity_month", "signup_month"])
    n0 = len(df)
    log(f"Loaded raw panel: {n0} rows, {df.shape[1]} columns, "
        f"{df['customer_id'].nunique()} unique customers")

    # 1. Exact duplicate rows
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    log(f"Removed {dupes} exact duplicate rows -> {len(df)} rows remain")

    # 2. Standardize categorical text
    for col in ["region", "customer_segment", "contract_type", "payment_method", "plan"]:
        df[col] = df[col].astype(str).str.strip().str.title()
        df[col] = df[col].replace("Nan", np.nan)  # astype(str) turns real NaNs into the string "Nan" above; restore them so step 4's fillna can catch them
    log("Standardized categorical text fields (trimmed whitespace, title-cased)")

    # 3. Invalid values
    invalid_age = (df["age"] <= 0) | (df["age"] > 100)
    log(f"Found {invalid_age.sum()} invalid age rows (<=0 or >100) -> set to NaN, imputed by segment median")
    df.loc[invalid_age, "age"] = np.nan

    # 4. Missing values
    miss_before = df.isna().sum()
    df["age"] = df["age"].fillna(df.groupby("customer_segment")["age"].transform("median"))
    df["usage_gb"] = df["usage_gb"].fillna(df.groupby("plan")["usage_gb"].transform("median"))
    df["payment_method"] = df["payment_method"].fillna("Unknown")
    for col in miss_before[miss_before > 0].index:
        log(f"Imputed {miss_before[col]} missing values in '{col}'")

    # 5. Outlier handling on monthly_charges (cap using IQR within each row's plan, since price legitimately
    #    varies by plan and calendar-time tariff hikes — capping globally would wrongly flag late-window prices)
    df["monthly_charges_outlier_flag"] = 0
    for plan_name, grp in df.groupby("plan"):
        q1, q3 = grp["monthly_charges"].quantile([0.25, 0.75])
        upper = q3 + 3 * (q3 - q1)
        outlier_idx = grp.index[grp["monthly_charges"] > upper]
        df.loc[outlier_idx, "monthly_charges_outlier_flag"] = 1
        df.loc[outlier_idx, "monthly_charges"] = upper
    log(f"Flagged and capped {df['monthly_charges_outlier_flag'].sum()} monthly_charges outliers (per-plan 3x IQR rule)")

    # 6. Drop leakage-risk generator-internal column
    if "churn_hazard_latent" in df.columns:
        df = df.drop(columns=["churn_hazard_latent"])
        log("Dropped 'churn_hazard_latent' — generator-internal churn probability, would leak the target")

    # 7. Derived features
    df["tenure_years"] = (df["tenure_months"] / 12).round(2)
    df["tenure_group"] = pd.cut(
        df["tenure_months"], bins=[-1, 6, 12, 24, 48, 999],
        labels=["0-6mo", "7-12mo", "1-2yr", "2-4yr", "4yr+"]
    )
    df["is_high_value"] = (df["monthly_charges"] > df.groupby("activity_month")["monthly_charges"].transform("quantile", 0.75)).astype(int)
    df["complaint_flag"] = (df["support_calls_this_month"] >= 2).astype(int)
    df["cohort_quarter"] = df["cohort_year"].astype(str) + "-Q" + ((df["cohort_month"] - 1) // 3 + 1).astype(str)
    log("Added derived features: tenure_years, tenure_group, is_high_value (relative to that month's cross-section), "
        "complaint_flag, cohort_quarter")

    # 8. Sanity checks
    assert (df["monthly_charges"] > 0).all()
    assert df["age"].between(1, 100).all()
    assert df["activity_month"].min() >= pd.Timestamp("2021-01-01")
    assert df["activity_month"].max() <= pd.Timestamp("2025-12-01")
    # no customer-month row should exist after that customer's own churn month
    churn_months = df[df["churn_this_month"] == 1].groupby("customer_id")["activity_month"].min()
    for cid, cmonth in churn_months.items():
        after = df[(df["customer_id"] == cid) & (df["activity_month"] > cmonth)]
        assert len(after) == 0, f"Leakage: rows exist after churn month for {cid}"
    log("Passed sanity checks: positive charges, valid ages, window bounds respected, "
        "no rows exist after any customer's churn month (leakage guard)")

    churn_events = int(df["churn_this_month"].sum())
    ever_churn_rate = df.groupby("customer_id")["churn_this_month"].max().mean()
    log(f"Churn events in window: {churn_events} | share of customers who ever churn in window: {ever_churn_rate:.3f}")

    df.to_csv(OUT_MASTER, index=False)
    with open(OUT_LOG, "w") as f:
        f.write("\n".join(log_lines))
    log(f"Saved clean panel -> {OUT_MASTER} ({df.shape[0]} rows, {df.shape[1]} cols)")


if __name__ == "__main__":
    main()