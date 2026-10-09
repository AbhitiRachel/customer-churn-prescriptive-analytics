"""
Longitudinal Synthetic Customer-Month Panel Generator
======================================================
SYNTHETIC DATA. Generates a monthly customer-level panel (2021-01 to 2025-12)
for ~6,000 customers, using a MONTHLY HAZARD model: each active customer has
a churn probability recomputed every month from their *current* state, and
churn occurs in whichever month that draw first fires. This avoids the
leakage problem of a static label — no customer-month row contains
information from after that customer's actual churn month.

CALIBRATION NOTE (industry benchmarks, NOT simulated companies):
Real Indian telecom industry data (TRAI regulatory reports; see
data/reference/industry_benchmarks.csv and industry_benchmarks.py for exact
figures and sources) shows blended monthly ARPU rose roughly 55-60% from
2021 to 2025, driven largely by discrete tariff hikes (notably July 2024).
We calibrate ONLY the shape/magnitude of our synthetic pricing trend to this
real trajectory — a step-up pricing multiplier applied economy-wide in this
synthetic dataset, mirroring the timing and rough scale of real industry
tariff hikes. No synthetic customer represents an actual customer of any
real company; only the aggregate trend shape is benchmarked.

BUSINESS RULES PRESERVED FROM v1 (still drive churn hazard each month):
  - Month-to-month contracts churn more than 1yr/2yr contracts
  - Higher price relative to usage (value perception) raises churn hazard
  - More support complaints raise churn hazard
  - Manual payment methods (electronic check) raise churn hazard vs autopay
  - Higher usage/engagement lowers churn hazard
  - Tenure has a "new-customer fragility" effect (elevated hazard in first
    6 months) followed by a loyalty effect (hazard decays with tenure)
  - A temporary hazard bump follows the synthetic tariff-hike month (price
    shock effect), consistent with real-world churn spikes after price rises

NEW IN THIS VERSION: time-varying fields (contract_type, payment_method,
plan, usage, complaints) can change month to month via small transition
probabilities; cohort fields (signup_month, cohort_year, cohort_month)
support acquisition-cohort retention analysis.

Deliberately injected data-quality problems (missing values, duplicates,
invalid values, inconsistent categories, outliers) are preserved — see
inject_messiness().
"""

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime as dt

RNG = np.random.default_rng(42)

N_CUSTOMERS = 6000
SIGNUP_START = dt.date(2016, 1, 1)   # customers can pre-date the observation window (real tenure history)
SIGNUP_END = dt.date(2025, 11, 1)
WINDOW_START = dt.date(2021, 1, 1)   # <-- observation window is exactly Jan 2021 - Dec 2025
WINDOW_END = dt.date(2025, 12, 1)

REGIONS = ["North", "South", "East", "West"]
REGION_HAZARD_ADJ = {"North": -0.002, "South": 0.004, "East": 0.0, "West": 0.001}

CONTRACTS = ["Month-to-month", "One year", "Two year"]
CONTRACT_PROBS_AT_SIGNUP = [0.55, 0.25, 0.20]
CONTRACT_HAZARD_ADJ = {"Month-to-month": 0.013, "One year": 0.003, "Two year": -0.005}
CONTRACT_UPGRADE_PROB = {"Month-to-month": 0.02, "One year": 0.01, "Two year": 0.0}
CONTRACT_UPGRADE_TO = {"Month-to-month": "One year", "One year": "Two year"}

PAYMENT_METHODS = ["Electronic check", "Mailed check", "Bank transfer (auto)", "Credit card (auto)"]
PAYMENT_PROBS_AT_SIGNUP = [0.35, 0.20, 0.22, 0.23]
PAYMENT_HAZARD_ADJ = {
    "Electronic check": 0.007, "Mailed check": 0.002,
    "Bank transfer (auto)": -0.004, "Credit card (auto)": -0.004,
}
PAYMENT_AUTOPAY_MIGRATION_PROB = 0.012

PLANS = ["Basic", "Standard", "Premium"]
PLAN_PROBS_AT_SIGNUP = [0.40, 0.38, 0.22]
PLAN_BASE_PRICE = {"Basic": 29.99, "Standard": 59.99, "Premium": 94.99}
PLAN_HAZARD_ADJ = {"Basic": 0.002, "Standard": 0.0, "Premium": -0.002}
PLAN_USAGE_MEAN = {"Basic": 20, "Standard": 55, "Premium": 110}

SEGMENTS = ["Consumer", "SMB", "Enterprise"]
SEGMENT_PROBS = [0.70, 0.24, 0.06]
SEGMENT_ADDON_MEAN = {"Consumer": 0.8, "SMB": 1.8, "Enterprise": 2.6}

TARIFF_HIKE_MONTH = dt.date(2024, 7, 1)


def price_multiplier(month: dt.date) -> float:
    if month < dt.date(2022, 1, 1):
        return 1.00
    elif month < dt.date(2023, 1, 1):
        return 1.05
    elif month < TARIFF_HIKE_MONTH:
        return 1.15
    elif month < dt.date(2025, 1, 1):
        return 1.40
    else:
        return 1.55


def months_between(d1: dt.date, d2: dt.date) -> int:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def month_add(d: dt.date, n: int) -> dt.date:
    return d + relativedelta(months=n)


def sample_signup_dates(n):
    """
    Signups spread 2016-2025. Weighted so a majority of customers already
    existed before the observation window starts (giving realistic long
    tenure histories), while still generating enough new signups *within*
    2021-2025 to support genuine acquisition-cohort comparisons.
    """
    total_months = months_between(SIGNUP_START, SIGNUP_END) + 1
    month_dates = [month_add(SIGNUP_START, m) for m in range(total_months)]
    weights = np.array([6.0 if d < WINDOW_START else 1.0 for d in month_dates])
    weights = weights / weights.sum()
    chosen = RNG.choice(total_months, size=n, p=weights)
    days_jitter = RNG.integers(0, 28, size=n)
    return [month_add(SIGNUP_START, int(m)) + dt.timedelta(days=int(d)) for m, d in zip(chosen, days_jitter)]


def generate_panel():
    signup_dates = sample_signup_dates(N_CUSTOMERS)
    customer_ids = [f"CUST-{100000+i}" for i in range(N_CUSTOMERS)]

    region = RNG.choice(REGIONS, size=N_CUSTOMERS, p=[0.28, 0.24, 0.26, 0.22])
    segment = RNG.choice(SEGMENTS, size=N_CUSTOMERS, p=SEGMENT_PROBS)
    age = np.clip(RNG.normal(42, 14, size=N_CUSTOMERS).astype(int), 18, 85)
    init_contract = RNG.choice(CONTRACTS, size=N_CUSTOMERS, p=CONTRACT_PROBS_AT_SIGNUP)
    init_payment = RNG.choice(PAYMENT_METHODS, size=N_CUSTOMERS, p=PAYMENT_PROBS_AT_SIGNUP)
    init_plan = RNG.choice(PLANS, size=N_CUSTOMERS, p=PLAN_PROBS_AT_SIGNUP)
    addon_mean = np.array([SEGMENT_ADDON_MEAN[s] for s in segment])
    init_addons = RNG.poisson(addon_mean).clip(0, 5)

    rows = []
    for i in range(N_CUSTOMERS):
        cid = customer_ids[i]
        signup = signup_dates[i]
        signup_month0 = dt.date(signup.year, signup.month, 1)
        cohort_year, cohort_month = signup_month0.year, signup_month0.month

        contract = init_contract[i]
        payment = init_payment[i]
        plan = init_plan[i]
        num_addons = int(init_addons[i])
        usage = max(1.0, RNG.normal(PLAN_USAGE_MEAN[plan], PLAN_USAGE_MEAN[plan] * 0.3))
        complaints_recent = 0.0

        month = max(signup_month0, WINDOW_START)

        while month <= WINDOW_END:
            tenure_months = months_between(signup_month0, month)

            if RNG.random() < CONTRACT_UPGRADE_PROB.get(contract, 0):
                contract = CONTRACT_UPGRADE_TO[contract]
            if payment not in ("Bank transfer (auto)", "Credit card (auto)") and RNG.random() < PAYMENT_AUTOPAY_MIGRATION_PROB:
                payment = RNG.choice(["Bank transfer (auto)", "Credit card (auto)"])
            if RNG.random() < 0.01:
                plan = RNG.choice(PLANS, p=PLAN_PROBS_AT_SIGNUP)

            seasonal_mult = 1.15 if month.month in (11, 12) else (0.92 if month.month == 1 else 1.0)
            usage = max(1.0, 0.85 * usage + 0.15 * RNG.normal(PLAN_USAGE_MEAN[plan], PLAN_USAGE_MEAN[plan] * 0.2))
            usage_this_month = max(0.5, usage * seasonal_mult * RNG.normal(1.0, 0.05))

            base_price = PLAN_BASE_PRICE[plan] + num_addons * 7.5
            contract_discount = {"Month-to-month": 0, "One year": 0.05, "Two year": 0.10}[contract]
            monthly_charges = base_price * (1 - contract_discount) * price_multiplier(month)
            monthly_charges = max(10.0, RNG.normal(monthly_charges, monthly_charges * 0.04))

            price_per_usage = monthly_charges / (usage_this_month + 1)
            complaint_lambda = np.clip(0.15 + price_per_usage * 0.10 + 0.3 * (complaints_recent > 2), 0.05, 3)
            support_calls_this_month = RNG.poisson(complaint_lambda)
            complaints_recent = 0.7 * complaints_recent + 0.3 * support_calls_this_month

            tenure_effect = 0.008 if tenure_months < 6 else max(0.0015, 0.007 - 0.00012 * tenure_months)
            price_shock = 0.008 if TARIFF_HIKE_MONTH <= month < month_add(TARIFF_HIKE_MONTH, 3) and contract == "Month-to-month" else 0.0

            hazard = (
                tenure_effect
                + CONTRACT_HAZARD_ADJ[contract]
                + PAYMENT_HAZARD_ADJ[payment]
                + PLAN_HAZARD_ADJ[plan]
                + REGION_HAZARD_ADJ[region[i]]
                + 0.006 * support_calls_this_month
                - 0.00008 * (usage_this_month - 50)
                + price_shock
                + RNG.normal(0, 0.004)
            )
            hazard = float(np.clip(hazard, 0.001, 0.35))
            churn_this_month = int(RNG.random() < hazard)

            rows.append({
                "customer_id": cid,
                "activity_month": month,
                "tenure_months": tenure_months,
                "region": region[i],
                "customer_segment": segment[i],
                "age": age[i],
                "contract_type": contract,
                "payment_method": payment,
                "plan": plan,
                "monthly_charges": round(monthly_charges, 2),
                "usage_gb": round(usage_this_month, 1),
                "num_addon_services": num_addons,
                "support_calls_this_month": support_calls_this_month,
                "is_active": 1,
                "churn_this_month": churn_this_month,
                "signup_month": signup_month0,
                "cohort_year": cohort_year,
                "cohort_month": cohort_month,
                "churn_hazard_latent": round(hazard, 5),
            })

            if churn_this_month:
                break
            month = month_add(month, 1)

    panel = pd.DataFrame(rows)
    panel = inject_messiness(panel)
    return panel


def inject_messiness(df):
    n = len(df)
    df = df.copy()

    for col, frac in [("age", 0.015), ("usage_gb", 0.012), ("payment_method", 0.008)]:
        idx = RNG.choice(n, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    messy_region_map = {"North": ["North", "north ", "NORTH"], "South": ["South", "south"]}
    idx = RNG.choice(n, size=int(n * 0.015), replace=False)
    for i in idx:
        r = df.loc[i, "region"]
        if r in messy_region_map:
            df.loc[i, "region"] = RNG.choice(messy_region_map[r])

    idx = RNG.choice(n, size=30, replace=False)
    df.loc[idx, "monthly_charges"] = df.loc[idx, "monthly_charges"] * RNG.uniform(3, 5, size=30)
    idx2 = RNG.choice(n, size=10, replace=False)
    df.loc[idx2, "age"] = 0

    dup_sample = df.sample(n=int(n * 0.006), random_state=1)
    df = pd.concat([df, dup_sample], ignore_index=True)

    return df.sample(frac=1, random_state=7).reset_index(drop=True)


if __name__ == "__main__":
    panel = generate_panel()
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    panel.to_csv(raw_dir / "customer_month_raw.csv", index=False)

    n_customers = panel["customer_id"].nunique()
    n_churn_events = panel["churn_this_month"].sum()
    print(f"customer_month_raw: {panel.shape}")
    print(f"unique customers: {n_customers}")
    print(f"churn events (rows where churn_this_month=1): {n_churn_events}")
    print(f"customers who ever churn within window: {panel.groupby('customer_id')['churn_this_month'].max().mean():.3f}")
    print(f"avg monthly hazard: {panel['churn_hazard_latent'].mean():.4f}")
    print(f"date range: {panel['activity_month'].min()} to {panel['activity_month'].max()}")
