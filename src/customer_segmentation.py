"""
Customer Segmentation
=====================
Two complementary views:

1. BEHAVIOURAL SEGMENTS (unsupervised, K-Means)
   One row per customer, built from lifetime behaviour in the panel. The
   churn label is NOT a clustering input — it is only used afterwards to
   check that the segments separate churn risk (external validation).
   Charges are expressed as an index vs. that calendar month's median so
   that customers observed in 2021 and 2025 are comparable despite the
   industry-calibrated price rises.

2. RETENTION PRIORITY MATRIX (model-driven)
   Active customers in the latest month are scored with the churn model
   (models/churn_model.joblib, run churn_model.py first) and placed in a
   risk x value grid with a recommended action per quadrant.

Outputs:
  outputs/customer_segments.csv, outputs/segment_profiles.csv,
  outputs/segmentation_quality.csv, outputs/retention_priority.csv,
  outputs/segment_selection.png, outputs/segment_churn_rates.png,
  outputs/retention_priority_matrix.png
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from features import FEATURES, MODELS_DIR, OUT, TARGET, build_features, load_panel

sns.set_style("whitegrid")
SEED = 42
K_RANGE = range(3, 8)
CLUSTER_FEATURES = [
    "tenure_months", "avg_charges_index", "avg_usage_gb", "avg_support_calls",
    "num_addon_services", "contract_level", "is_autopay",
]
CONTRACT_LEVEL = {"Month-To-Month": 0, "One Year": 1, "Two Year": 2}  # spelling as title-cased by clean_data.py


def customer_table(df):
    df = df.copy()
    df["charges_index"] = df["monthly_charges"] / df.groupby("activity_month")["monthly_charges"].transform("median")
    df["contract_level"] = df["contract_type"].astype(str).map(CONTRACT_LEVEL)
    unmapped = df.loc[df["contract_level"].isna(), "contract_type"].unique()
    assert len(unmapped) == 0, f"Unmapped contract types: {list(unmapped)}"
    g = df.groupby("customer_id")
    last = g.tail(1).set_index("customer_id")
    cust = pd.DataFrame({
        "tenure_months": last["tenure_months"],
        "avg_charges_index": g["charges_index"].mean(),
        "avg_usage_gb": g["usage_gb"].mean(),
        "avg_support_calls": g["support_calls_this_month"].mean(),
        "num_addon_services": last["num_addon_services"],
        "contract_level": last["contract_level"],
        "is_autopay": last["is_autopay"],
        "customer_segment": last["customer_segment"].astype(str),
        "plan": last["plan"].astype(str),
        "churned": g[TARGET].max(),
    })
    return cust


def name_segment(row, overall):
    """Readable label from how a cluster's centroid differs from the overall customer base."""
    if row["avg_support_calls"] > overall["avg_support_calls"] * 1.3:
        return "Frustrated / high-complaint"
    if row["avg_usage_gb"] > overall["avg_usage_gb"] * 1.5:
        return "Heavy-usage premium"
    if row["num_addon_services"] > overall["num_addon_services"] * 2:
        return "Add-on rich bundlers"
    if row["contract_level"] < 0.6:
        return "Manual-pay, no contract" if row["is_autopay"] < 0.3 else "Autopay, no contract"
    if row["contract_level"] >= 1.4:
        return "Committed autopay loyalists" if row["is_autopay"] >= 0.8 else "Contracted, manual pay"
    return "Steady mainstream"


def main():
    df = build_features(load_panel())
    cust = customer_table(df)
    X = StandardScaler().fit_transform(cust[CLUSTER_FEATURES])

    # --- 1. Choose k by silhouette ---
    scores = []
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
        scores.append({"k": k, "silhouette": silhouette_score(X, labels),
                       "davies_bouldin": davies_bouldin_score(X, labels),
                       "inertia": KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X).inertia_})
    scores = pd.DataFrame(scores)
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    print(scores.round(4).to_string(index=False))
    print(f"Selected k = {best_k} (highest silhouette)")

    km = KMeans(n_clusters=best_k, n_init=20, random_state=SEED).fit(X)
    cust["cluster"] = km.labels_

    # stability: agreement of labels across different random initialisations
    ari = [adjusted_rand_score(km.labels_, KMeans(n_clusters=best_k, n_init=10, random_state=s).fit_predict(X))
           for s in range(1, 6)]
    quality = pd.DataFrame([{
        "k": best_k,
        "silhouette": silhouette_score(X, km.labels_),
        "davies_bouldin": davies_bouldin_score(X, km.labels_),
        "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
        "stability_mean_ARI_5_seeds": float(np.mean(ari)),
    }])
    quality.to_csv(OUT / "segmentation_quality.csv", index=False)

    # --- 2. Profile + name segments, validate against churn ---
    overall = cust[CLUSTER_FEATURES].mean()
    profiles = cust.groupby("cluster").agg(
        customers=("churned", "size"), churn_rate=("churned", "mean"),
        **{f: (f, "mean") for f in CLUSTER_FEATURES},
    )
    names = profiles.apply(name_segment, axis=1, overall=overall)
    # disambiguate if two clusters land on the same label
    names = names + names.groupby(names).cumcount().map(lambda i: "" if i == 0 else f" ({i + 1})")
    profiles.insert(0, "segment_name", names)
    profiles["share_of_customers"] = profiles["customers"] / profiles["customers"].sum()
    profiles = profiles.sort_values("churn_rate", ascending=False)
    cust["segment_name"] = cust["cluster"].map(names)
    profiles.to_csv(OUT / "segment_profiles.csv")
    cust.to_csv(OUT / "customer_segments.csv")

    print("\n=== Segment quality ===")
    print(quality.round(4).to_string(index=False))
    print("\n=== Segment profiles (sorted by churn rate in window) ===")
    print(profiles.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(scores["k"], scores["silhouette"], marker="o")
    axes[0].axvline(best_k, color="gray", linestyle="--")
    axes[0].set(xlabel="k", ylabel="Silhouette", title="Choosing k (higher is better)")
    axes[1].plot(scores["k"], scores["inertia"], marker="o", color="#DD8452")
    axes[1].set(xlabel="k", ylabel="Inertia", title="Elbow check")
    plt.tight_layout(); plt.savefig(OUT / "segment_selection.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    (100 * profiles.set_index("segment_name")["churn_rate"]).plot(kind="barh", ax=ax, color="#C44E52")
    ax.axvline(100 * cust["churned"].mean(), color="k", linestyle="--", alpha=0.5, label="All customers")
    ax.set(xlabel="% of customers who churned in 2021-2025", ylabel="", title="Churn rate by behavioural segment")
    ax.invert_yaxis(); ax.legend()
    plt.tight_layout(); plt.savefig(OUT / "segment_churn_rates.png", dpi=110); plt.close()

    # --- 3. Retention priority matrix for currently active customers ---
    model_path = MODELS_DIR / "churn_model.joblib"
    if not model_path.exists():
        print(f"\n{model_path} not found — run src/churn_model.py first to build the retention priority matrix.")
        return
    bundle = joblib.load(model_path)
    latest = df[df["activity_month"] == df["activity_month"].max()].copy()
    latest["churn_probability"] = bundle["model"].predict_proba(latest[FEATURES])[:, 1]
    risk_cut = latest["churn_probability"].quantile(0.75)
    latest["risk"] = np.where(latest["churn_probability"] >= risk_cut, "High risk", "Low risk")
    latest["value"] = np.where(latest["is_high_value"] == 1, "High value", "Standard value")
    actions = {
        ("High risk", "High value"): "Priority save: account manager call + loyalty discount / contract upgrade offer",
        ("High risk", "Standard value"): "Automated save: autopay incentive, plan right-sizing, proactive support",
        ("Low risk", "High value"): "Nurture: premium perks, upsell add-ons, referral programme",
        ("Low risk", "Standard value"): "Maintain: low-cost digital engagement",
    }
    latest["recommended_action"] = [actions[(r, v)] for r, v in zip(latest["risk"], latest["value"])]
    latest = latest.merge(cust[["segment_name"]], left_on="customer_id", right_index=True, how="left")
    priority = latest[["customer_id", "churn_probability", "risk", "value", "monthly_charges",
                       "segment_name", "contract_type", "payment_method", "tenure_months",
                       "recommended_action"]].sort_values("churn_probability", ascending=False)
    priority.to_csv(OUT / "retention_priority.csv", index=False)

    matrix = priority.groupby(["risk", "value"]).agg(
        customers=("customer_id", "size"), avg_churn_prob=("churn_probability", "mean"),
        monthly_revenue=("monthly_charges", "sum"))
    matrix["expected_monthly_revenue_at_risk"] = (
        priority.assign(r=priority["churn_probability"] * priority["monthly_charges"])
        .groupby(["risk", "value"])["r"].sum())
    print(f"\n=== Retention priority matrix — active customers in {latest['activity_month'].max().date()} ===")
    print(matrix.round(3).to_string())

    pivot = matrix["customers"].unstack("value")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax)
    ax.set(title="Retention priority matrix (active customers)", xlabel="", ylabel="")
    plt.tight_layout(); plt.savefig(OUT / "retention_priority_matrix.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
