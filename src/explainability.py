"""
Model Explainability — SHAP + Permutation Importance
=====================================================
Explains the churn model saved by churn_model.py (run that first).

  - GLOBAL: which features drive churn risk overall (mean |SHAP|), and in
    which direction (beeswarm + dependence plots)
  - LOCAL: why each of the highest-risk active customers is flagged — top
    3 reasons per customer, plus a waterfall chart for the riskiest one
  - VALIDATION of the explanations themselves:
      1. Additivity — SHAP values must sum to the model's raw output
      2. Agreement — SHAP ranking vs. model-agnostic permutation importance
      3. Ground truth — this data is synthetic, so the true churn drivers are
         known from generate_synthetic_data.py; the top SHAP features should
         recover them

One-hot columns are summed back to their original feature (SHAP values are
additive, so this is exact), e.g. contract_type_* -> contract_type.

Outputs:
  outputs/shap_global_importance.png, outputs/shap_beeswarm.png,
  outputs/shap_dependence_tenure.png, outputs/shap_dependence_support_calls.png,
  outputs/shap_waterfall_top_risk_customer.png,
  outputs/feature_importance.csv, outputs/explainability_checks.csv,
  outputs/top_risk_customer_reasons.csv
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from features import CATEGORICAL_FEATURES, FEATURES, MODELS_DIR, OUT, TARGET, build_features, load_panel

TEST_START = pd.Timestamp("2025-01-01")
SHAP_SAMPLE = 8000
SEED = 42

# Drivers that enter the churn hazard in generate_synthetic_data.py (directly, or as the
# engineered/derived form of a direct driver). Everything else is a distractor for the model.
TRUE_DRIVERS = {
    "contract_type", "payment_method", "is_autopay", "plan", "region", "tenure_months",
    "is_new_customer", "months_observed", "support_calls_this_month", "complaint_flag",
    "support_calls_3m", "support_calls_6m_avg", "usage_gb", "price_per_gb",
}


def original_feature(name):
    for col in CATEGORICAL_FEATURES:
        if name.startswith(col + "_"):
            return col
    return name


def aggregate(shap_matrix, encoded_names):
    groups = [original_feature(n) for n in encoded_names]
    return pd.DataFrame(shap_matrix, columns=encoded_names).T.groupby(groups).sum().T[FEATURES]


def main():
    bundle = joblib.load(MODELS_DIR / "churn_model.joblib")
    pipe = bundle["model"]
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    encoded_names = list(prep.get_feature_names_out())

    df = build_features(load_panel())
    test = df[df["activity_month"] >= TEST_START]
    sample = test.sample(n=min(SHAP_SAMPLE, len(test)), random_state=SEED)
    X_enc = prep.transform(sample[FEATURES])

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_enc)
    base = float(np.ravel(explainer.expected_value)[0])

    # --- Check 1: additivity (SHAP sums to the model's log-odds output) ---
    raw_output = clf.decision_function(X_enc)
    additivity_err = float(np.max(np.abs(base + sv.sum(axis=1) - raw_output)))

    # --- Global importance ---
    shap_orig = aggregate(sv, encoded_names)
    shap_imp = shap_orig.abs().mean().sort_values(ascending=False)

    perm = permutation_importance(pipe, test[FEATURES], test[TARGET], scoring="roc_auc",
                                  n_repeats=5, random_state=SEED, n_jobs=-1)
    perm_imp = pd.Series(perm.importances_mean, index=FEATURES)

    importance = pd.DataFrame({
        "mean_abs_shap": shap_imp,
        "permutation_auc_drop": perm_imp.reindex(shap_imp.index),
        "is_true_driver_in_generator": [f in TRUE_DRIVERS for f in shap_imp.index],
    })
    importance["shap_rank"] = range(1, len(importance) + 1)
    importance.to_csv(OUT / "feature_importance.csv", index_label="feature")

    # --- Checks 2 + 3: agreement and ground-truth recovery ---
    rho, _ = spearmanr(importance["mean_abs_shap"], importance["permutation_auc_drop"])
    top5 = list(shap_imp.index[:5])
    top5_overlap = len(set(top5) & set(perm_imp.sort_values(ascending=False).index[:5])) / 5
    top10_true = float(np.mean([f in TRUE_DRIVERS for f in shap_imp.index[:10]]))
    checks = pd.DataFrame([
        {"check": "SHAP additivity: max |base + sum(SHAP) - model log-odds|", "value": additivity_err,
         "pass": additivity_err < 1e-6},
        {"check": "Spearman rank correlation, SHAP vs permutation importance", "value": rho, "pass": rho > 0.6},
        {"check": "Top-5 feature overlap, SHAP vs permutation importance", "value": top5_overlap, "pass": top5_overlap >= 0.6},
        {"check": "Share of SHAP top-10 features that are true generator drivers", "value": top10_true, "pass": top10_true >= 0.8},
    ])
    checks.to_csv(OUT / "explainability_checks.csv", index=False)

    print("=== Global feature importance (2025 test sample) ===")
    print(importance.round(5).to_string())
    print("\n=== Explanation validity checks ===")
    print(checks.to_string(index=False))

    # --- Direction of effect for the categorical drivers ---
    print("\n=== Average SHAP (log-odds) by category — positive = raises churn risk ===")
    for col in ["contract_type", "payment_method", "plan"]:
        print(f"\n{col}:")
        print(shap_orig[col].groupby(sample[col].astype(str).to_numpy()).mean().sort_values(ascending=False).round(4).to_string())

    # --- Charts ---
    fig, ax = plt.subplots(figsize=(8, 6))
    shap_imp.iloc[:15][::-1].plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set(xlabel="Mean |SHAP value| (log-odds of churn)", title="What drives churn risk — global SHAP importance")
    plt.tight_layout(); plt.savefig(OUT / "shap_global_importance.png", dpi=110); plt.close()

    shap.summary_plot(sv, X_enc, feature_names=encoded_names, max_display=15, show=False)
    plt.title("SHAP beeswarm — direction and spread of each feature's effect")
    plt.tight_layout(); plt.savefig(OUT / "shap_beeswarm.png", dpi=110); plt.close()

    for feat, fname in [("tenure_months", "shap_dependence_tenure.png"),
                        ("support_calls_this_month", "shap_dependence_support_calls.png")]:
        shap.dependence_plot(feat, sv, X_enc, feature_names=encoded_names, interaction_index=None, show=False)
        plt.title(f"SHAP dependence — {feat}")
        plt.tight_layout(); plt.savefig(OUT / fname, dpi=110); plt.close()

    # --- Local explanations: highest-risk active customers in the latest month ---
    latest = df[df["activity_month"] == df["activity_month"].max()].copy()
    latest["churn_probability"] = pipe.predict_proba(latest[FEATURES])[:, 1]
    top = latest.nlargest(20, "churn_probability")
    top_enc = prep.transform(top[FEATURES])
    top_shap = aggregate(explainer.shap_values(top_enc), encoded_names)

    reasons = []
    for (_, row), (_, srow) in zip(top.iterrows(), top_shap.iterrows()):
        drivers = srow.sort_values(ascending=False).head(3)
        entry = {"customer_id": row["customer_id"], "churn_probability": round(row["churn_probability"], 4)}
        for i, (feat, val) in enumerate(drivers.items(), 1):
            shown = row[feat] if feat in CATEGORICAL_FEATURES else round(float(row[feat]), 2)
            entry[f"reason_{i}"] = f"{feat} = {shown} (+{val:.3f} log-odds)"
        reasons.append(entry)
    reasons = pd.DataFrame(reasons)
    reasons.to_csv(OUT / "top_risk_customer_reasons.csv", index=False)
    print(f"\n=== Why the top-risk active customers ({latest['activity_month'].max().date()}) are flagged ===")
    print(reasons.head(10).to_string(index=False))

    first = top.iloc[0]
    display = [str(first[f]) if f in CATEGORICAL_FEATURES else round(float(first[f]), 2) for f in FEATURES]
    expl = shap.Explanation(values=top_shap.iloc[0].to_numpy(), base_values=base,
                            data=np.array(display, dtype=object), feature_names=FEATURES)
    shap.plots.waterfall(expl, max_display=12, show=False)
    plt.title(f"Why {first['customer_id']} is high risk (p = {first['churn_probability']:.3f})")
    plt.tight_layout(); plt.savefig(OUT / "shap_waterfall_top_risk_customer.png", dpi=110); plt.close()
    print(f"\nSaved explainability charts and tables to {OUT}")


if __name__ == "__main__":
    main()
