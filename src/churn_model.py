"""
Churn Prediction Model — Monthly Hazard Framing
================================================
Predicts `churn_this_month` for each customer-month row from that row's own
state plus past-only history (see features.py).

VALIDATION DESIGN (time-based, no random shuffling — rows from the same
customer and from future months must never inform the past):
  - fit:   2021-01 .. 2024-06   -> hyperparameter candidates trained here
  - val:   2024-07 .. 2024-12   -> pick hyperparameters + decision threshold
  - test:  2025-01 .. 2025-12   -> untouched until the final evaluation
The chosen configuration is refit on fit+val and scored once on test.

CLASS IMBALANCE: only ~0.9% of rows are churn events, so plain accuracy is
misleading (predicting "never churns" scores ~99%). Accuracy is reported
alongside ROC-AUC, PR-AUC, precision/recall/F1 and top-decile lift, and
compared against that majority-class baseline.

Outputs:
  models/churn_model.joblib
  outputs/churn_model_metrics.csv
  outputs/churn_roc_pr_curves.png
  outputs/churn_confusion_matrix.png
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import (
    CATEGORICAL_FEATURES, CLEAN_PATH, FEATURES, MODELS_DIR, NUMERIC_FEATURES, OUT, TARGET,
    build_features, load_panel,
)

VAL_START = pd.Timestamp("2024-07-01")
TEST_START = pd.Timestamp("2025-01-01")
SEED = 42


def make_preprocessor(scale_numeric):
    num = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
         ("num", num, NUMERIC_FEATURES)],
        verbose_feature_names_out=False,
    )


def make_gbm(**params):
    return Pipeline([
        ("prep", make_preprocessor(scale_numeric=False)),
        ("clf", HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **params)),
    ])


def make_logreg():
    return Pipeline([
        ("prep", make_preprocessor(scale_numeric=True)),
        ("clf", LogisticRegression(max_iter=2000, C=0.5)),
    ])


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i])


def top_decile_lift(y, p):
    k = max(1, int(round(0.10 * len(y))))
    top = np.argsort(-p)[:k]
    captured = y[top].sum() / max(1, y.sum())
    return float(y[top].mean() / y.mean()), float(captured)


def evaluate(name, y, p, threshold):
    pred = (p >= threshold).astype(int)
    lift, capture = top_decile_lift(y, p)
    return {
        "model": name,
        "threshold": round(threshold, 4),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "top_decile_lift": lift,
        "top_decile_churn_capture": capture,
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    df = build_features(load_panel())
    fit = df[df["activity_month"] < VAL_START]
    val = df[(df["activity_month"] >= VAL_START) & (df["activity_month"] < TEST_START)]
    train = df[df["activity_month"] < TEST_START]
    test = df[df["activity_month"] >= TEST_START]
    for name, part in [("fit", fit), ("val", val), ("test", test)]:
        print(f"{name:>5}: {len(part):>7} rows | churn rate {100 * part[TARGET].mean():.3f}%")

    # --- 1. Hyperparameter + threshold selection on the validation window ---
    grid = [
        {"learning_rate": lr, "max_iter": it, "max_leaf_nodes": leaves, "min_samples_leaf": 200, "l2_regularization": 1.0}
        for lr, it in [(0.03, 300), (0.05, 200)]
        for leaves in [7, 15]
    ]
    results = []
    for params in grid:
        m = make_gbm(**params).fit(fit[FEATURES], fit[TARGET])
        p = m.predict_proba(val[FEATURES])[:, 1]
        results.append((average_precision_score(val[TARGET], p), roc_auc_score(val[TARGET], p), params, p))
        print(f"  val PR-AUC {results[-1][0]:.4f} | ROC-AUC {results[-1][1]:.4f} | {params}")
    _, _, best_params, best_val_p = max(results, key=lambda r: r[0])
    gbm_threshold = best_f1_threshold(val[TARGET].to_numpy(), best_val_p)
    print(f"Selected params: {best_params} | F1-optimal threshold on val: {gbm_threshold:.4f}")

    lr_fit = make_logreg().fit(fit[FEATURES], fit[TARGET])
    lr_threshold = best_f1_threshold(val[TARGET].to_numpy(), lr_fit.predict_proba(val[FEATURES])[:, 1])

    # --- 2. Refit on all pre-2025 data, evaluate once on the 2025 hold-out ---
    gbm = make_gbm(**best_params).fit(train[FEATURES], train[TARGET])
    logreg = make_logreg().fit(train[FEATURES], train[TARGET])
    y_test = test[TARGET].to_numpy()
    p_gbm = gbm.predict_proba(test[FEATURES])[:, 1]
    p_lr = logreg.predict_proba(test[FEATURES])[:, 1]

    metrics = pd.DataFrame([
        evaluate("Majority class (always 'no churn')", y_test, np.zeros(len(y_test)), 0.5),
        evaluate("Logistic regression", y_test, p_lr, lr_threshold),
        evaluate("Gradient boosting (HistGB)", y_test, p_gbm, gbm_threshold),
    ])
    # top-10% targeting is the operational rule a retention team would actually use
    top10_thr = float(np.quantile(p_gbm, 0.90))
    metrics = pd.concat([metrics, pd.DataFrame([evaluate("Gradient boosting @ top-10% risk", y_test, p_gbm, top10_thr)])])
    metrics.loc[metrics["model"].str.startswith("Majority"),
                ["roc_auc", "pr_auc", "top_decile_lift", "top_decile_churn_capture"]] = [0.5, y_test.mean(), 1.0, 0.10]

    # EVALUATION-ONLY ceiling: the generator's true per-row churn probability (dropped from the clean panel
    # as leakage). Scoring it shows the best ranking any model could achieve on this noisy target.
    raw_path = CLEAN_PATH.parents[1] / "raw" / "customer_month_raw.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path, usecols=["customer_id", "activity_month", "churn_hazard_latent"],
                          parse_dates=["activity_month"]).drop_duplicates(["customer_id", "activity_month"])
        oracle = test[["customer_id", "activity_month"]].merge(raw, on=["customer_id", "activity_month"], how="left")
        p_oracle = oracle["churn_hazard_latent"].to_numpy()
        if not np.isnan(p_oracle).any():
            row = evaluate("Oracle ceiling (true latent hazard, eval only)", y_test, p_oracle, float(np.quantile(p_oracle, 0.90)))
            metrics = pd.concat([metrics, pd.DataFrame([row])])

    metrics.to_csv(OUT / "churn_model_metrics.csv", index=False)
    print("\n=== 2025 hold-out test metrics ===")
    print(metrics.round(4).to_string(index=False))

    # --- 3. Charts ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in [("Gradient boosting", p_gbm), ("Logistic regression", p_lr)]:
        fpr, tpr, _ = roc_curve(y_test, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, p)
        axes[1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y_test, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC — 2025 hold-out")
    axes[1].axhline(y_test.mean(), color="k", linestyle="--", alpha=0.4, label=f"Base rate {y_test.mean():.3%}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall — 2025 hold-out")
    for ax in axes:
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "churn_roc_pr_curves.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, (p_gbm >= gbm_threshold).astype(int)),
                           display_labels=["Stay", "Churn"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Gradient boosting @ threshold {gbm_threshold:.3f}")
    plt.tight_layout(); plt.savefig(OUT / "churn_confusion_matrix.png", dpi=110); plt.close()

    joblib.dump({"model": gbm, "threshold": gbm_threshold, "params": best_params, "features": FEATURES},
                MODELS_DIR / "churn_model.joblib")
    print(f"\nSaved model -> {MODELS_DIR / 'churn_model.joblib'}")


if __name__ == "__main__":
    main()
