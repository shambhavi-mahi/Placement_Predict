"""Module 5 – Topic 34: ROC and Precision-Recall Curves.

Compares ROC curves (TPR vs FPR) and Precision-Recall curves across classifiers,
highlighting why PR curves are far more informative on imbalanced data.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import config

PLOT_DIR = os.path.join(config.PLOTS_DIR, "Curves")


def run_roc_and_pr_curves(df: pd.DataFrame) -> dict:
    os.makedirs(PLOT_DIR, exist_ok=True)
    target = config.TARGET_COLUMN
    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]

    data = df[feat_cols + [target]].dropna().copy()
    if len(data) > 6000:
        data = data.sample(6000, random_state=config.RANDOM_STATE)

    X = data[feat_cols].values
    y = data[target].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=config.RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=300, random_state=config.RANDOM_STATE).fit(X_train_s, y_train),
            X_test_s,
            "#2563EB"
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=50, max_depth=6, random_state=config.RANDOM_STATE).fit(X_train, y_train),
            X_test,
            "#10B981"
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=4, random_state=config.RANDOM_STATE).fit(X_train, y_train),
            X_test,
            "#F59E0B"
        ),
    }

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.8))

    model_metrics = []
    base_rate = float(y_test.mean())

    for name, (clf, X_eval, color) in models.items():
        probs = clf.predict_proba(X_eval)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = float(roc_auc_score(y_test, probs))
        ax_roc.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = float(average_precision_score(y_test, probs))
        ax_pr.plot(rec, prec, color=color, linewidth=2, label=f"{name} (AP={pr_auc:.3f})")

        model_metrics.append({
            "name": name,
            "roc_auc": round(roc_auc, 3),
            "pr_auc": round(pr_auc, 3),
            "color": color
        })

    # ROC decoration
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", label="Random Chance (0.50)")
    ax_roc.set_title("ROC Curve (TPR vs FPR)", fontsize=11, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=9)
    ax_roc.grid(True, linestyle="--", alpha=0.3)
    ax_roc.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.5)

    # PR decoration
    ax_pr.axhline(base_rate, linestyle="--", color="#94A3B8", label=f"Baseline Rate ({base_rate:.2f})")
    ax_pr.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
    ax_pr.set_xlabel("Recall", fontsize=9)
    ax_pr.set_ylabel("Precision", fontsize=9)
    ax_pr.grid(True, linestyle="--", alpha=0.3)
    ax_pr.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.5)

    fig.tight_layout()
    plot_file = os.path.join(PLOT_DIR, "roc_pr_comparison.png")
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)

    best_roc = max(model_metrics, key=lambda m: m["roc_auc"])
    best_pr = max(model_metrics, key=lambda m: m["pr_auc"])

    return {
        "n_test": len(y_test),
        "base_rate": round(base_rate * 100, 1),
        "best_roc_model": best_roc["name"],
        "best_roc_auc": best_roc["roc_auc"],
        "best_pr_model": best_pr["name"],
        "best_pr_auc": best_pr["pr_auc"],
        "models": model_metrics
    }
