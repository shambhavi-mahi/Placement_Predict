"""Module 5 – Topic 37: Learning and Validation Curves.

Diagnoses High Bias vs High Variance by plotting:
1. Learning Curve (Training sample size vs performance)
2. Validation Curve (Model complexity / tree depth vs performance)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import config

PLOT_DIR = os.path.join(config.PLOTS_DIR, "Curves")


def run_learning_and_validation_curves(df: pd.DataFrame) -> dict:
    os.makedirs(PLOT_DIR, exist_ok=True)
    target = config.TARGET_COLUMN
    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]

    data = df[feat_cols + [target]].dropna().copy()
    if len(data) > 4000:
        data = data.sample(4000, random_state=config.RANDOM_STATE)

    X = data[feat_cols].values
    y = data[target].astype(int).values

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)

    # 1. Learning Curve
    train_sizes, train_scores_lc, test_scores_lc = learning_curve(
        DecisionTreeClassifier(max_depth=5, random_state=config.RANDOM_STATE),
        X, y, cv=cv, scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5),
        random_state=config.RANDOM_STATE
    )

    lc_train_mean = np.mean(train_scores_lc, axis=1) * 100
    lc_test_mean = np.mean(test_scores_lc, axis=1) * 100

    # 2. Validation Curve over max_depth
    param_range = np.array([1, 2, 3, 5, 7, 10, 15])
    train_scores_vc, test_scores_vc = validation_curve(
        DecisionTreeClassifier(random_state=config.RANDOM_STATE),
        X, y, param_name="max_depth", param_range=param_range,
        cv=cv, scoring="accuracy"
    )

    vc_train_mean = np.mean(train_scores_vc, axis=1) * 100
    vc_test_mean = np.mean(test_scores_vc, axis=1) * 100

    best_depth_idx = int(np.argmax(vc_test_mean))
    optimal_depth = int(param_range[best_depth_idx])
    optimal_score = float(vc_test_mean[best_depth_idx])

    # Plot Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3))

    # Learning Curve
    ax1.plot(train_sizes, lc_train_mean, "o-", color="#2563EB", label="Training Score")
    ax1.plot(train_sizes, lc_test_mean, "s-", color="#10B981", label="Validation Score")
    ax1.fill_between(train_sizes, lc_test_mean, lc_train_mean, alpha=0.1, color="#64748B")
    ax1.set_title("Learning Curve (Bias vs Variance)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Training Samples (N)", fontsize=9)
    ax1.set_ylabel("Accuracy (%)", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.5)

    # Validation Curve
    ax2.plot(param_range, vc_train_mean, "o-", color="#2563EB", label="Training Score")
    ax2.plot(param_range, vc_test_mean, "s-", color="#F59E0B", label="Validation Score")
    ax2.axvline(optimal_depth, linestyle="--", color="#DC2626", label=f"Optimal Depth={optimal_depth}")
    ax2.set_title("Validation Curve (Model Capacity)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Tree Max Depth", fontsize=9)
    ax2.set_ylabel("Accuracy (%)", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.5)

    fig.tight_layout()
    plot_file = os.path.join(PLOT_DIR, "learning_validation_curves.png")
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)

    final_gap = round(float(lc_train_mean[-1] - lc_test_mean[-1]), 2)
    diagnosis = "Low Variance (converged well)" if final_gap < 5 else "High Variance / Slight Overfitting"

    return {
        "n_samples": len(data),
        "optimal_depth": optimal_depth,
        "optimal_val_acc": round(optimal_score, 2),
        "generalization_gap": final_gap,
        "diagnosis": diagnosis,
        "max_train_samples": int(train_sizes[-1])
    }
