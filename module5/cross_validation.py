"""Module 5 – Topic 31: Cross-Validation Strategies.

Compares standard K-Fold, Stratified K-Fold, and ShuffleSplit (Repeated Random Subsampling).
Examines fold variance, confidence intervals, and why stratification is essential.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import config

PLOT_DIR = os.path.join(config.PLOTS_DIR, "CV")


def run_cross_validation(df: pd.DataFrame) -> dict:
    os.makedirs(PLOT_DIR, exist_ok=True)
    target = config.TARGET_COLUMN
    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]

    data = df[feat_cols + [target]].dropna().copy()
    if len(data) > 6000:
        data = data.sample(6000, random_state=config.RANDOM_STATE)

    X = data[feat_cols].values
    y = data[target].astype(int).values

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=config.RANDOM_STATE))

    # 1. Stratified K-Fold (K=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    skf_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy") * 100

    # 2. Standard K-Fold (K=5)
    kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    kf_scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy") * 100

    # 3. ShuffleSplit (5 iterations, 80/20 train/test)
    ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=config.RANDOM_STATE)
    ss_scores = cross_val_score(model, X, y, cv=ss, scoring="accuracy") * 100

    # Per-fold breakdown
    fold_details = []
    for i in range(5):
        fold_details.append({
            "fold": i + 1,
            "skf": round(float(skf_scores[i]), 2),
            "kf": round(float(kf_scores[i]), 2),
            "ss": round(float(ss_scores[i]), 2),
        })

    # Plot fold scores comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_idx = np.arange(1, 6)
    width = 0.25

    ax.bar(x_idx - width, skf_scores, width, label="Stratified K-Fold", color="#2563EB", alpha=0.9)
    ax.bar(x_idx, kf_scores, width, label="Standard K-Fold", color="#F59E0B", alpha=0.9)
    ax.bar(x_idx + width, ss_scores, width, label="ShuffleSplit (80/20)", color="#10B981", alpha=0.9)

    ax.set_ylim([max(0, min(skf_scores.min(), kf_scores.min(), ss_scores.min()) - 4), 100])
    ax.set_xlabel("Fold / Split Index", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_title("Cross-Validation Fold Stability Across 3 Strategies", fontsize=11, fontweight="bold")
    ax.set_xticks(x_idx)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    plot_file = os.path.join(PLOT_DIR, "cv_comparison.png")
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)

    return {
        "n_samples": len(data),
        "n_folds": 5,
        "skf_mean": round(float(skf_scores.mean()), 2),
        "skf_std": round(float(skf_scores.std()), 2),
        "kf_mean": round(float(kf_scores.mean()), 2),
        "kf_std": round(float(kf_scores.std()), 2),
        "ss_mean": round(float(ss_scores.mean()), 2),
        "ss_std": round(float(ss_scores.std()), 2),
        "fold_details": fold_details,
    }
