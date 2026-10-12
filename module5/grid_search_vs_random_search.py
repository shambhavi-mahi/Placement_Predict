"""Module 5 – Topic 36: Grid Search vs Random Search.

Compares exhaustive Grid Search against Randomized Search on Random Forest
hyperparameter tuning, measuring time efficiency and performance tradeoff.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import config

PLOT_DIR = os.path.join(config.PLOTS_DIR, "Search")


def run_grid_search_vs_random_search(df: pd.DataFrame) -> dict:
    os.makedirs(PLOT_DIR, exist_ok=True)
    target = config.TARGET_COLUMN
    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]

    data = df[feat_cols + [target]].dropna().copy()
    if len(data) > 2000:
        data = data.sample(2000, random_state=config.RANDOM_STATE)

    X = data[feat_cols].values
    y = data[target].astype(int).values

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    base_rf = RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)

    # 1. Grid Search: 2 x 3 = 6 fits
    grid_params = {
        "n_estimators": [30, 60],
        "max_depth": [4, 8, 12]
    }
    t0 = time.time()
    grid = GridSearchCV(base_rf, grid_params, cv=cv, scoring="accuracy")
    grid.fit(X, y)
    grid_time = time.time() - t0
    grid_score = float(grid.best_score_) * 100

    # 2. Random Search: 4 fits
    rand_params = {
        "n_estimators": randint(20, 80),
        "max_depth": randint(3, 15)
    }
    t0 = time.time()
    rand = RandomizedSearchCV(base_rf, rand_params, n_iter=4, cv=cv, scoring="accuracy", random_state=config.RANDOM_STATE)
    rand.fit(X, y)
    rand_time = time.time() - t0
    rand_score = float(rand.best_score_) * 100

    # Speedup
    speedup = grid_time / rand_time if rand_time > 0 else 1.0

    # Plot Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.2))

    methods = ["Grid Search (12 combos)", "Random Search (6 combos)"]
    scores = [grid_score, rand_score]
    times = [grid_time, rand_time]

    ax1.bar(methods, scores, color=["#2563EB", "#10B981"], width=0.55, alpha=0.9)
    ax1.set_ylim([max(0, min(scores) - 5), 100])
    ax1.set_ylabel("Best CV Accuracy (%)", fontsize=9, fontweight="bold")
    ax1.set_title("Optimization Accuracy", fontsize=10, fontweight="bold")
    for i, v in enumerate(scores):
        ax1.text(i, v + 0.5, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2.bar(methods, times, color=["#F59E0B", "#8B5CF6"], width=0.55, alpha=0.9)
    ax2.set_ylabel("Execution Time (seconds)", fontsize=9, fontweight="bold")
    ax2.set_title(f"Search Runtime ({speedup:.1f}x Speedup)", fontsize=10, fontweight="bold")
    for i, v in enumerate(times):
        ax2.text(i, v + 0.05, f"{v:.2f}s", ha="center", fontweight="bold", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    plot_file = os.path.join(PLOT_DIR, "search_comparison.png")
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)

    return {
        "n_samples": len(data),
        "grid_score": round(grid_score, 2),
        "grid_time": round(grid_time, 2),
        "grid_params": str(grid.best_params_),
        "grid_fits": 6 * 3,
        "rand_score": round(rand_score, 2),
        "rand_time": round(rand_time, 2),
        "rand_params": str(rand.best_params_),
        "rand_fits": 4 * 3,
        "speedup": round(speedup, 1),
        "score_diff": round(rand_score - grid_score, 2)
    }
