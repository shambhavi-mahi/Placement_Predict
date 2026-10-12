"""Module 6 – Probability Calibration.

Transforms raw classifier confidence scores into true posterior probabilities.
Compares Uncalibrated models (e.g. Random Forest) against:
1. Platt Scaling (Parametric Sigmoid)
2. Isotonic Regression (Non-parametric Piecewise-Constant)

Computes Brier Score, Log Loss, and Expected Calibration Error (ECE),
and visualizes Reliability Diagrams and probability distributions.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import config

PLOT_DIR = os.path.join(config.PLOTS_DIR, "ProbabilityCalibration")


def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculates Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * np.abs(bin_acc - bin_conf)
    return float(ece) * 100


def run_probability_calibration(df: pd.DataFrame) -> dict:
    os.makedirs(PLOT_DIR, exist_ok=True)
    target = config.TARGET_COLUMN
    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]

    data = df[feat_cols + [target]].dropna().copy()
    if len(data) > 5000:
        data = data.sample(5000, random_state=config.RANDOM_STATE)

    X = data[feat_cols].values
    y = data[target].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=config.RANDOM_STATE, stratify=y
    )

    # 1. Base Uncalibrated Random Forest
    rf_base = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=config.RANDOM_STATE)
    rf_base.fit(X_train, y_train)
    probs_uncal = rf_base.predict_proba(X_test)[:, 1]

    # 2. Platt Scaling (Sigmoid)
    rf_sigmoid = CalibratedClassifierCV(rf_base, method="sigmoid", cv=3)
    rf_sigmoid.fit(X_train, y_train)
    probs_sigmoid = rf_sigmoid.predict_proba(X_test)[:, 1]

    # 3. Isotonic Regression
    rf_isotonic = CalibratedClassifierCV(rf_base, method="isotonic", cv=3)
    rf_isotonic.fit(X_train, y_train)
    probs_isotonic = rf_isotonic.predict_proba(X_test)[:, 1]

    models = [
        {"name": "Uncalibrated (RF Base)", "probs": probs_uncal, "color": "#EF4444", "style": "s--"},
        {"name": "Platt Scaling (Sigmoid)", "probs": probs_sigmoid, "color": "#2563EB", "style": "o-"},
        {"name": "Isotonic Regression", "probs": probs_isotonic, "color": "#10B981", "style": "^-"},
    ]

    metrics = []
    for m in models:
        p = m["probs"]
        bs = float(brier_score_loss(y_test, p))
        ll = float(log_loss(y_test, p))
        ece = calculate_ece(y_test, p, n_bins=10)
        metrics.append({
            "name": m["name"],
            "color": m["color"],
            "brier": round(bs, 4),
            "log_loss": round(ll, 4),
            "ece": round(ece, 2),
        })

    # Plot 1: Reliability Curve & Plot 2: Probability Distribution Histogram
    fig, (ax_rel, ax_hist) = plt.subplots(1, 2, figsize=(11, 4.6))

    for m in models:
        prob_true, prob_pred = calibration_curve(y_test, m["probs"], n_bins=8)
        ax_rel.plot(prob_pred, prob_true, m["style"], color=m["color"], linewidth=2, label=m["name"])

    ax_rel.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Perfect Calibration")
    ax_rel.set_title("Reliability Diagram (Calibration Curve)", fontsize=10, fontweight="bold")
    ax_rel.set_xlabel("Mean Predicted Probability", fontsize=9, fontweight="bold")
    ax_rel.set_ylabel("True Empirical Fraction (Positive)", fontsize=9, fontweight="bold")
    ax_rel.grid(True, linestyle="--", alpha=0.3)
    ax_rel.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8)

    # Histogram of predicted probabilities
    ax_hist.hist(probs_uncal, bins=20, alpha=0.45, color="#EF4444", label="Uncalibrated")
    ax_hist.hist(probs_isotonic, bins=20, alpha=0.45, color="#10B981", label="Isotonic")
    ax_hist.set_title("Predicted Probability Distribution", fontsize=10, fontweight="bold")
    ax_hist.set_xlabel("Predicted Probability", fontsize=9, fontweight="bold")
    ax_hist.set_ylabel("Number of Samples", fontsize=9, fontweight="bold")
    ax_hist.grid(True, linestyle="--", alpha=0.3)
    ax_hist.legend(loc="upper center", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8)

    fig.tight_layout()
    plot_file = os.path.join(PLOT_DIR, "calibration_plots.png")
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)

    brier_base = metrics[0]["brier"]
    brier_best = min(m["brier"] for m in metrics[1:])
    improvement = round((brier_base - brier_best) / brier_base * 100, 1)

    return {
        "n_samples": len(data),
        "n_test": len(y_test),
        "brier_uncal": metrics[0]["brier"],
        "brier_sigmoid": metrics[1]["brier"],
        "brier_isotonic": metrics[2]["brier"],
        "ece_uncal": metrics[0]["ece"],
        "ece_isotonic": metrics[2]["ece"],
        "brier_improvement": improvement,
        "metrics_table": metrics,
        "recommended_method": "Isotonic Regression" if len(data) >= 2000 else "Platt Scaling (Sigmoid)"
    }
