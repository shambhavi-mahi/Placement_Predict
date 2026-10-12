"""Module 5 – Topic 39: Probability Calibration & Statistical Significance.

1. Probability Calibration: Platt Scaling (Sigmoid) and Isotonic Regression to minimize Brier Score.
2. Statistical Significance: McNemar's Test for comparing two classifiers on paired sample predictions.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from scipy.stats import chi2
import config

PLOT_DIR = os.path.join(config.PLOTS_DIR, "Calibration")


def run_calibration_and_statistical_significance(df: pd.DataFrame) -> dict:
    os.makedirs(PLOT_DIR, exist_ok=True)
    target = config.TARGET_COLUMN
    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]

    data = df[feat_cols + [target]].dropna().copy()
    if len(data) > 4000:
        data = data.sample(4000, random_state=config.RANDOM_STATE)

    X = data[feat_cols].values
    y = data[target].astype(int).values

    # Train / Calibrate / Test split (60 / 20 / 20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=config.RANDOM_STATE, stratify=y
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config.RANDOM_STATE, stratify=y_temp
    )

    # 1. Base Uncalibrated Random Forest
    rf_base = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=config.RANDOM_STATE)
    rf_base.fit(X_train, y_train)
    probs_uncal = rf_base.predict_proba(X_test)[:, 1]
    brier_uncal = float(brier_score_loss(y_test, probs_uncal))

    # 2. Calibrated Models (Platt Scaling & Isotonic)
    rf_sigmoid = CalibratedClassifierCV(rf_base, method="sigmoid", cv=3)
    rf_sigmoid.fit(X_train, y_train)
    probs_sigmoid = rf_sigmoid.predict_proba(X_test)[:, 1]
    brier_sigmoid = float(brier_score_loss(y_test, probs_sigmoid))

    rf_isotonic = CalibratedClassifierCV(rf_base, method="isotonic", cv=3)
    rf_isotonic.fit(X_train, y_train)
    probs_isotonic = rf_isotonic.predict_proba(X_test)[:, 1]
    brier_isotonic = float(brier_score_loss(y_test, probs_isotonic))

    # Reliability Diagram
    fig, ax = plt.subplots(figsize=(7.5, 4.3))

    for name, p, color in [
        ("Uncalibrated RF", probs_uncal, "#EF4444"),
        ("Platt / Sigmoid", probs_sigmoid, "#2563EB"),
        ("Isotonic Regression", probs_isotonic, "#10B981")
    ]:
        prob_true, prob_pred = calibration_curve(y_test, p, n_bins=8)
        ax.plot(prob_pred, prob_true, "s-", color=color, linewidth=2, label=f"{name} (Brier={brier_score_loss(y_test, p):.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=9, fontweight="bold")
    ax.set_ylabel("Fraction of Positives (Empirical)", fontsize=9, fontweight="bold")
    ax.set_title("Reliability Diagram (Calibration Curves)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.5)

    fig.tight_layout()
    plot_file = os.path.join(PLOT_DIR, "reliability_curve.png")
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)

    # 3. Statistical Significance: McNemar's Test between Logistic Regression and Random Forest
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=300, random_state=config.RANDOM_STATE).fit(X_train_s, y_train)

    preds_lr = lr.predict(X_test_s)
    preds_rf = rf_base.predict(X_test)

    correct_lr = (preds_lr == y_test)
    correct_rf = (preds_rf == y_test)

    # Contingency Table:
    # b: LR correct, RF wrong
    # c: LR wrong, RF correct
    n_both_correct = int(np.sum(correct_lr & correct_rf))
    b = int(np.sum(correct_lr & (~correct_rf)))
    c = int(np.sum((~correct_lr) & correct_rf))
    n_both_wrong = int(np.sum((~correct_lr) & (~correct_rf)))

    # McNemar's test statistic with continuity correction
    if (b + c) > 0:
        stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        p_val = float(chi2.sf(stat, df=1))
    else:
        stat, p_val = 0.0, 1.0

    is_significant = p_val < 0.05

    return {
        "n_test": len(y_test),
        "brier_uncal": round(brier_uncal, 4),
        "brier_sigmoid": round(brier_sigmoid, 4),
        "brier_isotonic": round(brier_isotonic, 4),
        "brier_improvement": round((brier_uncal - brier_isotonic) / brier_uncal * 100, 1),
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_both_correct": n_both_correct,
        "mcnemar_both_wrong": n_both_wrong,
        "mcnemar_stat": round(stat, 3),
        "mcnemar_pval": round(p_val, 4),
        "is_significant": is_significant,
        "lr_acc": round(float(correct_lr.mean()) * 100, 2),
        "rf_acc": round(float(correct_rf.mean()) * 100, 2)
    }
