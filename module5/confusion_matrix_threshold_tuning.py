"""Module 5 – Topic 33: Confusion Matrix & Decision Threshold Tuning.

Analyzes how varying the classification probability threshold shifts
TP, FP, TN, and FN, and finds the threshold that maximizes business utility.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import config

PLOT_DIR = os.path.join(config.PLOTS_DIR, "Curves")


def run_confusion_matrix_threshold_tuning(df: pd.DataFrame) -> dict:
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

    clf = LogisticRegression(max_iter=300, random_state=config.RANDOM_STATE)
    clf.fit(X_train_s, y_train)

    probs = clf.predict_proba(X_test_s)[:, 1]

    # Business Utility Weights
    # TP: Successful Hire (+$1,000 value)
    # FP: Failed Probation / Unnecessary Interview (-$200 cost)
    # FN: Missed High-Potential Student (-$500 opportunity cost)
    # TN: Correctly Screened Out ($0)
    V_TP = 1000
    C_FP = -200
    C_FN = -500
    V_TN = 0

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    table_rows = []
    utilities = []
    recalls = []
    precisions = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        p = precision_score(y_test, preds, zero_division=0) * 100
        r = recall_score(y_test, preds, zero_division=0) * 100
        f = f1_score(y_test, preds, zero_division=0) * 100
        net_value = (tp * V_TP) + (fp * C_FP) + (fn * C_FN) + (tn * V_TN)

        utilities.append(net_value)
        precisions.append(p)
        recalls.append(r)

        table_rows.append({
            "threshold": t,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "precision": round(float(p), 1),
            "recall": round(float(r), 1),
            "f1": round(float(f), 1),
            "net_value": int(net_value)
        })

    best_idx = int(np.argmax(utilities))
    best_threshold = thresholds[best_idx]
    best_utility = utilities[best_idx]

    # Default 0.5 threshold stats
    def_idx = thresholds.index(0.5)
    def_utility = utilities[def_idx]
    def_row = table_rows[def_idx]

    # Plot Tradeoff
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.plot(thresholds, precisions, color="#2563EB", marker="o", linewidth=2, label="Precision (%)")
    ax1.plot(thresholds, recalls, color="#10B981", marker="s", linewidth=2, label="Recall (%)")
    ax1.set_xlabel("Decision Threshold", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Metric (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim([0, 105])

    ax2 = ax1.twinx()
    ax2.plot(thresholds, [u / 1000 for u in utilities], color="#F59E0B", marker="^", linewidth=2.5, linestyle="--", label="Net Utility ($k)")
    ax2.set_ylabel("Net Utility ($k)", color="#D97706", fontsize=10, fontweight="bold")
    ax2.axvline(best_threshold, color="#DC2626", linestyle=":", linewidth=2, label=f"Optimal T={best_threshold}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=True, facecolor="white", edgecolor="#E2E8F0")

    plt.title("Threshold vs Precision, Recall & Expected Business Utility", fontsize=11, fontweight="bold")
    fig.tight_layout()
    plot_file = os.path.join(PLOT_DIR, "threshold_curve.png")
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)

    return {
        "n_test": len(y_test),
        "best_threshold": best_threshold,
        "best_utility": f"${best_utility:,}",
        "default_utility": f"${def_utility:,}",
        "utility_gain": f"${(best_utility - def_utility):,}",
        "default_row": def_row,
        "best_row": table_rows[best_idx],
        "table_rows": table_rows
    }
