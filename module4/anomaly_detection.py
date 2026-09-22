"""Module 4 – Anomaly Detection.

Identifies unusual students whose profiles are statistically inconsistent
with the majority of the data. Two complementary algorithms:

  IsolationForest — builds random trees; points that are isolated early
    (in shallow trees) are anomalies. Fast, works well in high dimensions.
    contamination parameter: expected fraction of anomalies.

  LocalOutlierFactor (LOF) — compares each point's local density to its
    neighbours' densities. Points in sparse neighbourhoods score high (anomalous).
    More sensitive to local structure than IsolationForest.

The dataset already has an 'IsAnomaly' flag (IQR-based). We use it to
benchmark how well each unsupervised method agrees with that known label.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
)
import config

ANOMALY_DIR = config.ANOMALY_DIR
SAMPLE_CAP = 4_000   # LOF is O(n²) — 4K gives fast results while still meaningful


def run_anomaly_detection(df: pd.DataFrame) -> dict:
    os.makedirs(ANOMALY_DIR, exist_ok=True)

    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]
    has_flag = "IsAnomaly" in df.columns

    keep_cols = feat_cols + ([config.TARGET_COLUMN] + (["IsAnomaly"] if has_flag else []))
    data = df[keep_cols].dropna().copy()

    n_full = len(data)
    if n_full > SAMPLE_CAP:
        data = data.sample(SAMPLE_CAP, random_state=config.RANDOM_STATE)
    n_sample = len(data)

    ss = StandardScaler()
    X = ss.fit_transform(data[feat_cols].values)

    # True labels (IQR-based flag in the dataset)
    y_true = data["IsAnomaly"].values.astype(int) if has_flag else None
    contamination = float(y_true.mean()) if has_flag else 0.05

    # ── IsolationForest ────────────────────────────────────────────────────
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    iso_raw = iso.fit_predict(X)             # -1 = anomaly, 1 = normal
    iso_labels = (iso_raw == -1).astype(int) # 1 = anomaly, 0 = normal
    iso_scores = -iso.score_samples(X)       # higher = more anomalous

    # ── LocalOutlierFactor ─────────────────────────────────────────────────
    lof = LocalOutlierFactor(
        n_neighbors=10,          # fewer neighbors → faster
        contamination=contamination,
        algorithm='ball_tree',   # faster than brute force
        n_jobs=-1,
    )
    lof_raw = lof.fit_predict(X)
    lof_labels = (lof_raw == -1).astype(int)
    lof_scores = -lof.negative_outlier_factor_

    # ── Metrics vs known flag ──────────────────────────────────────────────
    def metrics(y_pred, name):
        if y_true is None:
            return {"name": name, "precision": None, "recall": None, "f1": None,
                    "accuracy": None, "n_flagged": int(y_pred.sum())}
        cm = confusion_matrix(y_true, y_pred)
        return {
            "name": name,
            "n_flagged": int(y_pred.sum()),
            "pct_flagged": round(y_pred.sum() / len(y_pred) * 100, 1),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)) * 100, 2),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)) * 100, 2),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)) * 100, 2),
            "accuracy": round(float(accuracy_score(y_true, y_pred)) * 100, 2),
            "tp": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
            "fp": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
            "fn": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
            "tn": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        }

    iso_metrics = metrics(iso_labels, "IsolationForest")
    lof_metrics = metrics(lof_labels, "LocalOutlierFactor")

    # ── PCA 2D scatter ─────────────────────────────────────────────────────
    pca2 = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X2d = pca2.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, lbls, title, col_anom in [
        (axes[0], iso_labels, "IsolationForest", "#EF4444"),
        (axes[1], lof_labels, "LocalOutlierFactor", "#F59E0B"),
    ]:
        normal_mask = lbls == 0
        ax.scatter(X2d[normal_mask, 0], X2d[normal_mask, 1],
                   c="#2563EB", s=4, alpha=0.3, label="Normal")
        ax.scatter(X2d[~normal_mask, 0], X2d[~normal_mask, 1],
                   c=col_anom, marker="x", s=20, alpha=0.7,
                   label=f"Anomaly ({(~normal_mask).sum():,})")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("PCA-1"); ax.set_ylabel("PCA-2")
        ax.legend(markerscale=2, fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.suptitle("Anomaly Detection — Normal vs Flagged Points (PCA 2D View)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(ANOMALY_DIR, "anomaly_scatter.png"), dpi=120)
    plt.close(fig)

    # ── Score distribution ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(iso_scores, bins=60, color="#2563EB", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Anomaly Score", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].set_title("IsolationForest — Score Distribution", fontsize=10, fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].hist(lof_scores, bins=60, color="#F59E0B", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("LOF Score (−negative_outlier_factor)", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].set_title("LocalOutlierFactor — Score Distribution", fontsize=10, fontweight="bold")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ANOMALY_DIR, "score_distributions.png"), dpi=120)
    plt.close(fig)

    # ── Anomaly profile vs normal profile ─────────────────────────────────
    data_copy = data[feat_cols].copy()
    iso_anom_profile = data_copy[iso_labels == 1].mean().round(2).to_dict()
    iso_norm_profile = data_copy[iso_labels == 0].mean().round(2).to_dict()
    profile_comparison = [
        {
            "feature": f,
            "normal_mean": iso_norm_profile.get(f, 0),
            "anomaly_mean": iso_anom_profile.get(f, 0),
            "diff": round(iso_anom_profile.get(f, 0) - iso_norm_profile.get(f, 0), 2),
        }
        for f in feat_cols
    ]
    profile_comparison.sort(key=lambda x: abs(x["diff"]), reverse=True)

    # Dataset flag stats
    flag_stats = None
    if has_flag:
        n_flagged_iqr = int(data["IsAnomaly"].sum())
        flag_stats = {
            "n_flagged": n_flagged_iqr,
            "pct": round(n_flagged_iqr / n_sample * 100, 1),
        }

    return {
        "n_full": n_full,
        "n_sample": n_sample,
        "n_features": len(feat_cols),
        "contamination": round(contamination * 100, 1),
        "has_flag": has_flag,
        "flag_stats": flag_stats,
        "iso_metrics": iso_metrics,
        "lof_metrics": lof_metrics,
        "profile_comparison": profile_comparison,
    }
