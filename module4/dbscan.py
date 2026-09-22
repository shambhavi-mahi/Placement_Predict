"""Module 4 – DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

Unlike K-Means and Hierarchical clustering which force every point into a cluster,
DBSCAN identifies dense regions as clusters and explicitly labels outliers as NOISE.
Key insight: no need to specify K upfront — cluster count emerges from the data's density.

Two parameters:
  eps         — radius around each point to search for neighbors
  min_samples — minimum neighbors needed within eps to be a 'core point'

Three point types:
  Core   — has ≥ min_samples neighbors within eps → anchors a cluster
  Border — within eps of a core point but not core itself → belongs to that cluster
  Noise  — not a core point and not within eps of one → labeled -1
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import config

DBSCAN_DIR = config.DBSCAN_DIR
SUBSAMPLE_SIZE = 5_000   # DBSCAN neighbour search is O(n²) in dense regions
MIN_SAMPLES = 20         # Rule of thumb: 2 × n_features; capped for speed


def run_dbscan(df: pd.DataFrame) -> dict:
    os.makedirs(DBSCAN_DIR, exist_ok=True)

    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]
    data = df[feat_cols + [config.TARGET_COLUMN]].dropna().copy()

    # ── Subsample ──────────────────────────────────────────────────────────
    n_full = len(data)
    rng = np.random.RandomState(config.RANDOM_STATE)
    idx = rng.choice(n_full, min(SUBSAMPLE_SIZE, n_full), replace=False)
    sample = data.iloc[idx].reset_index(drop=True)

    ss = StandardScaler()
    X = ss.fit_transform(sample[feat_cols].values)

    # ── K-Distance plot to find eps elbow ─────────────────────────────────
    nbrs = NearestNeighbors(n_neighbors=MIN_SAMPLES)
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dist = np.sort(distances[:, -1])

    # Kneedle / max-curvature approach (normalise, find farthest point from diagonal)
    n_pts = len(k_dist)
    x_norm = np.linspace(0, 1, n_pts)
    y_norm = (k_dist - k_dist.min()) / (k_dist.max() - k_dist.min() + 1e-12)
    perp = np.abs((y_norm[-1] - y_norm[0]) * x_norm
                  - (x_norm[-1] - x_norm[0]) * y_norm
                  + x_norm[-1] * y_norm[0] - y_norm[-1] * x_norm[0])
    perp /= (np.sqrt((y_norm[-1] - y_norm[0])**2 + (x_norm[-1] - x_norm[0])**2) + 1e-12)
    elbow_idx = int(np.argmax(perp))
    suggested_eps = round(float(k_dist[elbow_idx]), 2)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(k_dist, color="#6E2436", linewidth=1.5)
    ax.axvline(elbow_idx, color="#F59E0B", linestyle="--", linewidth=1.5,
               label=f"Elbow → eps ≈ {suggested_eps}")
    ax.set_xlabel("Points (sorted by distance to k-th neighbour)", fontsize=10)
    ax.set_ylabel(f"Distance to {MIN_SAMPLES}-th neighbour", fontsize=10)
    ax.set_title("K-Distance Plot — Find the Elbow to Choose eps", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DBSCAN_DIR, "k_distance_plot.png"), dpi=120)
    plt.close(fig)

    # ── Fit DBSCAN ─────────────────────────────────────────────────────────
    EPS = suggested_eps
    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, algorithm="ball_tree", n_jobs=-1)
    labels = db.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    n_core = int(len(db.core_sample_indices_))

    cluster_sizes = []
    for cid in sorted(set(labels)):
        count = int((labels == cid).sum())
        pct = round(count / len(labels) * 100, 1)
        cluster_sizes.append({
            "cluster": "Noise" if cid == -1 else f"Cluster {cid}",
            "count": count, "pct": pct,
            "is_noise": cid == -1
        })

    # ── eps sensitivity ─────────────────────────────────────────────────────
    sensitivity = []
    for eps_val in [round(EPS * 0.5, 2), EPS, round(EPS * 2, 2)]:
        tmp = DBSCAN(eps=eps_val, min_samples=MIN_SAMPLES, algorithm="ball_tree", n_jobs=-1)
        tmp_labels = tmp.fit_predict(X)
        sensitivity.append({
            "eps": eps_val,
            "n_clusters": len(set(tmp_labels)) - (1 if -1 in tmp_labels else 0),
            "n_noise": int((tmp_labels == -1).sum()),
            "noise_pct": round((tmp_labels == -1).sum() / len(tmp_labels) * 100, 1),
            "is_chosen": eps_val == EPS,
        })

    # ── Anomaly overlap ────────────────────────────────────────────────────
    anomaly_overlap = None
    if "IsAnomaly" in sample.columns:
        is_noise_mask = labels == -1
        anom_mask = sample["IsAnomaly"].values.astype(bool)
        both = int((is_noise_mask & anom_mask).sum())
        anomaly_overlap = {
            "both": both,
            "noise_that_is_anomaly": round(both / max(n_noise, 1) * 100, 1),
            "anomaly_caught_as_noise": round(both / max(anom_mask.sum(), 1) * 100, 1),
        }

    # ── PCA 2D scatter: clusters vs noise ─────────────────────────────────
    pca2 = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X2d = pca2.fit_transform(X)

    noise_mask = labels == -1
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        ax.scatter(X2d[mask, 0], X2d[mask, 1],
                   color=colors[cid % len(colors)], s=6, alpha=0.5, label=f"Cluster {cid}")
    ax.scatter(X2d[noise_mask, 0], X2d[noise_mask, 1],
               c="black", marker="x", s=15, alpha=0.5, label=f"Noise ({n_noise:,})")
    ax.set_xlabel("PCA Component 1", fontsize=10)
    ax.set_ylabel("PCA Component 2", fontsize=10)
    ax.set_title(f"DBSCAN — Clusters vs Noise (eps={EPS}, min_samples={MIN_SAMPLES})",
                 fontsize=11, fontweight="bold")
    if n_clusters <= 8:
        ax.legend(markerscale=2, fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(DBSCAN_DIR, "dbscan_clusters_pca_2d.png"), dpi=120)
    plt.close(fig)

    # ── Placement rate: clustered vs noise ─────────────────────────────────
    placement_clustered = round(float(sample.loc[~noise_mask, config.TARGET_COLUMN].mean()) * 100, 1) \
        if (~noise_mask).any() else None
    placement_noise = round(float(sample.loc[noise_mask, config.TARGET_COLUMN].mean()) * 100, 1) \
        if noise_mask.any() else None

    return {
        "n_full": n_full,
        "n_sample": len(sample),
        "n_features": len(feat_cols),
        "suggested_eps": suggested_eps,
        "eps": EPS,
        "min_samples": MIN_SAMPLES,
        "n_clusters": n_clusters,
        "n_core": n_core,
        "n_noise": n_noise,
        "noise_pct": round(n_noise / len(labels) * 100, 1),
        "core_pct": round(n_core / len(labels) * 100, 1),
        "cluster_sizes": cluster_sizes,
        "sensitivity": sensitivity,
        "anomaly_overlap": anomaly_overlap,
        "placement_clustered": placement_clustered,
        "placement_noise": placement_noise,
    }
