"""Module 4 – UMAP (Uniform Manifold Approximation and Projection).

UMAP is a non-linear dimensionality reduction technique — it preserves the
local and global structure of high-dimensional data far better than PCA.
Where PCA finds the directions of maximum variance (a linear operation),
UMAP constructs a graph of nearest-neighbour relationships and then finds a
low-dimensional layout that preserves those relationships.

Key parameters:
  n_neighbors  — controls local vs global structure (small = local, large = global)
  min_dist     — controls how tightly points are packed in the embedding
  n_components — output dimensions (2 for visualisation)

UMAP is primarily used for visualisation and as a preprocessing step before
clustering — you can run K-Means in UMAP-space instead of the original space.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import config

UMAP_DIR = config.UMAP_DIR
SAMPLE_CAP = 8_000   # UMAP is fast but scales poorly above ~10K rows for plotting


def run_umap(df: pd.DataFrame) -> dict:
    os.makedirs(UMAP_DIR, exist_ok=True)

    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]
    data = df[feat_cols + [config.TARGET_COLUMN]].dropna().copy()

    # Subsample
    n_full = len(data)
    if n_full > SAMPLE_CAP:
        data = data.sample(SAMPLE_CAP, random_state=config.RANDOM_STATE)
    n_sample = len(data)

    ss = StandardScaler()
    X = ss.fit_transform(data[feat_cols].values)
    y = data[config.TARGET_COLUMN].values

    try:
        import umap as umap_lib

        umap_available = True

        # ── Fit 3 configurations to show n_neighbors effect ─────────────────
        configs = [
            {"n_neighbors": 5,  "min_dist": 0.1,  "label": "n_neighbors=5 (local)",  "color": "#2563EB"},
            {"n_neighbors": 30, "min_dist": 0.1,  "label": "n_neighbors=30 (balanced)", "color": "#10B981"},
            {"n_neighbors": 100,"min_dist": 0.5,  "label": "n_neighbors=100 (global)", "color": "#F59E0B"},
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        embeddings = []
        for i, cfg in enumerate(configs):
            reducer = umap_lib.UMAP(
                n_neighbors=cfg["n_neighbors"],
                min_dist=cfg["min_dist"],
                n_components=2,
                random_state=config.RANDOM_STATE,
                verbose=False,
            )
            emb = reducer.fit_transform(X)
            embeddings.append(emb)
            sc = axes[i].scatter(emb[:, 0], emb[:, 1], c=y, cmap="coolwarm",
                                 s=4, alpha=0.4)
            axes[i].set_title(cfg["label"], fontsize=10, fontweight="bold")
            axes[i].set_xlabel("UMAP-1"); axes[i].set_ylabel("UMAP-2")
            axes[i].grid(True, alpha=0.2)
        fig.colorbar(sc, ax=axes[-1], label="Placed (1) / Not Placed (0)")
        fig.suptitle("UMAP Embeddings: Effect of n_neighbors on Structure",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(UMAP_DIR, "umap_neighbors_comparison.png"), dpi=120)
        plt.close(fig)

        # ── Best embedding (balanced) coloured by individual features ────────
        emb_best = embeddings[1]
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        highlight_feats = feat_cols[:6]
        for ax, feat in zip(axes.flatten(), highlight_feats):
            vals = data[feat].values
            sc = ax.scatter(emb_best[:, 0], emb_best[:, 1], c=vals, cmap="viridis",
                            s=4, alpha=0.4)
            ax.set_title(feat, fontsize=9, fontweight="bold")
            ax.set_xlabel("UMAP-1", fontsize=8); ax.set_ylabel("UMAP-2", fontsize=8)
            fig.colorbar(sc, ax=ax)
            ax.grid(True, alpha=0.15)
        fig.suptitle("UMAP Embedding (n_neighbors=30) — Coloured by Each Feature",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(UMAP_DIR, "umap_feature_colours.png"), dpi=120)
        plt.close(fig)

        # ── UMAP vs PCA 2D side-by-side ─────────────────────────────────────
        from sklearn.decomposition import PCA
        pca2 = PCA(n_components=2, random_state=config.RANDOM_STATE)
        X_pca = pca2.fit_transform(X)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", s=4, alpha=0.35)
        ax1.set_title("PCA 2D Projection", fontsize=11, fontweight="bold")
        ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
        ax1.grid(True, alpha=0.2)
        sc2 = ax2.scatter(emb_best[:, 0], emb_best[:, 1], c=y, cmap="coolwarm", s=4, alpha=0.35)
        ax2.set_title("UMAP 2D Projection (n_neighbors=30)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("UMAP-1"); ax2.set_ylabel("UMAP-2")
        ax2.grid(True, alpha=0.2)
        fig.colorbar(sc2, ax=ax2, label="Placed")
        fig.suptitle("PCA vs UMAP — Linear vs Non-Linear Dimensionality Reduction",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(UMAP_DIR, "pca_vs_umap.png"), dpi=120)
        plt.close(fig)

        config_results = [
            {"label": c["label"], "n_neighbors": c["n_neighbors"], "min_dist": c["min_dist"]}
            for c in configs
        ]

    except ImportError:
        umap_available = False
        config_results = []

    return {
        "umap_available": umap_available,
        "n_full": n_full,
        "n_sample": n_sample,
        "n_features": len(feat_cols),
        "feat_cols": feat_cols,
        "configs": config_results,
    }
