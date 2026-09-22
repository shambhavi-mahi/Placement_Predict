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
# Single fit with n_neighbors=15 (balanced local/global).
# Only one UMAP.fit_transform() call → ~8-10s on first load, cached forever after.
SAMPLE_CAP = 1_000


def run_umap(df: pd.DataFrame) -> dict:
    os.makedirs(UMAP_DIR, exist_ok=True)

    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]
    data = df[feat_cols + [config.TARGET_COLUMN]].dropna().copy()

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

        # ── Single embedding (balanced: n_neighbors=15) ───────────────────────
        N_NEIGHBORS = 15
        MIN_DIST = 0.1
        reducer = umap_lib.UMAP(
            n_neighbors=N_NEIGHBORS,
            min_dist=MIN_DIST,
            n_components=2,
            random_state=config.RANDOM_STATE,
            verbose=False,
            low_memory=True,        # faster on small samples
        )
        emb = reducer.fit_transform(X)

        # Plot 1 — Coloured by placement outcome
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="coolwarm", s=8, alpha=0.6)
        ax.set_title(f"UMAP Embedding — Placement Outcome\n"
                     f"(n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST}, n={n_sample:,})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.grid(True, alpha=0.2)
        fig.colorbar(sc, ax=ax, label="Placed (1) / Not Placed (0)")
        fig.tight_layout()
        fig.savefig(os.path.join(UMAP_DIR, "umap_neighbors_comparison.png"), dpi=120)
        plt.close(fig)

        # Plot 2 — Same embedding coloured by 4 features
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, feat in zip(axes.flatten(), feat_cols[:4]):
            vals = data[feat].values
            sc2 = ax.scatter(emb[:, 0], emb[:, 1], c=vals, cmap="viridis", s=8, alpha=0.5)
            ax.set_title(feat, fontsize=9, fontweight="bold")
            ax.set_xlabel("UMAP-1", fontsize=8); ax.set_ylabel("UMAP-2", fontsize=8)
            fig.colorbar(sc2, ax=ax)
            ax.grid(True, alpha=0.15)
        fig.suptitle(f"UMAP Embedding — Coloured by Each Feature", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(UMAP_DIR, "umap_feature_colours.png"), dpi=120)
        plt.close(fig)

        # Plot 3 — PCA vs UMAP side-by-side (PCA is instant)
        from sklearn.decomposition import PCA
        pca2 = PCA(n_components=2, random_state=config.RANDOM_STATE)
        X_pca = pca2.fit_transform(X)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", s=6, alpha=0.45)
        ax1.set_title("PCA 2D Projection", fontsize=11, fontweight="bold")
        ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
        ax1.grid(True, alpha=0.2)
        sc3 = ax2.scatter(emb[:, 0], emb[:, 1], c=y, cmap="coolwarm", s=6, alpha=0.45)
        ax2.set_title(f"UMAP 2D Projection (n_neighbors={N_NEIGHBORS})", fontsize=11, fontweight="bold")
        ax2.set_xlabel("UMAP-1"); ax2.set_ylabel("UMAP-2")
        ax2.grid(True, alpha=0.2)
        fig.colorbar(sc3, ax=ax2, label="Placed")
        fig.suptitle("PCA vs UMAP — Linear vs Non-Linear Dimensionality Reduction",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(UMAP_DIR, "pca_vs_umap.png"), dpi=120)
        plt.close(fig)

        config_results = [{"label": f"n_neighbors={N_NEIGHBORS} (balanced)",
                           "n_neighbors": N_NEIGHBORS, "min_dist": MIN_DIST}]

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
