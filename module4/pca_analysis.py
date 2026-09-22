"""Module 4 – Principal Component Analysis (PCA).

PCA finds a new coordinate system where axes (principal components) are ordered
by how much variance they capture. The first component captures the most spread,
the second captures the most remaining spread while orthogonal to the first, etc.

Why it matters: 10 features can often be summarised in 3-4 components with very
little information loss — enabling faster training, better visualisations, and
insight into which original features really drive variation in the data.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import config

PCA_DIR = config.PCA_DIR


def run_pca(df: pd.DataFrame) -> dict:
    os.makedirs(PCA_DIR, exist_ok=True)

    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]
    data = df[feat_cols + [config.TARGET_COLUMN]].dropna().copy()

    ss = StandardScaler()
    X = ss.fit_transform(data[feat_cols].values)

    # ── Fit PCA with all components ────────────────────────────────────────
    pca_full = PCA(n_components=None, random_state=config.RANDOM_STATE)
    pca_full.fit(X)

    evr = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(evr)
    n_components = len(evr)

    # Per-component table (first 10)
    component_table = [
        {
            "pc": i + 1,
            "variance_pct": round(float(evr[i]) * 100, 2),
            "cumulative_pct": round(float(cum_var[i]) * 100, 2),
        }
        for i in range(min(10, n_components))
    ]

    # Components needed for threshold
    thresholds = {}
    for t in [0.80, 0.90, 0.95]:
        n_needed = int(np.searchsorted(cum_var, t) + 1)
        thresholds[f"{int(t*100)}%"] = n_needed

    pc1_pc2_variance = round(float(cum_var[1]) * 100, 1)

    # ── Scree + cumulative variance plot ───────────────────────────────────
    n_show = min(10, n_components)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, n_show + 1), evr[:n_show] * 100, color="#2563EB", edgecolor="white")
    axes[0].set_xlabel("Principal Component", fontsize=10)
    axes[0].set_ylabel("Variance Explained (%)", fontsize=10)
    axes[0].set_title("Scree Plot", fontsize=11, fontweight="bold")
    axes[0].set_xticks(range(1, n_show + 1))
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(range(1, n_show + 1), cum_var[:n_show] * 100,
                 marker="o", color="#10B981", linewidth=2)
    for t_val in [80, 90, 95]:
        axes[1].axhline(y=t_val, color="#94A3B8", linestyle="--", linewidth=1)
        axes[1].text(n_show - 0.5, t_val + 0.5, f"{t_val}%", fontsize=8, color="#64748B")
    axes[1].set_xlabel("Number of Components", fontsize=10)
    axes[1].set_ylabel("Cumulative Variance (%)", fontsize=10)
    axes[1].set_title("Cumulative Variance Explained", fontsize=11, fontweight="bold")
    axes[1].set_xticks(range(1, n_show + 1))
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PCA_DIR, "scree_cumulative.png"), dpi=120)
    plt.close(fig)

    # ── Loadings: which features drive PC1 and PC2 ────────────────────────
    loadings = pd.DataFrame(
        pca_full.components_[:2].T,
        columns=["PC1", "PC2"],
        index=feat_cols,
    )
    top_pc1 = loadings["PC1"].abs().sort_values(ascending=False)
    top_pc2 = loadings["PC2"].abs().sort_values(ascending=False)

    loading_table = [
        {
            "feature": f,
            "pc1": round(float(loadings.loc[f, "PC1"]), 3),
            "pc2": round(float(loadings.loc[f, "PC2"]), 3),
            "pc1_direction": "+" if loadings.loc[f, "PC1"] > 0 else "−",
            "pc2_direction": "+" if loadings.loc[f, "PC2"] > 0 else "−",
        }
        for f in top_pc1.index
    ]

    # Loadings biplot (arrow circle)
    fig, ax = plt.subplots(figsize=(7, 7))
    for feat in feat_cols:
        x_val = loadings.loc[feat, "PC1"]
        y_val = loadings.loc[feat, "PC2"]
        ax.arrow(0, 0, x_val, y_val, head_width=0.025, color="#2563EB", alpha=0.7)
        ax.text(x_val * 1.12, y_val * 1.12, feat, fontsize=7.5, ha="center", color="#1E3A5F")
    circle = plt.Circle((0, 0), 1, fill=False, linestyle="--", color="#94A3B8")
    ax.add_artist(circle)
    ax.axhline(0, color="#E2E8F0"); ax.axvline(0, color="#E2E8F0")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(f"PC1 ({round(evr[0]*100, 1)}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({round(evr[1]*100, 1)}% variance)", fontsize=10)
    ax.set_title("PCA Loadings Biplot — How Features Map onto PC1/PC2",
                 fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(PCA_DIR, "loadings_biplot.png"), dpi=120)
    plt.close(fig)

    # ── 2D projection coloured by placement ───────────────────────────────
    X2d = pca_full.transform(X)[:, :2]
    y = data[config.TARGET_COLUMN].values
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(X2d[:, 0], X2d[:, 1], c=y, cmap="coolwarm", s=4, alpha=0.35)
    ax.set_xlabel(f"PC1 ({round(evr[0]*100, 1)}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({round(evr[1]*100, 1)}% variance)", fontsize=10)
    ax.set_title(f"Students Projected onto PC1/PC2  ({pc1_pc2_variance}% of total variance)\n"
                 f"Coloured by Placement Outcome",
                 fontsize=11, fontweight="bold")
    plt.colorbar(sc, ax=ax, label="Placed (1) / Not Placed (0)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(PCA_DIR, "pca_2d_placement.png"), dpi=120)
    plt.close(fig)

    return {
        "n_samples": len(data),
        "n_features": len(feat_cols),
        "n_components": n_components,
        "component_table": component_table,
        "thresholds": thresholds,
        "pc1_pc2_variance": pc1_pc2_variance,
        "pc1_variance": round(float(evr[0]) * 100, 2),
        "pc2_variance": round(float(evr[1]) * 100, 2),
        "loading_table": loading_table,
        "top_pc1_feature": top_pc1.index[0],
        "top_pc2_feature": top_pc2.index[0],
    }
