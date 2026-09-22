"""Module 4 – Hierarchical Clustering.
Builds agglomerative clusters and visualizes the dendrogram.
All fitting is done on a fixed subsample for speed — 50K rows would be O(n²).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import os
import config

HIER_DIR = os.path.join(config.PLOTS_DIR, 'Hierarchical')

# Use a small fixed sample — hierarchical clustering is O(n²) memory
WORK_SAMPLE = 3_000
DENDRO_SAMPLE = 300


def run_hierarchical(df: pd.DataFrame) -> dict:
    os.makedirs(HIER_DIR, exist_ok=True)
    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]
    data = df[feat_cols].dropna().copy()

    # ── Subsample for all fitting ────────────────────────────────────────────
    np.random.seed(config.RANDOM_STATE)
    n = len(data)
    work_idx = np.random.choice(n, min(WORK_SAMPLE, n), replace=False)
    data_work = data.iloc[work_idx].copy()

    ss = StandardScaler()
    X_work = ss.fit_transform(data_work.values)

    # ── Grid search: linkage × n_clusters ───────────────────────────────────
    linkage_methods = ['ward', 'complete', 'average']
    results = []
    for method in linkage_methods:
        for n_clusters in [3, 4, 5]:
            agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
            labels = agg.fit_predict(X_work)
            sil = round(float(silhouette_score(X_work, labels, sample_size=min(1000, len(X_work)))), 4)
            results.append({
                'method': method.capitalize(),
                'n_clusters': n_clusters,
                'silhouette': sil,
            })

    best = max(results, key=lambda x: x['silhouette'])

    # ── Dendrogram on a tiny sample (scipy linkage is O(n²) memory) ─────────
    dend_idx = np.random.choice(len(X_work), min(DENDRO_SAMPLE, len(X_work)), replace=False)
    X_dendro = X_work[dend_idx]
    Z = linkage(X_dendro, method='ward')
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, ax=ax, truncate_mode='level', p=5,
               color_threshold=0.7 * max(Z[:, 2]),
               above_threshold_color='#94A3B8')
    ax.set_title(f'Dendrogram — Ward Linkage ({DENDRO_SAMPLE}-sample preview)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=10)
    ax.set_ylabel('Distance', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HIER_DIR, 'dendrogram.png'), dpi=120)
    plt.close(fig)

    # ── Cluster profiles using the best config ───────────────────────────────
    best_agg = AgglomerativeClustering(
        n_clusters=best['n_clusters'], linkage=best['method'].lower()
    )
    data_work = data_work.copy()
    data_work['Cluster'] = best_agg.fit_predict(X_work)

    profiles = []
    for k in range(best['n_clusters']):
        cluster_data = data_work[data_work['Cluster'] == k]
        profile = {
            'cluster': k,
            'size': int(len(cluster_data)),
            'pct': round(len(cluster_data) / len(data_work) * 100, 1),
        }
        for col in feat_cols[:5]:
            profile[col] = round(float(cluster_data[col].mean()), 2)
        profiles.append(profile)

    return {
        'results': results,
        'best': best,
        'profiles': profiles,
        'feat_cols': feat_cols[:5],
        'n_samples': n,
        'sample_size': len(data_work),
    }
