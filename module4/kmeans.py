"""Module 4 – K-Means Clustering.
Clusters students by academic profile and reveals natural groupings.
Uses Elbow method and Silhouette score to find optimal K.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import config

KMEANS_DIR = os.path.join(config.PLOTS_DIR, 'KMeans')


def run_kmeans(df: pd.DataFrame) -> dict:
    os.makedirs(KMEANS_DIR, exist_ok=True)
    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]
    data = df[feat_cols].dropna().copy()
    ss = StandardScaler()
    X = ss.fit_transform(data.values)

    # Elbow method: inertia for K=2..10
    k_range = list(range(2, 11))
    inertias = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(round(float(km.inertia_), 2))
        sil_scores.append(round(float(silhouette_score(X, labels, sample_size=min(2000, len(X)))), 4))

    best_k = k_range[int(np.argmax(sil_scores))]

    # Elbow plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(k_range, inertias, marker='o', color='#2563EB', linewidth=2)
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=10)
    axes[0].set_ylabel('Inertia', fontsize=10)
    axes[0].set_title('Elbow Method', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(k_range, sil_scores, marker='s', color='#10B981', linewidth=2)
    axes[1].axvline(best_k, color='#F59E0B', linestyle='--', linewidth=1.5, label=f'Best K={best_k}')
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=10)
    axes[1].set_ylabel('Silhouette Score', fontsize=10)
    axes[1].set_title('Silhouette Score', fontsize=11, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(KMEANS_DIR, 'elbow_silhouette.png'), dpi=120)
    plt.close(fig)

    # Fit best K and get cluster profiles
    km_best = KMeans(n_clusters=best_k, random_state=config.RANDOM_STATE, n_init=10)
    data['Cluster'] = km_best.fit_predict(X)

    # Cluster profiles (mean of each feature per cluster)
    profiles = []
    for k in range(best_k):
        cluster_data = data[data['Cluster'] == k]
        profile = {'cluster': k, 'size': int(len(cluster_data)),
                   'pct': round(len(cluster_data) / len(data) * 100, 1)}
        for col in feat_cols:
            profile[col] = round(float(cluster_data[col].mean()), 2)
        profiles.append(profile)

    # Scatter plot: CGPA vs MockInterviewScore, colored by cluster
    colors = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16']
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in range(best_k):
        mask = data['Cluster'] == k
        ax.scatter(
            data.loc[mask, 'CGPA'] if 'CGPA' in data.columns else data.index[mask],
            data.loc[mask, 'MockInterviewScore'] if 'MockInterviewScore' in data.columns else np.zeros(mask.sum()),
            c=colors[k % len(colors)], alpha=0.3, s=8, label=f'Cluster {k}'
        )
    ax.set_xlabel('CGPA', fontsize=10)
    ax.set_ylabel('Mock Interview Score', fontsize=10)
    ax.set_title(f'K-Means Clusters (K={best_k})', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(KMEANS_DIR, 'cluster_scatter.png'), dpi=120)
    plt.close(fig)

    return {
        'best_k': best_k,
        'k_range': k_range,
        'inertias': inertias,
        'sil_scores': sil_scores,
        'best_sil': round(max(sil_scores), 4),
        'profiles': profiles,
        'feat_cols': feat_cols,
        'n_samples': len(data),
    }
