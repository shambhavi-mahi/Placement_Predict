"""Module 3 – Feature Importance and SHAP Values."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from models.feature_engg import _apply_encoding
import os
import config

SHAP_DIR = os.path.join(config.PLOTS_DIR, 'SHAP')
SAMPLE_CAP = 10_000


def run_shap_importance(df: pd.DataFrame) -> dict:
    os.makedirs(SHAP_DIR, exist_ok=True)

    # Sample BEFORE encoding
    if len(df) > SAMPLE_CAP:
        df = df.sample(SAMPLE_CAP, random_state=config.RANDOM_STATE)

    enc_df, _, _, _ = _apply_encoding(df)
    drop_cols = ['StudentID', 'IsAnomaly', 'Salary Package', config.TARGET_COLUMN]
    feat_cols = [c for c in enc_df.columns if c not in drop_cols]
    data = enc_df.dropna(subset=feat_cols + [config.TARGET_COLUMN]).copy()
    X = data[feat_cols].values
    y = data[config.TARGET_COLUMN].astype(int).values
    n = len(data)
    split = int(0.8 * n)
    Xtr, Xva = X[:split], X[split:]
    ytr, yva = y[:split], y[split:]

    # Fewer trees, parallel
    rf = RandomForestClassifier(n_estimators=50, random_state=config.RANDOM_STATE, n_jobs=-1)
    rf.fit(Xtr, ytr)
    val_acc = round(float(accuracy_score(yva, rf.predict(Xva))) * 100, 2)

    # 1. MDI
    mdi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    mdi_data = [{'feature': f, 'importance': round(float(v), 4)} for f, v in mdi.items()]

    # 2. Permutation importance — fewer repeats, cap val set
    perm_X = Xva[:min(1000, len(Xva))]
    perm_y = yva[:min(1000, len(yva))]
    perm = permutation_importance(rf, perm_X, perm_y, n_repeats=5,
                                  random_state=config.RANDOM_STATE, n_jobs=-1)
    perm_series = pd.Series(perm.importances_mean, index=feat_cols).sort_values(ascending=False)
    feat_idx = {f: i for i, f in enumerate(feat_cols)}
    perm_data = [
        {'feature': f, 'importance': round(float(v), 4),
         'std': round(float(perm.importances_std[feat_idx[f]]), 4)}
        for f, v in perm_series.items()
    ]

    # MDI plot
    fig, ax = plt.subplots(figsize=(7, max(4, len(mdi_data[:10]) * 0.5)))
    top = mdi.head(10)
    ax.barh(top.index[::-1], top.values[::-1], color='#2563EB')
    ax.set_xlabel('Mean Decrease Impurity', fontsize=10)
    ax.set_title('Feature Importance (MDI)', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SHAP_DIR, 'mdi_importance.png'), dpi=120)
    plt.close(fig)

    # Permutation plot
    fig, ax = plt.subplots(figsize=(7, max(4, len(perm_data[:10]) * 0.5)))
    top_p = perm_series.head(10)
    ax.barh(top_p.index[::-1], top_p.values[::-1], color='#10B981')
    ax.set_xlabel('Accuracy Drop (Permutation)', fontsize=10)
    ax.set_title('Permutation Importance (Validation Set)', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SHAP_DIR, 'perm_importance.png'), dpi=120)
    plt.close(fig)

    shap_available = False
    shap_data = []
    try:
        import shap
        sample_size = min(200, len(Xva))
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(Xva[:sample_size])
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap_mean = np.abs(sv).mean(axis=0)
        shap_series = pd.Series(shap_mean, index=feat_cols).sort_values(ascending=False)
        shap_data = [{'feature': f, 'importance': round(float(v), 4)} for f, v in shap_series.items()]

        fig, ax = plt.subplots(figsize=(7, max(4, len(shap_data[:10]) * 0.5)))
        top_s = shap_series.head(10)
        ax.barh(top_s.index[::-1], top_s.values[::-1], color='#F59E0B')
        ax.set_xlabel('Mean |SHAP Value|', fontsize=10)
        ax.set_title('SHAP Feature Importance', fontsize=11, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(SHAP_DIR, 'shap_importance.png'), dpi=120)
        plt.close(fig)
        shap_available = True
    except Exception:
        pass

    return {
        'val_acc': val_acc,
        'n_features': len(feat_cols),
        'mdi_data': mdi_data,
        'perm_data': perm_data,
        'shap_data': shap_data,
        'shap_available': shap_available,
        'n_train': split,
        'n_val': n - split,
    }
