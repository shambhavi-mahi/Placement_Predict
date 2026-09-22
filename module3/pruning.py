"""Module 3 – Pruning and Splitting Criteria.
Compares Gini impurity vs Entropy for splitting, and
post-pruning via min_samples_leaf / max_depth constraints.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from models.feature_engg import _apply_encoding
import config

# Decision trees are fast enough on 50K, but encoding is slow — sample first
SAMPLE_CAP = 10_000


def run_pruning(df: pd.DataFrame) -> dict:
    # Sample before encoding to keep things fast
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

    results = []
    for criterion in ['gini', 'entropy']:
        for max_depth in [None, 3, 5, 10]:
            dt = DecisionTreeClassifier(
                criterion=criterion, max_depth=max_depth,
                random_state=config.RANDOM_STATE
            )
            dt.fit(Xtr, ytr)
            train_acc = round(float(accuracy_score(ytr, dt.predict(Xtr))) * 100, 2)
            val_acc = round(float(accuracy_score(yva, dt.predict(Xva))) * 100, 2)
            results.append({
                'criterion': criterion.capitalize(),
                'max_depth': str(max_depth) if max_depth else 'None (full)',
                'train_acc': train_acc,
                'val_acc': val_acc,
                'n_leaves': int(dt.get_n_leaves()),
                'depth': int(dt.get_depth()),
                'gap': round(train_acc - val_acc, 2),
            })

    leaf_results = []
    for msl in [1, 5, 10, 20, 50]:
        dt = DecisionTreeClassifier(min_samples_leaf=msl, random_state=config.RANDOM_STATE)
        dt.fit(Xtr, ytr)
        leaf_results.append({
            'min_samples_leaf': msl,
            'train_acc': round(float(accuracy_score(ytr, dt.predict(Xtr))) * 100, 2),
            'val_acc': round(float(accuracy_score(yva, dt.predict(Xva))) * 100, 2),
            'n_leaves': int(dt.get_n_leaves()),
        })

    return {
        'results': results,
        'leaf_results': leaf_results,
        'n_train': split,
        'n_val': n - split,
        'n_features': len(feat_cols),
    }
