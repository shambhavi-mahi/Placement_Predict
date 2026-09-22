"""Module 3 – Comparing Model Tracks.
Runs all trained models against each other on the same val split
and produces a unified comparison matrix.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from models.feature_engg import _apply_encoding
import config

SAMPLE_CAP = 10_000


def run_compare_models(df: pd.DataFrame) -> dict:
    # Sample before encoding for speed
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

    ss = StandardScaler()
    Xtr_s = ss.fit_transform(Xtr)
    Xva_s = ss.transform(Xva)

    models = [
        ('Logistic Regression',    LogisticRegression(max_iter=500, random_state=config.RANDOM_STATE), True),
        ('Decision Tree (full)',    DecisionTreeClassifier(random_state=config.RANDOM_STATE), False),
        ('Decision Tree (depth=3)', DecisionTreeClassifier(max_depth=3, random_state=config.RANDOM_STATE), False),
        ('Bagging',                BaggingClassifier(n_estimators=30, random_state=config.RANDOM_STATE, n_jobs=-1), False),
        ('Random Forest',          RandomForestClassifier(n_estimators=50, random_state=config.RANDOM_STATE, n_jobs=-1), False),
        ('AdaBoost',               AdaBoostClassifier(n_estimators=50, random_state=config.RANDOM_STATE), False),
        ('Gradient Boosting',      GradientBoostingClassifier(n_estimators=50, random_state=config.RANDOM_STATE), False),
    ]

    results = []
    for name, model, scaled in models:
        Xt = Xtr_s if scaled else Xtr
        Xv = Xva_s if scaled else Xva
        model.fit(Xt, ytr)
        ypred = model.predict(Xv)
        try:
            yprob = model.predict_proba(Xv)[:, 1]
            auc = round(float(roc_auc_score(yva, yprob)), 4)
        except Exception:
            auc = None
        train_acc = round(float(accuracy_score(ytr, model.predict(Xt))) * 100, 2)
        val_acc = round(float(accuracy_score(yva, ypred)) * 100, 2)
        f1 = round(float(f1_score(yva, ypred, zero_division=0)) * 100, 2)
        gap = round(train_acc - val_acc, 2)
        category = 'Linear' if name == 'Logistic Regression' else ('Tree' if 'Tree' in name else 'Ensemble')
        results.append({
            'name': name,
            'category': category,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'f1': f1,
            'auc': auc,
            'gap': gap,
            'overfit': gap > 5,
        })

    results = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    best = results[0]

    return {
        'results': results,
        'best': best,
        'n_train': split,
        'n_val': n - split,
        'n_models': len(results),
    }
