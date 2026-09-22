"""Module 3 – Modern Boosted Trees (XGBoost / LightGBM).
Compares sklearn GB with XGBoost and LightGBM.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from models.feature_engg import _apply_encoding
import config

SAMPLE_CAP = 10_000


def run_modern_boosting(df: pd.DataFrame) -> dict:
    # Sample BEFORE encoding to avoid slow encoding on 50K rows
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

    # Sklearn Gradient Boosting (reduced estimators for speed)
    gb = GradientBoostingClassifier(n_estimators=50, random_state=config.RANDOM_STATE)
    gb.fit(Xtr, ytr)
    results.append({
        'name': 'Sklearn GradientBoosting',
        'library': 'scikit-learn',
        'train_acc': round(float(accuracy_score(ytr, gb.predict(Xtr))) * 100, 2),
        'val_acc': round(float(accuracy_score(yva, gb.predict(Xva))) * 100, 2),
        'note': 'Classic sequential boosting. Reliable baseline for modern methods.',
    })

    # XGBoost
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators=50, random_state=config.RANDOM_STATE,
            eval_metric='logloss', verbosity=0, n_jobs=-1
        )
        xgb_model.fit(Xtr, ytr)
        results.append({
            'name': 'XGBoost',
            'library': 'xgboost',
            'train_acc': round(float(accuracy_score(ytr, xgb_model.predict(Xtr))) * 100, 2),
            'val_acc': round(float(accuracy_score(yva, xgb_model.predict(Xva))) * 100, 2),
            'note': 'Regularized boosting with column subsampling. Fast and accurate.',
        })
    except ImportError:
        results.append({'name': 'XGBoost', 'library': 'xgboost', 'train_acc': None,
                        'val_acc': None, 'note': 'xgboost not installed (pip install xgboost)'})

    # LightGBM
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(
            n_estimators=50, random_state=config.RANDOM_STATE, verbose=-1, n_jobs=-1
        )
        lgb_model.fit(Xtr, ytr)
        results.append({
            'name': 'LightGBM',
            'library': 'lightgbm',
            'train_acc': round(float(accuracy_score(ytr, lgb_model.predict(Xtr))) * 100, 2),
            'val_acc': round(float(accuracy_score(yva, lgb_model.predict(Xva))) * 100, 2),
            'note': 'Leaf-wise growth. Fastest on large datasets.',
        })
    except ImportError:
        results.append({'name': 'LightGBM', 'library': 'lightgbm', 'train_acc': None,
                        'val_acc': None, 'note': 'lightgbm not installed (pip install lightgbm)'})

    return {
        'results': results,
        'n_train': split,
        'n_val': n - split,
        'n_features': len(feat_cols),
    }
