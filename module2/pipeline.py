"""Module 2 – Building an End-to-End sklearn Pipeline.
Demonstrates how to chain preprocessing + model into a single Pipeline object.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import config

# Cap training rows for speed — results are statistically representative
SAMPLE_CAP = 10_000


def run_pipeline(df: pd.DataFrame) -> dict:
    target = config.TARGET_COLUMN
    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]
    data = df[feat_cols + [target]].dropna().copy()

    # Subsample if needed
    if len(data) > SAMPLE_CAP:
        data = data.sample(SAMPLE_CAP, random_state=config.RANDOM_STATE)

    X = data[feat_cols].values
    y = data[target].values
    n = len(data)
    split = int(0.8 * n)
    Xtr, Xva = X[:split], X[split:]
    ytr, yva = y[:split], y[split:]

    pipelines = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=500, random_state=config.RANDOM_STATE))
        ]),
        'Random Forest': Pipeline([
            ('model', RandomForestClassifier(n_estimators=50, random_state=config.RANDOM_STATE, n_jobs=-1))
        ]),
        'Gradient Boosting': Pipeline([
            ('model', GradientBoostingClassifier(n_estimators=50, random_state=config.RANDOM_STATE))
        ]),
    }

    results = []
    for name, pipe in pipelines.items():
        pipe.fit(Xtr, ytr)
        train_acc = round(float(accuracy_score(ytr, pipe.predict(Xtr))) * 100, 2)
        val_acc = round(float(accuracy_score(yva, pipe.predict(Xva))) * 100, 2)
        steps = [s[0] for s in pipe.steps]
        results.append({
            'name': name,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'steps': steps,
            'n_steps': len(steps),
        })

    step_descriptions = {
        'scaler': 'StandardScaler — normalizes features to zero mean and unit variance',
        'model': 'The ML model — receives preprocessed features and learns decision boundaries',
    }

    return {
        'results': results,
        'step_descriptions': step_descriptions,
        'feat_cols': feat_cols,
        'n_features': len(feat_cols),
        'n_train': split,
        'n_val': n - split,
    }
