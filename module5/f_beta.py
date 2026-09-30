"""Module 5 – F-Beta Score.

The F-beta score is a generalization of the F1 score that adds a weight
(beta) to recall.
- beta = 1: F1 score (harmonic mean, precision and recall weighted equally)
- beta = 2: F2 score (recall is twice as important as precision)
- beta = 0.5: F0.5 score (precision is twice as important as recall)

This is crucial when business constraints mean false positives and
false negatives have different costs.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, fbeta_score
import config

def run_f_beta(df: pd.DataFrame) -> dict:
    target = config.TARGET_COLUMN
    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]
    data = df[feat_cols + [target]].dropna().copy()
    
    # Cap size for speed
    if len(data) > 10_000:
        data = data.sample(10_000, random_state=config.RANDOM_STATE)

    X = data[feat_cols].values
    y = data[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=config.RANDOM_STATE, stratify=y
    )

    ss = StandardScaler()
    X_train_s = ss.fit_transform(X_train)
    X_test_s = ss.transform(X_test)

    # Train a standard Logistic Regression
    lr = LogisticRegression(max_iter=300, random_state=config.RANDOM_STATE)
    lr.fit(X_train_s, y_train)

    # Predict with different thresholds to see effect
    y_prob = lr.predict_proba(X_test_s)[:, 1]
    
    # Calculate for standard threshold = 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    f1 = fbeta_score(y_test, y_pred, beta=1.0, zero_division=0)
    f2 = fbeta_score(y_test, y_pred, beta=2.0, zero_division=0)
    f05 = fbeta_score(y_test, y_pred, beta=0.5, zero_division=0)

    # Generate threshold curve data
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    curve_data = []
    
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        p = precision_score(y_test, yp, zero_division=0)
        r = recall_score(y_test, yp, zero_division=0)
        curve_data.append({
            "threshold": t,
            "precision": round(float(p) * 100, 1),
            "recall": round(float(r) * 100, 1),
            "f1": round(float(fbeta_score(y_test, yp, beta=1.0, zero_division=0)) * 100, 1),
            "f2": round(float(fbeta_score(y_test, yp, beta=2.0, zero_division=0)) * 100, 1),
            "f05": round(float(fbeta_score(y_test, yp, beta=0.5, zero_division=0)) * 100, 1),
        })

    return {
        "precision": round(float(precision) * 100, 1),
        "recall": round(float(recall) * 100, 1),
        "f1": round(float(f1) * 100, 1),
        "f2": round(float(f2) * 100, 1),
        "f05": round(float(f05) * 100, 1),
        "curve_data": curve_data
    }
