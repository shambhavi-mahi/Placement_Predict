"""Module 1 – Baseline Model and Serving.
Builds the simplest possible classifier: majority-class predictor.
Teaches the concept of a 'dumb baseline' that all real models must beat.
"""
import os
import numpy as np
import pandas as pd
import config


def run_baseline(df: pd.DataFrame) -> dict:
    """Fit a majority-class baseline and compute metrics."""
    target = config.TARGET_COLUMN
    if target not in df.columns:
        return None

    data = df[[target]].dropna().copy()
    data[target] = pd.to_numeric(data[target], errors='coerce').dropna()
    data = data.dropna()

    n = len(data)
    split = int(0.8 * n)
    y_train = data[target].iloc[:split]
    y_val = data[target].iloc[split:]

    # Majority class
    majority_class = int(y_train.mode()[0])
    majority_label = 'Placed' if majority_class == 1 else 'Not Placed'

    # Predictions = always predict majority class
    y_pred = np.full(len(y_val), majority_class)
    acc = round(float((y_pred == y_val.values).mean()) * 100, 2)

    # Class distribution
    placed_count = int((data[target] == 1).sum())
    not_placed_count = int((data[target] == 0).sum())
    placed_pct = round(placed_count / n * 100, 1)
    not_placed_pct = round(100 - placed_pct, 1)

    # Confusion matrix for baseline
    tp = int(((y_pred == 1) & (y_val.values == 1)).sum())
    tn = int(((y_pred == 0) & (y_val.values == 0)).sum())
    fp = int(((y_pred == 1) & (y_val.values == 0)).sum())
    fn = int(((y_pred == 0) & (y_val.values == 1)).sum())

    precision = round(tp / (tp + fp) * 100, 2) if (tp + fp) else 0.0
    recall = round(tp / (tp + fn) * 100, 2) if (tp + fn) else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) else 0.0

    # What a smart model achieves (logistic regression)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]
        df2 = df[feat_cols + [target]].dropna()
        X = df2[feat_cols].values
        y = df2[target].values
        sp2 = int(0.8 * len(df2))
        ss = StandardScaler()
        Xtr = ss.fit_transform(X[:sp2])
        Xva = ss.transform(X[sp2:])
        lr = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
        lr.fit(Xtr, y[:sp2])
        smart_acc = round(float((lr.predict(Xva) == y[sp2:]).mean()) * 100, 2)
    except Exception:
        smart_acc = None

    return {
        'majority_class': majority_class,
        'majority_label': majority_label,
        'placed_count': placed_count,
        'not_placed_count': not_placed_count,
        'placed_pct': placed_pct,
        'not_placed_pct': not_placed_pct,
        'n_train': split,
        'n_val': n - split,
        'n_total': n,
        'baseline_acc': acc,
        'smart_acc': smart_acc,
        'gap': round(smart_acc - acc, 2) if smart_acc else None,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
    }
