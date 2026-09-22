"""Module 2 – Train / Validation / Test Splitting.
Explains why we split data and what each set is used for.
"""
import numpy as np
import pandas as pd
import config


def run_splitting(df: pd.DataFrame) -> dict:
    """Compute split statistics for the dataset."""
    target = config.TARGET_COLUMN
    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]
    data = df[feat_cols + [target]].dropna().copy()
    n = len(data)

    # 60 / 20 / 20 split
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_df = data.iloc[:train_end]
    val_df = data.iloc[train_end:val_end]
    test_df = data.iloc[val_end:]

    def split_stats(sdf, label):
        placed = int((sdf[target] == 1).sum())
        total = len(sdf)
        return {
            'label': label,
            'n': total,
            'pct': round(total / n * 100, 1),
            'placed': placed,
            'not_placed': total - placed,
            'placed_pct': round(placed / total * 100, 1) if total else 0,
            'cgpa_mean': round(float(sdf['CGPA'].mean()), 3) if 'CGPA' in sdf.columns else None,
        }

    # Demonstrate data leakage: fit scaler on full data vs train only
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    X = data[feat_cols].values
    y = data[target].values
    Xtr, Xva = X[:train_end], X[train_end:val_end]
    ytr, yva = y[:train_end], y[train_end:val_end]

    # Correct: fit scaler on train only
    ss_correct = StandardScaler()
    Xtr_c = ss_correct.fit_transform(Xtr)
    Xva_c = ss_correct.transform(Xva)
    lr1 = LogisticRegression(max_iter=500, random_state=config.RANDOM_STATE)
    lr1.fit(Xtr_c, ytr)
    acc_correct = round(float(accuracy_score(yva, lr1.predict(Xva_c))) * 100, 2)

    # Incorrect: fit scaler on all data (leakage)
    ss_leak = StandardScaler()
    X_all_scaled = ss_leak.fit_transform(X)
    lr2 = LogisticRegression(max_iter=500, random_state=config.RANDOM_STATE)
    lr2.fit(X_all_scaled[:train_end], ytr)
    acc_leak = round(float(accuracy_score(yva, lr2.predict(X_all_scaled[train_end:val_end]))) * 100, 2)

    return {
        'n_total': n,
        'train': split_stats(train_df, 'Train'),
        'val': split_stats(val_df, 'Validation'),
        'test': split_stats(test_df, 'Test'),
        'acc_correct': acc_correct,
        'acc_leak': acc_leak,
        'leak_diff': round(acc_leak - acc_correct, 2),
        'feat_cols': feat_cols,
    }
