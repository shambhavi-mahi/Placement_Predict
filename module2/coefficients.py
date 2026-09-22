"""Module 2 – Interpreting Model Coefficients.
Explains what logistic regression coefficients mean and
how to read them as feature importance proxies.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import config

COEFF_DIR = os.path.join(config.PLOTS_DIR, 'Coefficients')


def run_coefficients(df: pd.DataFrame) -> dict:
    os.makedirs(COEFF_DIR, exist_ok=True)
    target = config.TARGET_COLUMN
    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]
    data = df[feat_cols + [target]].dropna().copy()
    if len(data) > 15_000:
        data = data.sample(15_000, random_state=config.RANDOM_STATE)
    X = data[feat_cols].values
    y = data[target].values
    n = len(data)
    split = int(0.8 * n)
    ss = StandardScaler()
    Xtr = ss.fit_transform(X[:split])
    Xva = ss.transform(X[split:])
    model = LogisticRegression(max_iter=300, random_state=config.RANDOM_STATE)
    model.fit(Xtr, y[:split])
    val_acc = round(float(accuracy_score(y[split:], model.predict(Xva))) * 100, 2)

    coefs = model.coef_[0]
    coef_data = sorted(
        [{'feature': f, 'coef': round(float(c), 4), 'abs_coef': round(abs(float(c)), 4),
          'direction': 'Positive ↑' if c > 0 else 'Negative ↓',
          'color': '#10B981' if c > 0 else '#EF4444'}
         for f, c in zip(feat_cols, coefs)],
        key=lambda x: abs(x['coef']), reverse=True
    )
    intercept = round(float(model.intercept_[0]), 4)

    # Coefficient bar chart
    fig, ax = plt.subplots(figsize=(8, max(4, len(feat_cols) * 0.5)))
    colors = ['#10B981' if c > 0 else '#EF4444' for c in coefs]
    sorted_feats = [d['feature'] for d in coef_data[::-1]]
    sorted_coefs = [d['coef'] for d in coef_data[::-1]]
    sorted_colors = ['#10B981' if c > 0 else '#EF4444' for c in sorted_coefs]
    ax.barh(sorted_feats, sorted_coefs, color=sorted_colors, edgecolor='white')
    ax.axvline(0, color='#94A3B8', linewidth=1)
    ax.set_xlabel('Coefficient Value (Standardized)', fontsize=10)
    ax.set_title('Logistic Regression Coefficients\n(positive = more likely Placed)', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(COEFF_DIR, 'coefficients_bar.png'), dpi=120)
    plt.close(fig)

    # Odds ratios (exp of coef)
    for d in coef_data:
        d['odds_ratio'] = round(float(np.exp(d['coef'])), 4)
        d['interpretation'] = (
            f"1 std increase → {round((d['odds_ratio']-1)*100, 1)}% {'higher' if d['coef']>0 else 'lower'} odds of placement"
        )

    return {
        'coef_data': coef_data,
        'intercept': intercept,
        'val_acc': val_acc,
        'n_features': len(feat_cols),
        'top_positive': [d for d in coef_data if d['coef'] > 0][:3],
        'top_negative': [d for d in coef_data if d['coef'] < 0][:3],
        'has_plot': True,
    }
