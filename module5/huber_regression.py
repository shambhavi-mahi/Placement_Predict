"""Module 5 – Huber Regression.

Demonstrates robust regression when outliers are present.
We will try to predict 'Salary Package' using CGPA and AptitudeTestScore.
We manually inject some massive outliers to see how Ordinary Least Squares (OLS)
gets skewed while HuberRegressor remains robust.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import config

HUBER_DIR = os.path.join(config.PLOTS_DIR, "Huber")

def run_huber_regression(df: pd.DataFrame) -> dict:
    os.makedirs(HUBER_DIR, exist_ok=True)
    
    # We need CGPA and Salary Package
    if 'CGPA' not in df.columns or 'Salary Package' not in df.columns:
        return {"error": "Missing CGPA or Salary Package columns"}
        
    data = df[['CGPA', 'Salary Package']].dropna().copy()
    
    # Take a small random sample for clear visualization
    data = data.sample(200, random_state=config.RANDOM_STATE).sort_values('CGPA')
    
    X = data[['CGPA']].values
    y = data['Salary Package'].values
    
    # Inject 10 extreme outliers (simulate data entry errors)
    np.random.seed(config.RANDOM_STATE)
    outlier_idx = np.random.choice(len(y), size=10, replace=False)
    y_corrupted = y.copy()
    y_corrupted[outlier_idx] = y_corrupted[outlier_idx] * 4 # massive salaries

    # Fit OLS
    ols = LinearRegression()
    ols.fit(X, y_corrupted)
    y_pred_ols = ols.predict(X)
    
    # Fit Huber
    huber = HuberRegressor(epsilon=1.35)
    huber.fit(X, y_corrupted)
    y_pred_huber = huber.predict(X)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y_corrupted, color='#94A3B8', alpha=0.6, label='Data (with outliers)')
    ax.scatter(X[outlier_idx], y_corrupted[outlier_idx], color='#EF4444', label='Injected Outliers')
    
    ax.plot(X, y_pred_ols, color='#F59E0B', linewidth=2, linestyle='--', label='OLS (skewed)')
    ax.plot(X, y_pred_huber, color='#10B981', linewidth=2, label='Huber Regression (robust)')
    
    ax.set_xlabel('CGPA', fontsize=10)
    ax.set_ylabel('Salary Package', fontsize=10)
    ax.set_title('Huber vs OLS Regression on Corrupted Data', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plot_path = os.path.join(HUBER_DIR, 'huber_vs_ols.png')
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    # Evaluate on the CLEAN data to see which generalized better
    ols_mae = mean_absolute_error(y, y_pred_ols)
    huber_mae = mean_absolute_error(y, y_pred_huber)

    return {
        "n_samples": len(y),
        "n_outliers": len(outlier_idx),
        "ols_mae": round(float(ols_mae), 2),
        "huber_mae": round(float(huber_mae), 2),
        "improvement": round(float(ols_mae - huber_mae), 2)
    }
