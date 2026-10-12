"""Module 5 – Topic 35: Regression Metrics & Robustness.

Evaluates continuous regression models using MAE, MSE, RMSE, R², and MedAE.
Compares standard OLS Linear Regression against robust alternatives (Huber, RANSAC)
when corrupted with severe outliers.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, RANSACRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import config

PLOT_DIR = os.path.join(config.PLOTS_DIR, "RegMetrics")


def run_regression_metrics_and_robustness(df: pd.DataFrame) -> dict:
    os.makedirs(PLOT_DIR, exist_ok=True)
    if "Salary Package" not in df.columns or "CGPA" not in df.columns:
        return {"error": "Salary Package or CGPA column not found"}

    target = "Salary Package"
    feat_cols = [c for c in ["CGPA", "CodingTestScore", "AptitudeTestScore", "Projects"] if c in df.columns]

    data = df[feat_cols + [target]].dropna().copy()
    if len(data) > 4000:
        data = data.sample(4000, random_state=config.RANDOM_STATE)

    X = data[feat_cols].values
    y = data[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=config.RANDOM_STATE
    )

    # Inject extreme outliers in training data (5% of points corrupted 5x)
    np.random.seed(config.RANDOM_STATE)
    corrupted_y_train = y_train.copy()
    n_outliers = int(0.05 * len(corrupted_y_train))
    outlier_idx = np.random.choice(len(corrupted_y_train), size=n_outliers, replace=False)
    corrupted_y_train[outlier_idx] = corrupted_y_train[outlier_idx] * 5.0

    models = {
        "OLS Linear Regression": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=10.0),
        "Huber Regressor (Robust)": HuberRegressor(epsilon=1.35, max_iter=300),
        "RANSAC Regressor (Robust)": RANSACRegressor(random_state=config.RANDOM_STATE),
    }

    results = []
    y_preds = {}

    for name, model in models.items():
        model.fit(X_train, corrupted_y_train)
        y_pred = model.predict(X_test)
        y_preds[name] = y_pred

        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        medae = float(median_absolute_error(y_test, y_pred))

        results.append({
            "name": name,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 3),
            "medae": round(medae, 2),
            "is_robust": "Robust" in name
        })

    # Plot Residual Distributions on Test Set
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Bar comparison of MAE and RMSE
    names = [r["name"].replace(" Regression", "").replace(" Regressor", "") for r in results]
    maes = [r["mae"] for r in results]
    rmses = [r["rmse"] for r in results]
    x_pos = np.arange(len(names))

    axes[0].bar(x_pos - 0.18, maes, width=0.35, label="MAE (L1)", color="#2563EB", alpha=0.9)
    axes[0].bar(x_pos + 0.18, rmses, width=0.35, label="RMSE (L2)", color="#EF4444", alpha=0.9)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(names, rotation=15, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Error Metric", fontsize=9, fontweight="bold")
    axes[0].set_title("MAE vs RMSE Under Outlier Attack", fontsize=10, fontweight="bold")
    axes[0].legend(frameon=True, facecolor="white", edgecolor="#E2E8F0")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    # Actual vs Predicted for OLS vs Huber
    axes[1].scatter(y_test, y_preds["OLS Linear Regression"], alpha=0.3, s=12, label="OLS (Skewed)", color="#F59E0B")
    axes[1].scatter(y_test, y_preds["Huber Regressor (Robust)"], alpha=0.3, s=12, label="Huber (Robust)", color="#10B981")
    ideal_min = min(y_test.min(), min(y_preds["Huber Regressor (Robust)"].min(), y_preds["OLS Linear Regression"].min()))
    ideal_max = max(y_test.max(), max(y_preds["Huber Regressor (Robust)"].max(), y_preds["OLS Linear Regression"].max()))
    axes[1].plot([ideal_min, ideal_max], [ideal_min, ideal_max], "k--", alpha=0.5, label="Perfect 1:1 Line")
    axes[1].set_xlabel("True Salary Package", fontsize=9)
    axes[1].set_ylabel("Predicted Salary", fontsize=9)
    axes[1].set_title("True vs Predicted: OLS vs Huber", fontsize=10, fontweight="bold")
    axes[1].legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    plot_file = os.path.join(PLOT_DIR, "metrics_comparison.png")
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)

    best_mae_model = min(results, key=lambda r: r["mae"])

    return {
        "n_samples": len(data),
        "n_outliers_injected": n_outliers,
        "best_model": best_mae_model["name"],
        "best_mae": best_mae_model["mae"],
        "best_r2": best_mae_model["r2"],
        "results": results
    }
