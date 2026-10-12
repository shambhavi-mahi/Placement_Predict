"""Module 5 – Topic 38: Bayesian Optimization.

Demonstrates Sequential Model-Based Optimization (SMBO) using a Gaussian Process
surrogate model and Upper Confidence Bound (UCB) acquisition function.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import config

PLOT_DIR = os.path.join(config.PLOTS_DIR, "Bayes")


def run_bayesian_optimization(df: pd.DataFrame) -> dict:
    os.makedirs(PLOT_DIR, exist_ok=True)
    target = config.TARGET_COLUMN
    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]

    data = df[feat_cols + [target]].dropna().copy()
    if len(data) > 3000:
        data = data.sample(3000, random_state=config.RANDOM_STATE)

    X = data[feat_cols].values
    y = data[target].astype(int).values

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)

    # Black-box objective function: Evaluate C (in log space: log10(C) between -3 and 2)
    def evaluate_c(log_c):
        c_val = 10.0 ** float(log_c)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c_val, max_iter=200, random_state=config.RANDOM_STATE))
        return float(cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()) * 100

    # Search space: log10(C) in [-3.0, 2.0]
    np.random.seed(config.RANDOM_STATE)
    candidate_space = np.linspace(-3.0, 2.0, 50).reshape(-1, 1)

    # 1. Warm-up: 3 initial random evaluations
    X_sample = np.array([[-2.5], [0.0], [1.5]])
    y_sample = np.array([evaluate_c(x[0]) for x in X_sample])

    gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-3, random_state=config.RANDOM_STATE)

    bayes_history = []
    for step in range(8):
        gp.fit(X_sample, y_sample)
        mu, sigma = gp.predict(candidate_space, return_std=True)

        # UCB Acquisition Function (kappa = 1.96)
        ucb = mu + 1.96 * sigma

        # Exclude already sampled points
        for sampled_x in X_sample:
            idx = np.argmin(np.abs(candidate_space.ravel() - sampled_x[0]))
            ucb[idx] = -np.inf

        next_x = candidate_space[np.argmax(ucb)]
        next_y = evaluate_c(next_x[0])

        bayes_history.append({
            "iter": step + 1,
            "log_c": round(float(next_x[0]), 2),
            "actual_c": round(float(10.0 ** next_x[0]), 4),
            "score": round(float(next_y), 2),
            "predicted_mean": round(float(mu[np.argmax(ucb)]), 2),
            "uncertainty_sigma": round(float(sigma[np.argmax(ucb)]), 2),
        })

        X_sample = np.vstack((X_sample, next_x))
        y_sample = np.append(y_sample, next_y)

    # Random search comparison over 8 steps
    rand_x = np.random.uniform(-3.0, 2.0, size=8)
    rand_scores = [evaluate_c(x) for x in rand_x]

    # Best trajectories
    bayes_cumulative = np.maximum.accumulate(y_sample[3:])
    rand_cumulative = np.maximum.accumulate(rand_scores)

    # Plot Convergence
    fig, ax = plt.subplots(figsize=(8, 4.3))
    iters = np.arange(1, 9)

    ax.plot(iters, bayes_cumulative, "o-", color="#2563EB", linewidth=2.5, label="Bayesian Optimization (GP + UCB)")
    ax.plot(iters, rand_cumulative, "s--", color="#F59E0B", linewidth=2, label="Random Search (Equal Budget)")

    ax.set_xlabel("Sequential Iteration", fontsize=10, fontweight="bold")
    ax.set_ylabel("Best Observed Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_title("Hyperparameter Convergence: Bayes Opt vs Random Search", fontsize=11, fontweight="bold")
    ax.set_xticks(iters)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0")

    fig.tight_layout()
    plot_file = os.path.join(PLOT_DIR, "bayes_opt_convergence.png")
    fig.savefig(plot_file, dpi=120)
    plt.close(fig)

    best_bayes_score = float(np.max(bayes_cumulative))
    best_rand_score = float(np.max(rand_cumulative))
    best_log_c = float(X_sample[np.argmax(y_sample)][0])

    return {
        "n_samples": len(data),
        "best_bayes_score": round(best_bayes_score, 2),
        "best_rand_score": round(best_rand_score, 2),
        "optimal_c": round(float(10.0 ** best_log_c), 4),
        "optimal_log_c": round(best_log_c, 2),
        "iterations": bayes_history
    }
