"""Module 5 – Topic 32: Nested Cross-Validation.

Implements Outer Loop (unbiased generalization evaluation) and Inner Loop
(hyperparameter selection) to eliminate optimization bias.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import config


def run_nested_cross_validation(df: pd.DataFrame) -> dict:
    target = config.TARGET_COLUMN
    feat_cols = [c for c in config.NUMERIC_COLUMNS if c in df.columns]

    data = df[feat_cols + [target]].dropna().copy()
    if len(data) > 4000:
        data = data.sample(4000, random_state=config.RANDOM_STATE)

    X = data[feat_cols].values
    y = data[target].astype(int).values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=300, random_state=config.RANDOM_STATE))
    ])

    param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}

    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=config.RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)

    outer_results = []
    outer_scores = []
    inner_best_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        grid_search = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="accuracy")
        grid_search.fit(X_train_outer, y_train_outer)

        best_c = grid_search.best_params_["clf__C"]
        inner_score = float(grid_search.best_score_) * 100
        outer_score = float(grid_search.score(X_test_outer, y_test_outer)) * 100

        outer_scores.append(outer_score)
        inner_best_scores.append(inner_score)

        outer_results.append({
            "outer_fold": fold_idx + 1,
            "best_c": best_c,
            "inner_cv_acc": round(inner_score, 2),
            "outer_test_acc": round(outer_score, 2),
            "difference": round(outer_score - inner_score, 2)
        })

    # Standard non-nested grid search on full dataset
    full_grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="accuracy")
    full_grid.fit(X, y)
    non_nested_score = float(full_grid.best_score_) * 100

    nested_mean = float(np.mean(outer_scores))
    optimism_gap = non_nested_score - nested_mean

    return {
        "n_samples": len(data),
        "outer_folds": 4,
        "inner_folds": 3,
        "nested_mean": round(nested_mean, 2),
        "nested_std": round(float(np.std(outer_scores)), 2),
        "non_nested_score": round(non_nested_score, 2),
        "optimism_gap": round(optimism_gap, 2),
        "best_overall_c": full_grid.best_params_["clf__C"],
        "fold_rows": outer_results
    }
