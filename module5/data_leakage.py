"""Module 5 – Data Leakage & Cross-Validation.

Data leakage happens when information from outside the training set
"leaks" into the model during training, making validation metrics
optimistically inflated and test performance disappointing.

Key concepts covered:
  True Risk (Generalisation Error):
      R_true(f) = E_{(x,y)~P}[ L(f(x), y) ]
      Expected loss over the unknown true joint distribution P(X,Y).

  Empirical Risk (Training Error):
      R_emp(f) = (1/N) * sum_{i=1}^{N} L(f(x_i), y_i)
      Average loss on the observed sample D.

  The gap between R_emp and R_true is what we try to minimise through
  proper splitting, regularisation, and cross-validation.

  Pipeline: encapsulating scaler inside the pipeline ensures the scaler
  is fit only on the training fold, never seeing validation data.

  StratifiedKFold: preserves the class ratio in every fold — critical
  for imbalanced targets like placement status.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import config

# Column mapping to our dataset (mirrors the reference code's intent)
FEATURE_MAP = {
    "CGPA":          "CGPA",
    "Aptitude":      "AptitudeTestScore",
    "Coding":        "CodingTestScore",
    "Communication": "MockInterviewScore",
    "Internship":    "Internships",
    "Projects":      "Projects",
}


def run_data_leakage(df: pd.DataFrame) -> dict:
    # ── Prepare features ───────────────────────────────────────────────────
    available = {nice: col for nice, col in FEATURE_MAP.items() if col in df.columns}
    feat_cols = list(available.values())
    nice_names = list(available.keys())
    target = config.TARGET_COLUMN

    data = df[feat_cols + [target]].dropna().copy()
    # Cap at 15K — LR + CV is fast but no need for all 50K
    if len(data) > 15_000:
        data = data.sample(15_000, random_state=config.RANDOM_STATE)

    X = data[feat_cols].values
    y = data[target].values
    n = len(data)

    # ── 1. Vault-lock 20% test set (stratified) ───────────────────────────
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=config.RANDOM_STATE, stratify=y
    )

    # ── 2. Leakage-free pipeline ───────────────────────────────────────────
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))

    # ── 3. 4-fold Stratified Cross-Validation on dev set ──────────────────
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=config.RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_dev, y_dev, cv=skf, scoring="accuracy")
    mean_cv = round(float(cv_scores.mean()) * 100, 2)
    std_cv  = round(float(cv_scores.std() * 2) * 100, 2)

    # ── 4. Retrain on full dev set → evaluate vault-locked test ───────────
    pipeline.fit(X_dev, y_dev)
    test_acc = round(float(pipeline.score(X_test, y_test)) * 100, 2)
    gap = round(mean_cv - test_acc, 2)

    # Per-fold results
    fold_results = [
        {"fold": i + 1, "accuracy": round(float(s) * 100, 2)}
        for i, s in enumerate(cv_scores)
    ]

    # ── 5. Leakage demo: fit scaler on ALL data (wrong) vs train only ──────
    ss_leak = StandardScaler()
    X_all_scaled = ss_leak.fit_transform(X)
    lr_leak = LogisticRegression(max_iter=300)
    lr_leak.fit(X_all_scaled[:len(X_dev)], y_dev)
    leak_val_acc = round(float(accuracy_score(y_test, lr_leak.predict(X_all_scaled[len(X_dev):]))) * 100, 2)

    ss_ok = StandardScaler()
    lr_ok = LogisticRegression(max_iter=300)
    lr_ok.fit(ss_ok.fit_transform(X_dev), y_dev)
    ok_val_acc = round(float(accuracy_score(y_test, lr_ok.predict(ss_ok.transform(X_test)))) * 100, 2)

    leakage_demo = {
        "leaked_acc": leak_val_acc,
        "clean_acc": ok_val_acc,
        "inflation": round(leak_val_acc - ok_val_acc, 2),
    }

    return {
        "n_total": n,
        "n_dev": len(X_dev),
        "n_test": len(X_test),
        "n_features": len(feat_cols),
        "feature_names": nice_names,
        "mean_cv": mean_cv,
        "std_cv": std_cv,
        "test_acc": test_acc,
        "gap": gap,
        "fold_results": fold_results,
        "leakage_demo": leakage_demo,
    }