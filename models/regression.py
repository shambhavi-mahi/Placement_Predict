import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SKLinearRegression, LogisticRegression as SKLogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, r2_score

import config

MLR_FEATURES = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]
SLR_FEATURE  = "CGPA"
TARGET_COL   = "Salary Package"

def _run_multilinear_regression(df):
    """Manual Normal-Equation MLR. Returns dict with coefs, metrics, etc."""
    data = df[MLR_FEATURES + [TARGET_COL]].dropna().copy()
    for col in MLR_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data = data.dropna()

    n = len(data)
    X_raw = data[MLR_FEATURES].values.astype(float)
    y     = data[TARGET_COL].values.astype(float)

    # 80/20 split (deterministic: first 80% train)
    split  = int(0.8 * n)
    X_train, X_test = X_raw[:split], X_raw[split:]
    y_train, y_test = y[:split],     y[split:]

    # Standardise using train stats
    mu  = X_train.mean(axis=0)
    sig = X_train.std(axis=0) + 1e-9
    X_tr_s = (X_train - mu) / sig
    X_te_s = (X_test  - mu) / sig

    # Normal equation: theta = (X^T X)^-1 X^T y
    ones   = np.ones((X_tr_s.shape[0], 1))
    X_b    = np.hstack([ones, X_tr_s])
    theta  = np.linalg.lstsq(X_b, y_train, rcond=None)[0]
    intercept = theta[0]
    coefs     = theta[1:]

    # Predict on test
    ones_te = np.ones((X_te_s.shape[0], 1))
    X_b_te  = np.hstack([ones_te, X_te_s])
    y_pred  = X_b_te @ theta

    mse  = float(np.mean((y_test - y_pred) ** 2))
    rmse = math.sqrt(mse)
    mae  = float(np.mean(np.abs(y_test - y_pred)))
    ss_res = float(np.sum((y_test - y_pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2   = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # Sample predictions (first 10 test rows)
    sample = []
    for i in range(min(10, len(y_test))):
        sample.append({"actual": round(float(y_test[i]), 2), "predicted": round(float(y_pred[i]), 2)})

    return {
        "intercept":  round(intercept, 4),
        "coefs":      {f: round(float(c), 4) for f, c in zip(MLR_FEATURES, coefs)},
        "mse":        round(mse, 4),
        "rmse":       round(rmse, 4),
        "mae":        round(mae, 4),
        "r2":         round(r2, 4),
        "n_train":    split,
        "n_test":     n - split,
        "sample":     sample,
        # keep normalisation params for prediction
        "_mu":  mu.tolist(),
        "_sig": sig.tolist(),
        "_theta": theta.tolist(),
    }


def _run_simple_regression(df, alpha=0.01, epochs=1000):
    """Gradient-Descent Simple Linear Regression (CGPA → Salary Package)."""
    data = df[[SLR_FEATURE, TARGET_COL]].dropna().copy()
    data[SLR_FEATURE]  = pd.to_numeric(data[SLR_FEATURE],  errors="coerce")
    data[TARGET_COL]   = pd.to_numeric(data[TARGET_COL],   errors="coerce")
    data = data.dropna()

    n = len(data)
    split = int(0.8 * n)
    X_all = data[SLR_FEATURE].values.astype(float)
    y_all = data[TARGET_COL].values.astype(float)
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    # Normalise
    X_mu, X_sig = X_train.mean(), X_train.std() + 1e-9
    Xn_train = (X_train - X_mu) / X_sig
    Xn_test  = (X_test  - X_mu) / X_sig

    theta0, theta1 = 0.0, 0.0
    m = len(Xn_train)
    cost_history = []
    for _ in range(epochs):
        y_pred_tr = theta0 + theta1 * Xn_train
        error     = y_pred_tr - y_train
        cost_history.append(float(np.mean(error ** 2)))
        theta0 -= alpha * (2 / m) * np.sum(error)
        theta1 -= alpha * (2 / m) * np.sum(error * Xn_train)

    y_pred_te = theta0 + theta1 * Xn_test
    mse  = float(np.mean((y_test - y_pred_te) ** 2))
    rmse = math.sqrt(mse)
    mae  = float(np.mean(np.abs(y_test - y_pred_te)))
    ss_res = float(np.sum((y_test - y_pred_te) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2   = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    # Scatter: sample 200 test points for the chart
    idx = np.linspace(0, len(Xn_test) - 1, min(200, len(Xn_test)), dtype=int)
    scatter_x = (Xn_test[idx] * X_sig + X_mu).tolist()
    scatter_y = y_test[idx].tolist()
    # Regression line: 30 points across training CGPA range
    line_x_norm = np.linspace(Xn_test.min(), Xn_test.max(), 30)
    line_x_real = (line_x_norm * X_sig + X_mu).tolist()
    line_y      = (theta0 + theta1 * line_x_norm).tolist()

    return {
        "intercept": round(float(theta0), 4),
        "slope":     round(float(theta1), 4),
        "mse":       round(mse, 4),
        "rmse":      round(rmse, 4),
        "mae":       round(mae, 4),
        "r2":        round(r2, 4),
        "n_train":   split,
        "n_test":    n - split,
        "epochs":    epochs,
        "alpha":     alpha,
        "cost_history": [round(c, 4) for c in cost_history[::20]],  # every 20th
        "scatter_x":    [round(v, 3) for v in scatter_x],
        "scatter_y":    [round(v, 3) for v in scatter_y],
        "line_x":       [round(v, 3) for v in line_x_real],
        "line_y":       [round(v, 3) for v in line_y],
        # for prediction
        "_theta0": theta0,
        "_theta1": theta1,
        "_X_mu":   X_mu,
        "_X_sig":  X_sig,
    }


def _run_regularization_models(df):
    data = df[MLR_FEATURES + [TARGET_COL]].dropna().copy()
    for col in MLR_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data = data.dropna()

    n = len(data)
    X = data[MLR_FEATURES].values.astype(float)
    y = data[TARGET_COL].values.astype(float)

    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    ss = StandardScaler()
    X_train_s = ss.fit_transform(X_train)
    X_test_s = ss.transform(X_test)

    results = {}
    models = {
        "ridge": Ridge(alpha=1.0, random_state=config.RANDOM_STATE),
        "lasso": Lasso(alpha=0.1, random_state=config.RANDOM_STATE),
        "elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=config.RANDOM_STATE)
    }

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        results[name] = {
            "r2": round(float(r2_score(y_test, preds)), 4),
            "rmse": round(float(mean_squared_error(y_test, preds) ** 0.5), 4),
            "mae": round(float(mean_absolute_error(y_test, preds)), 4)
        }
    return results


LOGR_FEATURES = [
    "CGPA", "AptitudeTestScore", "CodingTestScore",
    "MockInterviewScore", "AttendancePercent",
    "Internships", "Projects",
]
LOGR_TARGET = "PlacementStatus"


def _run_logistic_regression(df):
    """Full sklearn logistic regression pipeline matching the reference code."""
    import warnings
    warnings.filterwarnings("ignore")

    os.makedirs(config.LOGISTIC_DIR, exist_ok=True)

    # ── Prepare data ────────────────────────────────────────────────
    feature_cols = [c for c in LOGR_FEATURES if c in df.columns]
    target_col   = LOGR_TARGET
    needed        = feature_cols + [target_col]
    if "Salary Package" in df.columns:
        needed.append("Salary Package")

    data = df[needed].copy()
    for c in feature_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=feature_cols + [target_col])

    n     = len(data)
    split = int(0.8 * n)
    train = data.iloc[:split].copy()
    val   = data.iloc[split:].copy()

    x_train = train[feature_cols].copy()
    y_train = train[target_col].copy()
    x_val   = val[feature_cols].copy()
    y_val   = val[target_col].copy()

    # Fill missing with train median
    for col in feature_cols:
        fill = x_train[col].median()
        x_train[col] = x_train[col].fillna(fill)
        x_val[col]   = x_val[col].fillna(fill)

    # ── Class balance ────────────────────────────────────────────────
    placed_pct = float((y_train == 1).mean() * 100)

    # ── CGPA vs Placement S-curve ────────────────────────────────────
    bins = pd.cut(train["CGPA"], bins=15) if "CGPA" in train.columns else None
    if bins is not None:
        fraction_placed = train.groupby(bins, observed=True)[target_col].mean()
        bin_centers     = [interval.mid for interval in fraction_placed.index]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(bin_centers, fraction_placed.values, marker="o", color="#2563EB", linewidth=2)
        ax.set_xlabel("CGPA", fontsize=11)
        ax.set_ylabel("Fraction Placed", fontsize=11)
        ax.set_title("CGPA vs Placement — the S-shaped pattern", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(config.LOGISTIC_DIR, "cgpa_vs_placement_curve.png"), dpi=120)
        plt.close(fig)

    # ── Sigmoid function ─────────────────────────────────────────────
    z_vals = np.linspace(-8, 8, 200)
    sig_vals = 1 / (1 + np.exp(-z_vals))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z_vals, sig_vals, color="#2563EB", linewidth=2.5)
    ax.axhline(0.5, color="#94A3B8", linestyle="--", linewidth=1)
    ax.axvline(0,   color="#94A3B8", linestyle="--", linewidth=1)
    ax.set_xlabel("z",            fontsize=11)
    ax.set_ylabel("σ(z)",         fontsize=11)
    ax.set_title("The Sigmoid Function — always between 0 and 1", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.LOGISTIC_DIR, "sigmoid_function.png"), dpi=120)
    plt.close(fig)

    # ── AUC per feature ──────────────────────────────────────────────
    auc_features = ["CGPA", "MockInterviewScore", "CodingTestScore",
                    "AptitudeTestScore", "AttendancePercent"]
    auc_features = [f for f in auc_features if f in feature_cols]
    auc_scores = {}
    for col in auc_features:
        try:
            auc_scores[col] = round(float(roc_auc_score(y_train, x_train[col])), 4)
        except Exception:
            pass

    # ── Why not a straight line? ─────────────────────────────────────
    lin = SKLinearRegression()
    lin.fit(x_train[["CGPA"]], y_train)
    lin_min = round(float(lin.predict(x_train[["CGPA"]]).min()), 2)
    lin_max = round(float(lin.predict(x_train[["CGPA"]]).max()), 2)

    # ── Full logistic regression model ───────────────────────────────
    model = SKLogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    y_pred_val  = model.predict(x_val)
    y_prob_val  = model.predict_proba(x_val)[:, 1]
    val_acc     = round(float(accuracy_score(y_val, y_pred_val)) * 100, 2)
    val_logloss = round(float(log_loss(y_val, model.predict_proba(x_val))), 4)

    # Confusion matrix
    tp = int(np.sum((y_pred_val == 1) & (y_val == 1)))
    tn = int(np.sum((y_pred_val == 0) & (y_val == 0)))
    fp = int(np.sum((y_pred_val == 1) & (y_val == 0)))
    fn = int(np.sum((y_pred_val == 0) & (y_val == 1)))
    precision = round(tp / (tp + fp) * 100, 2) if (tp + fp) else 0.0
    recall    = round(tp / (tp + fn) * 100, 2) if (tp + fn) else 0.0
    f1_val    = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) else 0.0

    # Coefficients
    coefs = {f: round(float(w), 4) for f, w in zip(feature_cols, model.coef_[0])}

    # ── Decision boundary (CGPA + CodingTestScore) ───────────────────
    if "CodingTestScore" in feature_cols:
        bm = SKLogisticRegression(max_iter=1000, random_state=42)
        bm.fit(x_train[["CGPA", "CodingTestScore"]].to_numpy(), y_train)
        x_min_g = x_train["CGPA"].min() - 0.5
        x_max_g = x_train["CGPA"].max() + 0.5
        y_min_g = x_train["CodingTestScore"].min() - 5
        y_max_g = x_train["CodingTestScore"].max() + 5
        xx, yy = np.meshgrid(np.linspace(x_min_g, x_max_g, 200),
                             np.linspace(y_min_g, y_max_g, 200))
        zz = bm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.contourf(xx, yy, zz, alpha=0.2, levels=1, colors=["#FCA5A5", "#93C5FD"])
        sc = ax.scatter(x_val["CGPA"], x_val["CodingTestScore"],
                        c=y_val, cmap="coolwarm", alpha=0.35, s=8)
        ax.set_xlabel("CGPA", fontsize=11)
        ax.set_ylabel("Coding Test Score", fontsize=11)
        ax.set_title("Decision Boundary (CGPA + Coding Test Score)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(config.LOGISTIC_DIR, "decision_boundary.png"), dpi=120)
        plt.close(fig)

    # ── Scaler comparison ────────────────────────────────────────────
    def _eval_logr(Xtr, Xva, ytr, yva):
        m = SKLogisticRegression(max_iter=1000, random_state=42)
        m.fit(Xtr, ytr)
        return round(float(accuracy_score(yva, m.predict(Xva))) * 100, 2)

    scaler_results = {
        "Unscaled":      _eval_logr(x_train, x_val, y_train, y_val),
    }
    ss = StandardScaler()
    scaler_results["StandardScaler"] = _eval_logr(
        ss.fit_transform(x_train), ss.transform(x_val), y_train, y_val)
    mm = MinMaxScaler()
    scaler_results["MinMaxScaler"] = _eval_logr(
        mm.fit_transform(x_train), mm.transform(x_val), y_train, y_val)

    # Scaler bar chart
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(list(scaler_results.keys()), list(scaler_results.values()),
                  color=["#64748B", "#2563EB", "#10B981"], width=0.5, edgecolor="white")
    for bar, acc_val in zip(bars, scaler_results.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{acc_val}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Validation Accuracy (%)", fontsize=10)
    ax.set_title("Scaler Comparison — Same Model, Different Inputs", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.LOGISTIC_DIR, "scaler_comparison.png"), dpi=120)
    plt.close(fig)

    # ── Softmax: 3-class (Not Placed / Standard / Premium) ───────────
    softmax_acc = None
    if "Salary Package" in data.columns:
        salary_median = train.loc[train["Salary Package"] > 0, "Salary Package"].median() \
                        if "Salary Package" in train.columns else 0

        def _tier(s):
            try:
                val = float(s)
                if pd.isna(val) or val == 0:
                    return "Not Placed"
                if val < float(salary_median): 
                    return "Standard Package"
                return "Premium Package"
            except (ValueError, TypeError):
                return "Not Placed"

        y_train_tier = train["Salary Package"].apply(_tier)
        y_val_tier   = val["Salary Package"].apply(_tier)
        ts = StandardScaler()
        X_tr_tier = ts.fit_transform(x_train)
        X_va_tier  = ts.transform(x_val)
        sm = SKLogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        sm.fit(X_tr_tier, y_train_tier)
        softmax_acc = round(float(accuracy_score(y_val_tier, sm.predict(X_va_tier))) * 100, 2)

    # Sample predictions (10 rows)
    sample = []
    for i in range(min(10, len(y_val))):
        sample.append({
            "actual": int(list(y_val)[i]),
            "prob":   round(float(y_prob_val[i]) * 100, 1),
            "pred":   int(y_pred_val[i]),
        })

    # Probability distributions for chart
    y_val_arr = np.array(y_val)
    hist_placed    = [round(float(p), 3) for p in y_prob_val[y_val_arr == 1][:300]]
    hist_notplaced = [round(float(p), 3) for p in y_prob_val[y_val_arr == 0][:300]]

    # Save report
    os.makedirs(config.LOGISTIC_DIR, exist_ok=True)
    with open(os.path.join(config.LOGISTIC_DIR, "logistic_regression_report.txt"), "w") as f:
        f.write(f"Placed: {round(placed_pct,1)}%  Not Placed: {round(100-placed_pct,1)}%\n\n")
        f.write("Single-feature AUC:\n")
        for col, auc in auc_scores.items():
            f.write(f"  {col}: {auc}\n")
        f.write(f"\nFull model validation accuracy: {val_acc}%\n")
        f.write(f"Full model cross-entropy (log loss): {val_logloss}\n")
        f.write("\nScaler comparison:\n")
        for label, acc in scaler_results.items():
            f.write(f"  {label}: {acc}%\n")
        if softmax_acc is not None:
            f.write(f"\nSoftmax 3-class accuracy: {softmax_acc}%\n")

    # For live prediction: store scaler params manually (no pickle needed)
    mu_arr  = np.array(x_train.mean()).tolist()
    sig_arr = np.array(x_train.std() + 1e-9).tolist()

    return {
        "accuracy":      val_acc,
        "precision":     precision,
        "recall":        recall,
        "f1":            f1_val,
        "log_loss":      val_logloss,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n_train":       split,
        "n_test":        n - split,
        "placed_pct":    round(placed_pct, 1),
        "lin_range":     f"{lin_min} → {lin_max}",
        "auc_scores":    auc_scores,
        "coefs":         coefs,
        "intercept":     round(float(model.intercept_[0]), 4),
        "scaler_results":scaler_results,
        "softmax_acc":   softmax_acc,
        "sample":        sample,
        "hist_placed":   hist_placed,
        "hist_notplaced":hist_notplaced,
        "has_plots":     True,
        # For live prediction (mean/std of unscaled train)
        "_mu":    mu_arr,
        "_sig":   sig_arr,
        "_coef":  model.coef_[0].tolist(),
        "_intercept": float(model.intercept_[0]),
        "_features":  feature_cols,
    }
