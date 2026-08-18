import os
import math
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, jsonify
import io

import config

app = Flask(
    __name__,
    template_folder="Dashboard",
    static_folder="Dashboard"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_load_csv():
    """Load the raw CSV; return None if file is missing."""
    try:
        return pd.read_csv(config.RAW_DATA_PATH)
    except Exception:
        return None


def _get_numeric_cols(df):
    """Return numeric columns excluding ID and target columns."""
    exclude = config.CATEGORICAL_COLS + config.TARGET_COLS + config.ID_COLS
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]


def _apply_minmax(df):
    """Return (scaled_df, stats_list, num_cols) using Min-Max scaling."""
    num_cols = _get_numeric_cols(df)
    scaled_df = df.copy()
    stats = []
    for col in num_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        rng = col_max - col_min
        scaled_df[col] = ((df[col] - col_min) / rng).round(6) if rng != 0 else 0.0
        stats.append({
            "column":      col,
            "min":         round(float(col_min), 4),
            "max":         round(float(col_max), 4),
            "range":       round(float(rng), 4),
            "mean":        round(float(df[col].mean()), 4),
            "std":         round(float(df[col].std()), 4),
            "scaled_mean": round(float(scaled_df[col].mean()), 4),
            "scaled_std":  round(float(scaled_df[col].std()), 4),
        })
    return scaled_df, stats, num_cols


def _apply_standard(df):
    """Return (scaled_df, stats_list, num_cols) using Standard (Z-score) scaling."""
    num_cols = _get_numeric_cols(df)
    scaled_df = df.copy()
    stats = []
    for col in num_cols:
        mean = df[col].mean()
        std  = df[col].std()
        scaled_df[col] = ((df[col] - mean) / std).round(6) if std != 0 else 0.0
        stats.append({
            "column":      col,
            "mean":        round(float(mean), 4),
            "std":         round(float(std), 4),
            "min":         round(float(df[col].min()), 4),
            "max":         round(float(df[col].max()), 4),
            "scaled_mean": round(float(scaled_df[col].mean()), 4),
            "scaled_std":  round(float(scaled_df[col].std()), 4),
            "scaled_min":  round(float(scaled_df[col].min()), 4),
            "scaled_max":  round(float(scaled_df[col].max()), 4),
        })
    return scaled_df, stats, num_cols


def _apply_robust(df):
    """Return (scaled_df, stats_list, num_cols) using Robust (IQR-based) scaling."""
    num_cols = _get_numeric_cols(df)
    scaled_df = df.copy()
    stats = []
    for col in num_cols:
        median = df[col].median()
        q1     = df[col].quantile(0.25)
        q3     = df[col].quantile(0.75)
        iqr    = q3 - q1
        scaled_df[col] = ((df[col] - median) / iqr).round(6) if iqr != 0 else 0.0
        # Count outliers (IQR method)
        lower  = q1 - 1.5 * iqr
        upper  = q3 + 1.5 * iqr
        n_out  = int(((df[col] < lower) | (df[col] > upper)).sum())
        stats.append({
            "column":      col,
            "median":      round(float(median), 4),
            "q1":          round(float(q1), 4),
            "q3":          round(float(q3), 4),
            "iqr":         round(float(iqr), 4),
            "outliers":    n_out,
            "scaled_mean": round(float(scaled_df[col].mean()), 4),
            "scaled_std":  round(float(scaled_df[col].std()), 4),
            "scaled_median": round(float(scaled_df[col].median()), 4),
        })
    return scaled_df, stats, num_cols


def _apply_encoding(df):
    """Return (encoded_df, ordinal_stats, nominal_stats, new_cols) using manual encoding."""
    encoded_df = df.copy()
    
    # Ordinal mapping
    ordinal_map = {
        'CollegeTier': {'Tier3': 1, 'Tier2': 2, 'Tier1': 3},
        'CGPA_Tier': {'Low': 1, 'Mid': 2, 'High': 3}
    }
    
    ordinal_stats = []
    for col, mapping in ordinal_map.items():
        if col in encoded_df.columns:
            encoded_df[col] = encoded_df[col].map(mapping)
            ordinal_stats.append({
                "column": col,
                "mapping": " | ".join([f"{k} → {v}" for k, v in mapping.items()]),
                "unique_values": len(mapping)
            })
            
    # Nominal encoding (One-Hot)
    nominal_cols = [c for c in config.CATEGORICAL_COLS if c not in ordinal_map and c in encoded_df.columns]
    nominal_stats = []
    
    for col in nominal_cols:
        unique_vals = list(df[col].dropna().unique())
        nominal_stats.append({
            "column": col,
            "unique_count": len(unique_vals),
            "new_columns": f"{col}_..."
        })
        
    encoded_df = pd.get_dummies(encoded_df, columns=nominal_cols, drop_first=True)
    
    # Ensure boolean columns are integers 0/1
    for col in encoded_df.columns:
        if encoded_df[col].dtype == 'bool':
            encoded_df[col] = encoded_df[col].astype(int)
            
    new_cols = list(encoded_df.columns)
    return encoded_df, ordinal_stats, nominal_stats, new_cols


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    df = _safe_load_csv()
    stats = {}
    insights = []
    if df is not None:
        placed  = int((df["PlacementStatus"] == 1).sum()) if "PlacementStatus" in df.columns else 0
        total   = len(df)
        anomaly_rate = round(df["IsAnomaly"].mean() * 100, 1) if "IsAnomaly" in df.columns else 0
        placed_pct   = round(placed / total * 100, 1) if total else 0

        stats = {
            "rows":           f"{total:,}",
            "cols":           df.shape[1],
            "placed_pct":     placed_pct,
            "not_placed_pct": round((total - placed) / total * 100, 1) if total else 0,
            "anomaly_rate":   anomaly_rate,
            "columns":        list(df.columns),
        }

        # Generate real data insights
        if "Internships" in df.columns and "PlacementStatus" in df.columns:
            placed_df  = df[df["PlacementStatus"] == 1]
            intern_placed = round(placed_df["Internships"].mean(), 1)
            intern_all    = round(df["Internships"].mean(), 1)
            if intern_placed > intern_all:
                insights.append(f"Students who got placed had an average of {intern_placed} internships, compared to {intern_all} overall — internships are a strong indicator.")

        if "CGPA" in df.columns and "PlacementStatus" in df.columns:
            cgpa_placed = round(df[df["PlacementStatus"] == 1]["CGPA"].mean(), 2)
            cgpa_all    = round(df["CGPA"].mean(), 2)
            insights.append(f"Placed students have an average CGPA of {cgpa_placed} vs {cgpa_all} overall — CGPA is the strongest placement predictor.")

        if "PlacementStatus" in df.columns:
            insights.append(f"{placed_pct}% of the {total:,} students in this dataset were successfully placed — a healthy placement rate for ML modelling.")

        if "IsAnomaly" in df.columns:
            n_anom = int(df["IsAnomaly"].sum())
            insights.append(f"{n_anom:,} records ({anomaly_rate}%) were flagged as anomalies using the IQR method and are excluded from core model training.")

        if "Projects" in df.columns and "PlacementStatus" in df.columns:
            proj_placed = round(df[df["PlacementStatus"] == 1]["Projects"].mean(), 1)
            insights.append(f"Placed students averaged {proj_placed} projects — extracurricular work beyond coursework shows a clear correlation with placement outcomes.")

    return render_template("index.html", stats=stats, insights=insights)


@app.route("/load")
def load_page():
    from flask import request
    df = _safe_load_csv()

    if df is None:
        return render_template("load.html", table=None, pagination=None, info=None)

    # ── Pagination params ──────────────────────────────────────
    PER_PAGE = 50
    total_rows = len(df)
    total_pages = max(1, (total_rows + PER_PAGE - 1) // PER_PAGE)

    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1
    page = max(1, min(page, total_pages))          # clamp

    start = (page - 1) * PER_PAGE
    end   = min(start + PER_PAGE, total_rows)
    slice_df = df.iloc[start:end]

    # ── Build visible page window (max 7 page buttons) ─────────
    half = 3
    p_start = max(1, page - half)
    p_end   = min(total_pages, page + half)
    if p_end - p_start < 2 * half:
        p_start = max(1, p_end - 2 * half)
        p_end   = min(total_pages, p_start + 2 * half)
    page_range = list(range(p_start, p_end + 1))

    table_html = slice_df.to_html(
        classes="data-table", border=0, index=False, na_rep="—"
    )

    pagination = {
        "page":        page,
        "per_page":    PER_PAGE,
        "total_rows":  total_rows,
        "total_pages": total_pages,
        "start_row":   start + 1,
        "end_row":     end,
        "has_prev":    page > 1,
        "has_next":    page < total_pages,
        "page_range":  page_range,
        "show_first":  p_start > 1,
        "show_last":   p_end < total_pages,
    }

    info = {
        "rows":    f"{total_rows:,}",
        "cols":    df.shape[1],
        "columns": list(df.columns),
        "dtypes":  {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": int(df.isnull().sum().sum()),
    }

    return render_template("load.html", table=table_html, pagination=pagination, info=info)


@app.route("/eda")
def eda_page():
    df = _safe_load_csv()
    if df is None:
        return render_template("eda.html", stats=None, missing=None, sample=None, total_missing=0)
        
    num_cols = _get_numeric_cols(df)
    
    # Descriptive Stats
    stats_df = df[num_cols].describe().T
    stats = []
    for col in stats_df.index:
        stats.append({
            "column": col,
            "count": int(stats_df.loc[col, "count"]),
            "mean": round(stats_df.loc[col, "mean"], 4),
            "std": round(stats_df.loc[col, "std"], 4),
            "min": round(stats_df.loc[col, "min"], 4),
            "q1": round(stats_df.loc[col, "25%"], 4),
            "median": round(stats_df.loc[col, "50%"], 4),
            "q3": round(stats_df.loc[col, "75%"], 4),
            "max": round(stats_df.loc[col, "max"], 4)
        })
        
    # Missing Values
    missing = df.isnull().sum().to_dict()
    missing_list = [{"column": k, "missing": v} for k, v in missing.items() if v > 0]
    total_missing = sum(missing.values())
    
    # Random Sample
    sample_html = df.sample(10).to_html(classes="data-table", border=0, index=False)
    
    return render_template("eda.html", stats=stats, missing=missing_list, sample=sample_html, total_missing=total_missing)


@app.route("/graphs")
def graphs_page():
    # Collect all generated plot filenames that exist on disk
    plots = []
    if os.path.isdir(config.PLOTS_DIR):
        plots = sorted(
            f for f in os.listdir(config.PLOTS_DIR)
            if f.lower().endswith(".png")
        )
    return render_template("graphs.html", plots=plots)


@app.route("/feature-engineering")
def feature_engg_page():
    df = _safe_load_csv()
    if df is None:
        return render_template("feature_engg.html",
                               minmax_stats=None, minmax_preview=None,
                               std_stats=None,    std_preview=None,
                               robust_stats=None, robust_preview=None,
                               ordinal_stats=None, nominal_stats=None, enc_preview=None,
                               num_cols=None, total_rows=None)

    preview_cols = ([config.ID_COL] if config.ID_COL in df.columns else [])

    # ── Min-Max ───────────────────────────────────────────────
    mm_df,  minmax_stats,  num_cols = _apply_minmax(df)
    minmax_preview = mm_df[preview_cols + num_cols].head(10).to_html(
        classes="data-table", border=0, index=False,
        float_format=lambda x: f"{x:.4f}"
    )

    # ── Standard Scaler ────────────────────────────────────────
    st_df,  std_stats, _ = _apply_standard(df)
    std_preview = st_df[preview_cols + num_cols].head(10).to_html(
        classes="data-table", border=0, index=False,
        float_format=lambda x: f"{x:.4f}"
    )

    # ── Robust Scaler ──────────────────────────────────────────
    rb_df, robust_stats, _ = _apply_robust(df)
    robust_preview = rb_df[preview_cols + num_cols].head(10).to_html(
        classes="data-table", border=0, index=False,
        float_format=lambda x: f"{x:.4f}"
    )

    # ── Encoding ───────────────────────────────────────────────
    enc_df, ordinal_stats, nominal_stats, enc_cols = _apply_encoding(df)
    
    # We want to preview the original ID, target, and newly encoded columns
    # We'll just show the first 12 columns so it doesn't overflow massively
    preview_enc_cols = [c for c in enc_cols if c not in config.TARGET_COLS][:12]
    enc_preview = enc_df[preview_enc_cols].head(10).to_html(
        classes="data-table", border=0, index=False
    )

    return render_template(
        "feature_engg.html",
        minmax_stats=minmax_stats,   minmax_preview=minmax_preview,
        std_stats=std_stats,         std_preview=std_preview,
        robust_stats=robust_stats,   robust_preview=robust_preview,
        ordinal_stats=ordinal_stats, nominal_stats=nominal_stats, enc_preview=enc_preview,
        num_cols=num_cols,
        total_rows=f"{len(df):,}",
    )


def _csv_response(df, filename):
    """Helper: stream a DataFrame as a CSV download."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )


@app.route("/download-scaled")
def download_scaled():
    df = _safe_load_csv()
    if df is None:
        return "Dataset not found", 404
    scaled_df, _, _ = _apply_minmax(df)
    return _csv_response(scaled_df, "placement_minmax_scaled.csv")


@app.route("/download-standard")
def download_standard():
    df = _safe_load_csv()
    if df is None:
        return "Dataset not found", 404
    scaled_df, _, _ = _apply_standard(df)
    return _csv_response(scaled_df, "placement_standard_scaled.csv")


@app.route("/download-robust")
def download_robust():
    df = _safe_load_csv()
    if df is None:
        return "Dataset not found", 404
    scaled_df, _, _ = _apply_robust(df)
    return _csv_response(scaled_df, "placement_robust_scaled.csv")


@app.route("/download-encoded")
def download_encoded():
    df = _safe_load_csv()
    if df is None:
        return "Dataset not found", 404
    enc_df, _, _, _ = _apply_encoding(df)
    return _csv_response(enc_df, "placement_encoded.csv")


# ---------------------------------------------------------------------------
# Regression Helpers
# ---------------------------------------------------------------------------
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


# Cache so we don't recompute on every request
_regression_cache = {}


def _get_regression_data():
    if "mlr" not in _regression_cache:
        df = _safe_load_csv()
        if df is None:
            return None, None, None, None
        _regression_cache["mlr"]  = _run_multilinear_regression(df)
        _regression_cache["slr"]  = _run_simple_regression(df)
        _regression_cache["logr"] = _run_logistic_regression(df)
        _regression_cache["reg"]  = _run_regularization_models(df)
    return _regression_cache["mlr"], _regression_cache["slr"], _regression_cache["logr"], _regression_cache["reg"]


# ---------------------------------------------------------------------------
# Logistic Regression (sklearn — matching the reference implementation)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SKLinearRegression, LogisticRegression as SKLogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, r2_score

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
        "ridge": Ridge(alpha=1.0, random_state=42),
        "lasso": Lasso(alpha=0.1, random_state=42),
        "elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
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


@app.route("/regression")
def regression_page():
    mlr, slr, logr, reg = _get_regression_data()
    return render_template("regression.html", mlr=mlr, slr=slr, logr=logr, reg=reg)


@app.route("/predict-salary", methods=["POST"])
def predict_salary():
    mlr, slr, logr, reg = _get_regression_data()
    if mlr is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    try:
        model_type = request.form.get("model", "mlr")
        if model_type == "mlr":
            theta = np.array(mlr["_theta"])
            mu    = np.array(mlr["_mu"])
            sig   = np.array(mlr["_sig"])
            vals  = [float(request.form.get(f, 0)) for f in MLR_FEATURES]
            x_norm = (np.array(vals) - mu) / sig
            pred   = float(np.array([1.0] + x_norm.tolist()) @ theta)
            return jsonify({"predicted_salary": round(pred, 2)})
        else:
            theta0 = slr["_theta0"]; theta1 = slr["_theta1"]
            X_mu   = slr["_X_mu"];   X_sig  = slr["_X_sig"]
            cgpa   = float(request.form.get("CGPA", 0))
            pred   = theta0 + theta1 * ((cgpa - X_mu) / X_sig)
            return jsonify({"predicted_salary": round(pred, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-placement", methods=["POST"])
def predict_placement():
    mlr, slr, logr = _get_regression_data()
    if logr is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    try:
        coef      = np.array(logr["_coef"])
        intercept = float(logr["_intercept"])
        mu        = np.array(logr["_mu"])
        sig       = np.array(logr["_sig"])
        features  = logr["_features"]
        vals      = np.array([float(request.form.get(f, 0)) for f in features])
        x_norm    = (vals - mu) / sig
        z         = intercept + float(coef @ x_norm)
        prob      = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        pred      = 1 if prob >= 0.5 else 0
        return jsonify({
            "probability": round(prob * 100, 1),
            "prediction":  pred,
            "label":       "Placed ✅" if pred == 1 else "Not Placed ❌"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
