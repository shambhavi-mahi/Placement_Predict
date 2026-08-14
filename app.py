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
    if df is not None:
        placed = int((df["PlacementStatus"] == 1).sum()) if "PlacementStatus" in df.columns else 0
        total  = len(df)
        anomaly_rate = round(df["IsAnomaly"].mean() * 100, 1) if "IsAnomaly" in df.columns else 0
        stats = {
            "rows":         f"{total:,}",
            "cols":         df.shape[1],
            "placed_pct":   round(placed / total * 100, 1) if total else 0,
            "not_placed_pct": round((total - placed) / total * 100, 1) if total else 0,
            "anomaly_rate": anomaly_rate,
            "columns":      list(df.columns),
        }
    return render_template("index.html", stats=stats)


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
            return None, None
        _regression_cache["mlr"] = _run_multilinear_regression(df)
        _regression_cache["slr"] = _run_simple_regression(df)
    return _regression_cache["mlr"], _regression_cache["slr"]


@app.route("/regression")
def regression_page():
    mlr, slr = _get_regression_data()
    return render_template("regression.html", mlr=mlr, slr=slr)


@app.route("/predict-salary", methods=["POST"])
def predict_salary():
    mlr, slr = _get_regression_data()
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
            x_b    = np.array([1.0] + x_norm.tolist())
            pred   = float(x_b @ theta)
        else:
            theta0 = slr["_theta0"]; theta1 = slr["_theta1"]
            X_mu   = slr["_X_mu"];   X_sig  = slr["_X_sig"]
            cgpa   = float(request.form.get("CGPA", 0))
            x_norm = (cgpa - X_mu) / X_sig
            pred   = theta0 + theta1 * x_norm

        return jsonify({"predicted_salary": round(pred, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)