import os
import io
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, jsonify

import config

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
# Regression helpers
# ---------------------------------------------------------------------------
REGRESSION_TARGET = "Salary Package"
_REG_EXCLUDE = ["StudentID", "PlacementStatus", "Salary Package", "IsAnomaly", "CGPA_Tier"]

# ─── 4 features used for Multi-Linear Regression (matches user code) ────────
MLR_FEATURES = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"]


def _build_reg_preprocessor(X):
    """Build a ColumnTransformer that StandardScales numerics and OHE categoricals."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", ohe)])
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers, remainder="drop"), num_cols, cat_cols


def _batch_gradient_descent(X, y, lr=0.01, epochs=2000):
    """Pure-numpy Batch Gradient Descent. Returns (theta, cost_history)."""
    m, n = X.shape
    theta = np.zeros(n)
    history = []
    for _ in range(epochs):
        err = X @ theta - y
        history.append(float(np.mean(err ** 2)))
        theta -= (2 * lr / m) * (X.T @ err)
    return theta, history


def _simple_gd(X_1d, y, lr=0.01, epochs=1000):
    """
    Simple (univariate) Gradient Descent — matches user's code exactly.
    X_1d: 1-D numpy array.
    Returns (theta0, theta1, cost_history).
    """
    m = len(X_1d)
    theta0, theta1 = 0.0, 0.0
    history = []
    for _ in range(epochs):
        y_pred = theta0 + theta1 * X_1d
        err = y_pred - y
        history.append(float(np.mean(err ** 2)))
        theta0 -= (2 / m) * lr * np.sum(err)
        theta1 -= (2 / m) * lr * np.sum(err * X_1d)
    return theta0, theta1, history


def _run_multilinear(df):
    """
    Multilinear regression with the 4 explicit features from the user's code.
    Also runs a simple 1-feature GD (CGPA → Salary) to mirror both snippets.
    """
    feat_cols = [c for c in MLR_FEATURES if c in df.columns]
    data = df[feat_cols + [REGRESSION_TARGET]].dropna()
    X_df = data[feat_cols]
    y_arr = data[REGRESSION_TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_arr, test_size=0.20, random_state=42
    )

    # ── Sklearn Multi-Linear Regression ──────────────────────────────
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    y_pred = mlr.predict(X_test)

    mlr_metrics = {
        "mse":  round(float(mean_squared_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 4),
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
    }

    coefficients = [
        {"feature": feat, "coeff": round(float(c), 6)}
        for feat, c in zip(feat_cols, mlr.coef_)
    ]
    intercept = round(float(mlr.intercept_), 4)

    # Actual vs predicted (first 30 test rows)
    avp = [
        {"actual": round(float(a), 2), "predicted": round(float(p), 2),
         "residual": round(float(a - p), 2)}
        for a, p in zip(y_test[:30], y_pred[:30])
    ]

    # ── Simple GD: CGPA only → Salary ────────────────────────────────
    cgpa_arr = data["CGPA"].to_numpy() if "CGPA" in feat_cols else np.array([])
    simple_theta0, simple_theta1, simple_cost = 0.0, 0.0, []
    if len(cgpa_arr) > 0:
        simple_theta0, simple_theta1, simple_cost = _simple_gd(cgpa_arr, y_arr, lr=0.01, epochs=1000)

    # cost every 50 epochs for the chart
    cost_curve = [{"epoch": i + 1, "cost": round(c, 2)}
                  for i, c in enumerate(simple_cost) if i % 50 == 0]

    # Sample CGPA line: predict over [6, 6.5 … 9]
    cgpa_range = [round(v * 0.5, 1) for v in range(12, 19)]  # 6.0–9.0
    regression_line = [{"cgpa": x, "salary": round(simple_theta0 + simple_theta1 * x, 2)}
                       for x in cgpa_range]

    # Sample prediction: CGPA = 7.8
    pred_cgpa78 = round(simple_theta0 + simple_theta1 * 7.8, 2)

    return {
        "features": feat_cols,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "intercept": intercept,
        "coefficients": coefficients,
        "metrics": mlr_metrics,
        "avp": avp,
        "simple_theta0": round(simple_theta0, 6),
        "simple_theta1": round(simple_theta1, 6),
        "simple_pred_cgpa78": pred_cgpa78,
        "cost_curve": cost_curve,
        "regression_line": regression_line,
        # keep model for AJAX predict
        "_mlr": mlr,
        "_feat_cols": feat_cols,
    }


def _run_regression(df):
    """Train all three linear regression variants and return a results dict."""
    reg_df = df[df[REGRESSION_TARGET].notna()].copy()
    reg_df[REGRESSION_TARGET] = pd.to_numeric(reg_df[REGRESSION_TARGET], errors="coerce")
    reg_df = reg_df[reg_df[REGRESSION_TARGET] > 0].dropna(subset=[REGRESSION_TARGET])

    drop_cols = [c for c in _REG_EXCLUDE if c in reg_df.columns]
    X = reg_df.drop(columns=drop_cols)
    y = reg_df[REGRESSION_TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor, num_cols, cat_cols = _build_reg_preprocessor(X_train)
    X_tr = preprocessor.fit_transform(X_train)
    X_te = preprocessor.transform(X_test)

    # ── 1. Sklearn Linear Regression ─────────────────────────────
    sk_model = LinearRegression()
    sk_model.fit(X_tr, y_train)
    y_pred_sk = sk_model.predict(X_te)
    sk_metrics = {
        "mse":  round(float(mean_squared_error(y_test, y_pred_sk)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_sk))), 4),
        "mae":  round(float(mean_absolute_error(y_test, y_pred_sk)), 4),
        "r2":   round(float(r2_score(y_test, y_pred_sk)), 4),
    }

    # ── 2. Normal Equation ────────────────────────────────────────
    X_ne = np.c_[np.ones(X_tr.shape[0]), X_tr]
    X_ne_test = np.c_[np.ones(X_te.shape[0]), X_te]
    try:
        theta_ne = np.linalg.solve(X_ne.T @ X_ne, X_ne.T @ y_train)
    except np.linalg.LinAlgError:
        theta_ne = np.linalg.pinv(X_ne) @ y_train
    y_pred_ne = X_ne_test @ theta_ne
    ne_metrics = {
        "mse":  round(float(mean_squared_error(y_test, y_pred_ne)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_ne))), 4),
        "mae":  round(float(mean_absolute_error(y_test, y_pred_ne)), 4),
        "r2":   round(float(r2_score(y_test, y_pred_ne)), 4),
    }

    # ── 3. Batch Gradient Descent ────────────────────────────────
    X_gd = np.c_[np.ones(X_tr.shape[0]), X_tr]
    X_gd_test = np.c_[np.ones(X_te.shape[0]), X_te]
    theta_gd, cost_history = _batch_gradient_descent(X_gd, y_train, lr=0.01, epochs=2000)
    y_pred_gd = X_gd_test @ theta_gd
    gd_metrics = {
        "mse":  round(float(mean_squared_error(y_test, y_pred_gd)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred_gd))), 4),
        "mae":  round(float(mean_absolute_error(y_test, y_pred_gd)), 4),
        "r2":   round(float(r2_score(y_test, y_pred_gd)), 4),
    }

    # ── Actuals vs Predicted (first 30 test rows) ─────────────────
    actual_vs_pred = [
        {"actual": round(float(a), 2), "predicted": round(float(p), 2), "residual": round(float(a - p), 2)}
        for a, p in zip(y_test[:30], y_pred_sk[:30])
    ]

    # ── Top-10 coefficients by absolute weight ────────────────────
    try:
        feat_names = preprocessor.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_tr.shape[1])]
    coeff_df = pd.DataFrame({"feature": feat_names, "coeff": sk_model.coef_})
    coeff_df["abs"] = coeff_df["coeff"].abs()
    top10 = coeff_df.nlargest(10, "abs")[["feature", "coeff"]].to_dict("records")
    for row in top10:
        row["coeff"] = round(float(row["coeff"]), 4)

    # ── GD cost curve (every 50 epochs) ──────────────────────────
    cost_curve = [{"epoch": i + 1, "cost": round(c, 2)} for i, c in enumerate(cost_history) if i % 50 == 0]

    return {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "total_rows": len(reg_df),
        "num_features": X_tr.shape[1],
        "sklearn": sk_metrics,
        "normal_eq": ne_metrics,
        "grad_desc": gd_metrics,
        "actual_vs_pred": actual_vs_pred,
        "top_coefficients": top10,
        "cost_curve": cost_curve,
        "intercept": round(float(sk_model.intercept_), 4),
        "_preprocessor": preprocessor,
        "_sk_model": sk_model,
        "_X_cols": list(X.columns),
    }


# Store fitted models in-memory
_reg_cache = {}


@app.route("/regression")
def regression_page():
    df = _safe_load_csv()
    if df is None:
        return render_template("regression.html", result=None, mlr=None)
    result = _run_regression(df)
    _reg_cache["preprocessor"] = result.pop("_preprocessor")
    _reg_cache["sk_model"]     = result.pop("_sk_model")
    _reg_cache["X_cols"]       = result.pop("_X_cols")
    mlr = _run_multilinear(df)
    _reg_cache["mlr"]       = mlr.pop("_mlr")
    _reg_cache["feat_cols"] = mlr.pop("_feat_cols")
    return render_template("regression.html", result=result, mlr=mlr)


@app.route("/predict-salary", methods=["POST"])
def predict_salary():
    """AJAX endpoint — full-feature model."""
    if "sk_model" not in _reg_cache:
        return jsonify({"error": "Model not trained yet. Visit /regression first."}), 400
    data = request.get_json(force=True)
    df_full = _safe_load_csv()
    if df_full is None:
        return jsonify({"error": "Dataset unavailable."}), 500
    drop_cols = [c for c in _REG_EXCLUDE if c in df_full.columns]
    X_ref = df_full.drop(columns=drop_cols)
    row = {}
    for col in _reg_cache["X_cols"]:
        if col in data and data[col] != "":
            row[col] = data[col]
        elif X_ref[col].dtype == "object":
            row[col] = X_ref[col].mode().iloc[0]
        else:
            row[col] = float(X_ref[col].median())
    new_df = pd.DataFrame([row])[_reg_cache["X_cols"]]
    X_proc = _reg_cache["preprocessor"].transform(new_df)
    salary = float(_reg_cache["sk_model"].predict(X_proc)[0])
    return jsonify({"predicted_salary": round(salary, 2)})


@app.route("/predict-salary-multi", methods=["POST"])
def predict_salary_multi():
    """AJAX endpoint — 4-feature multi-linear model."""
    if "mlr" not in _reg_cache:
        return jsonify({"error": "Model not trained yet. Visit /regression first."}), 400
    data = request.get_json(force=True)
    row = [[
        float(data.get("CGPA", 7.5)),
        float(data.get("AptitudeTestScore", 80)),
        float(data.get("CodingTestScore", 85)),
        float(data.get("MockInterviewScore", 75)),
    ]]
    salary = float(_reg_cache["mlr"].predict(row)[0])
    return jsonify({"predicted_salary": round(salary, 2)})


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)