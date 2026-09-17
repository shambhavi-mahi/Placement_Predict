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


from models.feature_engg import _apply_minmax, _apply_standard, _apply_robust, _apply_encoding
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
from models.regression import _run_multilinear_regression, _run_simple_regression, _run_regularization_models, _run_logistic_regression, MLR_FEATURES
from models.decision_tree import _run_tree_models

from models.feature_engg import _apply_minmax, _apply_standard, _apply_robust, _apply_encoding
# ---------------------------------------------------------------------------
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


@app.route("/decision-tree")
def decision_tree_page():
    df = _safe_load_csv()
    dt_data = None
    if df is not None:
        dt_data = _run_tree_models(df)
    return render_template("decision_tree.html", dt=dt_data)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
