import os
import pandas as pd
from flask import Flask, render_template, request, send_file
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
    """Return (scaled_df, stats_list) using manual MinMax (no sklearn needed)."""
    num_cols = _get_numeric_cols(df)
    scaled_df = df.copy()
    stats = []
    for col in num_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        rng = col_max - col_min
        scaled_df[col] = ((df[col] - col_min) / rng).round(6) if rng != 0 else 0.0
        stats.append({
            "column":  col,
            "min":     round(float(col_min), 4),
            "max":     round(float(col_max), 4),
            "range":   round(float(rng), 4),
            "mean":    round(float(df[col].mean()), 4),
            "std":     round(float(df[col].std()), 4),
            "scaled_mean": round(float(scaled_df[col].mean()), 4),
            "scaled_std":  round(float(scaled_df[col].std()), 4),
        })
    return scaled_df, stats, num_cols


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
    # Collect all generated plot filenames that exist on disk
    plots = []
    if os.path.isdir(config.PLOTS_DIR):
        plots = sorted(
            f for f in os.listdir(config.PLOTS_DIR)
            if f.lower().endswith(".png")
        )
    return render_template("eda.html", plots=plots)


@app.route("/feature-engineering")
def feature_engg_page():
    df = _safe_load_csv()
    if df is None:
        return render_template("feature_engg.html",
                               scaling_stats=None, scaled_preview=None,
                               num_cols=None, total_rows=None)

    scaled_df, scaling_stats, num_cols = _apply_minmax(df)

    # Preview: first 8 rows, only numeric cols + StudentID
    preview_cols = ([config.ID_COL] if config.ID_COL in df.columns else []) + num_cols
    scaled_preview = scaled_df[preview_cols].head(8).to_html(
        classes="data-table", border=0, index=False,
        float_format=lambda x: f"{x:.4f}"
    )

    return render_template(
        "feature_engg.html",
        scaling_stats=scaling_stats,
        scaled_preview=scaled_preview,
        num_cols=num_cols,
        total_rows=f"{len(df):,}",
    )


@app.route("/download-scaled")
def download_scaled():
    """Stream the full MinMax-scaled CSV as a download."""
    df = _safe_load_csv()
    if df is None:
        return "Dataset not found", 404

    scaled_df, _, _ = _apply_minmax(df)

    buf = io.StringIO()
    scaled_df.to_csv(buf, index=False)
    buf.seek(0)

    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="placement_minmax_scaled.csv"
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)