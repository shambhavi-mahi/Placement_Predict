import os
import pandas as pd
from flask import Flask, render_template

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
    return render_template("feature_engg.html")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)