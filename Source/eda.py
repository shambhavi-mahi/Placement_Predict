import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import config
from src.data_utils import load_raw, load_cleaned, numeric_cols

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# ---------------------------------------------------------------------------
# Plot generation (same 11 charts)
# ---------------------------------------------------------------------------
def _save(fig, filename):
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(config.PLOTS_DIR, filename), bbox_inches="tight", dpi=120)
    plt.close(fig)


def generate_all_plots(df=None):
    df = df if df is not None else load_raw()
    num_cols = numeric_cols(df)

    # 01 missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        fig, ax = plt.subplots()
        sns.barplot(x=missing.index, y=(missing / len(df) * 100), ax=ax, color="indianred")
        ax.set_title("Missing Values by Column (%)")
        plt.xticks(rotation=45, ha="right")
        _save(fig, "01_missing_values.png")

    # 02 target balance
    fig, ax = plt.subplots()
    sns.countplot(x="PlacementStatus", data=df, hue="PlacementStatus",
                   palette=["#e74c3c", "#2ecc71"], legend=False, ax=ax)
    ax.set_title("Placement Status Distribution (0 = Not Placed, 1 = Placed)")
    _save(fig, "02_placement_status_distribution.png")

    # 03 numeric distributions
    dist_cols = ["CGPA", "AttendancePercent", "AptitudeTestScore",
                 "CodingTestScore", "MockInterviewScore", "SoftSkillsRating"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, col in zip(axes.flatten(), dist_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {col}")
    fig.tight_layout()
    _save(fig, "03_numeric_distributions.png")

    # 04 outlier boxplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    box_cols = ["CGPA", "AttendancePercent", "CodingTestScore",
                "AptitudeTestScore", "MockInterviewScore", "Salary Package"]
    for ax, col in zip(axes.flatten(), box_cols):
        sns.boxplot(y=df[col], ax=ax, color="lightsalmon")
        ax.set_title(f"Boxplot - {col}")
    fig.tight_layout()
    _save(fig, "04_outlier_boxplots.png")

    # 05 correlation heatmap
    corr = df[num_cols + ["PlacementStatus"]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap - Numeric Features")
    _save(fig, "05_correlation_heatmap.png")

    # 06 placement rate by category
    cat_cols = ["CollegeTier", "Stream", "Specialisation", "Hostel", "CGPA_Tier"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for ax, col in zip(axes.flatten(), cat_cols):
        rate = df.groupby(col)["PlacementStatus"].mean().sort_values(ascending=False) * 100
        sns.barplot(x=rate.index, y=rate.values, ax=ax, color="mediumseagreen")
        ax.set_title(f"Placement Rate (%) by {col}")
        ax.tick_params(axis="x", rotation=45)
    axes.flatten()[-1].axis("off")
    fig.tight_layout()
    _save(fig, "06_placement_rate_by_category.png")

    # 07 numeric vs placement
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, col in zip(axes, ["CGPA", "AttendancePercent", "CodingTestScore"]):
        sns.boxplot(x="PlacementStatus", y=col, data=df, hue="PlacementStatus",
                    palette=["#e74c3c", "#2ecc71"], legend=False, ax=ax)

        ax.set_title(f"{col} vs Placement Status")
        fig.tight_layout()
        _save(fig, "07_numeric_vs_placement_boxplots.png")

        # 08 pairplot (sampled)
        sample_df = df.sample(n=min(2000, len(df)), random_state=42)
        pair_cols = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "PlacementStatus"]
        pp = sns.pairplot(sample_df[pair_cols], hue="PlacementStatus",
                          palette=["#e74c3c", "#2ecc71"], plot_kws={"alpha": 0.5, "s": 15})
        pp.fig.suptitle("Pairwise Relationships (2000-row sample)", y=1.02)
        pp.savefig(os.path.join(config.PLOTS_DIR, "08_pairplot.png"), bbox_inches="tight", dpi=120)
        plt.close(pp.fig)

        # 09 / 10 salary
        placed_df = df[df["PlacementStatus"] == 1]
        fig, ax = plt.subplots()
        sns.histplot(placed_df["Salary Package"], kde=True, color="darkorange", ax=ax)
        ax.set_title("Salary Package Distribution (Placed Students Only)")
        _save(fig, "09_salary_distribution.png")

        fig, ax = plt.subplots()
        sns.boxplot(x="CollegeTier", y="Salary Package", data=placed_df,
                    order=["Tier1", "Tier2", "Tier3"], color="gold", ax=ax)
        ax.set_title("Salary Package by College Tier (Placed Students)")
        _save(fig, "10_salary_by_college_tier.png")

        # 11 SGPA trend
        sgpa_cols = [f"SGPA_Sem{i}" for i in range(1, 9)]
        sgpa_by_placement = df.groupby("PlacementStatus")[sgpa_cols].mean()
        fig, ax = plt.subplots()
        for status, label, color in [(0, "Not Placed", "#e74c3c"), (1, "Placed", "#2ecc71")]:
            ax.plot(range(1, 9), sgpa_by_placement.loc[status], marker="o", label=label, color=color)
        ax.set_xlabel("Semester")
        ax.set_ylabel("Average SGPA")
        ax.set_title("Average SGPA Trend by Placement Status")
        ax.legend()
        _save(fig, "11_sgpa_trend.png")

    # ---------------------------------------------------------------------------
    # Stats for each dashboard page (no plotting - templates read these dicts)
    # ---------------------------------------------------------------------------
    def get_overview_stats():
        df = load_raw()
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        return {
            "rows": df.shape[0],
            "cols": df.shape[1],
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_ids": int(df.duplicated(subset=[config.ID_COL]).sum()),
            "missing_table": [
                {"column": c, "count": int(v), "percent": round(v / len(df) * 100, 2)}
                for c, v in missing.items()
            ],
            "placement_counts": df["PlacementStatus"].value_counts().to_dict(),
            "placement_pct": df["PlacementStatus"].value_counts(normalize=True).mul(100).round(2).to_dict(),
            "anomaly_rate": round(df["IsAnomaly"].mean() * 100, 2),
            "dtypes": [{"column": c, "dtype": str(t)} for c, t in df.dtypes.items()],
        }

    def get_univariate_stats():
        df = load_raw()
        num_cols = numeric_cols(df)
        desc = df[num_cols].describe().T
        desc["skew"] = df[num_cols].skew()
        desc = desc.round(2)
        numeric_table = [
            {"column": c, **{k: (row[k] if pd.notna(row[k]) else None) for k in
                             ["mean", "std", "min", "25%", "50%", "75%", "max", "skew"]}}
            for c, row in desc.iterrows()
        ]
        categorical_freq = {
            col: df[col].value_counts(dropna=False).to_dict()
            for col in config.CATEGORICAL_COLS
        }
        return {"numeric_table": numeric_table, "categorical_freq": categorical_freq}

    def get_bivariate_stats():
        df = load_raw()
        cat_cols = ["CollegeTier", "Stream", "Specialisation", "Hostel", "CGPA_Tier"]
        placement_rate = {
            col: df.groupby(col)["PlacementStatus"].mean().mul(100).round(2)
            .sort_values(ascending=False).to_dict()
            for col in cat_cols
        }
        outlier_summary = {}
        for col in numeric_cols(df):
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_summary[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
            outlier_summary = dict(sorted(outlier_summary.items(), key=lambda x: -x[1]))
            return {"placement_rate": placement_rate, "outliers": outlier_summary}

        def get_multivariate_stats():
            df = load_raw()
            placed = df[df["PlacementStatus"] == 1]
            sgpa_cols = [f"SGPA_Sem{i}" for i in range(1, 9)]
            sgpa_trend = df.groupby("PlacementStatus")[sgpa_cols].mean().round(2).to_dict(orient="index")
            return {
                "salary_desc": placed["Salary Package"].describe().round(2).to_dict(),
                "sgpa_trend": sgpa_trend,
            }

        def get_correlation_stats():
            df = load_raw()
            num_cols = numeric_cols(df)
            corr = df[num_cols + ["PlacementStatus"]].corr(numeric_only=True)
            top_corr = corr["PlacementStatus"].drop("PlacementStatus").sort_values(ascending=False).round(3)
            return {"top_correlations": top_corr.to_dict()}