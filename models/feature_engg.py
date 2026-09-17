import pandas as pd
import config

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
