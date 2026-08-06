"""
Shared data-access helpers.
Every route that needs the dataset goes through here instead of
calling pandas.read_csv directly, so the cleaning rules live in
exactly one place.
"""

import os
import pandas as pd
import config


def load_raw():
    """Return the untouched dataset exactly as collected."""
    return pd.read_csv(config.RAW_DATA_PATH)

def clean_data(df=None, save=True):
    """
    Apply the cleaning rules decided during EDA (Day 1):
      - Workshops / AptitudeTestScore / SoftSkillsRating / CodingTestScore /
        MockInterviewScore have missing values -> median-impute per column
        (all are numeric, right-skew is mild, median is safe against outliers).
      - No duplicate rows / StudentIDs were found, so no dedup step is needed,
        but the check is kept here so future data drops still get verified.
    Returns the cleaned DataFrame and, by default, writes it to
    data/processed/cleaned_data.csv.
    """
    if df is None:
        df = load_raw()
    df = df.copy()

    impute_cols = ["Workshops", "AptitudeTestScore", "SoftSkillsRating",
                   "CodingTestScore", "MockInterviewScore"]
    for col in impute_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    before = len(df)
    df = df.drop_duplicates(subset=[config.ID_COL])
    dropped = before - len(df)

    if save:
        os.makedirs(os.path.dirname(config.CLEANED_DATA_PATH), exist_ok=True)
        df.to_csv(config.CLEANED_DATA_PATH, index=False)

    return df, dropped


def load_cleaned():
    """Return the cleaned dataset, generating it on first call if missing."""
    if not os.path.exists(config.CLEANED_DATA_PATH):
        df, _ = clean_data()
        return df
    return pd.read_csv(config.CLEANED_DATA_PATH)


def numeric_cols(df):
    return [c for c in df.columns
            if c not in config.CATEGORICAL_COLS + config.TARGET_COLS + [config.ID_COL]]