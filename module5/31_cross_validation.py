from module5.cross_validation import run_cross_validation

if __name__ == "__main__":
    import config, pandas as pd
    df = pd.read_csv(config.RAW_DATA_PATH)
    res = run_cross_validation(df)
    print("31 Cross-Validation result:", res)
