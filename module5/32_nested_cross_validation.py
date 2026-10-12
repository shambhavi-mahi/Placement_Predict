from module5.nested_cross_validation import run_nested_cross_validation

if __name__ == "__main__":
    import config, pandas as pd
    df = pd.read_csv(config.RAW_DATA_PATH)
    res = run_nested_cross_validation(df)
    print("32 Nested Cross-Validation result:", res)
