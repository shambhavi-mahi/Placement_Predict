from module5.regression_metrics_and_robustness import run_regression_metrics_and_robustness

if __name__ == "__main__":
    import config, pandas as pd
    df = pd.read_csv(config.RAW_DATA_PATH)
    res = run_regression_metrics_and_robustness(df)
    print("35 Regression Metrics and Robustness result:", res)
