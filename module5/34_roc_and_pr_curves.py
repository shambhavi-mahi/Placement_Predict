from module5.roc_and_pr_curves import run_roc_and_pr_curves

if __name__ == "__main__":
    import config, pandas as pd
    df = pd.read_csv(config.RAW_DATA_PATH)
    res = run_roc_and_pr_curves(df)
    print("34 ROC and PR Curves result:", res)
