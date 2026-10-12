from module5.bayesian_optimization import run_bayesian_optimization

if __name__ == "__main__":
    import config, pandas as pd
    df = pd.read_csv(config.RAW_DATA_PATH)
    res = run_bayesian_optimization(df)
    print("38 Bayesian Optimization result:", res)
