from module5.calibration_and_statistical_significance import run_calibration_and_statistical_significance

if __name__ == "__main__":
    import config, pandas as pd
    df = pd.read_csv(config.RAW_DATA_PATH)
    res = run_calibration_and_statistical_significance(df)
    print("39 Calibration and Statistical Significance result:", res)
