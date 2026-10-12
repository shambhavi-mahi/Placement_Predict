from module5.confusion_matrix_threshold_tuning import run_confusion_matrix_threshold_tuning

if __name__ == "__main__":
    import config, pandas as pd
    df = pd.read_csv(config.RAW_DATA_PATH)
    res = run_confusion_matrix_threshold_tuning(df)
    print("33 Confusion Matrix Threshold Tuning result:", res)
