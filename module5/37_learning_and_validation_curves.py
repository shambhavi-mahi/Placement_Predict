from module5.learning_and_validation_curves import run_learning_and_validation_curves

if __name__ == "__main__":
    import config, pandas as pd
    df = pd.read_csv(config.RAW_DATA_PATH)
    res = run_learning_and_validation_curves(df)
    print("37 Learning and Validation Curves result:", res)
