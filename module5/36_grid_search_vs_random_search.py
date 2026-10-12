from module5.grid_search_vs_random_search import run_grid_search_vs_random_search

if __name__ == "__main__":
    import config, pandas as pd
    df = pd.read_csv(config.RAW_DATA_PATH)
    res = run_grid_search_vs_random_search(df)
    print("36 Grid Search vs Random Search result:", res)
