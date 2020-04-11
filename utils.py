import pandas as pd


def read_csv(path):
    return pd.read_csv(path, encoding='utf-8', header=False, index_col=False)


def save_csv(df: pd.DataFrame, path):
    return df.to_csv(path, encoding='utf-8', header=True, index=False)
