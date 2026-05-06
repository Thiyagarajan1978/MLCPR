import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df
