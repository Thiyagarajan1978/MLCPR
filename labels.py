def create_labels(df, horizon=3):
    df["future"] = df["close"].shift(-horizon)
    df["label"] = (df["future"] > df["close"]).astype(int)
    df = df.drop("future", axis=1)  # prevent data leak into feature matrix
    return df.dropna()
