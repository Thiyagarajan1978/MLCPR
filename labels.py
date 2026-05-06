def create_labels(df, horizon=3):
    df["future"] = df["close"].shift(-horizon)
    df["label"] = (df["future"] > df["close"]).astype(int)
    return df.dropna()
