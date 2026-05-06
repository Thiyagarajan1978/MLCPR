def build_features(df):
    df["range"] = df["high"] - df["low"]
    df["body"] = df["close"] - df["open"]
    df["trend"] = df["close"].diff()
    df["volatility"] = df["range"].rolling(10).mean()
    return df.dropna()
