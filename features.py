def build_features(df):
    df["range"] = df["high"] - df["low"]
    df["body"] = df["close"] - df["open"]
    df["trend"] = df["close"].diff()
    df["volatility"] = df["range"].rolling(10).mean()
    if "volume" in df.columns:
        vol_ma = df["volume"].rolling(10).mean().replace(0, 1)
        df["vol_ratio"] = df["volume"] / vol_ma
    return df.dropna()
