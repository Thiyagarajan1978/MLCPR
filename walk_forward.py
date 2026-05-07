from model import train

_OHLC    = ["open", "high", "low", "close", "volume"]
_RAW_CPR = ["pp", "cpr_top", "cpr_bottom", "r1", "r2", "s1", "s2", "cpr_width"]
_EXCLUDE = set(["label"] + _OHLC + _RAW_CPR)


def run_walk_forward(df, train_size=200, test_size=50):
    results = []
    start = 0

    while start + train_size + test_size < len(df):
        train_df = df.iloc[start : start + train_size]
        test_df  = df.iloc[start + train_size : start + train_size + test_size]

        drop = [c for c in _EXCLUDE if c in train_df.columns]
        X_train = train_df.drop(columns=drop)
        y_train = train_df["label"]
        X_test  = test_df.drop(columns=drop)
        y_test  = test_df["label"]

        model = train(X_train, y_train)
        preds = model.predict(X_test)
        results.append((preds == y_test).mean())
        start += test_size

    return results
