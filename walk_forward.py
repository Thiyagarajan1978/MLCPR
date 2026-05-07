from model import train

# Same exclusion as main.py — OHLC absolute prices must not be in the feature matrix
_OHLC = ["open", "high", "low", "close", "volume"]


def run_walk_forward(df, train_size=200, test_size=50):
    results = []
    start = 0

    while start + train_size + test_size < len(df):
        train_df = df.iloc[start:start + train_size]
        test_df = df.iloc[start + train_size:start + train_size + test_size]

        drop_cols = ["label"] + [c for c in _OHLC if c in train_df.columns]
        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df["label"]

        X_test = test_df.drop(columns=drop_cols)
        y_test = test_df["label"]

        model = train(X_train, y_train)
        preds = model.predict(X_test)

        acc = (preds == y_test).mean()
        results.append(acc)

        start += test_size

    return results
