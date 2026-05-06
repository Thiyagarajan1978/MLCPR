from model import train


def run_walk_forward(df, train_size=200, test_size=50):
    results = []
    start = 0

    while start + train_size + test_size < len(df):
        train_df = df.iloc[start:start + train_size]
        test_df = df.iloc[start + train_size:start + train_size + test_size]

        X_train = train_df.drop(["label"], axis=1)
        y_train = train_df["label"]

        X_test = test_df.drop(["label"], axis=1)
        y_test = test_df["label"]

        model = train(X_train, y_train)
        preds = model.predict(X_test)

        acc = (preds == y_test).mean()
        results.append(acc)

        start += test_size

    return results
