import joblib


def save(model, path="model.pkl", feature_cols=None):
    joblib.dump({"model": model, "feature_cols": list(feature_cols) if feature_cols is not None else []}, path)


def save_long_short(long_model, short_model, feature_cols, path="model.pkl"):
    joblib.dump({
        "long_model":   long_model,
        "short_model":  short_model,
        "feature_cols": list(feature_cols),
    }, path)
