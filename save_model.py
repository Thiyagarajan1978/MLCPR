import joblib


def save(model, path="model.pkl", feature_cols=None):
    joblib.dump({"model": model, "feature_cols": list(feature_cols) if feature_cols is not None else []}, path)
