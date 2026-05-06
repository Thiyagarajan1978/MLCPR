import joblib


def save(model, path="model.pkl"):
    joblib.dump(model, path)
