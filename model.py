from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


def train(X, y):
    """Plain RF — used inside walk-forward loops (fast, no cv fold issues)."""
    model = RandomForestClassifier(n_estimators=100, random_state=42,
                                   class_weight="balanced")
    model.fit(X, y)
    return model


def train_calibrated(X, y):
    """Calibrated RF — used for the final saved model used in live prediction."""
    base = RandomForestClassifier(n_estimators=100, random_state=42,
                                  class_weight="balanced")
    model = CalibratedClassifierCV(base, cv=2, method="isotonic")
    model.fit(X, y)
    return model
