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


def train_long_short(X_long, y_long, X_short, y_short):
    """Two separate calibrated models — one per trade direction.

    Splitting by direction prevents the bullish-period training data from
    suppressing SHORT signal confidence across the board.
    """
    def _cal_rf():
        base = RandomForestClassifier(n_estimators=150, random_state=42,
                                      class_weight="balanced")
        return CalibratedClassifierCV(base, cv=2, method="isotonic")

    long_model  = _cal_rf()
    short_model = _cal_rf()
    long_model.fit(X_long,  y_long)
    short_model.fit(X_short, y_short)
    return long_model, short_model
