from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


def train(X, y):
    base = RandomForestClassifier(n_estimators=100, random_state=42)
    # Isotonic calibration corrects RF's tendency to output overconfident probabilities
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    model.fit(X, y)
    return model
