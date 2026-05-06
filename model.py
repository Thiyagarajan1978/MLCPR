from sklearn.ensemble import RandomForestClassifier


def train(X, y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model
