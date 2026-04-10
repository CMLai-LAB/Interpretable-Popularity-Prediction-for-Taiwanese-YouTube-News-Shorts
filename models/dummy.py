from sklearn.dummy import DummyClassifier


def train_dummy(X_train, y_train):
    model = DummyClassifier(strategy="prior")
    model.fit(X_train, y_train)
    return model
