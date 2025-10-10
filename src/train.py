from sklearn.ensemble import AdaBoostClassifier

def train(X, y):
    clf = AdaBoostClassifier()
    clf.fit(X, y)
    return clf

