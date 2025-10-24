from sklearn.metrics import accuracy_score, classification_report

def evaluate(y, yhat):
    acc = accuracy_score(y, yhat)
    report = classification_report(y, yhat, zero_division=0)
    return acc, report