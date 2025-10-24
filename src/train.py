import argparse
import joblib
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

def train_model(X, y, n_estimators):
    clf = AdaBoostClassifier(n_estimators=n_estimators)
    clf.fit(X, y)
    return clf

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train churn prediction model.")
    parser.add_argument("--input_data_path", type=str, default="data/processed/processed_1.csv")
    parser.add_argument("--output_model_path", type=str, default="models/result.pkl")

    args = parser.parse_args()

    df = pd.read_csv(args.input_data_path)
    X = df.drop('label').values
    y = df['label'].values
    clf = train_model(X, y, 25)

    # Save the model
    joblib.dump(clf, args.output_model_path)