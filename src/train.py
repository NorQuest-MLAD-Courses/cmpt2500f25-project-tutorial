import argparse
import joblib
import mlflow
import mlflow.sklearn
import os
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(X, y, n_estimators):
    clf = AdaBoostClassifier(n_estimators=n_estimators)
    clf.fit(X, y)
    return clf

if __name__ == "__main__":
    mlflow.set_experiment("cmpt2500-project")
    mlflow.autolog()

    parser = argparse.ArgumentParser(description="Train churn prediction model.")
    parser.add_argument("--input_data_path", type=str, default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--output_model_path", type=str, default="models/result.pkl")

    args = parser.parse_args()

    df = pd.read_csv(args.input_data_path)

    # pick target column automatically
    target_col = "Churn"
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # minimal features: use only numeric columns (keeps this step simple)
    X = df.drop(columns=[target_col]).select_dtypes(include=["number"]).values
    y = df[target_col].values
    with mlflow.start_run(run_name="ada_n25") as run:        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # train
        clf = train_model(X_train, y_train, 25)
        # eval
        acc = accuracy_score(y_test, clf.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        # save
        os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
        joblib.dump(clf, args.output_model_path)
        mlflow.sklearn.log_model(clf, artifact_path="model")
        print("Run ID:", run.info.run_id)