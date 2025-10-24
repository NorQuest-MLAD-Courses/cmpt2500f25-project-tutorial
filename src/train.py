import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(X, y, n_estimators):
    clf = AdaBoostClassifier(n_estimators=n_estimators)
    clf.fit(X, y)
    return clf

def load_config(cfg_filepath="configs/train_config.yaml"):
    cfg = yaml.safe_load(cfg_filepath)
    return cfg

def main():
    mlflow.set_experiment("cmpt2500-project")
    mlflow.autolog()

    cfg = load_config()

    n_estimators = int(cfg['n_estimators'])

    parser = argparse.ArgumentParser(description="Train churn prediction model.")
    parser.add_argument("--input_data_path", type=str, default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--output_model_path", type=str, default="models/result.pkl")

    args = parser.parse_args()

    df = pd.read_csv(args.input_data_path)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"]).select_dtypes(include=["number"]).values
    y = df['Churn'].values
    with mlflow.start_run(run_name="ada_n25") as run:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = train_model(X_train, y_train, n_estimators)
        acc = accuracy_score(y_test, clf.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, artifact_path="model")
        print("Run ID:", run.info.run_id)


if __name__ == "__main__":
    main()
