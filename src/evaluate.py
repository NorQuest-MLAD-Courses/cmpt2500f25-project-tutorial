# src/evaluate.py
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", type=str, required=True, help="MLflow run id to load model from")
    p.add_argument("--input_data_path", type=str, default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    args = p.parse_args()

    # Load model from MLflow artifacts
    model_uri = f"runs:/{args.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # Recreate the same basic preprocessing as train.py
    df = pd.read_csv(args.input_data_path)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"]).select_dtypes(include=["number"]).values
    y = df["Churn"].values

    # simple held-out split like train.py
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Loaded model from run {args.run_id}; accuracy={acc:.4f}")

if __name__ == "__main__":
    main()