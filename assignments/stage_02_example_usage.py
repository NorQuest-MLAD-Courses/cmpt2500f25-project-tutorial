"""
Example usage script demonstrating the workflow after Lab 02.
Includes DVC data pulling and MLflow experiment tracking.
"""

import logging
import os
import subprocess
import sys
import mlflow
import mlflow.sklearn
import numpy as np

# Ensure the src directory is in the Python path
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.evaluate import evaluate_model, print_evaluation_summary
from src.predict import ModelPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_PATH = "data/processed/preprocessed_data.npy"
MODEL_TYPE = "random_forest"
TUNE_HYPERPARAMETERS = False # Set to True to run tuning (slower)
EXPERIMENT_NAME = "telecom-churn-prediction" # Should match train.py

def run_command(command):
    """Executes a shell command and logs output."""
    logger.info(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
        logger.info(f"Command output:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Command stderr:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(command)}")
        logger.error(f"Error output:\n{e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"Command not found: {command[0]}. Is it installed and in PATH?")
        return False

def main():
    """Main workflow demonstrating Lab 02 features."""

    # Step 1: Ensure Data is Present using DVC
    logger.info("="*60)
    logger.info("Step 1: Checking and Pulling Data with DVC...")
    logger.info("="*60)
    if not run_command(["dvc", "pull", DATA_PATH]):
        logger.error("Failed to pull data using DVC. Ensure DVC is set up correctly.")
        # Attempt to continue assuming data might already exist locally
        # In a real pipeline, you might exit here.
        # sys.exit(1)
    else:
        logger.info("DVC data pull completed successfully.")


    # Step 2: Run Training with MLflow Tracking
    logger.info("\n" + "="*60)
    logger.info("Step 2: Running Training Script (with MLflow)...")
    logger.info("="*60)
    train_command = [
        "python", "-m", "src.train",
        "--data", DATA_PATH,
        "--model", MODEL_TYPE,
        "--experiment-name", EXPERIMENT_NAME
    ]
    if TUNE_HYPERPARAMETERS:
        train_command.append("--tune")

    if not run_command(train_command):
        logger.error("Training script failed. Check logs for details.")
        sys.exit(1)
    logger.info("Training script completed.")


    # Step 3: Load the Best Model from MLflow
    logger.info("\n" + "="*60)
    logger.info("Step 3: Loading Best Model from MLflow...")
    logger.info("="*60)
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        # Find the latest run for the specified model type (or best if sorted)
        runs = mlflow.search_runs(
            order_by=["start_time DESC"], # Get latest run
            # order_by=["metrics.accuracy DESC"], # Alternative: Get best accuracy run
            filter_string=f"tags.model_type = '{MODEL_TYPE}'",
            max_results=1
        )

        if runs.empty:
            logger.error(f"No MLflow runs found for model type '{MODEL_TYPE}' in experiment '{EXPERIMENT_NAME}'.")
            sys.exit(1)

        best_run_id = runs.iloc[0]['run_id']
        model_uri = f"runs:/{best_run_id}/model"
        logger.info(f"Found latest run ID: {best_run_id}")
        logger.info(f"Loading model from URI: {model_uri}")

        loaded_model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully from MLflow.")

    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        logger.error("Make sure MLflow tracking is working correctly and the experiment/run exists.")
        sys.exit(1)


    # Step 4: Evaluate the Loaded Model
    logger.info("\n" + "="*60)
    logger.info("Step 4: Evaluating the Loaded Model...")
    logger.info("="*60)
    try:
        # Load test data (needed for evaluation)
        data = np.load(os.path.join(PROJECT_ROOT, DATA_PATH), allow_pickle=True).item()
        X_test = data['X_test']
        y_test = data['y_test']

        logger.info(f"Test set loaded: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")

        results = evaluate_model(loaded_model, X_test, y_test)
        print_evaluation_summary(results, f"Loaded {MODEL_TYPE} (Run ID: {best_run_id[:8]}...)")

    except Exception as e:
        logger.error(f"Failed during model evaluation: {e}")
        sys.exit(1)


    # Step 5: Demonstrate Prediction with Loaded Model
    logger.info("\n" + "="*60)
    logger.info("Step 5: Making Predictions with Loaded Model...")
    logger.info("="*60)
    try:
        # Use a small subset for demonstration
        sample_X = X_test[:5]
        predictions = loaded_model.predict(sample_X)
        logger.info(f"Sample predictions on {len(sample_X)} instances: {predictions}")

        if hasattr(loaded_model, 'predict_proba'):
            probabilities = loaded_model.predict_proba(sample_X)
            logger.info(f"Sample probabilities:\n{probabilities}")

    except Exception as e:
        logger.error(f"Failed during prediction demonstration: {e}")
        sys.exit(1)

    logger.info("\n" + "="*60)
    logger.info("✅ Stage 02 Example Workflow Completed Successfully!")
    logger.info(f"Check the MLflow UI (run: mlflow ui --host 0.0.0.0 --port 5000) for experiment details.")
    logger.info("="*60)

if __name__ == "__main__":
    main()
