"""
Example usage script for the Telecom Churn Prediction project.
This demonstrates how to use the modules for a complete ML workflow.
"""

# Example 1: Complete Pipeline
print("="*60)
print("EXAMPLE 1: Complete ML Pipeline")
print("="*60)

import os
import sys

# Ensure the src directory is in the Python path
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.preprocess import preprocess_pipeline
from src.train import train_voting_classifier, save_model
from src.evaluate import evaluate_model, print_evaluation_summary
from src.predict import predict, predict_proba

# Step 1: Preprocess data
print("\n1. Preprocessing data...")
X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
    'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv',
    scale=True
)

# Step 2: Train model
print("\n2. Training Voting Classifier...")
model = train_voting_classifier(X_train, y_train)

# Step 3: Evaluate model
print("\n3. Evaluating model...")
results = evaluate_model(model, X_test, y_test)
print_evaluation_summary(results, "Voting Classifier")

# Step 4: Save model
print("\n4. Saving model...")
model_path = save_model(model, "voting_classifier")

# Step 5: Make predictions
print("\n5. Making predictions...")
predictions = predict(model, X_test)
probabilities = predict_proba(model, X_test)

print(f"Made predictions for {len(predictions)} samples")
print(f"First 5 predictions: {predictions[:5]}")

print("\n" + "="*60)
print("Pipeline completed successfully!")
print("="*60)


# Example 2: Train and Compare Multiple Models
print("\n\n" + "="*60)
print("EXAMPLE 2: Train and Compare Multiple Models")
print("="*60)

from src.train import train_all_models, save_all_models
from src.evaluate import compare_models

# Train all models
print("\nTraining all models...")
models = train_all_models(X_train, y_train)

# Compare models
print("\nComparing models...")
comparison_df = compare_models(models, X_test, y_test)
print("\nModel Comparison:")
print(comparison_df.to_string(index=False))

# Save all models
print("\nSaving all models...")
saved_paths = save_all_models(models)
print(f"Saved {len(saved_paths)} models")


# Example 3: Feature Engineering
print("\n\n" + "="*60)
print("EXAMPLE 3: Feature Engineering")
print("="*60)

from src.feature_engineering import get_feature_importance, calculate_correlation
import pandas as pd

# Get feature importance from Random Forest
print("\nCalculating feature importance...")
rf_model = models['random_forest']
# Need feature names for X_train
feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
importance_df = get_feature_importance(rf_model, feature_names, top_n=10)

if importance_df is not None:
    print("\nTop 10 Important Features:")
    print(importance_df.to_string(index=False))


# Example 4: Single Customer Prediction
print("\n\n" + "="*60)
print("EXAMPLE 4: Predict for Single Customer")
print("="*60)

from src.predict import predict_single, load_model

# Load saved model
print("\nLoading saved model...")
loaded_model = load_model(model_path)

# Example customer features (these would need to be preprocessed)
# This is just to show the API - in practice you'd preprocess first
example_customer = {
    'feature_0': 0.5,
    'feature_1': 1.2,
    # ... more features ...
}

print("\nNote: Single prediction requires properly preprocessed features")
print("See the README for complete preprocessing pipeline")


# Example 5: Batch Predictions
print("\n\n" + "="*60)
print("EXAMPLE 5: Batch Predictions")
print("="*60)

from src.predict import batch_predict

print("\nMaking batch predictions...")
batch_predictions = batch_predict(loaded_model, X_test, batch_size=500)
print(f"Generated {len(batch_predictions)} predictions in batches")


print("\n\n" + "="*60)
print("All examples completed successfully!")
print("="*60)
print()
print("Check models/ for saved models")
