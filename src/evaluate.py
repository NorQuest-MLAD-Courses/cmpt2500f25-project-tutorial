"""
Model evaluation module for telecom churn prediction.
Contains functions to evaluate model performance with CLI support.
"""

# Standard library imports
import argparse
import logging
from typing import Any, Dict

# Third-party imports
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"Accuracy: {accuracy:.4f}")
    return accuracy


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    # Determine the positive label automatically
    # If labels are numeric (0, 1), use 1 as pos_label
    # If labels are strings, use the second unique value alphabetically (usually 'Yes')
    unique_labels = np.unique(y_true)
    if len(unique_labels) > 0:
        if np.issubdtype(unique_labels.dtype, np.number):
            pos_label = 1
        else:
            # For string labels, sort and use the last one ('Yes' comes after 'No')
            pos_label = sorted(unique_labels)[-1]
    else:
        pos_label = 1  # Default fallback

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0)
    }

    logger.info(f"Metrics calculated: {metrics}")
    return metrics


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Classification report as string
    """
    report = classification_report(y_true, y_pred, zero_division=0)
    logger.info("Classification report generated")
    return report


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Generate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix as numpy array
    """
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")
    return cm


def calculate_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate ROC-AUC score.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        
    Returns:
        ROC-AUC score or None if cannot be calculated
    """
    try:
        # If y_proba is 2D (probability for each class), use second column
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        
        roc_auc = roc_auc_score(y_true, y_proba)
        logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
        return roc_auc
    except Exception as e:
        logger.warning(f"Could not calculate ROC-AUC: {e}")
        return None


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    results = calculate_metrics(y_test, y_pred)
    results['confusion_matrix'] = get_confusion_matrix(y_test, y_pred)
    results['classification_report'] = generate_classification_report(y_test, y_pred)
    
    # Calculate ROC-AUC if model supports probability predictions
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        roc_auc = calculate_roc_auc(y_test, y_proba)
        if roc_auc is not None:
            results['roc_auc'] = roc_auc
    
    logger.info("Model evaluation completed")
    return results


def compare_models(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Compare performance of multiple models.

    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels

    Returns:
        DataFrame with comparison results sorted by accuracy
    """
    logger.info("Comparing models...")

    results = []

    # Determine the positive label automatically
    unique_labels = np.unique(y_test)
    if len(unique_labels) > 0:
        if np.issubdtype(unique_labels.dtype, np.number):
            pos_label = 1
        else:
            # For string labels, sort and use the last one ('Yes' comes after 'No')
            pos_label = sorted(unique_labels)[-1]
    else:
        pos_label = 1  # Default fallback

    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        y_pred = model.predict(X_test)

        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='binary', pos_label=pos_label, zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='binary', pos_label=pos_label, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='binary', pos_label=pos_label, zero_division=0)
        }

        # Add ROC-AUC if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            roc_auc = calculate_roc_auc(y_test, y_proba)
            if roc_auc is not None:
                metrics['ROC-AUC'] = roc_auc

        results.append(metrics)

    comparison_df = pd.DataFrame(results)

    # Only sort if there are results
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

    logger.info("Model comparison completed")
    return comparison_df


def print_evaluation_summary(results: Dict[str, Any], model_name: str = "Model"):
    """
    Print a formatted evaluation summary.
    
    Args:
        results: Dictionary of evaluation results
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation Summary")
    print(f"{'='*60}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    
    if 'roc_auc' in results and results['roc_auc'] is not None:
        print(f"ROC-AUC:   {results['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    print(f"\nClassification Report:")
    print(results['classification_report'])
    print(f"{'='*60}\n")


def save_evaluation_results(
    results: Dict[str, Any],
    output_path: str,
    model_name: str = "model"
):
    """
    Save evaluation results to file.
    
    Args:
        results: Dictionary of evaluation results
        output_path: Path to save results
        model_name: Name of the model
    """
    import json
    from datetime import datetime
    
    # Prepare results for JSON serialization
    serializable_results = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1_score']),
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': results['classification_report']
    }
    
    if 'roc_auc' in results and results['roc_auc'] is not None:
        serializable_results['roc_auc'] = float(results['roc_auc'])
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {output_path}")


def main():
    """
    Main function for CLI evaluation.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate trained telecom churn prediction model'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to test data (numpy file with X_test and y_test)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save evaluation results (JSON format)'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='model',
        help='Name of the model for reporting'
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from: {args.model}")
    model = joblib.load(args.model)
    
    # Load test data
    logger.info(f"Loading test data from: {args.data}")
    data = np.load(args.data, allow_pickle=True).item()
    X_test = data['X_test']
    y_test = data['y_test']
    
    logger.info(f"Test set shape: {X_test.shape}")
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    # Print summary
    print_evaluation_summary(results, args.model_name)
    
    # Save results if output path provided
    if args.output:
        save_evaluation_results(results, args.output, args.model_name)
        print(f"\nResults saved to: {args.output}")
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
