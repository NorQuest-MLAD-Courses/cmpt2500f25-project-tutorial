"""
Model evaluation module for telecom churn prediction.
Contains functions to evaluate model performance.
"""

import logging
from typing import Any, Dict
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

logging.basicConfig(level=logging.INFO)
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
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary')
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
    report = classification_report(y_true, y_pred)
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
        ROC-AUC score
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
        results['roc_auc'] = calculate_roc_auc(y_test, y_proba)
    
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
        DataFrame with comparison results
    """
    logger.info("Comparing models...")
    
    results = []
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='binary'),
            'Recall': recall_score(y_test, y_pred, average='binary'),
            'F1-Score': f1_score(y_test, y_pred, average='binary')
        }
        
        # Add ROC-AUC if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            metrics['ROC-AUC'] = calculate_roc_auc(y_test, y_proba)
        
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
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
