"""
Tests for src/evaluate.py module.
Tests metric calculation, evaluation functions, and model comparison.
Note: Uses 'y' for true labels and 'yhat' for predictions per user preference.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.evaluate import (
    calculate_accuracy,
    calculate_metrics,
    calculate_roc_auc,
    compare_models,
    evaluate_model,
    generate_classification_report,
    get_confusion_matrix,
    print_evaluation_summary,
    save_evaluation_results
)


class TestCalculateAccuracy:
    """Tests for calculate_accuracy function."""
    
    def test_calculate_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        # --- THE FIX IS HERE ---
        y = np.array([0, 1, 0, 1, 0]) # Use numeric
        yhat = np.array([0, 1, 0, 1, 0])
        
        accuracy = calculate_accuracy(y, yhat)
        
        assert accuracy == 1.0
    
    def test_calculate_accuracy_half(self):
        """Test accuracy with 50% correct predictions."""
        # --- THE FIX IS HERE ---
        y = np.array([0, 1, 0, 1])
        yhat = np.array([0, 1, 1, 0])
        
        accuracy = calculate_accuracy(y, yhat)
        
        assert accuracy == 0.5
    
    def test_calculate_accuracy_zero(self):
        """Test accuracy with all wrong predictions."""
        # --- THE FIX IS HERE ---
        y = np.array([0, 1, 0, 1])
        yhat = np.array([1, 0, 1, 0])
        
        accuracy = calculate_accuracy(y, yhat)
        
        assert accuracy == 0.0


@pytest.fixture
def sample_predictions():
    """Fixture for sample predictions."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([
        [0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.6, 0.4], [0.3, 0.7],
        [0.8, 0.2], [0.1, 0.9], [0.9, 0.1], [0.6, 0.4], [0.7, 0.3]
    ])
    return {'y': y_true, 'yhat': y_pred, 'yprob': y_prob}


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""
    
    def test_calculate_metrics_values(self, sample_predictions):
        """Test calculation of various metrics."""
        y = sample_predictions['y']
        yhat = sample_predictions['yhat']
        
        metrics = calculate_metrics(y, yhat)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Based on y and yhat:
        # TP=3, FP=2, TN=3, FN=2
        # Acc = (3+3)/10 = 0.6
        # Prec = 3/(3+2) = 0.6
        # Rec = 3/(3+2) = 0.6
        # F1 = 2 * (0.6*0.6) / (0.6+0.6) = 0.6
        assert metrics['accuracy'] == pytest.approx(0.6)
        assert metrics['precision'] == pytest.approx(0.6)
        assert metrics['recall'] == pytest.approx(0.6)
        assert metrics['f1_score'] == pytest.approx(0.6)

    def test_calculate_metrics_zero_division(self):
        """Test zero_division behavior."""
        y = np.array([0, 0, 0])
        yhat = np.array([0, 0, 0])
        
        metrics = calculate_metrics(y, yhat)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1_score'] == 0.0


class TestROCAUC:
    """Tests for calculate_roc_auc function."""
    
    def test_calculate_roc_auc_binary(self, sample_predictions):
        """Test ROC AUC for binary classification."""
        y = sample_predictions['y']
        yprob = sample_predictions['yprob']
        
        roc_auc = calculate_roc_auc(y, yprob)
        
        assert isinstance(roc_auc, float)
        assert 0.5 <= roc_auc <= 1.0

    def test_calculate_roc_auc_single_class(self):
        """Test ROC AUC when only one class is present."""
        y = np.array([0, 0, 0, 0])
        yprob = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.9, 0.1]])
        
        roc_auc = calculate_roc_auc(y, yprob)
        
        # ROC AUC is not defined for single class, should return 0.0
        assert roc_auc == 0.0


class TestReportAndMatrix:
    """Tests for confusion matrix and classification report."""
    
    def test_get_confusion_matrix(self, sample_predictions):
        """Test confusion matrix generation."""
        y = sample_predictions['y']
        yhat = sample_predictions['yhat']
        
        cm = get_confusion_matrix(y, yhat)
        
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (2, 2)
        # TN=3, FP=2, FN=2, TP=3
        expected_cm = np.array([[3, 2], [2, 3]])
        np.testing.assert_array_equal(cm, expected_cm)
    
    def test_generate_classification_report(self, sample_predictions):
        """Test classification report generation."""
        y = sample_predictions['y']
        yhat = sample_predictions['yhat']
        
        report_str = generate_classification_report(y, yhat)
        
        assert isinstance(report_str, str)
        assert 'precision' in report_str
        assert 'recall' in report_str
        assert 'f1-score' in report_str
        assert '0.60' in report_str # Check for metrics


class TestEvaluateModel:
    """Tests for the main evaluate_model function."""
    
    def test_evaluate_model_workflow(self, trained_model, processed_data):
        """Test the main evaluation function."""
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        results = evaluate_model(trained_model, X_test, y_test)
        
        assert isinstance(results, dict)
        assert set(results.keys()) == set([
            'accuracy', 'precision', 'recall', 'f1_score',
            'roc_auc', 'confusion_matrix', 'classification_report'
        ])
        assert all(0.0 <= results[key] <= 1.0 for key in [
            'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'
        ])
    
    def test_evaluate_save_load_workflow(self, trained_model, processed_data, temp_output_dir):
        """Test evaluation, saving, and loading results."""
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # Evaluate
        results = evaluate_model(trained_model, X_test, y_test)
        
        # Save
        output_path = temp_output_dir / 'eval_results.json'
        save_evaluation_results(results, str(output_path), model_name='TestModel')
        
        # Load and verify
        with open(output_path, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['accuracy'] == results['accuracy']
        assert loaded_results['model_name'] == 'TestModel'


@pytest.mark.parametrize("metric_name", ['accuracy', 'precision', 'recall', 'f1_score'])
class TestParametrizedMetrics:
    """Parametrized tests for different metrics."""
    
    def test_metric_range(self, metric_name, sample_predictions):
        """Test that all metrics are in valid range [0, 1]."""
        metrics = calculate_metrics(
            sample_predictions['y'],
            sample_predictions['yhat']
        )
        assert 0.0 <= metrics[metric_name] <= 1.0
