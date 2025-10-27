"""
Tests for src/evaluate.py module.
Tests metric calculation, evaluation functions, and model comparison.
Note: Uses 'y' for true labels and 'yhat' for predictions per user preference.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix

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
        y = np.array(['Yes', 'No', 'Yes', 'No', 'Yes'])
        yhat = np.array(['Yes', 'No', 'Yes', 'No', 'Yes'])
        
        accuracy = calculate_accuracy(y, yhat)
        
        assert accuracy == 1.0
    
    def test_calculate_accuracy_half(self):
        """Test accuracy with 50% correct predictions."""
        y = np.array(['Yes', 'No', 'Yes', 'No'])
        yhat = np.array(['Yes', 'No', 'No', 'Yes'])
        
        accuracy = calculate_accuracy(y, yhat)
        
        assert accuracy == 0.5
    
    def test_calculate_accuracy_zero(self):
        """Test accuracy with all wrong predictions."""
        y = np.array(['Yes', 'No', 'Yes', 'No'])
        yhat = np.array(['No', 'Yes', 'No', 'Yes'])
        
        accuracy = calculate_accuracy(y, yhat)
        
        assert accuracy == 0.0


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""
    
    def test_calculate_metrics_all_present(self):
        """Test that all metrics are calculated."""
        y = np.array(['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'])
        yhat = np.array(['Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No'])
        
        metrics = calculate_metrics(y, yhat)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Check ranges
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0
    
    def test_calculate_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y = np.array(['Yes', 'No', 'Yes', 'No'])
        yhat = np.array(['Yes', 'No', 'Yes', 'No'])
        
        metrics = calculate_metrics(y, yhat)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_calculate_metrics_with_sample_predictions(self, sample_predictions):
        """Test metrics with fixture sample predictions."""
        metrics = calculate_metrics(
            sample_predictions['y'],
            sample_predictions['yhat']
        )
        
        assert all(key in metrics for key in ['accuracy', 'precision', 'recall', 'f1_score'])
        assert all(0.0 <= v <= 1.0 for v in metrics.values())


class TestGenerateClassificationReport:
    """Tests for generate_classification_report function."""
    
    def test_generate_report_format(self):
        """Test that classification report is generated correctly."""
        y = np.array(['Yes', 'No', 'Yes', 'No', 'Yes', 'No'])
        yhat = np.array(['Yes', 'No', 'Yes', 'Yes', 'Yes', 'No'])
        
        report = generate_classification_report(y, yhat)
        
        assert isinstance(report, str)
        assert 'precision' in report.lower()
        assert 'recall' in report.lower()
        assert 'f1-score' in report.lower()
    
    def test_generate_report_classes_included(self):
        """Test that report includes both classes."""
        y = np.array(['Yes', 'No', 'Yes', 'No'])
        yhat = np.array(['Yes', 'No', 'Yes', 'No'])
        
        report = generate_classification_report(y, yhat)
        
        assert 'Yes' in report or 'No' in report  # At least one class mentioned


class TestGetConfusionMatrix:
    """Tests for get_confusion_matrix function."""
    
    def test_confusion_matrix_shape(self):
        """Test confusion matrix has correct shape."""
        y = np.array(['Yes', 'No', 'Yes', 'No', 'Yes', 'No'])
        yhat = np.array(['Yes', 'No', 'No', 'Yes', 'Yes', 'No'])
        
        cm = get_confusion_matrix(y, yhat)
        
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (2, 2)  # Binary classification
    
    def test_confusion_matrix_values(self):
        """Test confusion matrix values are correct."""
        y = np.array(['Yes', 'No', 'Yes', 'No'])
        yhat = np.array(['Yes', 'No', 'Yes', 'No'])
        
        cm = get_confusion_matrix(y, yhat)
        
        # With perfect predictions, only diagonal should be non-zero
        assert cm[0, 0] + cm[1, 1] == len(y)
        assert cm[0, 1] + cm[1, 0] == 0
    
    def test_confusion_matrix_all_errors(self):
        """Test confusion matrix with all wrong predictions."""
        y = np.array(['Yes', 'No', 'Yes', 'No'])
        yhat = np.array(['No', 'Yes', 'No', 'Yes'])
        
        cm = get_confusion_matrix(y, yhat)
        
        # With all wrong predictions, diagonal should be zero
        assert cm[0, 0] + cm[1, 1] == 0
        assert cm[0, 1] + cm[1, 0] == len(y)


class TestCalculateRocAuc:
    """Tests for calculate_roc_auc function."""
    
    def test_roc_auc_perfect_predictions(self):
        """Test ROC-AUC with perfect probability predictions."""
        y = np.array(['Yes', 'No', 'Yes', 'No'])
        # Perfect probabilities
        y_proba = np.array([1.0, 0.0, 1.0, 0.0])
        
        roc_auc = calculate_roc_auc(y, y_proba)
        
        assert roc_auc == 1.0
    
    def test_roc_auc_random_predictions(self):
        """Test ROC-AUC with random-ish predictions."""
        np.random.seed(42)
        y = np.array(['Yes', 'No'] * 25)
        y_proba = np.random.rand(50)
        
        roc_auc = calculate_roc_auc(y, y_proba)
        
        assert 0.0 <= roc_auc <= 1.0
    
    def test_roc_auc_2d_probabilities(self):
        """Test ROC-AUC with 2D probability array."""
        y = np.array(['Yes', 'No', 'Yes', 'No'])
        # 2D array (probabilities for each class)
        y_proba = np.array([
            [0.2, 0.8],  # 80% Yes
            [0.9, 0.1],  # 10% Yes
            [0.3, 0.7],  # 70% Yes
            [0.8, 0.2]   # 20% Yes
        ])
        
        roc_auc = calculate_roc_auc(y, y_proba)
        
        assert 0.0 <= roc_auc <= 1.0
    
    def test_roc_auc_single_class(self):
        """Test ROC-AUC with single class (should return None or handle gracefully)."""
        y = np.array(['Yes', 'Yes', 'Yes', 'Yes'])
        y_proba = np.array([0.9, 0.8, 0.95, 0.85])
        
        # Should handle gracefully (return None or raise warning)
        roc_auc = calculate_roc_auc(y, y_proba)
        
        # Either None or some value, but shouldn't crash
        assert roc_auc is None or isinstance(roc_auc, float)


class TestEvaluateModel:
    """Tests for evaluate_model function."""
    
    def test_evaluate_model_complete(self, trained_model, processed_data):
        """Test complete model evaluation."""
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        results = evaluate_model(trained_model, X_test, y_test)
        
        # Check all metrics present
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 'confusion_matrix' in results
        assert 'classification_report' in results
        
        # Check confusion matrix
        assert isinstance(results['confusion_matrix'], np.ndarray)
        assert results['confusion_matrix'].shape == (2, 2)
        
        # Check classification report
        assert isinstance(results['classification_report'], str)
    
    def test_evaluate_model_with_roc_auc(self, trained_model, processed_data):
        """Test evaluation includes ROC-AUC when available."""
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        results = evaluate_model(trained_model, X_test, y_test)
        
        if hasattr(trained_model, 'predict_proba'):
            assert 'roc_auc' in results
            assert isinstance(results['roc_auc'], float)
            assert 0.0 <= results['roc_auc'] <= 1.0
    
    def test_evaluate_model_no_crash(self, trained_logistic_model, processed_data):
        """Test evaluation doesn't crash with different model types."""
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        results = evaluate_model(trained_logistic_model, X_test, y_test)
        
        assert 'accuracy' in results
        assert isinstance(results['accuracy'], float)


class TestCompareModels:
    """Tests for compare_models function."""
    
    def test_compare_models_basic(self, processed_data):
        """Test comparing multiple models."""
        from src.train import train_random_forest, train_logistic_regression
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        models = {
            'RandomForest': train_random_forest(X_train, y_train, tune_hyperparameters=False),
            'LogisticRegression': train_logistic_regression(X_train, y_train, tune_hyperparameters=False)
        }
        
        comparison_df = compare_models(models, X_test, y_test)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'Model' in comparison_df.columns
        assert 'Accuracy' in comparison_df.columns
        assert 'Precision' in comparison_df.columns
        assert 'Recall' in comparison_df.columns
        assert 'F1-Score' in comparison_df.columns
    
    def test_compare_models_sorted(self, processed_data):
        """Test that comparison results are sorted by accuracy."""
        from src.train import train_random_forest, train_decision_tree
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        models = {
            'Model1': train_random_forest(X_train, y_train, tune_hyperparameters=False),
            'Model2': train_decision_tree(X_train, y_train, tune_hyperparameters=False)
        }
        
        comparison_df = compare_models(models, X_test, y_test)
        
        # Check that accuracies are in descending order
        accuracies = comparison_df['Accuracy'].values
        assert all(accuracies[i] >= accuracies[i+1] for i in range(len(accuracies)-1))
    
    def test_compare_models_empty_dict(self, processed_data):
        """Test compare_models with empty dictionary."""
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        comparison_df = compare_models({}, X_test, y_test)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 0
    
    def test_compare_models_single_model(self, trained_model, processed_data):
        """Test comparison with single model."""
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        models = {'SingleModel': trained_model}
        
        comparison_df = compare_models(models, X_test, y_test)
        
        assert len(comparison_df) == 1
        assert comparison_df.iloc[0]['Model'] == 'SingleModel'


class TestPrintEvaluationSummary:
    """Tests for print_evaluation_summary function."""
    
    def test_print_evaluation_summary_output(self, capsys):
        """Test that evaluation summary prints correctly."""
        results = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'confusion_matrix': np.array([[10, 2], [3, 15]]),
            'classification_report': 'Sample report'
        }
        
        print_evaluation_summary(results, model_name='TestModel')
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert 'TestModel' in output
        assert '0.85' in output or '0.8500' in output  # Accuracy
        assert 'Confusion Matrix' in output
        assert 'Classification Report' in output
    
    def test_print_evaluation_summary_with_roc_auc(self, capsys):
        """Test summary includes ROC-AUC when present."""
        results = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'roc_auc': 0.90,
            'confusion_matrix': np.array([[10, 2], [3, 15]]),
            'classification_report': 'Sample report'
        }
        
        print_evaluation_summary(results)
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert 'ROC-AUC' in output
        assert '0.90' in output or '0.9000' in output


class TestSaveEvaluationResults:
    """Tests for save_evaluation_results function."""
    
    def test_save_evaluation_results_success(self, temp_output_dir):
        """Test saving evaluation results to JSON."""
        results = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'confusion_matrix': np.array([[10, 2], [3, 15]]),
            'classification_report': 'Sample report'
        }
        
        output_path = temp_output_dir / 'results.json'
        save_evaluation_results(results, str(output_path), model_name='TestModel')
        
        assert output_path.exists()
        
        # Load and verify
        with open(output_path, 'r') as f:
            saved_results = json.load(f)
        
        assert saved_results['model_name'] == 'TestModel'
        assert 'timestamp' in saved_results
        assert saved_results['accuracy'] == 0.85
        assert 'confusion_matrix' in saved_results
    
    def test_save_evaluation_results_with_roc_auc(self, temp_output_dir):
        """Test saving results with ROC-AUC."""
        results = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'roc_auc': 0.90,
            'confusion_matrix': np.array([[10, 2], [3, 15]]),
            'classification_report': 'Sample report'
        }
        
        output_path = temp_output_dir / 'results_with_auc.json'
        save_evaluation_results(results, str(output_path))
        
        with open(output_path, 'r') as f:
            saved_results = json.load(f)
        
        assert 'roc_auc' in saved_results
        assert saved_results['roc_auc'] == 0.90
    
    def test_load_saved_results(self, temp_output_dir):
        """Test that saved results can be loaded correctly."""
        results = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'confusion_matrix': np.array([[10, 2], [3, 15]]),
            'classification_report': 'Sample report'
        }
        
        output_path = temp_output_dir / 'loadable_results.json'
        save_evaluation_results(results, str(output_path))
        
        # Load back
        with open(output_path, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['accuracy'] == results['accuracy']
        assert loaded_results['precision'] == results['precision']
        assert isinstance(loaded_results['confusion_matrix'], list)


@pytest.mark.integration
class TestEvaluationWorkflow:
    """Integration tests for complete evaluation workflow."""
    
    def test_train_evaluate_workflow(self, processed_data):
        """Test complete workflow from training to evaluation."""
        from src.train import train_random_forest
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # Train
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        # Evaluate
        results = evaluate_model(model, X_test, y_test)
        
        # Verify
        assert all(key in results for key in [
            'accuracy', 'precision', 'recall', 'f1_score',
            'confusion_matrix', 'classification_report'
        ])
        assert all(0.0 <= results[key] <= 1.0 for key in [
            'accuracy', 'precision', 'recall', 'f1_score'
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
        
        assert metric_name in metrics
        assert 0.0 <= metrics[metric_name] <= 1.0
