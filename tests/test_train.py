"""
Tests for src/train.py module.
Tests model training, hyperparameter tuning, model saving, and MLflow integration.
"""

# Standard library imports
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Tuple # <-- This import is now correct

# Third-party imports
import joblib
import numpy as np
import pytest
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Local imports
from src.train import (
    evaluate_model,
    # save_all_models, <-- REMOVED
    save_model,
    train_adaboost,
    # train_all_models, <-- REMOVED
    train_decision_tree,
    train_gradient_boosting,
    train_logistic_regression,
    train_random_forest,
    train_voting_classifier,
    load_data_from_file # <-- This function is nested in main, so we can't test it directly
                        # We will remove tests that tried to use it.
)
from src.utils.config import RANDOM_STATE

# --- Model Training Tests ---

class TestTrainLogisticRegression:
    """Tests for train_logistic_regression function."""
    
    def test_train_logistic_regression_default(self, processed_data):
        """Test training with default parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_logistic_regression(X_train, y_train, tune_hyperparameters=False)
        
        assert isinstance(model, LogisticRegression)
        assert hasattr(model, 'coef_')
        assert model.random_state == RANDOM_STATE
    
    def test_train_logistic_regression_custom_params(self, processed_data):
        """Test training with custom parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_logistic_regression(X_train, y_train, tune_hyperparameters=False, C=0.5, solver='saga')
        
        assert isinstance(model, LogisticRegression)
        assert model.C == 0.5
        assert model.solver == 'saga'

    @pytest.mark.slow
    def test_train_logistic_regression_with_tuning(self, processed_data):
        """Test training with hyperparameter tuning."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        # Use patch to speed up GridSearchCV
        with patch('sklearn.model_selection.GridSearchCV.fit', MagicMock()) as mock_fit:
            model = train_logistic_regression(X_train, y_train, tune_hyperparameters=True)
            mock_fit.assert_called_once()
            
        assert isinstance(model, LogisticRegression)


class TestTrainRandomForest:
    """Tests for train_random_forest function."""
    
    def test_train_random_forest_default(self, processed_data):
        """Test training with default parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        assert isinstance(model, RandomForestClassifier)
        assert model.random_state == RANDOM_STATE
        
    def test_train_random_forest_custom_params(self, processed_data):
        """Test training with custom parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False, n_estimators=50, max_depth=10)
        
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50
        assert model.max_depth == 10

    @pytest.mark.slow
    def test_train_random_forest_with_tuning(self, processed_data):
        """Test training with hyperparameter tuning."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        with patch('sklearn.model_selection.GridSearchCV.fit', MagicMock()) as mock_fit:
            model = train_random_forest(X_train, y_train, tune_hyperparameters=True)
            mock_fit.assert_called_once()
            
        assert isinstance(model, RandomForestClassifier)


class TestTrainDecisionTree:
    """Tests for train_decision_tree function."""
    
    def test_train_decision_tree_default(self, processed_data):
        """Test training with default parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_decision_tree(X_train, y_train, tune_hyperparameters=False)
        
        assert isinstance(model, DecisionTreeClassifier)
        assert model.random_state == RANDOM_STATE

    @pytest.mark.slow
    def test_train_decision_tree_with_tuning(self, processed_data):
        """Test training with hyperparameter tuning."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        with patch('sklearn.model_selection.GridSearchCV.fit', MagicMock()) as mock_fit:
            model = train_decision_tree(X_train, y_train, tune_hyperparameters=True)
            mock_fit.assert_called_once()
            
        assert isinstance(model, DecisionTreeClassifier)


class TestTrainAdaBoost:
    """Tests for train_adaboost function."""
    
    def test_train_adaboost_default(self, processed_data):
        """Test training with default parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_adaboost(X_train, y_train, tune_hyperparameters=False)
        
        assert isinstance(model, AdaBoostClassifier)
        assert model.random_state == RANDOM_STATE

    def test_train_adaboost_custom_params(self, processed_data):
        """Test training with custom parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_adaboost(X_train, y_train, tune_hyperparameters=False, n_estimators=100, learning_rate=0.5)
        
        assert isinstance(model, AdaBoostClassifier)
        assert model.n_estimators == 100
        assert model.learning_rate == 0.5


class TestTrainGradientBoosting:
    """Tests for train_gradient_boosting function."""
    
    def test_train_gradient_boosting_default(self, processed_data):
        """Test training with default parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_gradient_boosting(X_train, y_train, tune_hyperparameters=False)
        
        assert isinstance(model, GradientBoostingClassifier)
        assert model.random_state == RANDOM_STATE

    def test_train_gradient_boosting_custom_params(self, processed_data):
        """Test training with custom parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_gradient_boosting(X_train, y_train, tune_hyperparameters=False, n_estimators=150, max_depth=5)
        
        assert isinstance(model, GradientBoostingClassifier)
        assert model.n_estimators == 150
        assert model.max_depth == 5


class TestTrainVotingClassifier:
    """Tests for train_voting_classifier function."""
    
    def test_train_voting_classifier_hard_voting(self, processed_data):
        """Test training with hard voting."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_voting_classifier(X_train, y_train, tune_hyperparameters=False, voting='hard')
        
        assert isinstance(model, VotingClassifier)
        assert model.voting == 'hard'


# --- Evaluation and Saving Tests ---

class TestEvaluateModel:
    """Tests for evaluate_model function."""
    
    def test_evaluate_model_basic(self, trained_model, processed_data):
        """Test basic model evaluation."""
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        with patch('src.train.mlflow.log_artifact', MagicMock()): # Mock mlflow
            metrics = evaluate_model(trained_model, X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0

    def test_evaluate_model_with_roc_auc(self, trained_model, processed_data):
        """Test ROC AUC calculation."""
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        with patch('src.train.mlflow.log_artifact', MagicMock()): # Mock mlflow
            metrics = evaluate_model(trained_model, X_test, y_test)
        
        assert 'roc_auc' in metrics
        assert 0.0 <= metrics['roc_auc'] <= 1.0

    def test_evaluate_model_confusion_matrix_values(self, processed_data):
        """Test confusion matrix logic with known values."""
        X_test = processed_data['X_test']
        # Create perfect predictions
        y_test = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        
        mock_model = MagicMock()
        mock_model.predict.return_value = y_pred
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        
        with patch('src.train.confusion_matrix') as mock_cm:
            with patch('src.train.mlflow.log_artifact', MagicMock()): # Mock mlflow
                evaluate_model(mock_model, X_test[:5], y_test)
                # Check if confusion_matrix was called correctly
                mock_cm.assert_called_once()
                np.testing.assert_array_equal(mock_cm.call_args[0][0], y_test)
                np.testing.assert_array_equal(mock_cm.call_args[0][1], y_pred)


class TestSaveModel:
    """Tests for save_model function."""
    
    def test_save_model_creates_file(self, trained_model, temp_output_dir):
        """Test that save_model creates a file."""
        model_path = save_model(trained_model, "test_model", str(temp_output_dir))
        
        assert os.path.exists(model_path)
        assert model_path.startswith(str(temp_output_dir))
        assert "test_model" in model_path
        assert ".pkl" in model_path
        
    def test_save_model_timestamped(self, trained_model, temp_output_dir):
        """Test that filenames are unique (timestamped)."""
        path1 = save_model(trained_model, "timed_model", str(temp_output_dir))
        path2 = save_model(trained_model, "timed_model", str(temp_output_dir))
        
        assert path1 != path2
        assert Path(path1).exists()
        assert Path(path2).exists()


# --- Test All Models and MLflow (Tests for deleted functions removed) ---

@pytest.mark.parametrize("model_type", [
    'logistic_regression',
    'random_forest',
    'decision_tree',
    'adaboost',
    'gradient_boosting',
    'voting_classifier'
])
class TestParametrizedModelTraining:
    """Parametrized tests for all model types."""
    
    def test_train_model_type(self, model_type, processed_data):
        """Test training each model type."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        train_functions = {
            'logistic_regression': train_logistic_regression,
            'random_forest': train_random_forest,
            'decision_tree': train_decision_tree,
            'adaboost': train_adaboost,
            'gradient_boosting': train_gradient_boosting,
            'voting_classifier': train_voting_classifier
        }
        
        train_func = train_functions[model_type]
        
        model = train_func(X_train, y_train, tune_hyperparameters=False)
        
        assert hasattr(model, 'predict')
        
        # Test prediction
        X_test = processed_data['X_test']
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
