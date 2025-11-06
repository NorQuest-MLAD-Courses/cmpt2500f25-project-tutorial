"""
Tests for src/train.py module.
Tests model training, hyperparameter tuning, model saving, and MLflow integration.
"""

import os
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

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

from src.train import (
    evaluate_model,
    save_all_models,
    save_model,
    train_adaboost,
    train_all_models,
    train_decision_tree,
    train_gradient_boosting,
    train_logistic_regression,
    train_random_forest,
    train_voting_classifier
)


# Load test config for fast hyperparameter tuning in tests
def load_test_config():
    """Load minimal test configuration for fast testing."""
    config_path = Path(__file__).parent.parent / "configs" / "test_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TestTrainLogisticRegression:
    """Tests for train_logistic_regression function."""
    
    def test_train_logistic_regression_default(self, processed_data):
        """Test training with default parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_logistic_regression(X_train, y_train, tune_hyperparameters=False)
        
        assert isinstance(model, LogisticRegression)
        assert hasattr(model, 'coef_')
        assert model.random_state == 42
    
    def test_train_logistic_regression_custom_params(self, processed_data):
        """Test training with custom parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_logistic_regression(
            X_train, y_train,
            tune_hyperparameters=False,
            C=0.5,
            penalty='l2'
        )
        
        assert isinstance(model, LogisticRegression)
        assert model.C == 0.5
    
    @pytest.mark.slow
    def test_train_logistic_regression_with_tuning(self, processed_data):
        """Test training with hyperparameter tuning using minimal test config."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']

        # Load minimal test config for fast testing
        test_config = load_test_config()

        model = train_logistic_regression(
            X_train, y_train,
            tune_hyperparameters=True,
            config=test_config  # Use test config with minimal grid
        )

        assert isinstance(model, LogisticRegression)
        assert hasattr(model, 'coef_')


class TestTrainRandomForest:
    """Tests for train_random_forest function."""
    
    def test_train_random_forest_default(self, processed_data):
        """Test training with default parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, 'estimators_')
        assert model.random_state == 42
    
    def test_train_random_forest_custom_params(self, processed_data):
        """Test training with custom parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_random_forest(
            X_train, y_train,
            tune_hyperparameters=False,
            n_estimators=50,
            max_depth=5
        )
        
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50
        assert model.max_depth == 5
    
    @pytest.mark.slow
    def test_train_random_forest_with_tuning(self, processed_data):
        """Test training with hyperparameter tuning using minimal test config."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']

        # Load minimal test config for fast testing
        test_config = load_test_config()

        model = train_random_forest(
            X_train, y_train,
            tune_hyperparameters=True,
            config=test_config  # Use test config with minimal grid
        )

        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, 'estimators_')
    
    def test_train_random_forest_prediction(self, processed_data):
        """Test that trained model can make predictions."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred in ['Yes', 'No'] for pred in predictions)


class TestTrainDecisionTree:
    """Tests for train_decision_tree function."""
    
    def test_train_decision_tree_default(self, processed_data):
        """Test training with default parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_decision_tree(X_train, y_train, tune_hyperparameters=False)
        
        assert isinstance(model, DecisionTreeClassifier)
        assert hasattr(model, 'tree_')
        assert model.random_state == 42
    
    def test_train_decision_tree_custom_params(self, processed_data):
        """Test training with custom parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_decision_tree(
            X_train, y_train,
            tune_hyperparameters=False,
            max_depth=3,
            min_samples_split=5
        )
        
        assert isinstance(model, DecisionTreeClassifier)
        assert model.max_depth == 3
        assert model.min_samples_split == 5


class TestTrainAdaBoost:
    """Tests for train_adaboost function."""
    
    def test_train_adaboost_default(self, processed_data):
        """Test training with default parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_adaboost(X_train, y_train, tune_hyperparameters=False)
        
        assert isinstance(model, AdaBoostClassifier)
        assert hasattr(model, 'estimators_')
        assert model.random_state == 42
    
    def test_train_adaboost_custom_params(self, processed_data):
        """Test training with custom parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_adaboost(
            X_train, y_train,
            tune_hyperparameters=False,
            n_estimators=25,
            learning_rate=0.5
        )
        
        assert isinstance(model, AdaBoostClassifier)
        assert model.n_estimators == 25
        assert model.learning_rate == 0.5


class TestTrainGradientBoosting:
    """Tests for train_gradient_boosting function."""
    
    def test_train_gradient_boosting_default(self, processed_data):
        """Test training with default parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_gradient_boosting(X_train, y_train, tune_hyperparameters=False)
        
        assert isinstance(model, GradientBoostingClassifier)
        assert hasattr(model, 'estimators_')
        assert model.random_state == 42
    
    def test_train_gradient_boosting_custom_params(self, processed_data):
        """Test training with custom parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_gradient_boosting(
            X_train, y_train,
            tune_hyperparameters=False,
            n_estimators=50,
            learning_rate=0.05
        )
        
        assert isinstance(model, GradientBoostingClassifier)
        assert model.n_estimators == 50
        assert model.learning_rate == 0.05


class TestTrainVotingClassifier:
    """Tests for train_voting_classifier function."""
    
    def test_train_voting_classifier_default(self, processed_data):
        """Test training ensemble with default parameters."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_voting_classifier(X_train, y_train)
        
        assert isinstance(model, VotingClassifier)
        assert len(model.estimators_) == 3
        assert model.voting == 'soft'
    
    def test_train_voting_classifier_hard_voting(self, processed_data):
        """Test training ensemble with hard voting."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        model = train_voting_classifier(X_train, y_train, voting='hard')
        
        assert isinstance(model, VotingClassifier)
        assert model.voting == 'hard'


class TestEvaluateModel:
    """Tests for evaluate_model function."""
    
    def test_evaluate_model_basic(self, trained_model, processed_data):
        """Test basic model evaluation."""
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        metrics = evaluate_model(trained_model, X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'true_positives' in metrics
        assert 'true_negatives' in metrics
        assert 'false_positives' in metrics
        assert 'false_negatives' in metrics
        
        # Check metric ranges
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['f1_score'] <= 1.0
    
    def test_evaluate_model_with_roc_auc(self, trained_model, processed_data):
        """Test evaluation includes ROC-AUC for models with predict_proba."""
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        metrics = evaluate_model(trained_model, X_test, y_test)
        
        if hasattr(trained_model, 'predict_proba'):
            assert 'roc_auc' in metrics
            assert 0.0 <= metrics['roc_auc'] <= 1.0
    
    def test_evaluate_model_confusion_matrix_values(self, trained_model, processed_data):
        """Test confusion matrix components are integers."""
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        metrics = evaluate_model(trained_model, X_test, y_test)
        
        assert isinstance(metrics['true_positives'], int)
        assert isinstance(metrics['true_negatives'], int)
        assert isinstance(metrics['false_positives'], int)
        assert isinstance(metrics['false_negatives'], int)


class TestSaveModel:
    """Tests for save_model function."""
    
    def test_save_model_success(self, trained_model, temp_models_dir):
        """Test successful model saving."""
        model_path = save_model(
            trained_model,
            model_name='test_model',
            output_dir=str(temp_models_dir)
        )
        
        assert Path(model_path).exists()
        assert Path(model_path).suffix == '.pkl'
        assert 'test_model' in Path(model_path).stem
    
    def test_save_model_creates_directory(self, trained_model, tmp_path):
        """Test that save_model creates output directory if needed."""
        output_dir = tmp_path / "new_models_dir"
        
        model_path = save_model(
            trained_model,
            model_name='test_model',
            output_dir=str(output_dir)
        )
        
        assert output_dir.exists()
        assert Path(model_path).exists()
    
    def test_save_model_timestamped(self, trained_model, temp_models_dir):
        """Test that saved models have timestamps."""
        path1 = save_model(trained_model, 'model1', str(temp_models_dir))
        path2 = save_model(trained_model, 'model1', str(temp_models_dir))
        
        # Different timestamps should result in different filenames
        assert path1 != path2
    
    def test_load_saved_model(self, trained_model, temp_models_dir):
        """Test that saved model can be loaded correctly."""
        model_path = save_model(
            trained_model,
            model_name='test_model',
            output_dir=str(temp_models_dir)
        )
        
        loaded_model = joblib.load(model_path)
        
        assert isinstance(loaded_model, type(trained_model))
        assert hasattr(loaded_model, 'predict')


class TestSaveAllModels:
    """Tests for save_all_models function."""
    
    def test_save_all_models_success(self, processed_data, temp_models_dir):
        """Test saving multiple models."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        models = {
            'model1': train_random_forest(X_train, y_train, tune_hyperparameters=False),
            'model2': train_logistic_regression(X_train, y_train, tune_hyperparameters=False)
        }
        
        saved_paths = save_all_models(models, output_dir=str(temp_models_dir))
        
        assert len(saved_paths) == 2
        assert 'model1' in saved_paths
        assert 'model2' in saved_paths
        assert all(Path(path).exists() for path in saved_paths.values())
    
    def test_save_all_models_empty_dict(self, temp_models_dir):
        """Test saving with empty models dictionary."""
        saved_paths = save_all_models({}, output_dir=str(temp_models_dir))
        
        assert len(saved_paths) == 0


class TestTrainAllModels:
    """Tests for train_all_models function."""
    
    @pytest.mark.slow
    def test_train_all_models_no_tuning(self, processed_data):
        """Test training all models without tuning."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        models = train_all_models(X_train, y_train, tune_hyperparameters=False)
        
        assert len(models) == 6
        assert 'logistic_regression' in models
        assert 'random_forest' in models
        assert 'decision_tree' in models
        assert 'adaboost' in models
        assert 'gradient_boosting' in models
        assert 'voting_classifier' in models
        
        # Verify all models are trained
        for model_name, model in models.items():
            assert hasattr(model, 'predict')


@pytest.mark.unit
class TestModelTrainingEdgeCases:
    """Tests for edge cases in model training."""
    
    def test_train_with_minimal_data(self):
        """Test training with minimal dataset."""
        X_train = np.random.randn(10, 5)
        y_train = np.array(['Yes', 'No'] * 5)
        
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        assert isinstance(model, RandomForestClassifier)
    
    def test_train_with_single_class(self):
        """Test training with single class (should work but produce warnings)."""
        X_train = np.random.randn(20, 5)
        y_train = np.array(['Yes'] * 20)
        
        # This should complete without error, though model may not be useful
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        assert isinstance(model, RandomForestClassifier)


@pytest.mark.integration
class TestMLflowIntegration:
    """Tests for MLflow integration in training."""
    
    def test_train_with_mlflow_mocked(self, processed_data, mock_mlflow_run):
        """Test training with MLflow logging (mocked)."""
        with patch('mlflow.log_param') as mock_log_param:
            with patch('mlflow.log_metric') as mock_log_metric:
                X_train = processed_data['X_train']
                y_train = processed_data['y_train']
                
                model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
                
                assert isinstance(model, RandomForestClassifier)
                # Note: In actual code, MLflow logging happens in main(), not in training functions
    
    def test_evaluate_model_metrics_logging(self, trained_model, processed_data):
        """Test that evaluate_model returns metrics suitable for MLflow."""
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        metrics = evaluate_model(trained_model, X_test, y_test)
        
        # All metrics should be serializable (numbers)
        for key, value in metrics.items():
            if key not in ['confusion_matrix']:  # Skip matrix
                assert isinstance(value, (int, float))


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
        
        if model_type == 'voting_classifier':
            model = train_func(X_train, y_train)
        else:
            model = train_func(X_train, y_train, tune_hyperparameters=False)
        
        assert hasattr(model, 'predict')
        
        # Test prediction
        X_test = processed_data['X_test']
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
