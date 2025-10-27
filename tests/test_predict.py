"""
Tests for src/predict.py module.
Tests model loading, predictions, and ModelPredictor class.
"""

import numpy as np
import pandas as pd
import pytest

from src.predict import (
    ModelPredictor,
    batch_predict,
    load_model,
    load_preprocessing_pipeline,
    predict,
    predict_proba,
    predict_single
)


class TestLoadModel:
    """Tests for load_model function."""
    
    def test_load_model_success(self, saved_model_file):
        """Test successful model loading with joblib."""
        model = load_model(str(saved_model_file))
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_load_model_file_not_found(self):
        """Test error handling for missing model file."""
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model.pkl")
    
    def test_load_model_pickle_format(self, trained_model, tmp_path):
        """Test loading model saved with pickle."""
        import pickle
        
        model_path = tmp_path / "pickle_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(trained_model, f)
        
        loaded_model = load_model(str(model_path))
        
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')


class TestLoadPreprocessingPipeline:
    """Tests for load_preprocessing_pipeline function."""
    
    def test_load_pipeline_success(self, tmp_path):
        """Test successful pipeline loading."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        pipeline = Pipeline([('scaler', StandardScaler())])
        pipeline_path = tmp_path / "pipeline.pkl"
        joblib.dump(pipeline, pipeline_path)
        
        loaded_pipeline = load_preprocessing_pipeline(str(pipeline_path))
        
        assert isinstance(loaded_pipeline, Pipeline)
    
    def test_load_pipeline_file_not_found(self):
        """Test error handling for missing pipeline file."""
        with pytest.raises(Exception):
            load_preprocessing_pipeline("nonexistent_pipeline.pkl")


class TestPredict:
    """Tests for predict function."""
    
    def test_predict_with_array(self, trained_model, processed_data):
        """Test prediction with numpy array."""
        X_test = processed_data['X_test']
        
        predictions = predict(trained_model, X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert all(pred in ['Yes', 'No'] for pred in predictions)
    
    def test_predict_with_dataframe(self, trained_model, processed_data):
        """Test prediction with pandas DataFrame."""
        X_test = pd.DataFrame(processed_data['X_test'])
        
        predictions = predict(trained_model, X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
    
    def test_predict_empty_input(self, trained_model):
        """Test error handling for empty input."""
        X_empty = np.array([])
        
        with pytest.raises(ValueError, match="empty"):
            predict(trained_model, X_empty)
    
    def test_predict_empty_dataframe(self, trained_model):
        """Test error handling for empty DataFrame."""
        X_empty = pd.DataFrame()
        
        with pytest.raises(ValueError, match="empty"):
            predict(trained_model, X_empty)


class TestPredictProba:
    """Tests for predict_proba function."""
    
    def test_predict_proba_success(self, trained_model, processed_data):
        """Test probability prediction with supporting model."""
        X_test = processed_data['X_test']
        
        if hasattr(trained_model, 'predict_proba'):
            probabilities = predict_proba(trained_model, X_test)
            
            assert probabilities is not None
            assert isinstance(probabilities, np.ndarray)
            assert len(probabilities) == len(X_test)
            assert probabilities.shape[1] == 2  # Binary classification
            
            # Probabilities should sum to 1
            assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_predict_proba_not_supported(self, processed_data):
        """Test predict_proba with model that doesn't support it."""
        from sklearn.svm import LinearSVC
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        # LinearSVC doesn't have predict_proba by default
        model = LinearSVC(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        probabilities = predict_proba(model, processed_data['X_test'])
        
        assert probabilities is None


class TestPredictSingle:
    """Tests for predict_single function."""
    
    def test_predict_single_success(self, trained_model):
        """Test prediction for single sample."""
        features = {f'feature_{i}': np.random.randn() for i in range(15)}
        
        prediction, proba = predict_single(trained_model, features)
        
        assert prediction in ['Yes', 'No']
        
        if hasattr(trained_model, 'predict_proba'):
            assert proba is not None
            assert len(proba) == 2
        else:
            assert proba is None
    
    def test_predict_single_with_real_features(self, trained_model):
        """Test prediction with realistic feature dictionary."""
        features = {
            'tenure': 12,
            'MonthlyCharges': 50.5,
            'SeniorCitizen': 0,
            'gender': 1,
            'Partner': 1,
            'Dependents': 0,
            'PhoneService': 1,
            'MultipleLines': 0,
            'InternetService': 1,
            'OnlineSecurity': 1,
            'OnlineBackup': 0,
            'DeviceProtection': 1,
            'TechSupport': 0,
            'StreamingTV': 1,
            'StreamingMovies': 0
        }
        
        prediction, proba = predict_single(trained_model, features)
        
        assert prediction in ['Yes', 'No']


class TestBatchPredict:
    """Tests for batch_predict function."""
    
    def test_batch_predict_small_batches(self, trained_model, processed_data):
        """Test batch prediction with small batch size."""
        X_test = processed_data['X_test']
        
        predictions = batch_predict(trained_model, X_test, batch_size=5)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert all(pred in ['Yes', 'No'] for pred in predictions)
    
    def test_batch_predict_single_batch(self, trained_model, processed_data):
        """Test batch prediction with batch size larger than data."""
        X_test = processed_data['X_test']
        
        predictions = batch_predict(trained_model, X_test, batch_size=1000)
        
        assert len(predictions) == len(X_test)
    
    def test_batch_predict_consistency(self, trained_model, processed_data):
        """Test that batch prediction gives same results as regular predict."""
        X_test = processed_data['X_test']
        
        regular_predictions = predict(trained_model, X_test)
        batch_predictions = batch_predict(trained_model, X_test, batch_size=5)
        
        np.testing.assert_array_equal(regular_predictions, batch_predictions)


class TestModelPredictor:
    """Tests for ModelPredictor class."""
    
    def test_model_predictor_init(self, saved_model_file):
        """Test ModelPredictor initialization."""
        predictor = ModelPredictor(str(saved_model_file))
        
        assert predictor.model is not None
        assert predictor.pipeline is None
    
    def test_model_predictor_with_pipeline(self, saved_model_file, tmp_path):
        """Test ModelPredictor initialization with pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        pipeline = Pipeline([('scaler', StandardScaler())])
        pipeline_path = tmp_path / "pipeline.pkl"
        joblib.dump(pipeline, pipeline_path)
        
        predictor = ModelPredictor(str(saved_model_file), str(pipeline_path))
        
        assert predictor.model is not None
        assert predictor.pipeline is not None
    
    def test_model_predictor_predict(self, saved_model_file, processed_data):
        """Test prediction using ModelPredictor."""
        predictor = ModelPredictor(str(saved_model_file))
        X_test = processed_data['X_test']
        
        predictions = predictor.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
    
    def test_model_predictor_predict_proba(self, saved_model_file, processed_data):
        """Test probability prediction using ModelPredictor."""
        predictor = ModelPredictor(str(saved_model_file))
        X_test = processed_data['X_test']
        
        probabilities = predictor.predict_proba(X_test)
        
        if hasattr(predictor.model, 'predict_proba'):
            assert probabilities is not None
            assert len(probabilities) == len(X_test)
    
    def test_model_predictor_predict_single(self, saved_model_file):
        """Test single prediction using ModelPredictor."""
        predictor = ModelPredictor(str(saved_model_file))
        features = {f'feature_{i}': np.random.randn() for i in range(15)}
        
        prediction, proba = predictor.predict_single(features)
        
        assert prediction in ['Yes', 'No']
    
    def test_model_predictor_with_preprocessing(self, saved_model_file, tmp_path, processed_data):
        """Test ModelPredictor applies preprocessing pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # Create and fit pipeline
        X_train = processed_data['X_train']
        pipeline = Pipeline([('scaler', StandardScaler())])
        pipeline.fit(X_train)
        
        pipeline_path = tmp_path / "pipeline.pkl"
        joblib.dump(pipeline, pipeline_path)
        
        # Create predictor with pipeline
        predictor = ModelPredictor(str(saved_model_file), str(pipeline_path))
        
        # Make predictions (should apply pipeline)
        X_test = processed_data['X_test']
        predictions = predictor.predict(X_test)
        
        assert len(predictions) == len(X_test)


@pytest.mark.integration
class TestPredictionWorkflow:
    """Integration tests for complete prediction workflow."""
    
    def test_train_save_load_predict_workflow(self, processed_data, tmp_path):
        """Test complete workflow from training to prediction."""
        from src.train import train_random_forest, save_model
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        
        # Train model
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        # Save model
        model_path = save_model(model, 'test_model', str(tmp_path))
        
        # Load model
        loaded_model = load_model(model_path)
        
        # Make predictions
        predictions = predict(loaded_model, X_test)
        
        assert len(predictions) == len(X_test)
        assert all(pred in ['Yes', 'No'] for pred in predictions)
    
    def test_predictor_class_workflow(self, processed_data, tmp_path):
        """Test workflow using ModelPredictor class."""
        from src.train import train_random_forest, save_model
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        
        # Train and save
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        model_path = save_model(model, 'test_model', str(tmp_path))
        
        # Create predictor
        predictor = ModelPredictor(model_path)
        
        # Test all prediction methods
        predictions = predictor.predict(X_test)
        assert len(predictions) == len(X_test)
        
        if hasattr(model, 'predict_proba'):
            probabilities = predictor.predict_proba(X_test)
            assert probabilities is not None
        
        features = {f'feature_{i}': X_test[0, i] for i in range(X_test.shape[1])}
        single_pred, single_proba = predictor.predict_single(features)
        assert single_pred in ['Yes', 'No']


@pytest.mark.unit
class TestPredictionEdgeCases:
    """Tests for edge cases in prediction."""
    
    def test_predict_single_sample(self, trained_model):
        """Test prediction with single sample."""
        X_single = np.random.randn(1, 15)
        
        predictions = predict(trained_model, X_single)
        
        assert len(predictions) == 1
        assert predictions[0] in ['Yes', 'No']
    
    def test_predict_large_batch(self, trained_model):
        """Test prediction with large dataset."""
        X_large = np.random.randn(10000, 15)
        
        predictions = batch_predict(trained_model, X_large, batch_size=1000)
        
        assert len(predictions) == 10000
    
    def test_predict_different_dtypes(self, trained_model):
        """Test prediction with different input data types."""
        X_float64 = np.random.randn(10, 15).astype(np.float64)
        X_float32 = np.random.randn(10, 15).astype(np.float32)
        
        pred_64 = predict(trained_model, X_float64)
        pred_32 = predict(trained_model, X_float32)
        
        assert len(pred_64) == 10
        assert len(pred_32) == 10


@pytest.mark.parametrize("batch_size", [1, 5, 10, 100])
class TestParametrizedBatchSizes:
    """Parametrized tests for different batch sizes."""
    
    def test_batch_predict_various_sizes(self, trained_model, processed_data, batch_size):
        """Test batch prediction with various batch sizes."""
        X_test = processed_data['X_test']
        
        predictions = batch_predict(trained_model, X_test, batch_size=batch_size)
        
        assert len(predictions) == len(X_test)
        assert all(pred in ['Yes', 'No'] for pred in predictions)
