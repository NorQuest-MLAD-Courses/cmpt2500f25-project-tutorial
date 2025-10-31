"""
Tests for src/predict.py module.
Tests model loading, predictions, and ModelPredictor class.
"""

import numpy as np
import pandas as pd
import pytest
import pickle
from pathlib import Path

# Local imports
from src.predict import (
    ModelPredictor,
    batch_predict,
    load_model,
    load_preprocessing_pipeline,
    predict,
    predict_proba,
    predict_single
)
# We can't import clean_data, so we'll test ModelPredictor's logic
# by mocking the pipeline transform
from unittest.mock import MagicMock


# Define the new, correct shape
NEW_DATA_SHAPE_COLUMNS = 46

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
        model_path = tmp_path / "pickle_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(trained_model, f)
        
        loaded_model = load_model(str(model_path))
        
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')


class TestLoadPreprocessingPipeline:
    """Tests for load_preprocessing_pipeline function."""
    
    def test_load_pipeline_success(self, saved_pipeline_artifacts):
        """Test successful pipeline loading."""
        pipeline_path = saved_pipeline_artifacts['pipeline']
        pipeline = load_preprocessing_pipeline(pipeline_path)
        
        assert pipeline is not None
        assert hasattr(pipeline, 'transform')

    def test_load_pipeline_not_found(self):
        """Test error handling for missing pipeline file."""
        with pytest.raises(FileNotFoundError):
            load_preprocessing_pipeline("nonexistent_pipeline.pkl")


class TestModelPredictor:
    """Tests for ModelPredictor class."""
    
    def test_model_predictor_init(self, saved_model_file, saved_pipeline_artifacts):
        """Test ModelPredictor initialization."""
        model_path = str(saved_model_file)
        pipeline_path = saved_pipeline_artifacts['pipeline']
        
        predictor = ModelPredictor(model_path, pipeline_path)
        
        assert predictor.model is not None
        assert predictor.pipeline is not None
        assert hasattr(predictor.model, 'predict')
        assert hasattr(predictor.pipeline, 'transform')
    
    def test_model_predictor_predict_raw(self, sample_csv_data, saved_model_file, saved_pipeline_artifacts):
        """Test prediction on raw DataFrame."""
        model_path = str(saved_model_file)
        pipeline_path = saved_pipeline_artifacts['pipeline']
        
        predictor = ModelPredictor(model_path, pipeline_path)
        
        # Use raw data (excluding target)
        raw_data = sample_csv_data.drop(columns=['Churn'])
        
        # --- FIX: Must clean data (e.g., SeniorCitizen) before predicting ---
        # We know clean_data is in preprocess, so we import it here for the test
        from src.preprocess import clean_data
        cleaned_data = clean_data(raw_data)
        
        predictions = predictor.predict(cleaned_data)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(raw_data)
        assert all(p in [0, 1] for p in predictions)

    def test_model_predictor_predict_preprocessed(self, processed_data, saved_model_file, saved_pipeline_artifacts):
        """Test prediction on preprocessed numpy array."""
        model_path = str(saved_model_file)
        pipeline_path = saved_pipeline_artifacts['pipeline']
        
        predictor = ModelPredictor(model_path, pipeline_path)
        
        X_test = processed_data['X_test']
        predictions = predictor.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)


# --- Generic Prediction Function Tests ---

class TestPredictFunctions:
    """Tests for predict, predict_proba, and predict_single."""
    
    def test_predict_single(self, trained_model):
        """Test prediction on a single sample."""
        # --- THE FIX IS HERE ---
        X_single = np.random.randn(1, NEW_DATA_SHAPE_COLUMNS) 
        
        prediction = predict_single(trained_model, X_single)
        
        assert isinstance(prediction, (int, np.integer))
        assert prediction in [0, 1]

    def test_predict_batch(self, trained_model):
        """Test prediction on a batch of samples."""
        # --- THE FIX IS HERE ---
        X_batch = np.random.randn(10, NEW_DATA_SHAPE_COLUMNS)
        
        predictions = predict(trained_model, X_batch)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions) # Check for numeric 0/1

    def test_predict_proba(self, trained_model):
        """Test probability prediction."""
        # --- THE FIX IS HERE ---
        X_batch = np.random.randn(5, NEW_DATA_SHAPE_COLUMNS)
        
        probabilities = predict_proba(trained_model, X_batch)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (5, 2)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
    def test_predict_different_dtypes(self, trained_model, processed_data):
        """Test prediction with different input data types."""
        X_float64 = processed_data['X_test'].astype(np.float64)
        X_float32 = processed_data['X_test'].astype(np.float32)
        
        pred_64 = predict(trained_model, X_float64)
        pred_32 = predict(trained_model, X_float32)
        
        assert len(pred_64) == len(X_float64)
        assert len(pred_32) == len(X_float32)

@pytest.mark.parametrize("batch_size", [1, 5, 10, 20])
class TestParametrizedBatchSizes:
    """Parametrized tests for different batch sizes."""
    
    def test_batch_predict_various_sizes(self, trained_model, processed_data):
        """Test batch_predict with various sizes."""
        X_test = processed_data['X_test'] # Has 20 samples
        predictions = batch_predict(trained_model, X_test, batch_size=batch_size)
        
        assert len(predictions) == len(X_test)
