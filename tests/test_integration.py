"""
Integration tests for complete ML workflows.
Tests end-to-end pipelines from data to deployment.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluate import compare_models, evaluate_model
from src.predict import ModelPredictor, predict
from src.preprocess import preprocess_pipeline, save_preprocessed_data, clean_data
from src.train import save_model, train_random_forest
# from src.train import train_all_models <-- REMOVED

# Define the new, correct shape
NEW_DATA_SHAPE_COLUMNS = 46

@pytest.mark.integration
class TestCompleteMLPipeline:
    """Tests for complete ML pipeline from data to predictions."""
    
    def test_full_pipeline_csv_to_predictions(self, sample_csv_file_large, tmp_path):
        """Test complete workflow: CSV → preprocessing → training → prediction."""
        # Step 1: Preprocess data
        X_train, X_test, y_train, y_test, pipeline, encoder = preprocess_pipeline(
            str(sample_csv_file_large),
            scale=True,
            use_sklearn_pipeline=True
        )
        
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        # --- THE FIX IS HERE ---
        assert X_train.shape[1] == NEW_DATA_SHAPE_COLUMNS
        assert X_test.shape[1] == NEW_DATA_SHAPE_COLUMNS
        
        # Step 2: Train model
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        assert hasattr(model, 'predict')
        
        # Step 3: Make predictions
        predictions_numeric = model.predict(X_test)
        predictions = encoder.inverse_transform(predictions_numeric) # <-- Convert to strings
        
        assert len(predictions) == len(X_test)
        assert all(pred in ['Yes', 'No'] for pred in predictions) # <-- Check for strings


@pytest.mark.integration
class TestModelPredictorIntegration:
    """Tests for ModelPredictor class in an integration context."""
    
    def test_model_predictor_workflow(self, sample_csv_file_large, tmp_path):
        """Test ModelPredictor loading artifacts and predicting."""
        # Step 1: Preprocess and save artifacts
        X_train, X_test, y_train, y_test, pipeline, encoder = preprocess_pipeline(
            str(sample_csv_file_large),
            scale=True,
            use_sklearn_pipeline=True
        )
        
        artifact_dir = tmp_path / "artifacts"
        artifact_paths = save_preprocessed_data(
            X_train, X_test, y_train, y_test,
            pipeline=pipeline,
            label_encoder=encoder,
            output_dir=str(artifact_dir)
        )
        
        # Step 2: Train and save model
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        model_path = save_model(model, "integration_model", str(artifact_dir))
        
        # Step 3: Load with ModelPredictor
        predictor = ModelPredictor(
            model_path=model_path,
            pipeline_path=artifact_paths['pipeline']
        )
        
        # Step 4: Predict on raw data (first 5 rows)
        raw_df = pd.read_csv(sample_csv_file_large).head(5)
        
        # --- FIX: Must clean data before predicting ---
        cleaned_df = clean_data(raw_df)

        predictions = predictor.predict(cleaned_df)
        
        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions) # ModelPredictor returns numeric


@pytest.mark.integration
class TestSaveLoadWorkflow:
    """Tests for saving and loading models."""
    
    def test_model_versioning(self, trained_model, tmp_path):
        """Test that save_model creates unique, versioned files."""
        # Save the same model twice
        path1 = save_model(trained_model, 'versioned_model', str(tmp_path))
        path2 = save_model(trained_model, 'versioned_model', str(tmp_path))
        
        # Paths should be different (timestamped)
        assert path1 != path2
        
        # Both files should exist
        assert Path(path1).exists()
        assert Path(path2).exists()
        
        # Both should be loadable
        model1 = joblib.load(path1)
        model2 = joblib.load(path2)
        
        assert hasattr(model1, 'predict')
        assert hasattr(model2, 'predict')


@pytest.mark.integration
class TestCLIWorkflow:
    """Tests for CLI-based workflows (would require subprocess calls)."""
    
    def test_preprocessing_saves_artifacts(self, sample_csv_file_large, tmp_path):
        """Test that preprocessing creates all expected artifacts."""
        X_train, X_test, y_train, y_test, pipeline, encoder = preprocess_pipeline(
            str(sample_csv_file_large),
            scale=True,
            use_sklearn_pipeline=True
        )
        
        paths = save_preprocessed_data(
            X_train, X_test, y_train, y_test,
            pipeline=pipeline,
            label_encoder=encoder,
            output_dir=str(tmp_path)
        )
        
        # Check all expected files exist
        expected_files = ['data', 'pipeline', 'label_encoder']
        assert all(key in paths for key in expected_files)
        assert all(Path(paths[key]).exists() for key in expected_files)
