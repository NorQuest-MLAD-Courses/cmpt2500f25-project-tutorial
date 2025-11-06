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
from src.preprocess import preprocess_pipeline, save_preprocessed_data
from src.train import save_model, train_all_models, train_random_forest


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
        
        # Step 2: Train model
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        assert hasattr(model, 'predict')
        
        # Step 3: Make predictions
        predictions = predict(model, X_test)

        assert len(predictions) == len(X_test)
        # Predictions are numeric (0, 1) since target was label-encoded
        # Decode them to original labels
        if encoder is not None:
            decoded_predictions = encoder.inverse_transform(predictions)
            assert all(pred in ['Yes', 'No'] for pred in decoded_predictions)
        else:
            # If no encoder, predictions should be numeric
            assert all(pred in [0, 1] for pred in predictions)
        
        # Step 4: Evaluate
        results = evaluate_model(model, X_test, y_test)
        
        assert 'accuracy' in results
        assert 0.0 <= results['accuracy'] <= 1.0
    
    def test_full_pipeline_with_saving(self, sample_csv_file_large, tmp_path):
        """Test complete workflow with saving all artifacts."""
        # Preprocess
        X_train, X_test, y_train, y_test, pipeline, encoder = preprocess_pipeline(
            str(sample_csv_file_large),
            scale=True,
            use_sklearn_pipeline=True
        )
        
        # Save preprocessed data
        data_paths = save_preprocessed_data(
            X_train, X_test, y_train, y_test,
            pipeline=pipeline,
            label_encoder=encoder,
            output_dir=str(tmp_path / 'data')
        )
        
        assert Path(data_paths['data']).exists()
        assert Path(data_paths['pipeline']).exists()
        assert Path(data_paths['label_encoder']).exists()
        
        # Train model
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        # Save model
        model_path = save_model(model, 'test_model', str(tmp_path / 'models'))
        
        assert Path(model_path).exists()
        
        # Load everything back
        loaded_data = np.load(data_paths['data'], allow_pickle=True).item()
        loaded_pipeline = joblib.load(data_paths['pipeline'])
        loaded_model = joblib.load(model_path)
        
        # Make predictions with loaded artifacts
        predictions = loaded_model.predict(loaded_data['X_test'])
        
        assert len(predictions) == len(loaded_data['X_test'])


@pytest.mark.integration
class TestModelTrainingPipeline:
    """Tests for model training workflows."""
    
    def test_train_multiple_models_workflow(self, processed_data, tmp_path):
        """Test training multiple models and comparing them."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # Train all models
        models = train_all_models(X_train, y_train, tune_hyperparameters=False)
        
        assert len(models) >= 6  # Should have at least 6 models
        
        # Compare models
        comparison_df = compare_models(models, X_test, y_test)
        
        assert len(comparison_df) == len(models)
        assert 'Model' in comparison_df.columns
        assert 'Accuracy' in comparison_df.columns
        
        # Get best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = models[best_model_name]
        
        # Save best model
        model_path = save_model(best_model, best_model_name, str(tmp_path))
        
        assert Path(model_path).exists()
    
    def test_hyperparameter_tuning_workflow(self, processed_data):
        """Test workflow with hyperparameter tuning."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # Train with tuning
        model_default = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        model_tuned = train_random_forest(X_train, y_train, tune_hyperparameters=True)
        
        # Evaluate both
        results_default = evaluate_model(model_default, X_test, y_test)
        results_tuned = evaluate_model(model_tuned, X_test, y_test)
        
        # Both should have valid metrics
        assert 0.0 <= results_default['accuracy'] <= 1.0
        assert 0.0 <= results_tuned['accuracy'] <= 1.0


@pytest.mark.integration
class TestPredictionPipeline:
    """Tests for prediction workflows."""
    
    def test_model_predictor_workflow(self, sample_csv_file_large, tmp_path):
        """Test complete workflow using ModelPredictor class."""
        # Preprocess
        X_train, X_test, y_train, y_test, pipeline, encoder = preprocess_pipeline(
            str(sample_csv_file_large),
            scale=True,
            use_sklearn_pipeline=True
        )
        
        # Train and save
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        model_path = save_model(model, 'test_model', str(tmp_path / 'models'))

        # Save pipeline
        pipeline_path = tmp_path / 'pipeline.pkl'
        joblib.dump(pipeline, pipeline_path)

        # Create predictor WITHOUT pipeline since X_test is already transformed
        # In real usage, you'd pass raw data to a predictor with pipeline,
        # but here X_test is already transformed
        predictor = ModelPredictor(str(model_path), pipeline_path=None)

        # Make predictions (data is already transformed)
        predictions = predictor.predict(X_test)
        
        assert len(predictions) == len(X_test)
        
        # Test probability predictions
        if hasattr(model, 'predict_proba'):
            probabilities = predictor.predict_proba(X_test)
            assert probabilities is not None
            assert len(probabilities) == len(X_test)
    
    def test_batch_prediction_workflow(self, processed_data, tmp_path):
        """Test batch prediction on large dataset."""
        from src.predict import batch_predict
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        # Train and save
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        model_path = save_model(model, 'batch_model', str(tmp_path))
        
        # Load model
        loaded_model = joblib.load(model_path)
        
        # Generate large test set
        X_large = np.random.randn(1000, X_train.shape[1])
        
        # Batch predict
        predictions = batch_predict(loaded_model, X_large, batch_size=100)
        
        assert len(predictions) == 1000


@pytest.mark.integration
class TestEvaluationPipeline:
    """Tests for evaluation workflows."""
    
    def test_evaluate_and_save_workflow(self, processed_data, tmp_path):
        """Test evaluation and saving results workflow."""
        from src.evaluate import save_evaluation_results
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # Train model
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        # Evaluate
        results = evaluate_model(model, X_test, y_test)
        
        # Save results
        results_path = tmp_path / 'results.json'
        save_evaluation_results(results, str(results_path), model_name='RandomForest')
        
        assert results_path.exists()
        
        # Load and verify
        with open(results_path, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['model_name'] == 'RandomForest'
        assert 'accuracy' in loaded_results
        assert 'timestamp' in loaded_results
    
    def test_compare_multiple_models_workflow(self, processed_data, tmp_path):
        """Test comparing multiple models and selecting best."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # Train multiple models
        models = train_all_models(X_train, y_train, tune_hyperparameters=False)
        
        # Compare
        comparison_df = compare_models(models, X_test, y_test)
        
        # Save comparison
        comparison_path = tmp_path / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        
        assert comparison_path.exists()
        
        # Load and verify
        loaded_comparison = pd.read_csv(comparison_path)
        
        assert len(loaded_comparison) == len(models)
        assert 'Model' in loaded_comparison.columns


@pytest.mark.integration
class TestReproducibilityWorkflow:
    """Tests for reproducibility of ML pipeline."""
    
    def test_pipeline_reproducibility(self, sample_csv_file_large):
        """Test that pipeline produces same results when run twice."""
        # Run pipeline twice
        result1 = preprocess_pipeline(
            str(sample_csv_file_large),
            scale=True,
            use_sklearn_pipeline=True
        )
        
        result2 = preprocess_pipeline(
            str(sample_csv_file_large),
            scale=True,
            use_sklearn_pipeline=True
        )
        
        # Compare results
        np.testing.assert_array_almost_equal(result1[0], result2[0])  # X_train
        np.testing.assert_array_equal(result1[2], result2[2])  # y_train
    
    def test_model_training_reproducibility(self, processed_data):
        """Test that model training is reproducible."""
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        
        # Train same model twice
        model1 = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        model2 = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        # Predictions should be identical
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        
        np.testing.assert_array_equal(pred1, pred2)


@pytest.mark.integration
class TestErrorHandlingWorkflow:
    """Tests for error handling in complete workflows."""
    
    def test_missing_file_handling(self):
        """Test handling of missing input files."""
        with pytest.raises(FileNotFoundError):
            preprocess_pipeline("nonexistent_file.csv")
    
    def test_invalid_model_path(self):
        """Test handling of invalid model path."""
        from src.predict import load_model
        
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model.pkl")
    
    def test_empty_data_handling(self, tmp_path):
        """Test handling of empty DataFrame."""
        from src.preprocess import handle_missing_values
        
        empty_df = pd.DataFrame()
        
        # Should handle gracefully
        result = handle_missing_values(empty_df)
        assert isinstance(result, pd.DataFrame)


@pytest.mark.integration
@pytest.mark.slow
class TestLargeDataWorkflow:
    """Tests for workflows with larger datasets."""
    
    def test_large_dataset_pipeline(self, tmp_path):
        """Test complete pipeline with larger synthetic dataset."""
        # Create large synthetic dataset
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'customerID': [f'C{i:04d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(0, 72, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'
            ], n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples).round(2),
            'TotalCharges': (np.random.uniform(100, 8000, n_samples).round(2)).astype(str),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
        }
        
        df = pd.DataFrame(data)
        csv_path = tmp_path / "large_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Run complete pipeline
        X_train, X_test, y_train, y_test, pipeline, encoder = preprocess_pipeline(
            str(csv_path),
            scale=True,
            use_sklearn_pipeline=True
        )
        
        assert X_train.shape[0] > 500  # Should have substantial training data
        
        # Train
        model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
        
        # Predict
        predictions = predict(model, X_test)
        
        assert len(predictions) > 0
        
        # Evaluate
        results = evaluate_model(model, X_test, y_test)
        
        assert 0.0 <= results['accuracy'] <= 1.0


@pytest.mark.integration
class TestSaveLoadWorkflow:
    """Tests for saving and loading all artifacts."""
    
    def test_complete_save_load_cycle(self, sample_csv_file_large, tmp_path):
        """Test saving and loading all components of ML pipeline."""
        # Preprocess and save
        X_train, X_test, y_train, y_test, pipeline, encoder = preprocess_pipeline(
            str(sample_csv_file_large),
            scale=True,
            use_sklearn_pipeline=True
        )
        
        data_paths = save_preprocessed_data(
            X_train, X_test, y_train, y_test,
            pipeline=pipeline,
            label_encoder=encoder,
            output_dir=str(tmp_path / 'data')
        )
        
        # Train and save multiple models
        models = {
            'model1': train_random_forest(X_train, y_train, tune_hyperparameters=False),
            'model2': train_random_forest(X_train, y_train, tune_hyperparameters=False)
        }
        
        from src.train import save_all_models
        model_paths = save_all_models(models, output_dir=str(tmp_path / 'models'))
        
        # Verify all files exist
        assert all(Path(path).exists() for path in data_paths.values())
        assert all(Path(path).exists() for path in model_paths.values())
        
        # Load everything back
        loaded_data = np.load(data_paths['data'], allow_pickle=True).item()
        loaded_pipeline = joblib.load(data_paths['pipeline'])
        loaded_encoder = joblib.load(data_paths['label_encoder'])
        loaded_models = {name: joblib.load(path) for name, path in model_paths.items()}
        
        # Verify loaded objects work
        predictions = loaded_models['model1'].predict(loaded_data['X_test'])
        assert len(predictions) == len(loaded_data['X_test'])
    
    def test_model_versioning(self, trained_model, tmp_path):
        """Test that models are versioned with timestamps."""
        # Save same model multiple times
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
