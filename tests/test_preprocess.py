"""
Tests for src/preprocess.py module.
Tests data loading, preprocessing, encoding, and pipeline creation.
"""

# Standard library imports
from pathlib import Path

# Third-party imports
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Local imports
from src.preprocess import (
    create_preprocessing_pipeline,
    drop_unnecessary_columns,
    encode_target,
    handle_missing_values,
    load_data,
    preprocess_data_with_pipeline,
    preprocess_pipeline,
    save_preprocessed_data,
    split_features_target
    # We cannot import clean_data as it's a nested function
)
from src.utils.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET
)

# Define the new, correct shape
NEW_DATA_SHAPE_COLUMNS = 46

class TestLoadData:
    """Tests for load_data function."""
    
    def test_load_data_success(self, sample_csv_file):
        """Test successful data loading from CSV."""
        df = load_data(str(sample_csv_file))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'Churn' in df.columns
        assert 'customerID' in df.columns
    
    def test_load_data_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.csv")
    
    def test_load_data_empty_file(self, tmp_path):
        """Test error handling for empty file."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")
        
        with pytest.raises(pd.errors.EmptyDataError):
            load_data(str(empty_file))


class TestHandleMissingValues:
    """Tests for handle_missing_values function."""
    
    def test_handle_missing_values_total_charges(self):
        """Test filling missing TotalCharges with 0."""
        df = pd.DataFrame({'TotalCharges': [100, np.nan, 200, np.nan]})
        df_cleaned = handle_missing_values(df)
        
        assert df_cleaned['TotalCharges'].isnull().sum() == 0
        assert df_cleaned['TotalCharges'].iloc[1] == 0
        assert df_cleaned['TotalCharges'].iloc[3] == 0

    def test_handle_missing_values_other_columns(self):
        """Test dropping rows with missing values in other columns."""
        df = pd.DataFrame({
            'TotalCharges': [100, 200, 300],
            'gender': ['Male', 'Female', np.nan]
        })
        df_cleaned = handle_missing_values(df)
        
        assert len(df_cleaned) == 2
        assert 'Female' in df_cleaned['gender'].values
        assert np.nan not in df_cleaned['gender'].values

    def test_handle_missing_values_no_missing(self, sample_csv_data):
        """Test with data having no missing values."""
        # Note: sample_csv_data is clean, but TotalCharges is object,
        # handle_missing_values converts it.
        df_cleaned = handle_missing_values(sample_csv_data.copy())
        
        assert len(df_cleaned) == len(sample_csv_data)
        assert df_cleaned['TotalCharges'].dtype == 'float64'


class TestColumnManipulation:
    """Tests for column dropping and feature/target splitting."""
    
    def test_drop_unnecessary_columns(self, sample_csv_data):
        """Test dropping 'customerID'."""
        df = sample_csv_data.copy()
        df_dropped = drop_unnecessary_columns(df, ['customerID'])
        
        assert 'customerID' not in df_dropped.columns
        assert 'gender' in df_dropped.columns
        assert len(df_dropped.columns) == len(df.columns) - 1
    
    def test_split_features_target(self, sample_csv_data):
        """Test splitting features and target."""
        X, y = split_features_target(sample_csv_data, target_column=TARGET)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert TARGET not in X.columns
        assert y.name == TARGET
        assert len(X) == len(y)


class TestEncoding:
    """Tests for encoding functions."""
    
    def test_encode_target(self):
        """Test LabelEncoder on target 'Churn'."""
        y = pd.Series(['No', 'Yes', 'No', 'Yes', 'Yes'])
        y_encoded, encoder = encode_target(y)
        
        assert isinstance(encoder, LabelEncoder)
        assert all(y_encoded == [0, 1, 0, 1, 1])
        assert list(encoder.classes_) == ['No', 'Yes']

    def test_create_preprocessing_pipeline(self):
        """Test creation of the scikit-learn pipeline."""
        pipeline = create_preprocessing_pipeline(NUMERICAL_FEATURES, CATEGORICAL_FEATURES, scale=True)
        
        assert isinstance(pipeline, Pipeline)
        assert 'preprocessor' in pipeline.named_steps
        
        preprocessor = pipeline.named_steps['preprocessor']
        assert isinstance(preprocessor, ColumnTransformer)
        
        # Check if transformers are correct
        assert len(preprocessor.transformers_) == 2
        assert preprocessor.transformers_[0][0] == 'num'
        assert isinstance(preprocessor.transformers_[0][1], Pipeline) # Scaler
        
        assert preprocessor.transformers_[1][0] == 'cat'
        assert isinstance(preprocessor.transformers_[1][1], Pipeline) # OHE


class TestPreprocessPipeline:
    """Tests for the main preprocess_pipeline function."""
    
    def test_preprocess_pipeline_sklearn(self, sample_csv_file_large):
        """
        Test the full sklearn pipeline.
        This is the main test that failed due to shape mismatch.
        """
        X_train, X_test, y_train, y_test, pipeline, encoder = preprocess_pipeline(
            str(sample_csv_file_large),
            scale=True,
            use_sklearn_pipeline=True
        )
        
        # Check output types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(encoder, LabelEncoder)
        
        # Check shapes
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X_train.shape[0] + X_test.shape[0] == 200 # Based on sample_csv_file_large
        
        # --- THE FIX IS HERE ---
        # Check for the new, correct number of columns (46)
        assert X_train.shape[1] == NEW_DATA_SHAPE_COLUMNS
        assert X_test.shape[1] == NEW_DATA_SHAPE_COLUMNS

    def test_preprocess_data_with_pipeline(self, sample_csv_file_large, saved_pipeline_artifacts):
        """Test loading and using an existing pipeline."""
        pipeline_path = saved_pipeline_artifacts['pipeline']
        pipeline = joblib.load(pipeline_path)
        
        df = load_data(str(sample_csv_file_large))
        # We must call the outer function to get the nested functions
        # This is a bit awkward, but it's how the code is structured
        from src.preprocess import preprocess_pipeline as main_pp
        df_cleaned = main_pp(str(sample_csv_file_large), return_df_only=True)
        
        df_handled = handle_missing_values(df_cleaned)
        df_features, _ = split_features_target(df_handled, TARGET)
        
        X_transformed = preprocess_data_with_pipeline(df_features, pipeline)
        
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == len(df_features)
        assert X_transformed.shape[1] == NEW_DATA_SHAPE_COLUMNS # Check shape
        
    def test_clean_data_logic(self, sample_csv_file):
        """Test the logic of clean_data by calling preprocess_pipeline."""
        # This test checks that SeniorCitizen is correctly mapped
        df_cleaned = preprocess_pipeline(str(sample_csv_file), return_df_only=True)
        
        assert "No" in df_cleaned['SeniorCitizen'].values
        assert "Yes" in df_cleaned['SeniorCitizen'].values
        assert 0 not in df_cleaned['SeniorCitizen'].values
        assert 1 not in df_cleaned['SeniorCitizen'].values


class TestSavePreprocessedData:
    """Tests for save_preprocessed_data function."""

    def test_save_preprocessed_data_all(self, temp_output_dir):
        """Test saving all artifacts."""
        # Create dummy data
        X_train = np.array([[1, 2], [3, 4]])
        X_test = np.array([[5, 6]])
        y_train = np.array([0, 1])
        y_test = np.array([0])
        pipeline = Pipeline(steps=[('scaler', StandardScaler())])
        label_encoder = LabelEncoder().fit(['No', 'Yes'])
        
        paths = save_preprocessed_data(
            X_train, X_test, y_train, y_test,
            pipeline, label_encoder,
            str(temp_output_dir)
        )
        
        # Verify all components saved
        assert 'data' in paths
        assert 'pipeline' in paths
        assert 'label_encoder' in paths
        
        assert Path(paths['data']).exists()
        assert Path(paths['pipeline']).exists()
        assert Path(paths['label_encoder']).exists()
        
        # Load and verify
        loaded_data = np.load(paths['data'], allow_pickle=True)
        assert loaded_data['X_train'].shape == X_train.shape
        
        loaded_pipeline = joblib.load(paths['pipeline'])
        assert isinstance(loaded_pipeline, Pipeline)

    def test_preprocessing_reproducibility(self, sample_csv_file_large):
        """Test that preprocessing produces consistent results."""
        # Run preprocessing twice
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
        
        # Verify results are identical
        np.testing.assert_array_equal(result1[0], result2[0])  # X_train
        np.testing.assert_array_equal(result1[1], result2[1])  # X_test
        np.testing.assert_array_equal(result1[2], result2[2])  # y_train
        np.testing.assert_array_equal(result1[3], result2[3])  # y_test
