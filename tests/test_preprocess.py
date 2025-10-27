"""
Tests for src/preprocess.py module.
Tests data loading, preprocessing, encoding, and pipeline creation.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

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
)


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
    
    def test_handle_missing_values_no_missing(self, sample_data):
        """Test handling when no missing values exist."""
        df_clean = handle_missing_values(sample_data)
        
        assert df_clean.shape == sample_data.shape
        assert df_clean.isnull().sum().sum() == 0
    
    def test_handle_missing_totalcharges(self, sample_data):
        """Test TotalCharges conversion and filling."""
        # Add missing TotalCharges
        df = sample_data.copy()
        df.loc[0, 'TotalCharges'] = ' '
        
        df_clean = handle_missing_values(df)
        
        assert df_clean['TotalCharges'].dtype in [np.float64, np.float32]
        assert df_clean['TotalCharges'].iloc[0] == 0.0
    
    def test_handle_missing_preserves_data(self, sample_data):
        """Test that function doesn't modify original DataFrame."""
        df_original = sample_data.copy()
        _ = handle_missing_values(sample_data)
        
        pd.testing.assert_frame_equal(sample_data, df_original)


class TestDropUnnecessaryColumns:
    """Tests for drop_unnecessary_columns function."""
    
    def test_drop_columns_success(self, sample_data):
        """Test dropping specified columns."""
        df_clean = drop_unnecessary_columns(sample_data, columns=['customerID'])
        
        assert 'customerID' not in df_clean.columns
        assert len(df_clean.columns) == len(sample_data.columns) - 1
    
    def test_drop_nonexistent_columns(self, sample_data):
        """Test dropping columns that don't exist."""
        df_clean = drop_unnecessary_columns(sample_data, columns=['nonexistent'])
        
        # Should not raise error, just return original columns
        assert len(df_clean.columns) == len(sample_data.columns)
    
    def test_drop_multiple_columns(self, sample_data):
        """Test dropping multiple columns."""
        df_clean = drop_unnecessary_columns(
            sample_data,
            columns=['customerID', 'gender']
        )
        
        assert 'customerID' not in df_clean.columns
        assert 'gender' not in df_clean.columns
        assert len(df_clean.columns) == len(sample_data.columns) - 2


class TestEncodeTarget:
    """Tests for encode_target function."""
    
    def test_encode_target_success(self, sample_data):
        """Test target encoding."""
        df_encoded, encoder = encode_target(sample_data, target_col='Churn')
        
        assert isinstance(encoder, LabelEncoder)
        assert df_encoded['Churn'].dtype in [np.int64, np.int32]
        assert set(df_encoded['Churn'].unique()).issubset({0, 1})
    
    def test_encode_target_missing_column(self, sample_data):
        """Test encoding with missing target column."""
        df_encoded, encoder = encode_target(sample_data, target_col='nonexistent')
        
        assert encoder is None
        pd.testing.assert_frame_equal(df_encoded, sample_data)
    
    def test_encode_target_preserves_mapping(self, sample_data):
        """Test that encoding preserves label mapping."""
        df_encoded, encoder = encode_target(sample_data, target_col='Churn')
        
        # Check that we can inverse transform
        original_labels = encoder.inverse_transform(df_encoded['Churn'])
        assert all(label in ['Yes', 'No'] for label in original_labels)


class TestSplitFeaturesTarget:
    """Tests for split_features_target function."""
    
    def test_split_features_target_success(self, sample_data):
        """Test successful feature/target split."""
        X, y = split_features_target(sample_data, target_col='Churn')
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert 'Churn' not in X.columns
        assert len(X) == len(y) == len(sample_data)
    
    def test_split_features_target_missing_target(self, sample_data):
        """Test error when target column missing."""
        with pytest.raises(ValueError, match="Target column.*not found"):
            split_features_target(sample_data, target_col='nonexistent')
    
    def test_split_features_target_shapes(self, sample_data):
        """Test output shapes are correct."""
        X, y = split_features_target(sample_data, target_col='Churn')
        
        assert X.shape[0] == sample_data.shape[0]
        assert X.shape[1] == sample_data.shape[1] - 1
        assert y.shape[0] == sample_data.shape[0]


class TestCreatePreprocessingPipeline:
    """Tests for create_preprocessing_pipeline function."""
    
    def test_create_pipeline_success(self):
        """Test pipeline creation."""
        numerical = ['tenure', 'MonthlyCharges']
        categorical = ['gender', 'Contract']
        
        pipeline = create_preprocessing_pipeline(numerical, categorical)
        
        assert isinstance(pipeline, Pipeline)
        assert 'preprocessor' in pipeline.named_steps
    
    def test_create_pipeline_empty_features(self):
        """Test pipeline with empty feature lists."""
        pipeline = create_preprocessing_pipeline([], [])
        
        assert isinstance(pipeline, Pipeline)
    
    def test_create_pipeline_onehot_encoding(self):
        """Test pipeline with one-hot encoding."""
        numerical = ['tenure']
        categorical = ['gender']
        
        pipeline = create_preprocessing_pipeline(
            numerical, categorical, encoding='onehot'
        )
        
        assert isinstance(pipeline, Pipeline)


class TestPreprocessDataWithPipeline:
    """Tests for preprocess_data_with_pipeline function."""
    
    def test_preprocess_with_pipeline_fit(self, sample_data_large):
        """Test preprocessing with pipeline fitting."""
        # Prepare data
        df = sample_data_large.copy()
        df = handle_missing_values(df)
        df = drop_unnecessary_columns(df, columns=['customerID'])
        df, _ = encode_target(df)
        X, y = split_features_target(df)
        
        X_transformed, pipeline = preprocess_data_with_pipeline(X, fit=True)
        
        assert isinstance(X_transformed, np.ndarray)
        assert isinstance(pipeline, Pipeline)
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_preprocess_with_pipeline_transform(self, sample_data_large):
        """Test preprocessing with existing pipeline (transform only)."""
        # Prepare training data
        df_train = sample_data_large.iloc[:80].copy()
        df_train = handle_missing_values(df_train)
        df_train = drop_unnecessary_columns(df_train, columns=['customerID'])
        df_train, _ = encode_target(df_train)
        X_train, y_train = split_features_target(df_train)
        
        # Fit pipeline
        X_train_transformed, pipeline = preprocess_data_with_pipeline(X_train, fit=True)
        
        # Prepare test data
        df_test = sample_data_large.iloc[80:].copy()
        df_test = handle_missing_values(df_test)
        df_test = drop_unnecessary_columns(df_test, columns=['customerID'])
        df_test, _ = encode_target(df_test)
        X_test, y_test = split_features_target(df_test)
        
        # Transform test data
        X_test_transformed, _ = preprocess_data_with_pipeline(
            X_test, pipeline=pipeline, fit=False
        )
        
        assert X_test_transformed.shape[0] == X_test.shape[0]
        assert X_test_transformed.shape[1] == X_train_transformed.shape[1]


class TestPreprocessPipeline:
    """Tests for complete preprocess_pipeline function."""
    
    def test_preprocess_pipeline_sklearn(self, sample_csv_file_large):
        """Test complete pipeline with sklearn pipelines."""
        result = preprocess_pipeline(
            str(sample_csv_file_large),
            scale=True,
            use_sklearn_pipeline=True
        )
        
        X_train, X_test, y_train, y_test, pipeline, label_encoder = result
        
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(label_encoder, LabelEncoder)
        
        # Check shapes
        assert len(X_train) > len(X_test)  # Default 80/20 split
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
    
    def test_preprocess_pipeline_legacy(self, sample_csv_file_large):
        """Test legacy preprocessing approach."""
        result = preprocess_pipeline(
            str(sample_csv_file_large),
            scale=True,
            use_sklearn_pipeline=False
        )
        
        X_train, X_test, y_train, y_test, scaler = result
        
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert len(X_train) > len(X_test)
    
    def test_preprocess_pipeline_no_scale(self, sample_csv_file_large):
        """Test pipeline without scaling."""
        result = preprocess_pipeline(
            str(sample_csv_file_large),
            scale=False,
            use_sklearn_pipeline=False
        )
        
        X_train, X_test, y_train, y_test = result
        
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)


class TestSavePreprocessedData:
    """Tests for save_preprocessed_data function."""
    
    def test_save_preprocessed_data_success(self, processed_data, temp_output_dir):
        """Test saving preprocessed data."""
        paths = save_preprocessed_data(
            processed_data['X_train'],
            processed_data['X_test'],
            processed_data['y_train'],
            processed_data['y_test'],
            output_dir=str(temp_output_dir)
        )
        
        assert 'data' in paths
        assert (temp_output_dir / 'preprocessed_data.npy').exists()
    
    def test_save_with_pipeline(self, processed_data, temp_output_dir):
        """Test saving with pipeline and encoder."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        pipeline = Pipeline([('scaler', StandardScaler())])
        encoder = LabelEncoder()
        encoder.fit(['Yes', 'No'])
        
        paths = save_preprocessed_data(
            processed_data['X_train'],
            processed_data['X_test'],
            processed_data['y_train'],
            processed_data['y_test'],
            pipeline=pipeline,
            label_encoder=encoder,
            output_dir=str(temp_output_dir)
        )
        
        assert 'data' in paths
        assert 'pipeline' in paths
        assert 'label_encoder' in paths
        assert (temp_output_dir / 'preprocessing_pipeline.pkl').exists()
        assert (temp_output_dir / 'label_encoder.pkl').exists()
    
    def test_load_saved_data(self, processed_data, temp_output_dir):
        """Test that saved data can be loaded correctly."""
        save_preprocessed_data(
            processed_data['X_train'],
            processed_data['X_test'],
            processed_data['y_train'],
            processed_data['y_test'],
            output_dir=str(temp_output_dir)
        )
        
        # Load the saved data
        loaded_data = np.load(
            temp_output_dir / 'preprocessed_data.npy',
            allow_pickle=True
        ).item()
        
        assert 'X_train' in loaded_data
        assert 'X_test' in loaded_data
        assert 'y_train' in loaded_data
        assert 'y_test' in loaded_data
        
        np.testing.assert_array_equal(loaded_data['X_train'], processed_data['X_train'])
        np.testing.assert_array_equal(loaded_data['y_train'], processed_data['y_train'])


@pytest.mark.integration
class TestPreprocessingIntegration:
    """Integration tests for complete preprocessing workflow."""
    
    def test_full_preprocessing_workflow(self, sample_csv_file_large, temp_output_dir):
        """Test complete preprocessing from CSV to saved data."""
        # Run complete pipeline
        X_train, X_test, y_train, y_test, pipeline, encoder = preprocess_pipeline(
            str(sample_csv_file_large),
            scale=True,
            use_sklearn_pipeline=True
        )
        
        # Save results
        paths = save_preprocessed_data(
            X_train, X_test, y_train, y_test,
            pipeline=pipeline,
            label_encoder=encoder,
            output_dir=str(temp_output_dir)
        )
        
        # Verify all components saved
        assert all(key in paths for key in ['data', 'pipeline', 'label_encoder'])
        
        # Load and verify
        loaded_data = np.load(paths['data'], allow_pickle=True).item()
        assert loaded_data['X_train'].shape == X_train.shape
        
        import joblib
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
        np.testing.assert_array_equal(result1[2], result2[2])  # y_train
