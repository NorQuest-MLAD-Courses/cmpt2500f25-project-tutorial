"""
pytest fixtures for test suite.
Provides reusable test data, models, and configurations.
"""

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


@pytest.fixture
def sample_data():
    """
    Create small sample dataset for testing.
    
    Returns:
        DataFrame with sample telecom churn data
    """
    data = {
        'customerID': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0, 0, 1],
        'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'Yes', 'No', 'Yes'],
        'tenure': [12, 24, 36, 6, 48],
        'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No', 'No', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic'],
        'OnlineSecurity': ['Yes', 'No', 'No', 'Yes', 'No'],
        'OnlineBackup': ['No', 'Yes', 'No', 'Yes', 'Yes'],
        'DeviceProtection': ['Yes', 'No', 'No', 'Yes', 'Yes'],
        'TechSupport': ['No', 'No', 'No', 'Yes', 'No'],
        'StreamingTV': ['Yes', 'No', 'No', 'Yes', 'Yes'],
        'StreamingMovies': ['No', 'Yes', 'No', 'No', 'Yes'],
        'Contract': ['Month-to-month', 'Two year', 'Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card', 'Electronic check'],
        'MonthlyCharges': [50.5, 70.25, 25.0, 45.75, 95.0],
        'TotalCharges': ['606', '1686.5', '900', '274.5', '4560'],
        'Churn': ['No', 'No', 'Yes', 'No', 'Yes']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_large():
    """
    Create larger sample dataset for testing (100 rows).
    
    Returns:
        DataFrame with larger sample data
    """
    np.random.seed(42)
    n_samples = 100
    
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
    
    return pd.DataFrame(data)


@pytest.fixture
def processed_data():
    """
    Create preprocessed training and test data.
    
    Returns:
        Dictionary with X_train, X_test, y_train, y_test
    """
    np.random.seed(42)
    
    # Create synthetic preprocessed data
    n_train = 80
    n_test = 20
    n_features = 15
    
    X_train = np.random.randn(n_train, n_features)
    X_test = np.random.randn(n_test, n_features)
    y_train = np.random.choice(['Yes', 'No'], n_train)
    y_test = np.random.choice(['Yes', 'No'], n_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


@pytest.fixture
def trained_model(processed_data):
    """
    Create a simple trained model for testing.
    
    Returns:
        Trained RandomForestClassifier
    """
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model


@pytest.fixture
def trained_logistic_model(processed_data):
    """
    Create a trained logistic regression model for testing.
    
    Returns:
        Trained LogisticRegression
    """
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    return model


@pytest.fixture
def scaler_fitted():
    """
    Create a fitted StandardScaler.
    
    Returns:
        Fitted StandardScaler
    """
    X_sample = np.random.randn(50, 10)
    scaler = StandardScaler()
    scaler.fit(X_sample)
    return scaler


@pytest.fixture
def label_encoder_fitted():
    """
    Create a fitted LabelEncoder.
    
    Returns:
        Fitted LabelEncoder
    """
    y_sample = np.array(['Yes', 'No', 'Yes', 'No', 'Yes'])
    encoder = LabelEncoder()
    encoder.fit(y_sample)
    return encoder


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Create temporary output directory for tests.
    
    Args:
        tmp_path: Pytest's tmp_path fixture
        
    Returns:
        Path to temporary output directory
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_models_dir(tmp_path):
    """
    Create temporary models directory for tests.
    
    Args:
        tmp_path: Pytest's tmp_path fixture
        
    Returns:
        Path to temporary models directory
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    """
    Create temporary data directory for tests.
    
    Args:
        tmp_path: Pytest's tmp_path fixture
        
    Returns:
        Path to temporary data directory
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_csv_file(tmp_path, sample_data):
    """
    Create temporary CSV file with sample data.
    
    Args:
        tmp_path: Pytest's tmp_path fixture
        sample_data: Sample DataFrame fixture
        
    Returns:
        Path to temporary CSV file
    """
    csv_path = tmp_path / "sample_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_csv_file_large(tmp_path, sample_data_large):
    """
    Create temporary CSV file with larger sample data.
    
    Args:
        tmp_path: Pytest's tmp_path fixture
        sample_data_large: Large sample DataFrame fixture
        
    Returns:
        Path to temporary CSV file
    """
    csv_path = tmp_path / "sample_data_large.csv"
    sample_data_large.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def saved_model_file(tmp_path, trained_model):
    """
    Create temporary saved model file.
    
    Args:
        tmp_path: Pytest's tmp_path fixture
        trained_model: Trained model fixture
        
    Returns:
        Path to saved model file
    """
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(trained_model, model_path)
    return model_path


@pytest.fixture
def saved_preprocessed_data(tmp_path, processed_data):
    """
    Create temporary saved preprocessed data file.
    
    Args:
        tmp_path: Pytest's tmp_path fixture
        processed_data: Processed data fixture
        
    Returns:
        Path to saved data file
    """
    data_path = tmp_path / "preprocessed_data.npy"
    np.save(data_path, processed_data)
    return data_path


@pytest.fixture
def sample_predictions():
    """
    Create sample predictions for testing.
    
    Returns:
        Dictionary with y (true labels) and yhat (predictions)
    """
    np.random.seed(42)
    n_samples = 50
    
    y = np.random.choice(['Yes', 'No'], n_samples)
    # Create predictions with some errors
    yhat = y.copy()
    error_indices = np.random.choice(n_samples, size=10, replace=False)
    yhat[error_indices] = np.where(yhat[error_indices] == 'Yes', 'No', 'Yes')
    
    return {'y': y, 'yhat': yhat}


@pytest.fixture
def empty_dataframe():
    """
    Create empty DataFrame for testing edge cases.
    
    Returns:
        Empty DataFrame
    """
    return pd.DataFrame()


@pytest.fixture
def dataframe_with_missing():
    """
    Create DataFrame with missing values for testing.
    
    Returns:
        DataFrame with NaN values
    """
    data = {
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [np.nan, 2, 3, 4, 5],
        'feature3': [1, 2, 3, np.nan, 5],
        'target': ['Yes', 'No', 'Yes', np.nan, 'No']
    }
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def reset_random_state():
    """
    Reset random state before each test for reproducibility.
    This fixture runs automatically for every test.
    """
    np.random.seed(42)


@pytest.fixture
def mock_mlflow_run(monkeypatch):
    """
    Mock MLflow tracking for testing without actual MLflow calls.
    
    Args:
        monkeypatch: Pytest's monkeypatch fixture
        
    Returns:
        Mock MLflow context manager
    """
    class MockRun:
        def __init__(self):
            self.info = type('obj', (object,), {'run_id': 'test_run_123'})()
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
    
    def mock_start_run(*args, **kwargs):
        return MockRun()
    
    def mock_log_param(*args, **kwargs):
        pass
    
    def mock_log_metric(*args, **kwargs):
        pass
    
    def mock_log_artifact(*args, **kwargs):
        pass
    
    def mock_set_tag(*args, **kwargs):
        pass
    
    def mock_set_experiment(*args, **kwargs):
        pass
    
    # Patch MLflow functions
    import mlflow
    monkeypatch.setattr(mlflow, "start_run", mock_start_run)
    monkeypatch.setattr(mlflow, "log_param", mock_log_param)
    monkeypatch.setattr(mlflow, "log_metric", mock_log_metric)
    monkeypatch.setattr(mlflow, "log_artifact", mock_log_artifact)
    monkeypatch.setattr(mlflow, "set_tag", mock_set_tag)
    monkeypatch.setattr(mlflow, "set_experiment", mock_set_experiment)
    
    return MockRun()
