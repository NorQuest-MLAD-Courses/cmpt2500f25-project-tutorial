"""
pytest fixtures for test suite.
Provides reusable test data, models, and configurations.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Generator

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Local imports
from src.preprocess import create_preprocessing_pipeline, save_preprocessed_data

# --- Constants ---
# Define the new, correct shape of our preprocessed data
# 3 numerical + 43 one-hot encoded categorical = 46
NEW_DATA_SHAPE_COLUMNS = 46


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Fixture to provide the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="module")
def temp_output_dir() -> Generator[Path, None, None]:
    """Fixture to create a temporary directory for test outputs."""
    dir_path = Path("./temp_test_output")
    dir_path.mkdir(exist_ok=True)
    yield dir_path
    # Teardown: remove the directory after tests
    shutil.rmtree(dir_path)

# --- Data Fixtures ---

@pytest.fixture(scope="session")
def sample_csv_data() -> pd.DataFrame:
    """Fixture to provide a small, sample DataFrame."""
    data = {
        'customerID': [f'1234-{i}' for i in range(5)],
        'gender': ['Female', 'Male', 'Female', 'Male', 'Male'],
        'SeniorCitizen': ["No", "Yes", "No", "No", "Yes"], # <-- FIX: Use strings
        'Partner': ['Yes', 'No', 'No', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'No', 'No', 'Yes'],
        'tenure': [1, 34, 2, 45, 12],
        'PhoneService': ['No', 'Yes', 'Yes', 'No', 'Yes'],
        'MultipleLines': ['No phone service', 'No', 'No', 'No phone service', 'Yes'],
        'InternetService': ['DSL', 'DSL', 'DSL', 'DSL', 'Fiber optic'],
        'OnlineSecurity': ['No', 'Yes', 'Yes', 'Yes', 'No'],
        'OnlineBackup': ['Yes', 'No', 'Yes', 'No', 'No'],
        'DeviceProtection': ['No', 'Yes', 'No', 'Yes', 'Yes'],
        'TechSupport': ['No', 'No', 'No', 'Yes', 'No'],
        'StreamingTV': ['No', 'No', 'No', 'No', 'Yes'],
        'StreamingMovies': ['No', 'No', 'No', 'No', 'Yes'],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'One year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 99.65],
        'TotalCharges': [29.85, 1889.5, 108.15, 1840.75, 1200.0],
        'Churn': ['No', 'No', 'Yes', 'No', 'Yes']
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sample_csv_file(tmp_path_factory, sample_csv_data: pd.DataFrame) -> Path:
    """Fixture to create a sample CSV file."""
    fn = tmp_path_factory.mktemp("data") / "sample_data.csv"
    sample_csv_data.to_csv(fn, index=False)
    return fn

@pytest.fixture(scope="session")
def processed_data() -> Dict[str, np.ndarray]:
    """
    Fixture to provide sample preprocessed data.
    Uses the NEW, CORRECT shape (46 columns).
    """
    # --- THE FIX IS HERE ---
    X = np.random.rand(100, NEW_DATA_SHAPE_COLUMNS) 
    y = np.random.randint(0, 2, 100)
    
    # Create train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

# --- Model Fixtures ---

@pytest.fixture(scope="module")
def trained_model(processed_data: Dict[str, np.ndarray]) -> LogisticRegression:
    """Fixture to provide a basic trained model."""
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    model = LogisticRegression(random_state=42, max_iter=100)
    model.fit(X_train, y_train)
    return model

@pytest.fixture(scope="module")
def saved_model_file(trained_model: LogisticRegression, tmp_path_factory) -> Path:
    """Fixture to create a sample saved model file."""
    fn = tmp_path_factory.mktemp("models") / "test_model.pkl"
    joblib.dump(trained_model, fn)
    return fn

# --- Preprocessing Fixtures ---

@pytest.fixture(scope="session")
def sample_csv_file_large(tmp_path_factory) -> Path:
    """Fixture for a larger, more realistic CSV file."""
    # Create 200 rows of synthetic data
    data = {
        'customerID': [f'cust_{i}' for i in range(200)],
        'gender': np.random.choice(['Male', 'Female'], 200),
        'SeniorCitizen': np.random.choice(["No", "Yes"], 200), # <-- FIX: Use strings
        'Partner': np.random.choice(['Yes', 'No'], 200),
        'Dependents': np.random.choice(['Yes', 'No'], 200),
        'tenure': np.random.randint(1, 72, 200),
        'PhoneService': np.random.choice(['Yes', 'No'], 200),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], 200),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 200),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], 200),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], 200),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], 200),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], 200),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], 200),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], 200),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 200),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], 200),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 200),
        'MonthlyCharges': np.random.uniform(20.0, 120.0, 200).round(2),
        'TotalCharges': np.random.uniform(20.0, 8000.0, 200).round(2),
        'Churn': np.random.choice(['Yes', 'No'], 200)
    }
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    df.loc[df.sample(n=10).index, 'TotalCharges'] = np.nan
    
    fn = tmp_path_factory.mktemp("data_large") / "large_sample.csv"
    df.to_csv(fn, index=False)
    return fn

@pytest.fixture(scope="module")
def saved_pipeline_artifacts(sample_csv_file_large: Path, tmp_path_factory) -> Dict[str, str]:
    """
    Fixture to run the full preprocessing pipeline and save artifacts.
    This uses the *actual* pipeline, not dummy data.
    """
    from src.preprocess import preprocess_pipeline # Local import to use config
    
    output_dir = tmp_path_factory.mktemp("processed_artifacts")
    
    X_train, X_test, y_train, y_test, pipeline, encoder = preprocess_pipeline(
        str(sample_csv_file_large),
        scale=True,
        use_sklearn_pipeline=True
    )
    
    paths = save_preprocessed_data(
        X_train, X_test, y_train, y_test,
        pipeline=pipeline,
        label_encoder=encoder,
        output_dir=str(output_dir)
    )
    
    return paths

# This fixture is from the original conftest.py, we keep it.
@pytest.fixture(autouse=True)
def reset_random_state():
    """
    Reset random state before each test for reproducibility.
    This fixture runs automatically for every test.
    """
    np.random.seed(42)

# This fixture is from the original conftest.py, we keep it.
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
