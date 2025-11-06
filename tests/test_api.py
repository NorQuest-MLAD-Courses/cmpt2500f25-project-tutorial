import pytest
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from src.app import app as flask_app # Import our Flask app
import src.app

# Create mock models and pipelines for testing
@pytest.fixture(autouse=True)
def setup_test_models(monkeypatch):
    """Set up mock models and pipelines for API testing."""

    # Create a simple preprocessing pipeline
    numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod", "SeniorCitizen"
    ]

    preprocessing_pipeline = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # Create a label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(['No', 'Yes'])

    # Create mock models - simple trained RandomForest classifiers
    # They need to be fitted with the right shape
    np.random.seed(42)

    # Create synthetic training data with the correct feature names
    import pandas as pd
    all_features = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
                    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaymentMethod", "tenure", "MonthlyCharges", "TotalCharges",
                    "SeniorCitizen"]

    synthetic_data = {
        "tenure": np.random.randint(1, 72, 100),
        "MonthlyCharges": np.random.uniform(20, 120, 100),
        "TotalCharges": np.random.uniform(100, 8000, 100),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], 100),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], 100),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], 100),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], 100),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], 100),
        "gender": np.random.choice(["Male", "Female"], 100),
        "SeniorCitizen": np.random.choice(["Yes", "No"], 100),
        "Partner": np.random.choice(["Yes", "No"], 100),
        "Dependents": np.random.choice(["Yes", "No"], 100),
        "PhoneService": np.random.choice(["Yes", "No"], 100),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], 100),
        "PaperlessBilling": np.random.choice(["Yes", "No"], 100),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], 100),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], 100),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], 100),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], 100),
    }

    X_train = pd.DataFrame(synthetic_data)
    y_train = np.random.choice(['No', 'Yes'], 100)

    # Encode labels numerically for training
    y_train_encoded = label_encoder.transform(y_train)

    # Fit the preprocessing pipeline
    preprocessing_pipeline.fit(X_train)
    X_train_processed = preprocessing_pipeline.transform(X_train)

    # Train mock models on encoded labels
    model_v1 = RandomForestClassifier(n_estimators=10, random_state=42)
    model_v1.fit(X_train_processed, y_train_encoded)

    model_v2 = RandomForestClassifier(n_estimators=10, random_state=43)
    model_v2.fit(X_train_processed, y_train_encoded)

    # Patch the global variables in src.app
    monkeypatch.setattr(src.app, 'pipeline', preprocessing_pipeline)
    monkeypatch.setattr(src.app, 'label_encoder', label_encoder)
    monkeypatch.setattr(src.app, 'model_v1', model_v1)
    monkeypatch.setattr(src.app, 'model_v2', model_v2)

# This is the pytest fixture
@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    # Set the app to testing mode
    flask_app.config['TESTING'] = True

    # Create a test client using the Flask application context
    with flask_app.test_client() as client:
        yield client # Provide this client to the test functions

def test_health_check(client):
    """Test the /health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json == {"status": "ok"}

def test_home_endpoint(client):
    """Test the /cmpt2500f25_tutorial_home endpoint."""
    response = client.get('/cmpt2500f25_tutorial_home')
    assert response.status_code == 200
    assert "message" in response.json
    assert "required_input_format" in response.json
    assert "numerical_features" in response.json["required_input_format"]

# A valid customer payload for testing
VALID_PAYLOAD = {
    "tenure": 12,
    "MonthlyCharges": 59.95,
    "TotalCharges": 720.50,
    "Contract": "One year",
    "PaymentMethod": "Electronic check",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "InternetService": "DSL",
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "PaperlessBilling": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No"
}

def test_v1_predict_single(client):
    """Test /v1/predict with a single valid record."""
    response = client.post('/v1/predict', json=VALID_PAYLOAD)
    
    assert response.status_code == 200
    assert "prediction" in response.json
    assert "probability" in response.json
    assert response.json["model_version"] == "v1"

def test_v1_predict_batch(client):
    """Test /v1/predict with a batch (list) of valid records."""
    # Create a batch of two identical valid records
    batch_payload = [VALID_PAYLOAD, VALID_PAYLOAD]
    response = client.post('/v1/predict', json=batch_payload)
    
    assert response.status_code == 200
    assert isinstance(response.json, list) # Check that the response is a list
    assert len(response.json) == 2 # Check that we got two predictions back
    assert response.json[0]["model_version"] == "v1"

def test_v1_predict_invalid_missing(client):
    """Test /v1/predict with missing features."""
    invalid_payload = {"tenure": 10, "MonthlyCharges": 50.0} # Missing most features
    response = client.post('/v1/predict', json=invalid_payload)
    
    assert response.status_code == 400 # Expect a Bad Request error
    assert "error" in response.json
    assert "Missing required features" in response.json["error"]

def test_v1_predict_invalid_type(client):
    """Test /v1/predict with an incorrect data type."""
    # Copy the valid payload and break it
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["tenure"] = "twelve" # Send a string instead of an int
    
    response = client.post('/v1/predict', json=invalid_payload)
    
    assert response.status_code == 400 # Expect a Bad Request error
    assert "error" in response.json
    assert "Invalid type for tenure" in response.json["error"]

def test_v2_predict_single(client):
    """Test /v2/predict with a single valid record."""
    response = client.post('/v2/predict', json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert "prediction" in response.json
    assert "probability" in response.json
    assert response.json["model_version"] == "v2" # Check for v2

def test_v2_predict_batch(client):
    """Test /v2/predict with a batch (list) of valid records."""
    # Create a batch of two identical valid records
    batch_payload = [VALID_PAYLOAD, VALID_PAYLOAD]
    response = client.post('/v2/predict', json=batch_payload)

    assert response.status_code == 200
    assert isinstance(response.json, list) # Check that the response is a list
    assert len(response.json) == 2 # Check that we got two predictions back
    assert response.json[0]["model_version"] == "v2"
    assert response.json[1]["model_version"] == "v2"

def test_v1_predict_empty_payload(client):
    """Test /v1/predict with an empty payload."""
    response = client.post('/v1/predict', json=None)

    assert response.status_code == 400
    assert "error" in response.json
    assert "No input data provided" in response.json["error"]

def test_v1_predict_null_totalcharges(client):
    """Test /v1/predict with TotalCharges=None (should be handled gracefully)."""
    payload = VALID_PAYLOAD.copy()
    payload["TotalCharges"] = None  # Set to None

    response = client.post('/v1/predict', json=payload)

    # Should succeed since the API handles None TotalCharges
    assert response.status_code == 200
    assert "prediction" in response.json
    assert "probability" in response.json

def test_v1_predict_missing_totalcharges(client):
    """Test /v1/predict without TotalCharges field (should be handled gracefully)."""
    payload = VALID_PAYLOAD.copy()
    del payload["TotalCharges"]  # Remove the field entirely

    response = client.post('/v1/predict', json=payload)

    # Should succeed since the API handles missing TotalCharges
    assert response.status_code == 200
    assert "prediction" in response.json
    assert "probability" in response.json

def test_v2_predict_invalid_type(client):
    """Test /v2/predict with an incorrect data type."""
    # Copy the valid payload and break it
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["MonthlyCharges"] = "fifty-nine"  # Send a string instead of a number

    response = client.post('/v2/predict', json=invalid_payload)

    assert response.status_code == 400 # Expect a Bad Request error
    assert "error" in response.json
    assert "Invalid type for MonthlyCharges" in response.json["error"]

def test_empty_list_prediction(client):
    """Test /v1/predict with an empty list."""
    response = client.post('/v1/predict', json=[])

    # Should succeed but return empty list
    assert response.status_code == 200
    assert isinstance(response.json, list)
    assert len(response.json) == 0
