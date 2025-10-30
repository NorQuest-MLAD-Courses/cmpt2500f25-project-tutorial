import pytest
import json
from src.app import app as flask_app # Import our Flask app

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
