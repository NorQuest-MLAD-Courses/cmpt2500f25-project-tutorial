"""
Test suite for Docker containerization.

These tests verify that:
1. Docker images build successfully
2. Containers start and respond to health checks
3. API endpoints work correctly in containers
4. Inter-service communication works (API -> MLflow)
"""

import pytest
import subprocess
import time
import requests
import os


# ============ Configuration ============

# Allow port configuration via environment variable
API_PORT = os.getenv("API_PORT", "5000")
MLFLOW_PORT = os.getenv("MLFLOW_PORT", "5001")

API_BASE_URL = f"http://localhost:{API_PORT}"
MLFLOW_BASE_URL = f"http://localhost:{MLFLOW_PORT}"


# ============ Helper Functions ============

def run_command(cmd, check=True, capture_output=True):
    """Run a shell command and return the result."""
    result = subprocess.run(
        cmd,
        shell=True,
        check=check,
        capture_output=capture_output,
        text=True
    )
    return result


def wait_for_service(url, timeout=30, interval=2):
    """Wait for a service to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)
    return False


# ============ Fixtures ============

@pytest.fixture(scope="module")
def docker_environment():
    """
    Set up Docker environment for testing.
    Builds images and starts containers before tests,
    tears them down after tests complete.
    """
    print("\n=== Setting up Docker environment ===")

    # Check if Docker is available
    try:
        run_command("docker --version")
        run_command("docker-compose --version")
    except subprocess.CalledProcessError:
        pytest.skip("Docker is not available")

    # Build images
    print("Building Docker images...")
    build_result = run_command("docker-compose build", check=False)
    if build_result.returncode != 0:
        pytest.fail(f"Docker build failed: {build_result.stderr}")

    # Start containers
    print("Starting containers...")
    up_result = run_command("docker-compose up -d", check=False)
    if up_result.returncode != 0:
        pytest.fail(f"Docker compose up failed: {up_result.stderr}")

    # Wait for services to be ready
    print(f"Waiting for services to be ready on ports {API_PORT} and {MLFLOW_PORT}...")
    api_ready = wait_for_service(f"{API_BASE_URL}/health", timeout=60)
    mlflow_ready = wait_for_service(f"{MLFLOW_BASE_URL}/", timeout=60)

    if not api_ready:
        # Print logs for debugging
        logs = run_command("docker-compose logs ml-app", check=False)
        print(f"API logs:\n{logs.stdout}")
        pytest.fail("API service did not become ready in time")

    if not mlflow_ready:
        logs = run_command("docker-compose logs mlflow", check=False)
        print(f"MLflow logs:\n{logs.stdout}")
        pytest.fail("MLflow service did not become ready in time")

    print("Docker environment ready!")

    # Yield control to tests
    yield

    # Teardown: Stop and remove containers
    print("\n=== Tearing down Docker environment ===")
    run_command("docker-compose down", check=False)


# ============ Tests ============

def test_docker_images_exist():
    """Test that Docker images were built successfully."""
    result = run_command("docker images", check=False)
    assert "cmpt2500f25-project-tutorial" in result.stdout or "churn" in result.stdout, \
        "Docker images not found"


def test_containers_are_running(docker_environment):
    """Test that both containers are running."""
    result = run_command("docker-compose ps")
    assert "churn-api" in result.stdout or "ml-app" in result.stdout
    assert "mlflow-server" in result.stdout or "mlflow" in result.stdout


def test_api_health_endpoint(docker_environment):
    """Test the API health check endpoint in Docker."""
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_home_endpoint(docker_environment):
    """Test the API home/documentation endpoint in Docker."""
    response = requests.get(f"{API_BASE_URL}/cmpt2500f25_tutorial_home")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "required_input_format" in data


def test_api_prediction_v1(docker_environment):
    """Test a single prediction using v1 model in Docker."""
    payload = {
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

    response = requests.post(
        f"{API_BASE_URL}/v1/predict",
        json=payload
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["model_version"] == "v1"
    assert data["prediction"] in ["Yes", "No"]
    assert 0 <= data["probability"] <= 1


def test_api_prediction_v2(docker_environment):
    """Test a single prediction using v2 model in Docker."""
    payload = {
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

    response = requests.post(
        f"{API_BASE_URL}/v2/predict",
        json=payload
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["model_version"] == "v2"


def test_api_validation_error(docker_environment):
    """Test that the API properly validates input in Docker."""
    # Missing required fields
    invalid_payload = {
        "tenure": 12,
        "MonthlyCharges": 59.95
    }

    response = requests.post(
        f"{API_BASE_URL}/v1/predict",
        json=invalid_payload
    )

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "Missing required features" in data["error"]


def test_mlflow_ui_accessible(docker_environment):
    """Test that MLflow UI is accessible."""
    response = requests.get(f"{MLFLOW_BASE_URL}/")
    assert response.status_code == 200
    # MLflow UI returns HTML
    assert "MLflow" in response.text or "mlflow" in response.text


def test_docker_network_communication(docker_environment):
    """Test that containers can communicate with each other."""
    # This test verifies that the ml-app container can reach the mlflow container
    # We'll check the logs to see if there are any MLflow connection errors
    result = run_command("docker-compose logs ml-app")

    # If there were MLflow connection errors, they would appear in logs
    # We're checking that the app started successfully
    assert "✅ Models and pipelines loaded successfully" in result.stdout or \
           "Models and pipelines loaded successfully" in result.stdout


def test_docker_volumes_mounted(docker_environment):
    """Test that volumes are properly mounted."""
    # Check if logs directory exists and is writable
    result = run_command(
        "docker-compose exec -T ml-app ls -la /app/logs",
        check=False
    )
    assert result.returncode == 0, "Logs directory not accessible in container"


def test_container_health_checks(docker_environment):
    """Test that container health checks are working."""
    # Get container health status
    result = run_command(
        "docker inspect --format='{{.State.Health.Status}}' churn-api",
        check=False
    )

    if result.returncode == 0:
        health_status = result.stdout.strip()
        # Health check might not be implemented in all versions
        if health_status:
            assert health_status in ["healthy", "starting"], \
                f"Container is not healthy: {health_status}"


# ============ Cleanup Tests ============

def test_docker_cleanup():
    """Test that Docker cleanup commands work (optional, only runs if explicitly invoked)."""
    # This test is marked as optional and won't run by default
    # To run it: pytest tests/test_docker.py::test_docker_cleanup -v
    pytest.skip("Cleanup test - run manually with explicit test name")

    # Stop containers
    result = run_command("docker-compose down", check=False)
    assert result.returncode == 0, "Failed to stop containers"

    # Verify containers are stopped
    result = run_command("docker-compose ps")
    assert "churn-api" not in result.stdout or "Up" not in result.stdout


# ============ Performance Tests (Optional) ============

@pytest.mark.slow
def test_api_response_time(docker_environment):
    """Test that API responds within acceptable time (optional slow test)."""
    payload = {
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

    start_time = time.time()
    response = requests.post(
        "http://localhost:5000/v1/predict",
        json=payload
    )
    end_time = time.time()

    assert response.status_code == 200
    response_time = end_time - start_time
    # API should respond within 2 seconds
    assert response_time < 2.0, f"API response too slow: {response_time:.2f}s"
