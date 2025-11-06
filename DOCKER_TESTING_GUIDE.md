# Docker Testing Guide - Local Machine

This guide provides comprehensive instructions for testing the Docker containerization on your local machine.

## Prerequisites

Before testing, ensure you have:
- ✅ Docker Desktop (macOS/Windows) or Docker Engine (Linux) installed
- ✅ Docker Compose installed
- ✅ Models in `models/` directory (`model_v1.pkl`, `model_v2.pkl`)
- ✅ Processed data in `data/processed/` directory (run `dvc pull` if needed)
- ✅ At least 2GB free disk space for Docker images

---

## Phase 1: Pre-Flight Checks

### 1.1 Verify Docker Installation

```bash
# Check Docker version
docker --version
# Expected: Docker version 24.x.x or higher

# Check Docker Compose version
docker-compose --version
# Expected: Docker Compose version v2.x.x or higher

# Verify Docker is running
docker ps
# Expected: Table showing containers (may be empty, but no errors)
```

### 1.2 Verify Required Files Exist

```bash
# Check models exist
ls -la models/
# Expected: model_v1.pkl and model_v2.pkl

# Check processed data exists
ls -la data/processed/
# Expected: preprocessing_pipeline.pkl and label_encoder.pkl

# Check Docker files exist
ls -la Dockerfile.mlapp Dockerfile.mlflow docker-compose.yml .dockerignore
# Expected: All four files present
```

### 1.3 Stop Conflicting Services

If you have local services running on ports 5000 or 5001, stop them:

```bash
# Check what's using port 5000
lsof -i :5000  # macOS/Linux
# or
netstat -ano | findstr :5000  # Windows

# Check what's using port 5001
lsof -i :5001  # macOS/Linux
# or
netstat -ano | findstr :5001  # Windows

# If needed, kill the process or stop services:
# - Stop local Flask app (Ctrl+C if running)
# - Stop local MLflow UI (Ctrl+C if running)
# - On macOS: Turn off AirPlay Receiver if using port 5000
```

---

## Phase 2: Build Docker Images

### 2.1 Build with Docker Compose

```bash
# Build both images (this may take 5-10 minutes the first time)
make docker-build

# Or directly with docker-compose:
docker-compose build

# Expected output:
# - Downloading Python base images
# - Installing dependencies from requirements.txt
# - Copying files into images
# - Successfully built messages for both services
```

### 2.2 Verify Images Were Created

```bash
# List Docker images
docker images

# Expected output should include:
# cmpt2500f25-project-tutorial-ml-app    (or similar name)
# cmpt2500f25-project-tutorial-mlflow    (or similar name)
# python:3.12-slim                        (base image)
```

### 2.3 Check Image Sizes

```bash
# View image sizes
docker images | grep cmpt2500

# Expected sizes (approximate):
# ml-app image: 400-600 MB
# mlflow image: 200-300 MB
```

**If build fails:**
- Check error messages carefully
- Verify `requirements.txt` is valid
- Ensure you have internet connection (to download base images)
- Check disk space: `df -h` (macOS/Linux) or `dir` (Windows)

---

## Phase 3: Start Containers

### 3.1 Start Services

```bash
# Start containers in detached mode
make docker-up

# Or directly:
docker-compose up -d

# Expected output:
# Creating network "ml-network"
# Creating mlflow-server ... done
# Creating churn-api ... done
```

### 3.2 Verify Containers Are Running

```bash
# Check container status
make docker-ps

# Or directly:
docker-compose ps

# Expected output:
# NAME            STATE    PORTS
# churn-api       Up       0.0.0.0:5000->5000/tcp
# mlflow-server   Up       0.0.0.0:5001->5001/tcp
```

### 3.3 Watch Logs (in a separate terminal)

```bash
# Follow logs from both containers
make docker-logs

# Or directly:
docker-compose logs -f

# Expected in logs:
# [ml-app] ✅ Models and pipelines loaded successfully
# [ml-app] Running on http://0.0.0.0:5000
# [mlflow] [INFO] Starting gunicorn
```

**Press Ctrl+C to stop following logs (containers keep running).**

**If containers fail to start:**
- Check logs: `docker-compose logs ml-app` or `docker-compose logs mlflow`
- Verify models and data files exist
- Check for port conflicts
- Ensure Docker has enough memory (Docker Desktop -> Settings -> Resources)

---

## Phase 4: Manual Testing - API Endpoints

### 4.1 Test Health Endpoint

```bash
# Test API health check
curl http://localhost:5000/health

# Expected output:
# {"status":"ok"}
```

### 4.2 Test Home/Documentation Endpoint

```bash
# Test home endpoint
curl http://localhost:5000/cmpt2500f25_tutorial_home

# Expected output:
# Large JSON with message, endpoints, and required_input_format
```

### 4.3 Test Prediction Endpoint (v1)

```bash
# Test single prediction
curl -X POST http://localhost:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

# Expected output:
# {
#   "model_version": "v1",
#   "prediction": "No",
#   "probability": 0.943...
# }
```

### 4.4 Test Prediction Endpoint (v2)

```bash
# Same payload, different endpoint
curl -X POST http://localhost:5000/v2/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'

# Expected output:
# {
#   "model_version": "v2",
#   "prediction": "No",
#   "probability": 0.880...
# }
```

### 4.5 Test Batch Prediction

```bash
# Test with array of 2 customers
curl -X POST http://localhost:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d '[
    {
      "tenure": 12, "MonthlyCharges": 59.95, "TotalCharges": 720.50,
      "Contract": "One year", "PaymentMethod": "Electronic check",
      "OnlineSecurity": "No", "TechSupport": "No", "InternetService": "DSL",
      "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes",
      "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
      "PaperlessBilling": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "No",
      "StreamingTV": "No", "StreamingMovies": "No"
    },
    {
      "tenure": 1, "MonthlyCharges": 70.70, "TotalCharges": 70.70,
      "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
      "OnlineSecurity": "No", "TechSupport": "No", "InternetService": "Fiber optic",
      "gender": "Male", "SeniorCitizen": "Yes", "Partner": "No",
      "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
      "PaperlessBilling": "Yes", "OnlineBackup": "No", "DeviceProtection": "No",
      "StreamingTV": "No", "StreamingMovies": "No"
    }
  ]'

# Expected output: Array with 2 predictions
```

### 4.6 Test Validation (Should Fail)

```bash
# Test with missing required fields
curl -X POST http://localhost:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 59.95
  }'

# Expected output (400 error):
# {
#   "error": "Missing required features: ..."
# }
```

---

## Phase 5: Manual Testing - MLflow UI

### 5.1 Test MLflow UI in Browser

1. Open your browser
2. Navigate to: `http://localhost:5001`
3. You should see the MLflow UI interface

**Expected:**
- MLflow Tracking UI loads
- Experiments list is visible
- No error messages

### 5.2 Test MLflow API (Optional)

```bash
# Test MLflow REST API
curl http://localhost:5001/api/2.0/mlflow/experiments/list

# Expected: JSON response with experiments
```

---

## Phase 6: Automated Testing with pytest

### 6.1 Run Docker Tests

```bash
# Run all Docker tests (requires containers to be running)
pytest tests/test_docker.py -v

# Expected output:
# tests/test_docker.py::test_docker_images_exist PASSED
# tests/test_docker.py::test_containers_are_running PASSED
# tests/test_docker.py::test_api_health_endpoint PASSED
# tests/test_docker.py::test_api_home_endpoint PASSED
# tests/test_docker.py::test_api_prediction_v1 PASSED
# tests/test_docker.py::test_api_prediction_v2 PASSED
# tests/test_docker.py::test_api_validation_error PASSED
# tests/test_docker.py::test_mlflow_ui_accessible PASSED
# ... and more
```

### 6.2 Run Tests in Container (Alternative)

```bash
# Run pytest inside the container
make docker-test

# Or directly:
docker-compose run --rm ml-app pytest tests/ -v

# This runs tests inside the container environment
```

---

## Phase 7: Container Inspection

### 7.1 Inspect Container Internals

```bash
# Open a shell in the API container
make docker-shell-app

# Or directly:
docker-compose exec ml-app /bin/bash

# Once inside, explore:
ls -la /app
ls -la /app/models
ls -la /app/data/processed
ls -la /app/logs
cat /app/src/app.py
python --version
pip list

# Exit the shell
exit
```

### 7.2 Inspect MLflow Container

```bash
# Open a shell in the MLflow container
make docker-shell-mlflow

# Or directly:
docker-compose exec mlflow /bin/bash

# Once inside:
ls -la /mlflow
ls -la /mlflow/mlruns
mlflow --version

# Exit
exit
```

### 7.3 Check Container Resource Usage

```bash
# View container stats (CPU, memory, network)
docker stats

# Expected:
# CONTAINER       CPU %   MEM USAGE / LIMIT   MEM %   NET I/O
# churn-api       0.5%    250MB / 2GB         12%     5kB / 3kB
# mlflow-server   0.2%    150MB / 2GB         7%      2kB / 1kB
```

---

## Phase 8: Network Testing

### 8.1 Verify Container Network

```bash
# List Docker networks
docker network ls

# Expected: ml-network should be listed

# Inspect the network
docker network inspect ml-network

# Expected: Both containers should be connected to this network
```

### 8.2 Test Inter-Container Communication

```bash
# From API container, ping MLflow container
docker-compose exec ml-app ping -c 3 mlflow

# Expected: Successful pings

# Test HTTP connection from API to MLflow
docker-compose exec ml-app curl http://mlflow:5001/

# Expected: MLflow UI HTML
```

---

## Phase 9: Log Analysis

### 9.1 View API Logs

```bash
# View last 50 lines of API logs
docker-compose logs --tail=50 ml-app

# Search for specific log entries
docker-compose logs ml-app | grep "prediction"
docker-compose logs ml-app | grep "ERROR"
```

### 9.2 View MLflow Logs

```bash
# View MLflow logs
docker-compose logs --tail=50 mlflow
```

### 9.3 Check Log Files in Volume

```bash
# If logs are mounted as volumes, check them on host
ls -la logs/
cat logs/api.log  # (if log files are configured)
```

---

## Phase 10: Performance Testing

### 10.1 Response Time Test

```bash
# Test response time with time command (macOS/Linux)
time curl -X POST http://localhost:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{...full payload...}'

# Expected: Response within 1-2 seconds
```

### 10.2 Load Testing (Optional)

```bash
# Install Apache Bench (if not already installed)
# macOS: brew install httpd
# Ubuntu: sudo apt-get install apache2-utils

# Send 100 requests with 10 concurrent connections
ab -n 100 -c 10 -p payload.json -T application/json http://localhost:5000/v1/predict

# Expected: All requests succeed, reasonable response times
```

---

## Phase 11: Cleanup and Restart

### 11.1 Stop Containers

```bash
# Stop containers (keeps images)
make docker-down

# Or directly:
docker-compose down

# Expected: Containers stopped and removed
```

### 11.2 Restart Containers

```bash
# Restart containers
make docker-restart

# Or stop and start:
make docker-down
make docker-up
```

### 11.3 Rebuild and Restart

```bash
# Rebuild images and restart (if you changed code)
docker-compose up -d --build
```

---

## Phase 12: Complete Cleanup (Optional)

⚠️ **Warning:** This removes all containers, images, and volumes!

```bash
# Complete cleanup
make docker-clean

# Or manually:
docker-compose down -v              # Stop and remove volumes
docker-compose down --rmi all       # Remove images
docker system prune -f              # Clean up unused resources

# Verify cleanup
docker images
docker ps -a
docker volume ls
```

---

## Common Issues and Troubleshooting

### Issue 1: Port Already in Use

**Error:** `Bind for 0.0.0.0:5000 failed: port is already allocated`

**Solution:**
```bash
# Find what's using the port
lsof -i :5000  # macOS/Linux

# Stop the conflicting service or change port in docker-compose.yml:
# ports:
#   - "5002:5000"  # Use host port 5002 instead
```

### Issue 2: Models Not Found

**Error:** `FileNotFoundError: models/model_v1.pkl`

**Solution:**
```bash
# Ensure models exist on host
ls -la models/

# Rebuild if you added models after building
docker-compose up -d --build
```

### Issue 3: Out of Disk Space

**Error:** `no space left on device`

**Solution:**
```bash
# Check Docker disk usage
docker system df

# Clean up unused resources
docker system prune -a -f

# Increase Docker Desktop disk limit (Desktop -> Settings -> Resources)
```

### Issue 4: Container Keeps Restarting

**Symptom:** `docker-compose ps` shows container constantly restarting

**Solution:**
```bash
# Check logs for errors
docker-compose logs ml-app

# Common causes:
# - Missing models/data files
# - Python dependencies not installed
# - Port conflicts
# - Application crash on startup
```

### Issue 5: Slow Build Times

**Solution:**
```bash
# Use .dockerignore to exclude unnecessary files
# Ensure you're not copying .venv, .git, mlruns, etc.

# Check build context size:
docker-compose build --no-cache --progress=plain
```

---

## Success Criteria Checklist

- ✅ Docker images build without errors
- ✅ Both containers start and remain running
- ✅ `/health` endpoint returns `{"status": "ok"}`
- ✅ `/cmpt2500f25_tutorial_home` returns full documentation JSON
- ✅ `/v1/predict` and `/v2/predict` return valid predictions
- ✅ Validation errors return 400 status with error messages
- ✅ MLflow UI accessible at `http://localhost:5001`
- ✅ All pytest tests in `tests/test_docker.py` pass
- ✅ Containers can communicate with each other on `ml-network`
- ✅ Logs are accessible via `docker-compose logs`
- ✅ Containers respond to stop/restart commands

---

## Next Steps

Once all tests pass:

1. ✅ Commit all Docker files to git
2. ✅ Update README.md with Docker instructions
3. ✅ Tag images for Docker Hub:
   ```bash
   docker tag cmpt2500f25-project-tutorial-ml-app:latest your-username/churn-api:latest
   docker tag cmpt2500f25-project-tutorial-mlflow:latest your-username/mlflow:latest
   ```
4. ✅ Push to Docker Hub:
   ```bash
   docker login
   docker push your-username/churn-api:latest
   docker push your-username/mlflow:latest
   ```
5. ✅ Ready for Lab 05: Cloud Deployment!

---

**Document Version:** 1.0
**Last Updated:** 2025-11-06
**For:** CMPT 2500 Lab 04 - Docker Containerization
