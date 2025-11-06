# API Documentation: CMPT 2500 Tutorial Project - Telecom Churn Prediction

## Overview

This API serves a machine learning model to predict customer churn. It exposes endpoints to check API health, get usage information, and receive predictions from two different model versions (v1 and v2).

This document provides instructions for setup and a comprehensive overview of all endpoints. For a detailed, interactive API reference, run the server and navigate to `/apidocs/`.

---

## Installation & Running

### 1. Setup Environment

Clone the repository and install the required Python packages.

```sh
git clone https://github.com/[YOUR_USERNAME]/cmpt2500f25-project-tutorial.git
cd cmpt2500f25-project-tutorial
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get Data & Artifacts

This project uses DVC to manage large data files and pipelines. You must set up your DagsHub credentials and pull the data and models.

(Follow the credential setup in `assignments/Lab 03 - REST API Development.md` if this is a new environment).

```sh
# Pull preprocessing artifacts
dvc pull data/processed

# Pull trained models
dvc pull models
```

This will download:
- `data/processed/preprocessing_pipeline.pkl` - Feature preprocessing pipeline
- `data/processed/label_encoder.pkl` - Label encoder for predictions
- `models/model_v1.pkl` - Best performing model (v1)
- `models/model_v2.pkl` - Second best model (v2)

### 3. Run the API Server

Ensure all artifacts are in place, then run the app:

```sh
python src/app.py
```

The server will start on `http://127.0.0.1:5000`.

---

## Endpoints

### 1. Health Check

**`GET /health`**

A simple health check to verify the API is running.

**Response (200 OK)**:
```json
{
  "status": "ok"
}
```

**Example**:
```sh
curl http://127.0.0.1:5000/health
```

---

### 2. API Information

**`GET /cmpt2500f25_tutorial_home`**

Provides usage information, available endpoints, and required input format.

**Response (200 OK)**:
```json
{
  "message": "Welcome to the CMPT 2500 F25 Project Tutorial API!",
  "api_documentation": "Find the interactive documentation at /apidocs/",
  "model_versions_available": ["/v1/predict", "/v2/predict"],
  "required_input_format": {
    "description": "A JSON object (or list of objects) with all 19 features.",
    "numerical_features": ["tenure", "MonthlyCharges", "TotalCharges"],
    "categorical_features": [
      "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
      "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
      "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
      "Contract", "PaymentMethod", "SeniorCitizen"
    ],
    "example_single_record": { ... }
  }
}
```

**Example**:
```sh
curl http://127.0.0.1:5000/cmpt2500f25_tutorial_home
```

---

### 3. Prediction Endpoints

**`POST /v1/predict`** - Best Model (v1)
**`POST /v2/predict`** - Second Best Model (v2)

Generates churn predictions using the specified model version. Accepts a single JSON object or a list of objects for batch prediction.

#### Model Information:
- **v1**: Best performing model from training (highest accuracy/F1-score)
- **v2**: Second best model - useful for comparison or ensemble approaches

#### Single Prediction

**Request Body**:
```json
{
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
```

**Success Response (200 OK)**:
```json
{
  "prediction": "No",
  "probability": 0.9431,
  "model_version": "v1"
}
```

**Example**:
```sh
curl -X POST http://127.0.0.1:5000/v1/predict \
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
```

#### Batch Prediction

**Request Body** (array of customer objects):
```json
[
  {
    "tenure": 12,
    "MonthlyCharges": 59.95,
    "TotalCharges": 720.50,
    ...
  },
  {
    "tenure": 24,
    "MonthlyCharges": 89.50,
    "TotalCharges": 2148.00,
    ...
  }
]
```

**Success Response (200 OK)**:
```json
[
  {
    "prediction": "No",
    "probability": 0.9431,
    "model_version": "v1"
  },
  {
    "prediction": "Yes",
    "probability": 0.7823,
    "model_version": "v1"
  }
]
```

**Example**:
```sh
curl -X POST http://127.0.0.1:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d '[
    {"tenure": 12, "MonthlyCharges": 59.95, ...},
    {"tenure": 24, "MonthlyCharges": 89.50, ...}
  ]'
```

#### Special Cases

**Empty List**: Returns empty list with 200 status
```json
Request: []
Response: []
```

**Missing or Null TotalCharges**: Automatically imputed to 0.0
```json
{
  "tenure": 12,
  "MonthlyCharges": 59.95,
  "TotalCharges": null,  // or omit this field entirely
  ...
}
```

---

## Error Responses

### 400 Bad Request

Returned when input validation fails.

**Missing Required Features**:
```json
{
  "error": "Missing required features: tenure, MonthlyCharges"
}
```

**Invalid Data Type**:
```json
{
  "error": "Invalid type for tenure: expected int or float, got str"
}
```

**No Input Data**:
```json
{
  "error": "No input data provided"
}
```

### 500 Internal Server Error

Returned when server-side processing fails.

**Models Not Loaded**:
```json
{
  "error": "Models or pipelines are not loaded. Check server logs."
}
```

**Prediction Error**:
```json
{
  "error": "An error occurred during prediction: [error details]"
}
```

---

## Required Input Features

All prediction requests must include these 19 features:

### Numerical Features (3):
- `tenure` - Number of months the customer has been with the company (integer)
- `MonthlyCharges` - Current monthly charge amount (float)
- `TotalCharges` - Total amount charged to date (float, can be null/missing)

### Categorical Features (16):
- `gender` - "Male" or "Female"
- `SeniorCitizen` - "Yes" or "No" (string format required)
- `Partner` - "Yes" or "No"
- `Dependents` - "Yes" or "No"
- `PhoneService` - "Yes" or "No"
- `MultipleLines` - "Yes", "No", or "No phone service"
- `InternetService` - "DSL", "Fiber optic", or "No"
- `OnlineSecurity` - "Yes", "No", or "No internet service"
- `OnlineBackup` - "Yes", "No", or "No internet service"
- `DeviceProtection` - "Yes", "No", or "No internet service"
- `TechSupport` - "Yes", "No", or "No internet service"
- `StreamingTV` - "Yes", "No", or "No internet service"
- `StreamingMovies` - "Yes", "No", or "No internet service"
- `Contract` - "Month-to-month", "One year", or "Two year"
- `PaperlessBilling` - "Yes" or "No"
- `PaymentMethod` - "Electronic check", "Mailed check", "Bank transfer", or "Credit card"

**Note**: All categorical features must be strings. SeniorCitizen should be "Yes" or "No", not 0/1.

---

## Interactive Documentation

**`GET /apidocs/`**

Provides a full, interactive "Swagger UI" for the API. You can:
- See all endpoints and their schemas
- View detailed request/response models
- Test endpoints directly from your browser
- Download OpenAPI specification

**Access**: Navigate to `http://127.0.0.1:5000/apidocs/` in your browser after starting the server.

---

## Testing the API

### Using Python

```python
import requests

# Single prediction
response = requests.post(
    "http://127.0.0.1:5000/v1/predict",
    json={
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
)

print(response.json())
# Output: {"prediction": "No", "probability": 0.9431, "model_version": "v1"}
```

### Using curl

```sh
# Health check
curl http://127.0.0.1:5000/health

# Get API info
curl http://127.0.0.1:5000/cmpt2500f25_tutorial_home

# Single prediction
curl -X POST http://127.0.0.1:5000/v1/predict \
  -H "Content-Type: application/json" \
  -d @sample_customer.json
```

### Using Postman or other API clients

1. Set request type to `POST`
2. Enter URL: `http://127.0.0.1:5000/v1/predict`
3. Set Headers: `Content-Type: application/json`
4. Add JSON body with all required features
5. Send request

---

## Troubleshooting

### "Models or pipelines are not loaded"

**Cause**: Model files not found in `models/` or `data/processed/` directories.

**Solution**:
```sh
dvc pull data/processed
dvc pull models
```

### "Missing required features" error

**Cause**: Request missing one or more of the 19 required features.

**Solution**: Check the error message for which features are missing and include them in your request.

### "Invalid type" error

**Cause**: Feature has wrong data type (e.g., string instead of number).

**Solution**:
- Numerical features must be numbers: `"tenure": 12` not `"tenure": "12"`
- Categorical features must be strings: `"SeniorCitizen": "No"` not `"SeniorCitizen": 0`

### Connection refused

**Cause**: Server not running.

**Solution**:
```sh
python src/app.py
```

---

## Additional Resources

- **Interactive API Docs**: `http://127.0.0.1:5000/apidocs/`
- **Project Repository**: See README.md for more information
- **Lab Assignment**: `assignments/Lab 03 - REST API Development.md`
