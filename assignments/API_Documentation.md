# API Documentation: CMPT 2500 Tutorial Project - Telecom Churn Prediction

## Overview

This API serves a machine learning model to predict customer churn. It exposes endpoints to check API health, get usage information, and receive predictions from two different model versions (v1 and v2).

This document provides instructions for setup and a high-level overview of the endpoints. For a detailed, interactive API reference, run the server and navigate to `/apidocs/`.

---

## Installation & Running

### 1. Setup Environment

Clone the repository and install the required Python packages.

```sh
git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/cmpt2500f25-project-tutorial.git
cd cmpt2500f25-project-tutorial
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get Data & Artifacts

This project uses DVC to manage large data files and pipelines. You must set up your DagsHub credentials and pull the data.

(Follow the credential setup in `assignments/Lab 03 - REST API Development.md` if this is a new environment).

```sh
dvc pull data/processed
```

This will download `preprocessing_pipeline.pkl` and `label_encoder.pkl`.

### 3. Run the API Server

Ensure your `models/model_v1.pkl` and `models/model_v2.pkl` files are in place. Then, run the app:

```sh
python src/app.py
```

The server will start on `http://127.0.0.1:5000`.

---

## Endpoints

`GET /health`

- **Purpose**: A simple health check.
- **Success Response (200 OK)**:

```json
{
  "message": "Welcome to the CMPT 2500 F25 Project Tutorial API!",
  "api_documentation": "Find the interactive documentation at /apidocs/",
  ...
}
```

`POST /v1/predict` (and `POST /v2/predict`)

- **Purpose**: Generates a prediction from Model v1 (or v2). Accepts a single JSON object or a list of objects for batch prediction.
- **Request Body (Example for one customer)**:

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

- **Success Response (200 OK)**:

```json
{
  "prediction": "No",
  "probability": 0.9431,
  "model_version": "v1"
}
```

- **Error Response (400 Bad Request)**:

```json
{"error": "Missing required features: tenure, ..."}
```

`GET /apidocs/`

- **Purpose**: Provides a full, interactive "Swagger UI" for the API. You can see all endpoints, data models, and test them live from your browser.
