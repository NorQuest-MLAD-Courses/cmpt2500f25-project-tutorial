# CMPT 2500 Project: Telecom Customer Churn Prediction

A production-ready machine learning project demonstrating industry best practices for MLOps, including modular code organization, data version control (DVC), experiment tracking (MLflow), automated testing, and a REST API for model serving.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![DVC](https://img.shields.io/badge/DVC-3.63.0-945DD6.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.5.1-0194E2.svg)](https://mlflow.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Swagger](https://img.shields.io/badge/Swagger-85EA2D?style=flat&logo=swagger&logoColor=black)](https://swagger.io/)

## Overview

Customer churn prediction for telecommunications companies. By predicting which customers are likely to leave, companies can take proactive retention measures—reducing costs since customer acquisition is typically 5× more expensive than retention.

This project demonstrates a complete ML workflow from data preprocessing to model deployment, incorporating:

- 🏗️ **Modular architecture** with separation of concerns
- 🖥️ **CLI interfaces** for all operations
- 🔧 **Hyperparameter tuning** with configurable grids (comprehensive, quick, and test)
- 📝 **YAML configuration** for preprocessing, training, and hyperparameter grids
- 📦 **Data version control (DVC)** with DagsHub remote
- 📊 **Experiment tracking (MLflow)** with models stored as artifacts
- 🧪 **Automated testing (pytest)** with fast test configurations
- 🐍 **Virtual environments** for reproducibility
- 🐳 **REST API** for serving models (Flask & Flasgger)

---

## API Documentation

This project is served via a REST API. For complete details on installation, running, and all available endpoints, please see the **[API Documentation](./API_Documentation.md)**.

For a live, interactive API specification (Swagger UI), run the server and navigate to:
`http://127.0.0.1:5000/apidocs/`

---

## Quick Start

### 1. Install Dependencies

Clone the repository and set up your virtual environment.

```sh
git clone https://github.com/ajallooe/cmpt2500f25-project-tutorial.git
cd cmpt2500f25-project-tutorial
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. How to setup DVC

This project uses DVC to manage large data files and ML models. You must pull the data from the remote storage.

**(One-Time Setup)**: If this is a new environment (like a fresh Codespace), you must first add your DagsHub credentials:

```sh
dvc remote modify origin --local access_key_id <YOUR_DVC_ACCESS_KEY_ID>
dvc remote modify origin --local secret_access_key <YOUR_DVC_SECRET_ACCESS_KEY>
```

**Pull Data**:

```sh
dvc pull
```

This will download `preprocessing_pipeline.pkl`, `label_encoder.pkl`, and `preprocessed_data.npy` from the DagsHub remote.

### 3. How to run preprocessing

The preprocessing pipeline can be run with the CLI, using configuration files for flexible parameter management.

```sh
# Use default config (configs/preprocess_config.yaml)
python -m src.preprocess --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --output-dir data/processed

# Use custom config
python -m src.preprocess --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --config configs/custom_preprocess.yaml

# Legacy mode (hardcoded defaults, no config file)
python -m src.preprocess --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --legacy
```

The preprocessing pipeline uses **`configs/preprocess_config.yaml`** for all settings including feature lists, split ratios, and validation options.

### 4. How to run training

You can run the full training pipeline using the CLI. This will train all models, perform hyperparameter tuning, and track every run in MLflow.

**Important**: Models are saved to MLflow by default (best practice). Use `--save` to also save locally.

#### Training Examples

```sh
# Train all models with comprehensive hyperparameter tuning (6-20 hours)
python -m src.train --data data/processed/preprocessed_data.npy --model all --tune

# Train all models with quick tuning config (10-30 minutes)
python -m src.train --data data/processed/preprocessed_data.npy --model all --tune --config configs/quick_tune_config.yaml

# Train a single model without tuning (fast)
python -m src.train --data data/processed/preprocessed_data.npy --model random_forest

# Train with custom config and save to local disk
python -m src.train --data data/processed/preprocessed_data.npy --model all --tune --config configs/custom_config.yaml --save
```

#### Configuration Files

- **`configs/train_config.yaml`**: Comprehensive hyperparameter grids for production (~193k combinations)
- **`configs/quick_tune_config.yaml`**: Reduced grids for faster tuning (~1.3k combinations)
- **`configs/test_config.yaml`**: Minimal grids for automated tests (~48 combinations)

To view the results, run the MLflow UI:

```sh
mlflow ui
```

Then navigate to `http://127.0.0.1:5000` to view experiments, compare models, and access saved model artifacts.

### 5. How to run tests

This project uses `pytest` for automated testing. You can run all tests (including the new API tests) with a single command.

```sh
pytest
```

For more targeted testing:

```sh
# Run only fast tests (skip slow hyperparameter tuning tests)
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run with verbose output
pytest -v
```

### 6. How to run the API Server (Lab 03)

To serve your trained models, run the Flask API server.

```sh
python src/app.py
```

The server will start on `http://127.0.0.1:5000`.

**Note**: If port 5000 conflicts with other services (e.g., macOS AirPlay), use a different port:

```sh
PORT=5002 python src/app.py
```

You can test if it's running in a new terminal:

```sh
curl http://127.0.0.1:5000/health
```

**Expected Output:**

```json
{"status":"ok"}
```

---

## Project Structure

```output
cmpt2500f25-project-tutorial/
├── .dvc/                  # DVC metadata
├── .github/               # GitHub Actions (CI/CD) workflows (Future)
├── .venv/                 # Python virtual environment (Ignored)
├── assignments/           # Lab instructions and guides
├── configs/               # YAML configuration files
│   ├── preprocess_config.yaml  # Preprocessing settings (features, splits, etc.)
│   ├── train_config.yaml       # Comprehensive hyperparameter grids (production)
│   ├── quick_tune_config.yaml  # Reduced grids for faster tuning
│   └── test_config.yaml        # Minimal grids for automated tests
├── data/
│   ├── processed/         # DVC-tracked processed data and pipelines
│   └── raw/               # DVC-tracked raw data
├── models/                # Optional local model storage (use --save flag)
├── mlruns/                # MLflow experiment tracking and model artifacts (Ignored)
├── notebooks/             # Jupyter notebooks for exploration (EDA)
├── src/                   # Main source code
│   ├── utils/             # Utility functions and config
│   │   ├── __init__.py
│   │   └── config.py      # Project constants and feature lists
│   ├── __init__.py
│   ├── app.py             # Lab 03: Flask API server
│   ├── evaluate.py        # CLI script for model evaluation
│   ├── predict.py         # CLI script for making predictions
│   ├── preprocess.py      # CLI script for data preprocessing
│   └── train.py           # CLI script for model training
├── tests/                 # Automated tests
│   ├── __init__.py
│   ├── conftest.py        # Pytest fixtures
│   ├── test_api.py        # Lab 03: Automated tests for the Flask API
│   ├── test_evaluate.py   # Unit tests for evaluate.py
│   ├── test_integration.py# Integration tests for full workflows
│   ├── test_predict.py    # Unit tests for predict.py
│   ├── test_preprocess.py # Unit tests for preprocess.py
│   └── test_train.py      # Unit tests for train.py
├── .dvcignore             # DVC ignore file
├── .gitignore             # Git ignore file
├── API_Documentation.md   # Lab 03: High-level API manual
├── Makefile               # (Optional) Helper commands
├── README.md              # This file
├── pytest.ini             # Pytest configuration
└── requirements.txt       # Project dependencies
```

---

## Project Status

### ✅ Completed (Lab 01)

- Modular project structure
- OOP principles applied
- Initial scripts (`preprocess.py`, `train.py`, etc.)
- `requirements.txt` and `.gitignore`

### ✅ Completed (Lab 02)

- Virtual environment setup
- Dependency management (`pip install -r requirements.txt`)
- CLI interfaces (`argparse`)
- Hyperparameter tuning (`GridSearchCV`)
- Scikit-learn pipelines (`pipeline.pkl`)
- YAML configuration files for preprocessing and training
- Configuration loading with `--config` CLI argument
- Flexible hyperparameter grids (comprehensive, quick, and test configs)
- Unit testing setup (`pytest`)
- DVC setup with DagsHub (`dvc pull`)
- MLflow integration (`mlflow ui`)
- MLflow best practices (models saved to MLflow by default)

### ✅ Completed (Lab 03)

- REST API development (`src/app.py` with Flask)
- API documentation (`flasgger` for `/apidocs/`)
- Manual API documentation (`API_Documentation.md`)
- Model serving (v1 and v2 endpoints)
- API testing (`tests/test_api.py`)

### 🔜 Upcoming (Future Labs)

- Containerization (Docker)
- Cloud deployment (AWS/GCP/Azure)
- CI/CD pipelines (GitHub Actions)
- Monitoring and logging

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Acknowledgments

- **Dataset**: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) from Kaggle
- **MLflow**: Databricks for the open-source experiment tracking platform
- **DVC**: Iterative for data version control
- **DagsHub**: For providing free DVC remote storage and MLflow hosting

---

## Contact

**Instructor**: Mohammad M. Ajallooeian
**Course**: CMPT 2500 - Machine Learning Deployment and Software Development
**Institution**: NorQuest College

---

**Version**: 4.0.0 (Configuration Architecture & Test Optimization)
**Last Updated**: 2025-11-06

### Recent Updates (v4.0.0)

- Refactored to YAML-based configuration for preprocessing and training
- Added flexible hyperparameter grid configs (comprehensive, quick, test)
- Optimized tests with minimal grids (11s vs hours)
- Implemented MLflow best practices (models in MLflow by default)
- Added `--config` CLI argument to all training and preprocessing commands
