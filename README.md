# CMPT 2500 Project: Telecom Customer Churn Prediction

A production-ready machine learning project demonstrating industry best practices for MLOps, including modular code organization, data version control (DVC), experiment tracking (MLflow), and automated workflows.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![DVC](https://img.shields.io/badge/DVC-3.63.0-945DD6.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.5.1-0194E2.svg)](https://mlflow.org/)

## Overview

Customer churn prediction for telecommunications companies. By predicting which customers are likely to leave, companies can take proactive retention measures—reducing costs since customer acquisition is typically 5× more expensive than retention.

This project demonstrates a complete ML workflow from data preprocessing to model deployment, incorporating:

- 🏗️ **Modular architecture** with separation of concerns
- 🖥️ **CLI interfaces** for all operations
- 🔧 **Hyperparameter tuning** for optimal performance
- 📝 **YAML configuration** for flexible deployment
- 📦 **Data version control (DVC)** with DagsHub remote
- 📊 **Experiment tracking (MLflow)** with comprehensive logging
- 🧪 **Automated testing** with pytest
- 🐍 **Virtual environments** for reproducibility

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/your-username/telecom-churn-prediction.git
cd telecom-churn-prediction

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Authenticate with DagsHub (for data access)
dagshub login
# Select token timeframe: "2 months"

# 5. Pull data from DVC remote
dvc pull

# 6. Preprocess data
python -m src.preprocess --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 7. Train model (with MLflow tracking)
python -m src.train --data data/processed/preprocessed_data.npy --model random_forest --tune

# 8. View experiments in MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
# In CodeSpaces: PORTS tab → Forward port 5000 → Open in Browser

# 9. Make predictions
python -m src.predict --model models/random_forest_*.pkl --data data/processed/preprocessed_data.npy

# 10. Evaluate model
python -m src.evaluate --model models/random_forest_*.pkl --data data/processed/preprocessed_data.npy
```

---

## Project Structure

```output
telecom-churn-prediction/
├── .venv/                      # Virtual environment (not in Git)
├── .dvc/                       # DVC configuration
│   ├── config                  # DVC remote config (in Git)
│   └── config.local            # Credentials (NOT in Git)
├── mlruns/                     # MLflow tracking data (NOT in Git)
│   └── 0/                      # Experiment ID
│       └── <run_id>/           # Individual runs with metrics/params/artifacts
├── configs/                    # YAML configuration files
│   ├── train_config.yaml       # Training configuration & hyperparameters
│   └── preprocess_config.yaml  # Data preprocessing configuration
├── data/
│   ├── .gitignore              # DVC-generated (ignores actual data)
│   ├── raw.dvc                 # DVC metadata for raw data (in Git)
│   ├── processed.dvc           # DVC metadata for processed data (in Git)
│   ├── raw/                    # Actual data (tracked by DVC, not Git)
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/              # Processed data (tracked by DVC, not Git)
│       ├── preprocessed_data.npy
│       ├── preprocessing_pipeline.pkl
│       └── label_encoder.pkl
├── models/                     # Trained models (tracked by MLflow)
├── notebooks/                  # Jupyter notebooks for exploration
│   └── proof_of_concept.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # Data preprocessing with sklearn pipelines
│   ├── train.py                # Model training with MLflow tracking ⭐
│   ├── predict.py              # Prediction CLI
│   ├── evaluate.py             # Model evaluation
│   └── utils/
│       ├── __init__.py
│       └── config.py           # Configuration constants
├── tests/                      # Unit tests
│   ├── test_preprocess.py
│   ├── test_train.py
│   └── test_predict.py
├── .gitignore                  # Git ignore (includes mlruns/, .dvc/cache/)
├── .dvcignore                  # DVC ignore patterns
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Technology Stack

**Core ML**:

- Python 3.12
- scikit-learn 1.7.2
- NumPy 2.3.4
- Pandas 2.3.3

**Data Version Control**:

- DVC 3.63.0
- DagsHub 0.6.3 (remote storage)

**Experiment Tracking**:

- MLflow 3.5.1 (tracking, models, artifacts)

**Visualization**:

- Matplotlib 3.10.7
- Seaborn 0.13.2

**Configuration**:

- PyYAML 6.0.3

**Testing**:

- pytest 8.4.2

---

## Data Version Control (DVC)

This project uses **DVC** to track data and models, with **DagsHub** as the remote storage.

**DVC Remote**: [https://dagshub.com/your-username/telecom-churn-prediction](https://dagshub.com/your-username/telecom-churn-prediction)

### Setup DVC

```bash
# Install DVC with S3 support
pip install dvc dvc-s3 dagshub

# Authenticate with DagsHub
dagshub login
# Browser opens → Sign in → Select token timeframe → Copy token
# Paste token in terminal

# Configure DVC remote (already done in repo)
dvc remote default origin
dvc remote modify origin url s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/your-username/repo.s3

# Pull data
dvc pull
```

### DVC Workflow

```bash
# After modifying data
dvc add data/raw
dvc add data/processed

# Commit DVC metadata to Git
git add data/raw.dvc data/processed.dvc .gitignore
git commit -m "Update data version"

# Push data to DagsHub
dvc push

# Push code to GitHub
git push
```

**Note**: In CodeSpaces, run `dagshub login` each time you start a new instance (tokens are stored locally).

---

## Experiment Tracking (MLflow)

This project uses **MLflow** for tracking experiments, comparing models, and managing the ML lifecycle.

### MLflow Workflow

**Train with tracking**:

```bash
# Single model
python -m src.train \
    --data data/processed/preprocessed_data.npy \
    --model random_forest

# With hyperparameter tuning
python -m src.train \
    --data data/processed/preprocessed_data.npy \
    --model random_forest \
    --tune

# All models for comparison
python -m src.train \
    --data data/processed/preprocessed_data.npy \
    --model all \
    --tune
```

**View experiments in UI**:

```bash
# Start MLflow UI (CodeSpaces: must use 0.0.0.0)
mlflow ui --host 0.0.0.0 --port 5000

# Access via forwarded port:
# 1. Open PORTS tab (bottom panel in VS Code)
# 2. Port 5000 should auto-forward
# 3. Right-click port 5000 → "Open in Browser"
```

### What MLflow Tracks

| Item | Tracked? | Description |
|------|----------|-------------|
| **Parameters** | ✅ | Model hyperparameters (n_estimators, max_depth, etc.) |
| **Metrics** | ✅ | accuracy, precision, recall, F1-score, ROC-AUC |
| **Artifacts** | ✅ | Trained models (.pkl files) |
| **Training Time** | ✅ | Duration in seconds |
| **Data Version** | ⚠️ Manual | Tag with DVC version hash |
| **Source Code** | ✅ | Git commit hash automatically tracked |
| **Environment** | ✅ | Python version, package versions |

### Load Best Model

```python
import mlflow.sklearn

# Load by run ID (from MLflow UI)
model = mlflow.sklearn.load_model("runs:/abc123.../model")

# Or search for best run
runs = mlflow.search_runs(
    order_by=["metrics.accuracy DESC"],
    max_results=1
)
best_run_id = runs.iloc[0]['run_id']
model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
```

### MLflow in CodeSpaces

**Critical**: CodeSpaces runs on a remote VM. To access MLflow UI:

1. Start server: `mlflow ui --host 0.0.0.0 --port 5000`
2. Open PORTS tab (bottom panel)
3. Port 5000 should auto-forward
4. Right-click port 5000 → "Open in Browser"

---

## Configuration Files

Config files in `configs/` directory:

### train_config.yaml

- Model hyperparameters (for each model type)
- Training settings (test_size, random_state)
- MLflow configuration (experiment name, tracking URI)
- DVC settings (data versioning)
- Hyperparameter tuning grids

### preprocess_config.yaml

- Data paths and column names
- Feature scaling methods
- Missing value handling
- Train-test split settings
- Pipeline configuration

**Note**: Config files added in v2.2.0. For older versions, unused sections are ignored. Code works without configs (uses CLI args).

---

## Usage Examples

### 1. Data Preprocessing

```bash
# Basic preprocessing
python -m src.preprocess \
    --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

# With custom output directory
python -m src.preprocess \
    --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
    --output data/processed
```

### 2. Model Training

```bash
# Train Random Forest with default hyperparameters
python -m src.train \
    --data data/processed/preprocessed_data.npy \
    --model random_forest

# Train with hyperparameter tuning (GridSearchCV)
python -m src.train \
    --data data/processed/preprocessed_data.npy \
    --model random_forest \
    --tune

# Train all models
python -m src.train \
    --data data/processed/preprocessed_data.npy \
    --model all

# Train all models with tuning
python -m src.train \
    --data data/processed/preprocessed_data.npy \
    --model all \
    --tune

# Available models:
# - logistic_regression
# - random_forest
# - decision_tree
# - adaboost
# - gradient_boosting
# - voting_classifier
```

### 3. Making Predictions

```bash
# Using trained model
python -m src.predict \
    --model models/random_forest_20241027_103045.pkl \
    --data data/processed/preprocessed_data.npy

# With custom output
python -m src.predict \
    --model models/random_forest_*.pkl \
    --data data/processed/preprocessed_data.npy \
    --output predictions.csv
```

### 4. Model Evaluation

```bash
# Evaluate model
python -m src.evaluate \
    --model models/random_forest_20241027_103045.pkl \
    --data data/processed/preprocessed_data.npy

# Save evaluation report
python -m src.evaluate \
    --model models/random_forest_*.pkl \
    --data data/processed/preprocessed_data.npy \
    --output outputs/evaluation_report.txt
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_train.py

# Run with verbose output
pytest -v
```

---

## Common Issues & Solutions

### DVC Issues

**Problem**: `dvc push` fails with authentication error

```bash
# Solution: Re-authenticate
dagshub login
# Select "2 months" token timeframe
```

**Problem**: `dvc pull` fails

```bash
# Check credentials are configured
cat .dvc/config.local

# If missing, re-authenticate
dagshub login
```

**Problem**: Data files not found after `git clone`

```bash
# Pull data from DVC remote
dvc pull
```

### MLflow Issues

**Problem**: MLflow UI not accessible in CodeSpaces

```bash
# 1. Verify MLflow is running
mlflow ui --host 0.0.0.0 --port 5000

# 2. Check PORTS tab (bottom panel)
# - Port 5000 should be listed
# - If not, click "Forward a Port" → enter 5000

# 3. Right-click port 5000 → "Open in Browser"
```

**Problem**: Runs not showing in MLflow UI

```bash
# 1. Refresh browser (F5)
# 2. Check experiment name matches
# 3. Verify mlruns/ directory exists
ls mlruns/
```

**Problem**: Large mlruns/ directory

```bash
# Delete old experiments
mlflow experiments delete --experiment-id 1

# Or delete via UI (select runs → delete button)

# Note: mlruns/ is in .gitignore (won't be committed)
```

---

## Project Roadmap

### ✅ Completed (Lab 01)

- Modular code organization
- Data preprocessing functions
- Model training functions
- Evaluation metrics
- Basic documentation

### ✅ Completed (Lab 02 - Part 1)

- Virtual environment setup
- Dependency management
- CLI interfaces (argparse)
- Hyperparameter tuning
- Scikit-learn pipelines
- YAML configuration
- Unit testing basics
- DVC setup with DagsHub
- Data version control

### ✅ Completed (Lab 02 - Part 2)

- MLflow integration
- Experiment tracking
- Model comparison
- Configuration files

### 🔜 Upcoming (Lab 03)

- REST API development (Flask/FastAPI)
- API documentation (Swagger)
- Model serving
- API testing

### 🔜 Future Labs

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

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Acknowledgments

- **Dataset**: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) from Kaggle
- **MLflow**: Databricks for the open-source experiment tracking platform
- **DVC**: Iterative for data version control
- **DagsHub**: For providing free DVC remote storage and MLflow hosting

---

## Contact

**Instructor**: [Your Name]  
**Course**: CMPT 2500 - ML/AI Deployment  
**Institution**: NorQuest College

---

**Version**: 2.2.0 (Lab 02 Complete - DVC + MLflow)  
**Last Updated**: October 2024  
**Python**: 3.12.12  
**DVC**: 3.63.0  
**MLflow**: 3.5.1
