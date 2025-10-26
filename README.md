# CMPT 2500 Project Tutorial: Telecom Customer Churn Prediction

A production-ready machine learning project to predict customer churn in the telecommunications industry. This project demonstrates industry best practices including modular code organization, CLI interfaces, hyperparameter tuning, experiment tracking, and data versioning.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![Code style: PEP 8](https://img.shields.io/badge/code%20style-PEP%208-black.svg)](https://www.python.org/dev/peps/pep-0008/)

## Overview

Customer churn prediction helps telecom companies identify customers who are likely to discontinue their services. By predicting churn, companies can take proactive measures to retain customers, reducing the cost of acquiring new customers (which is typically 5 times more expensive than retaining existing ones).

This project showcases a complete ML workflow from data preprocessing to model deployment, incorporating:

- 🏗️ **Modular architecture** with separation of concerns
- 🖥️ **CLI interfaces** for all major operations
- 🔧 **Hyperparameter tuning** for optimal model performance
- 📝 **YAML configuration** for flexible deployment
- 🧪 **Automated testing** with pytest
- 📦 **Virtual environments** for reproducibility

## Project Structure

```output
cmpt2500f25-project-tutorial/
├── .venv/                      # Virtual environment (not in Git)
├── configs/                    # YAML configuration files
│   ├── train_config.yaml
│   ├── preprocess_config.yaml
│   └── predict_config.yaml
├── data/
│   ├── raw/                    # Original data
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/              # Processed data & pipelines
│       ├── preprocessed_data.npy
│       ├── preprocessing_pipeline.pkl
│       └── label_encoder.pkl
├── models/                     # Trained models
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── proof_of_concept.ipynb
│   ├── eda.ipynb
│   └── model_experimentation.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py          # Data preprocessing with sklearn pipelines
│   ├── train.py               # Model training with hyperparameter tuning
│   ├── predict.py             # Prediction with CLI
│   ├── evaluate.py            # Model evaluation
│   ├── feature_engineering.py # Feature creation & selection
│   └── utils/
│       ├── __init__.py
│       └── config.py          # Configuration constants
├── tests/                      # Unit tests
│   ├── test_preprocess.py
│   ├── test_train.py
│   └── test_predict.py
├── outputs/                    # Plots, reports, results
├── experiments/                # Experiment notes
├── .gitignore
├── requirements.txt            # Python dependencies
└── README.md
```

## Dataset Features

The dataset includes the following features:

**Demographics**:

- Gender, Senior Citizen status, Partner, Dependents

**Services**:

- Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies

**Account Information**:

- Tenure, Contract type, Payment method, Paperless billing, Monthly charges, Total charges

**Target**:

- Churn (Yes/No) - Binary classification

## Models Implemented

1. **Logistic Regression** - Baseline linear model
2. **Random Forest Classifier** - Ensemble of decision trees
3. **Decision Tree Classifier** - Single decision tree
4. **AdaBoost Classifier** - Adaptive boosting ensemble
5. **Gradient Boosting Classifier** - Gradient boosting ensemble
6. **Voting Classifier** - Ensemble combining multiple models

All models support:

- ⚙️ **Hyperparameter tuning** with GridSearchCV
- 📊 **Cross-validation** for robust evaluation
- 💾 **Model persistence** with joblib
- 📈 **Experiment tracking** with MLflow

## Installation

### Prerequisites

- Python 3.12.x (recommended)
- Git
- pip

**Note**: Python 3.13 is available but some packages may not be fully compatible. Stick with Python 3.12.x for maximum compatibility.

1. **Clone the repository**:

   ```bash
   git clone https://github.com/NorQuest-MLAD-Courses/cmpt2500f25-project-tutorial.git
   cd cmpt2500f25-project-tutorial
   ```

2. **Create virtual environment**:

   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate it
   source .venv/bin/activate  # Mac/Linux
   # OR
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation**:

   ```bash
   python -c "import sklearn; import mlflow; import dvc; print('✅ All packages installed!')"
   ```

## Usage

### Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Preprocess data (creates sklearn pipeline)
python -m src.preprocess --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 3. Train model (with hyperparameter tuning)
python -m src.train --data data/processed/preprocessed_data.npy --model random_forest --tune

# 4. Make predictions
python -m src.predict --model models/random_forest_*.pkl --data data/processed/preprocessed_data.npy

# 5. Evaluate model
python -m src.evaluate --model models/random_forest_*.pkl --data data/processed/preprocessed_data.npy
```

### Data Preprocessing

The preprocessing module uses **scikit-learn pipelines** for reproducible preprocessing:

```bash
# Basic preprocessing
python -m src.preprocess \
    --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
    --output-dir data/processed

# Use legacy approach (not recommended)
python -m src.preprocess \
    --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
    --legacy

# Get help
python -m src.preprocess --help
```

**What it does**:

- Loads raw data from CSV
- Handles missing values
- Encodes categorical features
- Scales numerical features
- Splits into train/test sets
- Saves preprocessing pipeline for consistent predictions

**Output files**:

- `preprocessed_data.npy` - Train/test data
- `preprocessing_pipeline.pkl` - Sklearn pipeline
- `label_encoder.pkl` - Target encoder

### Model Training

```bash
# Train single model (fast, default hyperparameters)
python -m src.train \
    --data data/processed/preprocessed_data.npy \
    --model random_forest

# Train with hyperparameter tuning (slower, better performance)
python -m src.train \
    --data data/processed/preprocessed_data.npy \
    --model random_forest \
    --tune

# Train all models
python -m src.train \
    --data data/processed/preprocessed_data.npy \
    --model all

# Train all with tuning (will take time!)
python -m src.train \
    --data data/processed/preprocessed_data.npy \
    --model all \
    --tune

# Get help
python -m src.train --help
```

**Available models**:

- `logistic_regression`
- `random_forest`
- `decision_tree`
- `adaboost`
- `gradient_boosting`
- `voting_classifier`
- `all` (trains all models)

**Hyperparameter Tuning**:
When `--tune` flag is used, GridSearchCV performs exhaustive search over parameter grid with 5-fold cross-validation. This significantly improves model performance but takes longer.

### Making Predictions

```bash
# Predict with trained model
python -m src.predict \
    --model models/random_forest_20241027_143530.pkl \
    --data data/processed/preprocessed_data.npy

# Get probability predictions
python -m src.predict \
    --model models/random_forest_20241027_143530.pkl \
    --data data/processed/preprocessed_data.npy \
    --proba

# Save predictions to file
python -m src.predict \
    --model models/random_forest_20241027_143530.pkl \
    --data data/processed/preprocessed_data.npy \
    --output predictions/predictions.npy

# Predict with preprocessing pipeline
python -m src.predict \
    --model models/model.pkl \
    --data data/raw/new_data.csv \
    --pipeline data/processed/preprocessing_pipeline.pkl

# Get help
python -m src.predict --help
```

### Model Evaluation

```bash
# Evaluate model
python -m src.evaluate \
    --model models/random_forest_20241027_143530.pkl \
    --data data/processed/preprocessed_data.npy \
    --model-name "Random Forest"

# Save evaluation results
python -m src.evaluate \
    --model models/random_forest_20241027_143530.pkl \
    --data data/processed/preprocessed_data.npy \
    --output evaluation_results.json

# Get help
python -m src.evaluate --help
```

**Metrics calculated**:

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (if model supports probabilities)
- Confusion Matrix
- Classification Report

## Configuration Management

The project uses YAML files for flexible configuration management.

### Configuration Files

**`configs/train_config.yaml`** - Training settings:

```yaml
model:
  type: random_forest
  params:
    n_estimators: 100
    max_depth: 10

training:
  test_size: 0.2
  tune: false
```

**`configs/preprocess_config.yaml`** - Preprocessing settings:

```yaml
data:
  filename: WA_Fn-UseC_-Telco-Customer-Churn.csv
  target_column: Churn

scaling:
  method: standard
  
features:
  categorical: [gender, Contract, PaymentMethod]
  numerical: [tenure, MonthlyCharges, TotalCharges]
```

### Using Configurations

```python
import yaml

# Load configuration
with open('configs/train_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access settings
model_type = config['model']['type']
n_estimators = config['model']['params']['n_estimators']
```

## Model Performance

Typical performance metrics (with hyperparameter tuning):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Voting Classifier | 80.5% | 67.2% | 56.8% | 61.5% | 84.9% |
| Random Forest | 79.3% | 64.0% | 53.9% | 58.5% | 84.6% |
| Gradient Boosting | 80.1% | 66.5% | 55.2% | 60.3% | 84.2% |
| Logistic Regression | 79.8% | 65.8% | 54.1% | 59.4% | 84.1% |
| AdaBoost | 79.7% | 65.3% | 54.8% | 59.6% | 83.8% |
| Decision Tree | 73.5% | 53.2% | 48.9% | 50.9% | 71.2% |

**Note**: Performance varies based on data splits and hyperparameter settings.

## Key Insights

Based on the analysis of 7,043 telecom customers:

1. **Contract Type Impact**:
   - Month-to-month contracts show 42% churn rate
   - One-year contracts: 11% churn rate
   - Two-year contracts: 3% churn rate
   - **Action**: Incentivize longer contract commitments

2. **Tenure Correlation**:
   - Customers with <12 months tenure: 47% churn
   - Customers with >48 months tenure: 7% churn
   - **Action**: Focus retention efforts on new customers (first year)

3. **Service Bundle Effect**:
   - Customers with phone + internet + protection services: 25% churn
   - Customers with phone only: 35% churn
   - **Action**: Promote service bundles for retention

4. **Monthly Charges**:
   - Charges >$70/month associated with 33% churn
   - Charges <$30/month associated with 15% churn
   - **Action**: Review pricing strategy for high-value customers

5. **Payment Method**:
   - Electronic check users: 45% churn
   - Credit card / bank transfer: 15-18% churn
   - **Action**: Encourage automatic payment methods

## Technology Stack

**Core ML**:

- Python 3.12
- scikit-learn 1.7.2
- NumPy 2.3.4
- Pandas 2.3.3

**Advanced Models**:

- XGBoost 3.1.1
- CatBoost 1.2.8

**Visualization**:

- Matplotlib 3.10.7
- Seaborn 0.13.2
- Plotly 6.3.1

**Configuration**:

- PyYAML 6.0.3

**Testing**:

- pytest 8.4.2
- pytest-cov 7.0.0

**Development**:

- Jupyter 1.1.1

**Coming Soon (Lab 02 - Part 2)**:

- DVC 3.63.0 (data versioning)
- MLflow 3.5.1 (experiment tracking)

## Development Workflow

### 1. Experimentation Phase

```bash
# Use notebooks for exploration
jupyter notebook notebooks/

# Prototype in notebooks:
# - EDA (eda.ipynb)
# - Feature engineering (model_experimentation.ipynb)
# - Model selection (proof_of_concept.ipynb)
```

### 2. Development Phase

```bash
# Convert notebook code to modules
# - Extract preprocessing → src/preprocess.py
# - Extract training → src/train.py
# - Extract evaluation → src/evaluate.py

# Test modules individually
python -m src.preprocess --input data/raw/data.csv
python -m src.train --data data/processed/preprocessed_data.npy --model random_forest
```

### 3. Optimization Phase

```bash
# Enable hyperparameter tuning
python -m src.train --data data/processed/preprocessed_data.npy --model all --tune

# Compare results
python -m src.evaluate --model models/model1.pkl --data data/processed/preprocessed_data.npy
python -m src.evaluate --model models/model2.pkl --data data/processed/preprocessed_data.npy
```

### 4. Testing Phase

```bash
# Run unit tests
pytest tests/

# Check coverage
pytest --cov=src tests/
```

### 5. Version Control

```bash
# Commit changes
git add .
git commit -m "feat: Add optimized models with tuning"
git push
```

## Test

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_preprocess.py

# Run with verbose output
pytest -v tests/
```

## Project Roadmap

### ✅ Completed (Lab 01)

- Modular code organization
- Data preprocessing functions
- Model training functions
- Evaluation metrics
- Basic documentation

### ✅ Completed (Lab 02)

- Virtual environment setup
- Dependency management
- CLI interfaces (argparse)
- Hyperparameter tuning
- Scikit-learn pipelines
- YAML configuration
- Unit testing basics

### 🔜 Upcoming (Lab 02 - Part 2)

- DVC setup with DagsHub
- MLflow integration
- Comprehensive pytest suite

### 🔜 Upcoming (Lab 03)

- REST API with Flask/FastAPI
- API documentation (Swagger)
- Request/response validation
- API testing

### 🔜 Future (Lab 04-06)

- Docker containerization
- Cloud deployment (AWS/GCP/Azure)
- CI/CD pipeline
- Model monitoring
- A/B testing
- Model retraining automation

## Contributing

### Code Style

- Follow PEP 8 style guide
- Use type hints for all functions
- Write docstrings (Google style)
- Organize imports (standard → third-party → local)
- Use logging instead of print statements

### Testing

- Write tests for new features
- Maintain >80% code coverage
- Test edge cases
- Include integration tests

### Documentation

- Update README for new features
- Add docstrings to all functions
- Keep YAML configs in sync
- Document breaking changes

## Troubleshooting

### Virtual Environment Issues

**Problem**: `command not found: python`

```bash
# Solution: Activate virtual environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

**Problem**: Package import errors

```bash
# Solution: Reinstall requirements
pip install --force-reinstall -r requirements.txt
```

### Module Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

```bash
# Solution: Run from project root with -m flag
cd /path/to/project
python -m src.train --help
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: Telecom Customer Churn Dataset (IBM Sample Data)
- Course: CMPT 2500 - ML/AI Deployment, NorQuest College
- Instructor: [Your Name]

## Contact

For questions, issues, or contributions:

- Create an issue on GitHub
- Email: [your-email]
- Office Hours: [schedule]

---

**Last Updated**: October 2024  
**Version**: 2.0.0 (Lab 02 Complete)  
**Python Version**: 3.12.12
