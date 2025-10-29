# CMPT 2500 Project: Telecom Customer Churn Prediction

Am organized machine learning project demonstrating a machine learning project to predict customer churn in the telecommunications industry. This project uses various classification algorithms to identify customers at risk of leaving the service.

## Project Overview

Customer churn prediction helps telecom companies identify customers who are likely to discontinue their services. By predicting churn, companies can take proactive measures to retain customers, reducing the cost of acquiring new customers (which is typically 5 times more expensive than retaining existing ones).

## Project Structure

```output
cmpt2500f25-project-tutorial/
├── data/
│   ├── raw/                    # Original, immutable data
│   └── processed/              # Cleaned, processed data
├── configs/
│   ├── preprocess_config.yaml # Configuration for preprocessing
│   └── train_config.yaml      # Configuration for training
├── src/
│   ├── __init__.py
│   ├── preprocess.py          # Data loading & preprocessing
│   ├── feature_engineering.py # Feature creation & selection
│   ├── train.py               # Model training pipeline
│   ├── predict.py             # Prediction functions
│   ├── evaluate.py            # Model evaluation metrics
│   └── utils/
│       ├── __init__.py
│       └── config.py          # Configuration constants
├── models/                     # Saved model files
├── notebooks/                  # Jupyter notebooks for exploration
│   └── proof_of_concept.ipynb
├── outputs/                    # Plots, reports, results
├── tests/                      # Unit tests
├── .gitignore
├── Makefile
├── requirements.txt           # Python dependencies
└── README.md
```

## Features

The dataset includes the following features:

- **Demographics**: Gender, Senior Citizen status, Partner, Dependents
- **Services**: Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies
- **Account Information**: Tenure, Contract type, Payment method, Paperless billing, Monthly charges, Total charges
- **Target**: Churn (Yes/No)

## Models Implemented

1. **Logistic Regression** - Baseline linear model
2. **Random Forest Classifier** - Ensemble of decision trees
3. **Decision Tree Classifier** - Single decision tree
4. **AdaBoost Classifier** - Adaptive boosting ensemble
5. **Gradient Boosting Classifier** - Gradient boosting ensemble
6. **Voting Classifier** - Ensemble combining multiple models

## Installation

1. Clone the repository:

```bash
git clone https://github.com/NorQuest-MLAD-Courses/cmpt2500f25-project-tutorial.git
cd cmpt2500f25-project-tutorial
```

2. You can use the Makefile to set up the project. Running `make` will automatically create and activate the `.venv` virtual environment for you:

```bash
make
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

```python
from src.preprocess import preprocess_pipeline

# Run complete preprocessing pipeline
X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
    'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv',
    scale=True
)
```

### 2. Train Models

```python
from src.train import train_all_models, save_all_models

# Train all models
models = train_all_models(X_train, y_train)

# Save models
saved_paths = save_all_models(models)
```

### 3. Make Predictions

```python
from src.predict import load_model, predict

# Load trained model
model = load_model('models/voting_classifier_<timestamp>.pkl')

# Make predictions
predictions = predict(model, X_test)
```

### 4. Evaluate Models

```python
from src.evaluate import evaluate_model, compare_models

# Evaluate single model
results = evaluate_model(model, X_test, y_test)

# Compare multiple models
comparison_df = compare_models(models, X_test, y_test)
print(comparison_df)
```

## Model Performance

The Voting Classifier (ensemble) typically achieves the best performance:

- **Accuracy**: ~80%
- **Precision**: ~65-70%
- **Recall**: ~55-60%
- **F1-Score**: ~60-65%

(Note: Actual performance may vary based on data and hyperparameters)

## Key Insights

Based on the analysis:

1. **Contract Type**: Month-to-month contracts have higher churn rates
2. **Tenure**: Customers with shorter tenure are more likely to churn
3. **Internet Service**: Fiber optic customers show different churn patterns
4. **Charges**: Higher monthly charges correlate with increased churn risk
5. **Services**: Lack of online security and tech support increases churn

## Future Improvements

- Hyperparameter tuning using GridSearchCV/RandomizedSearchCV
- Cross-validation for more robust evaluation
- SHAP values for model interpretability
- Feature engineering for better predictive power
- API deployment for real-time predictions
- Docker containerization
- CI/CD pipeline setup

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.

## Acknowledgments

- Dataset source: [Telecom Customer Churn Dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
- This project is part of an ML deployment learning series
