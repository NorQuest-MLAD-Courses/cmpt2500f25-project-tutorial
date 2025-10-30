"""
This module contains constants used throughout the project.
"""

# Standard library imports
import os

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- 1. Data File Paths ---
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
DATA_RAW_PATH = os.path.join(DATA_PATH, 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
DATA_PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')

# --- 2. Model & Artifact Paths ---
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
PIPELINE_PATH = os.path.join(DATA_PROCESSED_PATH, 'preprocessing_pipeline.pkl')
LABEL_ENCODER_PATH = os.path.join(DATA_PROCESSED_PATH, 'label_encoder.pkl')

# --- 3. Feature Definitions ---
DROP_COLUMNS = ['customerID']
TARGET = 'Churn'

# Define all features (excluding target and drop columns)
ALL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
    'MonthlyCharges', 'TotalCharges'
]

# --- THE FIX IS HERE ---
# SeniorCitizen was incorrectly in NUMERICAL_FEATURES. It is categorical ("No"/"Yes").

CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
    'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen' # <-- Moved here
]

NUMERICAL_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges' # <-- Removed from here
]

# --- 4. Model Training Parameters ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- 5. MLflow Configuration ---
MLFLOW_TRACKING_URI = 'mlruns'
EXPERIMENT_NAME = 'Telecom Churn Prediction'
