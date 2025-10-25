"""
Configuration file for the Telecom Churn Prediction project.
"""

import os

# Paths
# Get the project root directory by going up three levels from the current file's location
# os.path.abspath(__file__) returns the absolute path of the current file
# Each os.path.dirname() removes one level from the path (goes up one directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.path.join() combines path components into a complete path using the OS-appropriate separator
# Example: if BASE_DIR = "/project", this creates "/project/data/raw" (Mac/Linux) or "\project\data\raw" (Windows)
DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
MODELS_PATH = os.path.join(BASE_DIR, "models")
OUTPUTS_PATH = os.path.join(BASE_DIR, "outputs")

# Model parameters
# RANDOM_STATE controls random number generation for reproducibility (same results every time)
# The value 42 is arbitrary but conventionally used; any integer works
RANDOM_STATE = 42
TEST_SIZE = 0.2

DATA_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Feature columns - YOU NEED TO UPDATE THESE based on your actual CSV columns!
# Run this first to see your columns:
# import pandas as pd
# df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# print(df.columns.tolist())

CATEGORICAL_FEATURES = [
    'customerID', 'gender', 'Partner', 'Dependents', 
    'PhoneService', 'MultipleLines', 'InternetService', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 
    'Contract', 'PaperlessBilling', 'PaymentMethod', 
    'TotalCharges', 'Churn'
]

NUMERICAL_FEATURES = ['SeniorCitizen', 'tenure', 'MonthlyCharges']

TARGET = 'Churn'

# Columns to drop
DROP_COLUMNS = ['customerID']