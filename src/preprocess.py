"""
Data preprocessing module for telecom churn prediction.
Handles data loading, cleaning, encoding, and splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple
import logging

from .utils.config import (
    CATEGORICAL_FEATURES, 
    NUMERICAL_FEATURES, 
    TARGET, 
    DROP_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load telecom churn data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Convert TotalCharges to numeric (handles spaces and empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with 0 (for new customers)
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Check for any remaining missing values
    missing_count = df.isnull().sum().sum()
    logger.info(f"Missing values handled. Remaining missing values: {missing_count}")
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using Label Encoding.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with encoded categorical features
    """
    df = df.copy()
    
    # Encode target variable
    if TARGET in df.columns:
        le = LabelEncoder()
        df[TARGET] = le.fit_transform(df[TARGET])
        logger.info(f"Target '{TARGET}' encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    logger.info(f"Encoded {len(CATEGORICAL_FEATURES)} categorical features")
    
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not needed for modeling.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with unnecessary columns removed
    """
    df = df.copy()
    
    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    
    logger.info(f"Dropped columns: {cols_to_drop}")
    
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features (X) and target (y).
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (features, target)
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series, 
                     test_size: float = TEST_SIZE, 
                     random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Testing features
        
    Returns:
        Tuple of (scaled_X_train, scaled_X_test, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(filepath: str, scale: bool = True) -> Tuple:
    """
    Complete preprocessing pipeline from raw data to train/test split.
    
    Args:
        filepath: Path to raw data CSV
        scale: Whether to scale features
        
    Returns:
        If scale=True: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
        If scale=False: (X_train, X_test, y_train, y_test)
    """
    logger.info("Starting preprocessing pipeline...")
    
    # Load data
    df = load_data(filepath)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Drop unnecessary columns
    df = drop_unnecessary_columns(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Split features and target
    X, y = split_features_target(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    if scale:
        # Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        logger.info("Preprocessing pipeline completed with scaling")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    else:
        logger.info("Preprocessing pipeline completed without scaling")
        return X_train, X_test, y_train, y_test
