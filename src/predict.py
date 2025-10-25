"""
Prediction module for telecom churn prediction.
Contains functions to load models and make predictions.
"""

import pickle
import logging
from typing import Any, Union
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model file
        
    Returns:
        Loaded model object
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from: {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def predict(model: Any, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model object
        X: Features for prediction
        
    Returns:
        Array of predictions
    """
    try:
        predictions = model.predict(X)
        logger.info(f"Predictions generated for {len(predictions)} samples")
        return predictions
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise


def predict_proba(model: Any, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Get prediction probabilities from a trained model.
    
    Args:
        model: Trained model object
        X: Features for prediction
        
    Returns:
        Array of prediction probabilities
    """
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            logger.info(f"Probabilities generated for {len(probabilities)} samples")
            return probabilities
        else:
            logger.warning("Model does not support probability predictions")
            return None
    except Exception as e:
        logger.error(f"Error generating probabilities: {e}")
        raise


def predict_single(model: Any, features: dict) -> tuple:
    """
    Make prediction for a single customer.
    
    Args:
        model: Trained model object
        features: Dictionary of feature values
        
    Returns:
        Tuple of (prediction, probability)
    """
    # Convert dict to DataFrame
    X = pd.DataFrame([features])
    
    # Get prediction
    prediction = predict(model, X)[0]
    
    # Get probability if available
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = predict_proba(model, X)[0]
    
    logger.info(f"Single prediction: {prediction}")
    
    return prediction, proba


def batch_predict(model: Any, X: Union[np.ndarray, pd.DataFrame], 
                  batch_size: int = 1000) -> np.ndarray:
    """
    Make predictions in batches for large datasets.
    
    Args:
        model: Trained model object
        X: Features for prediction
        batch_size: Number of samples per batch
        
    Returns:
        Array of all predictions
    """
    n_samples = len(X)
    predictions = []
    
    for i in range(0, n_samples, batch_size):
        batch = X[i:i+batch_size]
        batch_preds = predict(model, batch)
        predictions.extend(batch_preds)
    
    logger.info(f"Batch prediction completed for {n_samples} samples")
    return np.array(predictions)
