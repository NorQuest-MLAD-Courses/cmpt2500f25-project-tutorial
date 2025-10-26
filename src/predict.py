"""
Prediction module for telecom churn prediction.
Contains functions to load models and make predictions with CLI support.
"""

# Standard library imports
import argparse
import logging
import pickle
from typing import Any, Optional, Union

# Third-party imports
import joblib
import numpy as np
import pandas as pd

# Local imports
from .utils.config import MODELS_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk using joblib or pickle.
    
    Args:
        filepath: Path to the saved model file
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    try:
        # Try joblib first (more efficient for sklearn models)
        model = joblib.load(filepath)
        logger.info(f"Model loaded successfully (joblib) from: {filepath}")
        return model
    except:
        # Fall back to pickle
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded successfully (pickle) from: {filepath}")
            return model
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


def load_preprocessing_pipeline(filepath: str) -> Any:
    """
    Load a preprocessing pipeline from disk.
    
    Args:
        filepath: Path to the saved pipeline file
        
    Returns:
        Loaded pipeline object
    """
    try:
        pipeline = joblib.load(filepath)
        logger.info(f"Preprocessing pipeline loaded from: {filepath}")
        return pipeline
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        raise


def predict(model: Any, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model object
        X: Features for prediction
        
    Returns:
        Array of predictions
        
    Raises:
        ValueError: If input data is invalid
    """
    try:
        if isinstance(X, pd.DataFrame):
            # Ensure DataFrame has no issues
            if X.empty:
                raise ValueError("Input DataFrame is empty")
        elif isinstance(X, np.ndarray):
            # Ensure array has valid shape
            if X.size == 0:
                raise ValueError("Input array is empty")
        
        predictions = model.predict(X)
        logger.info(f"Predictions generated for {len(predictions)} samples")
        return predictions
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise


def predict_proba(model: Any, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
    """
    Get prediction probabilities from a trained model.
    
    Args:
        model: Trained model object
        X: Features for prediction
        
    Returns:
        Array of prediction probabilities, or None if model doesn't support it
    """
    try:
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model does not support probability predictions")
            return None
        
        probabilities = model.predict_proba(X)
        logger.info(f"Probabilities generated for {len(probabilities)} samples")
        return probabilities
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


def batch_predict(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    batch_size: int = 1000
) -> np.ndarray:
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
    
    logger.info(f"Processing {n_samples} samples in batches of {batch_size}...")
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch = X[i:batch_end]
        batch_preds = predict(model, batch)
        predictions.extend(batch_preds)
        
        logger.info(f"Processed samples {i+1} to {batch_end}")
    
    logger.info(f"Batch prediction completed for {n_samples} samples")
    return np.array(predictions)


class ModelPredictor:
    """
    Predictor class that handles model loading and predictions.
    
    This class provides a clean interface for making predictions and can
    optionally handle preprocessing automatically.
    """
    
    def __init__(self, model_path: str, pipeline_path: Optional[str] = None):
        """
        Initialize predictor with model and optional preprocessing pipeline.
        
        Args:
            model_path: Path to saved model
            pipeline_path: Optional path to preprocessing pipeline
        """
        self.model = load_model(model_path)
        self.pipeline = load_preprocessing_pipeline(pipeline_path) if pipeline_path else None
        logger.info(f"ModelPredictor initialized")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions, applying preprocessing if pipeline is available.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions
        """
        if self.pipeline is not None:
            X = self.pipeline.transform(X)
            logger.info("Applied preprocessing pipeline before prediction")
        
        return predict(self.model, X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """
        Get prediction probabilities, applying preprocessing if available.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of probabilities or None
        """
        if self.pipeline is not None:
            X = self.pipeline.transform(X)
            logger.info("Applied preprocessing pipeline before probability prediction")
        
        return predict_proba(self.model, X)
    
    def predict_single(self, features: dict) -> tuple:
        """
        Make prediction for a single sample.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Tuple of (prediction, probability)
        """
        X = pd.DataFrame([features])
        
        if self.pipeline is not None:
            X = self.pipeline.transform(X)
        
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0] if hasattr(self.model, 'predict_proba') else None
        
        return prediction, proba


def main():
    """
    Main function for CLI prediction.
    """
    parser = argparse.ArgumentParser(
        description='Make predictions using trained telecom churn model'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to input data (CSV or numpy file)'
    )
    
    parser.add_argument(
        '--pipeline',
        type=str,
        help='Path to preprocessing pipeline (optional)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save predictions (optional)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for predictions (default: 1000)'
    )
    
    parser.add_argument(
        '--proba',
        action='store_true',
        help='Output probabilities instead of class predictions'
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from: {args.model}")
    predictor = ModelPredictor(args.model, args.pipeline)
    
    # Load data
    logger.info(f"Loading data from: {args.data}")
    if args.data.endswith('.csv'):
        X = pd.read_csv(args.data)
        logger.info(f"Loaded CSV data with shape: {X.shape}")
    elif args.data.endswith('.npy'):
        data = np.load(args.data, allow_pickle=True).item()
        X = data['X_test']  # Assumes test data
        logger.info(f"Loaded numpy data with shape: {X.shape}")
    else:
        raise ValueError("Data file must be .csv or .npy format")
    
    # Make predictions
    if args.proba:
        logger.info("Generating probability predictions...")
        predictions = predictor.predict_proba(X)
        if predictions is None:
            print("Error: Model does not support probability predictions")
            return
    else:
        logger.info("Generating class predictions...")
        predictions = predictor.predict(X)
    
    # Display or save predictions
    if args.output:
        np.save(args.output, predictions)
        logger.info(f"Predictions saved to: {args.output}")
        print(f"\nPredictions saved to: {args.output}")
    else:
        print("\n" + "="*60)
        print("Predictions:")
        print("="*60)
        print(predictions[:20])  # Show first 20
        if len(predictions) > 20:
            print(f"... ({len(predictions) - 20} more predictions)")
        print("="*60)
    
    # Summary statistics
    if not args.proba:
        unique, counts = np.unique(predictions, return_counts=True)
        print("\nPrediction distribution:")
        for val, count in zip(unique, counts):
            print(f"  Class {val}: {count} samples ({count/len(predictions)*100:.2f}%)")
    
    logger.info("Prediction completed!")


if __name__ == '__main__':
    main()
