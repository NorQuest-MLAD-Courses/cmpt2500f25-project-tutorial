"""
Model training module for telecom churn prediction.
Contains functions to train various classification models.
"""

import pickle
import logging
from datetime import datetime
from typing import Any, Dict
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier

from .utils.config import RANDOM_STATE, MODELS_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> LogisticRegression:
    """
    Train Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for LogisticRegression
        
    Returns:
        Trained LogisticRegression model
    """
    logger.info("Training Logistic Regression...")
    
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    
    logger.info("Logistic Regression training completed")
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestClassifier:
    """
    Train Random Forest Classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for RandomForestClassifier
        
    Returns:
        Trained RandomForestClassifier model
    """
    logger.info("Training Random Forest...")
    
    # Set default parameters if not provided
    if 'random_state' not in kwargs:
        kwargs['random_state'] = RANDOM_STATE
    
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    
    logger.info("Random Forest training completed")
    return model


def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> DecisionTreeClassifier:
    """
    Train Decision Tree Classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for DecisionTreeClassifier
        
    Returns:
        Trained DecisionTreeClassifier model
    """
    logger.info("Training Decision Tree...")
    
    # Set default parameters if not provided
    if 'random_state' not in kwargs:
        kwargs['random_state'] = RANDOM_STATE
    
    model = DecisionTreeClassifier(**kwargs)
    model.fit(X_train, y_train)
    
    logger.info("Decision Tree training completed")
    return model


def train_adaboost(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> AdaBoostClassifier:
    """
    Train AdaBoost Classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for AdaBoostClassifier
        
    Returns:
        Trained AdaBoostClassifier model
    """
    logger.info("Training AdaBoost...")
    
    # Set default parameters if not provided
    if 'random_state' not in kwargs:
        kwargs['random_state'] = RANDOM_STATE
    
    model = AdaBoostClassifier(**kwargs)
    model.fit(X_train, y_train)
    
    logger.info("AdaBoost training completed")
    return model


def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> GradientBoostingClassifier:
    """
    Train Gradient Boosting Classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        **kwargs: Additional parameters for GradientBoostingClassifier
        
    Returns:
        Trained GradientBoostingClassifier model
    """
    logger.info("Training Gradient Boosting...")
    
    # Set default parameters if not provided
    if 'random_state' not in kwargs:
        kwargs['random_state'] = RANDOM_STATE
    
    model = GradientBoostingClassifier(**kwargs)
    model.fit(X_train, y_train)
    
    logger.info("Gradient Boosting training completed")
    return model


def train_voting_classifier(X_train: np.ndarray, y_train: np.ndarray, 
                           voting: str = 'soft', **kwargs) -> VotingClassifier:
    """
    Train Voting Classifier (ensemble of multiple models).
    
    Args:
        X_train: Training features
        y_train: Training target
        voting: Voting strategy ('hard' or 'soft')
        **kwargs: Additional parameters for base estimators
        
    Returns:
        Trained VotingClassifier model
    """
    logger.info("Training Voting Classifier (Ensemble)...")
    
    # Create base estimators
    clf1 = GradientBoostingClassifier(random_state=RANDOM_STATE)
    clf2 = LogisticRegression(max_iter=1000)
    clf3 = AdaBoostClassifier(random_state=RANDOM_STATE)
    
    # Create voting classifier
    model = VotingClassifier(
        estimators=[
            ('gbc', clf1),
            ('lr', clf2),
            ('abc', clf3)
        ],
        voting=voting
    )
    
    model.fit(X_train, y_train)
    
    logger.info("Voting Classifier training completed")
    return model


def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Train all available models and return them in a dictionary.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Dictionary of trained models
    """
    logger.info("Training all models...")
    
    models = {
        'logistic_regression': train_logistic_regression(X_train, y_train, max_iter=1000),
        'random_forest': train_random_forest(X_train, y_train),
        'decision_tree': train_decision_tree(X_train, y_train),
        'adaboost': train_adaboost(X_train, y_train),
        'gradient_boosting': train_gradient_boosting(X_train, y_train),
        'voting_classifier': train_voting_classifier(X_train, y_train)
    }
    
    logger.info(f"Trained {len(models)} models successfully")
    return models


def save_model(model: Any, model_name: str, output_dir: str = MODELS_PATH) -> str:
    """
    Save trained model to disk using pickle.
    
    Args:
        model: Trained model object
        model_name: Name for the model file
        output_dir: Directory to save the model
        
    Returns:
        Path to saved model file
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to: {filepath}")
    return filepath


def save_all_models(models: Dict[str, Any], output_dir: str = MODELS_PATH) -> Dict[str, str]:
    """
    Save all trained models to disk.
    
    Args:
        models: Dictionary of trained models
        output_dir: Directory to save models
        
    Returns:
        Dictionary mapping model names to file paths
    """
    saved_paths = {}
    
    for model_name, model in models.items():
        filepath = save_model(model, model_name, output_dir)
        saved_paths[model_name] = filepath
    
    logger.info(f"Saved {len(saved_paths)} models to {output_dir}")
    return saved_paths
