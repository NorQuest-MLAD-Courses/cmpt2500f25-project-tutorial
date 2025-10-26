"""
Model training module for telecom churn prediction.
Contains functions to train various classification models with hyperparameter tuning.
"""

# Standard library imports
import argparse
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

# Third-party imports
import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Local imports
from .utils.config import MODELS_PATH, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    tune_hyperparameters: bool = False,
    **kwargs
) -> LogisticRegression:
    """
    Train Logistic Regression model with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        tune_hyperparameters: Whether to perform hyperparameter tuning
        **kwargs: Additional parameters for LogisticRegression
        
    Returns:
        Trained LogisticRegression model
    """
    logger.info("Training Logistic Regression...")
    
    if tune_hyperparameters:
        logger.info("Performing hyperparameter tuning...")
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        }
        
        base_model = LogisticRegression(random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **kwargs)
        model.fit(X_train, y_train)
    
    logger.info("Logistic Regression training completed")
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune_hyperparameters: bool = False,
    **kwargs
) -> RandomForestClassifier:
    """
    Train Random Forest Classifier with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        tune_hyperparameters: Whether to perform hyperparameter tuning
        **kwargs: Additional parameters for RandomForestClassifier
        
    Returns:
        Trained RandomForestClassifier model
    """
    logger.info("Training Random Forest...")
    
    if tune_hyperparameters:
        logger.info("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        base_model = RandomForestClassifier(random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        if 'random_state' not in kwargs:
            kwargs['random_state'] = RANDOM_STATE
        
        model = RandomForestClassifier(**kwargs)
        model.fit(X_train, y_train)
    
    logger.info("Random Forest training completed")
    return model


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune_hyperparameters: bool = False,
    **kwargs
) -> DecisionTreeClassifier:
    """
    Train Decision Tree Classifier with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        tune_hyperparameters: Whether to perform hyperparameter tuning
        **kwargs: Additional parameters for DecisionTreeClassifier
        
    Returns:
        Trained DecisionTreeClassifier model
    """
    logger.info("Training Decision Tree...")
    
    if tune_hyperparameters:
        logger.info("Performing hyperparameter tuning...")
        param_grid = {
            'max_depth': [3, 5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy']
        }
        
        base_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        if 'random_state' not in kwargs:
            kwargs['random_state'] = RANDOM_STATE
        
        model = DecisionTreeClassifier(**kwargs)
        model.fit(X_train, y_train)
    
    logger.info("Decision Tree training completed")
    return model


def train_adaboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune_hyperparameters: bool = False,
    **kwargs
) -> AdaBoostClassifier:
    """
    Train AdaBoost Classifier with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        tune_hyperparameters: Whether to perform hyperparameter tuning
        **kwargs: Additional parameters for AdaBoostClassifier
        
    Returns:
        Trained AdaBoostClassifier model
    """
    logger.info("Training AdaBoost...")
    
    if tune_hyperparameters:
        logger.info("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
        }
        
        base_model = AdaBoostClassifier(random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        if 'random_state' not in kwargs:
            kwargs['random_state'] = RANDOM_STATE
        
        model = AdaBoostClassifier(**kwargs)
        model.fit(X_train, y_train)
    
    logger.info("AdaBoost training completed")
    return model


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune_hyperparameters: bool = False,
    **kwargs
) -> GradientBoostingClassifier:
    """
    Train Gradient Boosting Classifier with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        tune_hyperparameters: Whether to perform hyperparameter tuning
        **kwargs: Additional parameters for GradientBoostingClassifier
        
    Returns:
        Trained GradientBoostingClassifier model
    """
    logger.info("Training Gradient Boosting...")
    
    if tune_hyperparameters:
        logger.info("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        if 'random_state' not in kwargs:
            kwargs['random_state'] = RANDOM_STATE
        
        model = GradientBoostingClassifier(**kwargs)
        model.fit(X_train, y_train)
    
    logger.info("Gradient Boosting training completed")
    return model


def train_voting_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    voting: str = 'soft',
    **kwargs
) -> VotingClassifier:
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
    clf2 = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
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


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune_hyperparameters: bool = False
) -> Dict[str, Any]:
    """
    Train all available models and return them in a dictionary.
    
    Args:
        X_train: Training features
        y_train: Training target
        tune_hyperparameters: Whether to perform hyperparameter tuning
        
    Returns:
        Dictionary of trained models
    """
    logger.info("Training all models...")
    
    models = {
        'logistic_regression': train_logistic_regression(X_train, y_train, tune_hyperparameters),
        'random_forest': train_random_forest(X_train, y_train, tune_hyperparameters),
        'decision_tree': train_decision_tree(X_train, y_train, tune_hyperparameters),
        'adaboost': train_adaboost(X_train, y_train, tune_hyperparameters),
        'gradient_boosting': train_gradient_boosting(X_train, y_train, tune_hyperparameters),
        'voting_classifier': train_voting_classifier(X_train, y_train)
    }
    
    logger.info(f"Trained {len(models)} models successfully")
    return models


def save_model(model: Any, model_name: str, output_dir: str = MODELS_PATH) -> str:
    """
    Save trained model to disk using pickle/joblib.
    
    Args:
        model: Trained model object
        model_name: Name for the model file
        output_dir: Directory to save the model
        
    Returns:
        Path to saved model file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    # Save model using pickle
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


def main():
    """
    Main function for CLI training.
    """
    parser = argparse.ArgumentParser(
        description='Train machine learning models for telecom churn prediction'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to preprocessed training data (numpy file)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=[
            'logistic_regression',
            'random_forest',
            'decision_tree',
            'adaboost',
            'gradient_boosting',
            'voting_classifier',
            'all'
        ],
        default='all',
        help='Model type to train (default: all)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning using GridSearchCV'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=MODELS_PATH,
        help=f'Directory to save trained models (default: {MODELS_PATH})'
    )
    
    args = parser.parse_args()
    
    # Load preprocessed data
    logger.info(f"Loading data from {args.data}")
    data = np.load(args.data, allow_pickle=True).item()
    X_train = data['X_train']
    y_train = data['y_train']
    
    logger.info(f"Training set size: {X_train.shape}")
    
    # Train models
    if args.model == 'all':
        models = train_all_models(X_train, y_train, tune_hyperparameters=args.tune)
        saved_paths = save_all_models(models, args.output_dir)
        
        print("\nTrained and saved models:")
        for name, path in saved_paths.items():
            print(f"  - {name}: {path}")
    else:
        # Train single model
        model_trainers = {
            'logistic_regression': train_logistic_regression,
            'random_forest': train_random_forest,
            'decision_tree': train_decision_tree,
            'adaboost': train_adaboost,
            'gradient_boosting': train_gradient_boosting,
            'voting_classifier': train_voting_classifier
        }
        
        trainer = model_trainers[args.model]
        
        if args.model == 'voting_classifier':
            model = trainer(X_train, y_train)
        else:
            model = trainer(X_train, y_train, tune_hyperparameters=args.tune)
        
        model_path = save_model(model, args.model, args.output_dir)
        print(f"\nTrained and saved {args.model}: {model_path}")
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()
