"""
Model training module for telecom churn prediction.
Contains functions to train various classification models with hyperparameter tuning.
Includes MLflow experiment tracking.

CLI notes:
- Local saving is **disabled by default**.
- Use `--save` to write pickle files into `--output-dir` (default: MODELS_PATH).
- You can also pass `--no-save` explicitly (default), but it is redundant.
"""

# Standard library imports
import argparse
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

# Third-party imports
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Local imports
from .utils.config import MODELS_PATH, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def safe_log_params(params: dict):
    """
    Safely log parameters to MLflow, skipping if parameter already exists.

    Args:
        params: Dictionary of parameters to log
    """
    if mlflow.active_run() is None:
        return

    for key, value in params.items():
        try:
            mlflow.log_param(key, value)
        except mlflow.exceptions.MlflowException:
            # Parameter already logged, skip it
            logger.debug(f"Parameter '{key}' already logged, skipping")
            pass


def safe_log_metrics(metrics: dict):
    """
    Safely log metrics to MLflow.

    Args:
        metrics: Dictionary of metrics to log
    """
    if mlflow.active_run() is None:
        return

    for key, value in metrics.items():
        try:
            mlflow.log_metric(key, value)
        except mlflow.exceptions.MlflowException as e:
            logger.debug(f"Error logging metric '{key}': {e}")
            pass


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
        
        # Log parameter grid to MLflow
        safe_log_params({"param_grid": str(param_grid)})

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

        # Log best parameters and CV score to MLflow
        best_params = {f"best_{k}": v for k, v in grid_search.best_params_.items()}
        safe_log_params(best_params)
        safe_log_metrics({"cv_best_score": grid_search.best_score_})
        
        model = grid_search.best_estimator_
    else:
        # Log default parameters to MLflow
        default_params = {'C': 1.0, 'max_iter': 1000, 'penalty': 'l2', 'solver': 'lbfgs'}
        safe_log_params(default_params)

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
        
        # Log parameter grid to MLflow
        safe_log_params({"param_grid": str(param_grid)})
        
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
        
        # Log best parameters and CV score to MLflow
        # Log best parameters and CV score to MLflow
        best_params = {f"best_{k}": v for k, v in grid_search.best_params_.items()}
        safe_log_params(best_params)
        safe_log_metrics({"cv_best_score": grid_search.best_score_})
        
        model = grid_search.best_estimator_
    else:
        # Log default parameters to MLflow
        default_params = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 
                         'min_samples_leaf': 1, 'max_features': 'sqrt'}
        
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
        
        # Log parameter grid to MLflow
        safe_log_params({"param_grid": str(param_grid)})
        
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
        
        # Log best parameters and CV score to MLflow
        # Log best parameters and CV score to MLflow
        best_params = {f"best_{k}": v for k, v in grid_search.best_params_.items()}
        safe_log_params(best_params)
        safe_log_metrics({"cv_best_score": grid_search.best_score_})
        
        model = grid_search.best_estimator_
    else:
        # Log default parameters to MLflow
        default_params = {'max_depth': None, 'min_samples_split': 2, 
                         'min_samples_leaf': 1, 'criterion': 'gini'}
        
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
        
        # Log parameter grid to MLflow
        safe_log_params({"param_grid": str(param_grid)})
        
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
        
        # Log best parameters and CV score to MLflow
        # Log best parameters and CV score to MLflow
        best_params = {f"best_{k}": v for k, v in grid_search.best_params_.items()}
        safe_log_params(best_params)
        safe_log_metrics({"cv_best_score": grid_search.best_score_})
        
        model = grid_search.best_estimator_
    else:
        # Log default parameters to MLflow
        default_params = {'n_estimators': 50, 'learning_rate': 1.0, 'algorithm': 'SAMME.R'}
        
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
        
        # Log parameter grid to MLflow
        safe_log_params({"param_grid": str(param_grid)})
        
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
        
        # Log best parameters and CV score to MLflow
        # Log best parameters and CV score to MLflow
        best_params = {f"best_{k}": v for k, v in grid_search.best_params_.items()}
        safe_log_params(best_params)
        safe_log_metrics({"cv_best_score": grid_search.best_score_})
        
        model = grid_search.best_estimator_
    else:
        # Log default parameters to MLflow
        default_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 
                         'min_samples_split': 2, 'subsample': 1.0}
        
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
    
    # Log ensemble composition to MLflow
    safe_log_params({"voting_strategy": voting})
    safe_log_params({"ensemble_models": "GradientBoosting + LogisticRegression + AdaBoost"})
    
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


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Determine the positive label automatically
    unique_labels = np.unique(y_test)
    if len(unique_labels) > 0:
        if np.issubdtype(unique_labels.dtype, np.number):
            pos_label = 1
        else:
            # For string labels, sort and use the last one ('Yes' comes after 'No')
            pos_label = sorted(unique_labels)[-1]
    else:
        pos_label = 1  # Default fallback

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', pos_label=pos_label, zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', pos_label=pos_label, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='binary', pos_label=pos_label, zero_division=0),
    }

    # Calculate ROC-AUC if model supports predict_proba
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        # Convert y_test to binary (1 for positive class, 0 for negative)
        y_test_binary = (y_test == pos_label).astype(int)
        metrics['roc_auc'] = roc_auc_score(y_test_binary, y_proba)
    
    # Calculate confusion matrix components
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    })
    
    return metrics


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
    Save trained model to disk using pickle.
    
    Args:
        model: Trained model object
        model_name: Name for the model file
        output_dir: Directory to save the model
        
    Returns:
        Path to saved model file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp to filename (with microseconds to avoid collisions)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
    Main function for CLI training with MLflow tracking.
    """
    parser = argparse.ArgumentParser(
        description='Train machine learning models for telecom churn prediction with MLflow tracking'
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
    
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        '--no-save',
        action='store_true',
        dest='no_save',
        default=True,
        help='Do not save trained models to local disk (default)'
    )
    save_group.add_argument(
        '--save',
        action='store_false',
        dest='no_save',
        help='Save trained models to local disk'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='telecom-churn-prediction',
        help='MLflow experiment name (default: telecom-churn-prediction)'
    )
    
    args = parser.parse_args()
    
    # Set MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    logger.info(f"MLflow experiment: {args.experiment_name}")
    
    # Load preprocessed data
    logger.info(f"Loading data from {args.data}")
    data = np.load(args.data, allow_pickle=True).item()
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    
    # Train models
    if args.model == 'all':
        # Train all models, each in its own MLflow run
        models = {}
        saved_paths = {}

        model_names = [
            'logistic_regression',
            'random_forest',
            'decision_tree',
            'adaboost',
            'gradient_boosting',
            'voting_classifier'
        ]

        for model_name in model_names:
            run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with mlflow.start_run(run_name=run_name):
                start_time = datetime.now()

                # Log tags
                mlflow.set_tag("model_type", model_name)
                mlflow.set_tag("tuning", "enabled" if args.tune else "disabled")

                # Log basic parameters
                safe_log_params({"model_type": model_name})
                safe_log_params({"tune_hyperparameters": args.tune})
                safe_log_params({"random_state": RANDOM_STATE})

                # Train model
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {model_name}...")
                logger.info(f"{'='*60}")

                model_trainers = {
                    'logistic_regression': train_logistic_regression,
                    'random_forest': train_random_forest,
                    'decision_tree': train_decision_tree,
                    'adaboost': train_adaboost,
                    'gradient_boosting': train_gradient_boosting,
                    'voting_classifier': train_voting_classifier
                }

                trainer = model_trainers[model_name]

                if model_name == 'voting_classifier':
                    model = trainer(X_train, y_train)
                else:
                    model = trainer(X_train, y_train, tune_hyperparameters=args.tune)

                # Evaluate model
                metrics = evaluate_model(model, X_test, y_test)

                # Log metrics to MLflow (excluding non-numeric values)
                numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                safe_log_metrics(numeric_metrics)

                # Calculate training time
                elapsed = datetime.now() - start_time
                safe_log_metrics({"training_time_seconds": elapsed.total_seconds()})

                # Optionally save model locally
                model_path = None
                if not args.no_save:
                    model_path = save_model(model, model_name, args.output_dir)
                    saved_paths[model_name] = model_path
                models[model_name] = model

                # Log model to MLflow (kept independent of local saving)
                mlflow.sklearn.log_model(model, "model")

                # Log local model file as artifact
                if model_path is not None:
                    mlflow.log_artifact(model_path, "local_models")

                # Print results
                logger.info(f"\n{'='*60}")
                logger.info(f"Results for {model_name}")
                logger.info(f"{'='*60}")
                logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
                logger.info(f"Precision: {metrics['precision']:.4f}")
                logger.info(f"Recall:    {metrics['recall']:.4f}")
                logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
                if 'roc_auc' in metrics:
                    logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
                logger.info(f"Training time: {elapsed.total_seconds():.2f}s")
                if model_path is not None:
                    logger.info(f"Model saved: {model_path}")
                else:
                    logger.info("Model not saved to disk (default; use --save to enable)")
                logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
                logger.info(f"{'='*60}\n")

        print("\n" + "="*60)
        print("All models trained.")
        if saved_paths:
            print("="*60)
            print("Saved model artifacts:")
            for name, path in saved_paths.items():
                print(f"  - {name}: {path}")
            print("="*60)
        else:
            print("(No local model files were saved; pass --save to enable local saving.)")

    else:
        # Train single model
        run_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            start_time = datetime.now()

            # Log tags
            mlflow.set_tag("model_type", args.model)
            mlflow.set_tag("tuning", "enabled" if args.tune else "disabled")

            # Log basic parameters
            safe_log_params({"model_type": args.model})
            safe_log_params({"tune_hyperparameters": args.tune})
            safe_log_params({"random_state": RANDOM_STATE})

            # Train model
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

            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test)

            # Log metrics to MLflow (excluding non-numeric values)
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            safe_log_metrics(numeric_metrics)

            # Calculate training time
            elapsed = datetime.now() - start_time
            safe_log_metrics({"training_time_seconds": elapsed.total_seconds()})

            # Optionally save model locally
            model_path = None
            if not args.no_save:
                model_path = save_model(model, args.model, args.output_dir)

            # Log model to MLflow
            mlflow.sklearn.log_model(model, "model")

            # If saved locally, also log the file as an artifact
            if model_path is not None:
                mlflow.log_artifact(model_path, "local_models")

            # Print results
            if model_path is not None:
                print(f"\nTrained and saved {args.model}: {model_path}")
            else:
                print(f"\nTrained {args.model} (local saving disabled by default; use --save to enable)")
            print(f"\nMetrics:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            if 'roc_auc' in metrics:
                print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  Training time: {elapsed.total_seconds():.2f}s")
            print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
    
    logger.info("\n" + "="*60)
    logger.info("Training completed successfully!")
    logger.info("View experiments: mlflow ui --host 0.0.0.0 --port 5000")
    logger.info("="*60)


if __name__ == '__main__':
    main()
