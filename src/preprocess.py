"""
Data preprocessing module for telecom churn prediction.
Handles data loading, cleaning, encoding, and splitting using scikit-learn pipelines.
"""

# Standard library imports
import argparse
import logging
import os
from typing import Optional, Tuple

# Third-party imports
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Local imports
from .utils.config import (
    CATEGORICAL_FEATURES,
    DATA_PROCESSED_PATH,
    DATA_RAW_PATH,
    DROP_COLUMNS,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load telecom churn data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {filepath}")
        raise
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


def drop_unnecessary_columns(df: pd.DataFrame, columns: list = DROP_COLUMNS) -> pd.DataFrame:
    """
    Drop columns that are not needed for modeling.
    
    Args:
        df: Input DataFrame
        columns: List of column names to drop
        
    Returns:
        DataFrame with unnecessary columns removed
    """
    df = df.copy()
    
    cols_to_drop = [col for col in columns if col in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"Dropped columns: {cols_to_drop}")
    
    return df


def encode_target(df: pd.DataFrame, target_col: str = TARGET) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode the target variable using Label Encoding.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        Tuple of (DataFrame with encoded target, fitted LabelEncoder)
    """
    df = df.copy()
    
    if target_col in df.columns:
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        logger.info(f"Target '{target_col}' encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        return df, le
    else:
        logger.warning(f"Target column '{target_col}' not found in DataFrame")
        return df, None


def split_features_target(df: pd.DataFrame, target_col: str = TARGET) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features (X) and target (y).
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        Tuple of (features, target)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y


def create_preprocessing_pipeline(
    numerical_features: list,
    categorical_features: list,
    encoding: str = 'label'
) -> Pipeline:
    """
    Create a scikit-learn preprocessing pipeline using ColumnTransformer.
    
    This is the RECOMMENDED way to preprocess data in production as it:
    1. Ensures consistency between training and prediction
    2. Prevents data leakage
    3. Makes the pipeline reproducible and deployable
    4. Bundles all preprocessing steps together
    
    Args:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        encoding: Encoding method ('label' or 'onehot')
        
    Returns:
        Fitted preprocessing pipeline
    """
    logger.info("Creating preprocessing pipeline...")
    
    # Define transformers for numerical features
    numerical_transformer = StandardScaler()
    
    # Define transformers for categorical features
    if encoding == 'onehot':
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:  # label encoding
        # Note: LabelEncoder doesn't work directly in pipelines, so we use OneHotEncoder
        # and then manually handle label encoding separately if needed
        from sklearn.preprocessing import OrdinalEncoder
        categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Keep any remaining columns as-is
    )
    
    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    logger.info(f"Pipeline created with {encoding} encoding for categorical features")
    
    return pipeline


def preprocess_data_with_pipeline(
    df: pd.DataFrame,
    pipeline: Optional[Pipeline] = None,
    fit: bool = True
) -> Tuple[np.ndarray, Pipeline]:
    """
    Preprocess data using scikit-learn pipeline.
    
    Args:
        df: Input DataFrame
        pipeline: Existing pipeline (if None, creates new one)
        fit: Whether to fit the pipeline (True for training data, False for test data)
        
    Returns:
        Tuple of (transformed data, fitted pipeline)
    """
    if pipeline is None:
        # Identify categorical and numerical features
        cat_features = [col for col in CATEGORICAL_FEATURES if col in df.columns and col != TARGET]
        num_features = [col for col in NUMERICAL_FEATURES if col in df.columns]
        
        pipeline = create_preprocessing_pipeline(num_features, cat_features)
    
    if fit:
        X_transformed = pipeline.fit_transform(df)
        logger.info("Pipeline fitted and data transformed")
    else:
        X_transformed = pipeline.transform(df)
        logger.info("Data transformed using existing pipeline")
    
    return X_transformed, pipeline


def preprocess_pipeline(
    filepath: str,
    scale: bool = True,
    use_sklearn_pipeline: bool = True
) -> Tuple:
    """
    Complete preprocessing pipeline from raw data to train/test split.
    
    This function now uses scikit-learn pipelines (recommended) by default,
    but maintains backward compatibility with the original approach.
    
    Args:
        filepath: Path to raw data CSV
        scale: Whether to scale features
        use_sklearn_pipeline: Whether to use scikit-learn Pipeline (RECOMMENDED)
        
    Returns:
        If use_sklearn_pipeline=True:
            (X_train, X_test, y_train, y_test, pipeline, label_encoder)
        If use_sklearn_pipeline=False (legacy):
            (X_train, X_test, y_train, y_test, scaler) if scale=True
            (X_train, X_test, y_train, y_test) if scale=False
    """
    logger.info("Starting preprocessing pipeline...")
    
    # Load data
    df = load_data(filepath)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Drop unnecessary columns
    df = drop_unnecessary_columns(df)
    
    # Encode target variable
    df, label_encoder = encode_target(df)
    
    # Split features and target
    X, y = split_features_target(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    if use_sklearn_pipeline:
        # Use scikit-learn Pipeline (RECOMMENDED)
        X_train_transformed, pipeline = preprocess_data_with_pipeline(X_train, fit=True)
        X_test_transformed, _ = preprocess_data_with_pipeline(X_test, pipeline=pipeline, fit=False)
        
        logger.info("Preprocessing pipeline completed using scikit-learn Pipeline")
        return X_train_transformed, X_test_transformed, y_train, y_test, pipeline, label_encoder
    
    else:
        # Legacy approach (kept for backward compatibility)
        logger.warning("Using legacy preprocessing approach. Consider switching to sklearn pipelines.")
        
        # Manual encoding (label encoding)
        from sklearn.preprocessing import LabelEncoder
        
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        label_encoders = {}
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                le = LabelEncoder()
                X_train_encoded[col] = le.fit_transform(X_train[col])
                X_test_encoded[col] = le.transform(X_test[col])
                label_encoders[col] = le
        
        if scale:
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_encoded)
            X_test_scaled = scaler.transform(X_test_encoded)
            
            logger.info("Preprocessing pipeline completed with scaling (legacy)")
            return X_train_scaled, X_test_scaled, y_train, y_test, scaler
        else:
            logger.info("Preprocessing pipeline completed without scaling (legacy)")
            return X_train_encoded.values, X_test_encoded.values, y_train, y_test


def save_preprocessed_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    pipeline: Optional[Pipeline] = None,
    label_encoder: Optional[LabelEncoder] = None,
    output_dir: str = DATA_PROCESSED_PATH
) -> dict:
    """
    Save preprocessed data and preprocessing artifacts to disk.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        pipeline: Fitted preprocessing pipeline
        label_encoder: Fitted label encoder for target
        output_dir: Directory to save files
        
    Returns:
        Dictionary with paths to saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = {}
    
    # Save data as numpy files
    data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    data_path = os.path.join(output_dir, 'preprocessed_data.npy')
    np.save(data_path, data_dict)
    saved_paths['data'] = data_path
    logger.info(f"Preprocessed data saved to: {data_path}")
    
    # Save pipeline if provided
    if pipeline is not None:
        pipeline_path = os.path.join(output_dir, 'preprocessing_pipeline.pkl')
        joblib.dump(pipeline, pipeline_path)
        saved_paths['pipeline'] = pipeline_path
        logger.info(f"Preprocessing pipeline saved to: {pipeline_path}")
    
    # Save label encoder if provided
    if label_encoder is not None:
        encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
        joblib.dump(label_encoder, encoder_path)
        saved_paths['label_encoder'] = encoder_path
        logger.info(f"Label encoder saved to: {encoder_path}")
    
    return saved_paths


def main():
    """
    Main function for CLI preprocessing.
    """
    parser = argparse.ArgumentParser(
        description='Preprocess telecom churn data for model training'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DATA_PROCESSED_PATH,
        help=f'Directory to save preprocessed data (default: {DATA_PROCESSED_PATH})'
    )
    
    parser.add_argument(
        '--no-scale',
        action='store_true',
        help='Disable feature scaling'
    )
    
    parser.add_argument(
        '--legacy',
        action='store_true',
        help='Use legacy preprocessing approach instead of sklearn pipelines'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=TEST_SIZE,
        help=f'Proportion of data for testing (default: {TEST_SIZE})'
    )
    
    args = parser.parse_args()
    
    # Run preprocessing
    logger.info(f"Preprocessing data from: {args.input}")
    
    if args.legacy:
        # Legacy preprocessing
        if args.no_scale:
            X_train, X_test, y_train, y_test = preprocess_pipeline(
                args.input,
                scale=False,
                use_sklearn_pipeline=False
            )
            pipeline = None
            label_encoder = None
        else:
            X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
                args.input,
                scale=True,
                use_sklearn_pipeline=False
            )
            pipeline = scaler
            label_encoder = None
    else:
        # Sklearn pipeline preprocessing (recommended)
        X_train, X_test, y_train, y_test, pipeline, label_encoder = preprocess_pipeline(
            args.input,
            scale=not args.no_scale,
            use_sklearn_pipeline=True
        )
    
    # Save preprocessed data
    saved_paths = save_preprocessed_data(
        X_train, X_test, y_train, y_test,
        pipeline, label_encoder,
        args.output_dir
    )
    
    print("\n" + "="*60)
    print("Preprocessing completed successfully!")
    print("="*60)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"\nSaved files:")
    for key, path in saved_paths.items():
        print(f"  - {key}: {path}")
    print("="*60)
    
    logger.info("Preprocessing completed!")


if __name__ == '__main__':
    main()
