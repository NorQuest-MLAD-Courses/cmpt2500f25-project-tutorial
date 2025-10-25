"""
Feature engineering module for telecom churn prediction.
Contains functions for feature creation, selection, and analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_tenure_bins(df: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """
    Create tenure bins for categorical analysis.
    
    Args:
        df: Input DataFrame with 'tenure' column
        bins: Number of bins to create
        
    Returns:
        DataFrame with 'tenure_bin' column added
    """
    df = df.copy()
    
    df['tenure_bin'] = pd.cut(df['tenure'], bins=bins, labels=False)
    
    logger.info(f"Created {bins} tenure bins")
    return df


def get_feature_importance(model, feature_names: List[str], top_n: int = 10) -> pd.DataFrame:
    """
    Extract and return feature importance from tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with features and their importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    logger.info(f"Feature importance calculated for {len(feature_names)} features")
    
    if top_n:
        return importance_df.head(top_n)
    return importance_df


def select_top_features(X: pd.DataFrame, feature_importance_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Select top N features based on importance scores.
    
    Args:
        X: Input features DataFrame
        feature_importance_df: DataFrame with feature importance
        top_n: Number of top features to select
        
    Returns:
        DataFrame with only top N features
    """
    top_features = feature_importance_df.head(top_n)['feature'].tolist()
    X_selected = X[top_features]
    
    logger.info(f"Selected top {top_n} features: {top_features}")
    
    return X_selected


def calculate_correlation(df: pd.DataFrame, target: str = 'Churn', top_n: int = 10) -> pd.DataFrame:
    """
    Calculate correlation of features with target variable.
    
    Args:
        df: Input DataFrame
        target: Target variable name
        top_n: Number of top correlated features to return
        
    Returns:
        DataFrame with features and correlation scores
    """
    if target not in df.columns:
        logger.error(f"Target '{target}' not found in DataFrame")
        return None
    
    correlations = df.corr()[target].sort_values(ascending=False)
    correlation_df = pd.DataFrame({
        'feature': correlations.index,
        'correlation': correlations.values
    })
    
    # Remove target itself
    correlation_df = correlation_df[correlation_df['feature'] != target]
    
    logger.info(f"Calculated correlations with {target}")
    
    if top_n:
        return correlation_df.head(top_n)
    return correlation_df


def create_interaction_features(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create interaction features from pairs of features.
    
    Args:
        df: Input DataFrame
        feature_pairs: List of tuples containing feature pairs
        
    Returns:
        DataFrame with interaction features added
    """
    df = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f"{feat1}_x_{feat2}"
            df[interaction_name] = df[feat1] * df[feat2]
            logger.info(f"Created interaction feature: {interaction_name}")
    
    return df


def create_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated features specific to telecom churn.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with aggregated features added
    """
    df = df.copy()
    
    # Average monthly charge per tenure month
    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['avg_charge_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)  # +1 to avoid division by zero
    
    # Total to monthly charge ratio
    if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns:
        df['total_to_monthly_ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 0.01)  # Avoid division by zero
    
    logger.info("Created aggregated features")
    
    return df
