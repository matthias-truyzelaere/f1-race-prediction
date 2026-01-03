"""
Model training for F1 race prediction.

This module trains an XGBoost regression model to predict race finishing positions.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from utils import FEATURE_COLUMNS, get_latest_season, load_all_data, temporal_split

logger = logging.getLogger(__name__)


def prepare_data_for_training(
    df: pd.DataFrame, feature_cols: list, target_col: str = "finish_position"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target, keeping NaN in features (XGBoost handles them).

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_col: Target column name

    Returns:
        Tuple of (X, y)
    """
    # Only drop rows where the target is missing
    df_clean = df[feature_cols + [target_col]].copy()
    df_clean = df_clean[df_clean[target_col].notna()]

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    nan_counts = X.isna().sum()
    if nan_counts.sum() > 0:
        logger.info(f"   ℹ️ Features with NaN values: {nan_counts[nan_counts > 0].to_dict()}")

    return X, y


def create_model() -> XGBRegressor:
    """
    Create XGBoost regression model with optimized hyperparameters.

    Returns:
        Configured XGBoost model
    """
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=5,
        random_state=42,
        verbosity=1,
        enable_categorical=False,
    )
    return model


def evaluate_model(model: XGBRegressor, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Test") -> Dict[str, float]:
    """
    Evaluate model and print metrics.

    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        dataset_name: Name of dataset for display

    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    logger.info(f"\n📊 {dataset_name} Set Performance:")
    logger.info(f"   📊 MAE (Mean Absolute Error): {mae:.2f} positions")
    logger.info(f"   📊 RMSE (Root Mean Squared Error): {rmse:.2f} positions")
    logger.info(f"   📊 R² Score: {r2:.3f}")

    # Calculate accuracy within N positions
    errors = np.abs(y - y_pred)
    within_1 = (errors <= 1).sum() / len(errors) * 100
    within_2 = (errors <= 2).sum() / len(errors) * 100
    within_3 = (errors <= 3).sum() / len(errors) * 100

    logger.info("\n🎯 Prediction Accuracy:")
    logger.info(f"   🎯 Within 1 position: {within_1:.1f}%")
    logger.info(f"   🎯 Within 2 positions: {within_2:.1f}%")
    logger.info(f"   🎯 Within 3 positions: {within_3:.1f}%")

    return {"mae": mae, "rmse": rmse, "r2": r2, "within_3": within_3}


def show_feature_importance(model: XGBRegressor, feature_names: list, top_n: int = 10) -> None:
    """
    Display top N most important features.

    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of features to display
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        logger.info(f"\n🔍 Top {top_n} Most Important Features:")
        for i in range(min(top_n, len(feature_names))):
            idx = indices[i]
            logger.info(f"   🔍 {i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")


def save_model(model: XGBRegressor, feature_cols: list, model_dir: str, model_name: str = "f1_model.pkl") -> None:
    """
    Save trained model and feature columns.

    Args:
        model: Trained model
        feature_cols: List of feature column names
        model_dir: Directory to save model
        model_name: Filename for model
    """
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    model_file = model_path / model_name

    # Save model and metadata
    model_data = {"model": model, "feature_columns": feature_cols}

    with open(model_file, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(f"\n✅ Model saved to {model_file}")


def train_model() -> Tuple[XGBRegressor, Dict[str, float]]:
    """
    Main function to train the XGBoost model.

    Returns:
        Tuple of (trained_model, test_metrics)
    """
    import time

    training_start_time = time.time()

    # Auto-detect latest season
    data_dir = "data"
    model_dir = "models"
    test_season = get_latest_season(data_dir)

    # Load data
    logger.info("\n📂 Step 1/5: Loading historical data...")
    df = load_all_data(data_dir)
    logger.info(f"   📂 Loaded {len(df):,} races from {df['season'].nunique()} seasons")

    # Temporal split
    logger.info(f"\n📅 Step 2/5: Splitting data (test season: {test_season})...")
    train_df, test_df = temporal_split(df, test_season)

    # Get feature columns
    feature_cols = FEATURE_COLUMNS
    logger.info(f"\n🔧 Step 3/5: Preparing features ({len(feature_cols)} features)...")

    # Prepare training data
    X_train, y_train = prepare_data_for_training(train_df, feature_cols)
    logger.info(f"   📊 Training set: {len(X_train):,} samples")

    # Prepare test data
    X_test, y_test = prepare_data_for_training(test_df, feature_cols)
    logger.info(f"   📊 Test set: {len(X_test):,} samples")

    # Create model
    logger.info("\n🤖 Step 4/5: Training XGBoost model...")
    model = create_model()

    # Train model
    model_training_start = time.time()
    model.fit(X_train, y_train)
    model_training_time = time.time() - model_training_start
    logger.info(f"   ✅ Model trained in {model_training_time:.1f} seconds")

    # Evaluate on train set
    evaluate_model(model, X_train, y_train, dataset_name="Train")

    # Evaluate on test set
    test_metrics = evaluate_model(model, X_test, y_test, dataset_name="Test")

    # Show feature importance
    show_feature_importance(model, feature_cols)

    # Save model
    logger.info("\n💾 Step 5/5: Saving model...")
    save_model(model, feature_cols, model_dir)

    total_training_time = time.time() - training_start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"✅ Training pipeline completed in {total_training_time:.1f} seconds")
    logger.info(f"{'=' * 60}")

    return model, test_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_model()
