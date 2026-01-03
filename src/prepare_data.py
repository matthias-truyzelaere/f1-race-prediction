"""
Data preparation for F1 race prediction model.

This module loads historical data and performs temporal train/test splits.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from utils import FEATURE_COLUMNS, get_latest_season, load_all_data, temporal_split

logger = logging.getLogger(__name__)


def prepare_features_and_target(
    df: pd.DataFrame, feature_cols: list, target_col: str = "finish_position"
) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
    """
    Prepare feature matrix X and target vector y.

    IMPORTANT: Only removes rows with missing TARGET values.
    Missing feature values are kept (XGBoost handles them natively).

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_col: Target column name

    Returns:
        Tuple of (X, y, indices)
    """
    # Select features and target
    df_clean = df[feature_cols + [target_col]].copy()

    # Only drop rows where TARGET is missing (not features!)
    before_len = len(df_clean)
    df_clean = df_clean[df_clean[target_col].notna()]
    after_len = len(df_clean)

    if before_len > after_len:
        logger.warning(f"⚠️  Removed {before_len - after_len} rows with missing target values")

    # Check for NaN in features (this is OK for XGBoost)
    nan_counts = df_clean[feature_cols].isna().sum()
    if nan_counts.sum() > 0:
        logger.info(f"   ℹ️ Features with NaN values (XGBoost will handle): {nan_counts[nan_counts > 0].to_dict()}")

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    return X, y, df_clean.index


def save_prepared_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    """
    Save train and test sets to CSV files.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        output_dir: Directory to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / "train_data.csv"
    test_file = output_path / "test_data.csv"

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    logger.info(f"✅ Saved train data to {train_file}")
    logger.info(f"✅ Saved test data to {test_file}")


def prepare_data(
    data_dir: str, test_season: int, output_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Main function to prepare train/test data.

    Args:
        data_dir: Directory containing season CSV files
        test_season: Season to use for testing
        output_dir: Directory to save prepared data

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Load all data
    df = load_all_data(data_dir)

    # Temporal split
    train_df, test_df = temporal_split(df, test_season)

    # Prepare features and targets
    logger.info("\n🔧 Preparing training data...")
    X_train, y_train, train_idx = prepare_features_and_target(train_df, FEATURE_COLUMNS)
    train_df_clean = train_df.loc[train_idx]

    logger.info("\n🔧 Preparing test data...")
    X_test, y_test, test_idx = prepare_features_and_target(test_df, FEATURE_COLUMNS)
    test_df_clean = test_df.loc[test_idx]

    # Save prepared data
    logger.info("\n💾 Saving prepared data...")
    save_prepared_data(train_df_clean, test_df_clean, output_dir)

    # Print summary
    logger.info("\n📈 Data preparation summary:")
    logger.info(f"   📊 Features: {len(FEATURE_COLUMNS)}")
    logger.info(f"   📊 Train samples: {len(X_train)}")
    logger.info(f"   📊 Test samples: {len(X_test)}")
    logger.info(f"   📊 Target range: {y_train.min():.0f} - {y_train.max():.0f}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Auto-detect latest season and use defaults
    test_season = get_latest_season(data_dir="data")
    prepare_data(data_dir="data", test_season=test_season, output_dir="data")
