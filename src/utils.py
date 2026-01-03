"""
Shared utilities and constants for F1 race prediction.

This module contains common functions and constants used across multiple scripts.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

F1_POINTS_SYSTEM = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}

# API retry settings
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.0  # seconds

# Position thresholds
PODIUM_POSITION = 3
DNF_THRESHOLD = 20  # Positions > 20 considered DNF
Q3_THRESHOLD = 10  # Top 10 reach Q3

FEATURE_COLUMNS = [
    # Position-based features
    "grid_position",
    "qualifying_position",
    "qualifying_gap",
    "reached_q3",
    "has_grid_penalty",
    # Driver performance features
    "avg_finish_last_3",
    "avg_quali_last_3",
    "podium_rate_last_5",
    "dnf_rate_last_5",
    "avg_position_gain_last_3",
    "points_last_3",
    # Team features
    "team_avg_finish_season",
    "team_avg_quali_season",
    # Championship features
    "driver_championship_position",
    # Circuit-specific features
    "driver_circuit_avg",
    "team_circuit_avg",
]


def load_all_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load all season CSV files and combine them.

    Args:
        data_dir: Directory containing season CSV files

    Returns:
        Combined DataFrame sorted by season, round_number, and driver

    Raises:
        ValueError: If no season data files are found
    """
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*_season_data.csv"))

    if not csv_files:
        raise ValueError(f"No season data files found in {data_dir}")

    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(by=["season", "round_number", "driver"]).reset_index(drop=True)

    # Calculate memory usage
    memory_mb = combined_df.memory_usage(deep=True).sum() / 1024**2

    logger.info(f"✅ Loaded {len(combined_df):,} rows from {len(csv_files)} season files ({memory_mb:.1f} MB in memory)")

    return combined_df


def get_latest_season(data_dir: str = "data") -> int:
    """
    Automatically detect the latest season from CSV files.

    Args:
        data_dir: Directory containing season CSV files

    Returns:
        Latest season year

    Raises:
        ValueError: If no season data files are found
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*_season_data.csv"))

    if not csv_files:
        raise ValueError(f"No season data files found in {data_dir}")

    # Extract years from filenames
    years = [int(f.stem.split("_")[0]) for f in csv_files]
    latest_season = max(years)

    logger.info(f"🔍 Auto-detected latest season: {latest_season}")
    return latest_season


def get_available_years(data_dir: str = "data") -> List[int]:
    """
    Get all years that have data files.

    Args:
        data_dir: Directory containing season CSV files

    Returns:
        Sorted list of available years
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*_season_data.csv"))

    if not csv_files:
        return []

    years = sorted([int(f.stem.split("_")[0]) for f in csv_files])
    return years


def calculate_points(position: float) -> int:
    """
    Calculate F1 points based on finish position.

    Args:
        position: Finish position (1-20+)

    Returns:
        Points scored (0-25)
    """
    if pd.notna(position):
        return F1_POINTS_SYSTEM.get(int(position), 0)
    return 0


def temporal_split(df: pd.DataFrame, test_season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally - train on all seasons before test_season, test on test_season.

    Args:
        df: Combined DataFrame with all seasons
        test_season: Season to use for testing

    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = df[df["season"] < test_season].copy()
    test_df = df[df["season"] == test_season].copy()

    logger.info(f"📊 Train set: {len(train_df)} rows (seasons < {test_season})")
    logger.info(f"📊 Test set: {len(test_df)} rows (season {test_season})")

    return train_df, test_df
