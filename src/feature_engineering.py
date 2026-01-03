"""
Feature engineering for F1 race prediction.

This module adds engineered features to raw race data using optimized vectorized operations.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import DNF_THRESHOLD, PODIUM_POSITION, calculate_points

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Add engineered features to existing season CSV files")

    parser.add_argument(
        "--start-year",
        type=int,
        required=True,
        help="Start season year (e.g. 2024)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        required=False,
        help="End season year (e.g. 2025). If not provided, only start-year is used",
    )

    return parser.parse_args()


def add_avg_finish_last_3(df: pd.DataFrame, driver_state: Dict) -> List[float]:
    """Add rolling average of last 3 finish positions."""
    avg_finish_list = []

    for _, row in df.iterrows():
        driver = row["driver"]
        last_finishes = driver_state["finishes"].get(driver, [])

        if last_finishes:
            avg_finish = sum(last_finishes[-3:]) / min(len(last_finishes), 3)
        else:
            avg_finish = np.nan

        avg_finish_list.append(avg_finish)

        # Update state
        if pd.notna(row["finish_position"]):
            if driver not in driver_state["finishes"]:
                driver_state["finishes"][driver] = []
            driver_state["finishes"][driver].append(row["finish_position"])

    return avg_finish_list


def add_avg_quali_last_3(df: pd.DataFrame, driver_state: Dict) -> List[float]:
    """Add rolling average of last 3 qualifying positions."""
    avg_quali_list = []

    for _, row in df.iterrows():
        driver = row["driver"]
        last_quali = driver_state["quali"].get(driver, [])

        if last_quali:
            avg_quali = sum(last_quali[-3:]) / min(len(last_quali), 3)
        else:
            avg_quali = np.nan

        avg_quali_list.append(avg_quali)

        # Update state
        if pd.notna(row["qualifying_position"]):
            if driver not in driver_state["quali"]:
                driver_state["quali"][driver] = []
            driver_state["quali"][driver].append(row["qualifying_position"])

    return avg_quali_list


def add_podium_rate_last_5(df: pd.DataFrame, driver_state: Dict) -> List[float]:
    """Add podium rate (top 3 finishes) in last 5 races."""
    podium_rate_list = []

    for _, row in df.iterrows():
        driver = row["driver"]
        last_finishes = driver_state["finishes"].get(driver, [])

        if last_finishes:
            recent = last_finishes[-5:]
            podiums = sum(1 for pos in recent if pos <= PODIUM_POSITION)
            podium_rate = podiums / len(recent)
        else:
            podium_rate = np.nan

        podium_rate_list.append(podium_rate)

    return podium_rate_list


def add_dnf_rate_last_5(df: pd.DataFrame, driver_state: Dict) -> List[float]:
    """Add DNF (Did Not Finish) rate in last 5 races."""
    dnf_rate_list = []

    for _, row in df.iterrows():
        driver = row["driver"]
        last_finishes = driver_state["finishes"].get(driver, [])

        if last_finishes:
            recent = last_finishes[-5:]
            dnfs = sum(1 for pos in recent if pos > DNF_THRESHOLD or pd.isna(pos))
            dnf_rate = dnfs / len(recent)
        else:
            dnf_rate = np.nan

        dnf_rate_list.append(dnf_rate)

    return dnf_rate_list


def add_avg_position_gain_last_3(df: pd.DataFrame, driver_state: Dict) -> List[float]:
    """Add average position gain from qualifying to finish."""
    avg_gain_list = []

    for _, row in df.iterrows():
        driver = row["driver"]
        last_gains = driver_state["position_gains"].get(driver, [])

        if last_gains:
            avg_gain = sum(last_gains[-3:]) / min(len(last_gains), 3)
        else:
            avg_gain = np.nan

        avg_gain_list.append(avg_gain)

        # Update state (negative means gained positions)
        quali_pos = row["qualifying_position"]
        finish_pos = row["finish_position"]
        if pd.notna(quali_pos) and pd.notna(finish_pos):
            gain = quali_pos - finish_pos
            if driver not in driver_state["position_gains"]:
                driver_state["position_gains"][driver] = []
            driver_state["position_gains"][driver].append(gain)

    return avg_gain_list


def add_points_last_3(df: pd.DataFrame, driver_state: Dict) -> List[float]:
    """Add total points scored in last 3 races."""
    points_last_3_list = []

    for _, row in df.iterrows():
        driver = row["driver"]
        last_points = driver_state["points"].get(driver, [])

        if last_points:
            points_sum = sum(last_points[-3:])
        else:
            points_sum = np.nan

        points_last_3_list.append(points_sum)

        # Update state
        if pd.notna(row["finish_position"]):
            points = calculate_points(row["finish_position"])
            if driver not in driver_state["points"]:
                driver_state["points"][driver] = []
            driver_state["points"][driver].append(points)

    return points_last_3_list


def add_team_avg_finish_season(df: pd.DataFrame) -> pd.Series:
    """
    Add team's average finish position so far this season.
    VECTORIZED: Uses groupby + expanding mean - much faster than row iteration!
    """
    # Group by season and team, calculate expanding mean, shift to exclude current row
    team_avg = (
        df.groupby(["season", "team"])["finish_position"].expanding().mean().shift(1).reset_index(level=[0, 1], drop=True)
    )

    # Reindex to match original DataFrame
    return team_avg.reindex(df.index).fillna(np.nan)


def add_team_avg_quali_season(df: pd.DataFrame) -> pd.Series:
    """
    Add team's average qualifying position so far this season.
    VECTORIZED: Uses groupby + expanding mean.
    """
    team_quali_avg = (
        df.groupby(["season", "team"])["qualifying_position"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=[0, 1], drop=True)
    )

    return team_quali_avg.reindex(df.index).fillna(np.nan)


def add_driver_championship_position(df: pd.DataFrame) -> List[float]:
    """
    Add driver's position in championship before this race.
    OPTIMIZED: Cache championship standings per season/round.
    """
    championship_pos_list = []

    # Pre-calculate all championship standings
    championship_cache = {}

    for season in df["season"].unique():
        season_df = df[df["season"] == season]

        for round_num in season_df["round_number"].unique():
            key = (season, round_num)

            # Get all previous rounds
            previous_races = season_df[season_df["round_number"] < round_num]

            if len(previous_races) > 0:
                # Calculate points for each driver
                driver_points = {}
                for driver in previous_races["driver"].unique():
                    driver_races = previous_races[previous_races["driver"] == driver]
                    total_points = sum(calculate_points(pos) for pos in driver_races["finish_position"] if pd.notna(pos))
                    driver_points[driver] = total_points

                # Sort by points and create position mapping
                sorted_drivers = sorted(driver_points.items(), key=lambda x: x[1], reverse=True)
                position_map = {driver: i + 1 for i, (driver, _) in enumerate(sorted_drivers)}
                championship_cache[key] = position_map
            else:
                championship_cache[key] = {}

    # Now populate the list
    for _, row in df.iterrows():
        key = (row["season"], row["round_number"])
        position_map = championship_cache.get(key, {})
        championship_pos = position_map.get(row["driver"], np.nan)
        championship_pos_list.append(championship_pos)

    return championship_pos_list


def add_grid_penalty(df: pd.DataFrame) -> pd.Series:
    """
    Indicate if driver has grid penalty (grid != qualifying position).
    VECTORIZED: Single pandas comparison.
    """
    return ((df["grid_position"] != df["qualifying_position"]) & df["grid_position"].notna()).astype(int)


def add_driver_circuit_avg(df: pd.DataFrame) -> pd.Series:
    """
    Add driver's average finish at this circuit in previous years.
    VECTORIZED: Uses groupby + expanding mean with multi-index.
    """
    # Sort by driver, circuit, and season
    df_sorted = df.sort_values(["driver", "grand_prix", "season"]).copy()

    # Group by driver and circuit, calculate expanding mean of finish_position
    circuit_avg = (
        df_sorted.groupby(["driver", "grand_prix"])["finish_position"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=[0, 1], drop=True)
    )

    # Reindex to match original DataFrame order
    return circuit_avg.reindex(df.index).fillna(np.nan)


def add_team_circuit_avg(df: pd.DataFrame) -> pd.Series:
    """
    Add team's average finish at this circuit in previous years.
    VECTORIZED: Uses groupby + expanding mean.
    """
    # Sort by team, circuit, and season
    df_sorted = df.sort_values(["team", "grand_prix", "season"]).copy()

    # Group by team and circuit, calculate expanding mean
    team_circuit_avg = (
        df_sorted.groupby(["team", "grand_prix"])["finish_position"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=[0, 1], drop=True)
    )

    return team_circuit_avg.reindex(df.index).fillna(np.nan)


def add_all_features(season_files: List[Path]) -> None:
    """
    Add all engineered features to the dataset.

    Args:
        season_files: List of paths to season CSV files
    """
    # State dictionary to maintain rolling statistics across seasons
    driver_state = {
        "finishes": {},
        "quali": {},
        "position_gains": {},
        "points": {},
    }

    all_dfs = []

    # Load all data
    for season_file in season_files:
        df = pd.read_csv(season_file)
        all_dfs.append(df)

    # Combine all data
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure proper sort
    combined_df = combined_df.sort_values(
        by=[
            "season",
            "round_number",
            "driver",
        ]
    ).reset_index(drop=True)

    total_features = 12
    logger.info(f"🔧 Adding {total_features} engineered features...")

    # Add features that need state (rolling features)
    logger.info(f"   🔧 (Step 1/{total_features}) Rolling driver statistics...")
    combined_df["avg_finish_last_3"] = add_avg_finish_last_3(combined_df, driver_state)
    combined_df["avg_quali_last_3"] = add_avg_quali_last_3(combined_df, driver_state)
    combined_df["podium_rate_last_5"] = add_podium_rate_last_5(combined_df, driver_state)
    combined_df["dnf_rate_last_5"] = add_dnf_rate_last_5(combined_df, driver_state)
    combined_df["avg_position_gain_last_3"] = add_avg_position_gain_last_3(combined_df, driver_state)
    combined_df["points_last_3"] = add_points_last_3(combined_df, driver_state)

    # Add vectorized features (MUCH faster!)
    logger.info(f"   🔧 (Step 7/{total_features}) Team season statistics (vectorized)...")
    combined_df["team_avg_finish_season"] = add_team_avg_finish_season(combined_df)
    combined_df["team_avg_quali_season"] = add_team_avg_quali_season(combined_df)

    logger.info(f"   🔧 (Step 9/{total_features}) Championship positions...")
    combined_df["driver_championship_position"] = add_driver_championship_position(combined_df)

    logger.info(f"   🔧 (Step 10/{total_features}) Grid penalties...")
    combined_df["has_grid_penalty"] = add_grid_penalty(combined_df)

    logger.info(f"   🔧 (Step 11/{total_features}) Circuit-specific averages (vectorized)...")
    combined_df["driver_circuit_avg"] = add_driver_circuit_avg(combined_df)
    combined_df["team_circuit_avg"] = add_team_circuit_avg(combined_df)

    logger.info(f"✅ All {total_features} features added successfully")

    # Split back into separate season files and save
    for season_file in tqdm(season_files, desc="💾 Saving updated files"):
        year = int(season_file.stem.split("_")[0])
        season_df = combined_df[combined_df["season"] == year]
        season_df.to_csv(season_file, index=False)
        logger.info(f"✅ Updated {season_file.name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    # Determine years to process
    if args.end_year is None:
        years = [args.start_year]
    else:
        years = list(range(args.start_year, args.end_year + 1))

    # Find CSV files in data directory
    data_dir = Path("data")
    season_files = [data_dir / f"{year}_season_data.csv" for year in years]

    # Run feature calculation
    add_all_features(season_files)
