"""
Data fetching for F1 race prediction.

This module fetches race and qualifying data from the FastF1 API.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List

import fastf1
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import RETRY_ATTEMPTS, RETRY_DELAY

logger = logging.getLogger(__name__)

# Set log level to ERROR
fastf1.set_log_level("ERROR")

# Enable caching
fastf1.Cache.enable_cache(cache_dir="f1_cache")
logger.info("✅ Caching enabled")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Save all the data for the provided season(s)")

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


def extract_driver_data(
    race_results: pd.DataFrame,
    qualifying_results: pd.DataFrame,
    year: int,
    event_name: str,
    round_number: int,
) -> List[dict]:
    """
    Extract driver data from race and qualifying results.

    Args:
        race_results: DataFrame with race results
        qualifying_results: DataFrame with qualifying results
        year: Season year
        event_name: Name of the event
        round_number: Round number in the season

    Returns:
        List of driver data dictionaries
    """
    driver_data_list = []
    pole_time = qualifying_results["Q3"].min()

    for _, row in race_results.iterrows():
        driver = row["Abbreviation"]
        qualifying_row = qualifying_results[qualifying_results["Abbreviation"] == driver]

        if qualifying_row.empty:
            continue

        qualifying_row = qualifying_row.iloc[0]
        qualifying_gap = (qualifying_row["Q3"] - pole_time).total_seconds() if pd.notna(qualifying_row["Q3"]) else np.nan

        driver_data = {
            "season": year,
            "grand_prix": event_name,
            "round_number": round_number,
            "driver": driver,
            "team": row["TeamName"],
            "grid_position": row["GridPosition"],
            "finish_position": row["Position"],
            "win": int(row["Position"] == 1),
            "qualifying_position": qualifying_row["Position"],
            "qualifying_gap": qualifying_gap,
            "reached_q3": int(pd.notna(qualifying_row["Q3"])),
        }
        driver_data_list.append(driver_data)

    return driver_data_list


def fetch_single_race(year: int, event_name: str, round_number: int) -> List[dict]:
    """
    Fetch data for a single race with retry logic.

    Args:
        year: Season year
        event_name: Name of the event
        round_number: Round number in the season

    Returns:
        List of driver data dictionaries for this race
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            race_session = fastf1.get_session(year=year, gp=event_name, identifier="Race")
            qualifying_session = fastf1.get_session(year=year, gp=event_name, identifier="Qualifying")

            race_session.load()
            qualifying_session.load()

            race_results = race_session.results
            qualifying_results = qualifying_session.results

            if race_results is None or qualifying_results is None:
                return []

            return extract_driver_data(race_results, qualifying_results, year, event_name, round_number)

        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt + 1} failed for {event_name}: {e}")
            time.sleep(RETRY_DELAY)
            if attempt == RETRY_ATTEMPTS - 1:
                logger.warning(f"⚠️ Skipped {event_name} after {RETRY_ATTEMPTS} attempts")

    return []


def fetch_race_data(year: int) -> pd.DataFrame:
    """
    Fetch all race and qualifying data for the season.

    Args:
        year: Season year to fetch

    Returns:
        DataFrame with race and qualifying data
    """
    fetch_start_time = time.time()

    schedule = fastf1.get_event_schedule(year=year)
    races = schedule[schedule["EventFormat"] != "testing"]
    total_races = len(races)
    logger.info(f"🔍 Found {total_races} races in {year} season")

    all_data = []

    for _, race in tqdm(races.iterrows(), total=total_races, desc=f"Processing {year} season"):
        race_data = fetch_single_race(year, race["EventName"], race["RoundNumber"])
        all_data.extend(race_data)

    result_df = pd.DataFrame(all_data)
    fetch_elapsed_time = time.time() - fetch_start_time

    logger.info(f"✅ Fetched {len(result_df):,} driver results in {fetch_elapsed_time:.1f} seconds")

    return result_df


def save_data(df: pd.DataFrame, year: int, save_dir: str) -> None:
    """
    Save the dataset to CSV.

    Args:
        df: DataFrame to save
        year: Season year
        save_dir: Directory to save file
    """
    # Make sure the save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    filename = Path(save_dir) / f"{year}_season_data.csv"
    df.to_csv(filename, index=False)
    logger.info(f"💾 Successfully saved dataset to {filename}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    # Determine years to process
    if args.end_year is None:
        years = [args.start_year]
    else:
        years = list(range(args.start_year, args.end_year + 1))

    # Loop over years and save to data directory
    for year in years:
        try:
            logger.info(f"\n📅 Fetching {year} season...")
            df = fetch_race_data(year)

            if df.empty:
                logger.warning(f"⚠️  No data returned for {year}, skipping...")
                continue

            save_data(df, year, "data")
        except Exception as e:
            logger.error(f"❌ Failed to fetch {year} season: {e}")
            logger.info(f"⏭️  Skipping {year} and continuing with next year...")
            continue
