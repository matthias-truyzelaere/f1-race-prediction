#!/usr/bin/env python3

"""
Automatic F1 Model Update Script

Automatically:
1. Detects current year
2. Fetches latest race data
3. Re-engineers features across all seasons
4. Retrains the model

Usage: python src/update_model.py
"""

import logging
from datetime import datetime
from pathlib import Path

from feature_engineering import add_all_features
from fetch_data import fetch_race_data, save_data
from train_model import train_model as train_model_func
from utils import get_available_years

logger = logging.getLogger(__name__)


def get_current_year() -> int:
    """Get current year."""
    return datetime.now().year


def main() -> None:
    """Main update pipeline."""
    logger.info("=" * 60)
    logger.info("🏎️  F1 RACE PREDICTOR - AUTOMATIC MODEL UPDATE")
    logger.info("=" * 60)

    # Detect current year
    current_year = get_current_year()
    logger.info(f"\n📅 Current year detected: {current_year}")

    # Get available years in data
    available_years = get_available_years()

    if available_years:
        logger.info(f"📂 Existing data years: {', '.join(map(str, available_years))}")
    else:
        logger.warning("⚠️  No existing data found. Starting fresh...")

    # Step 1: Fetch current year data
    logger.info(f"\n{'=' * 60}")
    logger.info(f"📥 STEP 1: Fetching {current_year} season data...")
    logger.info("=" * 60)

    try:
        df = fetch_race_data(current_year)

        if df.empty:
            logger.warning(f"⚠️  No data available for {current_year} yet. Skipping fetch.")
        else:
            save_data(df, current_year, "data")
            logger.info(f"✅ Successfully fetched {current_year} data")
    except Exception as e:
        logger.error(f"❌ Error fetching {current_year} data: {e}")
        logger.info("⏭️  Continuing with existing data...")

    # Update available years after fetch
    available_years = get_available_years()

    if not available_years:
        logger.error("\n❌ No data available. Cannot proceed.")
        return

    # Step 2: Re-engineer features
    logger.info(f"\n{'=' * 60}")
    logger.info(f"🔧 STEP 2: Engineering features ({min(available_years)}-{max(available_years)})...")
    logger.info("=" * 60)

    try:
        data_dir = Path("data")
        season_files = [data_dir / f"{year}_season_data.csv" for year in available_years]

        # Only process files that exist
        season_files = [f for f in season_files if f.exists()]

        if season_files:
            add_all_features(season_files)
            logger.info("✅ Feature engineering completed")
        else:
            logger.error("❌ No season files found to process")
            return
    except Exception as e:
        logger.error(f"❌ Error during feature engineering: {e}")
        return

    # Step 3: Retrain model
    logger.info(f"\n{'=' * 60}")
    logger.info("🤖 STEP 3: Retraining model...")
    logger.info("=" * 60)

    try:
        train_model_func()
        logger.info("\n✅ Model retrained successfully")
    except Exception as e:
        logger.error(f"❌ Error during model training: {e}")
        return

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("✅ UPDATE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"📊 Data years: {min(available_years)}-{max(available_years)}")
    logger.info("🤖 Model ready for predictions")
    logger.info("🚀 Launch app: streamlit run src/app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
