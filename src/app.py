import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
import streamlit as st

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import DNF_THRESHOLD, F1_POINTS_SYSTEM, PODIUM_POSITION
from utils import load_all_data as load_all_data_util

# Configure logging for Streamlit app
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Set log level to ERROR
fastf1.set_log_level("ERROR")

# Create cache directory if it doesn't exist and enable caching
cache_dir = Path("f1_cache")
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(cache_dir=str(cache_dir))
logger.info("✅ Caching enabled")


@st.cache_resource
def load_model(model_path: str = "models/f1_model.pkl"):
    """
    Load trained model and feature columns from pickle file.

    Args:
        model_path: Path to the pickled model file

    Returns:
        Tuple of (model, feature_columns)

    Raises:
        FileNotFoundError: If model file does not exist
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. The model needs to be trained before using the app.")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["feature_columns"]


@st.cache_data
def load_historical_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load all historical race data using shared utility.

    Args:
        data_dir: Directory containing season CSV files

    Returns:
        Combined DataFrame with all historical race data
    """
    return load_all_data_util(data_dir)


@st.cache_data(ttl=3600)
def get_upcoming_race():
    """
    Get the next upcoming race or most recent race.

    Returns:
        Tuple of (next_race, current_year, is_upcoming)
    """
    current_year = datetime.now().year
    schedule = fastf1.get_event_schedule(current_year)

    # Filter out testing events
    races = schedule[schedule["EventFormat"] != "testing"]

    # Find next race (or most recent if season ended)
    now = datetime.now()
    # Convert EventDate to datetime for comparison
    event_dates = pd.to_datetime(races["EventDate"]).dt.tz_localize(None)
    upcoming = races[event_dates > now]

    if len(upcoming) > 0:
        next_race = upcoming.iloc[0]
        is_upcoming = True
    else:
        # Season ended, use last race
        next_race = races.iloc[-1]
        is_upcoming = False

    return next_race, current_year, is_upcoming


def check_race_has_happened(year: int, round_number: int) -> bool:
    """
    Check if a race has already happened based on the event date.

    Args:
        year: Season year
        round_number: Race round number

    Returns:
        True if race date has passed, False otherwise
    """
    try:
        schedule = fastf1.get_event_schedule(year)
        race_event = schedule[schedule["RoundNumber"] == round_number].iloc[0]
        race_date = pd.to_datetime(race_event["EventDate"])
        return race_date <= pd.Timestamp.now()
    except Exception:
        return True  # Assume race might have happened if we can't determine


def extract_drivers_from_session(year: int, round_number: int) -> tuple[dict, set]:
    """
    Extract driver names and teams from a race session.

    Args:
        year: Season year
        round_number: Race round number

    Returns:
        Tuple of (driver_name_map, teams_set)
    """
    driver_name_map = {}
    teams = set()

    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=False, weather=False, messages=False)
    results = session.results

    if len(results) == 0:
        return driver_name_map, teams

    for _, driver_row in results.iterrows():
        abbr = driver_row["Abbreviation"]
        full_name = driver_row["FullName"]
        team_name = driver_row["TeamName"]

        if pd.notna(abbr):
            if pd.notna(full_name):
                driver_name_map[abbr] = f"{full_name} ({abbr})"
            else:
                driver_name_map[abbr] = f"{abbr} (Full name unavailable)"

            if pd.notna(team_name):
                teams.add(team_name)

    return driver_name_map, teams


def lookup_driver_full_names(driver_codes: list, season: int) -> dict:
    """
    Look up full names for driver codes from recent race sessions.

    Args:
        driver_codes: List of driver abbreviation codes
        season: Season to search for names

    Returns:
        Dictionary mapping driver codes to display names
    """
    driver_name_map = {}

    try:
        schedule = fastf1.get_event_schedule(season)
        races = schedule[schedule["EventFormat"] != "testing"]

        for _, race in races.iloc[::-1].iterrows():
            try:
                session = fastf1.get_session(season, race["RoundNumber"], "R")
                session.load(telemetry=False, weather=False, messages=False)

                for _, driver_row in session.results.iterrows():
                    abbr = driver_row["Abbreviation"]
                    full_name = driver_row["FullName"]
                    if pd.notna(abbr) and pd.notna(full_name) and abbr in driver_codes:
                        driver_name_map[abbr] = f"{full_name} ({abbr})"

                if len(driver_name_map) == len(driver_codes):
                    break
            except Exception:
                continue
    except Exception:
        pass

    # Fill in any remaining codes without full names
    for code in driver_codes:
        if code not in driver_name_map:
            driver_name_map[code] = f"{code} (Full name unavailable)"

    return driver_name_map


@st.cache_data(ttl=3600)
def get_race_entry_list(year: int, round_number: int, historical_df: pd.DataFrame):
    """
    Get actual driver entry list for the race.

    Args:
        year: Season year
        round_number: Race round number
        historical_df: Historical race data DataFrame

    Returns:
        Tuple of (driver_name_map, available_teams)
    """
    driver_name_map = {}
    available_teams = set()

    # Try to get drivers from actual race session
    if check_race_has_happened(year, round_number):
        try:
            driver_name_map, available_teams = extract_drivers_from_session(year, round_number)
        except Exception:
            pass

    # Fall back to historical data if no drivers found
    if not driver_name_map:
        st.info("Using historical driver list (race session not yet available)")

        latest_season_data = historical_df[historical_df["season"] == year]
        if len(latest_season_data) == 0:
            latest_season_data = historical_df[historical_df["season"] == historical_df["season"].max()]

        driver_codes = latest_season_data["driver"].unique().tolist()
        available_teams = set(latest_season_data["team"].unique().tolist())
        actual_season = int(latest_season_data["season"].iloc[0])

        driver_name_map = lookup_driver_full_names(driver_codes, actual_season)

    # Add teams from most recent season
    most_recent_season = historical_df["season"].max()
    recent_teams = historical_df[historical_df["season"] == most_recent_season]["team"].unique().tolist()
    available_teams.update(recent_teams)

    return driver_name_map, sorted(list(available_teams))


def calculate_rolling_features(driver_history: pd.DataFrame) -> dict:
    """
    Calculate rolling performance features for a driver.

    Args:
        driver_history: DataFrame with driver's historical race data

    Returns:
        Dictionary with rolling feature values
    """
    if len(driver_history) == 0:
        return {
            "avg_finish_last_3": np.nan,
            "avg_quali_last_3": np.nan,
            "podium_rate_last_5": np.nan,
            "dnf_rate_last_5": np.nan,
            "avg_position_gain_last_3": np.nan,
            "points_last_3": np.nan,
        }

    recent_finishes = driver_history["finish_position"].dropna().tolist()
    recent_quali = driver_history["qualifying_position"].dropna().tolist()

    avg_finish_last_3 = np.mean(recent_finishes[-3:]) if len(recent_finishes) >= 1 else np.nan
    avg_quali_last_3 = np.mean(recent_quali[-3:]) if len(recent_quali) >= 1 else np.nan

    if len(recent_finishes) >= 1:
        recent_5 = recent_finishes[-5:]
        podium_rate_last_5 = sum(1 for p in recent_5 if p <= PODIUM_POSITION) / len(recent_5)
        dnf_rate_last_5 = sum(1 for p in recent_5 if p > DNF_THRESHOLD) / len(recent_5)
    else:
        podium_rate_last_5 = np.nan
        dnf_rate_last_5 = np.nan

    position_gains = []
    for _, row in driver_history.iterrows():
        if pd.notna(row["qualifying_position"]) and pd.notna(row["finish_position"]):
            gain = row["qualifying_position"] - row["finish_position"]
            position_gains.append(gain)

    avg_position_gain_last_3 = np.mean(position_gains[-3:]) if len(position_gains) >= 1 else np.nan

    recent_points = [F1_POINTS_SYSTEM.get(int(p), 0) for p in recent_finishes[-3:] if pd.notna(p)]
    points_last_3 = sum(recent_points) if recent_points else np.nan

    return {
        "avg_finish_last_3": avg_finish_last_3,
        "avg_quali_last_3": avg_quali_last_3,
        "podium_rate_last_5": podium_rate_last_5,
        "dnf_rate_last_5": dnf_rate_last_5,
        "avg_position_gain_last_3": avg_position_gain_last_3,
        "points_last_3": points_last_3,
    }


def calculate_team_features(
    historical_df: pd.DataFrame, team_name: str, current_season: int, current_round: int, grand_prix: str
) -> dict:
    """
    Calculate team performance features.

    Args:
        historical_df: Historical race data DataFrame
        team_name: Team name
        current_season: Current season year
        current_round: Current round number
        grand_prix: Grand Prix name

    Returns:
        Dictionary with team feature values
    """
    current_season_data = historical_df[
        (historical_df["season"] == current_season) & (historical_df["round_number"] < current_round)
    ]

    team_current_season = current_season_data[current_season_data["team"] == team_name]

    team_avg_finish_season = team_current_season["finish_position"].mean() if len(team_current_season) > 0 else np.nan
    team_avg_quali_season = team_current_season["qualifying_position"].mean() if len(team_current_season) > 0 else np.nan

    team_circuit_history = historical_df[
        (historical_df["team"] == team_name)
        & (historical_df["grand_prix"] == grand_prix)
        & (historical_df["season"] < current_season)
    ]
    team_circuit_avg = team_circuit_history["finish_position"].mean() if len(team_circuit_history) > 0 else np.nan

    return {
        "team_avg_finish_season": team_avg_finish_season,
        "team_avg_quali_season": team_avg_quali_season,
        "team_circuit_avg": team_circuit_avg,
    }


def calculate_championship_position(
    historical_df: pd.DataFrame, driver_code: str, current_season: int, current_round: int
) -> float:
    """
    Calculate driver's championship position before this race.

    Args:
        historical_df: Historical race data DataFrame
        driver_code: Driver abbreviation code
        current_season: Current season year
        current_round: Current round number

    Returns:
        Championship position or NaN if unavailable
    """
    current_season_data = historical_df[
        (historical_df["season"] == current_season) & (historical_df["round_number"] < current_round)
    ]

    if len(current_season_data) == 0:
        return np.nan

    driver_points = {}
    for _, row in current_season_data.iterrows():
        driver = row["driver"]
        points = F1_POINTS_SYSTEM.get(int(row["finish_position"]), 0) if pd.notna(row["finish_position"]) else 0
        driver_points[driver] = driver_points.get(driver, 0) + points

    if driver_code not in driver_points:
        return np.nan

    sorted_drivers = sorted(driver_points.items(), key=lambda x: x[1], reverse=True)
    return next((i + 1 for i, (d, _) in enumerate(sorted_drivers) if d == driver_code), np.nan)


def calculate_driver_circuit_avg(
    historical_df: pd.DataFrame, driver_code: str, grand_prix: str, current_season: int
) -> float:
    """
    Calculate driver's average finish at this circuit in previous years.

    Args:
        historical_df: Historical race data DataFrame
        driver_code: Driver abbreviation code
        grand_prix: Grand Prix name
        current_season: Current season year

    Returns:
        Average finish position or NaN if no history
    """
    driver_circuit_history = historical_df[
        (historical_df["driver"] == driver_code)
        & (historical_df["grand_prix"] == grand_prix)
        & (historical_df["season"] < current_season)
    ]
    return driver_circuit_history["finish_position"].mean() if len(driver_circuit_history) > 0 else np.nan


def calculate_driver_features(
    driver_code: str,
    team_name: str,
    quali_pos: int,
    grid_pos: int,
    quali_gap: float,
    reached_q3: int,
    historical_df: pd.DataFrame,
    current_season: int,
    current_round: int,
    grand_prix: str,
) -> dict:
    """
    Calculate all features for a single driver.

    Args:
        driver_code: Driver abbreviation code
        team_name: Team name
        quali_pos: Qualifying position
        grid_pos: Grid position
        quali_gap: Time gap to pole in seconds
        reached_q3: Whether driver reached Q3 (1 or 0)
        historical_df: Historical race data DataFrame
        current_season: Current season year
        current_round: Current round number
        grand_prix: Grand Prix name

    Returns:
        Dictionary of calculated features
    """
    driver_history = historical_df[historical_df["driver"] == driver_code].copy()

    rolling_features = calculate_rolling_features(driver_history)
    team_features = calculate_team_features(historical_df, team_name, current_season, current_round, grand_prix)
    championship_position = calculate_championship_position(historical_df, driver_code, current_season, current_round)
    driver_circuit_avg = calculate_driver_circuit_avg(historical_df, driver_code, grand_prix, current_season)
    has_grid_penalty = 1 if grid_pos != quali_pos else 0

    return {
        "grid_position": grid_pos,
        "qualifying_position": quali_pos,
        "qualifying_gap": quali_gap,
        "reached_q3": reached_q3,
        "has_grid_penalty": has_grid_penalty,
        **rolling_features,
        **team_features,
        "driver_championship_position": championship_position,
        "driver_circuit_avg": driver_circuit_avg,
    }


def render_driver_input(driver_index: int, available_drivers: list, available_teams: list) -> None:
    """
    Render input fields for a single driver.

    Args:
        driver_index: Index of the driver (0-based)
        available_drivers: List of available driver names
        available_teams: List of available team names
    """
    st.markdown(f"**Driver {driver_index + 1}**")

    is_dnf_in_qualifying = st.session_state.get(f"dnf_quali_{driver_index}", False)

    col1, col2 = st.columns(2)

    with col1:
        st.selectbox(
            "Driver",
            options=available_drivers,
            index=min(driver_index, len(available_drivers) - 1),
            key=f"driver_{driver_index}",
        )
        st.selectbox(
            "Team",
            options=available_teams,
            index=min(driver_index, len(available_teams) - 1),
            key=f"team_{driver_index}",
        )
        st.number_input(
            "Qualifying Position",
            min_value=1,
            max_value=22,
            value=20 if is_dnf_in_qualifying else driver_index + 1,
            key=f"quali_{driver_index}",
            disabled=is_dnf_in_qualifying,
            help="Auto-set to P20 if DNF in Qualifying" if is_dnf_in_qualifying else None,
        )

    with col2:
        st.number_input(
            "Grid Position",
            min_value=1,
            max_value=22,
            value=20 if is_dnf_in_qualifying else driver_index + 1,
            key=f"grid_{driver_index}",
            disabled=is_dnf_in_qualifying,
            help="Auto-set to P20 if DNF in Qualifying" if is_dnf_in_qualifying else None,
        )
        st.number_input(
            "Qualifying Gap (seconds)",
            min_value=0.0,
            value=5.0 if is_dnf_in_qualifying else float(driver_index) * 0.1,
            step=0.001,
            format="%.3f",
            key=f"gap_{driver_index}",
            disabled=is_dnf_in_qualifying,
            help="Auto-set to 5.0s if DNF in Qualifying" if is_dnf_in_qualifying else None,
        )
        st.checkbox(
            "Reached Q3",
            value=False if is_dnf_in_qualifying else (driver_index < 10),
            key=f"q3_{driver_index}",
            disabled=is_dnf_in_qualifying,
        )
        st.checkbox(
            "DNF in Qualifying",
            value=False,
            key=f"dnf_quali_{driver_index}",
            help="Driver crashed/retired in qualifying and didn't set a time. This will auto-fill other fields.",
        )

    st.markdown("---")


def extract_driver_code_from_display_name(display_name: str) -> str:
    """
    Extract driver abbreviation code from display name.

    Args:
        display_name: Display name like "Max Verstappen (VER)"

    Returns:
        Driver abbreviation code (e.g., "VER")
    """
    if display_name and "(" in display_name and ")" in display_name:
        return display_name.split("(")[1].split(")")[0]
    return display_name if display_name else ""


def collect_driver_form_data(driver_index: int) -> dict:
    """
    Collect form data for a single driver from session state.

    Args:
        driver_index: Index of the driver (0-based)

    Returns:
        Dictionary with driver form data
    """
    driver_display_name = st.session_state.get(f"driver_{driver_index}")
    team = st.session_state.get(f"team_{driver_index}")
    is_dnf_in_qualifying = st.session_state.get(f"dnf_quali_{driver_index}", False)

    quali_pos = st.session_state.get(f"quali_{driver_index}", driver_index + 1)
    grid_pos = st.session_state.get(f"grid_{driver_index}", driver_index + 1)
    quali_gap = st.session_state.get(f"gap_{driver_index}", float(driver_index) * 0.1)
    reached_q3 = st.session_state.get(f"q3_{driver_index}", driver_index < 10)

    driver_abbr = extract_driver_code_from_display_name(driver_display_name)

    if is_dnf_in_qualifying:
        quali_pos = 20
        grid_pos = 20
        quali_gap = 5.0
        reached_q3 = False

    return {
        "driver": driver_abbr,
        "team": team,
        "qualifying_position": quali_pos,
        "grid_position": grid_pos,
        "qualifying_gap": quali_gap,
        "reached_q3": int(reached_q3),
        "dnf_in_quali": is_dnf_in_qualifying,
    }


def display_predictions_table(predictions_df: pd.DataFrame) -> None:
    """
    Display the predictions results table.

    Args:
        predictions_df: DataFrame with prediction results
    """
    st.dataframe(
        predictions_df[["Predicted Position", "Driver", "Team", "Grid Position", "Position Change"]],
        hide_index=True,
        width="stretch",
    )


def display_podium(predictions_df: pd.DataFrame) -> None:
    """
    Display the podium prediction.

    Args:
        predictions_df: DataFrame with prediction results (sorted by predicted position)
    """
    st.markdown("### Podium Prediction")
    podium = predictions_df.head(3)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("P1", podium.iloc[0]["Driver"], f"{podium.iloc[0]['Team']}")
    with col2:
        st.metric("P2", podium.iloc[1]["Driver"], f"{podium.iloc[1]['Team']}")
    with col3:
        st.metric("P3", podium.iloc[2]["Driver"], f"{podium.iloc[2]['Team']}")


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")

    try:
        next_race, current_year, is_upcoming = get_upcoming_race()
        race_name = next_race["EventName"]
        round_number = next_race["RoundNumber"]
        race_date = next_race["EventDate"].strftime("%B %d, %Y")

        title_suffix = "" if is_upcoming else " (Season Ended)"
        st.title(f"🔮 F1 Race Prediction: {race_name}{title_suffix}")
        st.subheader(f"Round {round_number} - {race_date}")
    except Exception as e:
        st.title("F1 Race Prediction")
        st.error(f"Could not fetch upcoming race: {e}")
        return

    try:
        model, feature_cols = load_model()
        st.success("Model loaded successfully")
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.info("Please train a model first by running: python src/train_model.py")
        return

    try:
        historical_df = load_historical_data()
    except Exception as e:
        st.error(f"Could not load historical data: {e}")
        return

    st.markdown("---")

    driver_name_map, available_teams = get_race_entry_list(current_year, round_number, historical_df)
    available_drivers = sorted(driver_name_map.values())

    if not available_drivers:
        st.error("No drivers found. Please ensure historical data is available.")
        return

    st.header("Enter Qualifying Results")
    st.info(
        "**Grid Position vs Qualifying Position:**\n"
        "- **Qualifying Position**: Where driver qualified in Q3/Q2/Q1 (e.g., P1, P2, P3...)\n"
        "- **Grid Position**: Actual starting position on race day (may differ due to penalties)\n"
        "- **Qualifying Gap**: Time gap to pole position in seconds\n\n"
        f"**Tip**: Get real data from [F1 official website](https://www.formula1.com) "
        f"- Results - {current_year} Season - Qualifying"
    )

    num_drivers = st.number_input("Number of drivers", min_value=1, max_value=20, value=20)

    if "drivers_data" not in st.session_state:
        st.session_state.drivers_data = []

    cols = st.columns(2)
    for i in range(num_drivers):
        with cols[i % 2]:
            render_driver_input(i, available_drivers, available_teams)

    submitted = st.button("Predict Race Results", use_container_width=True, type="primary")

    if submitted:
        drivers_data = [collect_driver_form_data(i) for i in range(num_drivers)]
        st.session_state.drivers_data = drivers_data

    if submitted and st.session_state.drivers_data:
        st.markdown("---")
        st.header("Predicted Race Results")

        predictions = []
        with st.spinner("Calculating predictions..."):
            for driver_data in st.session_state.drivers_data:
                features = calculate_driver_features(
                    driver_code=driver_data["driver"],
                    team_name=driver_data["team"],
                    quali_pos=driver_data["qualifying_position"],
                    grid_pos=driver_data["grid_position"],
                    quali_gap=driver_data["qualifying_gap"],
                    reached_q3=driver_data["reached_q3"],
                    historical_df=historical_df,
                    current_season=current_year,
                    current_round=round_number,
                    grand_prix=race_name,
                )

                feature_vector = pd.DataFrame([features])[feature_cols]
                predicted_position = model.predict(feature_vector)[0]

                driver_code_abbr = driver_data["driver"]
                driver_display = driver_name_map.get(driver_code_abbr, driver_code_abbr)

                if driver_data["dnf_in_quali"]:
                    driver_display = f"(DNF) {driver_display}"

                predictions.append(
                    {
                        "Driver": driver_display,
                        "Team": driver_data["team"],
                        "Grid Position": driver_data["grid_position"],
                        "Predicted Finish": predicted_position,
                    }
                )

        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values("Predicted Finish")
        predictions_df["Predicted Position"] = range(1, len(predictions_df) + 1)
        predictions_df["Position Change"] = predictions_df["Grid Position"] - predictions_df["Predicted Position"]

        display_predictions_table(predictions_df)

        if any(d["dnf_in_quali"] for d in st.session_state.drivers_data):
            st.info("**(DNF) = Driver DNF'd in qualifying** (crashed/mechanical failure, starting from back)")

        display_podium(predictions_df)


if __name__ == "__main__":
    main()
