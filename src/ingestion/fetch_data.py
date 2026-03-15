"""
Live sensor data ingestion from Open-Meteo API.
Fetches hourly weather/atmospheric readings for 6 European cities
and stores in PostgreSQL.
"""

import os
import time
from datetime import datetime, timezone

import requests
import yaml
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_city(location: dict, variables: list, past_hours: int = 72) -> pd.DataFrame:
    """Fetch live hourly sensor data for one city from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":       location["latitude"],
        "longitude":      location["longitude"],
        "hourly":         ",".join(variables),
        "past_hours":     past_hours,
        "forecast_hours": 0,
        "timezone":       "UTC",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch {location['name']}: {e}")
        return pd.DataFrame()

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    rows = []
    for i, ts in enumerate(times):
        row = {
            "city":        location["name"],
            "country":     location.get("country", ""),
            "latitude":    location["latitude"],
            "longitude":   location["longitude"],
            "timestamp":   ts,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        for var in variables:
            vals = hourly.get(var, [])
            row[var] = vals[i] if i < len(vals) else None
        rows.append(row)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.info(f"{location['name']}: {len(df)} readings")
    return df


def fetch_all(config: dict) -> pd.DataFrame:
    """Fetch data for all configured cities."""
    cfg       = config["sources"]["open_meteo"]
    variables = cfg["variables"]
    past_hours = cfg["past_hours"]
    dfs       = []

    for location in cfg["locations"]:
        df = fetch_city(location, variables, past_hours)
        if not df.empty:
            dfs.append(df)
        time.sleep(0.5)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.success(f"Fetched {len(combined)} total readings from {len(dfs)} cities")
    return combined


def save_to_db(df: pd.DataFrame, config: dict) -> int:
    """Save readings to PostgreSQL, replacing existing data."""
    if df.empty:
        return 0
    engine = create_engine(config["database"]["url"])
    table  = config["database"]["table"]

    # Use replace to always have fresh data
    df.to_sql(table, engine, if_exists="replace", index=False)
    logger.success(f"Saved {len(df)} rows to {table}")
    return len(df)


def save_to_csv(df: pd.DataFrame, path: str = "outputs/sensor_readings.csv"):
    """Save to CSV as fallback when DB unavailable."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} rows to {path}")


def run_ingestion(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """Main ingestion — fetches live data, saves to DB or CSV."""
    cfg = load_config(config_path)
    df  = fetch_all(cfg)

    if df.empty:
        logger.error("No data fetched")
        return df

    # Try DB, fall back to CSV silently
    try:
        save_to_db(df, cfg)
    except Exception:
        try:
            save_to_csv(df)
        except Exception:
            pass  # On Streamlit Cloud, just return the df in memory

    return df


def load_data(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """Load sensor data — from DB if available, else fetch live from API."""
    cfg = load_config(config_path)

    # Try DB first
    try:
        engine = create_engine(cfg["database"]["url"])
        df = pd.read_sql(f"SELECT * FROM {cfg['database']['table']}", engine)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if len(df) > 0:
            logger.info(f"Loaded {len(df)} rows from DB")
            return df
    except Exception as e:
        logger.warning(f"DB unavailable: {e}")

    # Fetch live from API (always works — no DB needed)
    logger.info("Fetching live data from Open-Meteo API...")
    return fetch_all(cfg)


if __name__ == "__main__":
    df = run_ingestion()
    print(df.head())
    print(f"Shape: {df.shape}")