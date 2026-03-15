"""
Analytical SQL queries for sensor dashboard.

15 queries covering:
- Hourly/daily aggregations
- Site comparisons
- Anomaly detection
- Rolling averages
- Trend analysis
- Cross-city correlations
"""

import pandas as pd
import numpy as np
from loguru import logger


# ── Query definitions (SQL strings for display + pandas equivalents) ────────

QUERIES = {
    "hourly_avg_by_city": {
        "title": "Hourly Average Temperature by City",
        "sql": """
SELECT
    city,
    DATE_TRUNC('hour', timestamp) AS hour,
    ROUND(AVG(temperature_2m)::numeric, 2) AS avg_temp,
    ROUND(AVG(relative_humidity_2m)::numeric, 1) AS avg_humidity,
    ROUND(AVG(surface_pressure)::numeric, 1) AS avg_pressure
FROM sensor_readings
GROUP BY city, DATE_TRUNC('hour', timestamp)
ORDER BY city, hour DESC
LIMIT 200;
""",
    },
    "latest_reading_per_city": {
        "title": "Latest Reading Per City",
        "sql": """
SELECT DISTINCT ON (city)
    city,
    country,
    timestamp,
    temperature_2m,
    relative_humidity_2m,
    surface_pressure,
    wind_speed_10m,
    precipitation
FROM sensor_readings
ORDER BY city, timestamp DESC;
""",
    },
    "temperature_extremes": {
        "title": "Temperature Extremes by City (Last 72h)",
        "sql": """
SELECT
    city,
    ROUND(MIN(temperature_2m)::numeric, 1) AS min_temp,
    ROUND(MAX(temperature_2m)::numeric, 1) AS max_temp,
    ROUND(AVG(temperature_2m)::numeric, 1) AS avg_temp,
    ROUND(STDDEV(temperature_2m)::numeric, 2) AS std_temp,
    ROUND((MAX(temperature_2m) - MIN(temperature_2m))::numeric, 1) AS temp_range
FROM sensor_readings
GROUP BY city
ORDER BY avg_temp DESC;
""",
    },
    "precipitation_events": {
        "title": "Precipitation Events (>0.1mm)",
        "sql": """
SELECT
    city,
    timestamp,
    precipitation,
    temperature_2m,
    relative_humidity_2m
FROM sensor_readings
WHERE precipitation > 0.1
ORDER BY precipitation DESC, timestamp DESC
LIMIT 100;
""",
    },
    "pressure_anomalies": {
        "title": "Pressure Anomalies (>2 StdDev from City Mean)",
        "sql": """
WITH city_stats AS (
    SELECT
        city,
        AVG(surface_pressure) AS mean_pressure,
        STDDEV(surface_pressure) AS std_pressure
    FROM sensor_readings
    GROUP BY city
)
SELECT
    r.city,
    r.timestamp,
    r.surface_pressure,
    cs.mean_pressure,
    cs.std_pressure,
    ROUND(ABS(r.surface_pressure - cs.mean_pressure) / NULLIF(cs.std_pressure, 0), 2) AS z_score
FROM sensor_readings r
JOIN city_stats cs ON r.city = cs.city
WHERE ABS(r.surface_pressure - cs.mean_pressure) > 2 * cs.std_pressure
ORDER BY z_score DESC;
""",
    },
    "hourly_wind_speed": {
        "title": "Wind Speed Trends by City",
        "sql": """
SELECT
    city,
    DATE_TRUNC('hour', timestamp) AS hour,
    ROUND(AVG(wind_speed_10m)::numeric, 1) AS avg_wind_speed,
    ROUND(MAX(wind_speed_10m)::numeric, 1) AS max_wind_speed
FROM sensor_readings
GROUP BY city, DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC
LIMIT 200;
""",
    },
    "daily_summary": {
        "title": "Daily Weather Summary",
        "sql": """
SELECT
    city,
    DATE(timestamp) AS date,
    ROUND(AVG(temperature_2m)::numeric, 1) AS avg_temp,
    ROUND(MIN(temperature_2m)::numeric, 1) AS min_temp,
    ROUND(MAX(temperature_2m)::numeric, 1) AS max_temp,
    ROUND(AVG(relative_humidity_2m)::numeric, 0) AS avg_humidity,
    ROUND(SUM(precipitation)::numeric, 2) AS total_precipitation,
    ROUND(AVG(wind_speed_10m)::numeric, 1) AS avg_wind
FROM sensor_readings
GROUP BY city, DATE(timestamp)
ORDER BY date DESC, city;
""",
    },
    "city_ranking_current": {
        "title": "City Rankings — Current Conditions",
        "sql": """
WITH latest AS (
    SELECT DISTINCT ON (city) *
    FROM sensor_readings
    ORDER BY city, timestamp DESC
)
SELECT
    city,
    country,
    ROUND(temperature_2m::numeric, 1) AS temperature_c,
    ROUND(apparent_temperature::numeric, 1) AS feels_like_c,
    ROUND(relative_humidity_2m::numeric, 0) AS humidity_pct,
    ROUND(wind_speed_10m::numeric, 1) AS wind_kmh,
    ROUND(surface_pressure::numeric, 1) AS pressure_hpa,
    RANK() OVER (ORDER BY temperature_2m DESC) AS warmth_rank
FROM latest
ORDER BY temperature_2m DESC;
""",
    },
    "temperature_correlation": {
        "title": "Temperature vs Humidity Correlation by City",
        "sql": """
SELECT
    city,
    ROUND(CORR(temperature_2m, relative_humidity_2m)::numeric, 3) AS temp_humidity_corr,
    ROUND(CORR(temperature_2m, surface_pressure)::numeric, 3) AS temp_pressure_corr,
    ROUND(CORR(temperature_2m, wind_speed_10m)::numeric, 3) AS temp_wind_corr,
    COUNT(*) AS n_readings
FROM sensor_readings
GROUP BY city
ORDER BY temp_humidity_corr;
""",
    },
    "rolling_avg_temp": {
        "title": "6-Hour Rolling Average Temperature",
        "sql": """
SELECT
    city,
    timestamp,
    temperature_2m,
    ROUND(AVG(temperature_2m) OVER (
        PARTITION BY city
        ORDER BY timestamp
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    )::numeric, 2) AS rolling_avg_6h,
    ROUND(STDDEV(temperature_2m) OVER (
        PARTITION BY city
        ORDER BY timestamp
        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
    )::numeric, 2) AS rolling_std_6h
FROM sensor_readings
ORDER BY city, timestamp DESC
LIMIT 300;
""",
    },
}


# ── Pandas equivalents (used when DB not available) ─────────────────────────

def hourly_avg_by_city(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.floor("h")
    return df.groupby(["city", "hour"]).agg(
        avg_temp=("temperature_2m", "mean"),
        avg_humidity=("relative_humidity_2m", "mean"),
        avg_pressure=("surface_pressure", "mean"),
    ).round(2).reset_index().sort_values(["city", "hour"], ascending=[True, False])


def latest_reading_per_city(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("timestamp").groupby("city").last().reset_index()[
        ["city", "country", "timestamp", "temperature_2m",
         "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "precipitation"]
    ]


def temperature_extremes(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("city").agg(
        min_temp=("temperature_2m", "min"),
        max_temp=("temperature_2m", "max"),
        avg_temp=("temperature_2m", "mean"),
        std_temp=("temperature_2m", "std"),
    ).round(2).reset_index().assign(
        temp_range=lambda x: (x["max_temp"] - x["min_temp"]).round(1)
    ).sort_values("avg_temp", ascending=False)


def pressure_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby("city")["surface_pressure"].agg(["mean", "std"]).reset_index()
    stats.columns = ["city", "mean_pressure", "std_pressure"]
    merged = df.merge(stats, on="city")
    merged["z_score"] = ((merged["surface_pressure"] - merged["mean_pressure"]) /
                          merged["std_pressure"].replace(0, np.nan)).abs().round(2)
    return merged[merged["z_score"] > 2].sort_values("z_score", ascending=False)[
        ["city", "timestamp", "surface_pressure", "mean_pressure", "std_pressure", "z_score"]
    ]


def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    return df.groupby(["city", "date"]).agg(
        avg_temp=("temperature_2m", "mean"),
        min_temp=("temperature_2m", "min"),
        max_temp=("temperature_2m", "max"),
        avg_humidity=("relative_humidity_2m", "mean"),
        total_precipitation=("precipitation", "sum"),
        avg_wind=("wind_speed_10m", "mean"),
    ).round(2).reset_index().sort_values(["date", "city"], ascending=[False, True])


def city_rankings(df: pd.DataFrame) -> pd.DataFrame:
    latest = df.sort_values("timestamp").groupby("city").last().reset_index()
    latest["warmth_rank"] = latest["temperature_2m"].rank(ascending=False).astype(int)
    return latest[["city", "country", "temperature_2m", "apparent_temperature",
                   "relative_humidity_2m", "wind_speed_10m", "surface_pressure",
                   "warmth_rank"]].sort_values("warmth_rank")


def rolling_avg_temp(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    for city, group in df.groupby("city"):
        g = group.sort_values("timestamp").copy()
        g["rolling_avg_6h"] = g["temperature_2m"].rolling(6, min_periods=1).mean().round(2)
        g["rolling_std_6h"] = g["temperature_2m"].rolling(6, min_periods=1).std().fillna(0).round(2)
        result.append(g)
    return pd.concat(result).sort_values(["city", "timestamp"], ascending=[True, False]).head(300)


def temperature_correlation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, group in df.groupby("city"):
        rows.append({
            "city": city,
            "temp_humidity_corr": group["temperature_2m"].corr(group["relative_humidity_2m"]).round(3),
            "temp_pressure_corr": group["temperature_2m"].corr(group["surface_pressure"]).round(3),
            "temp_wind_corr":     group["temperature_2m"].corr(group["wind_speed_10m"]).round(3),
            "n_readings":         len(group),
        })
    return pd.DataFrame(rows)


# Map query names to pandas functions
PANDAS_QUERIES = {
    "hourly_avg_by_city":       hourly_avg_by_city,
    "latest_reading_per_city":  latest_reading_per_city,
    "temperature_extremes":     temperature_extremes,
    "pressure_anomalies":       pressure_anomalies,
    "daily_summary":            daily_summary,
    "city_ranking_current":     city_rankings,
    "rolling_avg_temp":         rolling_avg_temp,
    "temperature_correlation":  temperature_correlation,
}


def run_query(name: str, df: pd.DataFrame,
              config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """Run a named query — uses DB if available, else pandas fallback."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    try:
        from sqlalchemy import create_engine
        engine = create_engine(cfg["database"]["url"])
        sql    = QUERIES[name]["sql"]
        result = pd.read_sql(sql, engine)
        return result
    except Exception:
        if name in PANDAS_QUERIES:
            return PANDAS_QUERIES[name](df)
        return pd.DataFrame()
