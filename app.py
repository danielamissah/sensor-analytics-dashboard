"""
Live Sensor Analytics Dashboard
================================
Real-time weather analytics across 6 European cities.
Data: Open-Meteo Archive API — no auth required.

Tabs:
    1. Trends       — temperature, humidity, pressure time series
    2. Rankings     — current conditions comparison
    3. Anomalies    — statistical outlier detection (2-sigma)
    4. Correlations — cross-sensor Pearson correlations
    5. Export       — CSV and Excel download
"""

import io
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Live Sensor Analytics",
    layout="wide",
)


# ── Constants ─────────────────────────────────────────────────────────────────

LOCATIONS = [
    {"name": "Munich",    "latitude": 48.1351, "longitude": 11.5820, "country": "Germany"},
    {"name": "Berlin",    "latitude": 52.5200, "longitude": 13.4050, "country": "Germany"},
    {"name": "Hamburg",   "latitude": 53.5511, "longitude":  9.9937, "country": "Germany"},
    {"name": "Frankfurt", "latitude": 50.1109, "longitude":  8.6821, "country": "Germany"},
    {"name": "London",    "latitude": 51.5074, "longitude": -0.1278, "country": "UK"},
    {"name": "Paris",     "latitude": 48.8566, "longitude":  2.3522, "country": "France"},
]

CITY_COLORS = {
    "Munich":    "#2E75B6",
    "Berlin":    "#E74C3C",
    "Hamburg":   "#27AE60",
    "Frankfurt": "#F39C12",
    "London":    "#8E44AD",
    "Paris":     "#16A085",
}

ARCHIVE_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_city_archive(loc: dict, days: int = 3) -> pd.DataFrame:
    """
    Fetch historical hourly weather data for one city from the
    Open-Meteo Archive API. Returns an empty DataFrame on failure.
    """
    try:
        end_date   = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days)

        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":   loc["latitude"],
                "longitude":  loc["longitude"],
                "start_date": start_date.isoformat(),
                "end_date":   end_date.isoformat(),
                "hourly":     ",".join(ARCHIVE_VARIABLES),
                "timezone":   "UTC",
            },
            headers={"User-Agent": "SensorAnalyticsDashboard/1.0"},
            timeout=30,
        )
        response.raise_for_status()

        hourly = response.json().get("hourly", {})
        times  = hourly.get("time", [])
        if not times:
            return pd.DataFrame()

        rows = []
        for i, ts in enumerate(times):
            row = {"city": loc["name"], "country": loc["country"], "timestamp": ts}
            for var in ARCHIVE_VARIABLES:
                values  = hourly.get(var, [])
                row[var] = values[i] if i < len(values) else None
            rows.append(row)

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    except Exception:
        return pd.DataFrame()


def load_sample_data() -> pd.DataFrame:
    """
    Load pre-committed sample data as fallback when the API is unavailable.
    Timestamps are shifted to align with the current time.
    """
    try:
        df     = pd.read_csv("sample_data.csv", parse_dates=["timestamp"])
        max_ts = pd.to_datetime(df["timestamp"], utc=True).max()
        delta  = pd.Timestamp.now(tz="UTC") - max_ts
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True) + delta
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_data() -> tuple:
    """
    Fetch hourly sensor data for all cities from Open-Meteo Archive API.
    Falls back to sample data if the API is unavailable.

    Returns:
        (DataFrame, is_live) where is_live=True means data came from the API.
    """
    dfs = []
    for loc in LOCATIONS:
        df = fetch_city_archive(loc, days=3)
        if not df.empty:
            dfs.append(df)
        time.sleep(0.5)

    if dfs:
        return pd.concat(dfs, ignore_index=True), True

    return load_sample_data(), False


# ── Helper functions ──────────────────────────────────────────────────────────

def strip_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Remove timezone info from all datetime columns (required for Excel export)."""
    result = df.copy()
    for col in result.select_dtypes(include=["datetimetz"]).columns:
        result[col] = result[col].dt.tz_localize(None)
    return result


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag temperature readings that deviate more than 2 standard deviations
    from the 6-hour rolling mean within each city.
    """
    result = df.copy()
    for city, group in result.groupby("city"):
        rolling_mean = group["temperature_2m"].rolling(6, min_periods=1).mean()
        rolling_std  = group["temperature_2m"].rolling(6, min_periods=1).std().fillna(0)
        result.loc[group.index, "rolling_mean"] = rolling_mean
        result.loc[group.index, "rolling_std"]  = rolling_std
    result["is_anomaly"] = (
        (result["temperature_2m"] - result["rolling_mean"]).abs()
        > 2 * result["rolling_std"]
    )
    return result


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Sensor Analytics")
    st.caption("Historical weather data — Open-Meteo Archive API")
    st.divider()

    if st.button("Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    hours_back = st.slider(
        label="Hours back",
        min_value=6,
        max_value=72,
        value=48,
        step=6,
    )
    selected_cities = st.multiselect(
        label="Cities",
        options=[loc["name"] for loc in LOCATIONS],
        default=[loc["name"] for loc in LOCATIONS],
    )

    st.divider()
    st.caption("**Stack:** Open-Meteo Archive API · PostgreSQL · Streamlit · Plotly")
    st.caption(
        "**GitHub:** [sensor-analytics-dashboard]"
        "(https://github.com/danielamissah/sensor-analytics-dashboard)"
    )


# ── Load and filter data ──────────────────────────────────────────────────────

with st.spinner("Loading sensor data..."):
    df_full, is_live = get_data()

if df_full.empty:
    st.error("No data available. Please try refreshing.")
    st.stop()

if is_live:
    st.success("Live data from Open-Meteo Archive API")
else:
    st.info("Showing sample data — archive API temporarily unavailable.")

cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
df = df_full[
    (df_full["city"].isin(selected_cities)) &
    (df_full["timestamp"] >= cutoff)
].copy()

if df.empty:
    st.warning("No data for the selected filters. Try adjusting cities or hours.")
    st.stop()


# ── Page header and KPI cards ─────────────────────────────────────────────────

st.title("Live Sensor Analytics Dashboard")
st.caption(
    f"Weather analytics  |  {len(selected_cities)} cities  |  "
    f"{len(df):,} readings  |  Last {hours_back} hours"
)

latest   = df.sort_values("timestamp").groupby("city").last().reset_index()
kpi_cols = st.columns(len(latest))

for col, (_, row) in zip(kpi_cols, latest.iterrows()):
    with col:
        st.metric(
            label=row["city"],
            value=f"{row['temperature_2m']:.1f} C",
            delta=f"Humidity: {row['relative_humidity_2m']:.0f}%",
        )

st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_trends, tab_rankings, tab_anomalies, tab_correlations, tab_export = st.tabs([
    "Trends", "Rankings", "Anomalies", "Correlations", "Export",
])


# ── Tab 1: Trends ─────────────────────────────────────────────────────────────

with tab_trends:
    st.subheader("Temperature Over Time")
    st.caption(
        "Dotted lines show raw hourly readings. Solid lines show the 6-hour "
        "rolling average to smooth short-term fluctuations."
    )

    fig_temp = go.Figure()
    for city in selected_cities:
        city_df     = df[df["city"] == city].sort_values("timestamp")
        rolling_avg = city_df["temperature_2m"].rolling(6, min_periods=1).mean()
        color       = CITY_COLORS.get(city, "#888888")

        fig_temp.add_trace(go.Scatter(
            x=city_df["timestamp"],
            y=city_df["temperature_2m"],
            name=city,
            line=dict(color=color, width=1, dash="dot"),
            opacity=0.3,
            showlegend=False,
        ))
        fig_temp.add_trace(go.Scatter(
            x=city_df["timestamp"],
            y=rolling_avg,
            name=f"{city} (6h avg)",
            line=dict(color=color, width=2),
        ))

    fig_temp.update_layout(
        title="Hourly Temperature Trends Across European Cities (with 6-Hour Smoothing)",
        xaxis_title="Date and Time (UTC)",
        yaxis_title="Temperature (C)",
        height=420,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    col_humidity, col_pressure = st.columns(2)

    with col_humidity:
        fig_hum = px.line(
            df.sort_values("timestamp"),
            x="timestamp",
            y="relative_humidity_2m",
            color="city",
            color_discrete_map=CITY_COLORS,
            title="Relative Humidity — How Damp the Air Feels (%)",
            labels={
                "relative_humidity_2m": "Humidity (%)",
                "timestamp":            "Date and Time (UTC)",
                "city":                 "City",
            },
            template="plotly_white",
        )
        fig_hum.update_layout(height=320)
        st.plotly_chart(fig_hum, use_container_width=True)

    with col_pressure:
        fig_pres = px.line(
            df.sort_values("timestamp"),
            x="timestamp",
            y="surface_pressure",
            color="city",
            color_discrete_map=CITY_COLORS,
            title="Atmospheric Pressure — Higher Values Indicate Stable, Dry Weather (hPa)",
            labels={
                "surface_pressure": "Pressure (hPa)",
                "timestamp":        "Date and Time (UTC)",
                "city":             "City",
            },
            template="plotly_white",
        )
        fig_pres.update_layout(height=320)
        st.plotly_chart(fig_pres, use_container_width=True)

    daily = df.copy()
    daily["date"] = daily["timestamp"].dt.date
    daily_agg = (
        daily.groupby(["city", "date"])
        .agg(
            avg_temp     = ("temperature_2m",       "mean"),
            min_temp     = ("temperature_2m",       "min"),
            max_temp     = ("temperature_2m",       "max"),
            avg_humidity = ("relative_humidity_2m", "mean"),
            total_precip = ("precipitation",        "sum"),
        )
        .round(1)
        .reset_index()
        .sort_values(["date", "city"], ascending=[False, True])
    )
    st.subheader("Daily Summary")
    st.caption("Aggregated min, max, and average temperature per city per day.")
    st.dataframe(daily_agg, use_container_width=True, hide_index=True)


# ── Tab 2: Rankings ───────────────────────────────────────────────────────────

with tab_rankings:
    st.subheader("Current Conditions by City")
    st.caption("Based on the most recent hourly reading available for each city.")

    fig_rank = px.bar(
        latest,
        x="city",
        y="temperature_2m",
        color="temperature_2m",
        color_continuous_scale="RdYlBu_r",
        title="Which City is Warmest Right Now?",
        text="temperature_2m",
        labels={"temperature_2m": "Temperature (C)", "city": "City"},
        template="plotly_white",
    )
    fig_rank.update_traces(texttemplate="%{text:.1f} C", textposition="outside")
    fig_rank.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_rank, use_container_width=True)

    col_extremes, col_wind = st.columns(2)

    with col_extremes:
        extremes = (
            df.groupby("city")
            .agg(
                min_temp = ("temperature_2m", "min"),
                max_temp = ("temperature_2m", "max"),
                avg_temp = ("temperature_2m", "mean"),
            )
            .round(1)
            .reset_index()
        )
        fig_ext = go.Figure()
        fig_ext.add_trace(go.Bar(
            name="Minimum",
            x=extremes["city"],
            y=extremes["min_temp"],
            marker_color="#74B9FF",
        ))
        fig_ext.add_trace(go.Bar(
            name="Maximum",
            x=extremes["city"],
            y=extremes["max_temp"],
            marker_color="#E17055",
        ))
        fig_ext.update_layout(
            title="Temperature Range — Minimum vs Maximum Over Last 48 Hours",
            xaxis_title="City",
            yaxis_title="Temperature (C)",
            barmode="overlay",
            template="plotly_white",
            height=340,
        )
        st.plotly_chart(fig_ext, use_container_width=True)

    with col_wind:
        fig_wind = px.box(
            df,
            x="city",
            y="wind_speed_10m",
            color="city",
            color_discrete_map=CITY_COLORS,
            title="Wind Speed Distribution — How Gusty Has Each City Been?",
            labels={"wind_speed_10m": "Wind Speed (km/h)", "city": "City"},
            template="plotly_white",
        )
        fig_wind.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig_wind, use_container_width=True)

    latest["warmth_rank"] = latest["temperature_2m"].rank(ascending=False).astype(int)
    st.subheader("Full Rankings Table")
    st.dataframe(
        latest[[
            "city", "country", "temperature_2m", "apparent_temperature",
            "relative_humidity_2m", "wind_speed_10m", "surface_pressure", "warmth_rank",
        ]].sort_values("warmth_rank"),
        use_container_width=True,
        hide_index=True,
    )


# ── Tab 3: Anomalies ──────────────────────────────────────────────────────────

with tab_anomalies:
    st.subheader("Statistical Anomaly Detection")
    st.caption(
        "Readings are flagged as anomalous when they deviate more than 2 standard "
        "deviations from the 6-hour rolling mean for their city. These may indicate "
        "sudden weather changes, sensor noise, or unusual meteorological events."
    )

    anom_df = detect_anomalies(df)

    col_total, col_count, col_rate = st.columns(3)
    col_total.metric("Total Readings",      f"{len(anom_df):,}")
    col_count.metric("Anomalous Readings",  f"{anom_df['is_anomaly'].sum():,}")
    col_rate.metric("Anomaly Rate",         f"{anom_df['is_anomaly'].mean():.1%}")

    fig_anom = px.scatter(
        anom_df,
        x="timestamp",
        y="temperature_2m",
        color="city",
        symbol="is_anomaly",
        symbol_map={True: "x", False: "circle"},
        color_discrete_map=CITY_COLORS,
        title=(
            "Temperature Readings — Anomalous Points (marked X) Deviate "
            "More Than 2 Standard Deviations from the Recent Trend"
        ),
        labels={
            "temperature_2m": "Temperature (C)",
            "timestamp":      "Date and Time (UTC)",
            "city":           "City",
        },
        template="plotly_white",
    )
    fig_anom.update_layout(height=420)
    st.plotly_chart(fig_anom, use_container_width=True)

    anomalies_only = anom_df[anom_df["is_anomaly"]][[
        "city", "timestamp", "temperature_2m", "rolling_mean", "rolling_std",
    ]].copy()

    if not anomalies_only.empty:
        anomalies_only["deviation_C"] = (
            (anomalies_only["temperature_2m"] - anomalies_only["rolling_mean"])
            .abs().round(2)
        )
        st.subheader("Anomalous Readings Detail")
        st.dataframe(anomalies_only, use_container_width=True, hide_index=True)


# ── Tab 4: Correlations ───────────────────────────────────────────────────────

with tab_correlations:
    st.subheader("Cross-Sensor Correlations")
    st.caption(
        "Pearson correlation coefficients between temperature and other variables. "
        "Values close to -1 or +1 indicate a strong linear relationship. "
        "A negative correlation between temperature and humidity means warmer air "
        "tends to be drier."
    )

    corr_rows = []
    for city, group in df.groupby("city"):
        corr_rows.append({
            "city":             city,
            "temp vs humidity": round(group["temperature_2m"].corr(group["relative_humidity_2m"]), 3),
            "temp vs pressure": round(group["temperature_2m"].corr(group["surface_pressure"]), 3),
            "temp vs wind":     round(group["temperature_2m"].corr(group["wind_speed_10m"]), 3),
            "n readings":       len(group),
        })
    st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)

    fig_scatter = px.scatter(
        df,
        x="temperature_2m",
        y="relative_humidity_2m",
        color="city",
        color_discrete_map=CITY_COLORS,
        trendline="ols",
        title="Do Warmer Cities Have Drier Air? Temperature vs Relative Humidity by City",
        labels={
            "temperature_2m":       "Temperature (C)",
            "relative_humidity_2m": "Relative Humidity (%)",
            "city":                 "City",
        },
        template="plotly_white",
    )
    fig_scatter.update_layout(height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)


# ── Tab 5: Export ─────────────────────────────────────────────────────────────

with tab_export:
    st.subheader("Download Data")
    st.caption(
        "Download the currently filtered sensor readings as a CSV file or as a "
        "multi-sheet Excel report including daily summaries and city rankings."
    )

    col_csv, col_excel = st.columns(2)

    with col_csv:
        st.markdown("**CSV — Raw Sensor Readings**")
        st.caption(f"{len(df):,} rows  x  {len(df.columns)} columns")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "Download CSV",
            data      = csv_bytes,
            file_name = f"sensor_readings_{datetime.now().strftime('%Y%m%d')}.csv",
            mime      = "text/csv",
            use_container_width=True,
        )

    with col_excel:
        st.markdown("**Excel — Multi-Sheet Analytics Report**")
        st.caption("Sheets: Raw Readings | Daily Summary | City Rankings")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            strip_timezone(df.head(500)).to_excel(writer, sheet_name="Raw Readings", index=False)
            strip_timezone(daily_agg).to_excel(writer,   sheet_name="Daily Summary", index=False)
            strip_timezone(latest).to_excel(writer,      sheet_name="City Rankings", index=False)
        st.download_button(
            label     = "Download Excel Report",
            data      = buf.getvalue(),
            file_name = f"sensor_analytics_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.subheader("Data Preview — Last 20 Rows")
    st.dataframe(df.tail(20), use_container_width=True, hide_index=True)