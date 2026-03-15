"""
Live Sensor Analytics Dashboard
================================
Real-time weather analytics across 6 European cities.
Data: Tomorrow.io API (live) with sample data fallback.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import io

st.set_page_config(
    page_title="Live Sensor Analytics",
    page_icon="🌡️",
    layout="wide",
)

LOCATIONS = [
    {"name": "Munich",    "latitude": 48.1351, "longitude": 11.5820, "country": "Germany"},
    {"name": "Berlin",    "latitude": 52.5200, "longitude": 13.4050, "country": "Germany"},
    {"name": "Hamburg",   "latitude": 53.5511, "longitude":  9.9937, "country": "Germany"},
    {"name": "Frankfurt", "latitude": 50.1109, "longitude":  8.6821, "country": "Germany"},
    {"name": "London",    "latitude": 51.5074, "longitude": -0.1278, "country": "UK"},
    {"name": "Paris",     "latitude": 48.8566, "longitude":  2.3522, "country": "France"},
]

CITY_COLORS = {
    "Munich": "#2E75B6", "Berlin": "#E74C3C", "Hamburg": "#27AE60",
    "Frankfurt": "#F39C12", "London": "#8E44AD", "Paris": "#16A085",
}

# Tomorrow.io fields to fetch
ARCHIVE_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "surface_pressure", "wind_speed_10m",
    "wind_direction_10m", "precipitation",
]


def fetch_city_archive(loc: dict, days: int = 3) -> pd.DataFrame:
    """Fetch historical hourly data from Open-Meteo Archive API (no auth, separate rate limit)."""
    try:
        end_date   = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days)
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":  loc["latitude"],
                "longitude": loc["longitude"],
                "start_date": start_date.isoformat(),
                "end_date":   end_date.isoformat(),
                "hourly":     ",".join(ARCHIVE_VARIABLES),
                "timezone":   "UTC",
            },
            headers={"User-Agent": "SensorDashboard/1.0"},
            timeout=30,
        )
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        times  = hourly.get("time", [])
        if not times:
            return pd.DataFrame()
        rows = []
        for i, ts in enumerate(times):
            row = {"city": loc["name"], "country": loc["country"], "timestamp": ts}
            for var in ARCHIVE_VARIABLES:
                vals = hourly.get(var, [])
                row[var] = vals[i] if i < len(vals) else None
            rows.append(row)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except Exception:
        return pd.DataFrame()


def load_sample_data() -> pd.DataFrame:
    """Load committed sample data as fallback."""
    try:
        df = pd.read_csv("sample_data.csv", parse_dates=["timestamp"])
        max_ts = pd.to_datetime(df["timestamp"], utc=True).max()
        now    = pd.Timestamp.now(tz="UTC")
        delta  = now - max_ts
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True) + delta
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_data(_unused: str = "") -> tuple:
    """Fetch data from Open-Meteo Archive API — no auth required."""
    dfs = []
    for loc in LOCATIONS:
        df = fetch_city_archive(loc, days=3)
        if not df.empty:
            dfs.append(df)
        time.sleep(0.5)

    if dfs:
        return pd.concat(dfs, ignore_index=True), True

    return load_sample_data(), False


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌡️ Sensor Analytics")
    st.caption("Historical weather data — Open-Meteo Archive API")
    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    hours_back = st.slider("Hours back", 6, 72, 48, 6)
    cities_all = [loc["name"] for loc in LOCATIONS]
    selected_cities = st.multiselect("Cities", cities_all, default=cities_all)
    st.divider()
    st.caption("**Stack:** Open-Meteo Archive API · PostgreSQL · Streamlit · Plotly")
    st.caption("**GitHub:** [sensor-analytics-dashboard](https://github.com/danielamissah/sensor-analytics-dashboard)")


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading sensor data..."):
    df_full, is_live = get_data()

if df_full.empty:
    st.error("No data available.")
    st.stop()

if is_live:
    st.success("🟢 Live data from Open-Meteo Archive API")
else:
    st.info("Showing sample data — archive API temporarily unavailable.")

# Filter
cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
df = df_full[
    (df_full["city"].isin(selected_cities)) &
    (df_full["timestamp"] >= cutoff)
].copy()

if df.empty:
    st.warning("No data for selected filters.")
    st.stop()

# ── Header ─────────────────────────────────────────────────────────────────
st.title("Live Sensor Analytics Dashboard")
st.caption(f"Weather analytics · {len(selected_cities)} cities · {len(df):,} readings · Last {hours_back}h")

# ── KPI cards ───────────────────────────────────────────────────────────────
latest = df.sort_values("timestamp").groupby("city").last().reset_index()
cols   = st.columns(len(latest))
for i, (_, row) in enumerate(latest.iterrows()):
    with cols[i]:
        st.metric(
            label=f"{row['city']}",
            value=f"{row['temperature_2m']:.1f}°C",
            delta=f"{row['relative_humidity_2m']:.0f}%",
        )
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Trends", "Rankings", "Anomalies", "Correlations", "Export"
])

with tab1:
    st.subheader("Temperature Trends")
    fig = go.Figure()
    for city in selected_cities:
        cdf   = df[df["city"] == city].sort_values("timestamp")
        roll  = cdf["temperature_2m"].rolling(6, min_periods=1).mean()
        color = CITY_COLORS.get(city, "#888")
        fig.add_trace(go.Scatter(x=cdf["timestamp"], y=cdf["temperature_2m"],
            name=city, line=dict(color=color, width=1, dash="dot"),
            opacity=0.3, showlegend=False))
        fig.add_trace(go.Scatter(x=cdf["timestamp"], y=roll,
            name=f"{city} (6h avg)", line=dict(color=color, width=2)))
    fig.update_layout(title="Temperature (°C) with 6h rolling average",
        height=400, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        f2 = px.line(df.sort_values("timestamp"), x="timestamp", y="relative_humidity_2m",
            color="city", color_discrete_map=CITY_COLORS, title="Humidity (%)",
            template="plotly_white")
        f2.update_layout(height=300)
        st.plotly_chart(f2, use_container_width=True)
    with c2:
        f3 = px.line(df.sort_values("timestamp"), x="timestamp", y="surface_pressure",
            color="city", color_discrete_map=CITY_COLORS, title="Pressure (hPa)",
            template="plotly_white")
        f3.update_layout(height=300)
        st.plotly_chart(f3, use_container_width=True)

    daily = df.copy()
    daily["date"] = daily["timestamp"].dt.date
    daily_agg = daily.groupby(["city", "date"]).agg(
        avg_temp=("temperature_2m", "mean"), min_temp=("temperature_2m", "min"),
        max_temp=("temperature_2m", "max"), avg_humidity=("relative_humidity_2m", "mean"),
        total_precip=("precipitation", "sum"),
    ).round(1).reset_index().sort_values(["date", "city"], ascending=[False, True])
    st.subheader("Daily Summary")
    st.dataframe(daily_agg, use_container_width=True, hide_index=True)

with tab2:
    fig_rank = px.bar(latest, x="city", y="temperature_2m", color="temperature_2m",
        color_continuous_scale="RdYlBu_r", title="Current Temperature",
        text="temperature_2m", template="plotly_white")
    fig_rank.update_traces(texttemplate="%{text:.1f}°C", textposition="outside")
    fig_rank.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_rank, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        ext = df.groupby("city").agg(
            min_temp=("temperature_2m", "min"), max_temp=("temperature_2m", "max"),
            avg_temp=("temperature_2m", "mean")).round(1).reset_index()
        fe = go.Figure()
        fe.add_trace(go.Bar(name="Min", x=ext["city"], y=ext["min_temp"], marker_color="#74B9FF"))
        fe.add_trace(go.Bar(name="Max", x=ext["city"], y=ext["max_temp"], marker_color="#E17055"))
        fe.update_layout(title="Temperature Extremes", barmode="overlay",
            template="plotly_white", height=300)
        st.plotly_chart(fe, use_container_width=True)
    with c2:
        fw = px.box(df, x="city", y="wind_speed_10m", color="city",
            color_discrete_map=CITY_COLORS, title="Wind Speed (km/h)",
            template="plotly_white")
        fw.update_layout(height=300, showlegend=False)
        st.plotly_chart(fw, use_container_width=True)

    latest["rank"] = latest["temperature_2m"].rank(ascending=False).astype(int)
    st.dataframe(latest[["city","country","temperature_2m","apparent_temperature",
                          "relative_humidity_2m","wind_speed_10m","surface_pressure","rank"]
                         ].sort_values("rank"), use_container_width=True, hide_index=True)

with tab3:
    anom = df.copy()
    for city, grp in anom.groupby("city"):
        rm = grp["temperature_2m"].rolling(6, min_periods=1).mean()
        rs = grp["temperature_2m"].rolling(6, min_periods=1).std().fillna(0)
        anom.loc[grp.index, "roll_mean"] = rm
        anom.loc[grp.index, "roll_std"]  = rs
    anom["is_anomaly"] = (anom["temperature_2m"] - anom["roll_mean"]).abs() > 2 * anom["roll_std"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Readings", f"{len(anom):,}")
    c2.metric("Anomalies", f"{anom['is_anomaly'].sum():,}")
    c3.metric("Anomaly Rate", f"{anom['is_anomaly'].mean():.1%}")

    fa = px.scatter(anom, x="timestamp", y="temperature_2m", color="city",
        symbol="is_anomaly", symbol_map={True: "x", False: "circle"},
        color_discrete_map=CITY_COLORS,
        title="Temperature anomalies (>2σ from 6h rolling mean)",
        template="plotly_white")
    fa.update_layout(height=400)
    st.plotly_chart(fa, use_container_width=True)

with tab4:
    corr_rows = []
    for city, grp in df.groupby("city"):
        corr_rows.append({
            "city": city,
            "temp_humidity": round(grp["temperature_2m"].corr(grp["relative_humidity_2m"]), 3),
            "temp_pressure": round(grp["temperature_2m"].corr(grp["surface_pressure"]), 3),
            "temp_wind":     round(grp["temperature_2m"].corr(grp["wind_speed_10m"]), 3),
            "n": len(grp),
        })
    st.subheader("Cross-Sensor Correlations")
    st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)
    fs = px.scatter(df, x="temperature_2m", y="relative_humidity_2m",
        color="city", color_discrete_map=CITY_COLORS, trendline="ols",
        title="Temperature vs Humidity", template="plotly_white")
    fs.update_layout(height=400)
    st.plotly_chart(fs, use_container_width=True)

with tab5:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**CSV Export**")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv,
            f"sensor_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
            use_container_width=True)
    with c2:
        st.markdown("**Excel Report**")
        def strip_tz(df):
            d = df.copy()
            for col in d.select_dtypes(include=["datetimetz"]).columns:
                d[col] = d[col].dt.tz_localize(None)
            return d
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            strip_tz(df.head(500)).to_excel(w, sheet_name="Raw", index=False)
            strip_tz(daily_agg).to_excel(w, sheet_name="Daily", index=False)
            strip_tz(latest).to_excel(w, sheet_name="Rankings", index=False)
        st.download_button("📊 Download Excel", buf.getvalue(),
            f"analytics_{datetime.now().strftime('%Y%m%d')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    st.dataframe(df.tail(20), use_container_width=True, hide_index=True)