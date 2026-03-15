"""
Live Sensor Analytics Dashboard
================================
Streamlit dashboard for real-time weather sensor analytics
across 6 European cities — Open-Meteo API.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live Sensor Analytics",
    page_icon="🌡️",
    layout="wide",
)

# ── Hardcoded config ─────────────────────────────────────────────────────────
LOCATIONS = [
    {"name": "Munich",    "latitude": 48.1351, "longitude": 11.5820, "country": "Germany"},
    {"name": "Berlin",    "latitude": 52.5200, "longitude": 13.4050, "country": "Germany"},
    {"name": "Hamburg",   "latitude": 53.5511, "longitude":  9.9937, "country": "Germany"},
    {"name": "Frankfurt", "latitude": 50.1109, "longitude":  8.6821, "country": "Germany"},
    {"name": "London",    "latitude": 51.5074, "longitude": -0.1278, "country": "UK"},
    {"name": "Paris",     "latitude": 48.8566, "longitude":  2.3522, "country": "France"},
]
VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "surface_pressure", "wind_speed_10m",
    "wind_direction_10m", "precipitation",
]
CITY_COLORS = {
    "Munich": "#2E75B6", "Berlin": "#E74C3C", "Hamburg": "#27AE60",
    "Frankfurt": "#F39C12", "London": "#8E44AD", "Paris": "#16A085",
}


# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_cities(past_hours: int = 72) -> pd.DataFrame:
    dfs = []
    for loc in LOCATIONS:
        try:
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":    loc["latitude"],
                    "longitude":   loc["longitude"],
                    "hourly":      ",".join(VARIABLES),
                    "forecast_days": 3,
                    "timezone":    "UTC",
                },
                timeout=30,
            )
            resp.raise_for_status()
            hourly = resp.json().get("hourly", {})
            times  = hourly.get("time", [])
            if not times:
                continue
            rows = []
            for i, ts in enumerate(times):
                row = {"city": loc["name"], "country": loc["country"],
                       "timestamp": ts}
                for var in VARIABLES:
                    vals = hourly.get(var, [])
                    row[var] = vals[i] if i < len(vals) else None
                rows.append(row)
            dfs.append(pd.DataFrame(rows))
        except Exception as e:
            st.error(f"Failed to fetch {loc['name']}: {type(e).__name__}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌡️ Sensor Analytics")
    st.caption("Live weather data — Open-Meteo API")
    st.divider()

    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    hours_back = st.slider("Hours back", 6, 72, 48, 6)
    cities_all = [loc["name"] for loc in LOCATIONS]
    selected_cities = st.multiselect("Cities", cities_all, default=cities_all)
    st.divider()
    st.caption("**Stack:** Open-Meteo API · PostgreSQL · SQLAlchemy · Streamlit · Plotly")
    st.caption("**GitHub:** [sensor-analytics-dashboard](https://github.com/danielamissah/sensor-analytics-dashboard)")


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Fetching live sensor data..."):
    df_full = fetch_all_cities(past_hours=72)

if df_full.empty:
    st.error("Could not fetch data from Open-Meteo API.")
    st.info("Check your internet connection and try refreshing.")
    st.stop()

# Filter
cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
df = df_full[
    (df_full["city"].isin(selected_cities)) &
    (df_full["timestamp"] >= cutoff)
].copy()

if df.empty:
    st.warning("No data for selected filters.")
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌡️ Live Sensor Analytics Dashboard")
st.caption(f"Real-time weather analytics · {len(selected_cities)} cities · {len(df):,} readings · Last {hours_back}h")

# ── KPI cards ─────────────────────────────────────────────────────────────────
latest = df.sort_values("timestamp").groupby("city").last().reset_index()
cols = st.columns(len(latest))
for i, (_, row) in enumerate(latest.iterrows()):
    with cols[i]:
        st.metric(
            label=f"🏙️ {row['city']}",
            value=f"{row['temperature_2m']:.1f}°C",
            delta=f"💧{row['relative_humidity_2m']:.0f}%",
        )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trends", "🏆 Rankings", "⚠️ Anomalies", "📊 Correlations", "📥 Export"
])

# ── Tab 1: Trends ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Temperature Trends")
    fig = go.Figure()
    for city in selected_cities:
        cdf = df[df["city"] == city].sort_values("timestamp")
        cdf["rolling_6h"] = cdf["temperature_2m"].rolling(6, min_periods=1).mean()
        color = CITY_COLORS.get(city, "#888")
        fig.add_trace(go.Scatter(x=cdf["timestamp"], y=cdf["temperature_2m"],
            name=f"{city}", line=dict(color=color, width=1, dash="dot"),
            opacity=0.4, showlegend=False))
        fig.add_trace(go.Scatter(x=cdf["timestamp"], y=cdf["rolling_6h"],
            name=f"{city} (6h avg)", line=dict(color=color, width=2)))
    fig.update_layout(title="Temperature (°C) with 6h rolling average",
        xaxis_title="Time (UTC)", yaxis_title="Temperature (°C)",
        height=400, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig2 = px.line(df.sort_values("timestamp"), x="timestamp",
            y="relative_humidity_2m", color="city",
            color_discrete_map=CITY_COLORS, title="Humidity (%)",
            template="plotly_white")
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = px.line(df.sort_values("timestamp"), x="timestamp",
            y="surface_pressure", color="city",
            color_discrete_map=CITY_COLORS, title="Pressure (hPa)",
            template="plotly_white")
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    # Daily summary
    st.subheader("Daily Summary")
    daily = df.copy()
    daily["date"] = daily["timestamp"].dt.date
    daily_agg = daily.groupby(["city", "date"]).agg(
        avg_temp=("temperature_2m", "mean"),
        min_temp=("temperature_2m", "min"),
        max_temp=("temperature_2m", "max"),
        avg_humidity=("relative_humidity_2m", "mean"),
        total_precip=("precipitation", "sum"),
    ).round(1).reset_index().sort_values(["date", "city"], ascending=[False, True])
    st.dataframe(daily_agg, use_container_width=True, hide_index=True)

# ── Tab 2: Rankings ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("Current Conditions by City")
    fig_rank = px.bar(latest, x="city", y="temperature_2m",
        color="temperature_2m", color_continuous_scale="RdYlBu_r",
        title="Current Temperature", text="temperature_2m",
        template="plotly_white")
    fig_rank.update_traces(texttemplate="%{text:.1f}°C", textposition="outside")
    fig_rank.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_rank, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        extremes = df.groupby("city").agg(
            min_temp=("temperature_2m", "min"),
            max_temp=("temperature_2m", "max"),
            avg_temp=("temperature_2m", "mean"),
        ).round(1).reset_index()
        fig_ext = go.Figure()
        fig_ext.add_trace(go.Bar(name="Min", x=extremes["city"], y=extremes["min_temp"], marker_color="#74B9FF"))
        fig_ext.add_trace(go.Bar(name="Max", x=extremes["city"], y=extremes["max_temp"], marker_color="#E17055"))
        fig_ext.update_layout(title="Temperature Extremes", barmode="overlay",
            template="plotly_white", height=300)
        st.plotly_chart(fig_ext, use_container_width=True)
    with c2:
        fig_wind = px.box(df, x="city", y="wind_speed_10m", color="city",
            color_discrete_map=CITY_COLORS, title="Wind Speed Distribution (km/h)",
            template="plotly_white")
        fig_wind.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_wind, use_container_width=True)

    st.subheader("Full Rankings Table")
    latest["warmth_rank"] = latest["temperature_2m"].rank(ascending=False).astype(int)
    st.dataframe(latest[["city", "country", "temperature_2m", "apparent_temperature",
                           "relative_humidity_2m", "wind_speed_10m", "surface_pressure",
                           "warmth_rank"]].sort_values("warmth_rank"),
                 use_container_width=True, hide_index=True)

# ── Tab 3: Anomalies ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("Statistical Anomaly Detection")
    anom = df.copy()
    for city, group in anom.groupby("city"):
        roll_mean = group["temperature_2m"].rolling(6, min_periods=1).mean()
        roll_std  = group["temperature_2m"].rolling(6, min_periods=1).std().fillna(0)
        anom.loc[group.index, "roll_mean"] = roll_mean
        anom.loc[group.index, "roll_std"]  = roll_std
    anom["is_anomaly"] = (anom["temperature_2m"] - anom["roll_mean"]).abs() > 2 * anom["roll_std"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Readings", f"{len(anom):,}")
    c2.metric("Anomalies", f"{anom['is_anomaly'].sum():,}")
    c3.metric("Anomaly Rate", f"{anom['is_anomaly'].mean():.1%}")

    fig_anom = px.scatter(anom, x="timestamp", y="temperature_2m",
        color="city", symbol="is_anomaly",
        symbol_map={True: "x", False: "circle"},
        color_discrete_map=CITY_COLORS,
        title="Temperature — anomalies marked ×",
        template="plotly_white")
    fig_anom.update_layout(height=400)
    st.plotly_chart(fig_anom, use_container_width=True)

    precip = df[df["precipitation"] > 0.1][
        ["city", "timestamp", "precipitation", "temperature_2m"]
    ].sort_values("precipitation", ascending=False).head(50)
    if not precip.empty:
        st.subheader("Precipitation Events")
        fig_p = px.bar(precip, x="timestamp", y="precipitation", color="city",
            color_discrete_map=CITY_COLORS, title="Precipitation Events (>0.1mm)",
            template="plotly_white")
        fig_p.update_layout(height=300)
        st.plotly_chart(fig_p, use_container_width=True)

# ── Tab 4: Correlations ───────────────────────────────────────────────────────
with tab4:
    st.subheader("Cross-Sensor Correlations")
    corr_rows = []
    for city, group in df.groupby("city"):
        corr_rows.append({
            "city": city,
            "temp_humidity": round(group["temperature_2m"].corr(group["relative_humidity_2m"]), 3),
            "temp_pressure": round(group["temperature_2m"].corr(group["surface_pressure"]), 3),
            "temp_wind":     round(group["temperature_2m"].corr(group["wind_speed_10m"]), 3),
            "n": len(group),
        })
    st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)

    fig_sc = px.scatter(df, x="temperature_2m", y="relative_humidity_2m",
        color="city", color_discrete_map=CITY_COLORS,
        trendline="ols", title="Temperature vs Humidity",
        template="plotly_white")
    fig_sc.update_layout(height=400)
    st.plotly_chart(fig_sc, use_container_width=True)

# ── Tab 5: Export ─────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Export Data")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Raw sensor readings (CSV)**")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv,
            f"sensor_readings_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
            use_container_width=True)
    with c2:
        st.markdown("**Analytics report (Excel)**")
        import io
        daily_exp = df.copy()
        daily_exp["date"] = daily_exp["timestamp"].dt.date
        daily_exp["timestamp"] = daily_exp["timestamp"].dt.tz_localize(None)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            daily_exp.head(500).to_excel(writer, sheet_name="Raw Readings", index=False)
            daily_agg.to_excel(writer, sheet_name="Daily Summary", index=False)
            latest.to_excel(writer, sheet_name="City Rankings", index=False)
        st.download_button("📊 Download Excel",
            buf.getvalue(),
            f"sensor_analytics_{datetime.now().strftime('%Y%m%d')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)

    st.subheader("Data Preview")
    st.dataframe(df.tail(20), use_container_width=True, hide_index=True)