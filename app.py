"""
Live Sensor Analytics Dashboard
================================
Streamlit dashboard for real-time weather sensor analytics
across 6 European cities — Open-Meteo API + PostgreSQL + SQL analytics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone

from src.ingestion.fetch_data import load_data, run_ingestion, fetch_direct

@st.cache_data(ttl=3600)
def get_cached_data():
    """Fetch and cache sensor data for 1 hour."""
    import traceback
    errors = []
    
    # Test a single city first
    try:
        import requests
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 48.1351, "longitude": 11.5820,
                "hourly": "temperature_2m",
                "past_hours": 6, "forecast_hours": 0, "timezone": "UTC"
            },
            timeout=30
        )
        data = resp.json()
        times = data.get("hourly", {}).get("time", [])
        if times:
            # Full fetch
            return fetch_direct(past_hours=72)
        else:
            errors.append(f"API returned no times: {data}")
    except Exception as e:
        errors.append(f"Direct API test failed: {e}\n{traceback.format_exc()}")
    
    raise RuntimeError("\n".join(errors) or "Unknown error")
from src.queries.analytics import (
    QUERIES, PANDAS_QUERIES, run_query,
    latest_reading_per_city, temperature_extremes,
    daily_summary, city_rankings, rolling_avg_temp,
    pressure_anomalies, temperature_correlation,
    hourly_avg_by_city,
)
from src.export.exporter import to_csv, to_excel, export_filename

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Live Sensor Analytics",
    page_icon   = "🌡️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.metric-card {
    background: #f8fafc;
    border-left: 4px solid #2E75B6;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 4px 0;
}
.sql-box {
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 12px;
    border-radius: 6px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    white-space: pre;
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)

CITY_COLORS = {
    "Munich":    "#2E75B6",
    "Berlin":    "#E74C3C",
    "Hamburg":   "#27AE60",
    "Frankfurt": "#F39C12",
    "London":    "#8E44AD",
    "Paris":     "#16A085",
}

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌡️ Sensor Analytics")
    st.caption("Live weather data — Open-Meteo API")

    st.divider()

    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    if "last_refresh" in st.session_state:
        st.caption(f"Last refresh: {st.session_state['last_refresh'].strftime('%H:%M:%S')}")

    st.divider()

    # Filters
    st.subheader("Filters")

    # Load data using Streamlit cache (1 hour TTL)
    with st.spinner("Fetching live sensor data from Open-Meteo API..."):
        try:
            df_full = get_cached_data()
            st.session_state["last_refresh"] = datetime.now()
        except Exception as e:
            st.error(f"Fetch failed: {str(e)}")
            st.stop()

    if df_full.empty:
        st.error("Open-Meteo returned no data.")
        if st.button("Retry"):
            st.cache_data.clear()
            st.rerun()
        st.stop()

    cities        = sorted(df_full["city"].unique().tolist())
    selected_cities = st.multiselect("Cities", cities, default=cities)

    hours_back = st.slider("Hours back", min_value=6, max_value=72, value=48, step=6)

    st.divider()
    st.caption("**Stack:** Open-Meteo API · PostgreSQL · SQLAlchemy · Streamlit · Plotly")
    st.caption("**GitHub:** [sensor-analytics-dashboard](https://github.com/dkamissah/sensor-analytics-dashboard)")

# ── Filter data ─────────────────────────────────────────────────────────────
cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
df = df_full[
    (df_full["city"].isin(selected_cities)) &
    (df_full["timestamp"] >= cutoff)
].copy()

if df.empty:
    st.warning("No data for selected filters.")
    st.stop()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🌡️ Live Sensor Analytics Dashboard")
st.caption(f"Real-time weather analytics across {len(selected_cities)} European cities · {len(df):,} readings · Last {hours_back}h")

# ── KPI Cards ────────────────────────────────────────────────────────────────
latest = latest_reading_per_city(df)

cols = st.columns(len(latest))
for i, (_, row) in enumerate(latest.iterrows()):
    with cols[i]:
        delta = ""
        st.metric(
            label = f"🏙️ {row['city']}",
            value = f"{row['temperature_2m']:.1f}°C",
            delta = f"💧{row['relative_humidity_2m']:.0f}%  💨{row['wind_speed_10m']:.1f}km/h",
        )

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Trends", "🏆 Rankings", "⚠️ Anomalies",
    "📊 Correlations", "🗃️ SQL Explorer", "📥 Export"
])

# ── Tab 1: Trends ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Temperature Trends by City")

    roll_df = rolling_avg_temp(df)

    fig_temp = go.Figure()
    for city in selected_cities:
        city_df = roll_df[roll_df["city"] == city].sort_values("timestamp")
        color   = CITY_COLORS.get(city, "#888888")
        fig_temp.add_trace(go.Scatter(
            x=city_df["timestamp"], y=city_df["temperature_2m"],
            name=f"{city} (raw)", line=dict(color=color, width=1, dash="dot"),
            opacity=0.4, showlegend=False,
        ))
        fig_temp.add_trace(go.Scatter(
            x=city_df["timestamp"], y=city_df["rolling_avg_6h"],
            name=f"{city} (6h avg)", line=dict(color=color, width=2),
        ))

    fig_temp.update_layout(
        title="Temperature (°C) — raw + 6h rolling average",
        xaxis_title="Time (UTC)", yaxis_title="Temperature (°C)",
        height=400, template="plotly_white", hovermode="x unified",
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Humidity Trends")
        fig_hum = px.line(
            df.sort_values("timestamp"), x="timestamp", y="relative_humidity_2m",
            color="city", color_discrete_map=CITY_COLORS,
            title="Relative Humidity (%)", template="plotly_white",
            labels={"relative_humidity_2m": "Humidity (%)", "timestamp": "Time (UTC)"},
        )
        fig_hum.update_layout(height=300, font=dict(family="Inter"))
        st.plotly_chart(fig_hum, use_container_width=True)

    with col2:
        st.subheader("Atmospheric Pressure")
        fig_pres = px.line(
            df.sort_values("timestamp"), x="timestamp", y="surface_pressure",
            color="city", color_discrete_map=CITY_COLORS,
            title="Surface Pressure (hPa)", template="plotly_white",
            labels={"surface_pressure": "Pressure (hPa)", "timestamp": "Time (UTC)"},
        )
        fig_pres.update_layout(height=300, font=dict(family="Inter"))
        st.plotly_chart(fig_pres, use_container_width=True)

    # Daily summary table
    st.subheader("Daily Summary")
    daily_df = daily_summary(df)
    st.dataframe(daily_df.style.background_gradient(subset=["avg_temp"], cmap="RdYlBu_r"),
                 use_container_width=True, hide_index=True)

# ── Tab 2: Rankings ───────────────────────────────────────────────────────
with tab2:
    st.subheader("City Rankings — Current Conditions")

    ranking_df = city_rankings(df)

    # Bar chart
    fig_rank = px.bar(
        ranking_df, x="city", y="temperature_2m",
        color="temperature_2m", color_continuous_scale="RdYlBu_r",
        title="Current Temperature by City",
        labels={"temperature_2m": "Temperature (°C)", "city": "City"},
        template="plotly_white", text="temperature_2m",
    )
    fig_rank.update_traces(texttemplate="%{text:.1f}°C", textposition="outside")
    fig_rank.update_layout(height=350, font=dict(family="Inter"), showlegend=False)
    st.plotly_chart(fig_rank, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Temperature extremes
        st.subheader("Temperature Extremes (Last 72h)")
        ext_df = temperature_extremes(df)
        fig_ext = go.Figure()
        fig_ext.add_trace(go.Bar(name="Min", x=ext_df["city"], y=ext_df["min_temp"],
                                  marker_color="#74B9FF"))
        fig_ext.add_trace(go.Bar(name="Max", x=ext_df["city"], y=ext_df["max_temp"],
                                  marker_color="#E17055"))
        fig_ext.add_trace(go.Scatter(name="Avg", x=ext_df["city"], y=ext_df["avg_temp"],
                                      mode="markers+lines", marker=dict(size=10, color="#2D3436")))
        fig_ext.update_layout(barmode="overlay", template="plotly_white",
                               height=300, font=dict(family="Inter"))
        st.plotly_chart(fig_ext, use_container_width=True)

    with col2:
        # Wind speed
        st.subheader("Wind Speed Distribution")
        fig_wind = px.box(
            df, x="city", y="wind_speed_10m",
            color="city", color_discrete_map=CITY_COLORS,
            title="Wind Speed (km/h) Distribution",
            template="plotly_white",
        )
        fig_wind.update_layout(height=300, font=dict(family="Inter"), showlegend=False)
        st.plotly_chart(fig_wind, use_container_width=True)

    # Full rankings table
    st.subheader("Full Rankings Table")
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)

# ── Tab 3: Anomalies ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Anomaly Detection — Statistical Outliers")
    st.caption("Readings where temperature or pressure deviates >2 standard deviations from city rolling mean")

    # Detect anomalies
    anomaly_df = rolling_avg_temp(df).copy()
    anomaly_df["is_anomaly"] = (
        (anomaly_df["temperature_2m"] - anomaly_df["rolling_avg_6h"]).abs() >
        2 * anomaly_df["rolling_std_6h"]
    )
    anomaly_count = anomaly_df["is_anomaly"].sum()
    anomaly_rate  = anomaly_count / len(anomaly_df) if len(anomaly_df) > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Readings", f"{len(anomaly_df):,}")
    col2.metric("Anomalous Readings", f"{anomaly_count:,}")
    col3.metric("Anomaly Rate", f"{anomaly_rate:.1%}")

    # Anomaly scatter plot
    fig_anom = px.scatter(
        anomaly_df, x="timestamp", y="temperature_2m",
        color="city", color_discrete_map=CITY_COLORS,
        symbol="is_anomaly",
        symbol_map={True: "x", False: "circle"},
        title="Temperature Readings — anomalies marked with ×",
        template="plotly_white",
        labels={"temperature_2m": "Temperature (°C)"},
    )
    fig_anom.update_layout(height=400, font=dict(family="Inter"))
    st.plotly_chart(fig_anom, use_container_width=True)

    # Pressure anomalies
    st.subheader("Pressure Anomalies")
    pres_anom = pressure_anomalies(df)
    if pres_anom.empty:
        st.info("No pressure anomalies detected in the selected time window.")
    else:
        st.dataframe(pres_anom, use_container_width=True, hide_index=True)

    # Precipitation events
    st.subheader("Precipitation Events")
    precip_df = df[df["precipitation"] > 0.1][
        ["city", "timestamp", "precipitation", "temperature_2m"]
    ].sort_values("precipitation", ascending=False).head(50)
    if precip_df.empty:
        st.info("No significant precipitation in selected window.")
    else:
        fig_precip = px.bar(
            precip_df, x="timestamp", y="precipitation",
            color="city", color_discrete_map=CITY_COLORS,
            title="Precipitation Events (>0.1mm)",
            template="plotly_white",
        )
        fig_precip.update_layout(height=300, font=dict(family="Inter"))
        st.plotly_chart(fig_precip, use_container_width=True)

# ── Tab 4: Correlations ───────────────────────────────────────────────────
with tab4:
    st.subheader("Cross-Sensor Correlations")

    corr_df = temperature_correlation(df)
    st.dataframe(corr_df, use_container_width=True, hide_index=True)

    st.subheader("Scatter: Temperature vs Humidity")
    fig_scatter = px.scatter(
        df, x="temperature_2m", y="relative_humidity_2m",
        color="city", color_discrete_map=CITY_COLORS,
        trendline="ols",
        title="Temperature vs Relative Humidity",
        labels={"temperature_2m": "Temperature (°C)", "relative_humidity_2m": "Humidity (%)"},
        template="plotly_white",
    )
    fig_scatter.update_layout(height=400, font=dict(family="Inter"))
    st.plotly_chart(fig_scatter, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_s2 = px.scatter(
            df, x="temperature_2m", y="surface_pressure",
            color="city", color_discrete_map=CITY_COLORS,
            title="Temperature vs Pressure",
            template="plotly_white", trendline="ols",
        )
        fig_s2.update_layout(height=300, font=dict(family="Inter"))
        st.plotly_chart(fig_s2, use_container_width=True)

    with col2:
        fig_s3 = px.scatter(
            df, x="wind_speed_10m", y="temperature_2m",
            color="city", color_discrete_map=CITY_COLORS,
            title="Wind Speed vs Temperature",
            template="plotly_white", trendline="ols",
        )
        fig_s3.update_layout(height=300, font=dict(family="Inter"))
        st.plotly_chart(fig_s3, use_container_width=True)

# ── Tab 5: SQL Explorer ────────────────────────────────────────────────────
with tab5:
    st.subheader("SQL Query Explorer")
    st.caption("Select a query to see the SQL and results — powered by PostgreSQL + SQLAlchemy")

    query_names = list(QUERIES.keys())
    query_labels = [QUERIES[q]["title"] for q in query_names]

    selected_label = st.selectbox("Select Query", query_labels)
    selected_name  = query_names[query_labels.index(selected_label)]

    # Show SQL
    st.markdown("**SQL Query:**")
    st.code(QUERIES[selected_name]["sql"], language="sql")

    # Run query
    with st.spinner("Running query..."):
        result_df = run_query(selected_name, df)

    st.markdown(f"**Results:** {len(result_df):,} rows")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    # Download this query result
    st.download_button(
        label     = "📥 Download CSV",
        data      = to_csv(result_df),
        file_name = export_filename(f"query_{selected_name}", "csv"),
        mime      = "text/csv",
    )

# ── Tab 6: Export ──────────────────────────────────────────────────────────
with tab6:
    st.subheader("Export Data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Export raw sensor readings**")
        st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
        st.download_button(
            label     = "📥 Download CSV",
            data      = to_csv(df),
            file_name = export_filename("sensor_readings", "csv"),
            mime      = "text/csv",
            use_container_width=True,
        )

    with col2:
        st.markdown("**Export full analytics report (Excel)**")
        st.caption("All analytical query results in one Excel file, one sheet per query")
        with st.spinner("Generating Excel report..."):
            excel_data = to_excel({
                "Raw Readings":       df.head(500),
                "Daily Summary":      daily_summary(df),
                "City Rankings":      city_rankings(df),
                "Temp Extremes":      temperature_extremes(df),
                "Pressure Anomalies": pressure_anomalies(df),
                "Correlations":       temperature_correlation(df),
            })
        st.download_button(
            label     = "📊 Download Excel Report",
            data      = excel_data,
            file_name = export_filename("sensor_analytics_report", "xlsx"),
            mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.subheader("Data Preview")
    st.dataframe(df.tail(20), use_container_width=True, hide_index=True)