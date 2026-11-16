#This was made with the help of SI(ChatGPT)
from __future__ import annotations

import json
from calendar import monthrange
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from geopy.geocoders import Nominatim

from trendlab.analysis import (
    add_fdr,
    analyze_series,
    detect_candidate_shifts,
    kalman_local_linear_trend,
)
from trendlab.catalog import VARIABLES
from trendlab.grid import sample_power_grid
from trendlab.nasa_power import fetch_power_daily, to_monthly

st.set_page_config(page_title="TrendLab | Earth System Trend Detective", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1500px;}
.hero {padding: 1.35rem 1.55rem; border: 1px solid rgba(120,150,180,.28);
       border-radius: 16px; background: linear-gradient(115deg, rgba(24,65,94,.24), rgba(18,115,101,.10));}
.eyebrow {letter-spacing: .15em; text-transform: uppercase; font-size: .72rem; opacity: .72;}
.small-note {font-size: .84rem; opacity: .76;}
.kpi {font-size: .82rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Earth system trend detective</div>
  <h1 style="margin:.25rem 0;">TrendLab</h1>
  <p style="font-size:1.08rem;margin-bottom:.3rem;">
    Discover what is changing, where it is changing, how quickly it is changing,
    and whether the evidence is statistically significant.
  </p>
  <p class="small-note" style="margin:0;">
    Live observations from NASA POWER · variable discovery · regional divergence · spatial trend screening
  </p>
</div>
""", unsafe_allow_html=True)

REFERENCE_LOCATIONS = {
    "Tampa, Florida, USA": (27.9506, -82.4572),
    "Miami, Florida, USA": (25.7617, -80.1918),
    "New York City, USA": (40.7128, -74.0060),
    "Los Angeles, USA": (34.0522, -118.2437),
    "Seattle, USA": (47.6062, -122.3321),
    "Phoenix, Arizona, USA": (33.4484, -112.0740),
    "Toronto, Canada": (43.6532, -79.3832),
    "Mexico City, Mexico": (19.4326, -99.1332),
    "Reykjavik, Iceland": (64.1466, -21.9426),
    "London, United Kingdom": (51.5072, -0.1276),
    "Paris, France": (48.8566, 2.3522),
    "Cairo, Egypt": (30.0444, 31.2357),
    "Lagos, Nigeria": (6.5244, 3.3792),
    "Nairobi, Kenya": (-1.2921, 36.8219),
    "Cape Town, South Africa": (-33.9249, 18.4241),
    "Mumbai, India": (19.0760, 72.8777),
    "Delhi, India": (28.6139, 77.2090),
    "Singapore": (1.3521, 103.8198),
    "Tokyo, Japan": (35.6762, 139.6503),
    "Seoul, South Korea": (37.5665, 126.9780),
    "Beijing, China": (39.9042, 116.4074),
    "Jakarta, Indonesia": (-6.2088, 106.8456),
    "Sydney, Australia": (-33.8688, 151.2093),
    "Auckland, New Zealand": (-36.8509, 174.7645),
    "Manaus, Brazil": (-3.1190, -60.0217),
    "Sao Paulo, Brazil": (-23.5505, -46.6333),
    "Buenos Aires, Argentina": (-34.6037, -58.3816),
}


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_place(place_name: str):
    geocoder = Nominatim(user_agent="TrendLab_EarthSystem_Trend_Detective")
    result = geocoder.geocode(place_name, exactly_one=True, timeout=15)
    if result is None:
        return None
    return float(result.latitude), float(result.longitude), result.address


@st.cache_data(ttl=21600, show_spinner=False)
def cached_daily(latitude: float, longitude: float, start: str, end: str,
                parameters: tuple[str, ...]) -> pd.DataFrame:
    return fetch_power_daily(latitude, longitude, start, end, parameters)


# The API is daily; the interface selects calendar months so trend records have
# complete month boundaries. The current incomplete month is not offered.
today = date.today()
last_complete_month_end = today.replace(day=1) - timedelta(days=1)
first_available_date = date(1981, 1, 1)

with st.sidebar:
    st.header("Investigation setup")
    st.caption("Live data is retrieved from the public NASA POWER API.")
    st.divider()
    st.subheader("Study location")
    location_name = st.text_input("Location label", "Tampa, Florida, USA")
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0,
                               value=27.9506, step=0.01, format="%.4f")
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0,
                                value=-82.4572, step=0.01, format="%.4f")

    st.markdown("**Add a comparison location**")
    place_to_add = st.text_input("City, region, or country", placeholder="e.g. Bengaluru, India")
    if st.button("Find and add location"):
        if place_to_add.strip():
            try:
                with st.spinner(f"Looking up {place_to_add.strip()}..."):
                    match = geocode_place(place_to_add.strip())
                if match:
                    st.session_state.setdefault("custom_locations", {})[place_to_add.strip()] = (match[0], match[1])
                    st.success(f"Added {place_to_add.strip()}: {match[0]:.4f}, {match[1]:.4f}")
                else:
                    st.warning("No match found. Try adding a country or enter coordinates manually.")
            except Exception as exc:
                st.warning(f"Place lookup is unavailable: {exc}")

    with st.expander("Add location by coordinates"):
        custom_label = st.text_input("Location name", key="custom_location_label")
        custom_lat = st.number_input("Custom latitude", -90.0, 90.0, 0.0, format="%.5f", key="custom_lat")
        custom_lon = st.number_input("Custom longitude", -180.0, 180.0, 0.0, format="%.5f", key="custom_lon")
        if st.button("Save coordinate location"):
            if custom_label.strip():
                st.session_state.setdefault("custom_locations", {})[custom_label.strip()] = (float(custom_lat), float(custom_lon))
                st.success(f"Saved {custom_label.strip()}")
            else:
                st.warning("Enter a location name first.")

    st.divider()
    st.subheader("Analysis period")
    default_start = date(2001, 1, 1)
    start_input = st.date_input("Start month", value=default_start,
                                min_value=first_available_date,
                                max_value=last_complete_month_end)
    end_input = st.date_input("End month", value=last_complete_month_end,
                              min_value=first_available_date,
                              max_value=last_complete_month_end)
    start_date = date(start_input.year, start_input.month, 1)
    end_date = date(end_input.year, end_input.month, monthrange(end_input.year, end_input.month)[1])
    st.caption("Dates are expanded to full calendar months. The latest incomplete month is excluded.")
    if start_date > end_date:
        st.error("The start month must be before the end month.")

    st.divider()
    st.subheader("Variable discovery")
    all_labels = list(VARIABLES.keys())
    selected_scan_labels = st.multiselect(
        "Variables to screen",
        options=all_labels,
        default=all_labels,
        help="TrendLab screens the selected variables at the study coordinates and adjusts their p-values for multiple testing.",
    )
    primary_label = st.selectbox("Variable to investigate in detail", options=all_labels, index=0)
    st.caption("A minimum of five continuous years of monthly observations is required for trend inference.")

    refresh = st.button("Refresh NASA observations", use_container_width=True)

if start_date > end_date:
    st.stop()

primary_meta = VARIABLES[primary_label]
primary_code = primary_meta["code"]
scan_parameters = [VARIABLES[label]["code"] for label in selected_scan_labels]
request_parameters = tuple(dict.fromkeys(scan_parameters + [primary_code]))
start_key = start_date.strftime("%Y%m%d")
end_key = end_date.strftime("%Y%m%d")

if refresh:
    cached_daily.clear()

try:
    with st.spinner("Retrieving real NASA POWER observations..."):
        daily = cached_daily(float(latitude), float(longitude), start_key, end_key, request_parameters)
        monthly = to_monthly(daily)
except Exception as exc:
    st.error("NASA POWER observations could not be retrieved.")
    st.code(str(exc))
    st.info("Check your internet connection, coordinates, and selected date range. TrendLab does not substitute simulated data when NASA data are unavailable.")
    st.stop()

if monthly.empty or primary_code not in monthly.columns:
    st.error("NASA POWER returned no usable observations for the selected investigation.")
    st.stop()

primary_analysis = analyze_series(monthly, primary_code, primary_meta["units"], primary_meta["monthly_aggregation"])

# Scan every selected catalog variable using the same downloaded observation set.
scan_results = []
not_analyzed = []
for label in selected_scan_labels:
    meta = VARIABLES[label]
    result = analyze_series(monthly, meta["code"], meta["units"], meta["monthly_aggregation"])
    if result is None:
        not_analyzed.append(label)
        continue
    scan_results.append({
        "Variable": label,
        "Parameter": meta["code"],
        "Data family": meta["family"],
        "Trend per year": result["slope_per_year"],
        "Trend per decade": result["slope_per_decade"],
        "95% CI lower / year": result["ci_low"],
        "95% CI upper / year": result["ci_high"],
        "Direction": result["direction"],
        "HAC p-value": result["hac_p_value"],
        "Kendall tau": result["kendall_tau"],
        "Observations (months)": result["n"],
        "Standardized effect / decade": result["standardized_effect_per_decade"],
    })
if scan_results:
    scan_results = add_fdr(scan_results, "HAC p-value")
    scan_table = pd.DataFrame(scan_results).rename(columns={"q_value": "FDR q-value"})
    scan_table["Evidence after FDR"] = scan_table["FDR q-value"] < 0.05
    scan_table = scan_table.sort_values(["FDR q-value", "Standardized effect / decade"],
                                        ascending=[True, False], na_position="last")
else:
    scan_table = pd.DataFrame()

# Carry the primary variable's adjusted q-value into its detailed view, where
# possible. A p-value alone is not called a discovery after a multi-variable scan.
primary_q = None
if not scan_table.empty:
    match = scan_table.loc[scan_table["Parameter"] == primary_code, "FDR q-value"]
    if not match.empty:
        primary_q = float(match.iloc[0])

source_url = daily.attrs.get("source_url", "https://power.larc.nasa.gov/api/temporal/daily/point")
missing_parameters = daily.attrs.get("missing_parameters", [])


tab_discovery, tab_trend, tab_regions, tab_grid, tab_regime, tab_methods = st.tabs([
    "Variable discovery",
    "Trend explorer",
    "Regional divergence",
    "Spatial trend map",
    "Regime shifts",
    "Methods and data provenance",
])

with tab_discovery:
    st.subheader("Discover which variables are changing")
    st.write(
        f"TrendLab screened the selected NASA POWER variables at **{location_name}** "
        f"({latitude:.4f}, {longitude:.4f}) over {start_date:%b %Y} to {end_date:%b %Y}. "
        "Each trend is estimated after removing the mean calendar-month cycle."
    )
    d1, d2, d3 = st.columns(3)
    d1.metric("Variables screened", len(scan_table))
    d2.metric("FDR-significant trends", int((scan_table["FDR q-value"] < 0.05).sum()) if not scan_table.empty else 0)
    d3.metric("Monthly record", f"{len(monthly)} months")

    if not scan_table.empty:
        st.dataframe(scan_table, use_container_width=True, hide_index=True,
                     column_config={
                         "Trend per year": st.column_config.NumberColumn(format="%.4g"),
                         "Trend per decade": st.column_config.NumberColumn(format="%.4g"),
                         "HAC p-value": st.column_config.NumberColumn(format="%.3g"),
                         "FDR q-value": st.column_config.NumberColumn(format="%.3g"),
                         "95% CI lower / year": st.column_config.NumberColumn(format="%.4g"),
                         "95% CI upper / year": st.column_config.NumberColumn(format="%.4g"),
                         "Standardized effect / decade": st.column_config.NumberColumn(format="%.3g"),
                     })
        st.download_button(
            "Download variable discovery results",
            scan_table.to_csv(index=False).encode("utf-8"),
            "trendlab_variable_discovery.csv",
            "text/csv",
        )
        discoveries = scan_table[scan_table["FDR q-value"] < 0.05]
        if not discoveries.empty:
            st.success(
                f"{len(discoveries)} selected variable(s) passed the 5% Benjamini–Hochberg FDR threshold "
                "within this one location, date range, and selected variable set."
            )
            strongest = discoveries.iloc[0]
            st.markdown(
                f"**Strongest statistical candidate in this scan:** {strongest['Variable']} "
                f"({strongest['Direction'].lower()}, q={strongest['FDR q-value']:.3g}). "
                "This is a candidate trend for scientific follow-up, not proof of a cause."
            )
        else:
            st.info("No selected variable passed the 5% FDR threshold. This does not establish that the variables are unchanging.")
    else:
        st.warning("No selected variables had enough continuous monthly data for trend inference.")

    if not_analyzed:
        st.caption("Not analyzed because the data did not meet the continuity or five-year minimum: " + ", ".join(not_analyzed))

    st.markdown("#### What these variables measure")
    catalog_rows = []
    for label in selected_scan_labels:
        meta = VARIABLES[label]
        catalog_rows.append({
            "Variable": label,
            "NASA POWER code": meta["code"],
            "Family": meta["family"],
            "Units": meta["units"],
            "Definition": meta["description"],
            "Why investigate it?": meta["why_it_matters"],
        })
    if catalog_rows:
        st.dataframe(pd.DataFrame(catalog_rows), use_container_width=True, hide_index=True)
    if missing_parameters:
        st.warning("NASA POWER did not return these requested parameters for this location/date selection: " + ", ".join(missing_parameters))
    st.caption(
        "Discovery scope: this automated screen covers the selected variables in TrendLab's NASA POWER catalog, "
        "not every variable available from every Earth science mission or model. FDR correction applies only to "
        "the variables in this scan; changing locations, periods, or selections creates additional tests."
    )

with tab_trend:
    st.subheader("Investigate one environmental trend")
    st.write(f"**{primary_label}** near **{location_name}**")
    if primary_analysis is None:
        st.warning(
            "A trend test was withheld. TrendLab requires at least 60 continuous monthly observations. "
            "It will not compress missing months and treat them as consecutive time steps."
        )
        st.markdown("#### Monthly data returned by NASA POWER")
        st.dataframe(monthly[[primary_code]].rename(columns={primary_code: primary_label}), use_container_width=True)
    else:
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Trend per decade", f"{primary_analysis['slope_per_decade']:+.4g} {primary_meta['units']}/decade")
        a2.metric("95% HAC interval / year", f"[{primary_analysis['ci_low']:+.3g}, {primary_analysis['ci_high']:+.3g}]")
        a3.metric("HAC p-value", f"{primary_analysis['hac_p_value']:.3g}")
        a4.metric("FDR q-value", f"{primary_q:.3g}" if primary_q is not None else "Not screened")

        if primary_q is not None and primary_q < 0.05:
            st.success("This variable passed the 5% FDR threshold within the selected variable scan.")
        elif primary_analysis["hac_p_value"] < 0.05:
            st.warning("The unadjusted HAC p-value is below 0.05, but the variable did not pass the scan's FDR threshold. Treat this as exploratory.")
        else:
            st.info("The trend is not statistically significant at the 0.05 level under the current model. This is not proof of no change.")

        st.caption(
            f"{primary_analysis['n']} continuous monthly observations · seasonal cycle removed · "
            f"Newey–West HAC standard errors with {primary_analysis['hac_maxlags']} monthly lags · "
            f"Kendall tau={primary_analysis['kendall_tau']:+.3f} (supplementary, not autocorrelation-corrected)"
        )

        frame = primary_analysis["frame"]
        raw_fig = go.Figure()
        raw_fig.add_trace(go.Scatter(x=frame.index, y=frame["value"], mode="lines", name="Monthly value"))
        raw_fig.update_layout(title="Observed monthly values from NASA POWER", xaxis_title="Month",
                              yaxis_title=primary_meta["units"], height=370, margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(raw_fig, use_container_width=True)

        anomaly_fig = go.Figure()
        anomaly_fig.add_trace(go.Scatter(x=frame.index, y=frame["seasonal_anomaly"], mode="lines",
                                         name="Seasonal anomaly", line=dict(width=1.3)))
        anomaly_fig.add_trace(go.Scatter(x=frame.index, y=frame["fitted_linear_trend"], mode="lines",
                                         name="Fitted linear trend", line=dict(width=2.5)))
        anomaly_fig.update_layout(title="Seasonally adjusted signal and fitted linear trend",
                                  xaxis_title="Month", yaxis_title=f"Anomaly ({primary_meta['units']})",
                                  height=390, margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(anomaly_fig, use_container_width=True)
        st.write(primary_analysis["interpretation"])

        coverage = monthly.attrs.get("coverage")
        if isinstance(coverage, pd.DataFrame) and primary_code in coverage:
            cov = coverage[primary_code]
            st.caption(f"Monthly data completeness: {cov.min() * 100:.1f}% minimum across returned months; months below the defined completeness threshold are excluded from inference.")

        c1, c2 = st.columns(2)
        c1.download_button("Download NASA POWER daily data", daily.to_csv(index=True).encode("utf-8"),
                           "trendlab_nasa_power_daily.csv", "text/csv")
        c2.download_button("Download monthly analysis data", frame.to_csv(index=True).encode("utf-8"),
                           "trendlab_monthly_trend_data.csv", "text/csv")

        with st.expander("View monthly observations and anomalies"):
            st.dataframe(frame, use_container_width=True)

with tab_regions:
    st.subheader("Find regions with opposing trends")
    st.write(
        "Compare the same environmental variable at separate locations. Positive and negative slopes can reveal "
        "regional divergence even though all locations are analyzed with the same method."
    )
    locations = {**REFERENCE_LOCATIONS, **st.session_state.get("custom_locations", {})}
    default_locations = [
        name for name in ["Tampa, Florida, USA", "Phoenix, Arizona, USA", "Manaus, Brazil"]
        if name in locations
    ]
    selected_locations = st.multiselect(
        "Locations to compare", options=list(locations.keys()), default=default_locations,
        max_selections=8,
    )
    st.caption("These are point comparisons, not city-wide or country-wide averages. Each location uses the same dates and variable.")
    if st.button("Analyze selected locations", type="primary", key="run_regional_comparison"):
        if len(selected_locations) < 2:
            st.warning("Select at least two locations.")
        else:
            regional_rows = []
            anomaly_frames = {}
            failed_locations = []
            with st.spinner("Retrieving the selected variable for each location..."):
                for name in selected_locations:
                    point_lat, point_lon = locations[name]
                    try:
                        point_daily = cached_daily(float(point_lat), float(point_lon), start_key, end_key, (primary_code,))
                        point_monthly = to_monthly(point_daily)
                        result = analyze_series(point_monthly, primary_code, primary_meta["units"], primary_meta["monthly_aggregation"])
                        if result is None:
                            failed_locations.append(f"{name}: insufficient continuous monthly observations")
                            continue
                        regional_rows.append({
                            "Location": name,
                            "Latitude": point_lat,
                            "Longitude": point_lon,
                            "Trend per year": result["slope_per_year"],
                            "Trend per decade": result["slope_per_decade"],
                            "95% CI lower": result["ci_low"],
                            "95% CI upper": result["ci_high"],
                            "HAC p-value": result["hac_p_value"],
                            "Direction": result["direction"],
                            "Months": result["n"],
                        })
                        anomaly_frames[name] = result["frame"]["seasonal_anomaly"]
                    except Exception as exc:
                        failed_locations.append(f"{name}: {exc}")

            if regional_rows:
                regional_rows = add_fdr(regional_rows, "HAC p-value")
                regional = pd.DataFrame(regional_rows).rename(columns={"q_value": "FDR q-value"})
                regional["FDR-significant"] = regional["FDR q-value"] < 0.05
                st.dataframe(regional, use_container_width=True, hide_index=True)
                map_fig = px.scatter_geo(
                    regional, lat="Latitude", lon="Longitude", color="Trend per decade",
                    hover_name="Location",
                    hover_data={"HAC p-value": ":.3g", "FDR q-value": ":.3g", "Direction": True,
                                "Latitude": ":.3f", "Longitude": ":.3f"},
                    color_continuous_scale="RdBu", color_continuous_midpoint=0,
                    projection="natural earth", title=f"Regional trend divergence: {primary_label}",
                )
                map_fig.update_layout(height=470, margin=dict(l=0, r=0, t=55, b=0))
                st.plotly_chart(map_fig, use_container_width=True)

                anomaly_fig = go.Figure()
                for name, series in anomaly_frames.items():
                    anomaly_fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=name))
                anomaly_fig.update_layout(title="Seasonal anomalies across selected locations",
                                          xaxis_title="Month", yaxis_title=f"Anomaly ({primary_meta['units']})",
                                          height=420)
                st.plotly_chart(anomaly_fig, use_container_width=True)

                slopes = regional["Trend per year"].dropna()
                if len(slopes) >= 2 and slopes.min() < 0 < slopes.max():
                    st.success("Opposing trend directions were detected among the selected point locations.")
                else:
                    st.info("No positive-versus-negative trend split was detected among the analyzed points.")
                st.download_button("Download regional comparison", regional.to_csv(index=False).encode("utf-8"),
                                   "trendlab_regional_comparison.csv", "text/csv")
            else:
                st.warning("No selected location returned enough continuous observations for a trend test.")
            if failed_locations:
                with st.expander("Locations not analyzed"):
                    st.write("\n".join(failed_locations))

with tab_grid:
    st.subheader("Map where a trend is changing")
    st.write(
        "Sample independent NASA POWER point requests over a user-defined rectangle. "
        "Each point receives its own trend estimate; p-values are corrected across the analyzed points."
    )
    g1, g2, g3, g4 = st.columns(4)
    south = g1.number_input("South latitude", -90.0, 89.0, 29.5, 0.5)
    north = g2.number_input("North latitude", -89.0, 90.0, 31.5, 0.5)
    west = g3.number_input("West longitude", -180.0, 179.0, -82.5, 0.5)
    east = g4.number_input("East longitude", -179.0, 180.0, -80.5, 0.5)
    spacing = st.select_slider("Sampling spacing (degrees)", options=[0.5, 1.0, 2.0, 5.0], value=1.0)
    st.caption("Maximum 64 point requests. Sampling points more densely than the native source grid does not create new spatial resolution.")
    if st.button("Run spatial trend analysis", type="primary", key="run_grid"):
        try:
            with st.spinner("Retrieving NASA POWER observations at each grid point and estimating trends..."):
                grid = sample_power_grid(
                    south, north, west, east, spacing, primary_code, primary_meta["units"],
                    primary_meta["monthly_aggregation"], start_key, end_key,
                )
            grid_failures = grid.attrs.get("failed_points", [])
            if grid.empty:
                st.warning("No grid point returned enough continuous observations for inference.")
            else:
                grid_records = grid.to_dict("records")
                grid_records = add_fdr(grid_records, "HAC p-value")
                # sample_power_grid includes an empty FDR q-value placeholder.
                # Remove it before renaming the newly calculated q_value column,
                # otherwise pandas creates duplicate column labels.
                grid = (
                    pd.DataFrame(grid_records)
                    .drop(columns=["FDR q-value"], errors="ignore")
                    .rename(columns={"q_value": "FDR q-value"})
                )
                grid["FDR-significant"] = grid["FDR q-value"].lt(0.05)
                st.metric("Analyzed points", len(grid))
                st.dataframe(grid, use_container_width=True, hide_index=True)
                grid_fig = px.scatter_geo(
                    grid, lat="Latitude", lon="Longitude", color="Trend per decade",
                    size="Observations", color_continuous_scale="RdBu", color_continuous_midpoint=0,
                    hover_data={"HAC p-value": ":.3g", "FDR q-value": ":.3g", "FDR-significant": True},
                    projection="natural earth", title=f"Spatial trend pattern: {primary_label}",
                )
                grid_fig.update_layout(height=490, margin=dict(l=0, r=0, t=55, b=0))
                st.plotly_chart(grid_fig, use_container_width=True)
                st.download_button("Download spatial trend results", grid.to_csv(index=False).encode("utf-8"),
                                   "trendlab_spatial_trends.csv", "text/csv")
            if grid_failures:
                with st.expander(f"{len(grid_failures)} point(s) not analyzed"):
                    st.dataframe(pd.DataFrame(grid_failures), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Spatial analysis could not be completed: {exc}")

with tab_regime:
    st.subheader("Explore potential environmental regime shifts")
    st.write(
        "This is an exploratory extension to the challenge's core linear-trend analysis. "
        "It highlights possible changes in the behavior of the seasonally adjusted series; "
        "it does not establish that a physical regime shift occurred."
    )
    if primary_analysis is None:
        st.warning("At least five continuous years of monthly observations are required before shift analysis is displayed.")
    else:
        regime_frame = primary_analysis["frame"].dropna(subset=["seasonal_anomaly"]).copy()
        kalman = kalman_local_linear_trend(regime_frame["seasonal_anomaly"].to_numpy())
        regime_frame["filtered_level"] = kalman["level"]
        regime_frame["filtered_slope"] = kalman["slope"]
        candidate_dates = detect_candidate_shifts(regime_frame["seasonal_anomaly"], window=12)
        r1, r2 = st.columns(2)
        r1.metric("Candidate shift boundaries", len(candidate_dates))
        r2.metric("Latest filtered slope", f"{regime_frame['filtered_slope'].iloc[-1]:+.4g} {primary_meta['units']}/month")
        regime_fig = go.Figure()
        regime_fig.add_trace(go.Scatter(x=regime_frame.index, y=regime_frame["seasonal_anomaly"],
                                        mode="lines", name="Seasonal anomaly", opacity=0.55))
        regime_fig.add_trace(go.Scatter(x=regime_frame.index, y=regime_frame["filtered_level"],
                                        mode="lines", name="Kalman filtered level", line=dict(width=2.4)))
        for candidate in candidate_dates:
            regime_fig.add_vline(x=candidate, line_dash="dash", opacity=0.65)
        regime_fig.update_layout(title="Exploratory filtered signal and candidate mean shifts",
                                 xaxis_title="Month", yaxis_title=f"Anomaly ({primary_meta['units']})",
                                 height=440, margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(regime_fig, use_container_width=True)
        if candidate_dates:
            st.write("Candidate boundaries: " + ", ".join(pd.Timestamp(item).strftime("%Y-%m") for item in candidate_dates))
        else:
            st.info("No candidate mean-shift boundary was selected under the current penalty and minimum segment size.")
        st.warning(
            "Change-point dates depend on the segmentation penalty, data window, and minimum segment length. "
            "Check sensitivity to these choices and compare with independent physical evidence before interpreting them."
        )

with tab_methods:
    st.subheader("Methods, sources, and scientific limits")
    st.markdown("""
**Live source:** NASA POWER Daily Point API. TrendLab requests actual daily NASA POWER values for the entered coordinates and dates. The application does not generate replacement observations or silently switch to demo data.

**Variables currently available in the live catalog:**

- `T2M`, `T2M_MAX`, `T2M_MIN`: 2-metre air temperature statistics.
- `PRECTOTCORR`: corrected precipitation.
- `RH2M`: relative humidity at 2 metres.
- `WS10M`: wind speed at 10 metres.
- `ALLSKY_SFC_SW_DWN`: all-sky downward shortwave energy at the surface.

**Temporal processing:** Daily values are aggregated to complete calendar months. Precipitation is summed only when at least 95% of the days in the month are present. Other variables are averaged only when at least 80% of days are present. Months failing these completeness rules are not used for trend inference.

**Trend analysis:** TrendLab removes the mean calendar-month cycle, fits a linear trend to the resulting monthly anomalies, and estimates uncertainty with Newey–West heteroskedasticity-and-autocorrelation-consistent standard errors. It reports a 95% confidence interval and p-value. Benjamini–Hochberg false-discovery-rate adjustment is applied across the variables in the discovery scan, the locations in a regional comparison, or the points in one spatial scan.

**Minimum record:** At least 60 consecutive monthly observations (five years) are required. If months are missing, the trend test is withheld rather than compressing time and misrepresenting the HAC lag spacing.

**Data provenance:** NASA POWER is an analysis-ready NASA data service that combines Earth-observation and model-based inputs; individual parameter provenance and spatial resolution vary. TrendLab identifies its results as NASA POWER point time series.

**Limits:** This release is point-based for the trend explorer and regional comparison. The spatial tool samples multiple points but is not a native high-resolution satellite raster or an area-weighted regional average. FDR correction only covers the tests displayed in that one scan. Statistical significance does not establish causation, and a statistically significant slope may still be small or not practically important.
""")
    st.markdown("**NASA data links**")
    st.markdown(
        "- [NASA POWER Daily API documentation](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)\n"
        "- [NASA POWER data access](https://power.larc.nasa.gov/)"
    )
    st.markdown("**Current request provenance**")
    provenance = {
        "source": "NASA POWER Daily Point API",
        "endpoint": source_url,
        "location_label": location_name,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": start_key,
        "end_date": end_key,
        "requested_parameters": list(request_parameters),
        "monthly_observations_returned": int(len(monthly)),
        "downloaded_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    st.json(provenance)
    st.download_button("Download provenance record", json.dumps(provenance, indent=2),
                       "trendlab_data_provenance.json", "application/json")

