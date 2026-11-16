"""Bounded regional sampling using real NASA POWER point responses."""
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .analysis import analyze_series
from .nasa_power import fetch_power_daily, to_monthly


def build_grid_coordinates(south, north, west, east, spacing, max_points=64):
    if not (-90 <= south < north <= 90):
        raise ValueError("Latitude bounds must satisfy -90 <= south < north <= 90.")
    if not (-180 <= west < east <= 180):
        raise ValueError("Longitude bounds must satisfy -180 <= west < east <= 180.")
    if spacing <= 0:
        raise ValueError("Grid spacing must be positive.")
    latitudes = [south + i * spacing for i in range(int((north - south) / spacing) + 1)]
    longitudes = [west + i * spacing for i in range(int((east - west) / spacing) + 1)]
    coordinates = [(round(lat, 5), round(lon, 5)) for lat in latitudes for lon in longitudes]
    if len(coordinates) > max_points:
        raise ValueError(
            f"This selection creates {len(coordinates)} point requests; "
            f"the limit is {max_points}. Increase spacing or narrow the bounds."
        )
    return coordinates


def sample_power_grid(south, north, west, east, spacing, parameter, units,
                      aggregation, start, end, max_points=64, max_workers=4):
    coordinates = build_grid_coordinates(south, north, west, east, spacing, max_points)
    rows = []
    failures = []

    def fetch_one(latitude, longitude):
        daily = fetch_power_daily(latitude, longitude, start, end, [parameter])
        monthly = to_monthly(daily)
        analysis = analyze_series(monthly, parameter, units, aggregation)
        if analysis is None:
            return None
        return {
            "Latitude": latitude,
            "Longitude": longitude,
            "Trend per year": analysis["slope_per_year"],
            "Trend per decade": analysis["slope_per_decade"],
            "95% CI lower": analysis["ci_low"],
            "95% CI upper": analysis["ci_high"],
            "HAC p-value": analysis["hac_p_value"],
            "FDR q-value": None,
            "Observations": analysis["n"],
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_locations = {
            executor.submit(fetch_one, latitude, longitude): (latitude, longitude)
            for latitude, longitude in coordinates
        }
        for future in as_completed(future_locations):
            latitude, longitude = future_locations[future]
            try:
                result = future.result()
                if result is None:
                    failures.append({
                        "Latitude": latitude,
                        "Longitude": longitude,
                        "Reason": "Not enough continuous monthly observations for a trend test.",
                    })
                else:
                    rows.append(result)
            except Exception as exc:
                failures.append({"Latitude": latitude, "Longitude": longitude, "Reason": str(exc)})

    result_frame = pd.DataFrame(rows)
    if not result_frame.empty:
        result_frame = result_frame.sort_values(["Latitude", "Longitude"]).reset_index(drop=True)
    result_frame.attrs["requested_points"] = len(coordinates)
    result_frame.attrs["failed_points"] = failures
    return result_frame
