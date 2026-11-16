"""Public, no-login access to NASA POWER daily point observations."""
from __future__ import annotations

from calendar import monthrange
from datetime import datetime
from typing import Iterable

import pandas as pd
import requests

from .catalog import VARIABLES, PRECIPITATION_CODES

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
ALLOWED_PARAMETERS = {item["code"] for item in VARIABLES.values()}


def _validate_request(latitude: float, longitude: float, start: str, end: str,
                     parameters: Iterable[str]) -> list[str]:
    latitude = float(latitude)
    longitude = float(longitude)
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees.")
    try:
        start_date = datetime.strptime(start, "%Y%m%d").date()
        end_date = datetime.strptime(end, "%Y%m%d").date()
    except (TypeError, ValueError) as exc:
        raise ValueError("Dates must use YYYYMMDD format.") from exc
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    selected = list(dict.fromkeys(parameters))
    if not selected:
        raise ValueError("Select at least one NASA POWER parameter.")
    unknown = sorted(set(selected) - ALLOWED_PARAMETERS)
    if unknown:
        raise ValueError(f"Unsupported NASA POWER parameter(s): {', '.join(unknown)}")
    return selected


def fetch_power_daily(latitude: float, longitude: float, start: str, end: str,
                      parameters: Iterable[str] | None = None,
                      timeout: int = 90) -> pd.DataFrame:
    """Download real daily values from NASA POWER; no account is required.

    Missing POWER fill values (-999) are converted to NaN. The returned frame
    carries source and request information in ``DataFrame.attrs``.
    """
    selected = _validate_request(
        latitude, longitude, start, end,
        parameters if parameters is not None else ALLOWED_PARAMETERS,
    )
    query = {
        "parameters": ",".join(selected),
        "community": "RE",
        "longitude": float(longitude),
        "latitude": float(latitude),
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": "UTC",
    }
    try:
        response = requests.get(BASE_URL, params=query, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"NASA POWER request failed: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("NASA POWER returned a response that was not valid JSON.") from exc

    try:
        parameter_data = payload["properties"]["parameter"]
    except (KeyError, TypeError) as exc:
        messages = payload.get("messages") if isinstance(payload, dict) else None
        detail = "; ".join(map(str, messages)) if messages else str(payload)[:350]
        raise RuntimeError(f"NASA POWER response did not contain parameter data: {detail}") from exc

    columns: dict[str, pd.Series] = {}
    missing_parameters: list[str] = []
    for code in selected:
        values = parameter_data.get(code)
        if not isinstance(values, dict) or not values:
            missing_parameters.append(code)
            continue
        series = pd.Series(values, name=code)
        series.index = pd.to_datetime(series.index, format="%Y%m%d", errors="coerce")
        series = pd.to_numeric(series, errors="coerce")
        series = series[series.index.notna()].sort_index()
        series = series.replace([-999, -999.0], float("nan"))
        columns[code] = series

    if not columns:
        missing = ", ".join(missing_parameters) or ", ".join(selected)
        raise RuntimeError(f"NASA POWER returned no usable values for: {missing}.")

    frame = pd.concat(columns.values(), axis=1).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.index.name = "date"
    frame.attrs["source"] = "NASA POWER Daily Point API"
    frame.attrs["source_url"] = getattr(response, "url", BASE_URL)
    frame.attrs["requested_parameters"] = selected
    frame.attrs["missing_parameters"] = missing_parameters
    frame.attrs["latitude"] = float(latitude)
    frame.attrs["longitude"] = float(longitude)
    frame.attrs["start"] = start
    frame.attrs["end"] = end
    return frame


def to_monthly(daily: pd.DataFrame, mean_coverage: float = 0.80,
               precipitation_coverage: float = 0.95) -> pd.DataFrame:
    """Aggregate daily data to complete-enough calendar months.

    Means require at least 80% of calendar days. Monthly precipitation totals
    require at least 95% of days so a few missing wet days do not create a
    misleadingly low monthly total. Months failing a threshold remain missing.
    """
    if daily.empty:
        return pd.DataFrame()
    frame = daily.copy().sort_index()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.apply(pd.to_numeric, errors="coerce")

    month_index = pd.date_range(
        frame.index.min().to_period("M").to_timestamp(),
        frame.index.max().to_period("M").to_timestamp(),
        freq="MS",
        name="date",
    )
    expected_days = pd.Series(
        [monthrange(ts.year, ts.month)[1] for ts in month_index],
        index=month_index,
        dtype=float,
    )
    valid_counts = frame.notna().resample("MS").sum().reindex(month_index).fillna(0)
    coverage = valid_counts.div(expected_days, axis=0)
    means = frame.resample("MS").mean().reindex(month_index)
    totals = frame.resample("MS").sum(min_count=1).reindex(month_index)

    monthly = pd.DataFrame(index=month_index)
    for column in frame.columns:
        if column in PRECIPITATION_CODES:
            threshold = precipitation_coverage
            monthly[column] = totals[column].where(coverage[column] >= threshold)
        else:
            threshold = mean_coverage
            monthly[column] = means[column].where(coverage[column] >= threshold)
    monthly.index.name = "date"
    monthly.attrs["coverage"] = coverage
    monthly.attrs["source"] = daily.attrs.get("source", "NASA POWER Daily Point API")
    monthly.attrs["request"] = {
        key: daily.attrs.get(key)
        for key in ("latitude", "longitude", "start", "end", "source_url")
    }
    return monthly
