"""Trend estimation, multiplicity control, and exploratory shift detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

MIN_MONTHS_FOR_TREND = 60


def seasonal_anomalies(series: pd.Series) -> pd.Series:
    """Remove the mean calendar-month cycle while retaining interannual change."""
    values = series.astype(float).copy()
    climatology = values.groupby(values.index.month).transform("mean")
    return values - climatology


def analyze_series(monthly: pd.DataFrame, parameter: str, units: str,
                   aggregation: str = "mean", min_months: int = MIN_MONTHS_FOR_TREND):
    """Estimate a linear trend with Newey-West HAC inference.

    A continuous monthly record is required for the HAC lag interpretation.
    The Mann-Kendall statistic is reported only as a supplementary rank-based
    description; its p-value is not corrected for serial autocorrelation.
    """
    if parameter not in monthly.columns:
        return None
    raw = monthly[parameter].astype(float).copy()
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw = raw.dropna()
    if len(raw) < min_months:
        return None

    full_index = pd.date_range(raw.index.min().to_period("M").to_timestamp(),
                               raw.index.max().to_period("M").to_timestamp(), freq="MS")
    if len(raw) != len(full_index) or not raw.index.equals(full_index):
        return None
    if not np.isfinite(raw.to_numpy()).all():
        return None

    anomaly = seasonal_anomalies(raw)
    if len(anomaly) < min_months:
        return None

    x = (raw.index.year.to_numpy(dtype=float)
         + (raw.index.month.to_numpy(dtype=float) - 0.5) / 12.0)
    x_centered = x - x.mean()
    y = anomaly.to_numpy(dtype=float)
    design = sm.add_constant(x_centered)
    maxlags = min(12, max(1, len(y) // 4))
    fitted = sm.OLS(y, design).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}, use_t=False
    )

    slope = float(fitted.params[1])
    ci = fitted.conf_int(alpha=0.05)
    ci_low, ci_high = float(ci[1, 0]), float(ci[1, 1])
    p_value = float(fitted.pvalues[1])
    tau, mk_p = stats.kendalltau(x, y, nan_policy="omit")
    residual_fit = pd.Series(fitted.predict(design), index=raw.index)
    standard_deviation = float(np.std(y, ddof=1))
    standardized_per_decade = (slope * 10 / standard_deviation
                               if standard_deviation > 0 else float("nan"))

    frame = pd.DataFrame({
        "value": raw,
        "seasonal_anomaly": anomaly,
        "fitted_linear_trend": residual_fit,
    }, index=raw.index)
    direction = "Increasing" if slope > 0 else ("Decreasing" if slope < 0 else "No linear trend")
    return {
        "frame": frame,
        "slope_per_year": slope,
        "slope_per_decade": slope * 10,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_low_per_decade": ci_low * 10,
        "ci_high_per_decade": ci_high * 10,
        "p_value": p_value,
        "hac_p_value": p_value,
        "hac_maxlags": int(maxlags),
        "mk_p_value": float(mk_p),
        "kendall_tau": float(tau),
        "direction": direction,
        "n": int(len(y)),
        "coverage_percent": 100.0,
        "standardized_effect_per_decade": standardized_per_decade,
        "r_squared": float(fitted.rsquared),
        "interpretation": (
            f"The estimated linear trend in seasonally adjusted {parameter} is "
            f"{slope:+.4g} {units} per year ({slope * 10:+.4g} per decade). "
            f"The Newey-West HAC 95% confidence interval is "
            f"[{ci_low:+.4g}, {ci_high:+.4g}] {units}/year, with p={p_value:.3g}. "
            f"Kendall's tau={tau:+.3f} (p={mk_p:.3g}) is supplementary and is not "
            "autocorrelation-corrected."
        ),
    }


def add_fdr(results: list[dict], p_key: str = "p_value") -> list[dict]:
    """Add Benjamini-Hochberg adjusted q-values, preserving result order."""
    p_values = np.array([r.get(p_key, np.nan) for r in results], dtype=float)
    valid = np.isfinite(p_values)
    q_values = np.full(p_values.shape, np.nan, dtype=float)
    if valid.any():
        valid_p = p_values[valid]
        order = np.argsort(valid_p)
        ranked = valid_p[order]
        count = len(ranked)
        adjusted = ranked * count / np.arange(1, count + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        restored = np.empty_like(adjusted)
        restored[order] = np.clip(adjusted, 0, 1)
        q_values[valid] = restored
    for result, q_value in zip(results, q_values):
        result["q_value"] = float(q_value) if np.isfinite(q_value) else np.nan
    return results


def kalman_local_linear_trend(values):
    """Two-state local-linear Kalman smoother-like forward filter for display."""
    observations = np.asarray(values, dtype=float)
    if observations.size == 0:
        return {"level": np.array([]), "slope": np.array([])}
    finite = observations[np.isfinite(observations)]
    variance = max(float(np.nanvar(finite)) if finite.size else 1.0, 1e-8)
    observation_variance = variance * 0.20
    level_variance = variance * 0.015
    slope_variance = variance * 0.0008

    state = np.array([observations[0] if np.isfinite(observations[0]) else 0.0, 0.0])
    covariance = np.eye(2) * variance
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    observation_matrix = np.array([[1.0, 0.0]])
    process_noise = np.diag([level_variance, slope_variance])
    measurement_noise = np.array([[observation_variance]])
    levels, slopes = [], []

    for observation in observations:
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_noise
        if np.isfinite(observation):
            innovation = np.array([[observation]]) - observation_matrix @ state.reshape(2, 1)
            innovation_covariance = observation_matrix @ covariance @ observation_matrix.T + measurement_noise
            gain = covariance @ observation_matrix.T @ np.linalg.inv(innovation_covariance)
            state = state + (gain @ innovation).flatten()
            covariance = (np.eye(2) - gain @ observation_matrix) @ covariance
        levels.append(state[0])
        slopes.append(state[1])
    return {"level": np.asarray(levels), "slope": np.asarray(slopes)}


def detect_change_points(series: pd.Series, min_size: int = 6):
    """Penalized optimal partitioning for candidate mean shifts (exploratory)."""
    clean = series.dropna().astype(float)
    values = clean.to_numpy()
    count = len(values)
    if count < min_size * 3:
        return []
    cumulative = np.r_[0.0, np.cumsum(values)]
    cumulative_sq = np.r_[0.0, np.cumsum(values * values)]

    def segment_cost(start, stop):
        size = stop - start
        total = cumulative[stop] - cumulative[start]
        total_sq = cumulative_sq[stop] - cumulative_sq[start]
        return max(0.0, total_sq - total * total / size)

    variance = max(float(np.var(values)), 1e-8)
    penalty = variance * np.log(count) * 2.0
    best = np.full(count + 1, np.inf)
    previous = np.full(count + 1, -1, dtype=int)
    best[0] = -penalty
    for stop in range(min_size, count + 1):
        for start in range(0, stop - min_size + 1):
            if start and not np.isfinite(best[start]):
                continue
            candidate = best[start] + segment_cost(start, stop) + penalty
            if candidate < best[stop]:
                best[stop] = candidate
                previous[stop] = start
    boundaries = []
    cursor = count
    while cursor > 0 and previous[cursor] >= 0:
        cursor = int(previous[cursor])
        if cursor > 0:
            boundaries.append(cursor)
    return [clean.index[index] for index in sorted(boundaries)]


def detect_candidate_shifts(series: pd.Series, window: int = 12, threshold: float = 1.5):
    """Compatibility wrapper retained for the existing interface."""
    del threshold
    return detect_change_points(series, min_size=max(4, window // 2))
