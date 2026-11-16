"""Unit tests for data parsing and statistical methods (no live API calls)."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from trendlab.analysis import add_fdr, analyze_series, detect_change_points
from trendlab.nasa_power import fetch_power_daily, to_monthly


class FakeResponse:
    url = "https://power.larc.nasa.gov/test"

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "properties": {
                "parameter": {
                    "T2M": {"20200101": 10.0, "20200102": 11.0, "20200103": -999.0},
                    "PRECTOTCORR": {"20200101": 1.0, "20200102": 2.0, "20200103": 3.0},
                }
            }
        }


class TrendLabValidation(unittest.TestCase):
    def test_nasa_power_parser_uses_requested_real_api_contract(self):
        with patch("trendlab.nasa_power.requests.get", return_value=FakeResponse()) as mocked:
            frame = fetch_power_daily(30.0, -81.0, "20200101", "20200103", ["T2M", "PRECTOTCORR"])
        self.assertEqual(list(frame.columns), ["T2M", "PRECTOTCORR"])
        self.assertTrue(pd.isna(frame.loc[pd.Timestamp("2020-01-03"), "T2M"]))
        self.assertEqual(frame.attrs["source"], "NASA POWER Daily Point API")
        mocked.assert_called_once()
        self.assertEqual(mocked.call_args.kwargs["params"]["parameters"], "T2M,PRECTOTCORR")

    def test_monthly_aggregation_and_coverage(self):
        dates = pd.date_range("2020-01-01", "2020-01-31", freq="D")
        daily = pd.DataFrame({
            "T2M": np.full(len(dates), 20.0),
            "PRECTOTCORR": np.ones(len(dates)),
        }, index=dates)
        monthly = to_monthly(daily)
        self.assertAlmostEqual(float(monthly.loc[pd.Timestamp("2020-01-01"), "T2M"]), 20.0)
        self.assertAlmostEqual(float(monthly.loc[pd.Timestamp("2020-01-01"), "PRECTOTCORR"]), 31.0)
        self.assertGreaterEqual(float(monthly.attrs["coverage"].iloc[0]["T2M"]), 0.99)

    def test_incomplete_precipitation_month_is_not_treated_as_a_full_total(self):
        dates = pd.date_range("2020-02-01", "2020-02-29", freq="D")
        daily = pd.DataFrame({"PRECTOTCORR": np.ones(len(dates))}, index=dates)
        daily = daily.drop(dates[[0, 1]])  # below the 95% precipitation threshold
        monthly = to_monthly(daily)
        self.assertTrue(pd.isna(monthly.loc[pd.Timestamp("2020-02-01"), "PRECTOTCORR"]))

    def test_known_linear_trend_recovered(self):
        index = pd.date_range("2000-01-01", periods=180, freq="MS")
        x = np.arange(len(index), dtype=float) / 12
        seasonal = np.sin(2 * np.pi * (index.month.to_numpy() - 1) / 12)
        values = 12 + 0.04 * x + seasonal
        frame = pd.DataFrame({"T2M": values}, index=index)
        result = analyze_series(frame, "T2M", "°C")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["slope_per_year"], 0.04, delta=0.006)
        self.assertLess(result["ci_low"], result["slope_per_year"])
        self.assertGreater(result["ci_high"], result["slope_per_year"])

    def test_missing_month_withholds_hac_trend(self):
        index = pd.date_range("2000-01-01", periods=120, freq="MS")
        frame = pd.DataFrame({"T2M": np.linspace(10, 15, len(index))}, index=index)
        frame = frame.drop(index[40])
        self.assertIsNone(analyze_series(frame, "T2M", "°C"))

    def test_fdr_adjustment(self):
        results = [{"p_value": 0.01}, {"p_value": 0.04}, {"p_value": 0.20}, {"p_value": 0.03}]
        adjusted = add_fdr(results)
        q_values = [item["q_value"] for item in adjusted]
        self.assertAlmostEqual(q_values[0], 0.04)
        self.assertAlmostEqual(q_values[1], 0.0533333333, places=6)
        self.assertAlmostEqual(q_values[2], 0.20)
        self.assertAlmostEqual(q_values[3], 0.0533333333, places=6)

    def test_change_point_detects_large_step(self):
        index = pd.date_range("2000-01-01", periods=120, freq="MS")
        values = np.r_[np.zeros(60), np.full(60, 5.0)]
        points = detect_change_points(pd.Series(values, index=index), min_size=8)
        self.assertTrue(any(abs((point - index[60]).days) <= 31 for point in points))

    def test_invalid_parameter_rejected_before_network(self):
        with self.assertRaises(ValueError):
            fetch_power_daily(30.0, -81.0, "20200101", "20201231", ["NOT_A_POWER_PARAMETER"])


if __name__ == "__main__":
    unittest.main()
