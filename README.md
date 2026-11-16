#This was made with the help of SI
# TrendLab — Earth System Trend Detective

TrendLab is a Streamlit application built around the NASA Space Apps Challenge **Be an Earth Systems Trend Detective**. It retrieves real observations from the public NASA POWER Daily Point API and helps users discover candidate trends, compare point locations, and map spatial differences in trend direction.

## What this build does

- **Live NASA data, no account prompt:** downloads daily values directly from NASA POWER. The application does not ask for an Earthdata username or password and does not substitute demo data if a request fails.
- **Variable discovery:** screens seven documented environmental parameters at the selected coordinates: mean, daily maximum, and daily minimum 2-m air temperature; corrected precipitation; relative humidity; wind speed; and all-sky surface shortwave radiation.
- **Trend explorer:** aggregates daily observations to monthly values, removes the mean calendar-month cycle, estimates linear trends, and reports Newey–West HAC confidence intervals and p-values.
- **Multiple-testing control:** applies Benjamini–Hochberg false-discovery-rate adjustment across each displayed variable scan.
- **Regional divergence:** compares the same variable across selected cities or user-added coordinates and identifies opposing trend directions.
- **Spatial trend screening:** samples NASA POWER point locations across a bounded latitude/longitude rectangle and adjusts p-values across the analyzed points.
- **Exploratory regime-shift view:** uses a Kalman-filtered signal and penalized segmentation to highlight candidate mean shifts; these are not physical change declarations.
- **Reproducibility:** offers CSV downloads and a JSON provenance record containing coordinates, request dates, parameter codes, source endpoint, and download time.
- **No simulated interface mode:** when NASA POWER cannot be reached, the application reports the failure rather than fabricating observations.

## Data source and scope

The live data source is NASA POWER:

`https://power.larc.nasa.gov/api/temporal/daily/point`

The application uses the `RE` community and these parameter codes:

| Parameter | Description | Monthly aggregation |
|---|---|---|
| `T2M` | Mean air temperature at 2 m | Mean |
| `T2M_MAX` | Daily maximum air temperature at 2 m | Mean of daily maxima |
| `T2M_MIN` | Daily minimum air temperature at 2 m | Mean of daily minima |
| `PRECTOTCORR` | Corrected precipitation | Sum of daily values |
| `RH2M` | Relative humidity at 2 m | Mean |
| `WS10M` | Wind speed at 10 m | Mean |
| `ALLSKY_SFC_SW_DWN` | All-sky surface shortwave radiation | Mean |

**Data provenance:** NASA POWER is an analysis-ready NASA data service that combines Earth-observation and model-based inputs. Underlying sources and spatial resolutions vary by parameter. TrendLab reports the NASA POWER parameters it requests and does not represent point samples as native satellite pixels or area-weighted regional averages.

## Scientific processing

1. Retrieve daily values from NASA POWER for a point and selected time interval.
2. Convert POWER fill values (`-999`) to missing values.
3. Aggregate to complete calendar months. Mean variables require at least 80% daily coverage. Monthly precipitation totals require at least 95% coverage to avoid treating incomplete sums as complete monthly totals.
4. Remove the long-term mean for each calendar month to reduce recurring seasonality.
5. Require at least 60 consecutive monthly observations. If months are missing, withhold HAC trend inference rather than compressing the timeline and treating nonconsecutive observations as adjacent months.
6. Fit an ordinary least squares linear slope to seasonal anomalies and use Newey–West HAC covariance estimates for the slope p-value and 95% confidence interval.
7. Report Kendall's tau as a supplementary rank statistic. Its p-value is not autocorrelation-corrected and should not be treated as an independent robust significance test.
8. Apply Benjamini–Hochberg FDR correction within the displayed family of variables, regional points, or spatial points.

## Run on Windows

Open PowerShell in the extracted `TrendLab` folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

You need an internet connection to retrieve current NASA POWER observations. No NASA credentials are requested.

## Run tests

```powershell
python -m unittest discover -s trendlab -p "test_*.py" -v
```

The tests validate response parsing with a mocked API response, monthly aggregation, missing-data handling, trend recovery on a synthetic test series, FDR adjustment, and change-point behavior. Synthetic series are used only inside unit tests; they are not shown as NASA observations in the application. Tests do not replace a live end-to-end NASA API run.

## Important interpretation limits

- The trend explorer and regional comparison operate on coordinates, not administrative-area averages.
- The spatial tab samples multiple point requests. It is not a native raster, and a denser sampling interval does not increase the underlying product's spatial resolution.
- NASA POWER parameter provenance and native resolution vary. Consult NASA's parameter documentation before interpreting any variable physically.
- HAC inference helps account for short-range serial correlation and heteroskedasticity, but does not correct every form of temporal or spatial dependence or model misspecification.
- FDR correction only covers the analysis family shown in that one scan. Repeatedly changing variables, locations, dates, or grid bounds creates additional tests.
- Statistical significance is not causation. The magnitude and practical importance of a trend must also be considered.
- Candidate regime shifts are exploratory signals and require sensitivity checks and independent physical interpretation.

## Competition research protocol

See [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md) for a suggested question, region-selection approach, pre-analysis decisions, evidence standards, and a presentation outline. It contains no invented results; use TrendLab-generated NASA observations for the actual finding.

## NASA resources

- [NASA POWER](https://power.larc.nasa.gov/)
- [NASA POWER Daily API documentation](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)
