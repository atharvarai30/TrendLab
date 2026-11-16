# TrendLab — Project Brief

## Challenge alignment

TrendLab addresses the NASA Space Apps Challenge **Be an Earth Systems Trend Detective** by helping users:

1. Select environmental variables from a documented NASA data catalog.
2. Retrieve actual NASA POWER daily observations for chosen coordinates and dates.
3. Build monthly time series and visualize values and seasonal anomalies.
4. Estimate the rate of change and test trend significance.
5. Compare regions and identify locations with opposing trend directions.
6. Examine how trends vary over a bounded spatial sample.

## Core scientific output

For every eligible time series, TrendLab reports:

- Linear trend in native units per year and per decade.
- Newey–West HAC 95% confidence interval and p-value.
- Benjamini–Hochberg adjusted q-value within the displayed analysis family.
- Number of continuous monthly observations.
- Kendall's tau as a supplementary rank statistic, with a warning that its p-value is not autocorrelation-corrected.

## Data source

TrendLab currently retrieves data through the NASA POWER Daily Point API. Its live catalog and analysis are limited to the seven parameters listed in the application documentation. Parameter-level provenance and spatial resolution vary, so results are identified as NASA POWER point time series.

## Validation

The unit test suite checks API-response parsing, missing-value handling, monthly aggregation and coverage thresholds, trend estimation on a known synthetic test pattern, FDR adjustment, missing-month safeguards, and candidate change-point detection. These are software tests, not evidence of a real environmental discovery. A competition submission should include a documented real-data case study and independent validation.
