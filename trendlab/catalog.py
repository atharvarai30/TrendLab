"""Curated, documented NASA POWER daily variables used by TrendLab."""

VARIABLES = {
    "Mean air temperature at 2 m": {
        "code": "T2M",
        "units": "°C",
        "monthly_aggregation": "mean",
        "family": "Temperature",
        "description": "Monthly mean of daily near-surface air temperature at 2 metres.",
        "why_it_matters": "Tracks broad warming or cooling in near-surface air conditions.",
    },
    "Daily maximum air temperature at 2 m": {
        "code": "T2M_MAX",
        "units": "°C",
        "monthly_aggregation": "mean",
        "family": "Temperature",
        "description": "Monthly mean of the daily maximum 2-metre air temperature.",
        "why_it_matters": "Highlights changes in daytime or peak daily heat conditions.",
    },
    "Daily minimum air temperature at 2 m": {
        "code": "T2M_MIN",
        "units": "°C",
        "monthly_aggregation": "mean",
        "family": "Temperature",
        "description": "Monthly mean of the daily minimum 2-metre air temperature.",
        "why_it_matters": "Highlights changes in nighttime or minimum daily temperatures.",
    },
    "Precipitation": {
        "code": "PRECTOTCORR",
        "units": "mm/month",
        "monthly_aggregation": "sum",
        "family": "Water cycle",
        "description": "Corrected daily precipitation, accumulated to a monthly total.",
        "why_it_matters": "Can reveal long-term shifts in wetness and rainfall deficits.",
    },
    "Relative humidity at 2 m": {
        "code": "RH2M",
        "units": "%",
        "monthly_aggregation": "mean",
        "family": "Atmosphere",
        "description": "Monthly mean of daily relative humidity at 2 metres.",
        "why_it_matters": "Provides context for atmospheric moisture and evaporative demand.",
    },
    "Wind speed at 10 m": {
        "code": "WS10M",
        "units": "m/s",
        "monthly_aggregation": "mean",
        "family": "Atmosphere",
        "description": "Monthly mean of daily wind speed at 10 metres.",
        "why_it_matters": "Can reveal changes in near-surface wind conditions.",
    },
    "All-sky surface shortwave radiation": {
        "code": "ALLSKY_SFC_SW_DWN",
        "units": "kWh/m²/day",
        "monthly_aggregation": "mean",
        "family": "Energy",
        "description": "Monthly mean of daily all-sky downward shortwave solar energy at the surface.",
        "why_it_matters": "Tracks changes in surface solar energy, including cloud-related variability.",
    },
}

CODE_TO_LABEL = {item["code"]: label for label, item in VARIABLES.items()}
PRECIPITATION_CODES = {"PRECTOTCORR"}
