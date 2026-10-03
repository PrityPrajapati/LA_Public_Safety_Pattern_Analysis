"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "Crime_Data_from_2020_to_2024.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
APP_DATA = ROOT / "app" / "app_data"

for p in (PROCESSED, MODELS, FIGURES, RESULTS, APP_DATA):
    p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# LAPD switched to a new records system (NIBRS) on 7 March 2024. After that date the
# open dataset only holds incidents from the retiring system, so counts collapse.
# All time-based analysis and modelling therefore stops at the last full week before it.
ANALYSIS_START = "2020-01-06"   # first Monday of 2020, so every week is complete
ANALYSIS_END = "2024-03-03"     # last Sunday before the 7 March 2024 system change

# Time-based split for forecasting (weeks run Monday to Sunday, labelled by the Monday)
TRAIN_END = "2022-12-26"        # train:      weeks starting Jan 2020 - Dec 2022
VAL_END = "2023-06-26"          # validation: weeks starting Jan 2023 - Jun 2023
                                # test:       weeks starting Jul 2023 - Feb 2024

# Approximate km per degree at LA's latitude (about 34.05 N), used to project lat/lon
KM_PER_DEG_LAT = 110.95
KM_PER_DEG_LON = 92.29
