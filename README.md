# CrimeScope LA
### Spatio-Temporal Hotspot Detection and Weekly Incident Forecasting for Public-Safety Resource Planning

Machine Learning Case Study 111 (Public Safety Pattern Analysis), B.Tech CSE 2024-28, Semester V.

> **Problem statement:** A city authority wants to investigate patterns in historical incidents to support resource planning.

This project turns 1 million LAPD incident records into three planning answers:

| Question | ML task | Method | Headline result |
|---|---|---|---|
| **When** do incidents happen? | Exploratory pattern analysis | Hour x weekday x crime-group profiles | Peak 16:00-21:00 and Fridays; 18:00 is 3.4x busier than 05:00 |
| **Where** do they concentrate? | Grouping / detection | K-Means patrol zones; DBSCAN vs HDBSCAN hotspots | 79 hotspots hold 13% of incidents in 2.8% of the city area (4.6x lift) |
| **How many** next week, per area? | Forecasting (regression) | Ridge / Random Forest / Poisson GBM vs 3 baselines | 8.1% error (WAPE); 23% better than "same as last week" |

A Streamlit app converts the forecasts into a patrol-unit allocation per area and per shift.

---

## Quick start

```bash
# 1. Python 3.11+ and a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 2. Install
pip install -r requirements.txt

# 3. Run the app (works straight away: all data it needs is included)
streamlit run app/app.py
```

To **re-run the full analysis** (about 5-8 minutes), put the raw CSV in `data/raw/` (see `data/raw/README.md`), then:

```bash
python -m src.run_pipeline
```

or open `notebooks/Public_Safety_Pattern_Analysis.ipynb`, which walks through every step with outputs.

**Troubleshooting**
- *"The forecasting model could not be loaded"* in the app: your scikit-learn version differs from the one that saved the model. Run `python -m src.forecasting` (needs `data/processed/crimes_clean.parquet`, created by `python -m src.data_prep`).
- *Street map does not load* (no internet, e.g. during a viva): switch on **Offline map** on the Hotspot map page.

---

## Project structure

```
LA_Public_Safety_Pattern_Analysis/
├── app/
│   ├── app.py                  Streamlit application (6 pages)
│   └── app_data/               small pre-aggregated files the app reads (<1 MB)
├── data/
│   ├── raw/                    put Crime_Data_from_2020_to_2024.csv here (not included, 255 MB)
│   └── processed/              cleaned parquet (created by the pipeline)
├── figures/                    all charts used in the report (PNG)
├── models/                     saved models (zones, hotspots, 2 forecasters)
├── notebooks/                  executed notebook: the whole project, step by step
├── report/                     final report (Word and PDF)
├── results/                    metrics, comparisons and profiles (CSV / JSON)
├── src/
│   ├── config.py               paths, analysis window, time split
│   ├── data_prep.py            loading, data-quality audit, cleaning, features
│   ├── eda.py                  exploratory analysis and figures
│   ├── hotspots.py             concentration, K-Means zones, DBSCAN/HDBSCAN hotspots
│   ├── forecasting.py          weekly features, baselines, 3 ML models, evaluation
│   ├── build_app_data.py       aggregates for the app
│   ├── style.py                shared chart style
│   └── run_pipeline.py         runs everything in order
├── requirements.txt
└── README.md
```

---

## Data

LAPD **Crime Data from 2020 to Present**, City of Los Angeles open-data portal (public domain):
https://data.lacity.org/d/2nrs-mtv8. The file used has 1,004,894 rows and 28 columns (one row per reported incident).

**Key data-quality decisions**
- LAPD moved to a new records system on 7 March 2024; afterwards the open file holds only incidents from the old system and monthly volume falls 55%. **Analysis window: 6 Jan 2020 - 3 Mar 2024** (912,387 incidents, 217 full weeks).
- Placeholder times (00:01, and the excess of records at exactly 12:00) are excluded from hour-of-day analysis.
- Unknown dates are filed on the 1st of the month (4.6% vs 3.3% expected). This creates a monthly spike, so the forecaster gets a "week contains the 1st" feature.
- 2,240 records with (0, 0) coordinates are excluded from spatial analysis only. Victim age 0 (27%) means no individual victim.
- 140 offence descriptions are grouped into 6 planning groups: Violent, Vehicle, Burglary & Theft, Fraud & Identity, Vandalism, Other / Public Order.

## Methods and results

**Spatial.** Coordinates are projected to kilometres. K-Means (k = 30, chosen with silhouette and Davies-Bouldin within the operational range 16-30; mean ARI stability 0.70) defines patrol zones. DBSCAN and HDBSCAN are compared for dense hotspots using silhouette, Davies-Bouldin and **lift** (share of incidents / share of area). The rule, fixed in advance, selected DBSCAN (eps = 150 m, min_samples = 40 on a 60k sample): 79 hotspots, silhouette 0.48, lift 4.6x. HDBSCAN gave the best-separated clusters (silhouette up to 0.66) but they were broad regions (lift 1.1-1.4x), less useful for patrol targeting.

**Forecasting.** One-week-ahead incidents per area (21 areas). Time split: train 2021-2022, validation Jan-Jun 2023, test Jul 2023 - Feb 2024.

| Target | Selected model (by validation) | Test MAE | Test WAPE | R² | vs naive | vs best baseline | 80% interval coverage |
|---|---|---|---|---|---|---|---|
| All incidents | Ridge regression | 17.0 | 8.1% | 0.78 | -23% error | -3% error | 78% |
| Violent crime | Random Forest | 8.0 | 12.5% | 0.83 | -26% error | -7% error | 78% |

The strongest baseline, a 4-week moving average, is hard to beat. The ML models add a steady improvement, calibrated intervals and the ability to run what-if scenarios.

## Limitations
Reported incidents only; no usable data after March 2024; placeholder dates and times; hotspots are found on a 60,000-point sample; forecasts assume the near future looks like the recent past (no events, policy changes or weather inputs).
