"""Run the whole project end to end (about 5-8 minutes on a laptop).

    python -m src.run_pipeline

Steps: clean -> EDA -> hotspots/zones -> forecasting -> app data.
Needs data/raw/Crime_Data_from_2020_to_2024.csv (see data/raw/README.md).
"""
import time

from src import build_app_data, data_prep, eda, forecasting, hotspots

STEPS = [("1/5 Cleaning and data-quality report", data_prep.main),
         ("2/5 Exploratory analysis and figures", eda.main),
         ("3/5 Spatial patterns: zones and hotspots", hotspots.main),
         ("4/5 Weekly forecasting models", forecasting.main),
         ("5/5 Building app data", build_app_data.main)]

if __name__ == "__main__":
    t0 = time.time()
    for name, fn in STEPS:
        print(f"\n######## {name}")
        t = time.time()
        fn()
        print(f"######## done in {time.time() - t:.0f}s")
    print(f"\nPipeline finished in {(time.time() - t0) / 60:.1f} min. Start the app with:  streamlit run app/app.py")
