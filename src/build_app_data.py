"""Small, pre-aggregated files for the Streamlit app, so the app never needs the 255 MB raw CSV.

Run from the project root (after data_prep, hotspots and forecasting):  python -m src.build_app_data
"""
import shutil

import pandas as pd

from src.config import APP_DATA, RESULTS
from src.data_prep import load_clean


def main():
    df = load_clean()
    w = df[df["in_analysis_window"]].copy()
    w["area_name"] = w["area_name"].astype(str)
    w["crime_group"] = w["crime_group"].astype(str)

    # month x area x group counts (overview trends)
    w["month"] = w["date_occ"].dt.to_period("M").dt.to_timestamp()
    w.groupby(["month", "area_name", "crime_group"]).size().rename("n").reset_index() \
        .to_parquet(APP_DATA / "agg_month_area_group.parquet", index=False)

    # area x group x weekday x hour counts, placeholder times removed (time patterns)
    t = w[~w["time_placeholder"]]
    t.groupby(["area_name", "crime_group", "weekday", "hour"]).size().rename("n").reset_index() \
        .to_parquet(APP_DATA / "agg_area_group_weekday_hour.parquet", index=False)

    # area x group x premise (top premises)
    top_prem = w["premise"].value_counts().head(15).index
    p = w[w["premise"].isin(top_prem)].copy()
    p["premise"] = p["premise"].astype(str)
    p.groupby(["area_name", "crime_group", "premise"]).size().rename("n").reset_index() \
        .to_parquet(APP_DATA / "agg_area_group_premise.parquet", index=False)

    # area-level KPIs
    k = w.groupby("area_name").agg(incidents=("dr_no", "size"),
                                   violent_share=("crime_group", lambda s: (s == "Violent Crime").mean()),
                                   arrest_rate=("arrest_made", "mean"),
                                   weeks=("week_start", "nunique")).reset_index()
    k["per_week"] = k["incidents"] / k["weeks"]
    k.to_parquet(APP_DATA / "area_kpis.parquet", index=False)

    # copy result tables the app displays
    for f in RESULTS.glob("*"):
        shutil.copy(f, APP_DATA / f.name)
    print(sorted(x.name for x in APP_DATA.iterdir()))


if __name__ == "__main__":
    main()
