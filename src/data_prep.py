"""Load the raw LAPD file, document data quality, clean it and engineer features.

Run from the project root:  python -m src.data_prep
Outputs: data/processed/crimes_clean.parquet and results/data_quality.json
"""
import json

import numpy as np
import pandas as pd

from src.config import RAW_CSV, PROCESSED, RESULTS, ANALYSIS_START, ANALYSIS_END

USECOLS = ["DR_NO", "Date Rptd", "DATE OCC", "TIME OCC", "AREA", "AREA NAME",
           "Rpt Dist No", "Part 1-2", "Crm Cd", "Crm Cd Desc", "Vict Age",
           "Vict Sex", "Vict Descent", "Premis Desc", "Weapon Desc",
           "Status Desc", "LAT", "LON"]

# Ordered keyword rules. The first match wins, so specific rules come first
# (e.g. "BURGLARY FROM VEHICLE" must become Vehicle Crime, not Burglary & Theft,
#  and "THEFT OF IDENTITY" must become Fraud & Identity, not Burglary & Theft).
CATEGORY_RULES = [
    ("Vehicle Crime", r"VEHICLE - STOLEN|VEHICLE - ATTEMPT STOLEN|VEHICLE, STOLEN|"
                      r"BURGLARY FROM VEHICLE|THEFT FROM MOTOR VEHICLE|"
                      r"DRIVING WITHOUT OWNER CONSENT"),
    ("Fraud & Identity", r"IDENTITY|BUNCO|EMBEZZLEMENT|FORGERY|CREDIT CARD|"
                         r"DEFRAUDING|COUNTERFEIT|INSURANCE FRAUD|BRIBERY|"
                         r"UNAUTHORIZED COMPUTER|DOCUMENT WORTHLESS"),
    ("Violent Crime", r"HOMICIDE|MANSLAUGHTER|ASSAULT|BATTERY|ROBBERY|RAPE|"
                      r"KIDNAPPING|SHOTS FIRED|DISCHARGE FIREARMS|BRANDISH|"
                      r"CRIMINAL THREATS|LYNCHING|SODOMY|ORAL COPULATION|"
                      r"SEXUAL PENETRATION|HUMAN TRAFFICKING|CHILD ABUSE|"
                      r"EXTORTION|STALKING|INTIMATE PARTNER|THROWING OBJECT|"
                      r"SHOTS INHABITED|CHILD STEALING|FALSE IMPRISONMENT"),
    ("Burglary & Theft", r"BURGLARY|THEFT|SHOPLIFTING|PICKPOCKET|PURSE SNATCHING|"
                         r"STOLEN|DISHONEST EMPLOYEE|TILL TAP"),
    ("Vandalism", r"VANDALISM|ARSON|TELEPHONE PROPERTY"),
]
OTHER = "Other / Public Order"
GROUP_ORDER = ["Violent Crime", "Vehicle Crime", "Burglary & Theft",
               "Fraud & Identity", "Vandalism", OTHER]


def categorise(desc: pd.Series) -> pd.Series:
    """Map the 140 detailed LAPD crime descriptions to 6 planning-friendly groups."""
    out = pd.Series(OTHER, index=desc.index, dtype=object)
    unassigned = pd.Series(True, index=desc.index)
    for name, pattern in CATEGORY_RULES:
        hit = unassigned & desc.str.contains(pattern, regex=True, na=False)
        out[hit] = name
        unassigned &= ~hit
    return out


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path, usecols=USECOLS, dtype=str)


def quality_report(raw: pd.DataFrame) -> dict:
    """Collect the data-quality facts that the report documents."""
    occ = pd.to_datetime(raw["DATE OCC"], format="%m/%d/%Y %I:%M:%S %p")
    rpt = pd.to_datetime(raw["Date Rptd"], format="%m/%d/%Y %I:%M:%S %p")
    t = raw["TIME OCC"].astype(int)
    age = raw["Vict Age"].astype(int)
    lat, lon = raw["LAT"].astype(float), raw["LON"].astype(float)
    lag = (rpt - occ).dt.days
    monthly = occ.dt.to_period("M").value_counts().sort_index()
    pre = monthly.loc["2022-03":"2024-02"].mean()
    post = monthly.loc["2024-04":"2024-11"].mean()
    return {
        "rows": int(len(raw)),
        "columns_in_source": 28,
        "duplicate_DR_NO": int(raw["DR_NO"].duplicated().sum()),
        "date_occ_range": [str(occ.min().date()), str(occ.max().date())],
        "missing_share": {c: round(float(v), 4) for c, v in raw.isna().mean().items() if v > 0},
        "zero_coordinates": int(((lat == 0) | (lon == 0)).sum()),
        "time_exactly_1200_share": round(float((t == 1200).mean()), 4),
        "time_exactly_0001_share": round(float((t == 1).mean()), 4),
        "occurred_on_1st_of_month_share": round(float((occ.dt.day == 1).mean()), 4),
        "expected_1st_of_month_share": round(12 / 365.25, 4),
        "victim_age_zero_share": round(float((age == 0).mean()), 4),
        "victim_age_negative": int((age < 0).sum()),
        "victim_sex_codes": raw["Vict Sex"].fillna("NaN").value_counts().to_dict(),
        "report_lag_days_median": float(lag.median()),
        "report_lag_over_30d_share": round(float((lag > 30).mean()), 4),
        "report_lag_over_365d_share": round(float((lag > 365).mean()), 4),
        "monthly_mean_mar2022_feb2024": round(float(pre), 1),
        "monthly_mean_apr2024_nov2024": round(float(post), 1),
        "drop_after_system_change_pct": round(float(100 * (1 - post / pre)), 1),
        "distinct_crime_descriptions": int(raw["Crm Cd Desc"].nunique()),
        "distinct_reporting_districts": int(raw["Rpt Dist No"].nunique()),
    }


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["dr_no"] = raw["DR_NO"].astype(np.int64)
    df["date_occ"] = pd.to_datetime(raw["DATE OCC"], format="%m/%d/%Y %I:%M:%S %p")
    df["date_rptd"] = pd.to_datetime(raw["Date Rptd"], format="%m/%d/%Y %I:%M:%S %p")
    t = raw["TIME OCC"].astype(int)
    df["hour"] = (t // 100).astype(np.int8)
    # Placeholder times. 00:01 is almost never a real time, so all of it is flagged.
    # 12:00 is partly real (people round to the hour) and partly "time unknown".
    # We estimate the real level from the neighbouring round hours (11:00 and 13:00)
    # and flag only the excess, deterministically chosen by DR_NO, so hourly profiles
    # have neither a false spike nor a false dip at noon.
    expected_noon = (t.eq(1100).sum() + t.eq(1300).sum()) / 2
    keep_share = min(1.0, expected_noon / max(t.eq(1200).sum(), 1))
    bucket = raw["DR_NO"].astype(np.int64) % 1000
    df["time_placeholder"] = t.eq(1) | (t.eq(1200) & (bucket >= int(keep_share * 1000)))
    df["area"] = raw["AREA"].astype(np.int8)
    df["area_name"] = raw["AREA NAME"].astype("category")
    df["rpt_dist"] = raw["Rpt Dist No"].astype(np.int16)
    df["part"] = raw["Part 1-2"].astype(np.int8)
    df["crm_cd"] = raw["Crm Cd"].astype(np.int16)
    df["crime_desc"] = raw["Crm Cd Desc"].astype("category")
    df["crime_group"] = pd.Categorical(categorise(raw["Crm Cd Desc"]), categories=GROUP_ORDER)

    # Victim fields: age 0 or negative means "no individual victim, or unknown"
    age = raw["Vict Age"].astype(float)
    df["vict_age"] = age.where((age > 0) & (age <= 100)).astype("float32")
    sex = raw["Vict Sex"].where(raw["Vict Sex"].isin(["M", "F"]), "Unknown")
    df["vict_sex"] = sex.astype("category")
    df["vict_descent"] = raw["Vict Descent"].fillna("X").replace("-", "X").astype("category")

    df["premise"] = raw["Premis Desc"].fillna("UNKNOWN").astype("category")
    df["weapon"] = raw["Weapon Desc"].fillna("NONE / NOT RECORDED").astype("category")
    df["weapon_involved"] = raw["Weapon Desc"].notna()
    df["status"] = raw["Status Desc"].astype("category")
    df["arrest_made"] = raw["Status Desc"].isin(["Adult Arrest", "Juv Arrest"])

    lat, lon = raw["LAT"].astype(float), raw["LON"].astype(float)
    bad = (lat == 0) | (lon == 0)
    df["lat"] = lat.where(~bad).astype("float32")
    df["lon"] = lon.where(~bad).astype("float32")

    # Calendar features
    df["year"] = df["date_occ"].dt.year.astype(np.int16)
    df["month"] = df["date_occ"].dt.month.astype(np.int8)
    df["weekday"] = df["date_occ"].dt.dayofweek.astype(np.int8)   # 0 = Monday
    df["week_start"] = df["date_occ"].dt.to_period("W-SUN").dt.start_time
    df["report_lag_days"] = (df["date_rptd"] - df["date_occ"]).dt.days.astype(np.int32)
    df["in_analysis_window"] = df["date_occ"].between(ANALYSIS_START, ANALYSIS_END)
    return df


def load_clean() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "crimes_clean.parquet")


def main():
    raw = load_raw()
    q = quality_report(raw)
    df = clean(raw)
    del raw
    q["rows_missing_coordinates_after_cleaning"] = int(df["lat"].isna().sum())
    q["rows_in_analysis_window"] = int(df["in_analysis_window"].sum())
    q["crime_group_counts"] = {k: int(v) for k, v in df["crime_group"].value_counts().items()}
    df.to_parquet(PROCESSED / "crimes_clean.parquet", index=False)
    (RESULTS / "data_quality.json").write_text(json.dumps(q, indent=2, default=str))
    print(json.dumps(q, indent=2, default=str))


if __name__ == "__main__":
    main()
