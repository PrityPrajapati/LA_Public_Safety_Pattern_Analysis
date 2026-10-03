"""Exploratory data analysis: figures in figures/ and key numbers in results/eda_summary.json.

Run from the project root:  python -m src.eda
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import style
from src.config import FIGURES, RESULTS, ANALYSIS_END
from src.data_prep import load_clean, GROUP_ORDER

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def fig_monthly_trend(df):
    m = df.groupby(df["date_occ"].dt.to_period("M")).size()
    m.index = m.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ok = m.index <= ANALYSIS_END
    ax.plot(m.index[ok], m[ok], color=style.INK, lw=2.2, label="Used for analysis")
    ax.plot(m.index[~ok | (m.index == m.index[ok][-1])], m[~ok | (m.index == m.index[ok][-1])],
            color=style.ACCENT, lw=2.2, ls="--", label="After records-system change (excluded)")
    ax.axvline(pd.Timestamp("2024-03-07"), color=style.ACCENT, lw=1.2)
    ax.text(pd.Timestamp("2024-03-20"), m.max() * 0.97, "7 Mar 2024:\nLAPD moves to\nnew records system",
            color=style.ACCENT, fontsize=9.5, va="top")
    ax.set_title("Monthly incidents, 2020-2024")
    ax.set_ylabel("Incidents per month")
    ax.set_ylim(0, m.max() * 1.1)
    ax.legend(loc="lower left")
    style.save(fig, FIGURES / "01_monthly_trend.png")
    return m


def fig_groups(w):
    g = w["crime_group"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.barh(g.index[::-1], g.values[::-1], color=[style.GROUP_COLORS[k] for k in g.index[::-1]])
    for i, v in enumerate(g.values[::-1]):
        ax.text(v + g.max() * 0.01, i, f"{v:,}  ({v / g.sum():.0%})", va="center", fontsize=9.5, color=style.INK)
    ax.set_xlim(0, g.max() * 1.25)
    ax.set_title("Incidents by crime group")
    ax.set_xlabel("Incidents (Jan 2020 - Feb 2024)")
    ax.grid(axis="y", visible=False)
    style.save(fig, FIGURES / "02_crime_groups.png")
    return g


def fig_top_crimes(w):
    top = w["crime_desc"].value_counts().head(15)
    grp = w.drop_duplicates("crime_desc").set_index("crime_desc")["crime_group"]
    labels = [s.title()[:48] for s in top.index]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.barh(labels[::-1], top.values[::-1],
            color=[style.GROUP_COLORS[grp[k]] for k in top.index[::-1]])
    ax.set_title("Top 15 offence types (colour = crime group)")
    ax.set_xlabel("Incidents")
    ax.grid(axis="y", visible=False)
    style.save(fig, FIGURES / "03_top_offences.png")
    return top


def fig_areas(w):
    a = w.groupby("area_name", observed=True).agg(
        n=("dr_no", "size"), violent=("crime_group", lambda s: (s == "Violent Crime").mean()))
    a = a.sort_values("n")
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.barh(a.index, a["n"], color=style.BLUE)
    for i, (n, v) in enumerate(zip(a["n"], a["violent"])):
        ax.text(n + a["n"].max() * 0.01, i, f"{v:.0%} violent", va="center", fontsize=8.5, color=style.MUTED)
    ax.set_xlim(0, a["n"].max() * 1.2)
    ax.set_title("Incidents by LAPD area (with violent-crime share)")
    ax.set_xlabel("Incidents (Jan 2020 - Feb 2024)")
    ax.grid(axis="y", visible=False)
    style.save(fig, FIGURES / "04_areas.png")
    return a


def fig_hour_weekday(w):
    x = w[~w["time_placeholder"]]
    piv = x.pivot_table(index="weekday", columns="hour", values="dr_no", aggfunc="size")
    weeks = x["week_start"].nunique()
    piv = piv / weeks
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7), DAYS)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.grid(False)
    ax.set_title("When incidents happen: average incidents per hour slot per week")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Avg incidents / week")
    style.save(fig, FIGURES / "05_hour_weekday_heatmap.png")
    return piv


def fig_hour_by_group(w):
    x = w[~w["time_placeholder"]]
    h = x.groupby(["crime_group", "hour"], observed=True).size().unstack(0)
    h = h / h.sum()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for g in GROUP_ORDER:
        ax.plot(h.index, h[g] * 100, lw=2.4 if g == "Violent Crime" else 1.6,
                color=style.GROUP_COLORS[g], label=g)
    ax.set_xticks(range(0, 24, 2), [f"{i:02d}" for i in range(0, 24, 2)])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("% of the group's incidents")
    ax.set_title("Each crime group has its own daily rhythm")
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    style.save(fig, FIGURES / "06_hour_by_group.png")
    return h


def fig_yearly_groups(w):
    y = w[w["year"] <= 2023].groupby(["year", "crime_group"], observed=True).size().unstack()
    idx = y / y.loc[2020] * 100
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for g in GROUP_ORDER:
        ax.plot(idx.index, idx[g], marker="o", lw=2, color=style.GROUP_COLORS[g], label=g)
    ax.axhline(100, color=style.MUTED, lw=1, ls=":")
    ax.set_xticks([2020, 2021, 2022, 2023])
    ax.set_ylabel("Index (2020 = 100)")
    ax.set_title("Trend by crime group, full years 2020-2023")
    ax.legend(fontsize=9, ncol=2)
    style.save(fig, FIGURES / "07_yearly_trend_by_group.png")
    return y


def fig_premises(w):
    p = w["premise"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh([s.title()[:40] for s in p.index[::-1]], p.values[::-1], color=style.INK)
    for i, v in enumerate(p.values[::-1]):
        ax.text(v + p.max() * 0.01, i, f"{v / len(w):.1%}", va="center", fontsize=9, color=style.MUTED)
    ax.set_xlim(0, p.max() * 1.15)
    ax.set_title("Where incidents happen: top 10 premise types")
    ax.set_xlabel("Incidents")
    ax.grid(axis="y", visible=False)
    style.save(fig, FIGURES / "08_premises.png")
    return p


def fig_spatial(w):
    x = w.dropna(subset=["lat", "lon"])
    fig, ax = plt.subplots(figsize=(7.5, 8))
    hb = ax.hexbin(x["lon"], x["lat"], gridsize=85, bins="log", cmap="inferno", mincnt=1)
    ax.set_aspect(1 / np.cos(np.radians(34.05)))
    ax.set_title("Spatial density of incidents (log scale)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    cb = fig.colorbar(hb, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("Incidents per hexagon (log)")
    style.save(fig, FIGURES / "09_spatial_density.png")


def fig_report_lag_arrest(w):
    s = w.groupby("crime_group", observed=True).agg(
        lag=("report_lag_days", "median"), late=("report_lag_days", lambda v: (v > 30).mean()),
        arrest=("arrest_made", "mean"), weapon=("weapon_involved", "mean")).reindex(GROUP_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = [style.GROUP_COLORS[g] for g in s.index]
    axes[0].barh(s.index[::-1], s["late"][::-1] * 100, color=cols[::-1])
    axes[0].set_title("Reported more than 30 days late")
    axes[0].set_xlabel("% of incidents")
    axes[1].barh(s.index[::-1], s["arrest"][::-1] * 100, color=cols[::-1])
    axes[1].set_title("Arrest made")
    axes[1].set_xlabel("% of incidents")
    axes[1].set_yticklabels([])
    for a in axes:
        a.grid(axis="y", visible=False)
    style.save(fig, FIGURES / "10_report_lag_and_arrests.png")
    return s


def fig_victims(w):
    x = w[w["crime_group"] == "Violent Crime"].dropna(subset=["vict_age"])
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.arange(0, 91, 5)
    for sex, c in [("F", style.ACCENT), ("M", style.BLUE)]:
        ax.hist(x.loc[x["vict_sex"] == sex, "vict_age"], bins=bins, histtype="step", lw=2.2, color=c,
                label={"F": "Female", "M": "Male"}[sex])
    ax.set_title("Victim age, violent crime")
    ax.set_xlabel("Victim age")
    ax.set_ylabel("Incidents")
    ax.legend()
    style.save(fig, FIGURES / "11_victim_age.png")
    return x["vict_age"].median()


def fig_quality(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    raw_hours = df.groupby("hour").size()
    axes[0].bar(raw_hours.index, raw_hours.values, color=style.MUTED, label="All records")
    ph = df[df["time_placeholder"]].groupby("hour").size()
    axes[0].bar(ph.index, ph.values, color=style.ACCENT, label="Flagged placeholder (excess 12:00, all 00:01)")
    axes[0].set_title("Hour recorded: placeholder spikes")
    axes[0].set_xlabel("Hour recorded")
    axes[0].legend(fontsize=9)
    dom = df.groupby(df["date_occ"].dt.day).size()
    axes[1].bar(dom.index, dom.values, color=[style.ACCENT if d == 1 else style.MUTED for d in dom.index])
    axes[1].set_title("Day of month: unknown dates go to the 1st")
    axes[1].set_xlabel("Day of month")
    style.save(fig, FIGURES / "00_data_quality_time.png")


def main():
    style.apply()
    df = load_clean()
    monthly = fig_monthly_trend(df)
    fig_quality(df)
    w = df[df["in_analysis_window"]].copy()
    groups = fig_groups(w)
    top = fig_top_crimes(w)
    areas = fig_areas(w)
    hw = fig_hour_weekday(w)
    hg = fig_hour_by_group(w)
    yearly = fig_yearly_groups(w)
    prem = fig_premises(w)
    fig_spatial(w)
    lagtab = fig_report_lag_arrest(w)
    med_age = fig_victims(w)

    x = w[~w["time_placeholder"]]
    hourly = x.groupby("hour").size()
    weekly_total = w.groupby("week_start").size()
    summary = {
        "analysis_rows": int(len(w)),
        "analysis_weeks": int(w["week_start"].nunique()),
        "mean_incidents_per_week": round(float(weekly_total.mean()), 1),
        "mean_incidents_per_day": round(float(len(w) / w["date_occ"].dt.normalize().nunique()), 1),
        "group_share": {k: round(float(v / groups.sum()), 4) for k, v in groups.items()},
        "top5_offences": {k: int(v) for k, v in top.head(5).items()},
        "busiest_area": areas["n"].idxmax(), "busiest_area_n": int(areas["n"].max()),
        "quietest_area": areas["n"].idxmin(), "quietest_area_n": int(areas["n"].min()),
        "highest_violent_share_area": areas["violent"].idxmax(),
        "highest_violent_share": round(float(areas["violent"].max()), 4),
        "peak_hour": int(hourly.idxmax()), "trough_hour": int(hourly.idxmin()),
        "peak_to_trough_ratio": round(float(hourly.max() / hourly.min()), 2),
        "evening_16_22_share": round(float(hourly.loc[16:22].sum() / hourly.sum()), 4),
        "busiest_weekday": DAYS[int(x.groupby("weekday").size().idxmax())],
        "quietest_weekday": DAYS[int(x.groupby("weekday").size().idxmin())],
        "violent_peak_hour": int(hg["Violent Crime"].idxmax()),
        "vehicle_peak_hour": int(hg["Vehicle Crime"].idxmax()),
        "yearly_counts": {int(k): {g: int(v) for g, v in row.items()} for k, row in yearly.iterrows()},
        "yearly_total": {int(k): int(v) for k, v in yearly.sum(axis=1).items()},
        "top_premises_share": {k: round(float(v / len(w)), 4) for k, v in prem.head(5).items()},
        "street_share": round(float((w["premise"] == "STREET").mean()), 4),
        "late_report_share_by_group": {k: round(float(v), 4) for k, v in lagtab["late"].items()},
        "arrest_rate_by_group": {k: round(float(v), 4) for k, v in lagtab["arrest"].items()},
        "overall_arrest_rate": round(float(w["arrest_made"].mean()), 4),
        "violent_victim_median_age": float(med_age),
        "firearm_share_violent": round(float(w.loc[w["crime_group"] == "Violent Crime", "weapon"]
                                             .str.contains("GUN|PISTOL|FIREARM|RIFLE|REVOLVER|SHOTGUN", regex=True)
                                             .mean()), 4),
    }
    (RESULTS / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
