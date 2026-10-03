"""CrimeScope LA - Public Safety Pattern Analysis (Streamlit app).

Run from the project root:   streamlit run app/app.py
The app only reads the small files in app/app_data/ and models/, never the raw CSV.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar

APP_DIR = Path(__file__).resolve().parent
DATA = APP_DIR / "app_data"
MODELS = APP_DIR.parent / "models"

GROUP_COLORS = {
    "Violent Crime": "#C8433A", "Vehicle Crime": "#3A6EA5", "Burglary & Theft": "#E9A23B",
    "Fraud & Identity": "#1F8A84", "Vandalism": "#7A5C99", "Other / Public Order": "#9CA3AF",
}
GROUPS = list(GROUP_COLORS)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SHIFTS = {"Day (06-14)": range(6, 14), "Evening (14-22)": range(14, 22), "Night (22-06)": [22, 23, 0, 1, 2, 3, 4, 5]}
INK, ACCENT, BLUE = "#1B2A41", "#E07A2E", "#3A6EA5"

st.set_page_config(page_title="CrimeScope LA", page_icon="🗺️", layout="wide")


# ----------------------------------------------------------------------------- loading
def _str_cols(d):
    for c in d.columns:
        if isinstance(d[c].dtype, pd.CategoricalDtype) or d[c].dtype == object or str(d[c].dtype).startswith("str"):
            d[c] = d[c].astype(str)
    return d


@st.cache_data
def load(name):
    return _str_cols(pd.read_parquet(DATA / f"{name}.parquet"))


@st.cache_data
def load_csv(name):
    return pd.read_csv(DATA / name)


@st.cache_data
def load_json(name):
    return json.loads((DATA / name).read_text())


@st.cache_resource
def load_model(name):
    """Returns (object, error message). Pickled sklearn models can fail to load on a
    different scikit-learn version; the app then keeps working without that tool."""
    try:
        return joblib.load(MODELS / name), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def fmt(n):
    return f"{n:,.0f}"


def largest_remainder(weights, total):
    """Split an integer total in proportion to weights (sums exactly to total)."""
    w = np.asarray(weights, float)
    if total <= 0 or w.sum() <= 0:
        return np.zeros(len(w), int)
    raw = w / w.sum() * total
    base = np.floor(raw).astype(int)
    rest = total - base.sum()
    base[np.argsort(-(raw - base))[:rest]] += 1
    return base


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("CrimeScope LA")
st.sidebar.caption("Spatio-temporal pattern analysis and incident forecasting for public-safety resource planning")
page = st.sidebar.radio("Go to", ["Overview", "Time patterns", "Hotspot map", "Forecast & resource planner",
                                  "Model performance", "Data & methods"])
st.sidebar.divider()
st.sidebar.caption("Data: LAPD 'Crime Data from 2020 to Present' (City of Los Angeles open data). "
                   "Analysis window: 6 Jan 2020 - 3 Mar 2024, before LAPD's records-system change.")

month = load("agg_month_area_group")
month["month"] = pd.to_datetime(month["month"])
areas = sorted(month["area_name"].unique())


# ----------------------------------------------------------------------------- pages
def page_overview():
    st.title("Where, when and what: Los Angeles incidents 2020-2024")
    eda = load_json("eda_summary.json")
    dq = load_json("data_quality.json")
    c = st.columns(4)
    c[0].metric("Incidents analysed", fmt(eda["analysis_rows"]))
    c[1].metric("Average per week", fmt(eda["mean_incidents_per_week"]))
    c[2].metric("Violent-crime share", f"{eda['group_share']['Violent Crime']:.0%}")
    c[3].metric("Arrest rate", f"{eda['overall_arrest_rate']:.1%}")

    f1, f2 = st.columns([2, 3])
    sel_areas = f1.multiselect("LAPD areas (empty = all)", areas)
    sel_groups = f2.multiselect("Crime groups (empty = all)", GROUPS)
    d = month
    if sel_areas:
        d = d[d["area_name"].isin(sel_areas)]
    if sel_groups:
        d = d[d["crime_group"].isin(sel_groups)]
    m = d.groupby(["month", "crime_group"])["n"].sum().reset_index()
    m = m[m["month"] < "2024-03-01"]   # March 2024 is only partly covered
    fig = px.area(m, x="month", y="n", color="crime_group", color_discrete_map=GROUP_COLORS,
                  category_orders={"crime_group": GROUPS}, labels={"n": "Incidents per month", "month": ""})
    fig.update_layout(height=380, legend_title_text="", margin=dict(t=20, b=10))
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    by_area = d.groupby("area_name")["n"].sum().sort_values()
    fig = px.bar(by_area, orientation="h", labels={"value": "Incidents", "area_name": ""},
                 color_discrete_sequence=[BLUE])
    fig.update_layout(height=520, showlegend=False, title="Incidents by area", margin=dict(t=40))
    a.plotly_chart(fig, width="stretch")
    by_group = d.groupby("crime_group")["n"].sum().reindex(GROUPS).dropna()
    fig = px.pie(values=by_group.values, names=by_group.index, color=by_group.index,
                 color_discrete_map=GROUP_COLORS, hole=0.5)
    fig.update_layout(height=520, title="Crime-group mix", margin=dict(t=40))
    b.plotly_chart(fig, width="stretch")

    st.info(f"Data note: from April 2024 the open dataset drops by about {dq['drop_after_system_change_pct']:.0f}% "
            "because LAPD moved to a new records system on 7 March 2024. That is a reporting change, not a fall "
            "in crime, so everything here stops at 3 March 2024.")


def page_time():
    st.title("When incidents happen")
    st.caption("Placeholder times (00:01, and the excess of records at exactly 12:00) are removed here.")
    hw = load("agg_area_group_weekday_hour")
    c1, c2 = st.columns(2)
    area = c1.selectbox("Area", ["All areas"] + areas)
    groups = c2.multiselect("Crime groups (empty = all)", GROUPS)
    d = hw if area == "All areas" else hw[hw["area_name"] == area]
    if groups:
        d = d[d["crime_group"].isin(groups)]
    weeks = 217
    piv = d.pivot_table(index="weekday", columns="hour", values="n", aggfunc="sum").reindex(index=range(7), columns=range(24)).fillna(0) / weeks
    fig = px.imshow(piv.values, x=[f"{h:02d}" for h in range(24)], y=DAYS, color_continuous_scale="YlOrRd",
                    labels=dict(x="Hour", y="", color="Avg / week"), aspect="auto")
    fig.update_layout(height=330, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

    hourly = d.groupby("hour")["n"].sum().reindex(range(24), fill_value=0)
    shift_share = {s: hourly.loc[list(h)].sum() / max(hourly.sum(), 1) for s, h in SHIFTS.items()}
    a, b = st.columns([3, 2])
    hg = d.groupby(["crime_group", "hour"])["n"].sum().reset_index()
    hg["share"] = hg["n"] / hg.groupby("crime_group")["n"].transform("sum") * 100
    fig = px.line(hg, x="hour", y="share", color="crime_group", color_discrete_map=GROUP_COLORS,
                  labels={"share": "% of group's incidents", "hour": "Hour"})
    fig.update_layout(height=360, legend_title_text="", title="Daily rhythm by crime group", margin=dict(t=40))
    a.plotly_chart(fig, width="stretch")
    with b:
        st.subheader("Suggested shift weighting")
        st.caption("Share of this selection's incidents in each 8-hour shift.")
        for s, v in shift_share.items():
            st.metric(s, f"{v:.0%}")
        peak = int(hourly.idxmax())
        st.write(f"Peak hour: **{peak:02d}:00**. Busiest day: **{DAYS[int(piv.sum(axis=1).idxmax())]}**.")


def page_map():
    st.title("Hotspots and patrol zones")
    hs = load_json("hotspot_summary.json")
    sel = hs["hotspot_selected"]
    c = st.columns(4)
    c[0].metric("Dense hotspots (DBSCAN)", sel["n_clusters"])
    c[1].metric("Incidents inside hotspots", f"{sel['coverage']:.0%}")
    c[2].metric("City area they cover", f"{sel['area_share']:.1%}")
    c[3].metric("Lift (x denser than average)", f"{sel['lift']:.1f}x")

    pts = load("map_points")
    prof = load("hotspot_profiles")
    f1, f2, f3 = st.columns([2, 2, 1])
    colour_by = f1.radio("Colour points by", ["Hotspot vs rest", "K-Means patrol zone", "Crime group"],
                         horizontal=True, key="colour_by")
    groups = f2.multiselect("Crime groups (empty = all)", GROUPS)
    n_show = f3.slider("Points shown", 5_000, 60_000, 20_000, 5_000)
    offline = f3.toggle("Offline map (no street tiles)", key="offline",
                        help="Use this if the street map does not load (no internet).")
    d = pts if not groups else pts[pts["crime_group"].isin(groups)]
    d = d.sample(min(n_show, len(d)), random_state=0)
    if colour_by == "Hotspot vs rest":
        d = d.assign(label=np.where(d["hotspot_rank"] > 0, "In a hotspot", "Elsewhere"))
        fig = px.scatter_map(d, lat="lat", lon="lon", color="label", zoom=9.2, height=620,
                             color_discrete_map={"In a hotspot": ACCENT, "Elsewhere": "#9CA3AF"},
                             opacity=0.55, map_style="carto-positron")
    elif colour_by == "K-Means patrol zone":
        d = d.assign(zone=d["zone"].astype(str))
        fig = px.scatter_map(d, lat="lat", lon="lon", color="zone", zoom=9.2, height=620, opacity=0.55,
                             map_style="carto-positron")
        fig.update_layout(showlegend=False)
    else:
        fig = px.scatter_map(d, lat="lat", lon="lon", color="crime_group", color_discrete_map=GROUP_COLORS,
                             zoom=9.2, height=620, opacity=0.55, map_style="carto-positron")
    top = prof.head(15)
    if offline:
        fig = offline_map(d, colour_by, top)
        st.plotly_chart(fig, width="stretch")
    else:
        add_top_and_show(fig, top)
    hotspot_tables_and_check(prof)


def offline_map(d, colour_by, top):
    colour = {"Hotspot vs rest": "label", "K-Means patrol zone": "zone", "Crime group": "crime_group"}[colour_by]
    if colour_by == "Hotspot vs rest":
        cmap = {"In a hotspot": ACCENT, "Elsewhere": "#C9CCD1"}
    elif colour_by == "Crime group":
        cmap = GROUP_COLORS
    else:
        cmap = None
    fig = px.scatter(d, x="lon", y="lat", color=colour, color_discrete_map=cmap, opacity=0.6, height=640,
                     labels={"lon": "Longitude", "lat": "Latitude"})
    fig.update_traces(marker=dict(size=3))
    fig.add_trace(go.Scatter(x=top["lon"], y=top["lat"], mode="markers+text", text=[str(i + 1) for i in range(len(top))],
                             marker=dict(size=16, color=INK), textfont=dict(color="white", size=9), name="Top 15 hotspots"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1 / np.cos(np.radians(34.05)))
    fig.update_xaxes(constrain="domain")
    fig.update_layout(margin=dict(t=10), showlegend=colour_by != "K-Means patrol zone", plot_bgcolor="#F4F2EE")
    return fig


def add_top_and_show(fig, top):
    fig.add_trace(go.Scattermap(lat=top["lat"], lon=top["lon"], mode="markers+text",
                                marker=dict(size=14, color=INK), text=[str(i + 1) for i in range(len(top))],
                                textfont=dict(color="white", size=10), name="Top 15 hotspots",
                                hovertext=top["main_lapd_area"] + " - " + top["top_crime_group"]))
    fig.update_traces(marker=dict(size=4), selector=dict(mode="markers"))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                      map_center=dict(lat=34.05, lon=-118.35))
    st.plotly_chart(fig, width="stretch")


def hotspot_tables_and_check(prof):
    st.subheader("Top hotspots")
    show = prof.head(20).copy()
    show.insert(0, "rank", range(1, len(show) + 1))
    show = show[["rank", "main_lapd_area", "top_crime_group", "top_offence", "est_incidents_total",
                 "footprint_km2", "violent_share", "peak_hour"]]
    st.dataframe(show.rename(columns={"main_lapd_area": "LAPD area", "top_crime_group": "Dominant group",
                                      "top_offence": "Most common offence", "est_incidents_total": "Est. incidents (4 yrs)",
                                      "footprint_km2": "Footprint km²", "violent_share": "Violent share",
                                      "peak_hour": "Peak hour"}),
                 hide_index=True, width="stretch")

    st.subheader("Check a location")
    zones, e1 = load_model("kmeans_zones.joblib")
    dbs, e2 = load_model("dbscan_hotspots.joblib")
    if e1 or e2:
        st.warning(f"Location check unavailable: {e1 or e2}")
        return
    a, b, _ = st.columns([1, 1, 2])
    lat = a.number_input("Latitude", 33.70, 34.34, 34.0443, format="%.4f", key="chk_lat")
    lon = b.number_input("Longitude", -118.67, -118.15, -118.2536, format="%.4f", key="chk_lon")
    xy = np.array([(lon - zones["lon0"]) * zones["km_per_deg_lon"], (lat - zones["lat0"]) * zones["km_per_deg_lat"]])
    zone = int(np.argmin(((zones["centers_km"] - xy) ** 2).sum(axis=1)))
    dist = np.sqrt(((dbs["core_xy"] - xy) ** 2).sum(axis=1))
    j = int(np.argmin(dist))
    zp = load("zone_profiles").set_index("zone").loc[zone]
    st.write(f"This location is in **patrol zone {zone}** (mostly {zp['main_lapd_area']}), which holds "
             f"{zp['share_of_incidents']:.1%} of city incidents, {zp['violent_share']:.0%} of them violent.")
    if dist[j] <= dbs["eps_km"]:
        r = prof.iloc[int(dbs["core_rank"][j]) - 1]
        st.error(f"Inside **hotspot #{int(dbs['core_rank'][j])}** ({r['main_lapd_area']}): dominant group "
                 f"{r['top_crime_group']}, peak hour {int(r['peak_hour']):02d}:00, about {fmt(r['est_incidents_total'])} "
                 "incidents in four years.")
    else:
        st.success(f"Not inside a dense hotspot (nearest hotspot core is {dist[j] * 1000:.0f} m away).")


def build_scenario_features(area, lags, weekly, model_info, target_week):
    """Rebuild the model's inputs for one area and one week from (edited) recent counts."""
    hist = weekly[weekly["area_name"] == area].sort_values("week_start")["y"].to_numpy(float)
    recent = np.concatenate([hist[-13:-4], lags[::-1]])        # last 13 weeks, oldest first, edits at the end
    city = weekly.groupby("week_start")["y"].sum().sort_index().to_numpy(float)
    ws = pd.Timestamp(target_week)
    woy = ws.isocalendar().week
    hol = set(pd.Series(USFederalHolidayCalendar().holidays("2019-01-01", "2026-12-31")).dt.to_period("W-SUN").dt.start_time)

    def first_in(week):
        return int((week + pd.Timedelta(days=6)).month != week.month or week.day == 1)

    f = {"lag_1": lags[0], "lag_2": lags[1], "lag_3": lags[2], "lag_4": lags[3],
         "lag_52": weekly[(weekly["area_name"] == area) & (weekly["week_start"] == ws - pd.Timedelta(weeks=52))]["y"].sum(),
         "roll_mean_4": np.mean(lags), "roll_mean_13": recent.mean(), "roll_std_4": np.std(lags, ddof=1),
         "city_lag_1": city[-1] + (lags[0] - hist[-1]),
         "woy_sin": np.sin(2 * np.pi * woy / 52.18), "woy_cos": np.cos(2 * np.pi * woy / 52.18),
         "holiday_week": int(ws in hol), "month_start_week": first_in(ws),
         "prev_month_start_week": first_in(ws - pd.Timedelta(weeks=1)),
         "area_level": model_info["area_level"][area]}
    f["trend_4_13"] = f["roll_mean_4"] - f["roll_mean_13"]
    return pd.DataFrame([f])[model_info["features"]]


def page_forecast():
    st.title("Next-week forecast and resource planner")
    fa = load("forecast_all_next_week")
    fv = load("forecast_violent_next_week")
    kpis = load("area_kpis").set_index("area_name")
    wk = pd.to_datetime(fa["week_start"].iloc[0])
    st.caption(f"Forecast week: **{wk:%d %b %Y} - {wk + pd.Timedelta(days=6):%d %b %Y}** "
               "(the first week after the usable data ends). Ranges are 80% prediction intervals.")

    c1, c2, c3 = st.columns(3)
    units = c1.number_input("Patrol units available this week", 10, 2000, 210, 10)
    vweight = c2.slider("Weight of a violent incident vs other incidents", 1.0, 5.0, 2.0, 0.5,
                        help="1 = every incident counts the same. 2 = a violent incident counts double.")
    min_units = c3.number_input("Minimum units per area", 0, 20, 3)

    plan = fa[["area_name", "forecast", "lo80", "hi80"]].merge(
        fv[["area_name", "forecast"]].rename(columns={"forecast": "violent_forecast"}), on="area_name")
    plan["demand"] = (plan["forecast"] - plan["violent_forecast"]).clip(lower=0) + vweight * plan["violent_forecast"]
    floor = min(min_units, units // len(plan))
    plan["units"] = floor + largest_remainder(plan["demand"], units - floor * len(plan))
    hw = load("agg_area_group_weekday_hour")
    for s, hrs in SHIFTS.items():
        share = hw[hw["hour"].isin(list(hrs))].groupby("area_name")["n"].sum() / hw.groupby("area_name")["n"].sum()
        plan[s] = share.reindex(plan["area_name"]).to_numpy()
    shift_units = np.vstack([largest_remainder(plan.loc[i, list(SHIFTS)].to_numpy(float), int(plan.loc[i, "units"]))
                             for i in plan.index])
    for k, s in enumerate(SHIFTS):
        plan[s] = shift_units[:, k]
    plan["typical_week"] = plan["area_name"].map(kpis["per_week"])
    plan = plan.sort_values("demand", ascending=False)

    m = st.columns(4)
    m[0].metric("City forecast, all incidents", fmt(plan["forecast"].sum()),
                f"{plan['forecast'].sum() - plan['typical_week'].sum():+,.0f} vs 2020-24 average week")
    m[1].metric("City forecast, violent", fmt(plan["violent_forecast"].sum()))
    m[2].metric("Busiest area", plan.iloc[0]["area_name"])
    m[3].metric("Units allocated", fmt(plan["units"].sum()))

    fig = go.Figure()
    p2 = plan.sort_values("forecast")
    fig.add_trace(go.Bar(y=p2["area_name"], x=p2["forecast"], orientation="h", marker_color=BLUE, name="Forecast",
                         error_x=dict(type="data", symmetric=False, array=p2["hi80"] - p2["forecast"],
                                      arrayminus=p2["forecast"] - p2["lo80"], color=INK, thickness=1)))
    fig.add_trace(go.Scatter(y=p2["area_name"], x=p2["typical_week"], mode="markers", marker=dict(color=ACCENT, size=9,
                             symbol="line-ns-open", line=dict(width=3)), name="Average week 2020-24"))
    fig.update_layout(height=560, xaxis_title="Incidents next week", margin=dict(t=10), legend=dict(orientation="h"))
    a, b = st.columns([3, 2])
    a.plotly_chart(fig, width="stretch")
    b.dataframe(plan[["area_name", "forecast", "violent_forecast", "units", *SHIFTS]].rename(
        columns={"area_name": "Area", "forecast": "All (fcst)", "violent_forecast": "Violent (fcst)", "units": "Units"}
    ).round(0), hide_index=True, width="stretch", height=560)
    st.download_button("Download the plan as CSV", plan.round(1).to_csv(index=False), "resource_plan.csv", "text/csv")

    st.divider()
    st.subheader("Scenario tool: what if recent weeks change?")
    info, err = load_model("forecast_all.joblib")
    if err:
        st.warning("The forecasting model could not be loaded with this scikit-learn version. "
                   f"Run `python -m src.forecasting` to re-create it. ({err})")
        return
    weekly = load("weekly_all")
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    area = st.selectbox("Area", areas, index=areas.index("Central"))
    hist = weekly[weekly["area_name"] == area].sort_values("week_start")["y"].to_numpy()
    cols = st.columns(4)
    lags = [cols[i].number_input(f"{i + 1} week{'s' if i else ''} ago", 0, 2000, int(hist[-1 - i]), key=f"lag{i}")
            for i in range(4)]
    X = build_scenario_features(area, np.array(lags, float), weekly, info, wk)
    pred = float(info["model"].predict(X)[0])
    base = float(fa.set_index("area_name").loc[area, "forecast"])
    r = st.columns(3)
    r[0].metric(f"Forecast for {area}", fmt(pred), f"{pred - base:+.0f} vs unedited forecast")
    r[1].metric("80% range", f"{max(pred + info['resid_q10'], 0):,.0f} - {pred + info['resid_q90']:,.0f}")
    r[2].metric("Model", info["name"])


def page_models():
    st.title("How well do the models work?")
    fs = load_json("forecast_summary.json")
    tab1, tab2 = st.tabs(["Forecasting", "Clustering"])
    with tab1:
        target = st.radio("Target", ["all", "violent"], format_func=lambda t: "All incidents" if t == "all" else "Violent crime",
                          horizontal=True, key="perf_target")
        s = fs[target]
        c = st.columns(4)
        c[0].metric("Selected model", s["best_model"])
        c[1].metric("Test MAE (per area-week)", f"{s['test_metrics']['MAE']:.1f}")
        c[2].metric("Test WAPE", f"{s['test_metrics']['WAPE_%']:.1f}%")
        c[3].metric("Better than naive by", f"{s['improvement_vs_naive_test_MAE_%']:.0f}%")
        comp = load_csv(f"forecast_{target}_model_comparison.csv")
        st.dataframe(comp[["model", "type", "params", "val_MAE", "test_MAE", "test_RMSE", "test_WAPE_%", "test_R2"]],
                     hide_index=True, width="stretch")
        st.caption("The model is chosen by validation MAE (Jan-Jun 2023). The test period (Jul 2023 - Feb 2024) is used "
                   f"once, at the end. 80% interval coverage on test: {s['interval80_test_coverage']:.0%}.")
        pred = load(f"forecast_{target}_test_predictions")
        pred["week_start"] = pd.to_datetime(pred["week_start"])
        area = st.selectbox("Area to inspect", areas, key="perf_area")
        p = pred[pred["area_name"] == area]
        fig = go.Figure([
            go.Scatter(x=p["week_start"], y=p["best_hi80"], line=dict(width=0), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=p["week_start"], y=p["best_lo80"], fill="tonexty", line=dict(width=0),
                       fillcolor="rgba(224,122,46,0.18)", name="80% interval"),
            go.Scatter(x=p["week_start"], y=p["y"], name="Actual", line=dict(color=INK, width=2)),
            go.Scatter(x=p["week_start"], y=p[s["best_model"]], name=s["best_model"], line=dict(color=ACCENT, width=2)),
            go.Scatter(x=p["week_start"], y=p["Naive (last week)"], name="Naive", line=dict(color="#9CA3AF", dash="dot")),
        ])
        fig.update_layout(height=380, yaxis_title="Incidents per week", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        a, b = st.columns(2)
        ea = load_csv(f"forecast_{target}_error_by_area.csv").sort_values("WAPE_%")
        a.plotly_chart(px.bar(ea, x="WAPE_%", y="area_name", orientation="h", color_discrete_sequence=[BLUE],
                              title="Error by area (WAPE %)").update_layout(height=520), width="stretch")
        imp = load_csv(f"forecast_{target}_permutation_importance.csv")
        imp.columns = ["feature", "importance"]
        b.plotly_chart(px.bar(imp.head(10)[::-1], x="importance", y="feature", orientation="h",
                              color_discrete_sequence=[INK], title="Permutation importance").update_layout(height=520),
                       width="stretch")
    with tab2:
        st.markdown("**Patrol zones: K-Means, choosing k**")
        st.dataframe(load_csv("kmeans_selection.csv").round(4), hide_index=True, width="stretch")
        hs = load_json("hotspot_summary.json")
        st.caption(f"Selected k = {hs['kmeans_k']} (best silhouette and Davies-Bouldin in the operational range 16-30). "
                   f"Stability across random half-samples: mean ARI {hs['kmeans_stability_ari_mean']:.2f}.")
        st.markdown("**Dense hotspots: DBSCAN vs HDBSCAN**")
        st.dataframe(load_csv("density_clustering_comparison.csv"), hide_index=True, width="stretch")
        st.caption("Rule fixed in advance: 20-80 hotspots, silhouette at least 0.25, then the highest lift "
                   "(share of incidents divided by share of city area).")


def page_data():
    st.title("Data, cleaning and methods")
    dq = load_json("data_quality.json")
    st.markdown(f"""
**Source.** LAPD *Crime Data from 2020 to Present*, City of Los Angeles open-data portal (public domain).
{dq['rows']:,} incident records, 28 columns, one row per reported incident.

**Data-quality issues found and how they were handled**

| Issue | Evidence | Decision |
|---|---|---|
| Records-system change on 7 Mar 2024 | Monthly volume falls {dq['drop_after_system_change_pct']:.0f}% after April 2024 | Analyse 6 Jan 2020 - 3 Mar 2024 only |
| Placeholder times | {dq['time_exactly_1200_share']:.1%} of records at exactly 12:00, {dq['time_exactly_0001_share']:.1%} at 00:01 | Flag 00:01 and the excess at 12:00; exclude from hour analysis |
| Unknown dates filed on the 1st | {dq['occurred_on_1st_of_month_share']:.1%} of records vs {dq['expected_1st_of_month_share']:.1%} expected | Kept, and added a "week contains the 1st" feature to the forecast |
| Missing coordinates | {dq['zero_coordinates']:,} records at (0, 0) | Excluded from spatial analysis only |
| Victim age 0 or negative | {dq['victim_age_zero_share']:.0%} of records | Treated as "no individual victim / unknown" |
| Victim sex X, H, blank | about {dq['missing_share']['Vict Sex']:.0%} blank | Mapped to Unknown |
| Weapon blank | {dq['missing_share']['Weapon Desc']:.0%} | Means no weapon recorded, kept as a category |
| Late reporting | {dq['report_lag_over_30d_share']:.1%} reported more than 30 days after the event | Documented; fraud is most affected |

**Methods.** 140 offence types grouped into 6 crime groups; K-Means patrol zones and DBSCAN/HDBSCAN hotspots on
kilometre-projected coordinates; one-week-ahead forecasts per area with naive, seasonal-naive and moving-average
baselines against Ridge, Random Forest and Poisson gradient boosting, with a strictly time-ordered split.
""")


{"Overview": page_overview, "Time patterns": page_time, "Hotspot map": page_map,
 "Forecast & resource planner": page_forecast, "Model performance": page_models,
 "Data & methods": page_data}[page]()
