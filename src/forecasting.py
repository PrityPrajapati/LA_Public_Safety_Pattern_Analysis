"""Forecast next week's incidents for each of the 21 LAPD areas (resource-planning horizon).

Two targets: all incidents, and violent crime only.
Split is strictly by time: train (2021-2022) -> validation (H1 2023, tuning) -> test (Jul 2023-Feb 2024).

Run from the project root:  python -m src.forecasting
"""
import itertools
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import style
from src.config import (FIGURES, MODELS, RESULTS, APP_DATA, RANDOM_STATE,
                        ANALYSIS_START, ANALYSIS_END, TRAIN_END, VAL_END)
from src.data_prep import load_clean

TARGETS = {"all": None, "violent": "Violent Crime"}
LAGS = [1, 2, 3, 4, 52]


# --------------------------------------------------------------------------- data
def weekly_counts(df, group=None):
    w = df[df["in_analysis_window"]]
    if group:
        w = w[w["crime_group"] == group]
    weeks = pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="W-MON")
    areas = sorted(df["area_name"].cat.categories)
    idx = pd.MultiIndex.from_product([areas, weeks], names=["area_name", "week_start"])
    c = w.groupby(["area_name", "week_start"], observed=True).size().reindex(idx, fill_value=0)
    return c.rename("y").reset_index()


def holiday_weeks():
    hol = USFederalHolidayCalendar().holidays("2019-01-01", "2025-12-31")
    return set(pd.Series(hol).dt.to_period("W-SUN").dt.start_time)


def contains_first(week_start):
    """1 if the Monday-Sunday week starting on week_start contains the 1st of a month."""
    ws = pd.to_datetime(week_start)
    return ((ws + pd.Timedelta(days=6)).dt.month != ws.dt.month).astype(int) | (ws.dt.day == 1).astype(int)


def build_features(wk):
    """Every feature uses only information available before the target week starts."""
    wk = wk.sort_values(["area_name", "week_start"]).copy()
    g = wk.groupby("area_name", observed=True)["y"]
    for L in LAGS:
        wk[f"lag_{L}"] = g.shift(L)
    wk["roll_mean_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    wk["roll_mean_13"] = g.transform(lambda s: s.shift(1).rolling(13).mean())
    wk["roll_std_4"] = g.transform(lambda s: s.shift(1).rolling(4).std())
    wk["trend_4_13"] = wk["roll_mean_4"] - wk["roll_mean_13"]
    city = wk.groupby("week_start")["y"].sum().shift(1)
    wk["city_lag_1"] = wk["week_start"].map(city)
    woy = wk["week_start"].dt.isocalendar().week.astype(int)
    wk["woy_sin"] = np.sin(2 * np.pi * woy / 52.18)
    wk["woy_cos"] = np.cos(2 * np.pi * woy / 52.18)
    wk["holiday_week"] = wk["week_start"].isin(holiday_weeks()).astype(int)
    # Data-quality echo: incidents with an unknown date are filed on the 1st of the month,
    # so weeks containing a 1st run ~9% higher. Both flags are pure calendar facts,
    # known in advance, so they are legitimate forecast inputs.
    wk["month_start_week"] = contains_first(wk["week_start"])
    wk["prev_month_start_week"] = contains_first(wk["week_start"] - pd.Timedelta(weeks=1))
    # Area level = mean weekly count in the TRAINING period only (no leakage)
    area_level = wk[wk["week_start"] <= TRAIN_END].groupby("area_name", observed=True)["y"].mean()
    wk["area_level"] = wk["area_name"].map(area_level).astype(float)
    return wk


FEATURES = [f"lag_{L}" for L in LAGS] + ["roll_mean_4", "roll_mean_13", "roll_std_4", "trend_4_13",
                                         "city_lag_1", "woy_sin", "woy_cos", "holiday_week",
                                         "month_start_week", "prev_month_start_week", "area_level"]


def split(feat):
    f = feat.dropna(subset=FEATURES)            # lag_52 -> first usable week is Jan 2021
    tr = f[f["week_start"] <= TRAIN_END]
    va = f[(f["week_start"] > TRAIN_END) & (f["week_start"] <= VAL_END)]
    te = f[f["week_start"] > VAL_END]
    return tr, va, te


# --------------------------------------------------------------------------- metrics
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": round(float(mean_absolute_error(y, p)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 2),
            "WAPE_%": round(float(100 * np.abs(y - p).sum() / y.sum()), 2),
            "MAPE_%": round(float(100 * np.mean(np.abs(y - p) / np.clip(y, 1, None))), 2),
            "R2": round(float(r2_score(y, p)), 3)}


# --------------------------------------------------------------------------- models
BASELINES = {
    "Naive (last week)": lambda d: d["lag_1"].values,
    "Seasonal naive (same week last year)": lambda d: d["lag_52"].values,
    "Moving average (last 4 weeks)": lambda d: d["roll_mean_4"].values,
}

GRIDS = {
    "Ridge regression": (lambda **k: make_pipeline(StandardScaler(), Ridge(**k)),
                         {"alpha": [0.1, 1.0, 10.0, 100.0]}),
    "Random forest": (lambda **k: RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, **k),
                      {"max_depth": [6, 12, None], "min_samples_leaf": [1, 5, 15]}),
    "Gradient boosting (Poisson)": (lambda **k: HistGradientBoostingRegressor(loss="poisson", max_iter=400,
                                                                              random_state=RANDOM_STATE, **k),
                                    {"learning_rate": [0.03, 0.08], "max_leaf_nodes": [8, 15, 31],
                                     "l2_regularization": [0.0, 1.0]}),
}


def tune(name, tr, va):
    build, grid = GRIDS[name]
    best = None
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        m = build(**params).fit(tr[FEATURES], tr["y"])
        mae = mean_absolute_error(va["y"], m.predict(va[FEATURES]))
        if best is None or mae < best[0]:
            best = (mae, params)
    return best[1]


def run_target(df, key, group):
    print(f"\n===== target: {key}")
    wk = weekly_counts(df, group)
    feat = build_features(wk)
    tr, va, te = split(feat)
    trva = pd.concat([tr, va])
    print("rows train/val/test:", len(tr), len(va), len(te))

    rows, preds_te, preds_va = [], {}, {}
    for name, f in BASELINES.items():
        preds_va[name], preds_te[name] = f(va), f(te)
        rows.append({"model": name, "type": "baseline", "params": "-",
                     **{f"val_{k}": v for k, v in metrics(va["y"], preds_va[name]).items()},
                     **{f"test_{k}": v for k, v in metrics(te["y"], preds_te[name]).items()}})
    fitted = {}
    for name in GRIDS:
        params = tune(name, tr, va)
        m_tr = GRIDS[name][0](**params).fit(tr[FEATURES], tr["y"])       # for validation score/residuals
        preds_va[name] = m_tr.predict(va[FEATURES])
        m = GRIDS[name][0](**params).fit(trva[FEATURES], trva["y"])      # refit on train+val for test
        preds_te[name] = m.predict(te[FEATURES])
        fitted[name] = (m, m_tr, params)
        rows.append({"model": name, "type": "ML", "params": json.dumps(params),
                     **{f"val_{k}": v for k, v in metrics(va["y"], preds_va[name]).items()},
                     **{f"test_{k}": v for k, v in metrics(te["y"], preds_te[name]).items()}})
        print(rows[-1])
    res = pd.DataFrame(rows)

    # Model selection uses VALIDATION MAE only; the test set is looked at once, afterwards.
    best_name = res[res["type"] == "ML"].sort_values("val_MAE").iloc[0]["model"]
    best_model, best_tr_model, best_params = fitted[best_name]

    # 80% prediction interval from validation residuals of the train-only model
    resid = va["y"].values - preds_va[best_name]
    q_lo, q_hi = np.quantile(resid, [0.10, 0.90])
    lo, hi = preds_te[best_name] + q_lo, preds_te[best_name] + q_hi
    coverage = float(((te["y"].values >= lo) & (te["y"].values <= hi)).mean())

    out = te[["area_name", "week_start", "y"]].copy()
    for name, p in preds_te.items():
        out[name] = p
    out["best_lo80"], out["best_hi80"] = lo, hi

    # Error analysis by area and by week
    err = out.assign(abs_err=np.abs(out["y"] - out[best_name]))
    naive_err = np.abs(out["y"] - out["Naive (last week)"])
    by_area = err.groupby("area_name", observed=True).agg(mean_weekly=("y", "mean"), MAE=("abs_err", "mean"))
    by_area["WAPE_%"] = 100 * by_area["MAE"] / by_area["mean_weekly"]
    by_area["naive_MAE"] = naive_err.groupby(out["area_name"], observed=True).mean()
    by_area["improvement_vs_naive_%"] = 100 * (1 - by_area["MAE"] / by_area["naive_MAE"])
    by_week = err.groupby("week_start").agg(actual=("y", "sum"), pred=(best_name, "sum"), MAE=("abs_err", "mean"))

    # Permutation importance on test data (which inputs does the model rely on?)
    pi = permutation_importance(best_model, te[FEATURES], te["y"], n_repeats=5,
                                random_state=RANDOM_STATE, scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)

    # Live forecast: the week after the data ends (4 Mar 2024), per area
    nxt = next_week_features(wk, feat)
    nxt["forecast"] = best_model.predict(nxt[FEATURES])
    nxt["lo80"], nxt["hi80"] = nxt["forecast"] + q_lo, (nxt["forecast"] + q_hi)
    nxt["lo80"] = nxt["lo80"].clip(lower=0)

    joblib.dump({"model": best_model, "name": best_name, "params": best_params, "features": FEATURES,
                 "resid_q10": float(q_lo), "resid_q90": float(q_hi),
                 "area_level": feat.groupby("area_name", observed=True)["area_level"].first().to_dict()},
                MODELS / f"forecast_{key}.joblib")
    res.to_csv(RESULTS / f"forecast_{key}_model_comparison.csv", index=False)
    by_area.round(2).to_csv(RESULTS / f"forecast_{key}_error_by_area.csv")
    imp.round(3).to_csv(RESULTS / f"forecast_{key}_permutation_importance.csv")
    out.to_parquet(APP_DATA / f"forecast_{key}_test_predictions.parquet", index=False)
    nxt[["area_name", "week_start", "forecast", "lo80", "hi80"]].to_parquet(
        APP_DATA / f"forecast_{key}_next_week.parquet", index=False)
    wk.to_parquet(APP_DATA / f"weekly_{key}.parquet", index=False)

    best_row = res[res["model"] == best_name].iloc[0]
    naive_row = res[res["model"] == "Naive (last week)"].iloc[0]
    best_base = res[res["type"] == "baseline"].sort_values("val_MAE").iloc[0]
    summary = {
        "target": key, "best_model": best_name, "best_params": best_params,
        "test_metrics": {k.replace("test_", ""): best_row[k] for k in res.columns if k.startswith("test_")},
        "val_metrics": {k.replace("val_", ""): best_row[k] for k in res.columns if k.startswith("val_")},
        "best_baseline": best_base["model"], "best_baseline_test_MAE": best_base["test_MAE"],
        "improvement_vs_naive_test_MAE_%": round(100 * (1 - best_row["test_MAE"] / naive_row["test_MAE"]), 1),
        "improvement_vs_best_baseline_test_MAE_%": round(100 * (1 - best_row["test_MAE"] / best_base["test_MAE"]), 1),
        "interval80_residual_q10_q90": [round(float(q_lo), 2), round(float(q_hi), 2)],
        "interval80_test_coverage": round(coverage, 3),
        "mean_weekly_per_area_test": round(float(te["y"].mean()), 1),
        "worst_areas_WAPE": by_area.sort_values("WAPE_%", ascending=False).head(3)["WAPE_%"].round(1).to_dict(),
        "best_areas_WAPE": by_area.sort_values("WAPE_%").head(3)["WAPE_%"].round(1).to_dict(),
        "top_features": imp.head(5).round(3).to_dict(),
        "worst_weeks": by_week.sort_values("MAE", ascending=False).head(3)["MAE"].round(1)
                              .rename(lambda d: str(d.date())).to_dict(),
        "next_week": str(nxt["week_start"].iloc[0].date()),
        "next_week_city_forecast": round(float(nxt["forecast"].sum()), 0),
    }
    plots(key, res, out, best_name, by_area, imp, wk)
    return summary


def next_week_features(wk, feat):
    """Features for the week straight after the last observed week."""
    last = wk["week_start"].max()
    nxt_week = last + pd.Timedelta(weeks=1)
    ext = pd.concat([wk, pd.DataFrame({"area_name": wk["area_name"].unique(), "week_start": nxt_week, "y": np.nan})])
    ext["area_name"] = ext["area_name"].astype(str)
    f = build_features(ext)
    # area_level must stay the training-period value
    f["area_level"] = f["area_name"].map(feat.groupby("area_name", observed=True)["area_level"].first())
    return f[f["week_start"] == nxt_week].reset_index(drop=True)


# --------------------------------------------------------------------------- plots
def plots(key, res, out, best_name, by_area, imp, wk):
    label = "all incidents" if key == "all" else "violent crime"
    # 1. model comparison
    r = res.sort_values("test_MAE", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = [style.ACCENT if m == best_name else (style.MUTED if t == "baseline" else style.BLUE)
              for m, t in zip(r["model"], r["type"])]
    ax.barh(r["model"][::-1], r["test_MAE"][::-1], color=colors[::-1])
    for i, v in enumerate(r["test_MAE"][::-1]):
        ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=9.5, color=style.INK)
    ax.set_xlabel("Test MAE (incidents per area per week, lower is better)")
    ax.set_title(f"Forecast accuracy on the unseen test period ({label})")
    ax.grid(axis="y", visible=False)
    style.save(fig, FIGURES / f"15_forecast_{key}_model_comparison.png")

    # 2. city-level actual vs forecast across the whole history
    city = wk.groupby("week_start")["y"].sum()
    city = city[city.index >= "2022-01-01"]
    tp = out.groupby("week_start")[[best_name, "Naive (last week)", "best_lo80", "best_hi80", "y"]].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(city.index, city.values, color=style.INK, lw=1.4, label="Actual (city total)")
    ax.fill_between(tp.index, tp["best_lo80"], tp["best_hi80"], color=style.ACCENT, alpha=0.18,
                    label="80% interval (sum of area intervals)")
    ax.plot(tp.index, tp[best_name], color=style.ACCENT, lw=2, label=f"{best_name} forecast")
    ax.axvspan(pd.Timestamp(VAL_END), tp.index.max(), color=style.BLUE, alpha=0.05)
    ax.text(pd.Timestamp(VAL_END) + pd.Timedelta(days=7), city.max() * 1.01, "test period", color=style.BLUE, fontsize=9)
    ax.set_title(f"Weekly {label}, city total (2022 onward): actual vs one-week-ahead forecast")
    ax.set_ylabel("Incidents per week")
    ax.legend(loc="lower left", fontsize=9)
    style.save(fig, FIGURES / f"16_forecast_{key}_timeline.png")

    # 3. error by area
    b = by_area.sort_values("WAPE_%")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(b.index, b["WAPE_%"], color=style.BLUE)
    ax.axvline(100 * (np.abs(out["y"] - out[best_name]).sum() / out["y"].sum()), color=style.ACCENT, ls="--", lw=1.2,
               label="Overall WAPE")
    ax.set_xlabel("WAPE % on test weeks (lower is better)")
    ax.set_title(f"Where the forecast is harder ({label})")
    ax.legend()
    ax.grid(axis="y", visible=False)
    style.save(fig, FIGURES / f"17_forecast_{key}_error_by_area.png")

    # 4. feature importance
    i = imp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(i.index, i.values, color=style.INK)
    ax.set_xlabel("Increase in test MAE when the feature is shuffled")
    ax.set_title(f"What drives the forecast ({label})")
    ax.grid(axis="y", visible=False)
    style.save(fig, FIGURES / f"18_forecast_{key}_feature_importance.png")

    # 5. residual distribution
    resid = out["y"] - out[best_name]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(resid, bins=40, color=style.BLUE, alpha=0.85)
    ax.axvline(0, color=style.INK, lw=1)
    ax.axvline(resid.mean(), color=style.ACCENT, lw=1.5, ls="--", label=f"Mean error {resid.mean():+.1f}")
    ax.set_xlabel("Actual minus forecast (incidents per area-week)")
    ax.set_title(f"Test residuals ({label})")
    ax.legend()
    style.save(fig, FIGURES / f"19_forecast_{key}_residuals.png")


def main():
    style.apply()
    df = load_clean()
    summaries = {key: run_target(df, key, group) for key, group in TARGETS.items()}
    (RESULTS / "forecast_summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(json.dumps(summaries, indent=2, default=str))


if __name__ == "__main__":
    main()
