"""Spatial pattern mining: crime concentration, patrol zones (K-Means) and dense hotspots
(DBSCAN vs HDBSCAN).

Run from the project root:  python -m src.hotspots
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN, MiniBatchKMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

from src import style
from src.config import (FIGURES, MODELS, RESULTS, APP_DATA, RANDOM_STATE,
                        KM_PER_DEG_LAT, KM_PER_DEG_LON)
from src.data_prep import load_clean

LAT0, LON0 = 34.05, -118.35   # reference point for the local km projection
CELL_KM = 0.5                 # grid cell size for the concentration analysis
LIFT_CELL_KM = 0.25           # finer grid used to measure hotspot footprint
EVAL_N = 20_000               # sample size for silhouette (O(n^2) metric)
HDB_N = 60_000                # sample size for density clustering


def to_km(lat, lon):
    """Project lat/lon to a flat x/y grid in kilometres (equirectangular, fine at city scale).
    Clustering in degrees would distort distances: 1 degree of longitude is only
    ~92 km here versus ~111 km for latitude."""
    return np.column_stack([(lon - LON0) * KM_PER_DEG_LON, (lat - LAT0) * KM_PER_DEG_LAT])


def to_latlon(xy):
    return xy[:, 1] / KM_PER_DEG_LAT + LAT0, xy[:, 0] / KM_PER_DEG_LON + LON0


def concentration(X):
    """How concentrated is crime? Share of incidents in the busiest x% of 0.5 km cells."""
    cells = pd.Series(list(zip((X[:, 0] // CELL_KM).astype(int), (X[:, 1] // CELL_KM).astype(int))))
    counts = cells.value_counts().sort_values(ascending=False).values
    cum = counts.cumsum() / counts.sum()
    n = len(counts)
    out = {"occupied_cells": int(n), "cell_km": CELL_KM,
           "city_area_km2_occupied": round(n * CELL_KM ** 2, 1)}
    for pct in (1, 5, 10, 20):
        out[f"share_in_top_{pct}pct_cells"] = round(float(cum[max(int(n * pct / 100), 1) - 1]), 4)
    cells_for_half = int(np.searchsorted(cum, 0.5) + 1)
    out["pct_cells_holding_50pct_incidents"] = round(100 * cells_for_half / n, 2)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(np.arange(1, n + 1) / n * 100, cum * 100, color=style.INK, lw=2.4)
    ax.plot([0, 100], [0, 100], color=style.MUTED, ls=":", lw=1, label="If crime were spread evenly")
    ax.axvline(out["pct_cells_holding_50pct_incidents"], color=style.ACCENT, lw=1.2)
    ax.text(out["pct_cells_holding_50pct_incidents"] + 2, 45,
            f"{out['pct_cells_holding_50pct_incidents']:.0f}% of cells hold\nhalf of all incidents",
            color=style.ACCENT, fontsize=10)
    ax.set_xlabel("% of 0.5 km grid cells (busiest first)")
    ax.set_ylabel("% of incidents")
    ax.set_title("Crime is highly concentrated in space")
    ax.legend(loc="lower right")
    style.save(fig, FIGURES / "12_concentration_curve.png")
    return out


def cluster_metrics(X, labels, rng_idx):
    """Internal validity on a fixed sample; noise (-1) excluded for density methods."""
    Xs, ls = X[rng_idx], labels[rng_idx]
    m = ls >= 0
    if len(np.unique(ls[m])) < 2:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    return {
        "silhouette": round(float(silhouette_score(Xs[m], ls[m])), 4),
        "davies_bouldin": round(float(davies_bouldin_score(Xs[m], ls[m])), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(Xs[m], ls[m])), 1),
    }


def kmeans_search(X, eval_idx):
    rows = []
    for k in (8, 12, 16, 21, 25, 30, 40, 50):
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, batch_size=8192,
                             n_init=5, max_iter=300).fit(X)
        r = {"k": k, "inertia": float(km.inertia_)}
        r.update(cluster_metrics(X, km.labels_, eval_idx))
        rows.append(r)
        print("kmeans", r)
    return pd.DataFrame(rows)


def kmeans_stability(X, k, rng):
    """Refit on 5 different random halves; compare labels on a shared holdout (ARI)."""
    hold = rng.choice(len(X), 20_000, replace=False)
    preds = []
    for seed in range(5):
        idx = rng.choice(len(X), len(X) // 2, replace=False)
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=8192, n_init=5).fit(X[idx])
        preds.append(km.predict(X[hold]))
    aris = [adjusted_rand_score(preds[i], preds[j]) for i in range(5) for j in range(i + 1, 5)]
    return round(float(np.mean(aris)), 4), round(float(np.min(aris)), 4)


def footprint(xy, cell=LIFT_CELL_KM):
    """Distinct 250 m grid cells touched by a set of points (ids)."""
    return np.unique(np.floor(xy[:, 0] / cell).astype(np.int64) * 100_000
                     + np.floor(xy[:, 1] / cell).astype(np.int64))


def density_methods(X, rng, eval_idx_sample):
    """DBSCAN (one global density threshold) vs HDBSCAN (adapts to local density).

    Besides internal metrics we measure what a planner cares about:
      coverage = share of incidents that fall inside hotspots
      area_share = share of the city's occupied 250 m cells the hotspots cover
      lift = coverage / area_share  (how many times denser than average the hotspots are)
    """
    s_idx = rng.choice(len(X), HDB_N, replace=False)
    Xs = X[s_idx]
    city_cells = len(footprint(Xs))
    rows, fitted = [], {}
    configs = [("DBSCAN", {"eps": 0.15, "min_samples": 40}),
               ("DBSCAN", {"eps": 0.2, "min_samples": 60}),
               ("DBSCAN", {"eps": 0.25, "min_samples": 40}),
               ("DBSCAN", {"eps": 0.3, "min_samples": 100}),
               ("HDBSCAN", {"min_cluster_size": 150, "min_samples": 30}),
               ("HDBSCAN", {"min_cluster_size": 300, "min_samples": 30, "cluster_selection_method": "leaf"}),
               ("HDBSCAN", {"min_cluster_size": 150, "min_samples": 150, "cluster_selection_method": "leaf"})]
    for name, params in configs:
        model = (DBSCAN(**params) if name == "DBSCAN" else HDBSCAN(copy=True, **params)).fit(Xs)
        lab = model.labels_
        inside = lab >= 0
        coverage = float(inside.mean())
        area_share = len(footprint(Xs[inside])) / city_cells if inside.any() else 0.0
        r = {"method": name, "params": json.dumps(params),
             "n_clusters": int(len(np.unique(lab[inside]))),
             "coverage": round(coverage, 4),
             "area_share": round(area_share, 4),
             "lift": round(coverage / area_share, 2) if area_share else None}
        r.update(cluster_metrics(Xs, lab, eval_idx_sample))
        rows.append(r)
        fitted[(name, r["params"])] = model
        print(r)
    return pd.DataFrame(rows), s_idx, fitted


def profile_hotspots(df_s, labels, xy):
    """Describe each hotspot so a planner can act on it."""
    d = df_s.assign(cluster=labels, x=xy[:, 0], y=xy[:, 1])
    d = d[d["cluster"] >= 0]
    rows = []
    for c, g in d.groupby("cluster"):
        hour = g.loc[~g["time_placeholder"], "hour"]
        rows.append({
            "cluster": int(c),
            "sample_incidents": len(g),
            "lat": float(g["lat"].mean()), "lon": float(g["lon"].mean()),
            "footprint_km2": round(len(footprint(g[["x", "y"]].values)) * LIFT_CELL_KM ** 2, 2),
            "main_lapd_area": g["area_name"].mode().iat[0],
            "top_crime_group": g["crime_group"].mode().iat[0],
            "violent_share": round(float((g["crime_group"] == "Violent Crime").mean()), 3),
            "vehicle_share": round(float((g["crime_group"] == "Vehicle Crime").mean()), 3),
            "street_share": round(float((g["premise"] == "STREET").mean()), 3),
            "peak_hour": int(hour.mode().iat[0]) if len(hour) else None,
            "top_offence": g["crime_desc"].mode().iat[0],
        })
    p = pd.DataFrame(rows)
    p["density_per_km2"] = (p["sample_incidents"] / p["footprint_km2"].clip(lower=0.0625)).round(1)
    return p.sort_values("sample_incidents", ascending=False).reset_index(drop=True)


def plot_kmeans_selection(tab, k_best):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(tab["k"], tab["silhouette"], marker="o", color=style.INK, lw=2, label="Silhouette (higher = better)")
    ax1.set_xlabel("Number of zones k")
    ax1.set_ylabel("Silhouette")
    ax2 = ax1.twinx()
    ax2.plot(tab["k"], tab["davies_bouldin"], marker="s", color=style.ACCENT, lw=2, label="Davies-Bouldin (lower = better)")
    ax2.set_ylabel("Davies-Bouldin")
    ax2.grid(False)
    ax1.axvline(k_best, color=style.BLUE, ls="--", lw=1.2)
    ax1.set_title(f"Choosing the number of patrol zones (selected k = {k_best})")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    style.save(fig, FIGURES / "13_kmeans_selection.png")


def _place_labels(pts, min_gap):
    """Greedy label placement: try 8 directions around each point, keep the first
    spot that is at least min_gap away from labels already placed."""
    placed = []
    dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1), (0, 1.3), (0, -1.3), (1.3, 0), (-1.3, 0)]
    for x, y in pts:
        best = None
        for mult in (1.0, 1.8, 2.6):
            for dx, dy in dirs:
                cx, cy = x + dx * min_gap * mult, y + dy * min_gap * mult
                if all((cx - px) ** 2 + (cy - py) ** 2 >= min_gap ** 2 for px, py in placed):
                    best = (cx, cy)
                    break
            if best:
                break
        placed.append(best or (x, y))
    return placed


def plot_maps(X, km_labels, centers, Xs, labels, prof, title_right):
    lat, lon = to_latlon(X)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), 60_000, replace=False)
    asp = 1 / np.cos(np.radians(34.05))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.scatter(lon[idx], lat[idx], c=km_labels[idx] % 20, cmap="tab20", s=0.6, alpha=0.6, rasterized=True)
    clat, clon = to_latlon(centers)
    ax.scatter(clon, clat, c=style.INK, s=26, marker="x")
    ax.set_title(f"K-Means patrol zones (k = {len(centers)})")
    ax.set_aspect(asp)
    ax.grid(False)

    ax = axes[1]
    ax.scatter(lon[idx], lat[idx], c="#DAD6CE", s=0.5, rasterized=True)
    size = 20 + 900 * prof["est_incidents_total"] / prof["est_incidents_total"].max()
    ax.scatter(prof["lon"], prof["lat"], s=size, c=[style.GROUP_COLORS[g] for g in prof["top_crime_group"]],
               alpha=0.8, edgecolor="white", linewidth=0.8, zorder=3)
    top = prof.head(10)
    spots = _place_labels(list(zip(top["lon"], top["lat"])), min_gap=0.011)
    for (rank, r), (tx, ty) in zip(top.iterrows(), spots):
        ax.annotate(str(rank), (r["lon"], r["lat"]), xytext=(tx, ty), fontsize=9, fontweight="bold",
                    color=style.INK, ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=style.INK, lw=0.8),
                    arrowprops=dict(arrowstyle="-", color=style.INK, lw=0.7))
    ax.set_xlim(-118.52, -118.17)
    ax.set_ylim(33.93, 34.16)
    for g, c in style.GROUP_COLORS.items():
        if g in set(prof["top_crime_group"]):
            ax.scatter([], [], s=60, c=c, label=g)
    ax.legend(loc="lower left", fontsize=8.5, title="Dominant group (size = volume)", title_fontsize=8.5)
    ax.set_title(title_right)
    ax.set_aspect(asp)
    ax.grid(False)
    for a in axes:
        a.set_xlabel("Longitude")
        a.set_ylabel("Latitude")
    style.save(fig, FIGURES / "14_zones_and_hotspots.png")


def main():
    style.apply()
    rng = np.random.default_rng(RANDOM_STATE)
    df = load_clean()
    df = df[df["in_analysis_window"] & df["lat"].notna()].reset_index(drop=True)
    X = to_km(df["lat"].values.astype(float), df["lon"].values.astype(float))
    print("points", X.shape)

    conc = concentration(X)
    print(conc)

    eval_idx = rng.choice(len(X), EVAL_N, replace=False)
    tab = kmeans_search(X, eval_idx)
    # Selection rule: best silhouette among k = 16-30 (the operationally useful range:
    # LAPD runs 21 areas, so zones in that range can be staffed as patrol sectors).
    cand = tab[(tab["k"] >= 16) & (tab["k"] <= 30)]
    k_best = int(cand.loc[cand["silhouette"].idxmax(), "k"])
    plot_kmeans_selection(tab, k_best)
    km = MiniBatchKMeans(n_clusters=k_best, random_state=RANDOM_STATE, batch_size=8192, n_init=10).fit(X)
    ari_mean, ari_min = kmeans_stability(X, k_best, rng)
    km_labels = km.labels_

    dens_tab, s_idx, fitted = density_methods(X, rng, np.arange(EVAL_N))
    # Selection rule (fixed before looking at the winner): a planner-sized list
    # (20-80 hotspots) with acceptable separation (silhouette >= 0.25), then the
    # highest lift, because the goal is to cover many incidents with little area.
    ok = dens_tab[dens_tab["n_clusters"].between(20, 80) & (dens_tab["silhouette"] >= 0.25)]
    best = ok.loc[ok["lift"].idxmax()]
    model = fitted[(best["method"], best["params"])]
    hdb_labels = model.labels_
    Xs = X[s_idx]
    df_s = df.iloc[s_idx].reset_index(drop=True)
    prof = profile_hotspots(df_s, hdb_labels, Xs)
    scale = len(X) / len(Xs)
    prof["est_incidents_total"] = (prof["sample_incidents"] * scale).round().astype(int)
    prof.index = prof.index + 1
    prof.index.name = "rank"

    # New-point rule for DBSCAN: a location is inside hotspot c if it lies within eps
    # of one of c's core points. Saved so the app can check any address.
    core_xy = Xs[model.core_sample_indices_]
    core_lab = hdb_labels[model.core_sample_indices_]
    eps = json.loads(best["params"]).get("eps", 0.2)

    plot_maps(X, km_labels, km.cluster_centers_, Xs, hdb_labels, prof,
              f"{best['method']} hotspots, zoom on central LA (eps = {eps * 1000:.0f} m, {len(prof)} hotspots city-wide)")

    # Zone profile for K-Means
    z = df.assign(zone=km_labels)
    zc = to_latlon(km.cluster_centers_)
    zone_prof = z.groupby("zone").agg(incidents=("dr_no", "size"),
                                      violent_share=("crime_group", lambda s: (s == "Violent Crime").mean()),
                                      main_lapd_area=("area_name", lambda s: s.mode().iat[0])).reset_index()
    zone_prof["lat"], zone_prof["lon"] = zc[0][zone_prof["zone"]], zc[1][zone_prof["zone"]]
    zone_prof["share_of_incidents"] = zone_prof["incidents"] / zone_prof["incidents"].sum()
    zone_prof["violent_share"] = zone_prof["violent_share"].round(3)

    # Save models and results
    # Saved as plain arrays (not sklearn objects) so they load on any library version
    joblib.dump({"centers_km": km.cluster_centers_, "lat0": LAT0, "lon0": LON0,
                 "km_per_deg_lat": KM_PER_DEG_LAT, "km_per_deg_lon": KM_PER_DEG_LON},
                MODELS / "kmeans_zones.joblib")
    rank_of = {c: r for r, c in zip(prof.index, prof["cluster"])}
    joblib.dump({"core_xy": core_xy, "core_rank": np.array([rank_of[c] for c in core_lab]),
                 "eps_km": eps, "lat0": LAT0, "lon0": LON0, "method": best["method"], "params": best["params"]},
                MODELS / "dbscan_hotspots.joblib")
    tab.to_csv(RESULTS / "kmeans_selection.csv", index=False)
    dens_tab.to_csv(RESULTS / "density_clustering_comparison.csv", index=False)
    prof.to_csv(RESULTS / "hotspot_profiles.csv")
    zone_prof.to_csv(RESULTS / "zone_profiles.csv", index=False)

    # Small point sample with labels for the Streamlit map
    app_pts = df_s[["lat", "lon", "area_name", "crime_group", "hour", "weekday", "year"]].copy()
    app_pts["hotspot"] = hdb_labels
    app_pts["hotspot_rank"] = app_pts["hotspot"].map(rank_of).fillna(0).astype(int)
    app_pts["zone"] = km.predict(Xs)
    app_pts.to_parquet(APP_DATA / "map_points.parquet", index=False)
    prof.to_parquet(APP_DATA / "hotspot_profiles.parquet")
    zone_prof.to_parquet(APP_DATA / "zone_profiles.parquet", index=False)

    km_row = tab[tab["k"] == k_best].iloc[0].to_dict()
    summary = {"concentration": conc, "kmeans_k": k_best, "kmeans_metrics": km_row,
               "kmeans_stability_ari_mean": ari_mean, "kmeans_stability_ari_min": ari_min,
               "hotspot_selected": best.to_dict(),
               "hotspot_min_samples_full_data_equiv": int(round(json.loads(best["params"]).get("min_samples", 0) * scale)),
               "top10_hotspots_share_of_all_incidents": round(float(prof.head(10)["sample_incidents"].sum() / len(Xs)), 4),
               "top10_hotspots_footprint_km2": round(float(prof.head(10)["footprint_km2"].sum()), 2),
               "all_hotspots_footprint_km2": round(float(prof["footprint_km2"].sum()), 2)}
    (RESULTS / "hotspot_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(prof.head(12).to_string())


if __name__ == "__main__":
    main()
