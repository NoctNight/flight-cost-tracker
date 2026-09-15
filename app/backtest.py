"""Out-of-sample validation.

Everything fitted -- booking curve, fare-index normalisation, per-series
regressions -- uses only observations with search_date <= cutoff AND
flight_date <= cutoff. The test window is then scored against reality.

Two evaluations:

1. Price-level forecast: predict the daily fare index for every series and
   flight_date in the test window; score MAPE / bias / out-of-sample R²
   against the realised index, and against two honest baselines
   (train-mean per series, and seasonal-naive = same date last year).

2. Buy-timing advice: for each test flight, stand at `anchor_dtd` days
   before departure knowing only the train-fitted booking curve, decide
   which remaining day to buy, then look up the fare actually observed on
   that day. Compare with buying immediately and with the oracle minimum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import models as M


def _seasonal_naive(idx_train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Same series, same date one year earlier (364d keeps the weekday)."""
    tr = idx_train.set_index(["series", "flight_date"])["fare"]
    keys = pd.MultiIndex.from_arrays([
        test["series"], test["flight_date"] - pd.Timedelta(days=364)])
    return tr.reindex(keys).to_numpy()


def price_backtest(fares: pd.DataFrame, oil: pd.Series,
                   cutoff: str, test_end: str) -> dict:
    cutoff, test_end = pd.Timestamp(cutoff), pd.Timestamp(test_end)
    train = fares[(fares["search_date"] <= cutoff) & (fares["flight_date"] <= cutoff)]
    curve = M.fit_booking_curve(train)
    idx_train = M.daily_index(train, curve, agg=M.INDEX_AGG)
    oa = M.oil_lag_analysis(idx_train, oil)
    oil_term = M.make_oil_term(oil, oa["best_lag_days"],
                               idx_train["flight_date"].min(),
                               pd.Timestamp(test_end), freeze_after=cutoff)
    oil_beta = M.estimate_oil_beta(idx_train, oil_term, **M.MODEL_CONFIG)
    models = M.fit_series_models(idx_train, oil_term, oil_beta,
                                 **M.MODEL_CONFIG)

    # realised index over the test window, normalised with the TRAIN curve
    test_obs = fares[(fares["flight_date"] > cutoff) & (fares["flight_date"] <= test_end)]
    idx_test = M.daily_index(test_obs, curve, agg=M.INDEX_AGG)

    rows, pooled = [], {"y": [], "yhat": [], "naive": [], "mean": []}
    for s, m in models.items():
        sub = idx_test[idx_test["series"] == s].dropna(subset=["fare"])
        if len(sub) < 30:
            continue
        y = sub["fare"].to_numpy()
        yhat, _ = m.predict(pd.DatetimeIndex(sub["flight_date"]))
        naive = _seasonal_naive(idx_train, sub)
        mean_b = np.full_like(y, idx_train.loc[idx_train["series"] == s, "fare"].mean())
        ok = np.isfinite(naive)
        rows.append({
            "series": s, "n": int(len(y)),
            "mape_model": float(np.mean(np.abs(yhat - y) / y) * 100),
            "mape_seasonal_naive": float(np.mean(np.abs(naive[ok] - y[ok]) / y[ok]) * 100),
            "mape_train_mean": float(np.mean(np.abs(mean_b - y) / y) * 100),
            "bias_pct": float(np.mean((yhat - y) / y) * 100),
            "r2_oos": float(1 - np.sum((yhat - y) ** 2) / np.sum((y - y.mean()) ** 2)),
        })
        pooled["y"].append(y); pooled["yhat"].append(yhat)
        pooled["naive"].append(naive[ok]); pooled["mean"].append(mean_b)
        pooled.setdefault("y_naive", []).append(y[ok])

    y = np.concatenate(pooled["y"]); yhat = np.concatenate(pooled["yhat"])
    yn = np.concatenate(pooled["y_naive"]); nv = np.concatenate(pooled["naive"])
    mb = np.concatenate(pooled["mean"])
    return {
        "cutoff": str(cutoff.date()), "test_end": str(test_end.date()),
        "n_series": len(rows), "n_obs": int(len(y)),
        "mape_model": round(float(np.mean(np.abs(yhat - y) / y) * 100), 2),
        "mape_seasonal_naive": round(float(np.mean(np.abs(nv - yn) / yn) * 100), 2),
        "mape_train_mean": round(float(np.mean(np.abs(mb - y) / y) * 100), 2),
        "bias_pct": round(float(np.mean((yhat - y) / y) * 100), 2),
        "r2_oos": round(float(1 - np.sum((yhat - y) ** 2) / np.sum((y - y.mean()) ** 2)), 3),
        "per_series": sorted(rows, key=lambda r: r["mape_model"]),
    }


def timing_backtest(fares: pd.DataFrame, cutoff: str, test_end: str,
                    lead_days: int = 30, anchor_tol: int = 8,
                    min_quotes: int = 8, sample_every_days: int = 7) -> dict:
    """Stand some days before departure knowing only the train-fitted booking
    curve, pick a buy day, and pay whatever was actually quoted then.

    The standing point is route-relative: `lead_days` before that route's own
    curve minimum, capped at the quote horizon. A fixed anchor would stand
    *past* the optimum on long-haul routes (their curve bottoms out ~75 days
    out vs ~46 short-haul), where the advice can only ever say "buy now" and
    scores a trivial zero.

    A real tracker samples horizons rather than quoting every flight every
    day, so the anchor is the observed quote nearest the standing point
    (within `anchor_tol`) and the purchase settles at the nearest quote to
    the recommended day -- you can only buy at a price you actually saw.
    """
    cutoff, test_end = pd.Timestamp(cutoff), pd.Timestamp(test_end)
    train = fares[(fares["search_date"] <= cutoff) & (fares["flight_date"] <= cutoff)]
    curve = M.fit_booking_curve(train)

    grid = np.arange(1, 181)
    plan = {}          # route -> (anchor_dtd, recommended_dtd_from_anchor)
    for route in fares["route"].astype(object).unique():
        cv = curve.value(grid, route)
        trough = int(grid[int(np.argmin(cv))])
        anchor = min(trough + lead_days, 170)
        cand = np.arange(1, anchor + 1)
        plan[route] = (anchor, int(cand[int(np.argmin(curve.value(cand, route)))]),
                       trough)

    max_anchor = max(a for a, _, _ in plan.values()) + anchor_tol
    test = fares[(fares["flight_date"] > cutoff + pd.Timedelta(days=max_anchor)) &
                 (fares["flight_date"] <= test_end) &
                 (fares["dtd"] >= 1) & (fares["dtd"] <= max_anchor)]
    # weekly sample of departure dates to keep this light
    test = test[test["flight_date"].dt.dayofyear % sample_every_days == 0]

    res, per_trough = [], {}
    for (s, fd), traj in test.groupby(["series", "flight_date"], observed=True):
        route = str(traj["route"].iloc[0])
        anchor_dtd, rec_dtd, trough = plan[route]
        traj = traj.sort_values("dtd", ascending=False)
        dtds = traj["dtd"].to_numpy()
        fares_t = traj["fare"].to_numpy()
        a = int(np.argmin(np.abs(dtds - anchor_dtd)))
        if abs(dtds[a] - anchor_dtd) > anchor_tol:
            continue
        dtds, fares_t = dtds[a:], fares_t[a:]      # only what is still buyable
        if len(dtds) < min_quotes:
            continue
        immediate = fares_t[0]
        realized = fares_t[np.argmin(np.abs(dtds - rec_dtd))]
        oracle = fares_t.min()
        res.append((immediate, realized, oracle))
        bucket = "books early (trough > 60d)" if trough > 60 else "books late (trough <= 60d)"
        per_trough.setdefault(bucket, []).append((immediate - realized) / immediate * 100)
    if not res:
        return {"error": "no test trajectories"}
    a = np.array(res)
    imm, real, orc = a[:, 0], a[:, 1], a[:, 2]
    saving = (imm - real) / imm * 100
    possible = (imm - orc) / imm * 100
    with np.errstate(invalid="ignore", divide="ignore"):
        capture = np.where(imm - orc > 1e-9, (imm - real) / (imm - orc), np.nan)
    dollars = imm - real
    return {
        "lead_days": lead_days,
        "by_curve_shape": {
            k: {"n": len(v), "avg_saving_pct": round(float(np.mean(v)), 2),
                "helped_pct": round(float(np.mean(np.array(v) > 0) * 100), 1)}
            for k, v in sorted(per_trough.items())},
        "n_flights": int(len(a)),
        "avg_saving_vs_buy_now_pct": round(float(saving.mean()), 2),
        "median_saving_vs_buy_now_pct": round(float(np.median(saving)), 2),
        "std_saving_pct": round(float(saving.std(ddof=1)), 2),
        "std_saving_usd": round(float(dollars.std(ddof=1)), 2),
        "avg_saving_usd": round(float(dollars.mean()), 2),
        "saving_pct_percentiles": {
            "p10": round(float(np.percentile(saving, 10)), 2),
            "p25": round(float(np.percentile(saving, 25)), 2),
            "p50": round(float(np.percentile(saving, 50)), 2),
            "p75": round(float(np.percentile(saving, 75)), 2),
            "p90": round(float(np.percentile(saving, 90)), 2),
        },
        "avg_possible_saving_pct": round(float(possible.mean()), 2),
        "capture_ratio": round(float(np.nanmean(np.clip(capture, -1, 1))), 3),
        "pct_flights_advice_helped": round(float((saving > 0).mean() * 100), 1),
        "pct_flights_advice_hurt": round(float((saving < -0.5).mean() * 100), 1),
    }


def run(fares: pd.DataFrame, oil: pd.Series, splits=None) -> dict:
    splits = splits or [("2024-12-31", "2025-12-31"), ("2025-12-31", "2026-09-01")]
    out = []
    for cutoff, test_end in splits:
        p = price_backtest(fares, oil, cutoff, test_end)
        t = timing_backtest(fares, cutoff, test_end)
        out.append({"price": p, "timing": t})
    return {"splits": out}
