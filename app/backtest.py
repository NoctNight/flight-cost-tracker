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


def price_backtest(fares: pd.DataFrame, cutoff: str, test_end: str) -> dict:
    cutoff, test_end = pd.Timestamp(cutoff), pd.Timestamp(test_end)
    train = fares[(fares["search_date"] <= cutoff) & (fares["flight_date"] <= cutoff)]
    curve = M.fit_booking_curve(train)
    idx_train = M.daily_index(train, curve)
    models = M.fit_series_models(idx_train)

    # realised index over the test window, normalised with the TRAIN curve
    test_obs = fares[(fares["flight_date"] > cutoff) & (fares["flight_date"] <= test_end)]
    idx_test = M.daily_index(test_obs, curve)

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
                    anchor_dtd: int = 60, sample_every_days: int = 7) -> dict:
    cutoff, test_end = pd.Timestamp(cutoff), pd.Timestamp(test_end)
    train = fares[(fares["search_date"] <= cutoff) & (fares["flight_date"] <= cutoff)]
    curve = M.fit_booking_curve(train)

    test = fares[(fares["flight_date"] > cutoff + pd.Timedelta(days=anchor_dtd)) &
                 (fares["flight_date"] <= test_end) &
                 (fares["dtd"] >= 1) & (fares["dtd"] <= anchor_dtd)]
    # weekly sample of departure dates to keep this light
    test = test[test["flight_date"].dt.dayofyear % sample_every_days == 0]

    res = []
    for (s, fd), traj in test.groupby(["series", "flight_date"]):
        traj = traj.sort_values("dtd", ascending=False)
        if traj["dtd"].max() < anchor_dtd - 3 or len(traj) < 20:
            continue
        route = traj["route"].iloc[0]
        dtds = traj["dtd"].to_numpy()
        fares_t = traj["fare"].to_numpy()
        immediate = fares_t[0]
        # recommendation from the train-fitted curve only
        cand = np.arange(1, dtds[0] + 1)
        rec_dtd = int(cand[np.argmin(curve.value(cand, route))])
        realized = fares_t[np.argmin(np.abs(dtds - rec_dtd))]
        oracle = fares_t.min()
        res.append((immediate, realized, oracle))
    if not res:
        return {"error": "no test trajectories"}
    a = np.array(res)
    imm, real, orc = a[:, 0], a[:, 1], a[:, 2]
    saving = (imm - real) / imm * 100
    possible = (imm - orc) / imm * 100
    with np.errstate(invalid="ignore", divide="ignore"):
        capture = np.where(imm - orc > 1e-9, (imm - real) / (imm - orc), np.nan)
    return {
        "anchor_dtd": anchor_dtd, "n_flights": int(len(a)),
        "avg_saving_vs_buy_now_pct": round(float(saving.mean()), 2),
        "median_saving_vs_buy_now_pct": round(float(np.median(saving)), 2),
        "avg_possible_saving_pct": round(float(possible.mean()), 2),
        "capture_ratio": round(float(np.nanmean(np.clip(capture, -1, 1))), 3),
        "pct_flights_advice_helped": round(float((saving > 0).mean() * 100), 1),
        "pct_flights_advice_hurt": round(float((saving < -0.5).mean() * 100), 1),
    }


def run(fares: pd.DataFrame, splits=None) -> dict:
    splits = splits or [("2024-12-31", "2025-12-31"), ("2025-12-31", "2026-09-01")]
    out = []
    for cutoff, test_end in splits:
        p = price_backtest(fares, cutoff, test_end)
        t = timing_backtest(fares, cutoff, test_end)
        out.append({"price": p, "timing": t})
    return {"splits": out}
