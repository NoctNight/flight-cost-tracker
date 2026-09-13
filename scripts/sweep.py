"""Hyperparameter sweep with a proper train/validation/test protocol.

  train    <= 2024-12-31
  validate 2025            (sweep picks the winner here)
  test     2026            (touched exactly once, with the winning config)

Sweeps: Fourier harmonics, trend on/off, oil term on/off, booking-curve
bucket granularity, the smoothing window used in the oil-lag estimation,
and (for timing advice) the wait-verdict threshold.

Run:  python scripts/sweep.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import models as M            # noqa: E402
from app.store import get_store        # noqa: E402

CUTOFF, VAL_END, TEST_END = "2024-12-31", "2025-12-31", "2026-09-01"

BUCKETS = {
    "coarse": [0, 3, 7, 14, 30, 60, 120],
    "default": M.DTD_BUCKETS,
    "fine": list(range(0, 22)) + [24, 28, 32, 36, 40, 45, 50, 56, 63, 70, 80, 95, 120],
}


def evaluate(fares, oil, cutoff, eval_start, eval_end,
             n_harm, use_trend, oil_on, buckets_key, lag_smooth):
    cutoff, eval_start, eval_end = map(pd.Timestamp, (cutoff, eval_start, eval_end))
    train = fares[(fares["search_date"] <= cutoff) & (fares["flight_date"] <= cutoff)]
    curve = M.fit_booking_curve(train, BUCKETS[buckets_key])
    idx_train = M.daily_index(train, curve, smooth=lag_smooth)
    if oil_on:
        oa = M.oil_lag_analysis(idx_train, oil)
        oil_term = M.make_oil_term(oil, oa["best_lag_days"],
                                   idx_train["flight_date"].min(), eval_end,
                                   freeze_after=cutoff)
        oil_beta = oa["elasticity"] or 0.0
    else:
        oil_term, oil_beta = None, 0.0
    models = M.fit_series_models(idx_train, oil_term, oil_beta,
                                 n_harm=n_harm, use_trend=use_trend)
    ev = fares[(fares["flight_date"] >= eval_start) & (fares["flight_date"] <= eval_end)]
    idx_ev = M.daily_index(ev, curve)
    errs, ys = [], []
    for s, m in models.items():
        sub = idx_ev[idx_ev["series"] == s].dropna(subset=["fare"])
        if len(sub) < 30:
            continue
        y = sub["fare"].to_numpy()
        yhat, _ = m.predict(pd.DatetimeIndex(sub["flight_date"]))
        errs.append(np.abs(yhat - y) / y)
        ys.append((yhat - y) / y)
    e = np.concatenate(errs)
    return {"mape": round(float(e.mean() * 100), 2),
            "bias": round(float(np.concatenate(ys).mean() * 100), 2)}


def timing_eval(fares, cutoff, eval_end, threshold, anchor=60):
    """Follow the advice only when the curve promises a saving > threshold."""
    cutoff, eval_end = pd.Timestamp(cutoff), pd.Timestamp(eval_end)
    train = fares[(fares["search_date"] <= cutoff) & (fares["flight_date"] <= cutoff)]
    curve = M.fit_booking_curve(train)
    test = fares[(fares["flight_date"] > cutoff + pd.Timedelta(days=anchor)) &
                 (fares["flight_date"] <= eval_end) &
                 (fares["dtd"] >= 1) & (fares["dtd"] <= anchor)]
    test = test[test["flight_date"].dt.dayofyear % 7 == 0]
    savings, hurt = [], 0
    for (s, fd), traj in test.groupby(["series", "flight_date"]):
        traj = traj.sort_values("dtd", ascending=False)
        if traj["dtd"].max() < anchor - 3 or len(traj) < 20:
            continue
        route = traj["route"].iloc[0]
        dtds = traj["dtd"].to_numpy(); f = traj["fare"].to_numpy()
        cand = np.arange(1, dtds[0] + 1)
        cv = curve.value(cand, route)
        ratio = cv.min() / curve.value(np.array([dtds[0]]), route)[0]
        rec_dtd = int(cand[np.argmin(cv)]) if ratio < threshold else int(dtds[0])
        realized = f[np.argmin(np.abs(dtds - rec_dtd))]
        sv = (f[0] - realized) / f[0] * 100
        savings.append(sv); hurt += sv < -0.5
    a = np.array(savings)
    return {"threshold": threshold, "avg_saving": round(float(a.mean()), 2),
            "hurt_pct": round(float(hurt / len(a) * 100), 1), "n": len(a)}


def main():
    s = get_store()
    fares, oil = s.fares, s.oil["brent"]

    grid = list(itertools.product([1, 2, 3], [True, False], [True, False],
                                  ["coarse", "default", "fine"], [7]))
    # lag_smooth sweep only for the best-so-far shape config, to keep the grid sane
    print(f"price sweep: {len(grid)} configs on validation (2025)\n")
    rows = []
    for n_harm, trend, oil_on, bk, ls in grid:
        r = evaluate(fares, oil, CUTOFF, "2025-01-01", VAL_END, n_harm, trend, oil_on, bk, ls)
        rows.append({"harm": n_harm, "trend": trend, "oil": oil_on,
                     "buckets": bk, "lag_smooth": ls, **r})
        print(f"  harm={n_harm} trend={int(trend)} oil={int(oil_on)} buckets={bk:7s}"
              f" -> MAPE {r['mape']:6.2f}%  bias {r['bias']:+.2f}%")
    rows.sort(key=lambda x: x["mape"])
    best = rows[0]
    print(f"\nbest shape config: {best}")

    print("\nlag-smoothing sweep (winner's shape):")
    for ls in [3, 7, 14, 28]:
        r = evaluate(fares, oil, CUTOFF, "2025-01-01", VAL_END,
                     best["harm"], best["trend"], best["oil"], best["buckets"], ls)
        print(f"  smooth={ls:2d}d -> MAPE {r['mape']:6.2f}%  bias {r['bias']:+.2f}%")
        if r["mape"] < best["mape"]:
            best = {**best, "lag_smooth": ls, **r}

    print("\ntiming threshold sweep (validation):")
    tbest = None
    for th in [1.00, 0.99, 0.97, 0.95, 0.93]:
        r = timing_eval(fares, CUTOFF, VAL_END, th)
        print(f"  wait if saving > {round((1-th)*100)}% -> avg saving {r['avg_saving']}% "
              f"hurt {r['hurt_pct']}% (n={r['n']})")
        if tbest is None or r["avg_saving"] > tbest["avg_saving"]:
            tbest = r

    print(f"\n=== FINAL: winning config, scored ONCE on held-out 2026 ===")
    final = evaluate(fares, oil, "2025-12-31", "2026-01-01", TEST_END,
                     best["harm"], best["trend"], best["oil"],
                     best["buckets"], best["lag_smooth"])
    final_t = timing_eval(fares, "2025-12-31", TEST_END, tbest["threshold"])
    print(f"  price: MAPE {final['mape']}% bias {final['bias']}% "
          f"(config: harm={best['harm']} trend={best['trend']} oil={best['oil']} "
          f"buckets={best['buckets']} lag_smooth={best['lag_smooth']})")
    print(f"  timing: threshold {tbest['threshold']} -> avg saving {final_t['avg_saving']}% "
          f"hurt {final_t['hurt_pct']}%")
    json.dump({"validation_grid": rows, "best": best,
               "timing_best": tbest, "test_price": final, "test_timing": final_t},
              open(Path(__file__).resolve().parent.parent / "data" / "sweep_results.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()
