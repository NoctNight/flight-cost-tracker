"""Data loading + model cache.

Priority order for fare data:
  1. data/fares.parquet  (real data: Kaggle compaction or collector output)
  2. synthetic generation (labelled as such), cached to data/fares_synthetic.parquet

Oil: data/brent-daily.csv (EIA via datasets/oil-prices, public domain).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest
from . import models as M
from . import synthetic

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_oil() -> pd.DataFrame:
    frames = {}
    for name in ("brent", "wti"):
        p = DATA_DIR / f"{name}-daily.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")
            frames[name] = df["Price"]
    if not frames:
        raise FileNotFoundError("no oil CSVs in data/ - run scripts/fetch_oil.py")
    return pd.DataFrame(frames)



def _f(v, nd=2):
    return None if pd.isna(v) else float(round(float(v), nd))

class Store:
    def __init__(self) -> None:
        self.oil = load_oil()
        self.fares, self.data_source = self._load_fares()
        self.fares = M.add_keys(self.fares)
        self.today = self.fares["search_date"].max()

        self.curve = M.fit_booking_curve(self.fares)
        self.index = M.daily_index(self.fares, self.curve, agg=M.INDEX_AGG)
        self.correlations = M.correlation_matrix(self.index, self._corr_groups())
        self.oil_analysis = M.oil_lag_analysis(self.index, self.oil["brent"])
        oil_term = M.make_oil_term(
            self.oil["brent"], self.oil_analysis["best_lag_days"],
            self.index["flight_date"].min(),
            self.today + pd.Timedelta(days=220), freeze_after=self.today)
        oil_beta = M.estimate_oil_beta(self.index, oil_term, **M.MODEL_CONFIG)
        self.oil_analysis["level_elasticity"] = round(oil_beta, 3)
        self.series_models = M.fit_series_models(
            self.index, oil_term, oil_beta, **M.MODEL_CONFIG)

    # ------------------------------------------------------------------ data
    def _load_fares(self) -> tuple[pd.DataFrame, str]:
        real = DATA_DIR / "fares.parquet"
        if real.exists():
            return pd.read_parquet(real), "real"
        cached = DATA_DIR / "fares_synthetic.parquet"
        if cached.exists():
            df = pd.read_parquet(cached)
            # regenerate if stale (> 3 days behind today)
            if (pd.Timestamp.today().normalize() - df["search_date"].max()).days <= 3:
                return df, "synthetic"
        df = synthetic.generate(self._brent_series())
        DATA_DIR.mkdir(exist_ok=True)
        df.to_parquet(cached, index=False)
        return df, "synthetic"

    def _brent_series(self) -> pd.Series:
        return self.oil["brent"]

    # ---------------------------------------------------------------- groups
    def _region_pair(self, origin: str, dest: str) -> str:
        ra = synthetic.REGION.get(origin, "??")
        rb = synthetic.REGION.get(dest, "??")
        a, b = sorted([ra, rb])
        na = synthetic.REGION_NAMES.get(a, a)
        nb = synthetic.REGION_NAMES.get(b, b)
        return f"{na} \u2194 {nb}" if a != b else f"Within {na}"

    def _corr_groups(self) -> dict:
        groups: dict[str, list[str]] = {}
        seen = set()
        for r in (self.fares[["origin", "dest"]].astype(object)
                  .drop_duplicates().itertuples()):
            key = tuple(sorted([r.origin, r.dest]))
            if key in seen:
                continue
            seen.add(key)
            groups.setdefault(self._region_pair(r.origin, r.dest), []).append(
                f"{r.origin}-{r.dest}")
        return dict(sorted(groups.items()))

    # ------------------------------------------------------------------ meta
    def meta(self) -> dict:
        combos = (self.fares[["origin", "dest", "airline"]].astype(object)
                  .drop_duplicates())
        routes = {}
        for r in combos.itertuples(index=False):
            routes.setdefault((r.origin, r.dest), []).append(r.airline)
        return {
            "data_source": self.data_source,
            "today": str(self.today.date()),
            "search_date_range": [str(self.fares["search_date"].min().date()),
                                  str(self.fares["search_date"].max().date())],
            "flight_date_max": str(self.fares["flight_date"].max().date()),
            "routes": [
                {"origin": o, "dest": d, "airlines": sorted(a),
                 "group": self._region_pair(o, d)}
                for (o, d), a in sorted(routes.items())
            ],
            "airline_names": synthetic.AIRLINE_NAMES,
            "airport_names": synthetic.AIRPORT_NAMES,
            "airport_region": synthetic.REGION,
            "region_names": synthetic.REGION_NAMES,
        }

    # -------------------------------------------------------------- forecast
    def forecast(self, origin: str, dest: str, airline: str,
                 horizon: int = 120) -> dict:
        s = M.series_key(origin, dest, airline)
        m = self.series_models.get(s)
        hist = self.index[self.index["series"] == s].dropna(subset=["fare"])
        if m is None or hist.empty:
            return {"error": f"no model for {s}"}
        h = hist[hist["flight_date"] <= self.today]
        future = pd.date_range(self.today + pd.Timedelta(days=1),
                               self.today + pd.Timedelta(days=horizon), freq="D")
        mu, sd = m.predict(future)
        fit_mu, _ = m.predict(pd.DatetimeIndex(h["flight_date"]))

        def smooth(a):
            return pd.Series(a).rolling(7, min_periods=1, center=True).mean().to_numpy()

        # display smoothing: the dow effect makes raw fit/forecast a sawtooth
        mu, sd, fit_mu = smooth(mu), smooth(sd), smooth(fit_mu)
        return {
            "series": s, "r2": _f(m.r2, 3), "n_obs": m.n,
            "history": {
                "dates": [str(d.date()) for d in h["flight_date"]],
                "fare": [_f(v) for v in h["fare"]],
                "fare_7d": [_f(v) for v in h["fare_7d"]],
                "fitted": [_f(v) for v in fit_mu],
            },
            "forecast": {
                "dates": [str(d.date()) for d in future],
                "mean": [_f(v) for v in mu],
                "lo": [_f(v) for v in (mu - 1.96 * sd)],
                "hi": [_f(v) for v in (mu + 1.96 * sd)],
            },
        }

    # ---------------------------------------------------------------- search
    def search(self, origin: str, dest: str, flight_date: str | None = None,
               window: int = 90) -> dict:
        r = self.fares[(self.fares["origin"] == origin) & (self.fares["dest"] == dest)]
        if r.empty:
            return {"error": f"no data for {origin}-{dest}"}

        # "current" = most recent search snapshot per (flight_date, airline)
        recent = r[r["search_date"] >= self.today - pd.Timedelta(days=2)]
        snap = (recent.sort_values("search_date")
                      .groupby(["flight_date", "airline"], as_index=False).last())
        if flight_date:
            fd = pd.Timestamp(flight_date)
            snap = snap[(snap["flight_date"] >= fd - pd.Timedelta(days=3)) &
                        (snap["flight_date"] <= fd + pd.Timedelta(days=3))]
        else:
            snap = snap[(snap["flight_date"] > self.today) &
                        (snap["flight_date"] <= self.today + pd.Timedelta(days=window))]
        snap = snap.sort_values("fare")

        cheapest_now = [{
            "flight_date": str(row.flight_date.date()),
            "dow": row.flight_date.strftime("%a"),
            "airline": row.airline,
            "fare": _f(row.fare),
            "dtd": int((row.flight_date - self.today).days),
        } for row in snap.head(12).itertuples()]

        # future-minimum per candidate flight: trajectory of predicted price
        # for every remaining buy day, anchored to today's observed price.
        best = []
        for row in snap.head(40).itertuples():
            traj = self._trajectory(origin, dest, row.airline,
                                    row.flight_date, row.fare)
            if traj is None:
                continue
            i_min = int(np.argmin(traj["mean"]))
            best.append({
                "flight_date": str(row.flight_date.date()),
                "dow": row.flight_date.strftime("%a"),
                "airline": row.airline,
                "fare_now": _f(row.fare),
                "predicted_min": _f(traj["mean"][i_min]),
                "best_buy_date": traj["dates"][i_min],
                "expected_saving": _f(row.fare - traj["mean"][i_min]),
                "verdict": "wait" if traj["mean"][i_min] < row.fare * 0.97 else "book now",
            })
        best.sort(key=lambda x: x["predicted_min"])
        return {
            "origin": origin, "dest": dest, "today": str(self.today.date()),
            "cheapest_now": cheapest_now,
            "cheapest_eventually": best[:12],
        }

    def quotes(self, origin: str, dest: str) -> dict:
        """Latest quote per future (flight_date, airline) plus the
        curve-implied predicted minimum and best buy date for each flight.
        Search/filtering (dates, return combos) happens client-side."""
        route = f"{origin}-{dest}"
        r = self.fares[(self.fares["origin"] == origin) & (self.fares["dest"] == dest)]
        if r.empty:
            return {"error": f"no data for {route}"}
        # 8-day lookback: the collector polls far-out horizons up to a week
        # apart, so a shorter window would miss quotes for distant flights
        recent = r[r["search_date"] >= self.today - pd.Timedelta(days=8)]
        snap = (recent.sort_values("search_date")
                      .groupby(["flight_date", "airline"], as_index=False).last())
        snap = snap[snap["flight_date"] > self.today]

        dgrid = np.arange(0, 181)
        cv = self.curve.value(dgrid, route)
        # over buy days 1..dtd: cheapest remaining curve point and its dtd
        cmin = np.minimum.accumulate(cv[1:])          # index i -> min over dtd 1..i+1
        argc = np.zeros(len(cv) - 1, dtype=int)
        best = 0
        for i in range(len(cmin)):
            if cv[1 + i] <= cv[1 + best]:
                best = i
            argc[i] = best + 1

        out = []
        for row in snap.itertuples():
            dtd = int((row.flight_date - self.today).days)
            if dtd < 1 or dtd > 180:
                continue
            ratio = float(cmin[dtd - 1] / cv[dtd])
            pm = row.fare * ratio
            bb_dtd = int(argc[dtd - 1])
            out.append({
                "flight_date": str(row.flight_date.date()),
                "dow": row.flight_date.strftime("%a"),
                "airline": row.airline,
                "fare": _f(row.fare),
                "dtd": dtd,
                "predicted_min": _f(pm),
                "best_buy_date": str((row.flight_date - pd.Timedelta(days=bb_dtd)).date()),
                "expected_saving": _f(row.fare - pm),
                "verdict": "wait" if pm < row.fare * 0.97 else "book now",
            })
        out.sort(key=lambda x: (x["flight_date"], x["airline"]))
        sigma = {a: round(self.series_models[k].sigma, 4)
                 for a in snap["airline"].unique()
                 if (k := M.series_key(origin, dest, a)) in self.series_models}
        return {"route": route, "today": str(self.today.date()),
                "curve": [_f(v, 4) for v in cv], "sigma": sigma, "quotes": out}

    def _trajectory(self, origin: str, dest: str, airline: str,
                    flight_date: pd.Timestamp, fare_now: float) -> dict | None:
        """Predicted fare for each remaining buy date, anchored to the
        currently observed fare (model supplies the *shape* via the booking
        curve; the level is calibrated to today's price)."""
        s = M.series_key(origin, dest, airline)
        if s not in self.series_models:
            return None
        route = f"{origin}-{dest}"
        buy_dates = pd.date_range(self.today, flight_date - pd.Timedelta(days=1), freq="D")
        if len(buy_dates) == 0:
            return None
        dtd = (flight_date - buy_dates).days.to_numpy()
        curve = self.curve.value(dtd, route)
        curve_now = self.curve.value(np.array([dtd[0]]), route)[0]
        mean = fare_now * curve / curve_now
        sigma = self.series_models[s].sigma
        # uncertainty grows with distance from the anchor
        grow = np.sqrt(np.arange(len(buy_dates)) / max(len(buy_dates), 1) + 1e-9)
        return {
            "dates": [str(d.date()) for d in buy_dates],
            "mean": mean.tolist(),
            "lo": (mean * np.exp(-1.28 * sigma * grow)).tolist(),
            "hi": (mean * np.exp(1.28 * sigma * grow)).tolist(),
        }

    def trajectory_api(self, origin, dest, airline, flight_date) -> dict:
        fd = pd.Timestamp(flight_date)
        r = self.fares[(self.fares["origin"] == origin) & (self.fares["dest"] == dest) &
                       (self.fares["airline"] == airline) & (self.fares["flight_date"] == fd)]
        recent = r[r["search_date"] >= self.today - pd.Timedelta(days=2)]
        if recent.empty:
            return {"error": "no current observation for that flight"}
        fare_now = float(recent.sort_values("search_date")["fare"].iloc[-1])
        # observed history of this exact flight's price so far
        hist = r.sort_values("search_date")
        traj = self._trajectory(origin, dest, airline, fd, fare_now)
        if traj is None:
            return {"error": "no model"}
        return {
            "flight_date": str(fd.date()), "airline": airline, "fare_now": _f(fare_now),
            "observed": {
                "dates": [str(d.date()) for d in hist["search_date"]],
                "fare": [_f(v) for v in hist["fare"]],
            },
            "predicted": traj,
        }

    # ----------------------------------------------------------------- oil api
    def oil_api(self) -> dict:
        a = self.oil_analysis
        days = pd.date_range(self.index["flight_date"].min(), self.today, freq="D")
        brent = self.oil["brent"].reindex(days).ffill()
        piv = self.index.pivot_table(index="flight_date", columns="series", values="fare_7d")
        fare_avg = piv.mean(axis=1).reindex(days)
        b0 = brent.dropna().iloc[0]
        f0 = fare_avg.dropna().iloc[0]
        return {
            **a,
            "dates": [str(d.date()) for d in days],
            "brent_indexed": [_f(100 * v / b0) for v in brent],
            "fare_indexed": [_f(100 * v / f0) for v in fare_avg],
        }

    def backtest_api(self) -> dict:
        if not hasattr(self, "_backtest"):
            r = backtest.run(self.fares, self.oil["brent"])
            for sp in r["splits"]:  # honest R2: median across series, not pooled
                med = float(np.median([x["r2_oos"] for x in sp["price"]["per_series"]]))
                sp["price"]["r2_oos_median_series"] = round(med, 3)
            self._backtest = r
        return self._backtest

    def booking_curve_api(self, origin: str | None = None, dest: str | None = None) -> dict:
        route = f"{origin}-{dest}" if origin and dest else None
        dtd = np.arange(0, 120)
        return {
            "route": route if route in self.curve.per_route else "all routes",
            "dtd": dtd.tolist(),
            "multiplier": [_f(v, 4) for v in self.curve.value(dtd, route)],
        }


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
