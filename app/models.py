"""Statistical models.

All models operate on the observation table produced by app.store:
    search_date, flight_date, origin, dest, airline, fare

Series key: "ORIGIN-DEST:AIRLINE" (a route flown by one airline).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REF_DTD_LO, REF_DTD_HI = 21, 45  # reference booking window for the daily index

# Forecast-model configuration. Chosen on the validation split (2025) by
# scripts/sweep.py -- see data/sweep_results.json; test (2026) scored once.
MODEL_CONFIG = {
    "n_harm": 2,          # annual Fourier harmonics
    "use_trend": False,   # linear trend extrapolates noise on short windows
    "half_life_days": 730,  # recency weighting: a 2-year-old obs counts half
    "pool_by_route": True,  # one seasonal fit per route, per-series offsets
}
INDEX_AGG = "mean"        # daily index aggregation across a day's quotes
DTD_BUCKETS = [0, 2, 4, 6, 9, 13, 18, 24, 31, 40, 52, 67, 83, 105, 135, 181]


def series_key(origin: str, dest: str, airline: str) -> str:
    return f"{origin}-{dest}:{airline}"


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Attach route / series keys and days-to-departure.

    origin/dest/airline arrive as categoricals (the table is ~500 series over
    millions of rows), so build the keys once per category combination and
    map, rather than concatenating strings row by row."""
    df = df.copy()
    cols = ["origin", "dest", "airline"]
    for c in cols:
        if not isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype("category")
    codes = df[cols].apply(lambda s: s.cat.codes)
    combo = pd.MultiIndex.from_frame(df[cols].astype(object))
    uniq = combo.unique()
    route_map = {k: f"{k[0]}-{k[1]}" for k in uniq}
    series_map = {k: f"{k[0]}-{k[1]}:{k[2]}" for k in uniq}
    df["route"] = pd.Categorical([route_map[k] for k in combo])
    df["series"] = pd.Categorical([series_map[k] for k in combo])
    df["dtd"] = (df["flight_date"] - df["search_date"]).dt.days.astype("int16")
    del codes
    return df


# ---------------------------------------------------------------- booking curve

@dataclass
class BookingCurve:
    """log-fare effect of days-to-departure, estimated with flight fixed
    effects: demean log fare within each (series, flight_date), then average
    the demeaned values per dtd bucket. Multiplier normalised to 1.0 at the
    bucket containing dtd=30."""
    bucket_edges: list
    bucket_mid: np.ndarray
    multiplier: np.ndarray          # per bucket, per-route dict may override
    per_route: dict = field(default_factory=dict)

    def value(self, dtd, route: str | None = None) -> np.ndarray:
        dtd = np.asarray(dtd, dtype=float).clip(0, 180)
        mult = self.per_route.get(route, self.multiplier)
        # piecewise-linear interpolation between bucket midpoints
        return np.interp(dtd, self.bucket_mid, mult)


def fit_booking_curve(df: pd.DataFrame, buckets=None) -> BookingCurve:
    buckets = buckets or DTD_BUCKETS
    d = df[(df["dtd"] >= 0) & (df["dtd"] <= 180)].copy()
    d["logf"] = np.log(d["fare"])
    d["logf_dm"] = d["logf"] - d.groupby(["series", "flight_date"])["logf"].transform("mean")
    d["bucket"] = np.digitize(d["dtd"], buckets[1:-1])

    mids = []
    for i in range(len(buckets) - 1):
        mids.append(0.5 * (buckets[i] + buckets[i + 1]))
    mids = np.array(mids)

    def curve_for(sub: pd.DataFrame) -> np.ndarray:
        eff = sub.groupby("bucket")["logf_dm"].mean()
        eff = eff.reindex(range(len(mids)))
        eff = eff.interpolate(limit_direction="both").to_numpy()
        ref_bucket = int(np.digitize(30, buckets[1:-1]))
        return np.exp(eff - eff[ref_bucket])

    global_curve = curve_for(d)
    per_route = {}
    for route, sub in d.groupby("route"):
        if sub["flight_date"].nunique() >= 60:
            per_route[route] = curve_for(sub)
    return BookingCurve(buckets, mids, global_curve, per_route)


# ------------------------------------------------------------- daily fare index

def daily_index(df: pd.DataFrame, curve: BookingCurve, smooth: int = 7,
                agg: str = "median") -> pd.DataFrame:
    """One row per (series, flight_date): booking-curve-normalised median fare.
    This is the 1-day-granularity series everything downstream uses."""
    d = df[(df["dtd"] >= 0) & (df["dtd"] <= 180)].copy()
    d["norm_fare"] = d["fare"] / curve.value(d["dtd"].to_numpy(), None)
    idx = (d.groupby(["series", "route", "airline", "flight_date"])
             ["norm_fare"].agg(agg).rename("fare").reset_index())
    idx = idx.sort_values(["series", "flight_date"])
    idx["fare_7d"] = (idx.groupby("series")["fare"]
                        .transform(lambda s: s.rolling(smooth, min_periods=max(2, smooth // 2),
                                                       center=True).mean()))
    return idx


# --------------------------------------------------------- per-series regression

def make_oil_term(oil: pd.Series, lag_days: int, start: pd.Timestamp,
                  end: pd.Timestamp, freeze_after: pd.Timestamp,
                  smooth: int = 30) -> pd.Series:
    """log(30d-smoothed oil) shifted by the pass-through lag, for use as a
    regressor indexed by flight_date. Oil after `freeze_after` is unknown at
    prediction time, so it is held at its last observed value -- no lookahead."""
    days = pd.date_range(start - pd.Timedelta(days=lag_days + 120), end, freq="D")
    o = oil.reindex(days).ffill()
    o = o.where(o.index <= freeze_after).ffill()
    sm = np.log(o.rolling(smooth, min_periods=max(3, smooth // 6)).mean())
    term = sm.shift(lag_days, freq="D").reindex(pd.date_range(start, end, freq="D"))
    return term.ffill().bfill()


@dataclass
class SeriesModel:
    series: str
    beta: np.ndarray
    sigma: float                 # residual std of log fare
    r2: float
    n: int
    t0: pd.Timestamp
    dow_means: np.ndarray
    oil_term: pd.Series | None = None
    oil_mean: float = 0.0
    oil_beta: float = 0.0        # fixed pass-through elasticity, not free-fitted
    n_harm: int = 2              # annual Fourier harmonics
    use_trend: bool = True

    def _oil(self, dates: pd.DatetimeIndex) -> np.ndarray:
        if self.oil_term is None or self.oil_beta == 0.0:
            return np.zeros(len(dates))
        return self.oil_beta * (self.oil_term.reindex(dates).ffill().bfill().to_numpy()
                                - self.oil_mean)

    def design(self, dates: pd.DatetimeIndex) -> np.ndarray:
        t = (dates - self.t0).days.to_numpy(dtype=float) / 365.25
        doy = dates.dayofyear.to_numpy(dtype=float)
        x = 2 * np.pi * doy / 365.25
        dow = dates.dayofweek.to_numpy()
        dow_dum = np.zeros((len(dates), 6))
        for j in range(6):
            dow_dum[:, j] = (dow == j + 1).astype(float)
        cols = [np.ones(len(dates))]
        if self.use_trend:
            cols.append(t)
        for k in range(1, self.n_harm + 1):
            cols += [np.sin(k * x), np.cos(k * x)]
        X = np.column_stack(cols + [dow_dum])
        return X

    def predict(self, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        X = self.design(dates)
        mu = X @ self.beta + self._oil(dates)
        return np.exp(mu), np.exp(mu) * self.sigma  # approx sd on level scale


def estimate_oil_beta(idx: pd.DataFrame, oil_term: pd.Series,
                      n_harm: int = 2, use_trend: bool = False,
                      half_life_days: float | None = None,
                      **_ignored) -> float:
    """Level elasticity of fares to (lagged, smoothed) oil, estimated JOINTLY
    with per-route seasonality via Frisch-Waugh-Lovell: residualise both log
    fare and the oil term on each route's seasonal design, then pool.

    Identification comes from seasonal phases differing across routes while
    the oil path is common. (The change-on-change slope in oil_lag_analysis
    finds the LAG well but attenuates the magnitude; a naive residual
    regression is also biased low because within one route a year of oil is
    collinear with seasonality. FWL is the exact joint-OLS answer.)
    """
    o_mean = float(oil_term.mean())
    num = den = 0.0
    ridx = (idx.dropna(subset=["fare"])
               .groupby(["route", "flight_date"], as_index=False)["fare"].mean())
    for r, sub in ridx.groupby("route"):
        if len(sub) < 90:
            continue
        dates = pd.DatetimeIndex(sub["flight_date"])
        proto = SeriesModel(r, None, None, None, len(sub), dates.min(), None,
                            None, 0.0, 0.0, n_harm, use_trend)
        X = proto.design(dates)
        y = np.log(sub["fare"].to_numpy())
        x = oil_term.reindex(dates).ffill().bfill().to_numpy() - o_mean
        if half_life_days:
            w = 0.5 ** ((dates.max() - dates).days.to_numpy(dtype=float) / half_life_days)
        else:
            w = np.ones(len(y))
        ry = y - X @ _wls(X, y, w)
        rx = x - X @ _wls(X, x, w)
        num += float(np.sum(w * rx * ry))
        den += float(np.sum(w * rx * rx))
    return num / den if den > 0 else 0.0


def _wls(X, y, w, ridge=0.0):
    """(weighted) least squares with an optional ridge penalty on all
    non-intercept coefficients."""
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    if ridge > 0:
        P = np.eye(X.shape[1]) * ridge
        P[0, 0] = 0.0
        return np.linalg.solve(Xw.T @ Xw + P, Xw.T @ yw)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return beta


def fit_series_models(idx: pd.DataFrame,
                      oil_term: pd.Series | None = None,
                      oil_beta: float = 0.0,
                      n_harm: int = 2,
                      use_trend: bool = True,
                      half_life_days: float | None = None,
                      ridge: float = 0.0,
                      pool_by_route: bool = False) -> dict[str, SeriesModel]:
    """The oil coefficient is FIXED to the pooled elasticity from the lag
    analysis (free-fitting it per series is collinear with trend+seasonality
    on short training windows and overfits): the oil effect is subtracted
    from log fares before the OLS and added back at prediction."""
    models = {}
    t0 = idx["flight_date"].min()
    oil_mean = float(oil_term.mean()) if oil_term is not None else 0.0

    def fit_one(key, sub):
        dates = pd.DatetimeIndex(sub["flight_date"])
        m = SeriesModel(key, None, None, None, len(sub), t0, None,
                        oil_term, oil_mean, oil_beta, n_harm, use_trend)
        y = np.log(sub["fare"].to_numpy()) - m._oil(dates)
        X = m.design(dates)
        if half_life_days:
            # recency weighting: an observation half_life_days old counts half
            age = (dates.max() - dates).days.to_numpy(dtype=float)
            w = 0.5 ** (age / half_life_days)
        else:
            w = np.ones(len(y))
        beta = _wls(X, y, w, ridge)
        resid = y - X @ beta
        m.sigma = float(np.sqrt(np.sum(w * resid ** 2) / w.sum()))
        ybar = np.sum(w * y) / w.sum()
        ss_tot = np.sum(w * (y - ybar) ** 2)
        m.r2 = float(1 - np.sum(w * resid ** 2) / ss_tot) if ss_tot > 0 else 0.0
        m.beta = beta
        return m

    if pool_by_route:
        # one seasonal/dow/oil fit per ROUTE (airlines on a route share the
        # demand pattern; per-series fits waste data re-estimating it), then a
        # per-series price-level offset on the intercept.
        route_models = {}
        for r, sub in idx.dropna(subset=["fare"]).groupby("route"):
            agg = sub.groupby("flight_date", as_index=False)["fare"].mean()
            agg["flight_date"] = pd.to_datetime(agg["flight_date"])
            if len(agg) < 90:
                continue
            route_models[r] = fit_one(r, agg)
        for s, sub in idx.dropna(subset=["fare"]).groupby("series"):
            r = sub["route"].iloc[0]
            if r not in route_models or len(sub) < 90:
                continue
            rm = route_models[r]
            dates = pd.DatetimeIndex(sub["flight_date"])
            y = np.log(sub["fare"].to_numpy())
            pred = np.log(rm.predict(dates)[0])
            offset = float(np.mean(y - pred))
            import copy
            m = copy.copy(rm)
            m.series = s
            m.beta = rm.beta.copy()
            m.beta[0] += offset
            m.sigma = float(np.std(y - pred - offset))
            m.n = len(sub)
            models[s] = m
        return models

    for s, sub in idx.groupby("series"):
        sub = sub.dropna(subset=["fare"])
        if len(sub) < 90:
            continue
        models[s] = fit_one(s, sub)
    return models


# ------------------------------------------------------------------ correlations

def correlation_matrix(idx: pd.DataFrame, groups: dict | None = None,
                       min_overlap: int = 120, max_series: int = 26) -> dict:
    """Correlation between series' weekly-smoothed log fare *changes*
    (levels correlate trivially through shared seasonality; changes measure
    co-movement).

    Summary statistics are computed over every pair. The matrices returned
    for display are scoped to `groups` (name -> list of routes) and capped at
    `max_series` per group, because a 500x500 heatmap is unreadable.
    """
    piv = idx.pivot_table(index="flight_date", columns="series",
                          values="fare_7d", observed=True)
    logret = np.log(piv).diff(7)
    corr = logret.corr(min_periods=min_overlap)
    labels = list(corr.columns)
    C = corr.to_numpy()

    def parts(s):
        route, airline = s.split(":")
        return route, airline

    meta = [parts(s) for s in labels]
    same_route, same_airline, cross = [], [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            v = C[i, j]
            if not np.isfinite(v):
                continue
            if meta[i][0] == meta[j][0]:
                same_route.append(v)
            elif meta[i][1] == meta[j][1]:
                same_airline.append(v)
            else:
                cross.append(v)

    # per-group display matrices
    counts = idx.groupby("series", observed=True)["fare"].count()
    out_groups = {}
    for name, routes in (groups or {}).items():
        rset = set(routes)
        members = [s for s in labels if s.split(":")[0] in rset]
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda s: -counts.get(s, 0))[:max_series]
        members.sort()
        sub = corr.loc[members, members].to_numpy()
        out_groups[name] = {
            "labels": members,
            "matrix": [[None if not np.isfinite(v) else round(float(v), 3)
                        for v in row] for row in sub],
        }

    return {
        "groups": out_groups,
        "n_series": len(labels),
        "n_pairs": len(same_route) + len(same_airline) + len(cross),
        "avg_same_route": round(float(np.mean(same_route)), 3) if same_route else None,
        "avg_same_airline": round(float(np.mean(same_airline)), 3) if same_airline else None,
        "avg_unrelated": round(float(np.mean(cross)), 3) if cross else None,
    }


# ------------------------------------------------------------------ oil lag model

def oil_lag_analysis(idx: pd.DataFrame, oil: pd.Series,
                     max_lag: int = 90) -> dict:
    """Find the delay with which oil price changes show up in fares.

    Method: cross-correlation between 28-day log changes of (30d-smoothed)
    oil and 28-day log changes of each series' weekly-smoothed fare index,
    at lags 0..max_lag. The per-series best lag distribution gives the
    'average time'; a pooled lagged regression quantifies pass-through.
    """
    days = pd.date_range(idx["flight_date"].min() - pd.Timedelta(days=max_lag + 40),
                         idx["flight_date"].max(), freq="D")
    o = oil.reindex(days).ffill()
    o_sm = np.log(o.rolling(30, min_periods=5).mean())
    o_chg = o_sm.diff(28)

    piv = idx.pivot_table(index="flight_date", columns="series", values="fare_7d")
    f_chg = np.log(piv).diff(28)

    lags = np.arange(0, max_lag + 1)
    mean_corr, per_series_best = [], {}
    corr_by_lag_series = {}
    for lag in lags:
        shifted = o_chg.shift(lag, freq="D").reindex(f_chg.index)
        cors = f_chg.corrwith(shifted)
        corr_by_lag_series[lag] = cors
        mean_corr.append(float(cors.mean()))
    mean_corr = np.array(mean_corr)
    best_lag = int(lags[np.nanargmax(mean_corr)])
    for s in piv.columns:
        cs = np.array([corr_by_lag_series[l].get(s, np.nan) for l in lags])
        if np.isfinite(cs).any():
            per_series_best[s] = int(lags[np.nanargmax(cs)])
    avg_series_lag = float(np.mean(list(per_series_best.values()))) if per_series_best else None

    # pooled regression: fare index changes ~ oil changes at best lag
    shifted = o_chg.shift(best_lag, freq="D").reindex(f_chg.index)
    xy = pd.concat([shifted.rename("oil"), f_chg.mean(axis=1).rename("fare")], axis=1).dropna()
    if len(xy) > 30:
        X = np.column_stack([np.ones(len(xy)), xy["oil"].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, xy["fare"].to_numpy(), rcond=None)
        resid = xy["fare"].to_numpy() - X @ beta
        ss_tot = ((xy["fare"] - xy["fare"].mean()) ** 2).sum()
        r2 = float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0
        elasticity = float(beta[1])
    else:
        elasticity, r2 = None, None

    return {
        "lags": lags.tolist(),
        "mean_correlation": [None if not np.isfinite(v) else round(float(v), 4) for v in mean_corr],
        "best_lag_days": best_lag,
        "avg_series_lag_days": round(avg_series_lag, 1) if avg_series_lag is not None else None,
        "per_series_lag": per_series_best,
        "elasticity": round(elasticity, 3) if elasticity is not None else None,
        "r2": round(r2, 3) if r2 is not None else None,
    }
