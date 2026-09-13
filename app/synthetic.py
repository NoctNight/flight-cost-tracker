"""Synthetic fare generator.

Real, free, ongoing daily route+airline fare data does not exist (see README).
Until real data is dropped into data/fares.parquet (Kaggle compactor or the
collector), this module generates a clearly-labelled synthetic dataset with
known structure so every model in the app can be exercised -- and validated,
because the ground truth (seasonality, booking curve, oil pass-through lag)
is known by construction.

Schema matches the Kaggle dilwong/flightprices compaction:
    search_date, flight_date, origin, dest, airline, fare, source
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Ground truth used by tests: oil feeds into fares with this lag (days).
TRUE_OIL_LAG_DAYS = 45
OIL_ELASTICITY = 0.30  # d log(fare) / d log(oil), applied to 30d-smoothed oil

# route: (base_fare_usd, airlines with price-level factor)
ROUTES = {
    ("JFK", "LAX"): (285.0, {"AA": 1.00, "DL": 1.04, "B6": 0.90, "UA": 1.02}),
    ("LGA", "ORD"): (180.0, {"AA": 1.00, "UA": 0.98, "DL": 1.03}),
    ("SFO", "JFK"): (295.0, {"UA": 1.00, "DL": 1.05, "B6": 0.88, "AA": 1.02}),
    ("LAX", "SFO"): (95.0,  {"WN": 0.92, "UA": 1.00, "DL": 1.02, "AS": 0.96}),
    ("ATL", "MIA"): (150.0, {"DL": 1.00, "AA": 1.03, "WN": 0.90}),
    ("BOS", "LAX"): (270.0, {"B6": 0.92, "DL": 1.00, "UA": 1.01, "AA": 1.03}),
    ("ORD", "DEN"): (140.0, {"UA": 1.00, "WN": 0.91, "AA": 1.02}),
    ("SEA", "SFO"): (120.0, {"AS": 0.94, "UA": 1.00, "DL": 1.03}),
    # Kangaroo route: long-haul one-ways, higher fuel share, peak around the
    # southern-summer/Christmas period, best booked much earlier than short-haul
    ("SYD", "LHR"): (1150.0, {"QF": 1.06, "BA": 1.03, "SQ": 1.00, "EK": 0.92,
                              "QR": 0.90, "CX": 0.96},
                     {"amp": 0.16, "peak_doy": 358, "oil_mult": 1.6, "dtd_scale": 0.55}),
    ("LHR", "SYD"): (1120.0, {"QF": 1.05, "BA": 1.04, "SQ": 1.00, "EK": 0.93,
                              "QR": 0.91, "CX": 0.97},
                     {"amp": 0.15, "peak_doy": 352, "oil_mult": 1.6, "dtd_scale": 0.55}),
}

# Every route is flyable in both directions (return-trip search needs the
# reverse leg): auto-add the reverse with the same economics.
for _od, _spec in list(ROUTES.items()):
    _rev = (_od[1], _od[0])
    if _rev not in ROUTES:
        ROUTES[_rev] = _spec

# Observation grid: days-to-departure at which each flight's fare is "seen".
DTD_GRID = np.arange(1, 91)  # daily quotes out to 90 days, like real scraped data

AIRLINE_NAMES = {
    "AA": "American", "DL": "Delta", "UA": "United", "B6": "JetBlue",
    "WN": "Southwest", "AS": "Alaska", "NK": "Spirit",
    "QF": "Qantas", "BA": "British Airways", "SQ": "Singapore Airlines",
    "EK": "Emirates", "QR": "Qatar Airways", "CX": "Cathay Pacific",
}


def booking_curve(dtd: np.ndarray) -> np.ndarray:
    """Fare multiplier vs days-to-departure (1.0 at dtd=30).

    Classic shape: mild premium far out, soft trough ~3-8 weeks out,
    steep climb inside 21 days.
    """
    dtd = np.asarray(dtd, dtype=float)
    late = 0.55 * np.exp(-dtd / 9.0)          # last-minute ramp
    early = 0.06 * (1.0 - np.exp(-(dtd - 30.0).clip(0) / 40.0))  # far-out drift
    trough = -0.04 * np.exp(-((dtd - 45.0) ** 2) / (2 * 18.0 ** 2))
    raw = 1.0 + late + early + trough
    ref = 1.0 + 0.55 * np.exp(-30.0 / 9.0) + 0.0 - 0.04 * np.exp(-(225.0) / (2 * 324.0))
    return raw / ref


def _seasonal(doy: np.ndarray, phase: float, amp: float) -> np.ndarray:
    x = 2 * np.pi * doy / 365.25
    return 1.0 + amp * np.sin(x + phase) + 0.5 * amp * np.sin(2 * x + 0.7 * phase)


def generate(
    oil: pd.Series,
    start: str = "2024-01-01",
    end: str | None = None,
    horizon_days: int = 180,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate observations. `oil` is a Date-indexed daily price series."""
    rng = np.random.default_rng(seed)
    today = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(start)

    # Oil driver: log of 30d rolling mean, lagged, forward-filled to all days.
    oil_daily = oil.reindex(pd.date_range(oil.index.min(), today, freq="D")).ffill()
    oil_smooth = np.log(oil_daily.rolling(30, min_periods=5).mean()).ffill()
    oil_lagged = oil_smooth.shift(TRUE_OIL_LAG_DAYS, freq="D")
    oil_lagged = oil_lagged.reindex(pd.date_range(start_ts - pd.Timedelta(days=120), today + pd.Timedelta(days=horizon_days), freq="D")).ffill().bfill()
    oil_ref = float(oil_lagged.loc[start_ts:today].mean())

    flight_dates = pd.date_range(start_ts, today + pd.Timedelta(days=horizon_days), freq="D")
    doy = flight_dates.dayofyear.to_numpy(dtype=float)
    dow = flight_dates.dayofweek.to_numpy()
    # travel-date effects: Fri/Sun expensive, Tue/Wed cheap
    dow_fac = np.array([0.99, 0.955, 0.95, 0.99, 1.07, 0.97, 1.06])[dow]
    oil_dev = oil_lagged.reindex(flight_dates).to_numpy() - oil_ref

    rows = []
    for r_i, ((origin, dest), spec) in enumerate(ROUTES.items()):
        base, airlines = spec[0], spec[1]
        opts = spec[2] if len(spec) > 2 else {}
        if "peak_doy" in opts:
            # anchor the annual peak to a specific day of year
            phase = np.pi / 2 - 2 * np.pi * opts["peak_doy"] / 365.25
        else:
            phase = 0.6 + 0.5 * r_i
        season = _seasonal(doy, phase=phase,
                           amp=opts.get("amp", 0.10 + 0.02 * (r_i % 3)))
        # slow structural drift per route
        t = (flight_dates - start_ts).days.to_numpy(dtype=float)
        drift = 1.0 + 0.00006 * t * (1 if r_i % 2 == 0 else -0.6)
        # route-level daily demand shock, AR(1): shared across airlines on the
        # route (this is what the correlation matrix should find)
        n = len(flight_dates)
        shock = np.empty(n)
        shock[0] = 0.0
        eps = rng.normal(0, 0.035, n)
        for i in range(1, n):
            shock[i] = 0.88 * shock[i - 1] + eps[i]
        oil_fac = np.exp(OIL_ELASTICITY * opts.get("oil_mult", 1.0) * oil_dev)
        route_level = base * season * dow_fac * drift * oil_fac * np.exp(shock)

        for airline, fac in airlines.items():
            # airline-specific AR(1) noise on top of the shared route level
            a_shock = np.empty(n)
            a_shock[0] = 0.0
            a_eps = rng.normal(0, 0.02, n)
            for i in range(1, n):
                a_shock[i] = 0.8 * a_shock[i - 1] + a_eps[i]
            level = route_level * fac * np.exp(a_shock)

            dtd_scale = opts.get("dtd_scale", 1.0)
            for dtd in DTD_GRID:
                sd = flight_dates - pd.Timedelta(days=int(dtd))
                mask = (sd >= start_ts) & (sd <= today)
                if not mask.any():
                    continue
                fare = level[mask] * booking_curve(np.array([dtd * dtd_scale]))[0]
                fare = fare * np.exp(rng.normal(0, 0.015, mask.sum()))  # obs noise
                rows.append(pd.DataFrame({
                    "search_date": sd[mask],
                    "flight_date": flight_dates[mask],
                    "origin": origin,
                    "dest": dest,
                    "airline": airline,
                    "fare": np.round(fare, 2),
                }))

    df = pd.concat(rows, ignore_index=True)
    df["source"] = "synthetic"
    return df
