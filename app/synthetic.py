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


# ---------------------------------------------------------------- network
#
# Routes are declared by region and haul length; the economics (seasonal
# amplitude and peak, oil sensitivity, how early the booking curve bottoms
# out) are derived from those tags rather than hand-set per route, so adding
# a route is one line.
#
#   haul:  S  < 3h    intra-region hops
#          M  3-7h    medium haul
#          L  > 7h    intercontinental

AIRPORTS = {
    "EU": {"LHR": "London", "CDG": "Paris", "AMS": "Amsterdam", "FRA": "Frankfurt",
           "MAD": "Madrid", "BCN": "Barcelona", "FCO": "Rome", "MUC": "Munich",
           "ZRH": "Zurich", "DUB": "Dublin", "CPH": "Copenhagen", "IST": "Istanbul",
           "LIS": "Lisbon", "VIE": "Vienna", "ATH": "Athens", "ARN": "Stockholm"},
    "AS": {"SIN": "Singapore", "HKG": "Hong Kong", "NRT": "Tokyo Narita",
           "HND": "Tokyo Haneda", "ICN": "Seoul", "BKK": "Bangkok",
           "KUL": "Kuala Lumpur", "TPE": "Taipei", "PVG": "Shanghai",
           "DEL": "Delhi", "BOM": "Mumbai", "CGK": "Jakarta", "MNL": "Manila",
           "SGN": "Ho Chi Minh City"},
    "AU": {"SYD": "Sydney", "MEL": "Melbourne", "BNE": "Brisbane",
           "PER": "Perth", "AKL": "Auckland", "ADL": "Adelaide"},
    "NA": {"JFK": "New York JFK", "LGA": "New York LGA", "LAX": "Los Angeles",
           "SFO": "San Francisco", "ORD": "Chicago", "ATL": "Atlanta",
           "MIA": "Miami", "BOS": "Boston", "DEN": "Denver", "SEA": "Seattle"},
}
AIRPORT_NAMES = {c: n for reg in AIRPORTS.values() for c, n in reg.items()}
REGION = {c: reg for reg, ports in AIRPORTS.items() for c in ports}
REGION_NAMES = {"EU": "Europe", "AS": "Asia", "AU": "Australia / NZ",
                "NA": "North America"}

# (origin, dest, haul, base one-way fare USD, {airline: price-level factor})
NETWORK = [
    # ---- intra-Europe --------------------------------------------------
    ("LHR", "CDG", "S", 118, {"BA": 1.04, "AF": 1.00, "U2": 0.80}),
    ("LHR", "AMS", "S", 125, {"BA": 1.03, "KL": 1.00, "U2": 0.81}),
    ("LHR", "DUB", "S", 96,  {"EI": 1.00, "BA": 1.05, "FR": 0.72}),
    ("LHR", "FCO", "S", 158, {"BA": 1.02, "AZ": 1.00, "FR": 0.74}),
    ("LHR", "MAD", "S", 152, {"IB": 1.00, "BA": 1.03, "U2": 0.79}),
    ("LHR", "MUC", "S", 149, {"LH": 1.00, "BA": 1.04}),
    ("LHR", "ZRH", "S", 163, {"LX": 1.00, "BA": 1.03}),
    ("LHR", "LIS", "S", 146, {"TP": 1.00, "BA": 1.04, "FR": 0.75}),
    ("LHR", "ARN", "S", 144, {"SK": 1.00, "BA": 1.04, "FR": 0.77}),
    ("CDG", "BCN", "S", 121, {"AF": 1.00, "VY": 0.78, "IB": 1.02}),
    ("CDG", "FCO", "S", 134, {"AF": 1.00, "AZ": 1.01, "FR": 0.73}),
    ("AMS", "BCN", "S", 129, {"KL": 1.00, "VY": 0.79, "U2": 0.82}),
    ("AMS", "CPH", "S", 112, {"KL": 1.00, "SK": 1.02}),
    ("FRA", "VIE", "S", 138, {"LH": 1.00, "OS": 1.01}),
    ("MAD", "BCN", "S", 88,  {"IB": 1.00, "VY": 0.80}),
    ("FCO", "ATH", "S", 127, {"AZ": 1.00, "A3": 1.01, "FR": 0.76}),
    ("FRA", "IST", "M", 214, {"LH": 1.00, "TK": 0.96}),
    # ---- intra-Asia ----------------------------------------------------
    ("SIN", "HKG", "M", 244, {"SQ": 1.00, "CX": 1.02, "TR": 0.78}),
    ("SIN", "BKK", "M", 162, {"SQ": 1.00, "TG": 1.01, "TR": 0.76}),
    ("SIN", "KUL", "S", 88,  {"SQ": 1.00, "MH": 0.98, "AK": 0.70}),
    ("SIN", "CGK", "S", 118, {"SQ": 1.00, "GA": 0.99, "TR": 0.77}),
    ("SIN", "MNL", "M", 196, {"SQ": 1.00, "PR": 0.97, "CZ": 0.88}),
    ("SIN", "DEL", "M", 289, {"SQ": 1.00, "AI": 0.96, "TR": 0.79}),
    ("HKG", "NRT", "M", 286, {"CX": 1.00, "JL": 1.03, "NH": 1.02}),
    ("HKG", "TPE", "S", 154, {"CX": 1.00, "BR": 0.99, "CI": 0.98}),
    ("HKG", "PVG", "S", 178, {"CX": 1.00, "MU": 0.93, "CZ": 0.92}),
    ("HKG", "BKK", "M", 187, {"CX": 1.00, "TG": 1.01, "AK": 0.73}),
    ("NRT", "ICN", "S", 189, {"JL": 1.00, "KE": 0.99, "OZ": 0.98}),
    ("HND", "ICN", "S", 196, {"NH": 1.00, "KE": 0.99, "JL": 1.01}),
    ("ICN", "PVG", "S", 172, {"KE": 1.00, "MU": 0.93, "OZ": 0.99}),
    ("TPE", "NRT", "M", 232, {"BR": 1.00, "CI": 0.99, "JL": 1.03}),
    ("BKK", "SGN", "S", 108, {"TG": 1.00, "VN": 0.98, "AK": 0.72}),
    ("BOM", "DEL", "S", 92,  {"AI": 1.00, "6E": 0.84, "UK": 0.95}),
    # ---- intra-Australia / NZ ------------------------------------------
    ("SYD", "MEL", "S", 96,  {"QF": 1.00, "VA": 0.93, "JQ": 0.71}),
    ("SYD", "BNE", "S", 104, {"QF": 1.00, "VA": 0.94, "JQ": 0.72}),
    ("SYD", "ADL", "S", 118, {"QF": 1.00, "VA": 0.94, "JQ": 0.72}),
    ("SYD", "PER", "M", 268, {"QF": 1.00, "VA": 0.94, "JQ": 0.75}),
    ("MEL", "BNE", "S", 108, {"QF": 1.00, "VA": 0.93, "JQ": 0.71}),
    ("MEL", "PER", "M", 252, {"QF": 1.00, "VA": 0.95, "JQ": 0.76}),
    ("SYD", "AKL", "M", 182, {"QF": 1.00, "NZ": 1.01, "JQ": 0.74}),
    # ---- Europe <-> Asia -------------------------------------------------
    ("LHR", "SIN", "L", 742, {"SQ": 1.00, "BA": 1.02, "QR": 0.89, "EK": 0.90}),
    ("LHR", "HKG", "L", 706, {"CX": 1.00, "BA": 1.03, "QR": 0.88}),
    ("LHR", "NRT", "L", 768, {"JL": 1.00, "BA": 1.02, "NH": 1.01, "TK": 0.86}),
    ("LHR", "BKK", "L", 648, {"TG": 1.00, "BA": 1.04, "EK": 0.89, "QR": 0.88}),
    ("LHR", "DEL", "L", 552, {"AI": 1.00, "BA": 1.05, "EK": 0.88}),
    ("CDG", "HND", "L", 784, {"AF": 1.00, "JL": 1.02, "TK": 0.87}),
    ("CDG", "SIN", "L", 728, {"AF": 1.00, "SQ": 1.02, "QR": 0.88}),
    ("FRA", "SIN", "L", 736, {"LH": 1.00, "SQ": 1.02, "TK": 0.86}),
    ("FRA", "PVG", "L", 698, {"LH": 1.00, "MU": 0.91, "CZ": 0.90}),
    ("FRA", "ICN", "L", 724, {"LH": 1.00, "KE": 0.99, "TK": 0.87}),
    ("AMS", "BKK", "L", 662, {"KL": 1.00, "TG": 1.01, "EK": 0.89}),
    ("ZRH", "SIN", "L", 754, {"LX": 1.00, "SQ": 1.02, "QR": 0.89}),
    ("MUC", "HKG", "L", 716, {"LH": 1.00, "CX": 1.01, "EK": 0.90}),
    ("FCO", "SIN", "L", 706, {"AZ": 1.00, "SQ": 1.03, "QR": 0.88}),
    ("IST", "BKK", "L", 512, {"TK": 1.00, "TG": 1.02}),
    # ---- Europe <-> Australia (kangaroo) --------------------------------
    ("SYD", "LHR", "L", 1150, {"QF": 1.06, "BA": 1.03, "SQ": 1.00, "EK": 0.92,
                               "QR": 0.90, "CX": 0.96}),
    ("MEL", "LHR", "L", 1124, {"QF": 1.05, "BA": 1.03, "SQ": 1.00, "EK": 0.92,
                               "QR": 0.90}),
    ("PER", "LHR", "L", 1058, {"QF": 1.04, "SQ": 1.00, "EK": 0.92, "QR": 0.91}),
    ("SYD", "CDG", "L", 1168, {"QF": 1.04, "AF": 1.02, "SQ": 1.00, "EK": 0.92}),
    ("SYD", "FRA", "L", 1156, {"LH": 1.02, "SQ": 1.00, "QF": 1.04, "EK": 0.92}),
    # ---- Asia <-> Australia ----------------------------------------------
    ("SYD", "SIN", "L", 486, {"SQ": 1.00, "QF": 1.03, "JQ": 0.76, "TR": 0.78}),
    ("MEL", "SIN", "L", 472, {"SQ": 1.00, "QF": 1.03, "JQ": 0.75}),
    ("BNE", "SIN", "L", 494, {"SQ": 1.00, "QF": 1.03, "JQ": 0.76}),
    ("PER", "SIN", "M", 328, {"SQ": 1.00, "QF": 1.02, "JQ": 0.74, "TR": 0.77}),
    ("SYD", "HKG", "L", 512, {"CX": 1.00, "QF": 1.03, "JQ": 0.77}),
    ("MEL", "HKG", "L", 504, {"CX": 1.00, "QF": 1.03, "JQ": 0.77}),
    ("SYD", "BKK", "L", 498, {"TG": 1.00, "QF": 1.02, "JQ": 0.76}),
    ("SYD", "NRT", "L", 588, {"JL": 1.00, "QF": 1.02, "NH": 1.01, "JQ": 0.78}),
    ("SYD", "ICN", "L", 566, {"KE": 1.00, "QF": 1.02, "OZ": 0.99}),
    ("SYD", "KUL", "L", 482, {"MH": 1.00, "QF": 1.03, "AK": 0.72}),
    ("MEL", "KUL", "L", 476, {"MH": 1.00, "QF": 1.03, "AK": 0.72}),
    # ---- North America ---------------------------------------------------
    ("JFK", "LAX", "M", 285, {"AA": 1.00, "DL": 1.04, "B6": 0.90, "UA": 1.02}),
    ("LGA", "ORD", "S", 180, {"AA": 1.00, "UA": 0.98, "DL": 1.03}),
    ("SFO", "JFK", "M", 295, {"UA": 1.00, "DL": 1.05, "B6": 0.88, "AA": 1.02}),
    ("LAX", "SFO", "S", 95,  {"WN": 0.92, "UA": 1.00, "DL": 1.02, "AS": 0.96}),
    ("ATL", "MIA", "S", 150, {"DL": 1.00, "AA": 1.03, "WN": 0.90}),
    ("BOS", "LAX", "M", 270, {"B6": 0.92, "DL": 1.00, "UA": 1.01, "AA": 1.03}),
    ("ORD", "DEN", "S", 140, {"UA": 1.00, "WN": 0.91, "AA": 1.02}),
    ("SEA", "SFO", "S", 120, {"AS": 0.94, "UA": 1.00, "DL": 1.03}),
]

# haul -> seasonal amplitude, oil sensitivity, booking-curve time-scale
# (< 1 pushes the trough further from departure: long-haul is booked earlier)
HAUL_PARAMS = {
    "S": {"amp": 0.10, "oil_mult": 1.00, "dtd_scale": 1.00},
    "M": {"amp": 0.12, "oil_mult": 1.25, "dtd_scale": 0.80},
    "L": {"amp": 0.15, "oil_mult": 1.60, "dtd_scale": 0.55},
}
NORTH_PEAK_DOY = 200   # northern summer
SOUTH_PEAK_DOY = 358   # southern summer / Christmas

ROUTES = {}
for _o, _d, _haul, _base, _airlines in NETWORK:
    _p = dict(HAUL_PARAMS[_haul])
    _p["peak_doy"] = (SOUTH_PEAK_DOY if "AU" in (REGION.get(_o), REGION.get(_d))
                      else NORTH_PEAK_DOY)
    _p["haul"] = _haul
    ROUTES[(_o, _d)] = (float(_base), _airlines, _p)

# Every route is flyable in both directions (return-trip search needs the
# reverse leg): auto-add the reverse with the same economics.
for _od, _spec in list(ROUTES.items()):
    _rev = (_od[1], _od[0])
    if _rev not in ROUTES:
        ROUTES[_rev] = _spec

# Quote horizon and polling cadence. A real tracker polls near-term dates
# daily and far-out dates progressively less often; gaps stay <= 7 days so a
# search always finds a recent quote for every future flight date.
DTD_GRID = np.unique(np.r_[
    np.arange(1, 36),          # daily inside 5 weeks
    np.arange(36, 61, 2),      # every 2nd day to 60
    np.arange(63, 91, 3),      # every 3rd day to 90
    np.arange(97, 181, 7),     # weekly to 180
])

# A tracker cannot re-poll every flight at every horizon: it samples. Each
# (flight, horizon) cell is kept with the probability below, so a flight is
# quoted ~14 times over its life -- except inside RECENT_DAYS of "today",
# where the full grid is kept because that is what a live search reads.
SAMPLE_P = {35: 0.30, 90: 0.12, 181: 0.06}   # upper dtd bound -> keep prob
RECENT_DAYS = 45


def _keep_prob(dtd: int) -> float:
    for bound, p in sorted(SAMPLE_P.items()):
        if dtd <= bound:
            return p
    return min(SAMPLE_P.values())

AIRLINE_NAMES = {
    # Europe
    "BA": "British Airways", "AF": "Air France", "KL": "KLM", "LH": "Lufthansa",
    "IB": "Iberia", "AZ": "ITA Airways", "LX": "SWISS", "EI": "Aer Lingus",
    "TP": "TAP Portugal", "SK": "SAS", "OS": "Austrian", "A3": "Aegean",
    "VY": "Vueling", "U2": "easyJet", "FR": "Ryanair", "TK": "Turkish Airlines",
    # Asia
    "SQ": "Singapore Airlines", "CX": "Cathay Pacific", "JL": "Japan Airlines",
    "NH": "ANA", "KE": "Korean Air", "OZ": "Asiana", "TG": "Thai Airways",
    "MH": "Malaysia Airlines", "BR": "EVA Air", "CI": "China Airlines",
    "MU": "China Eastern", "CZ": "China Southern", "GA": "Garuda",
    "PR": "Philippine Airlines", "VN": "Vietnam Airlines", "AI": "Air India",
    "UK": "Vistara", "6E": "IndiGo", "AK": "AirAsia", "TR": "Scoot",
    # Middle East
    "EK": "Emirates", "QR": "Qatar Airways",
    # Oceania
    "QF": "Qantas", "VA": "Virgin Australia", "JQ": "Jetstar",
    "NZ": "Air New Zealand",
    # North America
    "AA": "American", "DL": "Delta", "UA": "United", "B6": "JetBlue",
    "WN": "Southwest", "AS": "Alaska", "NK": "Spirit",
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
        dtd_scale = opts.get("dtd_scale", 1.0)

        for airline, fac in airlines.items():
            # airline-specific AR(1) noise on top of the shared route level
            a_shock = np.empty(n)
            a_shock[0] = 0.0
            a_eps = rng.normal(0, 0.02, n)
            for i in range(1, n):
                a_shock[i] = 0.8 * a_shock[i - 1] + a_eps[i]
            level = route_level * fac * np.exp(a_shock)

            sd_parts, fd_parts, fare_parts = [], [], []
            for dtd in DTD_GRID:
                dtd = int(dtd)
                sd = flight_dates - pd.Timedelta(days=dtd)
                mask = np.asarray((sd >= start_ts) & (sd <= today))
                if not mask.any():
                    continue
                # poll-cadence sampling, always dense near "today"
                keep = rng.random(n) < _keep_prob(dtd)
                keep |= np.asarray(sd >= today - pd.Timedelta(days=RECENT_DAYS))
                mask = mask & keep
                if not mask.any():
                    continue
                fare = level[mask] * booking_curve(np.array([dtd * dtd_scale]))[0]
                fare = fare * np.exp(rng.normal(0, 0.015, int(mask.sum())))
                sd_parts.append(np.asarray(sd)[mask])
                fd_parts.append(np.asarray(flight_dates)[mask])
                fare_parts.append(fare.astype(np.float32))
            if not sd_parts:
                continue
            rows.append(pd.DataFrame({
                "search_date": np.concatenate(sd_parts),
                "flight_date": np.concatenate(fd_parts),
                "origin": origin,
                "dest": dest,
                "airline": airline,
                "fare": np.round(np.concatenate(fare_parts), 2),
            }))

    df = pd.concat(rows, ignore_index=True)
    df["source"] = "synthetic"
    for col in ("origin", "dest", "airline", "source"):
        df[col] = df[col].astype("category")
    return df
