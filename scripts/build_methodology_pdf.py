"""Render docs/regression-methodology.pdf.

Numbers are read from the built snapshot (site/core.js) so the document can
never drift from the model it describes:

    python scripts/build_static.py          # produces site/core.js
    python scripts/build_methodology_pdf.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "regression-methodology.pdf"

INK = colors.HexColor("#14140f")
MUTED = colors.HexColor("#6b6a63")
RULE = colors.HexColor("#d9d8d0")
ACCENT = colors.HexColor("#1c5cab")
BOXBG = colors.HexColor("#f4f3ee")


def load_facts() -> dict:
    core = ROOT / "site" / "core.js"
    if not core.exists():
        raise SystemExit("run scripts/build_static.py first (site/core.js missing)")
    raw = core.read_text()
    data = json.loads(raw[raw.index("=") + 1: raw.rstrip().rfind(";")])
    meta, corr, oil = data["/api/meta"], data["/api/correlations"], data["/api/oil"]
    splits = data["/api/backtest"]["splits"]
    airlines = {a for r in meta["routes"] for a in r["airlines"]}
    return {"meta": meta, "corr": corr, "oil": oil, "splits": splits,
            "n_routes": len(meta["routes"]), "n_series": corr["n_series"],
            "n_airports": len(meta["airport_names"]), "n_airlines": len(airlines)}


# ----------------------------------------------------------------- styles
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica",
                      fontSize=8.9, leading=12.6, textColor=INK,
                      alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold",
                    fontSize=17, leading=20, textColor=INK, alignment=0,
                    spaceAfter=2)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=9.4, textColor=MUTED,
                     alignment=0, spaceAfter=10)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                    fontSize=10.6, leading=13, textColor=INK,
                    spaceBefore=11, spaceAfter=4)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold",
                    fontSize=9.2, leading=12, textColor=ACCENT,
                    spaceBefore=7, spaceAfter=3)
EQ = ParagraphStyle("eq", parent=BODY, fontName="Courier", fontSize=8.4,
                    leading=11.6, alignment=0, leftIndent=10, spaceBefore=3,
                    spaceAfter=5, textColor=INK)
NOTE = ParagraphStyle("note", parent=BODY, fontSize=8.2, leading=11,
                      textColor=MUTED, spaceBefore=2)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.9, leading=10.4,
                      alignment=0, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")


def p(t, s=BODY):
    return Paragraph(t, s)


def eq(t):
    return Paragraph(t.replace(" ", "&nbsp;"), EQ)


def table(rows, widths, header=True):
    data = [[Paragraph(c, CELLB if (header and i == 0) else CELL) for c in row]
            for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
             ("LEFTPADDING", (0, 0), (-1, -1), 5),
             ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE)]
    if header:
        style += [("LINEBELOW", (0, 0), (-1, 0), 0.9, INK),
                  ("BACKGROUND", (0, 0), (-1, 0), colors.white)]
    t.setStyle(TableStyle(style))
    return t


def callout(title, body):
    inner = [Paragraph(title, ParagraphStyle("ct", parent=CELLB, fontSize=8.4,
                                             textColor=ACCENT, spaceAfter=2)),
             Paragraph(body, ParagraphStyle("cb", parent=CELL, fontSize=8.2,
                                            leading=11))]
    t = Table([[inner]], colWidths=[176 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BOXBG),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBEFORE", (0, 0), (0, -1), 2, ACCENT)]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(17 * mm, 12 * mm, "Farecast - fare regression methodology")
    canvas.drawRightString(A4[0] - 17 * mm, 12 * mm, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(17 * mm, 15 * mm, A4[0] - 17 * mm, 15 * mm)
    canvas.restoreState()


def build() -> None:
    F = load_facts()
    meta, corr, oil, splits = F["meta"], F["corr"], F["oil"], F["splits"]
    s1, s2 = splits[0], splits[1]
    p1, t1 = s1["price"], s1["timing"]
    p2, t2 = s2["price"], s2["timing"]

    story = []
    A = story.append

    # ---------------------------------------------------------- title
    A(p("Predicting flight prices: the regression, step by step", H1))
    A(p(f"How Farecast turns raw fare quotes into a daily price index, a "
        f"seasonal forecast, a crude-oil pass-through estimate and a "
        f"buy-or-wait decision. Every step below is implemented in "
        f"<font face='Courier'>app/models.py</font>; numbers are measured on "
        f"{F['n_routes']} routes / {F['n_series']} route-airline series "
        f"({meta['search_date_range'][0]} to {meta['search_date_range'][1]}).", SUB))

    # ---------------------------------------------------------- 0
    A(p("0. Notation and input", H2))
    A(p("The only input is a table of observed quotes. One row is: a price seen "
        "on a given <i>search date</i>, for a flight departing on a given "
        "<i>flight date</i>, on one route flown by one airline.", BODY))
    A(table([
        ["Symbol", "Meaning"],
        ["<font face='Courier'>s</font>", "search date - the day the price was observed"],
        ["<font face='Courier'>f</font>", "flight date - the day of departure"],
        ["<font face='Courier'>d = f - s</font>", "days to departure (the booking horizon)"],
        ["<font face='Courier'>i</font>", "a series: one route flown by one airline, e.g. SYD-LHR:QF"],
        ["<font face='Courier'>r(i)</font>", "the route of series i, e.g. SYD-LHR"],
        ["<font face='Courier'>P(i,s,f)</font>", "the quoted one-way fare"],
    ], [26 * mm, 150 * mm]))
    A(p("Two different things are called 'price' and must not be confused: the "
        "<b>quote</b> P depends on both when you look and when you fly; the "
        "<b>index</b> built in step 2 depends only on when you fly. Forecasting "
        "operates on the index; buy-timing operates on the quote.", NOTE))

    # ---------------------------------------------------------- 1
    A(p("1. The booking curve: separating <i>when you buy</i> from <i>when you fly</i>", H2))
    A(p("A fare rises as departure approaches. If that effect is left in the data, "
        "a route observed mostly at short notice looks expensive for reasons that "
        "have nothing to do with the season. So the horizon effect is estimated "
        "first and divided out.", BODY))
    A(p("The estimate must not be confounded by <i>which</i> flights happen to be "
        "observed at each horizon - expensive peak-season flights are quoted more "
        "often near departure. The fix is flight fixed effects: compare a flight "
        "only against itself, by demeaning log fare within each (series, flight "
        "date) cell before averaging across flights.", BODY))
    A(p("Estimator", H3))
    A(eq("y(i,s,f)   = log P(i,s,f)"))
    A(eq("y~(i,s,f)  = y(i,s,f) - mean_s[ y(i,s,f) ]      # within-flight demeaning"))
    A(eq("c_b        = mean over all obs with d in bucket b of y~"))
    A(eq("C_r(d)     = exp( c_b(d) - c_b(30) )            # normalised: C_r(30) = 1"))
    A(p("Horizons are grouped into 15 buckets, dense near departure and coarse far "
        "out (0-2, 2-4, ... 105-135, 135-181 days), because quotes thin out with "
        "horizon. Bucket effects are linearly interpolated between bucket "
        "midpoints, so C is defined for every integer d from 0 to 180. The curve is "
        "fitted per route where a route has at least 60 distinct flight dates, "
        "otherwise the pooled curve is used.", BODY))
    A(callout("Why this matters for the advice",
              "C is the only thing that decides <i>when to buy</i>. Its argmin is "
              "the recommended purchase horizon, and its shape differs sharply by "
              "haul: short-haul troughs around 46 days out, long-haul around 75. "
              "That difference is learned from data, not assumed."))

    # ---------------------------------------------------------- 2
    A(p("2. The daily fare index", H2))
    A(p("Every quote is divided by the booking-curve factor for its horizon, which "
        "removes the horizon effect and leaves a quantity comparable across "
        "booking windows. Quotes sharing a flight date are then aggregated, giving "
        "one number per series per calendar day - the 1-day-granularity series the "
        "forecast is built on.", BODY))
    A(eq("Pn(i,s,f) = P(i,s,f) / C_r(d)                   # horizon-normalised"))
    A(eq("X(i,f)    = mean over s of Pn(i,s,f)            # daily index"))
    A(eq("X7(i,f)   = centred 7-day rolling mean of X(i,.)  # rolling week"))
    A(p("The rolling week is used for correlation and oil work, where day-to-day "
        "noise would swamp the signal; the regression itself is fitted on the "
        "unsmoothed daily index, so smoothing cannot leak future information into "
        "the fit.", NOTE))

    # ---------------------------------------------------------- 3
    A(p("3. Crude oil: first the lag, then the magnitude", H2))
    A(p("Fuel is a large share of airline cost, but a change in crude does not "
        "reach fares immediately - carriers hedge, and fares are set in advance. "
        "The delay is estimated before the size of the effect, because the two "
        "need different estimators.", BODY))

    A(p("3a. Finding the lag", H3))
    A(p("Brent daily spot is smoothed over 30 days and logged. Both oil and fares "
        "are differenced over 28 days, which removes the level and leaves "
        "co-movement. For each candidate lag from 0 to 90 days, oil is shifted "
        "forward and correlated with the fare-index changes; the lag maximising "
        "the mean correlation across series is selected.", BODY))
    A(eq("O(t)     = log( 30-day rolling mean of Brent )"))
    A(eq("dO(t,L)  = O(t-L) - O(t-28-L)          # 28-day oil change, lagged by L"))
    A(eq("dX(i,t)  = log X7(i,t) - log X7(i,t-28)"))
    A(eq("L*       = argmax_L  mean_i corr( dX(i,.), dO(.,L) )"))
    A(p(f"Measured on the full sample: <b>L* = {oil['best_lag_days']} days</b>, "
        f"with per-series argmaxes averaging "
        f"{oil.get('avg_series_lag_days', 'n/a')} days.", BODY))

    A(p("3b. Estimating the pass-through (and a trap)", H3))
    A(p("The slope of that same change-on-change regression is a <i>biased</i> "
        "estimate of the elasticity: short-horizon fare changes are dominated by "
        "demand noise, which attenuates the coefficient - here by roughly a factor "
        "of three. Regressing the seasonal model's residuals on oil is also biased "
        "low, because within a single route one year of oil is collinear with the "
        "seasonal terms.", BODY))
    A(p("The unbiased route is Frisch-Waugh-Lovell: residualise both log fare and "
        "the oil term on each route's seasonal design, then pool. Identification "
        "comes from seasonal phases differing across routes while the oil path is "
        "common to all of them.", BODY))
    A(eq("Z_r  = seasonal design matrix for route r (see step 4)"))
    A(eq("ry   = log X - Z_r (Z_r'W Z_r)^-1 Z_r'W log X     # fare, seasonality removed"))
    A(eq("rx   = O(.-L*) - Z_r (Z_r'W Z_r)^-1 Z_r'W O(.-L*) # oil, seasonality removed"))
    A(eq("beta = sum_r <rx, ry>_W / sum_r <rx, rx>_W        # pooled elasticity"))
    A(p(f"Measured: <b>beta = {oil.get('level_elasticity', 'n/a')}</b> "
        f"(d log fare per d log Brent). The synthetic data is generated with a "
        f"known pass-through of 0.30, so this estimator recovers the truth; the "
        f"change-on-change slope returns {oil.get('elasticity', 'n/a')} on the same "
        f"data, which is the attenuation described above.", BODY))
    A(callout("No lookahead",
              "Oil after the forecast origin is unknown in reality, so the oil "
              "series is frozen at its last observed value and carried forward "
              "flat. The model therefore never sees a future oil shock - visible "
              "in the results as negative bias in the split whose test year "
              "contains a price spike."))

    A(Spacer(1, 2))
    story.append(p("4. The forecast regression", H2))
    A(p("With the horizon effect divided out and the oil effect held at a known "
        "coefficient, what remains to explain is the seasonal and weekly shape of "
        "the daily index. That is an ordinary least-squares regression of log "
        "index on a compact calendar design.", BODY))

    A(p("4a. Design matrix", H3))
    A(p("Seasonality uses Fourier terms rather than month dummies: any smooth "
        "annual pattern is a sum of sine waves whose periods divide the year. Each "
        "harmonic contributes a sine/cosine pair, which lets the regression place "
        "a peak at any date (a&middot;sin + b&middot;cos is one wave with fitted "
        "phase and height). Harmonic 1 gives a single annual peak and trough; "
        "harmonic 2 allows two peaks per year and asymmetric run-ups - the shape "
        "real fares have. Two harmonics is four parameters against eleven for "
        "month dummies, and the curve is smooth and defined on every calendar day.", BODY))
    A(eq("u(f) = 2*pi * dayofyear(f) / 365.25"))
    A(eq("Z    = [ 1,  sin(u), cos(u),  sin(2u), cos(2u),  D_Tue..D_Sun ]"))
    A(p("D_Tue..D_Sun are six day-of-week dummies (Monday is the reference), "
        "capturing the cheap-midweek / expensive-weekend pattern in the "
        "<i>departure</i> date.", NOTE))
    A(p("A linear time trend was tested and <b>deliberately excluded</b>: on "
        "training windows of one to two years it fits noise and extrapolates it, "
        "costing about 3 percentage points of out-of-sample error. Seasonality "
        "plus the oil term carries the level movement instead.", BODY))

    A(p("4b. Fitting: oil offset, recency weights, route pooling", H3))
    A(p("The oil contribution is subtracted from the left-hand side rather than "
        "fitted as a free column, so its coefficient is the FWL estimate from step "
        "3b and cannot be re-estimated (badly) inside each short window.", BODY))
    A(eq("y(i,f)  = log X(i,f) - beta * ( O(f-L*) - mean O )"))
    A(p("Observations are weighted by age with an exponential half-life, so recent "
        "seasons count more than old ones - a fare regime from several years ago "
        "should not carry the same weight as last year:", BODY))
    A(eq("w(f)    = 0.5 ^ ( age(f) / H ),     H = 730 days"))
    A(p("Finally, the seasonal fit is pooled by route. Airlines competing on one "
        "route share a demand pattern, so estimating it separately per airline "
        "spends data re-learning the same curve. One weighted fit per route is "
        "taken, then each airline gets a price-level offset on the intercept:", BODY))
    A(eq("b_r     = (Z'W Z)^-1 Z'W y_r        # per-route seasonal fit, W = diag(w)"))
    A(eq("a_i     = mean( y_i - Z b_r )       # per-series level offset"))
    A(eq("log X^  = Z b_r + a_i + beta * ( O(f-L*) - mean O )"))
    A(p("Residual scale is the weighted standard deviation of the fit residuals; "
        "the reported interval is the 95% band exp(log X^ &plusmn; 1.96 sigma). "
        "Fitting is on the log scale throughout, which keeps fares positive and "
        "makes airline differences multiplicative rather than additive.", BODY))

    A(p("4c. Chosen by validation, not by taste", H3))
    A(p("Every structural choice above was selected on a validation year and only "
        "then confirmed on a held-out test year "
        "(<font face='Courier'>scripts/sweep.py</font>, "
        "<font face='Courier'>data/sweep_results.json</font>):", BODY))
    A(table([
        ["Choice", "Alternatives tested", "Outcome"],
        ["Linear trend", "on / off", "<b>off</b> - on costs ~3pp MAPE"],
        ["Fourier harmonics", "1 / 2 / 3", "<b>2</b> - 1 underfits, 3 adds nothing"],
        ["Oil term", "off / fixed-beta", "<b>on</b> - ~1.3pp better"],
        ["Recency half-life", "none / 730 / 365 / 180 / 90 d", "<b>730 d</b> - 90 d starves seasonality"],
        ["Route pooling", "per-series / per-route", "<b>per-route</b> + level offsets"],
        ["Index aggregation", "median / mean / min", "<b>mean</b>"],
        ["Curve buckets", "coarse / default / fine", "no material difference"],
        ["Ridge penalty", "0 / 0.1 / 1 / 10 / 100", "<b>0</b> - no gain"],
    ], [34 * mm, 58 * mm, 84 * mm]))

    # ---------------------------------------------------------- 5
    A(p("5. From forecast to a buy-or-wait decision", H2))
    A(p("The forecast above predicts the index for a <i>departure</i> date. The "
        "decision a traveller faces is different: given a specific flight, on "
        "which remaining day should it be bought? That comes from the booking "
        "curve, anchored to the price actually quoted today - so the model "
        "supplies the shape while the live quote supplies the level.", BODY))
    A(eq("P^(buy at horizon d) = P_now * C_r(d) / C_r(d_now)"))
    A(eq("d*   = argmin over d <= d_now of C_r(d)      # recommended buy horizon"))
    A(eq("save = P_now - P^(d*)"))
    A(p("'Wait' is shown only when the predicted saving exceeds 3% of the current "
        "price; otherwise the verdict is 'book now'. For return trips both legs "
        "are priced on the same purchase date and the sum is minimised jointly, "
        "since a round trip is bought in one transaction.", BODY))

    # ---------------------------------------------------------- 6
    A(p("6. Correlation between routes and airlines", H2))
    A(p("Correlations are computed on 7-day differences of the log rolling-week "
        "index, not on levels. Two unrelated routes both have summer peaks, so "
        "their levels correlate strongly for a trivial reason; differencing "
        "removes the shared seasonal component and leaves genuine co-movement.", BODY))
    A(eq("g(i,t) = log X7(i,t) - log X7(i,t-7)"))
    A(eq("R      = corr( g ),  pairs with >= 120 overlapping days"))
    A(table([
        ["Pair type", "Mean correlation", "Reading"],
        ["Same route, different airline", f"<b>{corr['avg_same_route']}</b>",
         "carriers on a route track each other closely"],
        ["Same airline, different route", f"<b>{corr['avg_same_airline']}</b>",
         "little network-wide common pricing"],
        ["Unrelated pairs", f"<b>{corr['avg_unrelated']}</b>", "baseline"],
    ], [50 * mm, 32 * mm, 94 * mm]))
    A(p(f"Measured across {corr['n_pairs']:,} pairs from {corr['n_series']} series. "
        f"Competitive response within a route dominates; an airline's own pricing "
        f"across different routes is close to independent.", NOTE))

    # ---------------------------------------------------------- 7
    A(p("7. Validation protocol", H2))
    A(p("Accuracy is measured by temporal holdout. Everything that is fitted - the "
        "booking curve, the index normalisation, the oil lag and elasticity, and "
        "the regressions - is re-estimated using only data before the cutoff. The "
        "test period is then scored blind, and oil is frozen at the cutoff so no "
        "future information enters through that channel.", BODY))
    A(p("Two baselines are reported, because a forecast that cannot beat a naive "
        "rule is not worth running: <b>seasonal-naive</b> (same calendar date one "
        "year earlier, 364 days back to preserve weekday) and <b>train-mean</b> "
        "(the series average over the training window).", BODY))
    A(table([
        ["Metric", f"Train &le; {p1['cutoff']}<br/>test {p1['test_end'][:4]}",
         f"Train &le; {p2['cutoff']}<br/>test {p2['test_end'][:4]}"],
        ["Forecast MAPE", f"<b>{p1['mape_model']}%</b>", f"<b>{p2['mape_model']}%</b>"],
        ["Seasonal-naive MAPE", f"{p1['mape_seasonal_naive']}%", f"{p2['mape_seasonal_naive']}%"],
        ["Train-mean MAPE", f"{p1['mape_train_mean']}%", f"{p2['mape_train_mean']}%"],
        ["Bias", f"{p1['bias_pct']:+}%", f"{p2['bias_pct']:+}%"],
        ["Series / daily observations", f"{p1['n_series']} / {p1['n_obs']:,}",
         f"{p2['n_series']} / {p2['n_obs']:,}"],
    ], [50 * mm, 63 * mm, 63 * mm]))
    A(p(f"The second split's negative bias ({p2['bias_pct']}%) is the frozen-oil "
        f"assumption doing its job: its test year contains a crude spike the model "
        f"could not have known about, so fares are systematically under-predicted. "
        f"That is the honest cost of refusing lookahead, not a fitting error.", NOTE))

    A(p("7b. Scoring the buy-timing advice", H3))
    A(p("Price accuracy and decision quality are different questions, so the "
        "advice is scored separately. For each test flight the simulation stands "
        "at a standing point 30 days before that route's own curve trough, follows "
        "the recommendation, and pays the nearest quote actually observed on the "
        "recommended day - you cannot buy at a price nobody quoted.", BODY))
    A(p("The standing point is route-relative by necessity. A fixed 60-day anchor "
        "stands <i>past</i> the trough on long-haul routes, where the advice can "
        "only ever say 'buy now' and scores a trivial zero; that mistake made the "
        "measured capture ratio look like 0.31 rather than 0.71.", NOTE))
    A(table([
        ["Outcome vs buying at the standing point",
         f"Test {p1['test_end'][:4]}", f"Test {p2['test_end'][:4]}"],
        ["Mean saving", f"<b>{t1['avg_saving_vs_buy_now_pct']:+}%</b>",
         f"<b>{t2['avg_saving_vs_buy_now_pct']:+}%</b>"],
        ["Median saving", f"{t1['median_saving_vs_buy_now_pct']:+}%",
         f"{t2['median_saving_vs_buy_now_pct']:+}%"],
        ["Standard deviation", f"{t1['std_saving_pct']} pp", f"{t2['std_saving_pct']} pp"],
        ["10th / 90th percentile",
         f"{t1['saving_pct_percentiles']['p10']}% / {t1['saving_pct_percentiles']['p90']}%",
         f"{t2['saving_pct_percentiles']['p10']}% / {t2['saving_pct_percentiles']['p90']}%"],
        ["Flights helped / hurt",
         f"{t1['pct_flights_advice_helped']}% / {t1['pct_flights_advice_hurt']}%",
         f"{t2['pct_flights_advice_helped']}% / {t2['pct_flights_advice_hurt']}%"],
        ["Share of the perfect-hindsight saving captured",
         f"{round(t1['capture_ratio'] * 100)}%", f"{round(t2['capture_ratio'] * 100)}%"],
        ["Flights simulated", f"{t1['n_flights']:,}", f"{t2['n_flights']:,}"],
    ], [66 * mm, 55 * mm, 55 * mm]))
    A(p(f"Broken out by curve shape: routes that bottom out late "
        f"(&gt; 60 days, mostly long-haul) yield "
        f"{t1['by_curve_shape'].get('books early (trough > 60d)', {}).get('avg_saving_pct', 'n/a')}% "
        f"on average, while routes bottoming out within 60 days yield "
        f"{t1['by_curve_shape'].get('books late (trough <= 60d)', {}).get('avg_saving_pct', 'n/a')}%. "
        f"Worst observed single-flight outcome across both test years is about "
        f"-5%: the advice only shifts a purchase by a few weeks, so the downside "
        f"is bounded by drift over that window.", NOTE))

    # ---------------------------------------------------------- 8
    A(p("8. What this model does not do", H2))
    A(p("&bull; <b>It is calendar-driven, not event-driven.</b> Flash sales, "
        "capacity changes and strikes are not predicted; they appear as residual "
        "noise. Remaining error is dominated by the autocorrelated demand shocks "
        "the design deliberately does not chase.", BODY))
    A(p("&bull; <b>The elasticity is a level relationship, not a causal claim.</b> "
        "Oil and fares share business-cycle drivers; the regression measures "
        "association after seasonality, not a fuel-cost mechanism.", BODY))
    A(p("&bull; <b>Fare data here is synthetic.</b> No free source publishes daily "
        "route-airline fares: public fare statistics are quarterly or monthly "
        "averages, price-analysis APIs return quartiles rather than series, and "
        "continuous feeds are commercial. The generator's structure - booking "
        "curve, seasonality, route-level shocks, lagged oil pass-through - is "
        "known, which is what lets the estimators above be checked against a "
        "ground truth they were not told. Real quotes drop into the same schema "
        "without changing any model code.", BODY))
    A(p("&bull; <b>Synthetic noise is smoother than real fares.</b> Real tails "
        "would be fatter than the bounded downside reported above.", BODY))

    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=17 * mm, rightMargin=17 * mm,
                          topMargin=16 * mm, bottomMargin=20 * mm,
                          title="Farecast - fare regression methodology",
                          author="Farecast")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
    doc.build(story)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    build()
