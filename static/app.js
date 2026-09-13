/* Farecast dashboard — vanilla JS + hand-rolled SVG charts. */
"use strict";

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const $ = s => document.querySelector(s);
const fmt$ = v => v == null ? "–" : "$" + Number(v).toFixed(0);
const fmt2 = v => v == null ? "–" : Number(v).toFixed(2);

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

/* ---------------------------------------------------------------- charts */

// Generic multi-series line chart with crosshair tooltip + optional band.
// spec: {x:[labels], series:[{name,color,values,dash?,width?}], band:{lo,hi,color},
//        ylabel?, yfmt?, markers?:[{i,label}]}
function lineChart(el, spec) {
  el.innerHTML = "";
  const W = 920, H = 300, m = {t: 14, r: 16, b: 34, l: 52};
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const n = spec.x.length;
  const all = [];
  spec.series.forEach(s => s.values.forEach(v => { if (v != null) all.push(v); }));
  if (spec.band) { spec.band.lo.forEach(v => v != null && all.push(v)); spec.band.hi.forEach(v => v != null && all.push(v)); }
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
  if (spec.zero) lo = Math.min(lo, 0);
  const X = i => m.l + (n <= 1 ? 0 : i / (n - 1) * iw);
  const Y = v => m.t + (1 - (v - lo) / (hi - lo)) * ih;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  const put = (tag, attrs, parent = svg) => {
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    parent.appendChild(e); return e;
  };
  // gridlines + y ticks (clean numbers)
  const ticks = niceTicks(lo, hi, 5);
  for (const t of ticks) {
    put("line", {x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: css("--grid"), "stroke-width": 1});
    const txt = put("text", {x: m.l - 8, y: Y(t) + 4, "text-anchor": "end", fill: css("--muted"),
      "font-size": 11, style: "font-variant-numeric:tabular-nums"});
    txt.textContent = (spec.yfmt || (v => v)) (t);
  }
  // x labels (~6)
  const step = Math.max(1, Math.round(n / 6));
  for (let i = 0; i < n; i += step) {
    const txt = put("text", {x: X(i), y: H - 10, "text-anchor": "middle", fill: css("--muted"), "font-size": 11});
    txt.textContent = shortDate(spec.x[i]);
  }
  put("line", {x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, stroke: css("--axis"), "stroke-width": 1});
  // band
  if (spec.band) {
    let d = "";
    spec.band.hi.forEach((v, i) => { if (v != null) d += (d ? "L" : "M") + X(i) + " " + Y(v); });
    for (let i = n - 1; i >= 0; i--) { const v = spec.band.lo[i]; if (v != null) d += "L" + X(i) + " " + Y(v); }
    put("path", {d: d + "Z", fill: spec.band.color, opacity: 0.10});
  }
  // vertical markers
  (spec.markers || []).forEach(mk => {
    put("line", {x1: X(mk.i), x2: X(mk.i), y1: m.t, y2: m.t + ih, stroke: css("--axis"), "stroke-width": 1});
    const t = put("text", {x: X(mk.i) + 5, y: m.t + 12, fill: css("--muted"), "font-size": 11});
    t.textContent = mk.label;
  });
  // series lines
  for (const s of spec.series) {
    let d = "";
    s.values.forEach((v, i) => { d += v == null ? "" : (d && s.values[i-1] != null ? "L" : "M") + X(i) + " " + Y(v); });
    put("path", {d, fill: "none", stroke: s.color, "stroke-width": s.width || 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
      ...(s.dash ? {"stroke-dasharray": "5 4"} : {})});
    // end-dot with surface ring on the last non-null point
    for (let i = n - 1; i >= 0; i--) if (s.values[i] != null) {
      put("circle", {cx: X(i), cy: Y(s.values[i]), r: 4.5, fill: s.color,
        stroke: css("--surface"), "stroke-width": 2});
      break;
    }
  }
  // crosshair + tooltip
  const cross = put("line", {x1: 0, x2: 0, y1: m.t, y2: m.t + ih, stroke: css("--axis"),
    "stroke-width": 1, visibility: "hidden"});
  const dots = spec.series.map(s => put("circle", {r: 4.5, fill: s.color, stroke: css("--surface"),
    "stroke-width": 2, visibility: "hidden"}));
  el.appendChild(svg);
  const tip = document.createElement("div"); tip.className = "tooltip"; el.appendChild(tip);
  const hit = put("rect", {x: m.l, y: m.t, width: iw, height: ih, fill: "transparent"});
  const show = evt => {
    const r = svg.getBoundingClientRect();
    const px = (evt.clientX - r.left) * (W / r.width);
    const i = Math.max(0, Math.min(n - 1, Math.round((px - m.l) / iw * (n - 1))));
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i));
    cross.setAttribute("visibility", "visible");
    tip.style.display = "block";
    tip.innerHTML = "";
    const xr = document.createElement("div"); xr.className = "tt-x";
    xr.textContent = spec.x[i]; tip.appendChild(xr);
    spec.series.forEach((s, k) => {
      const v = s.values[i];
      if (v == null) { dots[k].setAttribute("visibility", "hidden"); return; }
      dots[k].setAttribute("cx", X(i)); dots[k].setAttribute("cy", Y(v));
      dots[k].setAttribute("visibility", "visible");
      const row = document.createElement("div"); row.className = "tt-row";
      const key = document.createElement("span"); key.className = "tt-key";
      key.style.borderTopColor = s.color;
      if (s.dash) key.style.borderTopStyle = "dashed";
      const nm = document.createElement("span"); nm.textContent = s.name;
      const vv = document.createElement("span"); vv.className = "tt-val";
      vv.textContent = (spec.yfmt || (x => x))(v);
      row.append(key, nm, vv); tip.appendChild(row);
    });
    const tx = X(i) / W * r.width;
    tip.style.left = Math.min(r.width - tip.offsetWidth - 8, Math.max(0, tx + 14)) + "px";
    tip.style.top = "10px";
  };
  hit.addEventListener("pointermove", show);
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden"); tip.style.display = "none";
    dots.forEach(d => d.setAttribute("visibility", "hidden"));
  });
  // legend (only for >= 2 series)
  if (spec.series.length >= 2) {
    const lg = document.createElement("div"); lg.className = "legend";
    for (const s of spec.series) {
      const k = document.createElement("span"); k.className = "key";
      const ln = document.createElement("span"); ln.className = "line" + (s.dash ? " dash" : "");
      ln.style.borderTopColor = s.color;
      const t = document.createElement("span"); t.textContent = s.name;
      k.append(ln, t); lg.appendChild(k);
    }
    el.prepend(lg);
  }
  el._tabledata = {
    head: ["", ...spec.series.map(s => s.name)],
    rows: spec.x.map((x, i) => [x, ...spec.series.map(s =>
      s.values[i] == null ? "–" : (spec.yfmt || (v => v))(s.values[i]))]),
  };
}

function niceTicks(lo, hi, count) {
  const span = hi - lo, step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  let step = mag;
  for (const s of [1, 2, 2.5, 5, 10]) if (mag * s >= step0) { step = mag * s; break; }
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(+t.toFixed(6));
  return out;
}
function shortDate(s) {
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) {
    const d = new Date(s + "T00:00:00");
    return d.toLocaleDateString(undefined, {month: "short", day: "numeric",
      year: d.getMonth() === 0 && d.getDate() < 8 ? "numeric" : undefined});
  }
  return s;
}

// Correlation heatmap: diverging blue <-> red, gray midpoint.
function heatmap(el, labels, matrix) {
  el.innerHTML = "";
  const nn = labels.length, cell = 26, gap = 2, left = 118, top = 96;
  const W = left + nn * (cell + gap) + 10, H = top + nn * (cell + gap) + 10;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.style.maxWidth = W + "px";
  const put = (tag, attrs) => {
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    svg.appendChild(e); return e;
  };
  const mid = css("--div-mid");
  const color = v => {
    if (v == null) return css("--grid");
    // interpolate mid -> pole in sRGB (adequate for a UI heatmap)
    const pole = v >= 0 ? [42, 120, 214] : [227, 73, 72];
    const t = Math.min(1, Math.abs(v));
    const m3 = mid.match(/#(..)(..)(..)/).slice(1).map(h => parseInt(h, 16));
    const c = m3.map((mv, i) => Math.round(mv + (pole[i] - mv) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };
  labels.forEach((lab, j) => {
    const t = put("text", {x: left + j * (cell + gap) + cell / 2, y: top - 8,
      fill: css("--muted"), "font-size": 10,
      transform: `rotate(-45 ${left + j * (cell + gap) + cell / 2} ${top - 8})`});
    t.textContent = lab;
  });
  labels.forEach((lab, i) => {
    const t = put("text", {x: left - 8, y: top + i * (cell + gap) + cell / 2 + 4,
      "text-anchor": "end", fill: css("--muted"), "font-size": 10});
    t.textContent = lab;
  });
  const tip = document.createElement("div"); tip.className = "tooltip";
  for (let i = 0; i < nn; i++) for (let j = 0; j < nn; j++) {
    const v = matrix[i][j];
    const r = put("rect", {x: left + j * (cell + gap), y: top + i * (cell + gap),
      width: cell, height: cell, rx: 4, fill: color(v)});
    r.addEventListener("pointermove", evt => {
      const box = el.getBoundingClientRect();
      tip.style.display = "block";
      tip.innerHTML = "";
      const xr = document.createElement("div"); xr.className = "tt-x";
      xr.textContent = labels[i] + " × " + labels[j]; tip.appendChild(xr);
      const row = document.createElement("div"); row.className = "tt-row";
      const vv = document.createElement("span"); vv.className = "tt-val";
      vv.textContent = v == null ? "n/a" : v.toFixed(2);
      row.appendChild(vv); tip.appendChild(row);
      tip.style.left = Math.min(box.width - 150, evt.clientX - box.left + 14) + "px";
      tip.style.top = (evt.clientY - box.top - 40) + "px";
    });
    r.addEventListener("pointerleave", () => tip.style.display = "none");
  }
  el.appendChild(svg); el.appendChild(tip);
  el._tabledata = {head: ["", ...labels],
    rows: labels.map((lab, i) => [lab, ...matrix[i].map(v => v == null ? "–" : v.toFixed(2))])};
}

/* table-view toggles (accessibility twin for every chart) */
document.addEventListener("click", e => {
  const btn = e.target.closest("[data-table-for]");
  if (!btn) return;
  const el = document.getElementById(btn.dataset.tableFor);
  const open = btn.getAttribute("aria-pressed") === "true";
  btn.setAttribute("aria-pressed", String(!open));
  let tw = el.parentElement.querySelector(".tablewrap");
  if (open) { tw && tw.remove(); return; }
  if (!el._tabledata) return;
  tw = document.createElement("div"); tw.className = "tablewrap";
  const t = document.createElement("table"); t.className = "dataview";
  const thead = document.createElement("tr");
  el._tabledata.head.forEach(h => { const th = document.createElement("th"); th.textContent = h; thead.appendChild(th); });
  t.appendChild(thead);
  el._tabledata.rows.forEach(r => {
    const tr = document.createElement("tr");
    r.forEach(c => { const td = document.createElement("td"); td.textContent = c; tr.appendChild(td); });
    t.appendChild(tr);
  });
  tw.appendChild(t); el.after(tw);
});

/* ------------------------------------------------------------------ tiles */
function tile(lab, val, note) {
  const d = document.createElement("div"); d.className = "tile";
  const l = document.createElement("div"); l.className = "lab"; l.textContent = lab;
  const v = document.createElement("div"); v.className = "val"; v.textContent = val;
  d.append(l, v);
  if (note) { const nt = document.createElement("div"); nt.className = "note"; nt.textContent = note; d.appendChild(nt); }
  return d;
}

/* ------------------------------------------------------------------- app */
let META;
const AIRLINE = c => (META && META.airline_names[c]) ? `${META.airline_names[c]} (${c})` : c;

async function boot() {
  META = await api("/api/meta");
  const badge = $("#src-badge");
  if (META.data_source === "synthetic") {
    badge.textContent = "SYNTHETIC DEMO DATA";
    badge.classList.add("synthetic");
    badge.title = "No real fare feed configured - prices are generated. See README to plug in real data.";
  } else {
    badge.textContent = "REAL FARE DATA";
  }
  // route selects
  const origins = [...new Set(META.routes.map(r => r.origin))].sort();
  $("#origin").innerHTML = origins.map(o => `<option>${o}</option>`).join("");
  syncDest();
  $("#origin").addEventListener("change", syncDest);
  $("#go").addEventListener("click", runSearch);
  $("#pill-oneway").addEventListener("click", () => setTrip("oneway"));
  $("#pill-return").addEventListener("click", () => setTrip("return"));
  // forecast series select
  const opts = [];
  for (const r of META.routes) for (const a of r.airlines)
    opts.push(`${r.origin}-${r.dest}:${a}`);
  $("#fc-series").innerHTML = opts.map(s => `<option value="${s}">${s.replace(":", "  ·  ")}</option>`).join("");
  $("#fc-series").addEventListener("change", loadForecast);

  loadForecast();
  loadBookingCurve();
  loadOil();
  loadCorr();
  loadBacktest();
  runSearch();
}

function syncDest() {
  const o = $("#origin").value;
  const dests = META.routes.filter(r => r.origin === o).map(r => r.dest).sort();
  $("#dest").innerHTML = dests.map(d => `<option>${d}</option>`).join("");
}

let TRIP = "oneway";
function setTrip(t) {
  TRIP = t;
  $("#pill-oneway").setAttribute("aria-pressed", String(t === "oneway"));
  $("#pill-return").setAttribute("aria-pressed", String(t === "return"));
  $("#rdate-wrap").hidden = t !== "return";
  runSearch();
}

const D1 = 86400000;
const toDate = s => new Date(s + "T00:00:00");
const addDays = (s, n) => new Date(toDate(s).getTime() + n * D1).toISOString().slice(0, 10);
const dayDiff = (a, b) => Math.round((toDate(a) - toDate(b)) / D1);

function filterLegs(quotes, today, dateStr, minDate) {
  return quotes.filter(q => {
    if (dateStr) return Math.abs(dayDiff(q.flight_date, dateStr)) <= 3;
    if (minDate && dayDiff(q.flight_date, minDate) < 1) return false;
    return dayDiff(q.flight_date, today) <= 90;
  });
}

// predicted price of one leg for each remaining buy day t = 0..maxT (days from today)
function legPath(q, quote, maxT) {
  const c = q.curve, dtd = quote.dtd, out = [];
  for (let t = 0; t <= maxT; t++) out.push(quote.fare * c[dtd - t] / c[dtd]);
  return out;
}

function comboPredict(qO, qI, o, i) {
  const maxT = Math.min(o.dtd - 1, i.dtd - 1);   // buy both legs together, before departure
  const po = legPath(qO, o, maxT), pi = legPath(qI, i, maxT);
  let best = 0, bestV = Infinity;
  for (let t = 0; t <= maxT; t++) {
    const v = po[t] + pi[t];
    if (v < bestV) { bestV = v; best = t; }
  }
  return { total: o.fare + i.fare, predicted_min: bestV, best_t: best,
           path: po.map((v, t) => v + pi[t]) };
}

async function runSearch() {
  $("#search-err").textContent = "";
  const o = $("#origin").value, d = $("#dest").value;
  const dd = $("#ddate").value, rd = $("#rdate").value;
  try {
    const qO = await api(`/api/quotes?origin=${o}&dest=${d}`);
    if (qO.error) { $("#search-err").textContent = qO.error; return; }
    $("#results").hidden = false;
    $("#traj-block").hidden = true;
    const today = qO.today;

    if (TRIP === "oneway") {
      const legs = filterLegs(qO.quotes, today, dd, null);
      $("#results").querySelectorAll("h3")[0].textContent = "Cheapest right now";
      $("#results").querySelectorAll("h3")[1].textContent = "Predicted cheapest by departure";
      renderLegs($("#list-now"), [...legs].sort((a, b) => a.fare - b.fare).slice(0, 12), false, qO, o, d);
      renderLegs($("#list-future"), [...legs].sort((a, b) => a.predicted_min - b.predicted_min).slice(0, 12), true, qO, o, d);
    } else {
      const qI = await api(`/api/quotes?origin=${d}&dest=${o}`);
      if (qI.error) { $("#search-err").textContent = qI.error; return; }
      const outs = filterLegs(qO.quotes, today, dd, null)
        .sort((a, b) => a.fare - b.fare).slice(0, 40);
      const ins = filterLegs(qI.quotes, today, rd, dd || today)
        .sort((a, b) => a.fare - b.fare).slice(0, 40);
      const combos = [];
      for (const ol of outs) for (const il of ins) {
        if (dayDiff(il.flight_date, ol.flight_date) < 1) continue;
        combos.push({ out: ol, in: il, total: ol.fare + il.fare });
      }
      combos.sort((a, b) => a.total - b.total);
      const top = combos.slice(0, 60);
      for (const c of top) {
        const p = comboPredict(qO, qI, c.out, c.in);
        c.predicted_min = p.predicted_min;
        c.best_buy_date = addDays(today, p.best_t);
        c.expected_saving = c.total - p.predicted_min;
        c.verdict = p.predicted_min < c.total * 0.97 ? "wait" : "book now";
      }
      $("#results").querySelectorAll("h3")[0].textContent = "Cheapest round trips right now";
      $("#results").querySelectorAll("h3")[1].textContent = "Predicted cheapest if you time the purchase";
      renderCombos($("#list-now"), combos.slice(0, 12), false, qO, qI, o, d);
      renderCombos($("#list-future"), [...top].sort((a, b) => a.predicted_min - b.predicted_min).slice(0, 12), true, qO, qI, o, d);
    }
  } catch (e) { $("#search-err").textContent = String(e); }
}

function selectRow(li) {
  document.querySelectorAll("#results li[aria-selected]").forEach(x => x.removeAttribute("aria-selected"));
  li.setAttribute("aria-selected", "true");
}

function renderLegs(ul, items, future, qO, o, d) {
  ul.innerHTML = "";
  if (!items.length) { ul.innerHTML = "<li class='f-note'>no flights in window</li>"; return; }
  for (const f of items) {
    const li = document.createElement("li");
    li.tabIndex = 0; li.setAttribute("role", "button");
    const dt = document.createElement("span"); dt.className = "f-date"; dt.textContent = f.flight_date;
    const dow = document.createElement("span"); dow.className = "f-dow"; dow.textContent = f.dow;
    const air = document.createElement("span"); air.className = "f-air"; air.textContent = AIRLINE(f.airline);
    li.append(dt, dow, air);
    if (future) {
      const chip = document.createElement("span");
      chip.className = "chip " + (f.verdict === "wait" ? "wait" : "now");
      chip.textContent = f.verdict === "wait" ? `wait → ${shortDate(f.best_buy_date)}` : "book now";
      li.appendChild(chip);
      if (f.expected_saving > 1) {
        const sv = document.createElement("span"); sv.className = "saving";
        sv.textContent = `save ~${fmt$(f.expected_saving)}`; li.appendChild(sv);
      }
    }
    const fare = document.createElement("span"); fare.className = "f-fare";
    fare.textContent = fmt$(future ? f.predicted_min : f.fare);
    if (future) fare.title = `now ${fmt$(f.fare)}`;
    li.appendChild(fare);
    const open = () => { selectRow(li); loadLegTrajectory(qO, f, o, d); };
    li.addEventListener("click", open);
    li.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    ul.appendChild(li);
  }
}

function renderCombos(ul, items, future, qO, qI, o, d) {
  ul.innerHTML = "";
  if (!items.length) { ul.innerHTML = "<li class='f-note'>no combinations in window</li>"; return; }
  for (const c of items) {
    const li = document.createElement("li");
    li.tabIndex = 0; li.setAttribute("role", "button");
    const dt = document.createElement("span"); dt.className = "f-date";
    dt.textContent = `${shortDate(c.out.flight_date)} → ${shortDate(c.in.flight_date)}`;
    dt.style.width = "128px";
    const air = document.createElement("span"); air.className = "f-leg";
    air.textContent = c.out.airline === c.in.airline ? AIRLINE(c.out.airline)
      : `${c.out.airline} out · ${c.in.airline} back`;
    li.append(dt, air);
    if (future) {
      const chip = document.createElement("span");
      chip.className = "chip " + (c.verdict === "wait" ? "wait" : "now");
      chip.textContent = c.verdict === "wait" ? `wait → ${shortDate(c.best_buy_date)}` : "book now";
      li.appendChild(chip);
      if (c.expected_saving > 1) {
        const sv = document.createElement("span"); sv.className = "saving";
        sv.textContent = `save ~${fmt$(c.expected_saving)}`; li.appendChild(sv);
      }
    }
    const fare = document.createElement("span"); fare.className = "f-fare";
    fare.textContent = fmt$(future ? c.predicted_min : c.total);
    if (future) fare.title = `now ${fmt$(c.total)}`;
    li.appendChild(fare);
    const open = () => { selectRow(li); showComboTrajectory(qO, qI, c, o, d); };
    li.addEventListener("click", open);
    li.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    ul.appendChild(li);
  }
}

function bandAround(mean, sigma) {
  const n = mean.length, lo = [], hi = [];
  for (let t = 0; t < n; t++) {
    const g = Math.sqrt(t / Math.max(n, 1) + 1e-9);
    lo.push(mean[t] * Math.exp(-1.28 * sigma * g));
    hi.push(mean[t] * Math.exp(1.28 * sigma * g));
  }
  return { lo, hi };
}

async function loadLegTrajectory(qO, f, o, d) {
  // observed history exists server-side (and for snapshot top flights);
  // otherwise predict client-side from the booking curve
  try {
    const r = await api(`/api/trajectory?origin=${o}&dest=${d}&airline=${f.airline}&flight_date=${f.flight_date}`);
    if (!r.error) return drawTrajectory(r, o, d, f.airline, f.flight_date);
  } catch (e) { /* fall through to client-side prediction */ }
  const mean = legPath(qO, f, f.dtd - 1);
  const dates = mean.map((_, t) => addDays(qO.today, t));
  const { lo, hi } = bandAround(mean, qO.sigma[f.airline] || 0.05);
  drawTrajectory({ fare_now: f.fare, observed: { dates: [], fare: [] },
                   predicted: { dates, mean, lo, hi } }, o, d, f.airline, f.flight_date);
}

function showComboTrajectory(qO, qI, c, o, d) {
  const p = comboPredict(qO, qI, c.out, c.in);
  const dates = p.path.map((_, t) => addDays(qO.today, t));
  const so = qO.sigma[c.out.airline] || 0.05, si = qI.sigma[c.in.airline] || 0.05;
  const { lo, hi } = bandAround(p.path, Math.sqrt((so * so + si * si) / 2));
  $("#traj-block").hidden = false;
  $("#traj-title").textContent =
    `${o} → ${d} → ${o} · ${shortDate(c.out.flight_date)} out (${c.out.airline}), ${shortDate(c.in.flight_date)} back (${c.in.airline})`;
  $("#traj-sub").textContent =
    `Round trip now ${fmt$(c.total)} · predicted minimum ${fmt$(p.predicted_min)} if bought ${dates[p.best_t]}`;
  lineChart($("#traj-chart"), {
    x: dates, yfmt: fmt$,
    band: { lo, hi, color: css("--s1") },
    markers: [{ i: p.best_t, label: "best buy" }],
    series: [{ name: "predicted round-trip price by buy date", color: css("--s1"), values: p.path, dash: true }],
  });
}

function drawTrajectory(r, o, d, airline, fdate) {
  $("#traj-block").hidden = false;
  $("#traj-title").textContent = `${o} → ${d} · ${AIRLINE(airline)} · departs ${fdate}`;
  const minI = r.predicted.mean.indexOf(Math.min(...r.predicted.mean));
  $("#traj-sub").textContent =
    `Now ${fmt$(r.fare_now)} · predicted minimum ${fmt$(r.predicted.mean[minI])} if bought ${r.predicted.dates[minI]}`;
  const nO = r.observed.dates.length;
  if (nO === 0) {
    lineChart($("#traj-chart"), {
      x: r.predicted.dates, yfmt: fmt$,
      band: { lo: r.predicted.lo, hi: r.predicted.hi, color: css("--s1") },
      markers: [{ i: minI, label: "best buy" }],
      series: [{ name: "predicted price by buy date", color: css("--s1"), values: r.predicted.mean, dash: true }],
    });
    return;
  }
  const xs = [...r.observed.dates, ...r.predicted.dates.slice(1)];
  const obs = [...r.observed.fare, ...Array(r.predicted.dates.length - 1).fill(null)];
  const pred = [...Array(nO - 1).fill(null), r.fare_now, ...r.predicted.mean.slice(1)];
  const lo = [...Array(nO - 1).fill(null), r.fare_now, ...r.predicted.lo.slice(1)];
  const hi = [...Array(nO - 1).fill(null), r.fare_now, ...r.predicted.hi.slice(1)];
  lineChart($("#traj-chart"), {
    x: xs, yfmt: fmt$,
    band: { lo, hi, color: css("--s1") },
    markers: [{ i: nO - 1 + minI, label: "best buy" }],
    series: [
      { name: "observed price", color: css("--s2"), values: obs },
      { name: "predicted price by buy date", color: css("--s1"), values: pred, dash: true },
    ],
  });
}

async function loadForecast() {
  const [o, rest] = $("#fc-series").value.split("-");
  const [d, a] = rest.split(":");
  const r = await api(`/api/forecast?origin=${o}&dest=${d}&airline=${a}`);
  if (r.error) return;
  const tiles = $("#fc-tiles"); tiles.innerHTML = "";
  const lastIdx = r.history.fare_7d.map(v => v != null).lastIndexOf(true);
  tiles.append(
    tile("Current fare index (7d avg)", fmt$(r.history.fare_7d[lastIdx])),
    tile("Forecast in 30 days", fmt$(r.forecast.mean[Math.min(29, r.forecast.mean.length - 1)])),
    tile("Forecast in 90 days", fmt$(r.forecast.mean[Math.min(89, r.forecast.mean.length - 1)])),
    tile("Model fit R²", fmt2(r.r2), `${r.n_obs} daily obs`),
  );
  const xs = [...r.history.dates, ...r.forecast.dates];
  const nH = r.history.dates.length, nF = r.forecast.dates.length;
  lineChart($("#fc-chart"), {
    x: xs, yfmt: fmt$,
    band: {lo: [...Array(nH).fill(null), ...r.forecast.lo],
           hi: [...Array(nH).fill(null), ...r.forecast.hi], color: css("--s1")},
    series: [
      {name: "daily fare index (7d avg)", color: css("--s2"), values: [...r.history.fare_7d, ...Array(nF).fill(null)]},
      {name: "model fit", color: css("--s1"), values: [...r.history.fitted, ...Array(nF).fill(null)], width: 1.5},
      {name: "forecast", color: css("--s1"), values: [...Array(nH).fill(null), ...r.forecast.mean], dash: true},
    ],
  });
}

async function loadBookingCurve() {
  const r = await api("/api/booking_curve");
  lineChart($("#bc-chart"), {
    x: r.dtd.map(String).reverse(),
    yfmt: v => v.toFixed(2) + "×",
    series: [{name: "fare multiplier", color: css("--s1"), values: [...r.multiplier].reverse()}],
  });
  const cap = document.createElement("p"); cap.className = "hint";
  cap.textContent = "x-axis: days before departure (left = book early, right = last minute)";
  $("#bc-chart").appendChild(cap);
}

async function loadOil() {
  const r = await api("/api/oil");
  const tiles = $("#oil-tiles"); tiles.innerHTML = "";
  tiles.append(
    tile("Best pooled lag", r.best_lag_days + " days", "argmax of mean correlation"),
    tile("Avg per-series lag", (r.avg_series_lag_days ?? "–") + " days", "mean of per-series argmax"),
    tile("Pass-through elasticity", fmt2(r.elasticity), "Δlog fare / Δlog Brent at best lag"),
    tile("R² at best lag", fmt2(r.r2)),
  );
  lineChart($("#lag-chart"), {
    x: r.lags.map(String), zero: true,
    yfmt: v => v.toFixed(2),
    markers: [{i: r.lags.indexOf(r.best_lag_days), label: `${r.best_lag_days}d`}],
    series: [{name: "mean correlation", color: css("--s3"), values: r.mean_correlation}],
  });
  const cap = document.createElement("p"); cap.className = "hint";
  cap.textContent = "x-axis: lag in days (oil leads fares)";
  $("#lag-chart").appendChild(cap);
  lineChart($("#oil-chart"), {
    x: r.dates, yfmt: v => v.toFixed(0),
    series: [
      {name: "avg fare index (=100 at start)", color: css("--s1"), values: r.fare_indexed},
      {name: "Brent crude (=100 at start)", color: css("--s2"), values: r.brent_indexed},
    ],
  });
}

async function loadBacktest() {
  const r = await api("/api/backtest");
  const wrap = $("#bt-splits"); wrap.innerHTML = "";
  for (const sp of r.splits) {
    const p = sp.price, t = sp.timing;
    const col = document.createElement("div");
    const h = document.createElement("h3");
    h.style.cssText = "font-size:13px;margin:0 0 8px;color:var(--ink-2)";
    h.textContent = `Train \u2264 ${p.cutoff} \u2192 test to ${p.test_end}`;
    const tiles = document.createElement("div"); tiles.className = "tiles";
    tiles.append(
      tile("Forecast MAPE", p.mape_model + "%", `${p.n_series} series · ${p.n_obs.toLocaleString()} daily obs`),
      tile("Seasonal-naive MAPE", p.mape_seasonal_naive + "%", "baseline: same date last year"),
      tile("Train-mean MAPE", p.mape_train_mean + "%", "baseline: series average"),
      tile("Bias", (p.bias_pct > 0 ? "+" : "") + p.bias_pct + "%", "OOS R\u00b2 (median series) " + p.r2_oos_median_series),
      tile("Timing: avg saving", (t.avg_saving_vs_buy_now_pct > 0 ? "+" : "") + t.avg_saving_vs_buy_now_pct + "%",
           `\u00b1${t.std_saving_pct}pp std · p10 ${t.saving_pct_percentiles.p10}% / p90 ${t.saving_pct_percentiles.p90}% · ${t.n_flights} flights`),
      tile("Advice helped", t.pct_flights_advice_helped + "%",
           `hurt ${t.pct_flights_advice_hurt}% · captured ${Math.round(t.capture_ratio * 100)}% of oracle`),
    );
    const btn = document.createElement("button");
    btn.className = "ghost"; btn.textContent = "Per-series table";
    const chartStub = document.createElement("div");
    chartStub.id = "bt-" + p.cutoff;
    chartStub._tabledata = {
      head: ["series", "n", "MAPE %", "naive %", "bias %", "R\u00b2 oos"],
      rows: p.per_series.map(x => [x.series, x.n, x.mape_model.toFixed(1),
        x.mape_seasonal_naive.toFixed(1), x.bias_pct.toFixed(1), x.r2_oos.toFixed(2)]),
    };
    btn.dataset.tableFor = chartStub.id;
    btn.setAttribute("aria-pressed", "false");
    col.append(h, tiles, btn, chartStub);
    wrap.appendChild(col);
  }
}

async function loadCorr() {
  const r = await api("/api/correlations");
  const tiles = $("#corr-tiles"); tiles.innerHTML = "";
  tiles.append(
    tile("Same route, different airline", fmt2(r.avg_same_route), "avg pairwise corr"),
    tile("Same airline, different route", fmt2(r.avg_same_airline), "avg pairwise corr"),
    tile("Unrelated pairs", fmt2(r.avg_unrelated), "baseline"),
  );
  heatmap($("#corr-chart"), r.labels, r.matrix);
}

boot().catch(e => { $("#src-badge").textContent = "error"; console.error(e); });
