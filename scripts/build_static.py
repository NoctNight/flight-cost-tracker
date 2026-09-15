"""Build the static dashboard snapshot into site/ (for Vercel or any static host).

With ~160 routes a single embedded payload would be tens of MB, so the build
splits it:

    site/core.js          meta, oil, correlations, booking curve, backtest
    site/r/<O>-<D>.js     that route's quotes, forecasts and trajectories

The page loads core.js up front and pulls a route chunk the first time a
search or forecast touches that route (injected as a <script>, so it works
from file:// and under any static host's CSP).

    python scripts/build_static.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.store import get_store  # noqa: E402

SITE = ROOT / "site"

API_SHIM = '''async function api(path) {
  const m = /^\\/api\\/(quotes|forecast|trajectory)\\?origin=([A-Z]{3})&dest=([A-Z]{3})/.exec(path);
  if (m) await loadRoute(`${m[2]}-${m[3]}`);
  const d = window.__FC_DATA[path];
  if (!d) throw new Error("not in this snapshot: " + path);
  return d;
}

// Route payloads are fetched lazily, one <script> per route.
const __routeLoads = {};
function loadRoute(route) {
  if (!__routeLoads[route]) {
    __routeLoads[route] = new Promise(resolve => {
      const s = document.createElement("script");
      s.src = `r/${route}.js`;
      s.onload = s.onerror = () => resolve();
      document.head.appendChild(s);
    });
  }
  return __routeLoads[route];
}'''

LIVE_SHIM = '''async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}'''


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "r").mkdir(parents=True)

    html = (ROOT / "static" / "index.html").read_text()
    html = html.replace('<script src="/static/app.js"></script>',
                        '<script src="core.js"></script>\n<script src="app.js"></script>')
    html = html.replace(
        "see the README for how to plug in real fare data (Kaggle compaction or a collector).",
        "see the repo README for how to plug in real fare data. This is a static snapshot "
        "(all models precomputed; searches run client-side on the embedded quotes).")
    (SITE / "index.html").write_text(html)

    js = (ROOT / "static" / "app.js").read_text()
    assert LIVE_SHIM in js, "api() shim anchor not found"
    (SITE / "app.js").write_text(js.replace(LIVE_SHIM, API_SHIM))

    s = get_store()
    core = {"/api/meta": s.meta(), "/api/oil": s.oil_api(),
            "/api/correlations": s.correlations,
            "/api/booking_curve": s.booking_curve_api(None, None),
            "/api/backtest": s.backtest_api()}
    with open(SITE / "core.js", "w") as fh:
        fh.write("window.__FC_DATA = ")
        json.dump(core, fh, separators=(",", ":"))
        fh.write(";\n")

    n_payloads = len(core)
    for r in core["/api/meta"]["routes"]:
        o, d = r["origin"], r["dest"]
        q = s.quotes(o, d)
        chunk = {f"/api/quotes?origin={o}&dest={d}": q}
        for a in r["airlines"]:
            fc = s.forecast(o, d, a)
            # the chart plots the smoothed index and the fit; the raw daily
            # series is never drawn, so it is not worth shipping
            fc["history"].pop("fare", None)
            chunk[f"/api/forecast?origin={o}&dest={d}&airline={a}"] = fc
        # observed price history for the flights the default lists surface
        tops = (sorted(q["quotes"], key=lambda x: x["fare"])[:8] +
                sorted(q["quotes"], key=lambda x: x["predicted_min"])[:8])
        seen = set()
        for f in tops:
            k = (f["airline"], f["flight_date"])
            if k in seen:
                continue
            seen.add(k)
            t = s.trajectory_api(o, d, f["airline"], f["flight_date"])
            if "error" not in t:
                chunk[f"/api/trajectory?origin={o}&dest={d}&airline={f['airline']}"
                      f"&flight_date={f['flight_date']}"] = t
        with open(SITE / "r" / f"{o}-{d}.js", "w") as fh:
            fh.write("Object.assign(window.__FC_DATA, ")
            json.dump(chunk, fh, separators=(",", ":"))
            fh.write(");\n")
        n_payloads += len(chunk)

    files = list(SITE.rglob("*"))
    total = sum(p.stat().st_size for p in files if p.is_file())
    core_kb = (SITE / "core.js").stat().st_size / 1e3
    chunks = [p.stat().st_size for p in (SITE / "r").iterdir()]
    print(f"site/ built: {n_payloads} payloads across {len(files)} files, "
          f"{total/1e6:.1f} MB total")
    print(f"  first load: core.js {core_kb:.0f} KB + app.js; "
          f"route chunk median {sorted(chunks)[len(chunks)//2]/1e3:.0f} KB, "
          f"max {max(chunks)/1e3:.0f} KB")


if __name__ == "__main__":
    main()
