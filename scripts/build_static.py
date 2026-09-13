"""Build the static dashboard snapshot into site/ (for Vercel or any static
host). All API payloads are precomputed and embedded; search, return combos
and trajectories run client-side.

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

def main() -> None:
    SITE.mkdir(exist_ok=True)

    html = (ROOT / "static" / "index.html").read_text()
    html = html.replace('<script src="/static/app.js"></script>',
                        '<script src="data.js"></script>\n<script src="app.js"></script>')
    html = html.replace(
        "see the README for how to plug in real fare data (Kaggle compaction or a collector).",
        "see the repo README for how to plug in real fare data. This is a static snapshot "
        "(all models precomputed; searches run client-side on the embedded quotes).")
    (SITE / "index.html").write_text(html)

    js = (ROOT / "static" / "app.js").read_text()
    old_api = '''async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}'''
    assert old_api in js, "api() shim anchor not found"
    js = js.replace(old_api, '''async function api(path) {
  const d = window.__FC_DATA[path];
  if (!d) throw new Error("not in this snapshot: " + path);
  return d;
}''')
    (SITE / "app.js").write_text(js)

    s = get_store()
    D = {"/api/meta": s.meta(), "/api/oil": s.oil_api(),
         "/api/correlations": s.correlations,
         "/api/booking_curve": s.booking_curve_api(None, None),
         "/api/backtest": s.backtest_api()}
    for r in D["/api/meta"]["routes"]:
        o, d = r["origin"], r["dest"]
        q = s.quotes(o, d)
        D[f"/api/quotes?origin={o}&dest={d}"] = q
        tops = (sorted(q["quotes"], key=lambda x: x["fare"])[:12] +
                sorted(q["quotes"], key=lambda x: x["predicted_min"])[:12])
        seen = set()
        for f in tops:
            k = (f["airline"], f["flight_date"])
            if k in seen:
                continue
            seen.add(k)
            t = s.trajectory_api(o, d, f["airline"], f["flight_date"])
            if "error" not in t:
                D[f"/api/trajectory?origin={o}&dest={d}&airline={f['airline']}&flight_date={f['flight_date']}"] = t
        for a in r["airlines"]:
            D[f"/api/forecast?origin={o}&dest={d}&airline={a}"] = s.forecast(o, d, a)

    with open(SITE / "data.js", "w") as fh:
        fh.write("window.__FC_DATA = ")
        json.dump(D, fh, separators=(",", ":"))
        fh.write(";\n")

    total = sum(p.stat().st_size for p in SITE.iterdir())
    print(f"site/ built: {len(D)} payloads, {total/1e6:.1f} MB total")


if __name__ == "__main__":
    main()
