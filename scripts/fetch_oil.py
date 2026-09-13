"""Refresh daily Brent/WTI spot prices (EIA data, public domain) from the
datasets/oil-prices mirror on GitHub."""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/datasets/oil-prices/main/data"
DATA = Path(__file__).resolve().parent.parent / "data"

for name in ("brent-daily.csv", "wti-daily.csv"):
    url = f"{BASE}/{name}"
    dest = DATA / name
    print(f"{url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"  {dest.stat().st_size} bytes")
