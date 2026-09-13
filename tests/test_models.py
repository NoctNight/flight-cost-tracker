"""Model validation against known synthetic ground truth + API smoke tests."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import synthetic
from app.main import app
from app.store import get_store


@pytest.fixture(scope="session")
def store():
    return get_store()


@pytest.fixture(scope="session")
def client(store):
    return TestClient(app)


def test_oil_lag_recovered(store):
    """The lag scan should land near the true generative lag (45d)."""
    a = store.oil_analysis
    assert abs(a["best_lag_days"] - synthetic.TRUE_OIL_LAG_DAYS) <= 20
    assert a["elasticity"] is not None and a["elasticity"] > 0


def test_correlation_structure(store):
    """Airlines on the same route must co-move far more than unrelated pairs."""
    c = store.correlations
    assert c["avg_same_route"] > c["avg_unrelated"] + 0.3
    n = len(c["labels"])
    assert len(c["matrix"]) == n and len(c["matrix"][0]) == n


def test_booking_curve_shape(store):
    """Last-minute fares must be pricier than the 30d reference."""
    curve = store.curve
    v = curve.value(np.array([1, 7, 30, 60]))
    assert v[0] > v[2] * 1.15      # day-before premium
    assert v[1] > v[2]             # 1 week out > 1 month out
    assert abs(v[2] - 1.0) < 0.1   # normalised near 1 at 30d


def test_forecast_sane(store):
    f = store.forecast("JFK", "LAX", "DL")
    assert "error" not in f
    m = f["forecast"]["mean"]
    assert len(m) == 120
    hist_med = np.median([v for v in f["history"]["fare"] if v])
    assert all(0.3 * hist_med < v < 3 * hist_med for v in m)
    assert all(l < mu < h for l, mu, h in
               zip(f["forecast"]["lo"], m, f["forecast"]["hi"]))


def test_search_and_trajectory(store):
    r = store.search("JFK", "LAX")
    assert r["cheapest_now"] and r["cheapest_eventually"]
    assert r["cheapest_now"] == sorted(r["cheapest_now"], key=lambda x: x["fare"])
    top = r["cheapest_eventually"][0]
    assert top["predicted_min"] <= top["fare_now"] + 0.01
    t = store.trajectory_api("JFK", "LAX", top["airline"], top["flight_date"])
    assert "error" not in t
    assert len(t["predicted"]["dates"]) == len(t["predicted"]["mean"])


def test_api_endpoints(client):
    for path in ["/api/meta", "/api/search?origin=JFK&dest=LAX",
                 "/api/forecast?origin=JFK&dest=LAX&airline=DL",
                 "/api/correlations", "/api/oil", "/api/booking_curve", "/"]:
        r = client.get(path)
        assert r.status_code == 200, path


def test_unknown_route(client):
    r = client.get("/api/search?origin=XXX&dest=YYY")
    assert r.status_code == 200 and "error" in r.json()
