"""Flight cost tracker API + dashboard server."""
from __future__ import annotations

from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .store import get_store

STATIC = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_store()  # warm model cache
    yield


app = FastAPI(title="Flight Cost Tracker", lifespan=lifespan)


@app.get("/api/meta")
def meta():
    return get_store().meta()


@app.get("/api/search")
def search(origin: str = Query(..., min_length=3, max_length=3),
           dest: str = Query(..., min_length=3, max_length=3),
           flight_date: str | None = None,
           window: int = 90):
    return get_store().search(origin.upper(), dest.upper(), flight_date, window)


@app.get("/api/forecast")
def forecast(origin: str, dest: str, airline: str, horizon: int = 120):
    return get_store().forecast(origin.upper(), dest.upper(), airline.upper(), horizon)


@app.get("/api/quotes")
def quotes(origin: str, dest: str):
    return get_store().quotes(origin.upper(), dest.upper())


@app.get("/api/trajectory")
def trajectory(origin: str, dest: str, airline: str, flight_date: str):
    return get_store().trajectory_api(origin.upper(), dest.upper(),
                                      airline.upper(), flight_date)


@app.get("/api/correlations")
def correlations():
    return get_store().correlations


@app.get("/api/oil")
def oil():
    return get_store().oil_api()


@app.get("/api/backtest")
def backtest():
    return get_store().backtest_api()


@app.get("/api/booking_curve")
def booking_curve(origin: str | None = None, dest: str | None = None):
    return get_store().booking_curve_api(origin, dest)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
