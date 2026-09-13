"""Compact the Kaggle dilwong/flightprices dump (~31 GB CSV) into the small
observation table this app uses (data/fares.parquet).

Run this on the machine where you downloaded the Kaggle file:
    kaggle datasets download -d dilwong/flightprices
    python scripts/compact_kaggle.py itineraries.csv

Keeps one row per (search_date, flight_date, route, airline): the minimum
total fare across itineraries (non-stop preferred if present).
"""
import sys
from pathlib import Path

import pandas as pd

USECOLS = ["searchDate", "flightDate", "startingAirport", "destinationAirport",
           "totalFare", "isNonStop", "segmentsAirlineCode"]
OUT = Path(__file__).resolve().parent.parent / "data" / "fares.parquet"


def main(path: str, chunksize: int = 2_000_000) -> None:
    parts = []
    for i, chunk in enumerate(pd.read_csv(path, usecols=USECOLS, chunksize=chunksize)):
        chunk = chunk[chunk["isNonStop"] == True]  # noqa: E712 - single carrier, clean airline attribution
        chunk["airline"] = chunk["segmentsAirlineCode"].str.split("|").str[0]
        g = (chunk.groupby(["searchDate", "flightDate", "startingAirport",
                            "destinationAirport", "airline"])["totalFare"]
                  .min().reset_index())
        parts.append(g)
        print(f"chunk {i}: {len(chunk):,} nonstop rows -> {len(g):,} groups")
    df = pd.concat(parts, ignore_index=True)
    df = (df.groupby(["searchDate", "flightDate", "startingAirport",
                      "destinationAirport", "airline"])["totalFare"]
            .min().reset_index())
    df.columns = ["search_date", "flight_date", "origin", "dest", "airline", "fare"]
    df["search_date"] = pd.to_datetime(df["search_date"])
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    df["source"] = "kaggle"
    df.to_parquet(OUT, index=False)
    print(f"wrote {len(df):,} rows -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
