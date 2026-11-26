"""Download Binance Vision monthly 1m klines for BTCUSDT and cache as pickle."""
import io
import pickle
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
CACHE = BASE / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
MONTHS = [f"{y}-{m:02d}" for y in range(2023, 2027) for m in range(1, 13)]
MONTHS = [m for m in MONTHS if m <= "2026-10"]

COLS = ["ts", "open", "high", "low", "close", "volume"]


def fetch(month: str) -> pd.DataFrame:
    url = (
        "https://data.binance.vision/data/spot/monthly/klines/"
        f"{SYMBOL}/1m/{SYMBOL}-1m-{month}.zip"
    )
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 200:
                break
            if r.status_code == 404:
                raise FileNotFoundError(f"{month} not published")
            time.sleep(2 * (attempt + 1))
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"failed to download {month}: {r.status_code}")

    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None)
    df = df[pd.to_numeric(df[0], errors="coerce").notna()].iloc[:, :6]
    df.columns = COLS
    raw_ts = df["ts"].astype("int64")
    unit = "us" if raw_ts.iloc[0] > 10**14 else "ms"
    df["ts"] = pd.to_datetime(raw_ts, unit=unit, utc=True)
    for c in COLS[1:]:
        df[c] = df[c].astype(float)
    return df.set_index("ts").sort_index()


def main() -> None:
    for month in MONTHS:
        out = CACHE / f"{SYMBOL}-1m-{month}.pkl"
        if out.exists():
            continue
        df = fetch(month)
        with out.open("wb") as fh:
            pickle.dump(df, fh, protocol=5)
        print(f"{month}: {len(df):>6} rows -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
