# ============================================================
# src/data/download.py — Fetch historical OHLCV from Binance
# ============================================================

import time
import pandas as pd
import requests

from config import SYMBOL, INTERVAL, DATA_START, RAW_DATA_PATH


def get_binance_data(
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    start_str: str = DATA_START,
) -> pd.DataFrame:

    url      = "https://api.binance.com/api/v3/klines"
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    all_data = []

    print("=" * 60)
    print("FETCHING BITCOIN DATA FROM BINANCE")
    print("=" * 60)
    print(f"Symbol   : {symbol}")
    print(f"Interval : {interval}  (DAILY – better for ML)")
    print(f"Start    : {start_str}")
    print("\nDownloading...")

    while True:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": start_ts,
            "limit":     1000,
        }
        response = requests.get(url, params=params)
        res      = response.json()

        if not res:
            break

        df_chunk = pd.DataFrame(res).iloc[:, :6]
        all_data.append(df_chunk)

        start_ts = res[-1][0] + 1

        if len(res) < 1000:
            break

        time.sleep(0.2)

    df          = pd.concat(all_data)
    df.columns  = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    print(f"\n✅ Download complete!")
    print(f"   Total days  : {len(df):,}")
    print(f"   Date range  : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print("=" * 60)

    return df.reset_index(drop=True)


def load_or_download(path: str = RAW_DATA_PATH, **kwargs) -> pd.DataFrame:
    """
    Loads cached CSV if it exists, otherwise downloads from Binance and saves.
    """
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        print(f"✅ Loaded cached data from {path}  ({len(df):,} rows)")
    except FileNotFoundError:
        df = get_binance_data(**kwargs)
        df.to_csv(path, index=False)
        print(f"✅ Data saved to {path}")
    return df