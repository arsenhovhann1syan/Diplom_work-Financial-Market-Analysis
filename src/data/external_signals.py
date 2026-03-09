# ============================================================
# src/data/external_signals.py — Fear & Greed | S&P 500 | VIX
# ============================================================

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

from config import DATA_START


def get_external_signals(
    start_str: str = DATA_START,
    end_str:   str = None,
) -> pd.DataFrame:
    """
    Fetches and aligns three external market signals onto a full
    daily calendar:
      - Fear & Greed Index  (alternative.me API)
      - S&P 500 daily return (yfinance: ^GSPC)
      - VIX closing level    (yfinance: ^VIX)
    """
    if end_str is None:
        end_str = datetime.today().strftime("%Y-%m-%d")

    date_range = pd.date_range(start=start_str, end=end_str, freq="D")

    print("=" * 60)
    print("FETCHING EXTERNAL SIGNALS")
    print(f"Period : {start_str}  →  {end_str}")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Fear & Greed
    # --------------------------------------------------------
    print("\n[1/3] Fear & Greed Index  (alternative.me) ...")

    fg_resp = requests.get(
        "https://api.alternative.me/fng/?limit=0&format=json",
        timeout=15,
    ).json()

    if "data" not in fg_resp:
        raise RuntimeError("Fear & Greed API returned unexpected response.")

    fg_df = (
        pd.DataFrame(fg_resp["data"])[["timestamp", "value"]]
        .assign(
            timestamp  = lambda x: pd.to_datetime(
                x["timestamp"].astype(int), unit="s"
            ).dt.normalize(),
            fear_greed = lambda x: x["value"].astype(float),
        )
        [["timestamp", "fear_greed"]]
        .drop_duplicates(subset="timestamp")
        .set_index("timestamp")
        .sort_index()
    )

    print(f"   ✅ {len(fg_df):,} obs  |  "
          f"{fg_df.index.min().date()} → {fg_df.index.max().date()}")

    # --------------------------------------------------------
    # 2. S&P 500
    # --------------------------------------------------------
    print("\n[2/3] S&P 500  (^GSPC, yfinance) ...")

    spx_raw = yf.download(
        "^GSPC", start=start_str, end=end_str,
        progress=False, auto_adjust=True,
    )

    if spx_raw.empty:
        raise RuntimeError("yfinance returned no data for ^GSPC.")

    spx_raw.columns = [
        "_".join(c).strip() if isinstance(c, tuple) else c
        for c in spx_raw.columns
    ]
    close_col = next(c for c in spx_raw.columns if "close" in c.lower())

    spx_df = (
        spx_raw[[close_col]]
        .rename(columns={close_col: "spx_close"})
        .assign(spx_return=lambda x: x["spx_close"].pct_change())
        [["spx_return"]]
    )
    spx_df.index = pd.to_datetime(spx_df.index).normalize()
    spx_df.index.name = "timestamp"

    print(f"   ✅ {len(spx_df):,} obs  |  "
          f"{spx_df.index.min().date()} → {spx_df.index.max().date()}")

    # --------------------------------------------------------
    # 3. VIX
    # --------------------------------------------------------
    print("\n[3/3] VIX  (^VIX, yfinance) ...")

    vix_raw = yf.download(
        "^VIX", start=start_str, end=end_str,
        progress=False, auto_adjust=True,
    )

    if vix_raw.empty:
        raise RuntimeError("yfinance returned no data for ^VIX.")

    vix_raw.columns = [
        "_".join(c).strip() if isinstance(c, tuple) else c
        for c in vix_raw.columns
    ]
    close_col = next(c for c in vix_raw.columns if "close" in c.lower())

    vix_df = vix_raw[[close_col]].rename(columns={close_col: "vix"})
    vix_df.index = pd.to_datetime(vix_df.index).normalize()
    vix_df.index.name = "timestamp"

    print(f"   ✅ {len(vix_df):,} obs  |  "
          f"{vix_df.index.min().date()} → {vix_df.index.max().date()}")

    # --------------------------------------------------------
    # Merge + fill
    # --------------------------------------------------------
    print("\n[4/4] Merging signals onto full daily calendar ...")

    signals = (
        pd.DataFrame(index=date_range)
        .rename_axis("timestamp")
        .join(fg_df,  how="left")
        .join(spx_df, how="left")
        .join(vix_df, how="left")
        .loc[start_str:end_str]
    )

    signals["fear_greed"] = signals["fear_greed"].ffill(limit=1)
    signals["spx_return"] = signals["spx_return"].fillna(0.0)
    signals["vix"]        = signals["vix"].ffill(limit=3)

    # --------------------------------------------------------
    # Integrity report
    # --------------------------------------------------------
    missing = signals.isnull().sum()

    print("\n" + "=" * 60)
    print("EXTERNAL SIGNALS — INTEGRITY REPORT")
    print("=" * 60)
    print(f"  Shape      : {signals.shape}")
    print(f"  Date range : {signals.index.min().date()} → "
          f"{signals.index.max().date()}")
    print(f"\n  Remaining NaNs (after fill policy):")
    for col, n in missing.items():
        status = "✅" if n == 0 else "⚠️ "
        print(f"    {status} {col:<15} : {n}")

    if missing.any():
        print("\n  ⚠️  Residual NaNs. Investigate before feature engineering.")
    else:
        print("\n  ✅ No missing values. Signals ready for feature engineering.")

    print("=" * 60)
    return signals