# ============================================================
# src/data/validation.py — Data validation and cleaning 
# ============================================================

import pandas as pd


def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("DATA INTEGRITY AUDIT")
    print("=" * 60)

    initial_rows = len(df)
    print(f"Initial rows : {initial_rows:,}")
    print(f"Date range   : {df['timestamp'].min()} → {df['timestamp'].max()}")

    # --------------------------------------------------
    # 1. Remove duplicates
    # --------------------------------------------------
    duplicates = df.duplicated(subset=["timestamp"]).sum()
    if duplicates > 0:
        print(f"Removing {duplicates} duplicate timestamps.")
        df = df.drop_duplicates(subset=["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------
    # 2. OHLC logical consistency
    # --------------------------------------------------
    invalid_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
    invalid_low  = (df["low"]  > df[["open", "close"]].min(axis=1)).sum()

    if invalid_high > 0 or invalid_low > 0:
        print(f"Correcting {invalid_high + invalid_low} OHLC inconsistencies.")
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"]  = df[["open", "low",  "close"]].min(axis=1)
    else:
        print("OHLC relationships valid.")

    # --------------------------------------------------
    # 3. Gap detection (no interpolation)
    # --------------------------------------------------
    df_indexed   = df.set_index("timestamp")
    full_range   = pd.date_range(df_indexed.index.min(),
                                  df_indexed.index.max(), freq="1D")
    missing_days = len(full_range) - len(df_indexed)

    if missing_days > 0:
        print(f"WARNING: {missing_days} missing daily observations detected.")
        print("Rows kept as-is (no interpolation).")
    else:
        print("No missing daily timestamps detected.")

    df = df_indexed.reset_index()

    # --------------------------------------------------
    # 4. Return diagnostics
    # --------------------------------------------------
    df["returns"] = df["close"].pct_change()

    print("\nReturn distribution summary:")
    print(df["returns"].describe())
    print("=" * 60)
    print(f"Final rows : {len(df):,}")
    print("=" * 60)

    return df