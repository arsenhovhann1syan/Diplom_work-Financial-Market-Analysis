# ============================================================
# src/features/engineering.py — Feature engineering pipeline
# ============================================================

import numpy as np
import pandas as pd


def engineer_features_ml_ready(
    df_cleaned:  pd.DataFrame,
    df_external: pd.DataFrame,
) -> pd.DataFrame:


    keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df_feat   = df_cleaned[keep_cols].copy().set_index("timestamp").sort_index()

    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print(f"Input rows : {len(df_feat):,}")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. External signals (raw, unlagged — lag applied later)
    # --------------------------------------------------------
    print("\n[1/9] External signals ...")
    df_feat = df_feat.join(df_external, how="left")
    df_feat["fear_greed"] = df_feat["fear_greed"].ffill(limit=1)
    df_feat["spx_return"] = df_feat["spx_return"].fillna(0.0)
    df_feat["vix"]        = df_feat["vix"].ffill(limit=3)

    assert df_feat[["fear_greed", "spx_return", "vix"]].isnull().sum().sum() <= 1, \
        "Unexpected NaNs in external signals after fill."

    # --------------------------------------------------------
    # 2. Momentum
    # --------------------------------------------------------
    print("[2/9] Momentum features ...")
    for w in [1, 2, 3, 5, 7, 14, 21, 30, 60]:
        df_feat[f"momentum_{w}d"] = (
            np.log(df_feat["close"] / df_feat["close"].shift(w)).shift(1)
        )

    # --------------------------------------------------------
    # 3. Volatility
    # --------------------------------------------------------
    print("[3/9] Volatility features ...")
    park_const = 1 / (4 * np.log(2))
    hl_ratio   = np.log(df_feat["high"] / df_feat["low"])
    park_daily = np.sqrt(park_const * hl_ratio ** 2)

    log_hl   = np.log(df_feat["high"] / df_feat["low"])
    log_co   = np.log(df_feat["close"] / df_feat["open"])
    gk_daily = np.sqrt(0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2)

    for w in [7, 14, 21, 30, 60]:
        df_feat[f"vol_parkinson_{w}d"] = park_daily.rolling(w).mean().shift(1)

    for w in [14, 30]:
        df_feat[f"vol_gk_{w}d"] = gk_daily.rolling(w).mean().shift(1)

    log_returns = np.log(df_feat["close"] / df_feat["close"].shift(1))
    df_feat["volatility_30d"] = log_returns.rolling(30).std().shift(1)

    # --------------------------------------------------------
    # 4. Technical indicators
    # --------------------------------------------------------
    print("[4/9] Technical indicators ...")

    for period in [14, 28]:
        delta = df_feat["close"].diff()
        gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss  = (-delta).where(delta < 0, 0.0).rolling(period).mean()
        rs    = gain / (loss + 1e-9)
        df_feat[f"rsi_{period}"] = (100 - 100 / (1 + rs)).shift(1)

    for fast, slow in [(12, 26), (20, 50)]:
        ema_fast = df_feat["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df_feat["close"].ewm(span=slow, adjust=False).mean()
        df_feat[f"macd_{fast}_{slow}"]  = (ema_fast - ema_slow).shift(1)
        df_feat[f"price_to_ema_{slow}"] = ((df_feat["close"] / ema_slow) - 1).shift(1)

    for period in [20, 50]:
        sma = df_feat["close"].rolling(period).mean()
        std = df_feat["close"].rolling(period).std()
        df_feat[f"bb_zscore_{period}"] = (
            (df_feat["close"] - sma) / (std + 1e-9)
        ).shift(1)

    ma_short = df_feat["close"].rolling(20).mean()
    ma_long  = df_feat["close"].rolling(50).mean()
    df_feat["ma_cross"] = ((ma_short / ma_long) - 1).shift(1)

    # --------------------------------------------------------
    # 5. Volume
    # --------------------------------------------------------
    print("[5/9] Volume features ...")
    for w in [7, 14, 30]:
        vol_ma = df_feat["volume"].rolling(w).mean()
        df_feat[f"volume_ratio_{w}d"] = (df_feat["volume"] / (vol_ma + 1e-9)).shift(1)

    df_feat["vwap_7d"] = (
        (df_feat["close"] * df_feat["volume"]).rolling(7).sum()
        / (df_feat["volume"].rolling(7).sum() + 1e-9)
    ).shift(1)

    df_feat["price_to_vwap"] = (
        (df_feat["close"] / (
            (df_feat["close"] * df_feat["volume"]).rolling(7).sum()
            / (df_feat["volume"].rolling(7).sum() + 1e-9)
        ) - 1)
    ).shift(1)

    df_feat["volume_trend"] = (
        df_feat["volume"].rolling(7).mean()
        / (df_feat["volume"].rolling(30).mean() + 1e-9)
    ).shift(1)

    # --------------------------------------------------------
    # 6. Price action
    # --------------------------------------------------------
    print("[6/9] Price action features ...")
    df_feat["body_ratio"] = (
        np.abs(df_feat["close"] - df_feat["open"]) / (df_feat["open"] + 1e-9)
    ).shift(1)
    df_feat["hl_range"] = (
        (df_feat["high"] - df_feat["low"]) / (df_feat["close"] + 1e-9)
    ).shift(1)

    # --------------------------------------------------------
    # 7. Time
    # --------------------------------------------------------
    print("[7/9] Time features ...")
    df_feat["day_of_week"] = df_feat.index.dayofweek
    df_feat["month"]       = df_feat.index.month
    df_feat["quarter"]     = df_feat.index.quarter

    # --------------------------------------------------------
    # 8. Regime (vol spike)
    # --------------------------------------------------------
    print("[8/9] Regime features ...")
    df_feat["vol_spike"] = (
        df_feat["vol_parkinson_30d"] > df_feat["vol_parkinson_60d"] * 1.5
    ).astype(int)

    # --------------------------------------------------------
    # 9. External signal features (lag applied here)
    # --------------------------------------------------------
    print("[9/9] External signal features ...")

    df_feat["fg_level"]         = df_feat["fear_greed"].shift(1)
    df_feat["fg_momentum"]      = (
        df_feat["fear_greed"] - df_feat["fear_greed"].shift(7)
    ).shift(1)
    df_feat["fg_extreme_fear"]  = (df_feat["fear_greed"] < 25).astype(int).shift(1)
    df_feat["fg_extreme_greed"] = (df_feat["fear_greed"] > 75).astype(int).shift(1)

    df_feat["vix_level"]  = df_feat["vix"].shift(1)
    df_feat["vix_change"] = (df_feat["vix"] - df_feat["vix"].shift(7)).shift(1)
    df_feat["vix_spike"]  = (
        df_feat["vix"] > df_feat["vix"].rolling(30).mean() * 1.5
    ).astype(int).shift(1)

    df_feat["spx_return_lag1"]   = df_feat["spx_return"].shift(1)
    df_feat["spx_5d_return"]     = df_feat["spx_return"].rolling(5).sum().shift(1)
    df_feat["spx_negative_flag"] = (df_feat["spx_return"] < -0.02).astype(int).shift(1)

    # --------------------------------------------------------
    # Target: raw log-return for next day
    # --------------------------------------------------------
    print("\n[TARGET] Computing raw future return ...")
    df_feat["future_return"] = np.log(
        df_feat["close"].shift(-1) / df_feat["close"]
    )
    print("   ✅ future_return stored as raw log-return.")
    print("   ⚠️  Classification threshold applied per training fold.")

    # --------------------------------------------------------
    # Cleanup — drop raw OHLCV and unlagged external columns
    # --------------------------------------------------------
    drop_cols = [
        "open", "high", "low", "close", "volume",
        "fear_greed", "spx_return", "vix",
    ]
    df_feat = df_feat.drop(columns=drop_cols)

    initial_rows = len(df_feat)
    df_feat = df_feat.dropna(
        subset=[c for c in df_feat.columns if c != "future_return"]
    )
    df_feat = df_feat.dropna(subset=["future_return"])

    # --------------------------------------------------------
    # Integrity report
    # --------------------------------------------------------
    feature_cols = [c for c in df_feat.columns if c != "future_return"]

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING — INTEGRITY REPORT")
    print("=" * 60)
    print(f"  Input rows     : {initial_rows:,}")
    print(f"  Output rows    : {len(df_feat):,}")
    print(f"  Dropped rows   : {initial_rows - len(df_feat):,}")
    print(f"  Feature count  : {len(feature_cols)}")
    print(f"  Remaining NaNs : {df_feat.isnull().sum().sum()}")
    print(f"  Date range     : {df_feat.index.min().date()} → "
          f"{df_feat.index.max().date()}")
    print(f"\n  future_return stats :")
    print(f"    mean = {df_feat['future_return'].mean():.6f}")
    print(f"    std  = {df_feat['future_return'].std():.6f}")
    print(f"    min  = {df_feat['future_return'].min():.6f}")
    print(f"    max  = {df_feat['future_return'].max():.6f}")
    print("=" * 60)
    print("✅ Feature matrix ready.")
    print("=" * 60)

    return df_feat