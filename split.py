# ============================================================
# src/pipeline/split.py — train + expanding test split
# ============================================================

import pandas as pd
import numpy as np

from Config import TRAIN_END_DATE, TEST_START_DATE, THRESHOLD_PERCENTILE


def apply_labels(future_ret: pd.Series, threshold: float) -> pd.Series:
    """
    Directional three-class labels:
      +1 = strong up  (future_return >  +threshold)
       0 = neutral    (|future_return| <= threshold)
      -1 = strong down (future_return < -threshold)
    """
    labels = pd.Series(0, index=future_ret.index, dtype=int)
    labels[future_ret >  threshold] =  1
    labels[future_ret < -threshold] = -1
    return labels


def train_test_split_pipeline(
    df:                     pd.DataFrame,
    threshold_percentile:   float = THRESHOLD_PERCENTILE,
    train_end:              str   = TRAIN_END_DATE,
    test_start:             str   = TEST_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:

    df = df.sort_index()

    assert "future_return" in df.columns, \
        "future_return column missing. Run feature engineering first."

    train_df = df[df.index <= train_end].copy()
    test_df  = df[df.index >= test_start].copy()

    assert len(train_df) > 0, "Train set is empty — check TRAIN_END_DATE."
    assert len(test_df)  > 0, "Test set is empty — check TEST_START_DATE."

    # Drop absolute price features
    for col in ["vwap_7d"]:
        train_df = train_df.drop(columns=[col], errors="ignore")
        test_df  = test_df.drop(columns=[col],  errors="ignore")

    # Threshold from train only
    threshold = train_df["future_return"].abs().quantile(
        threshold_percentile / 100
    )

    y_train = apply_labels(train_df["future_return"], threshold)
    y_test  = apply_labels(test_df["future_return"],  threshold)

    X_train = train_df.drop(columns=["future_return"])
    X_test  = test_df.drop(columns=["future_return"])

    print("=" * 60)
    print("FIXED TRAIN + EXPANDING TEST — INTEGRITY REPORT")
    print("=" * 60)
    print(f"Train window  : {X_train.index.min().date()} → {X_train.index.max().date()}")
    print(f"Test  window  : {X_test.index.min().date()} → {X_test.index.max().date()}")
    print(f"Train samples : {len(X_train):,}")
    print(f"Test samples  : {len(X_test):,}")
    print(f"DTA threshold : ±{threshold*100:.4f}%  (p{threshold_percentile})")
    print("=" * 60)
    print("✅ Train frozen. Test expands with new data.")
    print("=" * 60)

    return X_train, X_test, y_train, y_test, threshold