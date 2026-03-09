# ============================================================
# src/features/selection.py — Feature selection (corr + RF)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from config import (
    CORRELATION_THRESHOLD,
    MDI_THRESHOLD,
    PI_THRESHOLD,
    FEATURE_SEL_VAL_FRAC,
    FEATURE_SEL_RF_TREES,
    FEATURE_SEL_RF_DEPTH,
    FEATURE_SEL_RF_MIN_LEAF,
    RANDOM_STATE,
)


# ============================================================
# Stage 1 — Correlation-based reduction
# ============================================================

def correlation_feature_selection(
    X_train:   pd.DataFrame,
    X_test:    pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Removes linearly redundant features using Pearson |r| computed
    on the training set only.  Dropping is applied identically to
    the test set.

    Returns
    -------
    X_train_reduced, X_test_reduced, dropped_features
    """
    print("=" * 60)
    print("CORRELATION FEATURE REDUCTION")
    print(f"Threshold : |r| > {threshold}")
    print("=" * 60)

    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    X_train_reduced = X_train.drop(columns=to_drop)
    X_test_reduced  = X_test.drop(columns=to_drop)

    print(f"\n  Features before  : {X_train.shape[1]}")
    print(f"  Features dropped : {len(to_drop)}")
    print(f"  Features after   : {X_train_reduced.shape[1]}")

    if to_drop:
        print(f"\n  Dropped features:")
        for f in to_drop:
            print(f"    - {f}")

    print(f"\n  Correlated pairs (|r| > {threshold}):")
    pairs_found = False
    for col in upper.columns:
        for partner in upper.index[upper[col] > threshold]:
            print(f"    {partner}  ↔  {col}  "
                  f"|r|={corr_matrix.loc[partner, col]:.4f}  → dropped: {col}")
            pairs_found = True
    if not pairs_found:
        print("    None found above threshold.")

    return X_train_reduced, X_test_reduced, to_drop


# ============================================================
# Stage 2 — Tree-based feature selection (MDI + PI)
# ============================================================

def tree_based_feature_selection(
    X_train:       pd.DataFrame,
    X_test:        pd.DataFrame,
    y_train:       pd.Series,
    mdi_threshold: float = MDI_THRESHOLD,
    pi_threshold:  float = PI_THRESHOLD,
    val_fraction:  float = FEATURE_SEL_VAL_FRAC,
    random_state:  int   = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, list, pd.DataFrame]:

    print("=" * 60)
    print("TREE-BASED FEATURE SELECTION  (Random Forest)")
    print(f"  MDI threshold : >= {mdi_threshold}")
    print(f"  PI  threshold : >= {pi_threshold}")
    print(f"  Val fraction  : {val_fraction}")
    print("=" * 60)

    val_size = int(len(X_train) * val_fraction)
    X_tr     = X_train.iloc[:-val_size]
    y_tr     = y_train.iloc[:-val_size]
    X_val    = X_train.iloc[-val_size:]
    y_val    = y_train.iloc[-val_size:]

    print(f"\n  Train slice : {len(X_tr):,} "
          f"({X_tr.index.min().date()} → {X_tr.index.max().date()})")
    print(f"  Val slice   : {len(X_val):,} "
          f"({X_val.index.min().date()} → {X_val.index.max().date()})")

    print("\n  Fitting Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators     = FEATURE_SEL_RF_TREES,
        max_depth        = FEATURE_SEL_RF_DEPTH,
        min_samples_leaf = FEATURE_SEL_RF_MIN_LEAF,
        max_features     = "sqrt",
        class_weight     = "balanced",
        random_state     = random_state,
        n_jobs           = -1,
    )
    rf.fit(X_tr, y_tr)

    mdi_scores = pd.Series(
        rf.feature_importances_, index=X_train.columns, name="mdi_importance"
    ).sort_values(ascending=False)

    print("  Computing permutation importance on val slice ...")
    pi_result = permutation_importance(
        rf, X_val, y_val,
        n_repeats=20, random_state=random_state, n_jobs=-1,
    )
    pi_scores = pd.Series(
        pi_result.importances_mean, index=X_train.columns, name="pi_importance"
    )
    pi_std    = pd.Series(
        pi_result.importances_std,  index=X_train.columns, name="pi_std"
    )

    importance_df = pd.DataFrame({
        "mdi_importance": mdi_scores,
        "pi_importance":  pi_scores,
        "pi_std":         pi_std,
    }).sort_values("mdi_importance", ascending=False)

    importance_df["mdi_pass"] = importance_df["mdi_importance"] >= mdi_threshold
    importance_df["pi_pass"]  = importance_df["pi_importance"]  >= pi_threshold
    importance_df["retained"] = importance_df["mdi_pass"] & importance_df["pi_pass"]

    selected_features = importance_df[importance_df["retained"]].index.tolist()
    dropped_features  = importance_df[~importance_df["retained"]].index.tolist()

    X_train_final = X_train[selected_features]
    X_test_final  = X_test[selected_features]

    # Report
    print("\n" + "=" * 60)
    print("FEATURE SELECTION — IMPORTANCE REPORT")
    print("=" * 60)
    print(f"\n  {'Feature':<25} {'MDI':>8} {'PI':>8} {'PI+/-':>7} "
          f"{'MDI_OK':>6} {'PI_OK':>5} {'Keep':>6}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*5} {'-'*6}")

    for feat, row in importance_df.iterrows():
        print(f"  {feat:<25} "
              f"{row['mdi_importance']:>8.4f} "
              f"{row['pi_importance']:>8.4f} "
              f"+/-{row['pi_std']:>6.4f} "
              f"{'YES' if row['mdi_pass'] else 'NO':>6} "
              f"{'YES' if row['pi_pass'] else 'NO':>5} "
              f"{'YES' if row['retained'] else 'NO':>6}")

    print(f"\n  Features before  : {X_train.shape[1]}")
    print(f"  Features dropped : {len(dropped_features)}")
    print(f"  Features after   : {len(selected_features)}")

    print("\n" + "=" * 60)
    print("Non-linear reduction complete.")
    print("=" * 60)

    return X_train_final, X_test_final, selected_features, importance_df