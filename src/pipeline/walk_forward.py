# ============================================================
# src/pipeline/walk_forward.py — Walk-forward validation
# ============================================================

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from config import (
    WFV_INITIAL_TRAIN_MONTHS,
    WFV_STEP_MONTHS,
    WFV_THRESHOLD_PCT,
    REGIME_COLS_KW,
    N_REGIMES,
    RANDOM_STATE,
)
from src.models.regime import extract_posteriors

warnings.filterwarnings("ignore")


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _make_labels(future_ret: pd.Series, threshold: float) -> pd.Series:
    labels = pd.Series(0, index=future_ret.index, dtype=int)
    labels[future_ret >  threshold] =  1
    labels[future_ret < -threshold] = -1
    return labels


def _fit_hmm_and_posteriors(
    X_train_fold: pd.DataFrame,
    X_test_fold:  pd.DataFrame,
    regime_cols:  list,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fits HMM on a single fold's training data and returns posteriors."""
    from sklearn.preprocessing import StandardScaler
    from hmmlearn.hmm import GaussianHMM

    scaler  = StandardScaler()
    Xtr_sc  = scaler.fit_transform(X_train_fold[regime_cols].values)
    Xte_sc  = scaler.transform(X_test_fold[regime_cols].values)

    hmm = GaussianHMM(
        n_components    = N_REGIMES,
        covariance_type = "full",
        n_iter          = 300,
        random_state    = random_state,
        tol             = 1e-4,
    )
    hmm.fit(Xtr_sc)

    order    = np.argsort(hmm.means_[:, 0])
    remap    = {old: new for new, old in enumerate(order)}
    col_ord  = [k for k, v in sorted(remap.items(), key=lambda x: x[1])]

    tr_post = hmm.predict_proba(Xtr_sc)[:, col_ord]
    te_post = hmm.predict_proba(Xte_sc)[:, col_ord]

    cols   = ["p_low_vol", "p_mid_vol", "p_high_vol"]
    tr_df  = pd.DataFrame(tr_post, index=X_train_fold.index, columns=cols)
    te_df  = pd.DataFrame(te_post, index=X_test_fold.index,  columns=cols)
    return tr_df, te_df


# ============================================================
# Walk-forward engine
# ============================================================

def run_walk_forward(
    df_ml:                 pd.DataFrame,
    best_params:           dict,
    initial_train_months:  int = WFV_INITIAL_TRAIN_MONTHS,
    step_months:           int = WFV_STEP_MONTHS,
    random_state:          int = RANDOM_STATE,
) -> pd.DataFrame:

    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print(f"  Initial train window : {initial_train_months} months")
    print(f"  Step size            : {step_months} months")
    print(f"  Models               : LR | Global LGBM | Soft-Regime LGBM")
    print("=" * 60)

    feature_cols = [c for c in df_ml.columns if c != "future_return"]
    regime_cols  = [c for c in feature_cols
                    if any(k in c for k in REGIME_COLS_KW)]

    all_dates = df_ml.index
    start_dt  = all_dates.min()

    # Build fold boundaries
    folds = []
    train_end = start_dt + pd.DateOffset(months=initial_train_months)
    while True:
        test_end = train_end + pd.DateOffset(months=step_months)
        if test_end > all_dates.max():
            break
        folds.append((start_dt, train_end, test_end))
        train_end = test_end

    print(f"\n  Total folds : {len(folds)}")
    print(f"  {'Fold':<5} {'Train start':>12} {'Train end':>12} "
          f"{'Test end':>12} {'Test days':>10}")
    print(f"  {'-'*5} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    for i, (ts, te, td) in enumerate(folds):
        test_days = len(df_ml.loc[te:td]) - 1
        print(f"  {i+1:<5} {str(ts.date()):>12} {str(te.date()):>12} "
              f"{str(td.date()):>12} {test_days:>10}")

    fold_results = []

    for fold_idx, (train_start, train_end, test_end) in enumerate(folds):

        train_df = df_ml.loc[train_start:train_end].copy()
        test_df  = df_ml.loc[train_end:test_end].copy().iloc[1:]

        if len(test_df) < 10:
            continue

        threshold    = train_df["future_return"].abs().quantile(
            WFV_THRESHOLD_PCT / 100
        )
        y_train_fold = _make_labels(train_df["future_return"], threshold)
        y_test_fold  = _make_labels(test_df["future_return"],  threshold)

        X_train_fold = train_df[feature_cols]
        X_test_fold  = test_df[feature_cols]

        # HMM posteriors (re-fitted per fold)
        try:
            tr_post, te_post = _fit_hmm_and_posteriors(
                X_train_fold, X_test_fold, regime_cols, random_state
            )
            X_train_soft = pd.concat([X_train_fold, tr_post], axis=1)
            X_test_soft  = pd.concat([X_test_fold,  te_post], axis=1)
            hmm_ok = True
        except Exception:
            X_train_soft = X_train_fold.copy()
            X_test_soft  = X_test_fold.copy()
            hmm_ok = False

        fold_row = {
            "fold"        : fold_idx + 1,
            "train_start" : train_start.date(),
            "train_end"   : train_end.date(),
            "test_end"    : test_end.date(),
            "n_train"     : len(X_train_fold),
            "n_test"      : len(X_test_fold),
            "threshold"   : round(threshold, 6),
            "hmm_ok"      : hmm_ok,
        }

        # LR
        try:
            lr = Pipeline([
                ("sc",  StandardScaler()),
                ("clf", LogisticRegression(
                    multi_class="multinomial", solver="lbfgs",
                    class_weight="balanced", C=0.1,
                    max_iter=1000, random_state=random_state,
                )),
            ])
            lr.fit(X_train_fold, y_train_fold)
            fold_row["lr_macro_f1"] = round(
                f1_score(y_test_fold, lr.predict(X_test_fold),
                         average="macro", zero_division=0), 4)
        except Exception:
            fold_row["lr_macro_f1"] = np.nan

        # Global LGBM
        try:
            lgbm_g = lgb.LGBMClassifier(
                **{**best_params, "class_weight": "balanced",
                   "random_state": random_state, "n_jobs": -1, "verbose": -1}
            )
            lgbm_g.fit(X_train_fold, y_train_fold)
            fold_row["global_macro_f1"] = round(
                f1_score(y_test_fold, lgbm_g.predict(X_test_fold),
                         average="macro", zero_division=0), 4)
        except Exception:
            fold_row["global_macro_f1"] = np.nan

        # Soft-Regime LGBM
        try:
            lgbm_s = lgb.LGBMClassifier(
                **{**best_params, "class_weight": "balanced",
                   "random_state": random_state, "n_jobs": -1, "verbose": -1}
            )
            lgbm_s.fit(X_train_soft, y_train_fold)
            fold_row["soft_macro_f1"] = round(
                f1_score(y_test_fold, lgbm_s.predict(X_test_soft),
                         average="macro", zero_division=0), 4)
        except Exception:
            fold_row["soft_macro_f1"] = np.nan

        fold_results.append(fold_row)

        print(f"\n  Fold {fold_idx+1}  "
              f"[{train_end.date()} → {test_end.date()}]  "
              f"n_test={len(X_test_fold)}  thr={threshold*100:.2f}%")
        print(f"    LR={fold_row.get('lr_macro_f1','N/A')}  "
              f"Global={fold_row.get('global_macro_f1','N/A')}  "
              f"Soft={fold_row.get('soft_macro_f1','N/A')}")

    return pd.DataFrame(fold_results)


# ============================================================
# Summary stats
# ============================================================

def summarise_wfv(results_df: pd.DataFrame) -> None:
    """Prints aggregate walk-forward statistics."""
    print("\n" + "=" * 60)
    print("WALK-FORWARD RESULTS — FOLD-BY-FOLD")
    print("=" * 60)
    print(results_df[
        ["fold", "train_end", "test_end", "n_test",
         "lr_macro_f1", "global_macro_f1", "soft_macro_f1"]
    ].to_string(index=False))

    print("\n" + "=" * 60)
    print("WALK-FORWARD RESULTS — AGGREGATE")
    print("=" * 60)
    for col, name in [
        ("lr_macro_f1",     "LR Baseline     "),
        ("global_macro_f1", "Global LGBM     "),
        ("soft_macro_f1",   "Soft-Regime LGBM"),
    ]:
        vals = results_df[col].dropna()
        print(f"  {name}  mean={vals.mean():.4f}  "
              f"std={vals.std():.4f}  "
              f"min={vals.min():.4f}  "
              f"max={vals.max():.4f}")