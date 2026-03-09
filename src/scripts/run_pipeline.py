#!/usr/bin/env python
# ============================================================
# scripts/run_pipeline.py — End-to-end BTC ML pipeline runner
# ============================================================
# Usage:
#   python scripts/run_pipeline.py
#
# Runs all pipeline stages in order:
#   1.  Data download / load
#   2.  Validation & cleaning
#   3.  External signals
#   4.  Feature engineering
#   5.  Train/test split
#   6.  Correlation feature selection
#   7.  Tree-based feature selection
#   8.  HMM regime detection
#   9.  HMM posteriors extraction
#  10.  Optuna HPO + model training
#  11.  Walk-forward validation
#  12.  Backtesting
# ============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.config import DATA_START, RANDOM_STATE
from src.data.download import load_or_download
from src.data.validation import validate_and_clean_data
from src.data.external_signals import get_external_signals
from src.features.engineering import engineer_features_ml_ready
from src.features.selection import (
    correlation_feature_selection,
    tree_based_feature_selection,
)
from src.pipeline.split import train_test_split_pipeline
from src.models.regime import detect_market_regimes_hmm, extract_posteriors
from src.models.train import run_optuna_hpo, train_and_evaluate
from src.models.backtest import (
    calculate_metrics,
    build_binary_position,
    build_confidence_position,
    build_regime_aware_position,
    build_mixed_position,
    regime_breakdown,
)
from src.pipeline.walk_forward import run_walk_forward, summarise_wfv


def main():
    print("\n" + "=" * 70)
    print("  BTC ML PIPELINE — FULL RUN")
    print("=" * 70 + "\n")

    # ── 1. Data ───────────────────────────────────────────────
    print("STEP 1/12 — Data download / load")
    df = load_or_download(start_str=DATA_START)

    # ── 2. Validation ─────────────────────────────────────────
    print("\nSTEP 2/12 — Validation & cleaning")
    df_cleaned = validate_and_clean_data(df)

    # ── 3. External signals ───────────────────────────────────
    print("\nSTEP 3/12 — External signals")
    df_external = get_external_signals(start_str=DATA_START)

    # ── 4. Feature engineering ────────────────────────────────
    print("\nSTEP 4/12 — Feature engineering")
    df_ml = engineer_features_ml_ready(df_cleaned, df_external)

    # ── 5. Train/test split ───────────────────────────────────
    print("\nSTEP 5/12 — Train/test split")
    X_train, X_test, y_train, y_test, dta_threshold = \
        train_test_split_pipeline(df_ml)

    print(f"\n  DTA threshold : ±{dta_threshold * 100:.4f}%")

    # ── 6. Correlation selection ──────────────────────────────
    print("\nSTEP 6/12 — Correlation feature reduction")
    X_train_reduced, X_test_reduced, dropped_corr = \
        correlation_feature_selection(X_train, X_test)

    # ── 7. Tree-based selection ───────────────────────────────
    print("\nSTEP 7/12 — Tree-based feature selection")
    X_train_final, X_test_final, selected_features, importance_df = \
        tree_based_feature_selection(X_train_reduced, X_test_reduced, y_train)

    print(f"\n  Selected features ({len(selected_features)}):")
    for f in selected_features:
        mdi = importance_df.loc[f, "mdi_importance"]
        pi  = importance_df.loc[f, "pi_importance"]
        print(f"    • {f:<30}  MDI={mdi:.4f}  PI={pi:.4f}")

    # ── 8. Regime detection ───────────────────────────────────
    print("\nSTEP 8/12 — HMM regime detection")
    X_train_reg, X_test_reg, hmm_model, transition_matrix = \
        detect_market_regimes_hmm(
            X_train_final, X_test_final,
            random_state=RANDOM_STATE,
        )

    print("\n  Transition Matrix:")
    print(transition_matrix.round(3).to_string())

    # ── 9. HMM posteriors ─────────────────────────────────────
    print("\nSTEP 9/12 — Extracting HMM posteriors")
    train_post_df, test_post_df = extract_posteriors(
        hmm_model, X_train_final, X_test_final
    )
    X_train_soft = pd.concat([X_train_final, train_post_df], axis=1)
    X_test_soft  = pd.concat([X_test_final,  test_post_df],  axis=1)

    print(f"  Base  features : {X_train_final.shape[1]}")
    print(f"  Soft  features : {X_train_soft.shape[1]}  (+3 HMM posteriors)")

    # ── 10. HPO + training ────────────────────────────────────
    print("\nSTEP 10/12 — Optuna HPO + model training")
    best_params = run_optuna_hpo(
        X_train_soft, y_train,
        random_state=RANDOM_STATE,
    )
    summary_df, results, models = train_and_evaluate(
        X_train_soft,  X_test_soft,
        X_train_final, X_test_final,
        y_train, y_test,
        best_params,
    )
    print("\n  Model comparison (test set):")
    print(summary_df.to_string(index=False))

    # ── 11. Walk-forward validation ───────────────────────────
    print("\nSTEP 11/12 — Walk-forward validation")
    wfv_results = run_walk_forward(
        df_ml, best_params,
        random_state=RANDOM_STATE,
    )
    summarise_wfv(wfv_results)

    # ── 12. Backtesting ───────────────────────────────────────
    print("\nSTEP 12/12 — Backtesting")

    test_idx     = y_test.index
    log_ret_test = df_ml.loc[test_idx, "future_return"]
    regime_test  = X_test_reg["regime"].values

    lgbm_soft   = models["LightGBM Soft-Regime"]
    lgbm_global = models["LightGBM Global"]

    pred_soft    = lgbm_soft.predict(X_test_soft)
    proba_soft   = lgbm_soft.predict_proba(X_test_soft)
    pred_global  = lgbm_global.predict(X_test_final)
    proba_global = lgbm_global.predict_proba(X_test_final)

    # ── Position builders ─────────────────────────────────────
    pos_bah = pd.Series(1.0, index=log_ret_test.index)

    pos_b = build_binary_position(pred_global)
    pos_b.index = log_ret_test.index

    pos_c = build_confidence_position(pred_global, proba_global)
    pos_c.index = log_ret_test.index

    pos_d = build_regime_aware_position(pred_soft, proba_soft, regime_test)
    pos_d.index = log_ret_test.index

    pos_e = build_mixed_position(pred_soft, proba_soft, regime_test)
    pos_e.index = log_ret_test.index

    # ── Metrics ───────────────────────────────────────────────
    res_a = calculate_metrics(pos_bah, log_ret_test, "A) Buy & Hold",       0)
    res_b = calculate_metrics(pos_b,   log_ret_test, "B) Binary LGBM", 0.001)
    res_c = calculate_metrics(pos_c,   log_ret_test, "C) Confidence",  0.001)
    res_d = calculate_metrics(pos_d,   log_ret_test, "D) Regime v1",   0.001)
    res_e = calculate_metrics(pos_e,   log_ret_test, "E) Mixed (C×E)", 0.001)

    all_results = [res_a, res_b, res_c, res_d, res_e]

    summary_cols = ["Strategy", "Return %", "Ann Ret %",
                    "Sharpe", "Sortino", "Max DD %", "Trades"]

    backtest_df = pd.DataFrame([
        {k: v for k, v in r.items()
         if not k.startswith("_") and k != "Equity"}
        for r in all_results
    ])

    # ── Final summary ─────────────────────────────────────────
    print(f"\nTest Period Summary:")
    print(f"  Range         : {log_ret_test.index.min().date()} → "
          f"{log_ret_test.index.max().date()}")
    print(f"  Total Samples : {len(log_ret_test)}")

    print("\n" + "=" * 80)
    print("FINAL BACKTEST SUMMARY")
    print("=" * 80)
    print(backtest_df[summary_cols].to_string(index=False))

    # ── Regime breakdown for Strategy E ───────────────────────
    reg_e_df = regime_breakdown(
        res_e, log_ret_test, regime_test,
        "E) Mixed (C×E)", silent=True
    )
    print(f"\nRegime analysis for Strategy E:")
    print(reg_e_df.to_string(index=False))

    # ── Winning strategy ──────────────────────────────────────
    tradeable = [r for r in all_results if r["Trades"] > 30]
    best = max(tradeable, key=lambda x: x["Sharpe"])

    print("\n" + "=" * 50)
    print(f" WINNING STRATEGY: {best['Strategy']}")
    print("=" * 50)
    print(f" Sharpe      : {best['Sharpe']}")
    print(f" Return %    : {best['Return %']}")
    print(f" Max DD %    : {best['Max DD %']}")
    print(f" Trades      : {best['Trades']}")
    print("-" * 50)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()