# ============================================================
# retraining/retrain.py — Retrain production model on fresh data
# ============================================================

import os
import sys
import shutil
from datetime import datetime

import pandas as pd
from sklearn.metrics import f1_score

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATA_START,
    RAW_DATA_PATH,
    RANDOM_STATE,
    THRESHOLD_PERCENTILE,
    REGIME_COLS_KW,
)

from src.data.download import get_binance_data
from src.data.validation import validate_and_clean_data
from src.data.external_signals import get_external_signals
from src.features.engineering import engineer_features_ml_ready
from src.features.selection import (
    correlation_feature_selection,
    tree_based_feature_selection,
)
from src.models.regime import detect_market_regimes_hmm
from src.models.train import run_optuna_hpo, train_and_evaluate
from src.models.artifacts import save_production_artifacts


# Retraining setup
RETRAIN_TRAIN_END = "2026-03-10"
RETRAIN_TEST_START = "2026-03-11"

MIN_ACCEPTABLE_F1 = 0.30

ACTIVE_MODEL_DIR = "models"
BACKUP_DIR = "models_backup"
VERSION_DIR = "models_versions"


def make_labels(future_return: pd.Series, threshold: float) -> pd.Series:
    """
    Convert future returns into 3 classes:
      -1 = DOWN
       0 = NEUTRAL
       1 = UP
    """
    y = pd.Series(0, index=future_return.index)
    y[future_return > threshold] = 1
    y[future_return < -threshold] = -1
    return y


def extract_posteriors_with_saved_scaler(
    hmm_model,
    hmm_scaler,
    X_train_final: pd.DataFrame,
    X_test_final: pd.DataFrame,
    regime_features: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create HMM posterior probabilities using the same scaler
    that will be saved for production inference.
    """

    Xtr_sc = hmm_scaler.transform(X_train_final[regime_features])
    Xte_sc = hmm_scaler.transform(X_test_final[regime_features])

    means = hmm_model.means_[:, 0]
    order = means.argsort()

    train_post = hmm_model.predict_proba(Xtr_sc)[:, order]
    test_post = hmm_model.predict_proba(Xte_sc)[:, order]

    cols = ["p_low_vol", "p_mid_vol", "p_high_vol"]

    train_post_df = pd.DataFrame(
        train_post,
        index=X_train_final.index,
        columns=cols,
    )

    test_post_df = pd.DataFrame(
        test_post,
        index=X_test_final.index,
        columns=cols,
    )

    return train_post_df, test_post_df


def backup_current_model():
    """
    Backup current production artifacts before replacing them.
    """

    if not os.path.exists(ACTIVE_MODEL_DIR):
        print("No active models directory found. Skipping backup.")
        return None

    os.makedirs(BACKUP_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"model_backup_{timestamp}")

    shutil.copytree(ACTIVE_MODEL_DIR, backup_path)

    print(f"✅ Current production model backed up to: {backup_path}")

    return backup_path


def main():
    print("=" * 70)
    print("MODEL RETRAINING")
    print("=" * 70)

    os.makedirs(VERSION_DIR, exist_ok=True)

    # -----------------------------
    # 1. Download fresh data
    # -----------------------------
    print("\n[1/8] Downloading fresh Binance data...")
    df_raw = get_binance_data(start_str=DATA_START)
    df_raw.to_csv(RAW_DATA_PATH, index=False)
    print(f"✅ Fresh data saved to {RAW_DATA_PATH}")

    # -----------------------------
    # 2. Feature engineering
    # -----------------------------
    print("\n[2/8] Building features...")
    df_cleaned = validate_and_clean_data(df_raw)
    df_external = get_external_signals(start_str=DATA_START)
    df_ml = engineer_features_ml_ready(df_cleaned, df_external)

    # -----------------------------
    # 3. New train/test split
    # -----------------------------
    print("\n[3/8] Creating retraining split...")

    train_df = df_ml.loc[df_ml.index <= RETRAIN_TRAIN_END].copy()
    test_df = df_ml.loc[df_ml.index >= RETRAIN_TEST_START].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test set is empty. Check retraining dates.")

    threshold = train_df["future_return"].abs().quantile(
        THRESHOLD_PERCENTILE / 100
    )

    y_train = make_labels(train_df["future_return"], threshold)
    y_test = make_labels(test_df["future_return"], threshold)

    X_train = train_df.drop(columns=["future_return"])
    X_test = test_df.drop(columns=["future_return"])

    print(f"Train period : {X_train.index.min().date()} → {X_train.index.max().date()}")
    print(f"Test period  : {X_test.index.min().date()} → {X_test.index.max().date()}")
    print(f"Train rows   : {len(X_train)}")
    print(f"Test rows    : {len(X_test)}")
    print(f"Threshold    : ±{threshold:.6f}")

    # -----------------------------
    # 4. Feature selection
    # -----------------------------
    print("\n[4/8] Feature selection...")

    X_train_reduced, X_test_reduced, dropped_corr = \
        correlation_feature_selection(X_train, X_test)

    X_train_final, X_test_final, selected_features, importance_df = \
        tree_based_feature_selection(X_train_reduced, X_test_reduced, y_train)

    print(f"Selected features ({len(selected_features)}):")
    for feature in selected_features:
        print(f"  - {feature}")

    # -----------------------------
    # 5. HMM regimes
    # -----------------------------
    print("\n[5/8] HMM regime detection...")

    X_train_reg, X_test_reg, hmm_model, hmm_scaler, transition_matrix = \
        detect_market_regimes_hmm(
            X_train_final,
            X_test_final,
            random_state=RANDOM_STATE,
        )

    regime_features = [
        c for c in X_train_final.columns
        if any(k in c for k in REGIME_COLS_KW)
    ]

    train_post_df, test_post_df = extract_posteriors_with_saved_scaler(
        hmm_model=hmm_model,
        hmm_scaler=hmm_scaler,
        X_train_final=X_train_final,
        X_test_final=X_test_final,
        regime_features=regime_features,
    )

    X_train_soft = pd.concat([X_train_final, train_post_df], axis=1)
    X_test_soft = pd.concat([X_test_final, test_post_df], axis=1)

    # -----------------------------
    # 6. Train new model
    # -----------------------------
    print("\n[6/8] Training new model...")

    best_params = run_optuna_hpo(
        X_train_soft,
        y_train,
        random_state=RANDOM_STATE,
    )

    summary_df, results, models = train_and_evaluate(
        X_train_soft,
        X_test_soft,
        X_train_final,
        X_test_final,
        y_train,
        y_test,
        best_params,
    )

    print("\nNew model comparison:")
    print(summary_df.to_string(index=False))

    soft_row = summary_df[summary_df["model"] == "LightGBM Soft-Regime"].iloc[0]
    new_f1 = float(soft_row["macro_f1"])

    print(f"\nNew Soft-Regime Macro F1: {new_f1:.4f}")

    # -----------------------------
    # 7. Save versioned model
    # -----------------------------
    print("\n[7/8] Saving versioned candidate model...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_path = os.path.join(VERSION_DIR, f"model_v2_{timestamp}")

    metadata = {
        "model_version": f"model_v2_{timestamp}",
        "best_model": "LightGBM Soft-Regime",
        "base_features_count": len(X_train_final.columns),
        "soft_features_count": len(X_train_soft.columns),
        "train_start": str(X_train.index.min().date()),
        "train_end": str(X_train.index.max().date()),
        "test_start": str(X_test.index.min().date()),
        "test_end": str(X_test.index.max().date()),
        "macro_f1": round(new_f1, 4),
        "threshold": round(float(threshold), 6),
    }

    save_production_artifacts(
        model=models["LightGBM Soft-Regime"],
        hmm_model=hmm_model,
        hmm_scaler=hmm_scaler,
        selected_features=list(X_train_final.columns),
        soft_features=list(X_train_soft.columns),
        regime_features=regime_features,
        metadata=metadata,
        output_dir=version_path,
    )

    print(f"✅ Candidate model saved to: {version_path}")

    # -----------------------------
    # 8. Promote if acceptable
    # -----------------------------
    print("\n[8/8] Promotion decision...")

    if new_f1 >= MIN_ACCEPTABLE_F1:
        print(f"✅ New model accepted: F1={new_f1:.4f} >= {MIN_ACCEPTABLE_F1}")

        backup_current_model()

        save_production_artifacts(
            model=models["LightGBM Soft-Regime"],
            hmm_model=hmm_model,
            hmm_scaler=hmm_scaler,
            selected_features=list(X_train_final.columns),
            soft_features=list(X_train_soft.columns),
            regime_features=regime_features,
            metadata=metadata,
            output_dir=ACTIVE_MODEL_DIR,
        )

        print("✅ New model promoted to production models/")
    else:
        print(f"⚠️ New model rejected: F1={new_f1:.4f} < {MIN_ACCEPTABLE_F1}")
        print("Production model was not changed.")

    print("\n✅ Retraining complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
