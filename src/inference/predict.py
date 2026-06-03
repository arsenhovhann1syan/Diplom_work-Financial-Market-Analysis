# ============================================================
# src/inference/predict.py — Production inference logic
# ============================================================

import pandas as pd

from src.config import DATA_START
from src.data.download import load_or_download
from src.data.validation import validate_and_clean_data
from src.data.external_signals import get_external_signals
from src.features.engineering import engineer_features_ml_ready
from src.models.artifacts import load_production_artifacts


_ARTIFACTS = None


LABEL_MAP = {
    -1: "DOWN",
     0: "NEUTRAL",
     1: "UP",
}


def get_artifacts():
    """
    Lazy-load production artifacts.

    This avoids loading model.pkl during module import,
    which is important for CI checks where model files may not exist.
    """

    global _ARTIFACTS

    if _ARTIFACTS is None:
        _ARTIFACTS = load_production_artifacts(input_dir="models")

    return _ARTIFACTS


def predict_single(input_data: dict) -> dict:
    """
    Makes prediction using the saved LightGBM Soft-Regime model.

    Steps:
      1. Load production artifacts
      2. Convert JSON input to DataFrame
      3. Select base features
      4. Create HMM posterior probabilities
      5. Add posterior features to input
      6. Predict class and probability
    """

    artifacts = get_artifacts()

    model = artifacts["model"]
    hmm_model = artifacts["hmm_model"]
    hmm_scaler = artifacts["hmm_scaler"]

    selected_features = artifacts["selected_features"]
    soft_features = artifacts["soft_features"]
    regime_features = artifacts["regime_features"]
    metadata = artifacts["metadata"]

    # -----------------------------
    # 1. Validate input features
    # -----------------------------
    missing_features = [
        feature for feature in selected_features
        if feature not in input_data
    ]

    if missing_features:
        return {
            "error": "Missing required features",
            "missing_features": missing_features,
        }

    # -----------------------------
    # 2. Base feature DataFrame
    # -----------------------------
    X_base = pd.DataFrame([input_data])[selected_features]

    # -----------------------------
    # 3. HMM posterior probabilities
    # -----------------------------
    X_regime = X_base[regime_features]
    X_regime_scaled = hmm_scaler.transform(X_regime)

    means = hmm_model.means_[:, 0]
    order = means.argsort()

    posterior = hmm_model.predict_proba(X_regime_scaled)[:, order]

    posterior_df = pd.DataFrame(
        posterior,
        columns=["p_low_vol", "p_mid_vol", "p_high_vol"],
        index=X_base.index,
    )

    # -----------------------------
    # 4. Soft-Regime input
    # -----------------------------
    X_soft = pd.concat([X_base, posterior_df], axis=1)
    X_soft = X_soft[soft_features]

    # -----------------------------
    # 5. Prediction
    # -----------------------------
    pred = model.predict(X_soft)[0]
    proba = model.predict_proba(X_soft)[0]

    max_probability = float(proba.max())

    return {
        "prediction": LABEL_MAP.get(int(pred), str(pred)),
        "prediction_raw": int(pred),
        "probability": round(max_probability, 4),
        "model_type": metadata.get("model_type", "LightGBM Soft-Regime"),
    }


def build_latest_features() -> tuple[dict, str]:
    """
    Builds the latest available ML features using the same feature engineering pipeline.

    Important:
      - Feature selection is NOT re-run here.
      - Selected features are loaded from features.json.
      - This keeps inference consistent with the trained model.
    """

    artifacts = get_artifacts()
    selected_features = artifacts["selected_features"]

    # -----------------------------
    # 1. Load / download raw BTC data
    # -----------------------------
    df_raw = load_or_download(start_str=DATA_START)

    # -----------------------------
    # 2. Validate and clean data
    # -----------------------------
    df_cleaned = validate_and_clean_data(df_raw)

    # -----------------------------
    # 3. External signals
    # -----------------------------
    df_external = get_external_signals(start_str=DATA_START)

    # -----------------------------
    # 4. Feature engineering
    # -----------------------------
    df_ml = engineer_features_ml_ready(df_cleaned, df_external)

    # -----------------------------
    # 5. Take latest available row
    # -----------------------------
    latest_row = df_ml.iloc[-1]
    latest_date = str(df_ml.index[-1].date())

    # -----------------------------
    # 6. Keep only selected features
    # -----------------------------
    input_data = latest_row[selected_features].to_dict()

    return input_data, latest_date


def predict_latest() -> dict:
    """
    Builds latest features automatically and returns prediction.
    """

    input_data, latest_date = build_latest_features()

    result = predict_single(input_data)
    result["date"] = latest_date

    return result
