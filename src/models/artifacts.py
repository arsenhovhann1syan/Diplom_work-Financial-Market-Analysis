# ============================================================
# src/models/artifacts.py — Save / Load production artifacts
# ============================================================

import os
import json
import joblib
from datetime import datetime


def save_production_artifacts(
    model,
    hmm_model,
    hmm_scaler,
    selected_features,
    soft_features,
    regime_features,
    metadata=None,
    output_dir="models",
):
    """
    Saves all artifacts needed for production inference.

    Artifacts:
      - LightGBM Soft-Regime model
      - HMM model
      - HMM scaler
      - selected base features
      - soft model features
      - regime features
      - metadata
    """

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Save models
    # -----------------------------
    joblib.dump(model, os.path.join(output_dir, "model.pkl"))
    joblib.dump(hmm_model, os.path.join(output_dir, "hmm_model.pkl"))
    joblib.dump(hmm_scaler, os.path.join(output_dir, "hmm_scaler.pkl"))

    # -----------------------------
    # Save feature lists
    # -----------------------------
    with open(os.path.join(output_dir, "features.json"), "w") as f:
        json.dump(selected_features, f, indent=4)

    with open(os.path.join(output_dir, "soft_features.json"), "w") as f:
        json.dump(soft_features, f, indent=4)

    with open(os.path.join(output_dir, "regime_features.json"), "w") as f:
        json.dump(regime_features, f, indent=4)

    # -----------------------------
    # Save metadata
    # -----------------------------
    if metadata is None:
        metadata = {}

    metadata["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata["model_type"] = "LightGBM Soft-Regime"
    metadata["uses_hmm"] = True

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("=" * 60)
    print("PRODUCTION ARTIFACTS SAVED")
    print("=" * 60)
    print(f"Directory        : {output_dir}")
    print("Saved files:")
    print("  - model.pkl")
    print("  - hmm_model.pkl")
    print("  - hmm_scaler.pkl")
    print("  - features.json")
    print("  - soft_features.json")
    print("  - regime_features.json")
    print("  - metadata.json")
    print("=" * 60)


def load_production_artifacts(input_dir="models"):
    """
    Loads all production artifacts for FastAPI inference.
    """

    model = joblib.load(os.path.join(input_dir, "model.pkl"))
    hmm_model = joblib.load(os.path.join(input_dir, "hmm_model.pkl"))
    hmm_scaler = joblib.load(os.path.join(input_dir, "hmm_scaler.pkl"))

    with open(os.path.join(input_dir, "features.json"), "r") as f:
        selected_features = json.load(f)

    with open(os.path.join(input_dir, "soft_features.json"), "r") as f:
        soft_features = json.load(f)

    with open(os.path.join(input_dir, "regime_features.json"), "r") as f:
        regime_features = json.load(f)

    with open(os.path.join(input_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    return {
        "model": model,
        "hmm_model": hmm_model,
        "hmm_scaler": hmm_scaler,
        "selected_features": selected_features,
        "soft_features": soft_features,
        "regime_features": regime_features,
        "metadata": metadata,
    }