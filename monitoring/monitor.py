# ============================================================
# monitoring/monitor.py — Basic model monitoring
# ============================================================

import os
import sys
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_START, RAW_DATA_PATH
from src.data.download import get_binance_data
from src.data.validation import validate_and_clean_data
from src.data.external_signals import get_external_signals
from src.features.engineering import engineer_features_ml_ready
from src.pipeline.split import train_test_split_pipeline
from src.inference.predict import predict_single


MONITOR_START_DATE = "2026-03-11"
MIN_F1_THRESHOLD = 0.30

LOG_PATH = "logs/predictions.csv"
REPORT_PATH = "reports/monitoring_report.csv"


def label_from_return(value: float, threshold: float) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def main():
    print("=" * 60)
    print("MODEL MONITORING")
    print("=" * 60)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # -----------------------------
    # 1. Download fresh Binance data
    # -----------------------------
    print("\n[1/6] Downloading fresh Binance data...")
    df_raw = get_binance_data(start_str=DATA_START)
    df_raw.to_csv(RAW_DATA_PATH, index=False)
    print(f"✅ Fresh data saved to {RAW_DATA_PATH}")

    # -----------------------------
    # 2. Validation + feature engineering
    # -----------------------------
    print("\n[2/6] Building features...")
    df_cleaned = validate_and_clean_data(df_raw)
    df_external = get_external_signals(start_str=DATA_START)
    df_ml = engineer_features_ml_ready(df_cleaned, df_external)

    # -----------------------------
    # 3. Get same DTA threshold
    # -----------------------------
    print("\n[3/6] Getting original labeling threshold...")
    _, _, _, _, dta_threshold = train_test_split_pipeline(df_ml)
    print(f"Monitoring threshold: ±{dta_threshold:.6f}")

    # -----------------------------
    # 4. Select monitoring period
    # -----------------------------
    monitor_df = df_ml.loc[df_ml.index >= MONITOR_START_DATE].copy()

    if monitor_df.empty:
        print("No monitoring data found.")
        return

    print(f"\nMonitoring period:")
    print(f"  Start : {monitor_df.index.min().date()}")
    print(f"  End   : {monitor_df.index.max().date()}")
    print(f"  Rows  : {len(monitor_df)}")

    # -----------------------------
    # 5. Predict each day
    # -----------------------------
    print("\n[4/6] Running predictions...")

    rows = []

    for date, row in monitor_df.iterrows():
        input_data = row.to_dict()
        result = predict_single(input_data)

        if "error" in result:
            continue

        actual = label_from_return(row["future_return"], dta_threshold)

        rows.append({
            "date": date,
            "prediction": result["prediction"],
            "prediction_raw": result["prediction_raw"],
            "probability": result["probability"],
            "actual_raw": actual,
            "future_return": row["future_return"],
        })

    pred_df = pd.DataFrame(rows)

    if pred_df.empty:
        print("No valid predictions generated.")
        return

    pred_df.to_csv(LOG_PATH, index=False)
    print(f"✅ Predictions saved to {LOG_PATH}")

    # -----------------------------
    # 6. Metrics
    # -----------------------------
    print("\n[5/6] Calculating metrics...")

    y_true = pred_df["actual_raw"]
    y_pred = pred_df["prediction_raw"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report_df = pd.DataFrame([{
        "monitor_start": str(pred_df["date"].min()),
        "monitor_end": str(pred_df["date"].max()),
        "samples": len(pred_df),
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
        "min_f1_threshold": MIN_F1_THRESHOLD,
        "status": "OK" if f1 >= MIN_F1_THRESHOLD else "RETRAINING_REQUIRED",
    }])

    report_df.to_csv(REPORT_PATH, index=False)

    print("\n" + "=" * 60)
    print("MONITORING REPORT")
    print("=" * 60)
    print(report_df.to_string(index=False))

    if f1 < MIN_F1_THRESHOLD:
        print("\n⚠️ Warning: retraining required")
    else:
        print("\n✅ Model performance is acceptable")

    print(f"\n✅ Report saved to {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()


 