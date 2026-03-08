# ============================================================
# src/models/train.py — Optuna HPO, model training, evaluation
# ============================================================

import warnings
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from Config import HPO_N_TRIALS, HPO_VAL_FRAC, RANDOM_STATE

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# Optuna HPO
# ============================================================

def run_optuna_hpo(
    X_train_soft: pd.DataFrame,
    y_train:      pd.Series,
    n_trials:     int   = HPO_N_TRIALS,
    val_fraction: float = HPO_VAL_FRAC,
    random_state: int   = RANDOM_STATE,
) -> dict:

    val_cut   = int(len(X_train_soft) * (1 - val_fraction))
    X_tr_hpo  = X_train_soft.iloc[:val_cut]
    y_tr_hpo  = y_train.iloc[:val_cut]
    X_val_hpo = X_train_soft.iloc[val_cut:]
    y_val_hpo = y_train.iloc[val_cut:]

    print(f"\n  HPO train slice : {len(X_tr_hpo):,} samples  "
          f"({X_tr_hpo.index.min().date()} → {X_tr_hpo.index.max().date()})")
    print(f"  HPO val  slice  : {len(X_val_hpo):,} samples  "
          f"({X_val_hpo.index.min().date()} → {X_val_hpo.index.max().date()})")

    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators",   200, 800),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth         = trial.suggest_int("max_depth",       3, 7),
            num_leaves        = trial.suggest_int("num_leaves",      10, 50),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 30),
            subsample         = trial.suggest_float("subsample",      0.6, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha",     1e-3, 2.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda",    1e-3, 2.0, log=True),
            class_weight      = "balanced",
            random_state      = random_state,
            n_jobs            = -1,
            verbose           = -1,
        )
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr_hpo, y_tr_hpo)
        y_pred = m.predict(X_val_hpo)
        return f1_score(y_val_hpo, y_pred, average="macro", zero_division=0)

    study = optuna.create_study(
        direction = "maximize",
        sampler   = optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\n  Best macro F1 (val) : {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k:<22} : {v}")

    return study.best_params


# ============================================================
# Train all three models + compare
# ============================================================

def train_and_evaluate(
    X_train_soft: pd.DataFrame,
    X_test_soft:  pd.DataFrame,
    X_train_base: pd.DataFrame,
    X_test_base:  pd.DataFrame,
    y_train:      pd.Series,
    y_test:       pd.Series,
    best_params:  dict,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Trains three models and evaluates them on the test set.

    Models
    ------
    A) Logistic Regression baseline
    B) Global LightGBM   (no regime posteriors)
    C) Soft-Regime LGBM  (+ HMM posteriors)

    Returns
    -------
    summary_df : per-model metrics DataFrame
    results    : dict {name: y_pred array}
    models     : dict {name: fitted estimator}
    """
    results = {}
    models  = {}

    # A) LR Baseline
    print("\n  [A] Logistic Regression baseline ...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            class_weight="balanced", C=0.1,
            max_iter=1000, random_state=random_state,
        )),
    ])
    lr_pipe.fit(X_train_base, y_train)
    results["LR Baseline"] = lr_pipe.predict(X_test_base)
    models["LR Baseline"]  = lr_pipe

    # B) Global LightGBM
    print("  [B] Global LightGBM (no regime features) ...")
    lgbm_global = lgb.LGBMClassifier(
        **{**best_params, "class_weight": "balanced",
           "random_state": random_state, "n_jobs": -1, "verbose": -1}
    )
    lgbm_global.fit(X_train_base, y_train)
    results["LightGBM Global"] = lgbm_global.predict(X_test_base)
    models["LightGBM Global"]  = lgbm_global

    # C) Soft-Regime LightGBM
    print("  [C] Soft-Regime LightGBM (+ HMM posteriors) ...")
    lgbm_soft = lgb.LGBMClassifier(
        **{**best_params, "class_weight": "balanced",
           "random_state": random_state, "n_jobs": -1, "verbose": -1}
    )
    lgbm_soft.fit(X_train_soft, y_train)
    results["LightGBM Soft-Regime"] = lgbm_soft.predict(X_test_soft)
    models["LightGBM Soft-Regime"]  = lgbm_soft

    # Metrics
    print("\n" + "=" * 60)
    print("MODEL COMPARISON — TEST SET")
    print("=" * 60)
    print(f"\n  {'Model':<25} {'Acc':>7} {'MacroF1':>9} "
          f"{'LongF1':>8} {'ShortF1':>9}")
    print(f"  {'-'*25} {'-'*7} {'-'*9} {'-'*8} {'-'*9}")

    summary = []
    for name, y_pred in results.items():
        acc = accuracy_score(y_test, y_pred)
        mf1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        rep = classification_report(
            y_test, y_pred, labels=[-1, 0, 1],
            target_names=["Short", "Neutral", "Long"],
            output_dict=True, zero_division=0,
        )
        lf1 = rep.get("Long",  {}).get("f1-score", 0.0)
        sf1 = rep.get("Short", {}).get("f1-score", 0.0)
        print(f"  {name:<25} {acc:>7.4f} {mf1:>9.4f} {lf1:>8.4f} {sf1:>9.4f}")
        summary.append(dict(model=name, accuracy=acc, macro_f1=mf1,
                            long_f1=lf1, short_f1=sf1))

    return pd.DataFrame(summary), results, models