# ============================================================
# src/models/regime.py — Gaussian HMM market regime detection
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

from config import N_REGIMES, REGIME_NAMES, REGIME_COLS_KW, RANDOM_STATE


def _stable_regime_labels(model, train_regimes, test_regimes, vol_col_idx=0):
    means = model.means_[:, vol_col_idx]
    order = np.argsort(means)
    remap = {old: new for new, old in enumerate(order)}
    return (
        np.array([remap[r] for r in train_regimes]),
        np.array([remap[r] for r in test_regimes]),
        remap,
    )


def detect_market_regimes_hmm(
    X_train:      pd.DataFrame,
    X_test:       pd.DataFrame,
    n_regimes:    int = N_REGIMES,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, GaussianHMM, pd.DataFrame]:
    
    print("=" * 60)
    print("MARKET REGIME DETECTION  (Gaussian HMM)")
    print(f"Forced n_states = {n_regimes}  (Low / Mid / High Vol)")
    print("=" * 60)

    regime_cols = [c for c in X_train.columns
                   if any(k in c for k in REGIME_COLS_KW)]
    print(f"\n  Regime features ({len(regime_cols)}): {regime_cols}")

    scaler      = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train[regime_cols].values)
    X_te_scaled = scaler.transform(X_test[regime_cols].values)

    # BIC reference table (informational)
    print("\n  BIC reference (informational — not used for selection):")
    print(f"  {'n_states':<10} {'Log-L':>12} {'BIC':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12}")
    n_samples, n_features = X_tr_scaled.shape

    for n in range(2, 7):
        try:
            m = GaussianHMM(n_components=n, covariance_type="full",
                            n_iter=200, random_state=random_state, tol=1e-4)
            m.fit(X_tr_scaled)
            ll    = m.score(X_tr_scaled)
            k     = n*(n-1) + n*n_features + n*n_features*n_features
            bic   = -2 * ll + k * np.log(n_samples)
            tag   = "  ← forced" if n == n_regimes else ""
            print(f"  {n:<10} {ll:>12.2f} {bic:>12.2f}{tag}")
        except Exception:
            pass

    # Fit final HMM
    print(f"\n  Fitting final HMM (n_states={n_regimes}) ...")
    hmm_model = GaussianHMM(
        n_components    = n_regimes,
        covariance_type = "full",
        n_iter          = 500,
        random_state    = random_state,
        tol             = 1e-5,
    )
    hmm_model.fit(X_tr_scaled)

    train_raw = hmm_model.predict(X_tr_scaled)
    test_raw  = hmm_model.predict(X_te_scaled)

    train_regimes, test_regimes, remap = _stable_regime_labels(
        hmm_model, train_raw, test_raw, vol_col_idx=0
    )
    print(f"  Remap (raw→vol-ordered): {remap}")

    # Attach column
    X_train_out = X_train.copy()
    X_test_out  = X_test.copy()
    X_train_out["regime"] = train_regimes
    X_test_out["regime"]  = test_regimes

    # Summary
    print("\n" + "=" * 60)
    print("REGIME SUMMARY")
    print("=" * 60)
    print(f"\n  {'ID':<4} {'Name':<10} {'Train':>7} {'Train%':>8} "
          f"{'Test':>7} {'Test%':>8}")
    print(f"  {'-'*4} {'-'*10} {'-'*7} {'-'*8} {'-'*7} {'-'*8}")
    for i in range(n_regimes):
        tr_n = (train_regimes == i).sum()
        te_n = (test_regimes  == i).sum()
        print(f"  {i:<4} {REGIME_NAMES[i]:<10} "
              f"{tr_n:>7} {tr_n/len(train_regimes)*100:>7.1f}% "
              f"{te_n:>7} {te_n/len(test_regimes)*100:>7.1f}%")

    # Transition matrix
    order    = [k for k, v in sorted(remap.items(), key=lambda x: x[1])]
    trans    = hmm_model.transmat_[np.ix_(order, order)]
    trans_df = pd.DataFrame(
        trans,
        index  = [f"From {REGIME_NAMES[i]}" for i in range(n_regimes)],
        columns= [f"To {REGIME_NAMES[i]}"   for i in range(n_regimes)],
    )
    print(f"\n  Transition Matrix:")
    print(trans_df.round(3).to_string())

    return X_train_out, X_test_out, hmm_model, trans_df


def extract_posteriors(
    hmm_model:     GaussianHMM,
    X_train_final: pd.DataFrame,
    X_test_final:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    regime_cols = [c for c in X_train_final.columns
                   if any(k in c for k in REGIME_COLS_KW)]

    scaler  = StandardScaler()
    Xtr_sc  = scaler.fit_transform(X_train_final[regime_cols].values)
    Xte_sc  = scaler.transform(X_test_final[regime_cols].values)

    means     = hmm_model.means_[:, 0]
    order     = np.argsort(means)
    remap     = {old: new for new, old in enumerate(order)}
    col_order = [k for k, v in sorted(remap.items(), key=lambda x: x[1])]

    tr_post = hmm_model.predict_proba(Xtr_sc)[:, col_order]
    te_post = hmm_model.predict_proba(Xte_sc)[:, col_order]

    cols    = ["p_low_vol", "p_mid_vol", "p_high_vol"]
    tr_df   = pd.DataFrame(tr_post, index=X_train_final.index, columns=cols)
    te_df   = pd.DataFrame(te_post, index=X_test_final.index,  columns=cols)

    print(f"  Regime cols used  : {len(regime_cols)}")
    print(f"  Posterior shape   : train={tr_df.shape}  test={te_df.shape}")
    return tr_df, te_df