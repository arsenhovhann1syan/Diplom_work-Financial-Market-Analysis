# ============================================================
# src/models/backtest.py — Backtesting engine
# ============================================================

import numpy as np
import pandas as pd

from config import (
    COST_BPS,
    ANNUAL_FACTOR,
    CONF_THRESHOLD,
    HYSTERESIS_ZONE,
    MIN_HOLD_TIME,
    MIXED_REGIME_SCALARS,
    MIXED_CONF_BY_REGIME,
    MIXED_HYSTERESIS,
    MIXED_MIN_HOLD,
)


# ============================================================
# Position builders
# ============================================================

def build_binary_position(y_pred: np.ndarray) -> pd.Series:
    """Simple ±1 / 0 position from classifier output."""
    pos = pd.Series(y_pred.astype(float))
    pos[pos == 0] = 0.0
    return pos


def build_confidence_position(
    y_pred:         np.ndarray,
    y_proba:        np.ndarray,
    conf_threshold: float = CONF_THRESHOLD,
    hysteresis:     float = HYSTERESIS_ZONE,
    min_hold:       int   = MIN_HOLD_TIME,
) -> pd.Series:
    """
    Position = direction × confidence probability.
    Only opens when prob > conf_threshold.
    Hysteresis and min-hold prevent excessive churning.
    """
    n = len(y_pred)
    pos = np.zeros(n)
    current_pos, hold_count = 0.0, 0

    for i in range(n):
        pred    = y_pred[i]
        cls_idx = int(pred) + 1
        prob    = y_proba[i, cls_idx]
        target  = float(pred) * prob if (prob > conf_threshold and pred != 0) else 0.0

        if abs(target - current_pos) > hysteresis and hold_count >= min_hold:
            current_pos = target
            hold_count  = 0
        else:
            hold_count += 1

        pos[i] = current_pos

    return pd.Series(pos)


def build_regime_aware_position(
    y_pred:          np.ndarray,
    y_proba:         np.ndarray,
    regime:          np.ndarray,
    conf_threshold:  float = CONF_THRESHOLD,
    hysteresis:      float = HYSTERESIS_ZONE,
    min_hold:        int   = MIN_HOLD_TIME,
    high_vol_scalar: float = 0.5,
) -> pd.Series:
    """
    Strategy D: confidence position with a flat high-vol size reduction.
    Kept for fair comparison against Strategy E.
    """
    n = len(y_pred)
    pos = np.zeros(n)
    current_pos, hold_count = 0.0, 0

    for i in range(n):
        pred    = y_pred[i]
        cls_idx = int(pred) + 1
        prob    = y_proba[i, cls_idx]
        target  = float(pred) * prob if (prob > conf_threshold and pred != 0) else 0.0

        if regime[i] == 2:
            target *= high_vol_scalar

        if abs(target - current_pos) > hysteresis and hold_count >= min_hold:
            current_pos = target
            hold_count  = 0
        else:
            hold_count += 1

        pos[i] = current_pos

    return pd.Series(pos)


def build_mixed_position(
    y_pred:         np.ndarray,
    y_proba:        np.ndarray,
    regime:         np.ndarray,
    regime_scalars: dict  = MIXED_REGIME_SCALARS,
    conf_by_regime: dict  = MIXED_CONF_BY_REGIME,
    hysteresis:     float = MIXED_HYSTERESIS,
    min_hold:       int   = MIXED_MIN_HOLD,
) -> pd.Series:
    """
    Strategy E: Confidence × Regime (graduated, adaptive threshold).

    - position = direction × confidence × regime_scalar
    - Confidence threshold is regime-specific (higher bar in high-vol).
    - Ensures low-vol regime trades more freely, high-vol regime only
      trades when the model is highly confident.
    """
    n = len(y_pred)
    pos = np.zeros(n)
    current_pos, hold_count = 0.0, 0

    for i in range(n):
        pred    = y_pred[i]
        cls_idx = int(pred) + 1
        prob    = y_proba[i, cls_idx]
        reg     = int(regime[i])

        threshold = conf_by_regime.get(reg, 0.45)
        scalar    = regime_scalars.get(reg, 1.0)

        target = float(pred) * prob * scalar if (prob > threshold and pred != 0) else 0.0

        if abs(target - current_pos) > hysteresis and hold_count >= min_hold:
            current_pos = target
            hold_count  = 0
        else:
            hold_count += 1

        pos[i] = current_pos

    return pd.Series(pos)


# ============================================================
# Metrics engine
# ============================================================

def calculate_metrics(
    positions:   pd.Series,
    log_returns: pd.Series,
    label:       str,
    cost_rate:   float = COST_BPS,
) -> dict:
    """
    Computes a full suite of risk/return metrics for a strategy.

    Parameters
    ----------
    positions   : Raw (unshifted) position series aligned to log_returns.
    log_returns : Daily log-returns of the underlying.
    label       : Human-readable strategy name.
    cost_rate   : Transaction cost per unit of position change (decimal).

    Returns
    -------
    dict with keys: Strategy, Return %, Ann Ret %, Sharpe, Sortino,
    Win Rate %, Max DD %, Avg Exposure, Trades, Equity, _net_ret, _exec_pos.
    """
    exec_pos = positions.shift(1).fillna(0)
    exec_pos.index = log_returns.index

    costs   = exec_pos.diff().abs().fillna(0) * cost_rate
    net_ret = (exec_pos * log_returns) - costs
    cum_ret = net_ret.cumsum().apply(np.exp)

    total_ret = cum_ret.iloc[-1] - 1
    ann_ret   = net_ret.mean() * 365
    ann_vol   = net_ret.std() * ANNUAL_FACTOR
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0
    downside  = net_ret[net_ret < 0].std() * ANNUAL_FACTOR
    sortino   = ann_ret / downside if downside > 0 else 0.0
    active    = net_ret[exec_pos != 0]
    win_rate  = (active > 0).mean() if len(active) > 0 else 0.0
    peak      = cum_ret.cummax()
    mdd       = ((cum_ret - peak) / peak).min()
    n_trades  = int((costs > 0).sum())
    avg_exp   = exec_pos.abs().mean()

    return {
        "Strategy"    : label,
        "Return %"    : round(total_ret * 100, 3),
        "Ann Ret %"   : round(ann_ret   * 100, 3),
        "Sharpe"      : round(sharpe,           3),
        "Sortino"     : round(sortino,          3),
        "Win Rate %"  : round(win_rate  * 100,  2),
        "Max DD %"    : round(mdd       * 100,  3),
        "Avg Exposure": round(avg_exp,          3),
        "Trades"      : n_trades,
        "Equity"      : cum_ret,
        "_net_ret"    : net_ret,
        "_exec_pos"   : exec_pos,
    }


# ============================================================
# Regime breakdown
# ============================================================

def regime_breakdown(
    result:        dict,
    log_returns:   pd.Series,
    regime_array:  np.ndarray,
    label:         str,
    silent:        bool = False,
) -> pd.DataFrame:
    """
    Per-regime trade count, win rate, avg P&L per trade, and
    regime-specific Sharpe.  Used to verify that overall Sharpe
    gains come from genuine edge, not from simply staying flat.

    Parameters
    ----------
    silent : If True, returns the DataFrame without printing anything.
             Caller is responsible for display.
    """
    net_ret  = result["_net_ret"]
    exec_pos = result["_exec_pos"]
    names    = {0: "Low Vol", 1: "Mid Vol", 2: "High Vol"}
    rows     = []

    for reg_id in [0, 1, 2]:
        mask    = pd.Series(regime_array, index=log_returns.index) == reg_id
        active  = exec_pos[mask] != 0
        trades  = active.sum()

        if trades == 0:
            rows.append({"Regime": names[reg_id], "Trades": 0,
                         "Win Rate %": np.nan, "Avg P&L/Trade": np.nan,
                         "Regime Sharpe": np.nan})
            continue

        r_slice  = net_ret[mask]
        r_active = r_slice[exec_pos[mask] != 0]
        win_rate = (r_active > 0).mean() * 100
        avg_pnl  = r_active.mean() * 100
        vol_r    = r_slice.std() * ANNUAL_FACTOR
        ann_r    = r_slice.mean() * 365
        sharpe_r = ann_r / vol_r if vol_r > 0 else np.nan

        rows.append({
            "Regime"        : names[reg_id],
            "Trades"        : int(trades),
            "Win Rate %"    : round(win_rate, 1),
            "Avg P&L/Trade" : round(avg_pnl,  4),
            "Regime Sharpe" : round(sharpe_r, 3) if not np.isnan(sharpe_r) else np.nan,
        })

    df = pd.DataFrame(rows)

    if not silent:
        print(f"\n  Regime breakdown — {label}")
        print(f"  {'-'*60}")
        print(df.to_string(index=False))

    return df