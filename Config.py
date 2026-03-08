# ============================================================
# config.py — Central configuration for the BTC ML pipeline
# ============================================================

# ── Data ────────────────────────────────────────────────────
SYMBOL          = "BTCUSDT"
INTERVAL        = "1d"
DATA_START      = "2020-01-01"
RAW_DATA_PATH   = "bitcoin_daily_data.csv"

# ── Train / Test split ───────────────────────────────────────
TRAIN_END_DATE  = "2024-12-17"
TEST_START_DATE = "2024-12-18"

# ── Labels ──────────────────────────────────────────────────
THRESHOLD_PERCENTILE = 70       # DTA threshold (pct of abs future return)

# ── Feature selection ───────────────────────────────────────
CORRELATION_THRESHOLD   = 0.9
MDI_THRESHOLD           = 0.005
PI_THRESHOLD            = -0.005
FEATURE_SEL_VAL_FRAC    = 0.2
FEATURE_SEL_RF_TREES    = 300
FEATURE_SEL_RF_DEPTH    = 8
FEATURE_SEL_RF_MIN_LEAF = 10

# ── HMM regime detection ────────────────────────────────────
N_REGIMES    = 3
REGIME_NAMES = {0: "Low Vol", 1: "Mid Vol", 2: "High Vol"}
REGIME_COLS_KW = ['vol', 'vix', 'fg', 'momentum_7d', 'momentum_21d', 'bb_zscore']

# ── Optuna HPO ───────────────────────────────────────────────
HPO_N_TRIALS    = 60
HPO_VAL_FRAC    = 0.2
RANDOM_STATE    = 42

# ── Walk-forward validation ──────────────────────────────────
WFV_INITIAL_TRAIN_MONTHS = 24
WFV_STEP_MONTHS          = 3
WFV_THRESHOLD_PCT        = 75

# ── Backtesting ──────────────────────────────────────────────
COST_BPS         = 0.001
ANNUAL_FACTOR    = 365 ** 0.5
CONF_THRESHOLD   = 0.45
HYSTERESIS_ZONE  = 0.15
MIN_HOLD_TIME    = 1

# Strategy E knobs
MIXED_REGIME_SCALARS  = {0: 1.15, 1: 1.0, 2: 0.9}
MIXED_CONF_BY_REGIME  = {0: 0.40, 1: 0.45, 2: 0.55}
MIXED_HYSTERESIS      = 0.12
MIXED_MIN_HOLD        = 1