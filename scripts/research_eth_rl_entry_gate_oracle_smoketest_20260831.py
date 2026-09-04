#!/usr/bin/env python3
"""Stage 0 "oracle smoke test" for the RL-entry-gate design (docs/eth_rl_autotrading_agent_design_
20260831.md, Section 8, "0단계"). Diagnostic/exploratory research only -- NOT a promotion claim.
No live code changes, no TabM/GPU training. Checks whether a richer supervised state (regime probs
+ raw microstructure/market context + [v1] Homer evidence-signal rule triggers + the direction
head's own output) can beat the existing live entry quality-gate (`quality_for_action >= 0.50` for
ETH's h48qual component) at separating good raw direction-head candidates from bad ones.

Candidate pool: every row in the h48qual bundle's precomputed OOF prediction export where
`dir_action != 0` (the direction head proposed a trade), regardless of whether the existing
quality gate would have accepted it -- both the existing-gate rule and the new classifier(s) are
evaluated as decision rules over this SAME pool, so the comparison is apples-to-apples.

Oracle label: simulate forward from next-bar open using the ATR-adaptive TP/SL barrier
(atr_pct(window=192)*12.0/6.0, floored/capped per omega4_6_1_live.py's _ComponentConfig defaults),
walking intrabar high/low with SL checked before TP (matches evaluate_exit's live convention).
label = 1 iff the trade's raw (pre-cost) price-move return is positive -- this is exactly
equivalent to "TP hit, or (timeout AND net return > 0)" since a resolved SL always returns a
negative price-move and a resolved TP always returns a positive one by construction (see
core.causal_futures_backtest._resolve_trade).

Split discipline: fit + calibrate everything on TRAIN only (with a chronological, embargoed
internal holdout inside TRAIN for probability-threshold calibration). VALIDATION is scored exactly
once, at the end, for both v0 and v1 (two pre-specified, non-iterative variants) -- never touched
during fitting/tuning. OOS and HOLDOUT are never loaded by this script.

Outputs (this run): tmp/causal_regen_20260516/eth_rl_entry_gate_oracle_smoketest_20260831/
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

from core.causal_futures_backtest import simulate_single_position  # noqa: E402
from scripts.eval_omega4_1_atr_safety_sltp_20260622 import _atr_pct  # noqa: E402
from scripts.live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from scripts.retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------------------------
MANIFEST_PATH = ROOT / "docs/model_contracts/CURRENT_LIVE_MANIFEST.json"
REGIME_MODEL_PATH = ROOT / "tmp/eth_regime_gbm3_independent_20260826/model.joblib"
ETH_KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_KLINES_PATH = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
TRAINING_FEATURES_PATHS = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_rl_entry_gate_oracle_smoketest_20260831"

# ATR-adaptive TP/SL -- omega4_6_1_live.py _ComponentConfig defaults (CLAUDE.md Position-Feature
# Parity Contract / Futures Risk Sizing Contract). Side-independent price-move fractions.
ATR_WINDOW = 192
TP_MULT = 12.0
SL_MULT = 6.0
MIN_TP = 0.075
MIN_SL = 0.040
MAX_TP = 0.22
MAX_SL = 0.12

# Max-hold-bars: documented default from scripts/train_eval_omega4_3head_parent72_loose_entry_
# quality_20260620.py's `--quality-max-hold-bars` argparse default (288) -- the training script
# for this exact h48qual bundle family. Also matches the task's own suggested 288=24h fallback.
MAX_HOLD_BARS = 288

# Standard taker roundtrip cost, no fee-discount assumption (feedback_no_fee_discount_assumptions).
ROUNDTRIP_COST = 0.0010

EXISTING_GATE_THRESHOLD = 0.50  # h48qual q050 live quality gate

INTERNAL_HOLDOUT_FRAC = 0.20
EMBARGO_BARS = MAX_HOLD_BARS  # purge/embargo width = the label's own max forward horizon

RANDOM_SEED = 20260831

HGB_PARAMS = dict(
    max_depth=8,
    learning_rate=0.05,
    max_iter=300,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=RANDOM_SEED,
)

# v0 feature set -----------------------------------------------------------------------------
PREDICTION_FEATURES = [
    "dir_p_cash", "dir_p_long", "dir_p_short", "dir_confidence",
    "dir_side_edge", "dir_trade_prob", "quality_for_action",
]
REGIME_FEATURES = ["regime_bull_prob", "regime_bear_prob", "regime_chop_prob"]
MARKET_CONTEXT_RAW_COLS = [
    "log_return", "realized_vol_ratio", "garman_klass_vol", "atr_pct_rank_288", "bb_width_z",
    "hour_sin", "hour_cos", "session_europe", "session_us",
    "net_taker_ratio", "taker_acceleration", "cvd_slope_48", "price_cvd_divergence",
    "oi_change_rate", "funding_oi_divergence", "last_funding_rate", "funding_z_score",
    "btc_corr_60", "chop_index",
]
RETURN_HORIZONS = [1, 3, 6, 12, 24]
MARKET_CONTEXT_COMPUTED_COLS = [f"ret_{k}" for k in RETURN_HORIZONS] + ["atr_pct_192"]

V0_FEATURE_COLS = PREDICTION_FEATURES + REGIME_FEATURES + MARKET_CONTEXT_RAW_COLS + MARKET_CONTEXT_COMPUTED_COLS

# v1 adds 6 of the 8 Homer evidence-signal RAW rule triggers (bottom_*/top_*, not the sustained
# `_active` display columns and not the TabPFN-calibrated proba -- that's a v2 refinement).
# demarker_extreme / kalman_deviation_meanrev skipped per task spec (training data not local).
V1_SIGNAL_NAMES = [
    "orthogonal_combo", "fib_extension_exhaustion", "smt_divergence",
    "liquidity_sweep", "short_term_return_z", "taker_delta_z_climax",
]
V1_SIGNAL_COLS = [f"sig_{name}_{side}" for name in V1_SIGNAL_NAMES for side in ("bottom", "top")]
V1_FEATURE_COLS = V0_FEATURE_COLS + V1_SIGNAL_COLS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------
# Step 1: manifest + prediction CSVs
# --------------------------------------------------------------------------------------------
def load_h48qual_bundle_dir() -> Path:
    manifest = json.loads(MANIFEST_PATH.read_text())
    rel = manifest["artifacts"]["eth"]["h48qual_bundle"]["path"]
    quality_threshold = manifest["artifacts"]["eth"]["h48qual_sidecar"]["quality_threshold"]
    assert abs(float(quality_threshold) - EXISTING_GATE_THRESHOLD) < 1e-9, (
        f"manifest quality_threshold {quality_threshold} != assumed {EXISTING_GATE_THRESHOLD}"
    )
    bundle_dir = (ROOT / rel).parent
    assert bundle_dir.is_dir(), f"bundle dir not found: {bundle_dir}"
    return bundle_dir


def detect_prefix(df: pd.DataFrame) -> str:
    hits = [c for c in df.columns if c.endswith("router_expert")]
    assert len(hits) == 1, f"expected exactly 1 router_expert column, found {hits}"
    return hits[0][: -len("router_expert")]


def load_prediction_csv(path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    prefix = detect_prefix(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    rename = {
        f"{prefix}dir_p_cash": "dir_p_cash", f"{prefix}dir_p_long": "dir_p_long",
        f"{prefix}dir_p_short": "dir_p_short", f"{prefix}dir_confidence": "dir_confidence",
        f"{prefix}dir_side_edge": "dir_side_edge", f"{prefix}dir_trade_prob": "dir_trade_prob",
        f"{prefix}dir_action": "dir_action", f"{prefix}quality_for_action": "quality_for_action",
        f"{prefix}quality_threshold": "quality_threshold", f"{prefix}final_action": "final_action",
        f"{prefix}router_expert": "router_expert",
    }
    df = df.rename(columns=rename)
    return df, prefix


def spot_check_threshold_export_consistency(bundle_dir: Path) -> dict:
    """Verify dir_p_cash/long/short, dir_action etc. are identical across the two available
    differently-thresholded exports (q045 vs q050) for the VALIDATION split -- only the gate
    threshold / final_action should differ."""
    p045 = bundle_dir / "validation_predictions_q045.csv"
    p050 = bundle_dir / "validation_predictions_q050.csv"
    if not p045.exists() or not p050.exists():
        return {"performed": False, "reason": "second qXXX export not found"}
    a, prefix_a = load_prediction_csv(p045)
    b, prefix_b = load_prediction_csv(p050)
    merged = a.merge(b, on="timestamp", suffixes=("_45", "_50"))
    assert len(merged) == len(a) == len(b), "row count mismatch on timestamp merge"
    numeric_cols = ["dir_p_cash", "dir_p_long", "dir_p_short", "dir_confidence",
                    "dir_side_edge", "dir_trade_prob", "dir_action", "quality_for_action"]
    max_abs_diffs = {c: float((merged[f"{c}_45"] - merged[f"{c}_50"]).abs().max()) for c in numeric_cols}
    router_equal = bool((merged["router_expert_45"].astype(str) == merged["router_expert_50"].astype(str)).all())
    all_identical = all(v < 1e-12 for v in max_abs_diffs.values()) and router_equal
    return {
        "performed": True,
        "n_rows_compared": int(len(merged)),
        "max_abs_diffs": max_abs_diffs,
        "router_expert_identical": router_equal,
        "quality_threshold_45_unique": sorted(merged[f"quality_threshold_45"].unique().tolist()),
        "quality_threshold_50_unique": sorted(merged[f"quality_threshold_50"].unique().tolist()),
        "final_action_differs_count": int((merged["final_action_45"] != merged["final_action_50"]).sum()),
        "verdict_identical_except_threshold_gated_fields": all_identical,
    }


def build_candidate_pool(pred_df: pd.DataFrame) -> pd.DataFrame:
    cand = pred_df[pred_df["dir_action"] != 0].reset_index(drop=True).copy()
    # sanity cross-check of the existing gate's own reproduction logic (documented assumption,
    # not a defect if it ever fails -- would just mean the export's final_action convention
    # differs from what's assumed).
    reproduced = np.where(cand["quality_for_action"].to_numpy() >= cand["quality_threshold"].to_numpy(),
                           cand["dir_action"].to_numpy(), 0)
    mismatch = int((reproduced != cand["final_action"].to_numpy()).sum())
    if mismatch:
        log(f"WARNING: existing-gate reproduction mismatch on {mismatch}/{len(cand)} rows")
    cand["sim_side"] = np.where(cand["dir_action"].to_numpy() == 1, 1, -1)
    return cand


# --------------------------------------------------------------------------------------------
# Step 2: klines + ATR + evidence-signal triggers
# --------------------------------------------------------------------------------------------
def load_klines(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    assert df["timestamp"].is_monotonic_increasing
    return df


def attach_kline_pos_and_atr(cand: pd.DataFrame, eth_klines: pd.DataFrame) -> pd.DataFrame:
    pos_lookup = pd.Series(np.arange(len(eth_klines)), index=eth_klines["timestamp"].to_numpy())
    n_before = len(cand)
    cand = cand.copy()
    cand["kline_pos"] = cand["timestamp"].map(pos_lookup)
    n_unmatched = int(cand["kline_pos"].isna().sum())
    cand = cand.dropna(subset=["kline_pos"]).reset_index(drop=True)
    cand["kline_pos"] = cand["kline_pos"].astype(np.int64)
    cand["atr_pct_192"] = eth_klines["atr_pct_192"].to_numpy()[cand["kline_pos"].to_numpy()]
    cand["tp_move"] = np.clip(np.maximum(MIN_TP, cand["atr_pct_192"].to_numpy() * TP_MULT), 0.0, MAX_TP)
    cand["sl_move"] = np.clip(np.maximum(MIN_SL, cand["atr_pct_192"].to_numpy() * SL_MULT), 0.0, MAX_SL)
    return cand, {"n_before_kline_join": n_before, "n_after_kline_join": len(cand), "n_unmatched_timestamp": n_unmatched}


def simulate_candidates(cand: pd.DataFrame, eth_klines: pd.DataFrame, max_hold_bars: int) -> pd.DataFrame:
    eth_ts = pd.DatetimeIndex(eth_klines["timestamp"])
    eth_open = eth_klines["open"].to_numpy(dtype=np.float64)
    eth_high = eth_klines["high"].to_numpy(dtype=np.float64)
    eth_low = eth_klines["low"].to_numpy(dtype=np.float64)
    eth_close = eth_klines["close"].to_numpy(dtype=np.float64)
    total_bars = len(eth_close)

    n = len(cand)
    reasons = np.array([None] * n, dtype=object)
    price_moves = np.full(n, np.nan, dtype=np.float64)
    bars_held = np.full(n, -1, dtype=np.int64)
    ok = np.zeros(n, dtype=bool)

    kline_pos = cand["kline_pos"].to_numpy()
    sides = cand["sim_side"].to_numpy()
    tp_moves = cand["tp_move"].to_numpy()
    sl_moves = cand["sl_move"].to_numpy()

    t0 = time.time()
    for i in range(n):
        pos = int(kline_pos[i])
        if pos + 1 >= total_bars:
            continue
        end = min(pos + max_hold_bars + 2, total_bars)
        score = 1.0 if sides[i] > 0 else -1.0
        result = simulate_single_position(
            timestamps=eth_ts[pos:end],
            open_px=eth_open[pos:end],
            high=eth_high[pos:end],
            low=eth_low[pos:end],
            close=eth_close[pos:end],
            decision_indices=np.array([0], dtype=np.int64),
            scores=np.array([score], dtype=np.float64),
            tp_moves=np.array([tp_moves[i]], dtype=np.float64),
            sl_moves=np.array([sl_moves[i]], dtype=np.float64),
            upper_threshold=0.0,
            lower_threshold=0.0,
            horizon_bars=max_hold_bars,
            margin_fraction=1.0,
            leverage=1.0,
            roundtrip_cost_rate=0.0,
        )
        if len(result.ledger) != 1:
            continue
        row = result.ledger.iloc[0]
        reasons[i] = row["reason"]
        price_moves[i] = float(row["price_move"])
        bars_held[i] = int(row["bars_held"])
        ok[i] = True
        if (i + 1) % 20000 == 0:
            log(f"  simulated {i + 1}/{n} candidates ({time.time() - t0:.1f}s elapsed)")

    out = cand.copy()
    out["sim_ok"] = ok
    out["reason"] = reasons
    out["price_move_raw"] = price_moves
    out["bars_held"] = bars_held
    out = out[out["sim_ok"]].reset_index(drop=True)
    # oracle label: TP always resolves price_move_raw>0, SL always resolves <0 by construction of
    # _resolve_trade, so "TP hit OR (timeout AND net>0)" collapses to this single sign check.
    out["oracle_label"] = (out["price_move_raw"].to_numpy() > 0.0).astype(np.int64)
    return out


# --------------------------------------------------------------------------------------------
# Step 3: regime GBM3 scoring + raw market-context features
# --------------------------------------------------------------------------------------------
def load_training_features_full() -> pd.DataFrame:
    frames = []
    for p in TRAINING_FEATURES_PATHS:
        df = pd.read_csv(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values("timestamp").reset_index(drop=True)
    dup = int(full["timestamp"].duplicated().sum())
    assert dup == 0, f"{dup} duplicate timestamps in concatenated training_features"
    return full


def score_regime_gbm3(training_features_full: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    payload = joblib.load(REGIME_MODEL_PATH)
    feature_cols = payload["feature_cols"]
    classes = payload["classes"]
    med = pd.Series(payload["feature_medians"])

    missing_before = [c for c in feature_cols if c not in training_features_full.columns]
    feats = _with_raw_state12(training_features_full)  # derives the state7_*/state12_* columns
    missing_after = [c for c in feature_cols if c not in feats.columns]
    for c in missing_after:  # should be empty after _with_raw_state12; defensive fallback only
        feats[c] = med.get(c, 0.0)

    x = feats[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    proba = payload["model"].predict_proba(x)
    regime_df = pd.DataFrame({"timestamp": feats["timestamp"].reset_index(drop=True)})
    for i, cls_name in enumerate(classes):
        regime_df[f"regime_{cls_name}_prob"] = proba[:, i]
    diag = {
        "feature_cols_missing_from_training_features_csv_before_state12_derivation": missing_before,
        "feature_cols_still_missing_after_state12_derivation": missing_after,
        "classes_order": list(classes),
        "model_classes_": payload["model"].classes_.tolist(),
    }
    return regime_df, diag


def build_market_context(training_features_full: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "close"] + MARKET_CONTEXT_RAW_COLS
    missing = [c for c in cols if c not in training_features_full.columns]
    assert not missing, f"missing market-context columns in training_features: {missing}"
    ctx = training_features_full[cols].copy()
    close = ctx["close"]
    for k in RETURN_HORIZONS:
        ctx[f"ret_{k}"] = close.pct_change(k)
    ctx = ctx.drop(columns=["close"])
    return ctx


# --------------------------------------------------------------------------------------------
# Step 4: joins
# --------------------------------------------------------------------------------------------
def join_all_features(
    cand: pd.DataFrame,
    market_context: pd.DataFrame,
    regime_df: pd.DataFrame,
    sig_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict]:
    n0 = len(cand)
    merged = cand.merge(market_context, on="timestamp", how="inner")
    n1 = len(merged)
    merged = merged.merge(regime_df, on="timestamp", how="inner")
    n2 = len(merged)
    join_report = {
        "n_candidates_simulated": n0,
        "n_after_market_context_join": n1,
        "n_after_regime_join": n2,
    }
    if sig_df is not None:
        merged = merged.merge(sig_df, on="timestamp", how="inner")
        join_report["n_after_signal_join"] = len(merged)
    return merged, join_report


def build_signal_features(eth_klines_with_atr: pd.DataFrame, btc_klines: pd.DataFrame) -> pd.DataFrame:
    sig = compute_signals(eth_klines_with_atr, btc_df=btc_klines, funding_df=None)
    out = sig[["timestamp"]].copy()
    for name in V1_SIGNAL_NAMES:
        out[f"sig_{name}_bottom"] = sig[f"bottom_{name}"].fillna(False).astype(int)
        out[f"sig_{name}_top"] = sig[f"top_{name}"].fillna(False).astype(int)
    return out


# --------------------------------------------------------------------------------------------
# Step 5: model fit / calibrate / evaluate
# --------------------------------------------------------------------------------------------
def internal_train_holdout_split(df: pd.DataFrame, holdout_frac: float, embargo_bars: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    split_pos = int(n * (1.0 - holdout_frac))
    holdout_start_ts = df_sorted.loc[split_pos, "timestamp"]
    embargo_start_ts = holdout_start_ts - pd.Timedelta(minutes=5 * embargo_bars)
    fit_df = df_sorted[df_sorted["timestamp"] < embargo_start_ts].reset_index(drop=True)
    holdout_df = df_sorted[df_sorted["timestamp"] >= holdout_start_ts].reset_index(drop=True)
    info = {
        "n_total": n,
        "holdout_start_ts": str(holdout_start_ts),
        "embargo_start_ts": str(embargo_start_ts),
        "n_fit": len(fit_df),
        "n_holdout": len(holdout_df),
        "n_embargoed_dropped": int(n - len(fit_df) - len(holdout_df)),
    }
    return fit_df, holdout_df, info


def fit_hgb(x: pd.DataFrame, y: np.ndarray) -> HistGradientBoostingClassifier:
    w = compute_sample_weight("balanced", y)
    model = HistGradientBoostingClassifier(**HGB_PARAMS)
    model.fit(x, y, sample_weight=w)
    return model


def calibrate_threshold_matching_gate_rate(probs: np.ndarray, target_accept_rate: float) -> float:
    if not (0.0 < target_accept_rate < 1.0):
        return float(np.median(probs))
    return float(np.quantile(probs, 1.0 - target_accept_rate))


def prep_x(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def evaluate_rule(accept: np.ndarray, oracle_label: np.ndarray, price_move_raw: np.ndarray,
                   score: np.ndarray | None) -> dict:
    n_total = len(oracle_label)
    n_accept = int(accept.sum())
    out = {
        "n_total_candidates": int(n_total),
        "n_accepted": n_accept,
        "accept_rate": float(n_accept / n_total) if n_total else None,
        "accuracy_vs_oracle": float(accuracy_score(oracle_label, accept.astype(int))),
    }
    if score is not None:
        try:
            out["auc_vs_oracle"] = float(roc_auc_score(oracle_label, score))
        except ValueError as e:
            out["auc_vs_oracle"] = None
            out["auc_error"] = str(e)
    if n_accept > 0:
        acc_label = oracle_label[accept]
        acc_net = price_move_raw[accept] - ROUNDTRIP_COST
        out["win_rate_accepted"] = float(acc_label.mean())
        out["avg_net_return_accepted_bp"] = float(acc_net.mean() * 10000.0)
        out["median_net_return_accepted_bp"] = float(np.median(acc_net) * 10000.0)
        out["sum_net_return_accepted_bp"] = float(acc_net.sum() * 10000.0)
    else:
        out["win_rate_accepted"] = None
        out["avg_net_return_accepted_bp"] = None
    return out


def run_variant(name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    log(f"=== variant {name}: feature_cols={len(feature_cols)} ===")
    fit_df, holdout_df, split_info = internal_train_holdout_split(train_df, INTERNAL_HOLDOUT_FRAC, EMBARGO_BARS)
    log(f"  internal split: fit={split_info['n_fit']} holdout={split_info['n_holdout']} embargoed={split_info['n_embargoed_dropped']}")

    x_fit = prep_x(fit_df, feature_cols)
    y_fit = fit_df["oracle_label"].to_numpy()
    model_a = fit_hgb(x_fit, y_fit)

    x_hold = prep_x(holdout_df, feature_cols)
    hold_probs = model_a.predict_proba(x_hold)[:, 1]
    gate_accept_rate_holdout = float((holdout_df["quality_for_action"].to_numpy() >= EXISTING_GATE_THRESHOLD).mean())
    threshold = calibrate_threshold_matching_gate_rate(hold_probs, gate_accept_rate_holdout)
    log(f"  calibrated threshold={threshold:.4f} (target accept rate {gate_accept_rate_holdout:.4f} from holdout gate rate)")

    # Permutation importance on the (untouched-by-refit) internal holdout, using model_a.
    try:
        pi = permutation_importance(model_a, x_hold, holdout_df["oracle_label"].to_numpy(),
                                     n_repeats=5, random_state=RANDOM_SEED, scoring="roc_auc")
        importance_pairs = sorted(zip(feature_cols, pi.importances_mean.tolist()), key=lambda t: -t[1])
        top_importance = [{"feature": f, "importance_mean_auc_drop": v} for f, v in importance_pairs[:15]]
    except Exception as e:  # pragma: no cover - diagnostic only
        top_importance = None
        log(f"  permutation_importance failed: {e}")

    # Refit on the FULL TRAIN pool (fit + holdout) with the threshold already frozen.
    x_full = prep_x(train_df, feature_cols)
    y_full = train_df["oracle_label"].to_numpy()
    model_final = fit_hgb(x_full, y_full)

    x_val = prep_x(val_df, feature_cols)
    val_probs = model_final.predict_proba(x_val)[:, 1]
    val_accept_model = val_probs >= threshold

    oracle_val = val_df["oracle_label"].to_numpy()
    price_move_val = val_df["price_move_raw"].to_numpy()
    gate_accept_val = val_df["quality_for_action"].to_numpy() >= EXISTING_GATE_THRESHOLD
    quality_score_val = val_df["quality_for_action"].to_numpy()

    metrics_gate = evaluate_rule(gate_accept_val, oracle_val, price_move_val, quality_score_val)
    metrics_model = evaluate_rule(val_accept_model, oracle_val, price_move_val, val_probs)
    metrics_oracle_ceiling = evaluate_rule(oracle_val.astype(bool), oracle_val, price_move_val, None)

    return {
        "variant": name,
        "feature_cols": feature_cols,
        "internal_split": split_info,
        "threshold_calibrated_on_train_holdout": threshold,
        "gate_accept_rate_on_train_holdout_used_for_calibration": gate_accept_rate_holdout,
        "top_permutation_importance_on_train_holdout": top_importance,
        "validation_metrics": {
            "existing_quality_gate_ge_0p50": metrics_gate,
            f"classifier_{name}": metrics_model,
            "oracle_ceiling_reference_hindsight_cheating": metrics_oracle_ceiling,
        },
        "validation_scored_rows": pd.DataFrame({
            "timestamp": val_df["timestamp"].to_numpy(),
            "oracle_label": oracle_val,
            "price_move_raw": price_move_val,
            "quality_for_action": quality_score_val,
            "gate_accept": gate_accept_val,
            f"{name}_prob": val_probs,
            f"{name}_accept": val_accept_model,
        }),
    }


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "script": "scripts/research_eth_rl_entry_gate_oracle_smoketest_20260831.py",
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "design_doc": "docs/eth_rl_autotrading_agent_design_20260831.md (Section 5, Section 8 '0단계')",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "assumptions": [],
    }

    log("Step 1: manifest + prediction CSVs")
    bundle_dir = load_h48qual_bundle_dir()
    report["h48qual_bundle_dir"] = str(bundle_dir.relative_to(ROOT))

    spot_check = spot_check_threshold_export_consistency(bundle_dir)
    report["q045_vs_q050_export_consistency_check"] = spot_check
    if spot_check.get("performed") and not spot_check.get("verdict_identical_except_threshold_gated_fields"):
        log("WARNING: q045 vs q050 export consistency check FAILED -- see report")
    else:
        log(f"  consistency check: {spot_check.get('verdict_identical_except_threshold_gated_fields')}")

    train_pred, _ = load_prediction_csv(bundle_dir / "train_predictions_q050.csv")
    val_pred, _ = load_prediction_csv(bundle_dir / "validation_predictions_q050.csv")
    report["train_predictions_ts_range"] = [str(train_pred["timestamp"].min()), str(train_pred["timestamp"].max())]
    report["validation_predictions_ts_range"] = [str(val_pred["timestamp"].min()), str(val_pred["timestamp"].max())]
    report["train_predictions_total_rows"] = int(len(train_pred))
    report["validation_predictions_total_rows"] = int(len(val_pred))

    train_cand_raw = build_candidate_pool(train_pred)
    val_cand_raw = build_candidate_pool(val_pred)
    report["raw_candidate_pool"] = {
        "train_n_candidates_dir_action_ne_cash": int(len(train_cand_raw)),
        "train_dir_action_value_counts": train_pred["dir_action"].value_counts().to_dict(),
        "validation_n_candidates_dir_action_ne_cash": int(len(val_cand_raw)),
        "validation_dir_action_value_counts": val_pred["dir_action"].value_counts().to_dict(),
    }
    log(f"  TRAIN candidates (dir_action!=cash): {len(train_cand_raw)} / {len(train_pred)} total bars")
    log(f"  VALIDATION candidates (dir_action!=cash): {len(val_cand_raw)} / {len(val_pred)} total bars")

    log("Step 2: klines + ATR")
    eth_klines = load_klines(ETH_KLINES_PATH)
    btc_klines = load_klines(BTC_KLINES_PATH)
    eth_klines["atr_pct_192"] = _atr_pct(eth_klines, window=ATR_WINDOW)
    report["klines"] = {
        "eth_rows": int(len(eth_klines)),
        "eth_ts_range": [str(eth_klines["timestamp"].min()), str(eth_klines["timestamp"].max())],
        "btc_rows": int(len(btc_klines)),
        "btc_ts_range": [str(btc_klines["timestamp"].min()), str(btc_klines["timestamp"].max())],
    }

    train_cand, train_kline_join = attach_kline_pos_and_atr(train_cand_raw, eth_klines)
    val_cand, val_kline_join = attach_kline_pos_and_atr(val_cand_raw, eth_klines)
    report["kline_join"] = {"train": train_kline_join, "validation": val_kline_join}

    log("Step 3: simulate TP/SL outcomes (this can take a while)")
    train_sim = simulate_candidates(train_cand, eth_klines, MAX_HOLD_BARS)
    log(f"  TRAIN simulated: {len(train_sim)} / {len(train_cand)} candidates resolved")
    val_sim = simulate_candidates(val_cand, eth_klines, MAX_HOLD_BARS)
    log(f"  VALIDATION simulated: {len(val_sim)} / {len(val_cand)} candidates resolved")

    def sanity_rates(df: pd.DataFrame) -> dict:
        n = len(df)
        vc = df["reason"].value_counts()
        return {
            "n": int(n),
            "tp_rate": float(vc.get("tp", 0) / n) if n else None,
            "sl_rate": float(vc.get("sl", 0) / n) if n else None,
            "timeout_rate": float(vc.get("timeout", 0) / n) if n else None,
            "oracle_label_positive_rate": float(df["oracle_label"].mean()) if n else None,
            "bars_held_p50": float(df["bars_held"].median()) if n else None,
            "bars_held_p90": float(df["bars_held"].quantile(0.90)) if n else None,
        }

    report["sanity_check_simulation_rates"] = {
        "train": sanity_rates(train_sim),
        "validation": sanity_rates(val_sim),
        "historical_report_rough_reference_shape_not_expected_to_match": {
            "source": "docs/experiments/omega1_2_quality_gate_rl_problem_report_20260618.md (different bundle/threshold, sanity shape only)",
            "n_candidates": 20071, "tp_rate": 0.3135, "sl_rate": 0.5578,
        },
    }
    log(f"  TRAIN sanity: {report['sanity_check_simulation_rates']['train']}")
    log(f"  VAL sanity:   {report['sanity_check_simulation_rates']['validation']}")

    log("Step 4: training_features (regime scoring + market context)")
    training_features_full = load_training_features_full()
    regime_df, regime_diag = score_regime_gbm3(training_features_full)
    report["regime_gbm3_scoring_diagnostics"] = regime_diag
    market_context = build_market_context(training_features_full)
    report["market_context_columns"] = {
        "from_training_features_csv": MARKET_CONTEXT_RAW_COLS,
        "computed_from_close": MARKET_CONTEXT_COMPUTED_COLS,
    }

    log("Step 4b: v1 evidence-signal triggers (compute_signals, vectorized over full ETH history)")
    sig_df = build_signal_features(eth_klines, btc_klines)
    for name in V1_SIGNAL_NAMES:
        report.setdefault("v1_signal_fire_counts_full_history", {})[name] = {
            "bottom": int(sig_df[f"sig_{name}_bottom"].sum()),
            "top": int(sig_df[f"sig_{name}_top"].sum()),
        }

    log("Step 5: join features")
    train_v0, train_join_report = join_all_features(train_sim, market_context, regime_df, None)
    val_v0, val_join_report_v0 = join_all_features(val_sim, market_context, regime_df, None)
    report["feature_join"] = {"train_v0": train_join_report, "validation_v0": val_join_report_v0}

    train_v1, train_join_report_v1 = join_all_features(train_sim, market_context, regime_df, sig_df)
    val_v1, val_join_report_v1 = join_all_features(val_sim, market_context, regime_df, sig_df)
    report["feature_join"]["train_v1"] = train_join_report_v1
    report["feature_join"]["validation_v1"] = val_join_report_v1

    log(f"  TRAIN v0 modeling pool: {len(train_v0)}; VALIDATION v0 modeling pool: {len(val_v0)}")
    log(f"  TRAIN v1 modeling pool: {len(train_v1)}; VALIDATION v1 modeling pool: {len(val_v1)}")

    # Save labeled candidate tables (v1 superset columns) for user spot-checking.
    keep_cols_meta = ["timestamp", "dir_action", "sim_side", "tp_move", "sl_move", "bars_held",
                       "reason", "price_move_raw", "oracle_label"]
    train_v1[keep_cols_meta + V1_FEATURE_COLS].to_csv(OUT_DIR / "candidates_train_labeled.csv", index=False)
    val_v1[keep_cols_meta + V1_FEATURE_COLS].to_csv(OUT_DIR / "candidates_validation_labeled.csv", index=False)

    log("Step 6: sanity gate before proceeding to modeling")
    train_sane = 0.05 < report["sanity_check_simulation_rates"]["train"]["tp_rate"] < 0.95
    val_sane = len(val_v0) > 200 and len(train_v0) > 1000
    report["v0_sanity_gate_passed"] = bool(train_sane and val_sane)
    if not report["v0_sanity_gate_passed"]:
        report["assumptions"].append("v0 sanity gate FAILED -- see sanity_check_simulation_rates / feature_join for diagnosis")
        log("SANITY GATE FAILED -- writing partial report and stopping before modeling")
        (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
        return

    log("Step 7: v0 model (no evidence-signal features)")
    v0_result = run_variant("v0", train_v0, val_v0, V0_FEATURE_COLS)
    v0_result["validation_scored_rows"].to_csv(OUT_DIR / "validation_v0_scored.csv", index=False)
    v0_summary = {k: v for k, v in v0_result.items() if k != "validation_scored_rows"}
    report["v0"] = v0_summary

    log("Step 8: v1 model (adds 6 evidence-signal rule triggers)")
    v1_result = run_variant("v1", train_v1, val_v1, V1_FEATURE_COLS)
    v1_result["validation_scored_rows"].to_csv(OUT_DIR / "validation_v1_scored.csv", index=False)
    v1_summary = {k: v for k, v in v1_result.items() if k != "validation_scored_rows"}
    report["v1"] = v1_summary

    report["assumptions"] = [
        "Oracle label uses PRE-COST (raw) price-move return sign; the 10bp roundtrip cost is applied only "
        "at the metrics stage (avg_net_return_accepted_bp), not baked into the label itself.",
        f"max_hold_bars={MAX_HOLD_BARS} (24h) taken from scripts/train_eval_omega4_3head_parent72_loose_entry_"
        "quality_20260620.py's --quality-max-hold-bars argparse default -- the training script for this exact "
        "h48qual bundle family -- not an arbitrary fallback.",
        "TRAIN/VALIDATION here are this bundle's OWN precomputed OOF export split (TRAIN=2025-01-01..2025-09-30, "
        "VALIDATION=2025-10-01..2025-12-31), NOT the CLAUDE.md standard Fresh-Forward split (VAL 2025-09-01..12-31)"
        " -- flagged since Sept 2025 falls in TRAIN here, not VALIDATION.",
        "orthogonal_combo's bottom leg used funding_df=None (delta_z-only pre-2026-08-27 formula) since the "
        "preprocessed funding_z cache (fetch_funding_history's own format) wasn't reconstructed for this smoke "
        "test; top leg is unaffected either way.",
        "Regime GBM3's 8 state7_*/state12_* feature_cols (not present verbatim in training_features_*.csv) were "
        "derived via scripts.retrain_clean_regime_hmm_raw_state12_20260517._with_raw_state12 (the same function "
        "the live regime scorer scripts/live_regime_gbm3_signal_20260826.py uses after its 2026-08-26 bugfix), "
        "not median-filled -- median-fill was the prior, explicitly-flagged-as-buggy live fallback.",
        "Market-context feature list is a deliberately-scoped ~25-column subset (19 from training_features_*.csv "
        "+ 5 computed multi-horizon returns + the same atr_pct_192 that drives TP/SL sizing), documented in "
        "report['market_context_columns'] -- not exhaustive over all 142 columns.",
        "v0 and v1 are two fixed, pre-specified pipelines each run and VALIDATION-scored exactly once in this "
        "single script pass; VALIDATION was not used for any fitting/threshold-tuning decision, and no "
        "adjustment loop was run after seeing either variant's VALIDATION numbers.",
        "Threshold calibration matches the classifier's accept rate on the internal TRAIN holdout to the "
        "existing gate's accept rate on that SAME holdout slice, then freezes that threshold for VALIDATION.",
        "HistGradientBoostingClassifier used with a single fixed, unduned hyperparameter set (no HP search); "
        "NaNs passed through natively (sklearn HGB's built-in missing-value handling), not imputed.",
    ]

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"Done. report.json + CSVs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
