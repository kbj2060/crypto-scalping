#!/usr/bin/env python3
"""Stage 0 "oracle smoke test" for the RL-exit-gate design (docs/eth_rl_autotrading_agent_design_
20260831.md, Section 8, "0단계" -- EXIT sub-task). Diagnostic/exploratory research only -- NOT a
promotion claim. No live code changes, no TabM/GPU training, no live exit_head replay.

Companion to scripts/research_eth_rl_entry_gate_oracle_smoketest_20260831.py (run 2026-08-31,
entry sub-task, FAILED -- 6th failure of that axis). That run left the exit sub-task unexecuted
(design doc Section 10 point 5 flagged the exit oracle-label definition as an open decision). This
script resolves it and runs it, on the SAME candidate population and TRAIN/VALIDATION split the
entry script already produced and saved to disk.

Question: does a supervised classifier over position-context (Tier D) + market-context/regime/
evidence-signal (Tier A/B) state, trained to predict "would exiting right now beat riding this
trade to its already-realized outcome", beat a naive **no-early-exit** policy (always ride every
candidate to its own realized TP/SL/timeout terminal outcome) on VALIDATION?

Candidate population: reuses candidates_{train,validation}_labeled.csv from the entry smoke test's
OUT_DIR verbatim -- i.e. the SAME full raw dir_action!=0 pool (not filtered by the existing quality
gate), for the same apples-to-apples-with-existing-mechanism reason the entry script gave. Metrics
are additionally reported on the quality_for_action>=0.50 (existing-gate-approved) subset, since
that is the population that would actually reach an open position live.

Checkpoint sampling (NEW methodology decision, since exit is a sequential per-bar decision and
every candidate's realized bars_held sums to >15M bar-instances across TRAIN alone -- too large for
a "cheap stage-0" budget): for each candidate with realized bars_held=H, sample up to 9 interior
decile checkpoints t=round(H*k/10) for k=1..9, deduplicated, restricted to 1<=t<H (t=0 is the entry
bar itself -- no decision to make there; t=H is the already-realized terminal bar -- also not an
"early" exit). Candidates with H<=1 contribute zero checkpoint rows (there is no early-exit
opportunity in a 1-bar trade) -- this is a correct consequence of the population, not a bug.

Oracle label (closed-form, no new simulation needed -- reuses the ALREADY-COMPUTED terminal
price_move_raw from the entry script's simulation): at checkpoint t, oracle_exit_label_t = 1 iff
unrealized_move_t (closing right now, side-adjusted so positive=profit exactly like price_move_raw)
> price_move_raw (the realized outcome of riding this exact candidate all the way to its own
TP/SL/timeout resolution). This directly answers "would cashing out here have beaten what actually
happened" -- it needs no new hyperparameter and ties to the same barrier mechanics price_move_raw
already used.

unrealized_move_t / mfe_t / mae_t use the EXACT price-move convention core.causal_futures_backtest.
_resolve_trade uses (entry price = ETH open at kline_pos+1, i.e. one bar after the candidate's own
signal bar; side-adjusted so LONG=close/entry-1, SHORT=1-close/entry) and the EXACT mfe/mae update
rule trading_bot.py uses live (mfe=max(0, running max), mae=min(0, running min); trading_bot.py:
9179-9180) -- both cumulative from an implicit 0.0 anchor at the entry bar itself, matching
CLAUDE.md's Position-Feature Train/Inference Parity Contract (raw, unscaled price-move fractions,
NOT multiplied by notional/leverage).

pos_notional/pos_leverage/pos_exposure: NOT sourced from the real per-candidate risk sidecar --
using it faithfully needs the exact base_cols frame + regime3_current wide24 expert-routing state
the live Omega461LiveAdapter builds internally, which is a materially larger, more failure-prone
lift than this stage-0 budget justifies (and getting the routing subtly wrong would silently
corrupt results worse than a documented constant). Falls back to the h48qual component's own fixed
BASE_TEMPLATE (notional=0.45, leverage=2.0, train_eval_omega1_2_tabm_diffusion_risk_20260603.
BASE_TEMPLATE) WITHOUT the regime-expert EXPERT_SCALES multiplier, held constant across every bar
of a given candidate's hold (sizing is an entry-time decision in this architecture, not re-decided
bar to bar) -- flagged per CLAUDE.md's Position-Feature Parity Contract via risk_sizing_source in
this file's docstring and report.json.

Existing-baseline comparison: docs/experiments/omega1_2_quality_gate_rl_problem_report_20260618.md
section 6's "Exit-head-only selected" (Validation PnL 0.46, OOS PnL -6.05, OOS WR 0.266) is a
DIFFERENT model generation (Omega1.2's Regime3-router 3-head TabM, quality threshold 0.8, a
different 20,071-candidate pool with TP/SL rates 31.35%/55.78%) tested 2026-06-18, and its own exit
mechanism fully REPLACED exit decisions rather than layering an early-exit on top of hard ATR
barriers the way the current live Omega4.6.1 evaluate_exit does. Cited as rough historical context
only, exactly like the entry script treated the same report's candidate-pool shape -- NOT
reproduced or reconciled here.

Split discipline: fit + calibrate everything on TRAIN only (chronological, embargoed internal
holdout inside TRAIN for probability-threshold calibration). VALIDATION is scored exactly once, at
the end, for both v0 and v1 (two pre-specified, non-iterative variants) -- never touched during
fitting/tuning. OOS and HOLDOUT are never loaded (inherited from the reused candidate CSVs' own
TRAIN/VALIDATION split).

Outputs: tmp/causal_regen_20260516/eth_rl_exit_gate_oracle_smoketest_20260901/
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import research_eth_rl_entry_gate_oracle_smoketest_20260831 as entry_smoke  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as _omega  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------------------------
ENTRY_OUT_DIR = entry_smoke.OUT_DIR
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_rl_exit_gate_oracle_smoketest_20260901"

ROUNDTRIP_COST = entry_smoke.ROUNDTRIP_COST  # 0.0010, unchanged by exit timing (paid once per trade)
EXISTING_GATE_THRESHOLD = entry_smoke.EXISTING_GATE_THRESHOLD  # 0.50
RANDOM_SEED = 20260901

# h48qual BASE_TEMPLATE fallback for pos_notional/pos_leverage (see module docstring).
_BASE_TEMPLATE = _omega.BASE_TEMPLATE

DECILE_FRACTIONS = [k / 10.0 for k in range(1, 10)]  # 9 interior checkpoints per candidate

INTERNAL_HOLDOUT_FRAC = 0.20
EMBARGO_BARS = 288  # matches entry script's MAX_HOLD_BARS / label horizon

# Tier D (position context) -- computed per checkpoint, see module docstring for conventions.
POS_FEATURE_COLS = [
    "pos_side", "pos_hold_bars", "pos_unrealized", "pos_mfe", "pos_mae", "pos_giveback",
    "pos_dist_to_tp", "pos_dist_to_sl", "pos_notional", "pos_leverage", "pos_exposure",
    "pos_tp", "pos_sl",
]

# Tier A/B (reused verbatim from the entry script for consistency).
REGIME_FEATURES = entry_smoke.REGIME_FEATURES
MARKET_CONTEXT_RAW_COLS = entry_smoke.MARKET_CONTEXT_RAW_COLS
MARKET_CONTEXT_COMPUTED_COLS = entry_smoke.MARKET_CONTEXT_COMPUTED_COLS
V1_SIGNAL_NAMES = entry_smoke.V1_SIGNAL_NAMES
V1_SIGNAL_COLS = entry_smoke.V1_SIGNAL_COLS

V0_FEATURE_COLS = POS_FEATURE_COLS + REGIME_FEATURES + MARKET_CONTEXT_RAW_COLS + MARKET_CONTEXT_COMPUTED_COLS
V1_FEATURE_COLS = V0_FEATURE_COLS + V1_SIGNAL_COLS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------
# Step 1: reload entry script's saved candidate tables + klines
# --------------------------------------------------------------------------------------------
def load_saved_candidates(split: str) -> pd.DataFrame:
    path = ENTRY_OUT_DIR / f"candidates_{split}_labeled.csv"
    if not path.exists():
        raise RuntimeError(
            f"missing {path} -- run scripts/research_eth_rl_entry_gate_oracle_smoketest_20260831.py first"
        )
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def build_kline_lookup(eth_klines: pd.DataFrame) -> pd.Series:
    return pd.Series(np.arange(len(eth_klines)), index=eth_klines["timestamp"].to_numpy())


# --------------------------------------------------------------------------------------------
# Step 2: per-candidate checkpoint construction (Tier D pos_* features + oracle label)
# --------------------------------------------------------------------------------------------
def build_checkpoints(cand: pd.DataFrame, eth_klines: pd.DataFrame, pos_lookup: pd.Series) -> pd.DataFrame:
    """One row per (candidate, sampled interior checkpoint). See module docstring for the exact
    entry-price / unrealized-move / mfe/mae / oracle-label conventions."""
    eth_ts = eth_klines["timestamp"].to_numpy()
    eth_open = eth_klines["open"].to_numpy(dtype=np.float64)
    eth_close = eth_klines["close"].to_numpy(dtype=np.float64)
    eth_atr_pct_192 = eth_klines["atr_pct_192"].to_numpy(dtype=np.float64)
    total_bars = len(eth_close)

    base_notional = float(_BASE_TEMPLATE["notional"]) if _BASE_TEMPLATE else 0.45
    base_leverage = float(_BASE_TEMPLATE["leverage"]) if _BASE_TEMPLATE else 2.0
    base_exposure = base_notional * base_leverage

    kline_pos = cand["timestamp"].map(pos_lookup)
    n_unmatched = int(kline_pos.isna().sum())
    cand = cand.assign(kline_pos=kline_pos).dropna(subset=["kline_pos"]).reset_index(drop=True)
    cand["kline_pos"] = cand["kline_pos"].astype(np.int64)

    rows: list[dict] = []
    t0 = time.time()
    n = len(cand)
    for i in range(n):
        pos = int(cand.at[i, "kline_pos"])
        entry_i = pos + 1
        if entry_i >= total_bars:
            continue
        side = int(cand.at[i, "sim_side"])
        bars_held = int(cand.at[i, "bars_held"])
        tp_move = float(cand.at[i, "tp_move"])
        sl_move = float(cand.at[i, "sl_move"])
        price_move_terminal = float(cand.at[i, "price_move_raw"])
        entry_price = eth_open[entry_i]
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            continue

        checkpoints = sorted({int(round(bars_held * f)) for f in DECILE_FRACTIONS})
        checkpoints = [t for t in checkpoints if 1 <= t < bars_held and entry_i + t < total_bars]
        if not checkpoints:
            continue

        max_t = checkpoints[-1]
        close_slice = eth_close[entry_i:entry_i + max_t + 1]  # index 0 == entry_i itself (bar after signal)
        if side > 0:
            moves = close_slice / entry_price - 1.0
        else:
            moves = 1.0 - close_slice / entry_price
        # mfe/mae are cumulative running max/min, seeded at 0.0 at the entry bar itself
        # (trading_bot.py:8991-8992,9179-9180 -- matches live bookkeeping exactly).
        running_max = np.maximum.accumulate(np.concatenate([[0.0], moves]))
        running_min = np.minimum.accumulate(np.concatenate([[0.0], moves]))

        cand_meta = cand.iloc[i]
        for t in checkpoints:
            unrealized = float(moves[t - 1])  # moves[0] is offset 0 = entry_i itself = hold_bars 1
            mfe = float(running_max[t])
            mae = float(running_min[t])
            giveback = (mfe - unrealized) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
            row = {
                "timestamp": eth_ts[entry_i + t],
                "cand_timestamp": cand_meta["timestamp"],
                "reason": cand_meta["reason"],
                "quality_for_action": cand_meta["quality_for_action"],
                "bars_held_terminal": bars_held,
                "price_move_terminal": price_move_terminal,
                "checkpoint_t": t,
                "pos_side": float(side),
                "pos_hold_bars": float(t),
                "pos_unrealized": unrealized,
                "pos_mfe": mfe,
                "pos_mae": mae,
                "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
                "pos_dist_to_tp": float(tp_move - unrealized),
                "pos_dist_to_sl": float(unrealized + abs(sl_move)),
                "pos_notional": base_notional,
                "pos_leverage": base_leverage,
                "pos_exposure": base_exposure,
                "pos_tp": tp_move,
                "pos_sl": sl_move,
                "atr_pct_192": float(eth_atr_pct_192[entry_i + t]),
                "oracle_exit_label": int(unrealized > price_move_terminal),
            }
            rows.append(row)
        if (i + 1) % 10000 == 0:
            log(f"  checkpoints built for {i + 1}/{n} candidates ({time.time() - t0:.1f}s elapsed, {len(rows)} rows so far)")

    out = pd.DataFrame(rows)
    return out, {"n_candidates_in": n, "n_unmatched_timestamp": n_unmatched, "n_checkpoint_rows": len(out)}


# --------------------------------------------------------------------------------------------
# Step 3: join Tier A/B context at the CHECKPOINT timestamp
# --------------------------------------------------------------------------------------------
def join_context(ckpt: pd.DataFrame, market_context: pd.DataFrame, regime_df: pd.DataFrame,
                  sig_df: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    n0 = len(ckpt)
    merged = ckpt.merge(market_context, on="timestamp", how="inner")
    n1 = len(merged)
    merged = merged.merge(regime_df, on="timestamp", how="inner")
    n2 = len(merged)
    report = {"n_checkpoints": n0, "n_after_market_context_join": n1, "n_after_regime_join": n2}
    if sig_df is not None:
        merged = merged.merge(sig_df, on="timestamp", how="inner")
        report["n_after_signal_join"] = len(merged)
    return merged, report


# --------------------------------------------------------------------------------------------
# Step 4: model fit / calibrate (reuses entry script's fit_hgb / split / prep_x for consistency)
# --------------------------------------------------------------------------------------------
def run_variant(name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    log(f"=== variant {name}: feature_cols={len(feature_cols)}, train_rows={len(train_df)}, val_rows={len(val_df)} ===")
    fit_df, holdout_df, split_info = entry_smoke.internal_train_holdout_split(
        train_df, INTERNAL_HOLDOUT_FRAC, EMBARGO_BARS
    )
    log(f"  internal split: fit={split_info['n_fit']} holdout={split_info['n_holdout']} embargoed={split_info['n_embargoed_dropped']}")

    x_fit = entry_smoke.prep_x(fit_df, feature_cols)
    y_fit = fit_df["oracle_exit_label"].to_numpy()
    model_a = entry_smoke.fit_hgb(x_fit, y_fit)

    x_hold = entry_smoke.prep_x(holdout_df, feature_cols)
    hold_probs = model_a.predict_proba(x_hold)[:, 1]
    holdout_base_rate = float(holdout_df["oracle_exit_label"].mean())
    threshold = entry_smoke.calibrate_threshold_matching_gate_rate(hold_probs, holdout_base_rate)
    log(f"  calibrated threshold={threshold:.4f} (target trigger rate {holdout_base_rate:.4f} = oracle label's own TRAIN-holdout base rate)")

    try:
        pi = permutation_importance(model_a, x_hold, holdout_df["oracle_exit_label"].to_numpy(),
                                     n_repeats=5, random_state=RANDOM_SEED, scoring="roc_auc")
        importance_pairs = sorted(zip(feature_cols, pi.importances_mean.tolist()), key=lambda t: -t[1])
        top_importance = [{"feature": f, "importance_mean_auc_drop": v} for f, v in importance_pairs[:15]]
    except Exception as e:  # pragma: no cover - diagnostic only
        top_importance = None
        log(f"  permutation_importance failed: {e}")

    x_full = entry_smoke.prep_x(train_df, feature_cols)
    y_full = train_df["oracle_exit_label"].to_numpy()
    model_final = entry_smoke.fit_hgb(x_full, y_full)

    x_val = entry_smoke.prep_x(val_df, feature_cols)
    val_probs = model_final.predict_proba(x_val)[:, 1]

    try:
        val_auc = float(roc_auc_score(val_df["oracle_exit_label"].to_numpy(), val_probs))
    except ValueError as e:
        val_auc = None
        log(f"  VALIDATION AUC error: {e}")

    val_scored = val_df[["cand_timestamp", "checkpoint_t", "bars_held_terminal", "reason",
                          "quality_for_action", "price_move_terminal", "pos_unrealized",
                          "oracle_exit_label"]].copy()
    val_scored[f"{name}_prob"] = val_probs
    val_scored[f"{name}_trigger"] = val_probs >= threshold

    return {
        "variant": name,
        "feature_cols": feature_cols,
        "internal_split": split_info,
        "threshold_calibrated_on_train_holdout": threshold,
        "train_holdout_oracle_label_base_rate": holdout_base_rate,
        "train_full_oracle_label_base_rate": float(train_df["oracle_exit_label"].mean()),
        "validation_oracle_label_base_rate": float(val_df["oracle_exit_label"].mean()),
        "validation_checkpoint_auc": val_auc,
        "top_permutation_importance_on_train_holdout": top_importance,
        "val_scored": val_scored,
    }


# --------------------------------------------------------------------------------------------
# Step 5: candidate-level POLICY evaluation (sequential first-trigger, VALIDATION only)
# --------------------------------------------------------------------------------------------
def evaluate_policy(val_cand: pd.DataFrame, val_scored: pd.DataFrame, trigger_col: str,
                     label_gate: np.ndarray | None) -> dict:
    """For each candidate, scan its checkpoints in increasing hold-bar order; at the first
    trigger==True checkpoint, realize pos_unrealized there; otherwise fall through to the
    candidate's own already-realized price_move_terminal (== the no-early-exit baseline)."""
    scored = val_scored.sort_values(["cand_timestamp", "checkpoint_t"])
    first_trigger = (
        scored[scored[trigger_col]]
        .groupby("cand_timestamp", as_index=False)
        .first()[["cand_timestamp", "pos_unrealized"]]
        .rename(columns={"pos_unrealized": "early_exit_move"})
    )
    merged = val_cand.merge(first_trigger, left_on="timestamp", right_on="cand_timestamp", how="left")
    realized = merged["early_exit_move"].where(merged["early_exit_move"].notna(), merged["price_move_raw"])
    triggered = merged["early_exit_move"].notna()

    def _stats(mask: np.ndarray) -> dict:
        r = realized[mask]
        if len(r) == 0:
            return {"n": 0}
        net = r.to_numpy() - ROUNDTRIP_COST
        return {
            "n": int(len(r)),
            "win_rate": float((r > 0).mean()),
            "avg_net_bp": float(net.mean() * 10000.0),
            "median_net_bp": float(np.median(net) * 10000.0),
            "sum_net_bp": float(net.sum() * 10000.0),
        }

    out = {
        "n_candidates": int(len(merged)),
        "n_early_exit_triggered": int(triggered.sum()),
        "trigger_rate": float(triggered.mean()) if len(merged) else None,
        "full_pool": _stats(np.ones(len(merged), dtype=bool)),
        "gate_approved_subset": _stats(label_gate) if label_gate is not None else None,
    }
    return out


def evaluate_no_exit_baseline(val_cand: pd.DataFrame, label_gate: np.ndarray | None) -> dict:
    def _stats(mask: np.ndarray) -> dict:
        r = val_cand.loc[mask, "price_move_raw"]
        if len(r) == 0:
            return {"n": 0}
        net = r.to_numpy() - ROUNDTRIP_COST
        return {
            "n": int(len(r)),
            "win_rate": float((r > 0).mean()),
            "avg_net_bp": float(net.mean() * 10000.0),
            "median_net_bp": float(np.median(net) * 10000.0),
            "sum_net_bp": float(net.sum() * 10000.0),
        }
    return {
        "full_pool": _stats(np.ones(len(val_cand), dtype=bool)),
        "gate_approved_subset": _stats(label_gate) if label_gate is not None else None,
    }


def evaluate_oracle_ceiling(val_cand: pd.DataFrame, val_ckpt: pd.DataFrame, label_gate: np.ndarray | None) -> dict:
    """Best achievable outcome per candidate: max(ride-to-end, best sampled checkpoint). Bounded by
    the 9-decile checkpoint grid, not every bar -- a slightly conservative (lower) ceiling than a
    true continuous-time oracle, documented as such."""
    best_ckpt = val_ckpt.groupby("cand_timestamp", as_index=False)["pos_unrealized"].max().rename(
        columns={"pos_unrealized": "best_checkpoint_move"}
    )
    merged = val_cand.merge(best_ckpt, left_on="timestamp", right_on="cand_timestamp", how="left")
    merged["best_checkpoint_move"] = merged["best_checkpoint_move"].fillna(-np.inf)
    best = np.maximum(merged["price_move_raw"].to_numpy(), merged["best_checkpoint_move"].to_numpy())

    def _stats(mask: np.ndarray) -> dict:
        r = best[mask]
        if len(r) == 0:
            return {"n": 0}
        net = r - ROUNDTRIP_COST
        return {
            "n": int(len(r)),
            "win_rate": float((r > 0).mean()),
            "avg_net_bp": float(net.mean() * 10000.0),
            "median_net_bp": float(np.median(net) * 10000.0),
            "sum_net_bp": float(net.sum() * 10000.0),
        }
    return {
        "full_pool": _stats(np.ones(len(merged), dtype=bool)),
        "gate_approved_subset": _stats(label_gate) if label_gate is not None else None,
    }


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "script": "scripts/research_eth_rl_exit_gate_oracle_smoketest_20260901.py",
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "design_doc": "docs/eth_rl_autotrading_agent_design_20260831.md (Section 5, Section 8 '0단계' exit sub-task)",
        "companion_script": "scripts/research_eth_rl_entry_gate_oracle_smoketest_20260831.py",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "risk_sizing_source": "BASE_TEMPLATE constant (notional=0.45, leverage=2.0), NOT the real "
                               "per-candidate risk sidecar -- see module docstring.",
        "existing_exit_head_baseline_replayed": False,
        "existing_exit_head_historical_reference_only": {
            "source": "docs/experiments/omega1_2_quality_gate_rl_problem_report_20260618.md section 6, "
                       "'Exit-head-only selected'",
            "validation_pnl": 0.46, "oos_pnl": -6.05, "oos_wr": 0.266,
            "caveat": "different model generation (Omega1.2), different candidate pool (20071, TP 31.35%/"
                      "SL 55.78%), different exit mechanism (full replacement, not early-exit-on-top-of-"
                      "hard-barriers) -- NOT reproduced or reconciled here, rough context only.",
        },
        "assumptions": [],
    }

    log("Step 1: reload entry-script candidate tables + klines")
    train_cand = load_saved_candidates("train")
    val_cand = load_saved_candidates("validation")
    report["reused_candidate_pool"] = {
        "train_n": int(len(train_cand)), "validation_n": int(len(val_cand)),
        "source_dir": str(ENTRY_OUT_DIR.relative_to(ROOT)),
    }
    log(f"  TRAIN candidates reused: {len(train_cand)}; VALIDATION candidates reused: {len(val_cand)}")

    eth_klines = entry_smoke.load_klines(entry_smoke.ETH_KLINES_PATH)
    btc_klines = entry_smoke.load_klines(entry_smoke.BTC_KLINES_PATH)
    eth_klines["atr_pct_192"] = entry_smoke._atr_pct(eth_klines, window=entry_smoke.ATR_WINDOW)
    pos_lookup = build_kline_lookup(eth_klines)

    log("Step 2: build per-candidate interior checkpoints (Tier D pos_* + oracle label)")
    train_ckpt_raw, train_ckpt_report = build_checkpoints(train_cand, eth_klines, pos_lookup)
    log(f"  TRAIN checkpoints: {train_ckpt_report}")
    val_ckpt_raw, val_ckpt_report = build_checkpoints(val_cand, eth_klines, pos_lookup)
    log(f"  VALIDATION checkpoints: {val_ckpt_report}")
    report["checkpoint_construction"] = {"train": train_ckpt_report, "validation": val_ckpt_report}
    report["oracle_exit_label_base_rate"] = {
        "train": float(train_ckpt_raw["oracle_exit_label"].mean()) if len(train_ckpt_raw) else None,
        "validation": float(val_ckpt_raw["oracle_exit_label"].mean()) if len(val_ckpt_raw) else None,
    }
    report["oracle_exit_label_base_rate_by_terminal_reason"] = {
        split: df.groupby("reason")["oracle_exit_label"].mean().to_dict()
        for split, df in (("train", train_ckpt_raw), ("validation", val_ckpt_raw))
    }
    log(f"  oracle_exit_label base rate: {report['oracle_exit_label_base_rate']}")
    log(f"  oracle_exit_label base rate by terminal reason: {report['oracle_exit_label_base_rate_by_terminal_reason']}")

    log("Step 3: training_features (regime scoring + market context + evidence signals)")
    training_features_full = entry_smoke.load_training_features_full()
    regime_df, regime_diag = entry_smoke.score_regime_gbm3(training_features_full)
    report["regime_gbm3_scoring_diagnostics"] = regime_diag
    market_context = entry_smoke.build_market_context(training_features_full)
    sig_df = entry_smoke.build_signal_features(eth_klines, btc_klines)

    log("Step 4: join Tier A/B context at checkpoint timestamps")
    train_v0, train_join_v0 = join_context(train_ckpt_raw, market_context, regime_df, None)
    val_v0, val_join_v0 = join_context(val_ckpt_raw, market_context, regime_df, None)
    train_v1, train_join_v1 = join_context(train_ckpt_raw, market_context, regime_df, sig_df)
    val_v1, val_join_v1 = join_context(val_ckpt_raw, market_context, regime_df, sig_df)
    report["feature_join"] = {
        "train_v0": train_join_v0, "validation_v0": val_join_v0,
        "train_v1": train_join_v1, "validation_v1": val_join_v1,
    }
    log(f"  TRAIN v0 modeling pool: {len(train_v0)}; VALIDATION v0 modeling pool: {len(val_v0)}")
    log(f"  TRAIN v1 modeling pool: {len(train_v1)}; VALIDATION v1 modeling pool: {len(val_v1)}")

    log("Step 5: sanity gate before modeling")
    sane = len(train_v0) > 5000 and len(val_v0) > 500
    report["sanity_gate_passed"] = bool(sane)
    if not sane:
        report["assumptions"].append("sanity gate FAILED -- see checkpoint_construction/feature_join for diagnosis")
        log("SANITY GATE FAILED -- writing partial report and stopping before modeling")
        (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
        return

    log("Step 6: v0 model (no evidence-signal features)")
    v0 = run_variant("v0", train_v0, val_v0, V0_FEATURE_COLS)
    v0["val_scored"].to_csv(OUT_DIR / "validation_v0_scored.csv", index=False)
    v0_summary = {k: v for k, v in v0.items() if k != "val_scored"}
    report["v0"] = v0_summary

    log("Step 7: v1 model (adds 6 evidence-signal rule triggers)")
    v1 = run_variant("v1", train_v1, val_v1, V1_FEATURE_COLS)
    v1["val_scored"].to_csv(OUT_DIR / "validation_v1_scored.csv", index=False)
    v1_summary = {k: v for k, v in v1.items() if k != "val_scored"}
    report["v1"] = v1_summary

    log("Step 8: candidate-level policy evaluation (VALIDATION, sequential first-trigger)")
    gate_mask = (val_cand["quality_for_action"] >= EXISTING_GATE_THRESHOLD).to_numpy()
    report["gate_approved_subset_n"] = int(gate_mask.sum())

    no_exit = evaluate_no_exit_baseline(val_cand, gate_mask)
    oracle_ceiling = evaluate_oracle_ceiling(val_cand, val_ckpt_raw, gate_mask)
    policy_v0 = evaluate_policy(val_cand, v0["val_scored"], "v0_trigger", gate_mask)
    policy_v1 = evaluate_policy(val_cand, v1["val_scored"], "v1_trigger", gate_mask)

    report["validation_policy_comparison"] = {
        "no_early_exit_baseline_ride_to_terminal": no_exit,
        "oracle_ceiling_reference_hindsight_cheating_bounded_by_9_decile_grid": oracle_ceiling,
        "classifier_v0_policy": policy_v0,
        "classifier_v1_policy": policy_v1,
    }
    log(f"  no-exit baseline (full pool): {no_exit['full_pool']}")
    log(f"  oracle ceiling (full pool):   {oracle_ceiling['full_pool']}")
    log(f"  v0 policy (full pool):        {policy_v0['full_pool']}")
    log(f"  v1 policy (full pool):        {policy_v1['full_pool']}")

    report["assumptions"] = [
        "Reused the entry smoke test's saved candidate pool verbatim (same full raw dir_action!=0 "
        "pool, same TRAIN=2025-01-01..2025-09-30 / VALIDATION=2025-10-01..2025-12-31 split as that "
        "bundle's own OOF export -- NOT the CLAUDE.md standard Fresh-Forward split). Metrics are "
        "additionally broken out on the quality_for_action>=0.50 (existing-gate-approved) subset.",
        "Checkpoint sampling: up to 9 interior decile checkpoints per candidate (t=round(bars_held*k/10) "
        "for k=1..9, deduped, restricted to 1<=t<bars_held) instead of every held bar -- bar-by-bar over "
        "the full reused pool would be >15M rows on TRAIN alone, far outside a stage-0 budget. Candidates "
        "with bars_held<=1 contribute zero checkpoint rows by construction (no early-exit opportunity).",
        "Oracle label is closed-form: oracle_exit_label_t = 1 iff unrealized_move_t (closing now) > "
        "price_move_raw (the candidate's own already-realized terminal outcome) -- reuses the terminal "
        "outcome the entry script already simulated rather than defining a new 'local optimum' search.",
        "unrealized_move / mfe / mae use core.causal_futures_backtest._resolve_trade's entry-price "
        "convention (ETH open at kline_pos+1) and trading_bot.py's live mfe/mae update rule (running "
        "max/min seeded at 0.0 at the entry bar; trading_bot.py:9179-9180) -- raw, unscaled price-move "
        "fractions per CLAUDE.md's Position-Feature Train/Inference Parity Contract, NOT multiplied by "
        "notional/leverage.",
        "pos_notional/pos_leverage/pos_exposure use the h48qual component's fixed BASE_TEMPLATE "
        "(notional=0.45, leverage=2.0) WITHOUT the regime-expert EXPERT_SCALES multiplier, held constant "
        "across a candidate's whole hold -- NOT the real per-candidate risk sidecar output (see module "
        "docstring for why; risk_sizing_source flagged at top level of this report per CLAUDE.md).",
        "pos_tp/pos_sl reuse tp_move/sl_move already computed by the entry script from the live "
        "ATR-adaptive formula (atr_pct_192-driven, side-independent) -- these do not depend on the "
        "model/sidecar simplification above and are exact.",
        "The existing-live-exit_head baseline was NOT replayed (no live PyTorch bundle inference in this "
        "script) -- see existing_exit_head_baseline_replayed=false and "
        "existing_exit_head_historical_reference_only above. The 06-18 Omega1.2 number is cited as rough "
        "context only, exactly as the entry script treated its own historical reference.",
        "Threshold calibration matches the classifier's VALIDATION-checkpoint... TRAIN-holdout trigger "
        "rate to the oracle label's OWN empirical base rate on that same TRAIN-holdout slice (the "
        "natural analog of the entry script's 'match the existing gate's accept rate' calibration, since "
        "there is no external gate for exit).",
        "Policy evaluation is candidate-level and sequential: for each VALIDATION candidate, checkpoints "
        "are scanned in increasing hold-bar order and the FIRST trigger realizes pos_unrealized there; "
        "candidates with no trigger fall through to their own already-realized price_move_raw (identical "
        "to the no-early-exit baseline for those candidates) -- this is the decision-relevant PnL/WR "
        "metric, not just checkpoint-level AUC.",
        "v0 and v1 are two fixed, pre-specified pipelines each run and VALIDATION-scored exactly once in "
        "this single script pass; VALIDATION was not used for any fitting/threshold-tuning decision.",
        "HistGradientBoostingClassifier (entry script's fit_hgb, same fixed hyperparameters, "
        "balanced class weights) -- no HP search; NaNs passed through natively, not imputed.",
    ]

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"Done. report.json + CSVs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
