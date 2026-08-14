#!/usr/bin/env python3
"""RESEARCH ONLY -- follow-up to `research_eth_omega461_exit_head_h48cons_relabel_20260813.py`
(see `docs/experiments/eth_omega461_live_exit_head_h48cons_relabel_20260813.md`, verdict: FAILED).

That attempt kept the "every zigzag_action bar is its own entry candidate, simulate forward to a
barrier resolution bar, label near that bar as exit=1" structure (37,158 candidates, a clean 50x
density win over the original 732-813 zigzag-segment recipe), but sourced the barrier width from
the Odyssey sub-project's h48_conservative triple-barrier CSV (ATR mult 1.2/0.8, floor 0.6%/0.4%).
That barrier is ~10-12x TIGHTER than the barrier the live h48qual/zig075 components actually run
under (ATR mult 12/6, floor 7.5%/4.0%), so its median resolution time (9-10 bars) was ~67-72x
shorter than the frozen-baseline's actual average hold (670/726 bars). The retrained exit_head
learned "exit soon after entry" and over-fired catastrophically (VAL PnL +5.45%/+40.31% ->
-3.90%/-9.94%).

This script keeps the exact same dense-candidate structure but replaces the barrier SOURCE: instead
of reading a precomputed CSV, it simulates the barrier forward from each candidate using the LIVE
component's own ATR-adaptive formula, imported unchanged from
`eval_omega4_1_atr_safety_sltp_20260622` (`_atr_pct` / `_apply_atr_safety_sltp` -- not reimplemented)
with the exact live defaults (`trading_bot_modules/omega4_6_1_live.py` `_ComponentConfig`:
atr_window=192, tp_mult=12.0, sl_mult=6.0, min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12).
`terminal_window`/`adverse_unreal`/`min_mfe_for_giveback`/`giveback_min` and the position-feature
construction (`exit_head._position_feature_row`) are unchanged from both prior recipes.

Because the live-ATR barrier resolves over hundreds of bars (not tens), the full 37K-candidate
population would need on the order of 30M dataset rows -- infeasible to feature-construct on a
CPU-only dev box in reasonable time. This script therefore draws a seeded random subsample of
candidates (`--max-candidates`, default 2000) from the same zigzag_action-bar population, which
still gives ~2.7x the original recipe's 732-813 independent segments while keeping the realistic
per-candidate duration. This deviation (subsampling instead of using every action bar) is called
out explicitly in the checkpoint output and the companion doc, not silently introduced.

Per the coordinator's explicit gate: a fast, feature-row-free timescale checkpoint runs BEFORE any
dataset build / retrain, comparing the new barrier's resolution-bar distribution against the live
baseline's actual avg_hold_bars (670.3 h48qual / 725.6 zig075, from the h48cons doc's VAL baseline
row). Only if the median is the same order of magnitude (tens-to-hundreds of bars, not single/low
double digits) does the script proceed past `--stage checkpoint_only`.

fresh_forward_bar_by_bar=true (VAL replay is a single causal forward pass via the existing,
already-certified `research_eth_omega461_exit_sweep_20260721.replay_exit_variant`, reused via the
h48cons script's `_evaluate_val`, unchanged). trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false. Training uses only the
pre-2025-10-01 TRAIN split. VAL only -- this script never loads or scores OOS data.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Does NOT overwrite any live checkpoint or the h48cons script/bundles.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as pricemove_retrain  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import research_eth_omega461_exit_head_h48cons_relabel_20260813 as h48cons  # noqa: E402

MODEL_ID = "eth_omega461_exit_head_liveatr_relabel_20260813"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DIRECTION_LABEL_DIR = h48cons.DIRECTION_LABEL_DIR

# Exact live defaults, verbatim from trading_bot_modules/omega4_6_1_live.py:91-97 (_ComponentConfig
# dataclass field defaults) -- read for the values only, the live module itself is never imported.
# Matches research_eth_omega461_exit_sweep_20260721.COMPONENTS (both h48qual q050 and zig075 q075
# use the same atr_window/tp_mult/sl_mult/min_tp/min_sl/max_tp/max_sl there too).
LIVE_ATR_CFG = {
    "atr_window": 192,
    "tp_mult": 12.0,
    "sl_mult": 6.0,
    "min_tp": 0.075,
    "min_sl": 0.040,
    "max_tp": 0.22,
    "max_sl": 0.12,
}

# baseline VAL avg_hold_bars from docs/experiments/eth_omega461_live_exit_head_h48cons_relabel_20260813.md
# (frozen SLTP + original exit_head, exit_head never fires -> pure ATR-safety-SLTP hold time).
BASELINE_AVG_HOLD_BARS = {"h48qual": 670.3103448275862, "zig075": 725.551724137931}
TIMESCALE_GATE_MIN_MEDIAN_BARS = 30  # below this, "still too short" per the coordinator's gate

# Memory safety valve. The first server attempt (--max-candidates 1500, unbounded row
# accumulation) coincided with an ops_watchdog "disk and memory headroom CRITICAL" alert and the
# shared server (also running the live trading bot + BTC/JM shadow loops) going down for over an
# hour. This machine's dataset-build loop now checks system-wide available memory and its own RSS
# periodically and stops accumulating EARLY (same graceful path as hitting --max-rows) rather than
# risk an OOM on a shared box. Deliberately conservative -- being a "good neighbor" on a live
# trading server matters more than finishing this one research dataset in a single pass.
MIN_AVAILABLE_MEMORY_GB = 3.0
MAX_PROCESS_RSS_GB = 6.0
MEMORY_CHECK_EVERY_CANDIDATES = 20


def _available_memory_gb() -> float:
    """Linux/WSL only (both dev and the server are Linux-family). Returns +inf if unreadable so a
    missing /proc never blocks the run -- the RSS cap and chunked flushing are the real limiters."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return float("inf")


def _process_rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return 0.0


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _fast_timescale_checkpoint(
    frame: pd.DataFrame,
    *,
    atr_cfg: dict[str, float],
    max_horizon_bars: int,
) -> dict[str, Any]:
    """Feature-row-free simulation: for EVERY zigzag_action in (1,2) bar, walk forward with the
    live ATR-adaptive barrier (entry price = next-bar open, no fee/slip -- a cheap proxy, the full
    builder below uses the proper `omega._try_execution` fill) and record bars-to-resolution +
    reason. Used only to answer the pre-training timescale question; not the training dataset."""
    required = {"timestamp", "zigzag_action", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"timescale checkpoint missing columns: {missing}")
    open_ = pd.to_numeric(frame["open"], errors="raise").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    action = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    atr_pct = atr_eval._atr_pct(frame, int(atr_cfg["atr_window"]))

    n = len(frame)
    last_i = n - 2
    tp_mult, sl_mult = float(atr_cfg["tp_mult"]), float(atr_cfg["sl_mult"])
    min_tp, min_sl = float(atr_cfg["min_tp"]), float(atr_cfg["min_sl"])
    max_tp, max_sl = float(atr_cfg["max_tp"]), float(atr_cfg["max_sl"])

    bars_long: list[int] = []
    bars_short: list[int] = []
    valid_candidate_idx: list[int] = []
    reason_counts: dict[str, int] = {}
    truncated = 0
    used = 0

    for i in range(0, max(last_i, 0)):
        side_action = int(action[i])
        if side_action not in (1, 2):
            continue
        side = 1 if side_action == 1 else -1
        entry_i = min(i + 1, n - 1)
        entry_price = float(open_[entry_i])
        if entry_price <= 0.0:
            continue
        a = float(atr_pct[i])
        tp_move = min(max(min_tp, a * tp_mult), max_tp)
        sl_move = min(max(min_sl, a * sl_mult), max_sl)
        if side > 0:
            tp_level = entry_price * (1.0 + tp_move)
            sl_level = entry_price * (1.0 - sl_move)
        else:
            tp_level = entry_price * (1.0 - tp_move)
            sl_level = entry_price * (1.0 + sl_move)
        end_bound = min(entry_i + int(max_horizon_bars), n - 1)
        reason = "timeout"
        bars = end_bound - entry_i + 1
        for row_i in range(entry_i, end_bound + 1):
            hi = high[row_i]
            lo = low[row_i]
            hit_sl = (lo <= sl_level) if side > 0 else (hi >= sl_level)
            hit_tp = (hi >= tp_level) if side > 0 else (lo <= tp_level)
            if hit_sl:
                reason = "sl"
                bars = row_i - entry_i + 1
                break
            if hit_tp:
                reason = "tp"
                bars = row_i - entry_i + 1
                break
        used += 1
        valid_candidate_idx.append(int(i))
        if reason == "timeout":
            truncated += 1
        (bars_long if side > 0 else bars_short).append(bars)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    def _stats(vals: list[int]) -> dict[str, float]:
        if not vals:
            return {}
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "count": int(len(arr)), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "p10": float(np.quantile(arr, 0.10)), "p25": float(np.quantile(arr, 0.25)),
            "p75": float(np.quantile(arr, 0.75)), "p90": float(np.quantile(arr, 0.90)),
            "max": float(arr.max()),
        }

    return {
        "used_candidates": int(used),
        "truncated_at_horizon": int(truncated),
        "truncated_rate": float(truncated / used) if used else 0.0,
        "reason_counts": reason_counts,
        "long_bars_stats": _stats(bars_long),
        "short_bars_stats": _stats(bars_short),
        "max_horizon_bars": int(max_horizon_bars),
        "atr_cfg": dict(atr_cfg),
        "valid_candidate_idx": valid_candidate_idx,
    }


def _build_exit_dataset_entry_label_live_atr_barrier(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    *,
    candidate_idx: np.ndarray,
    fee: float,
    slip: float,
    cost_mult: float,
    atr_cfg: dict[str, float],
    max_horizon_bars: int,
    terminal_window: int = 3,
    adverse_unreal: float = -0.010,
    min_mfe_for_giveback: float = 0.006,
    giveback_min: float = 0.65,
    max_rows: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Same structure as h48cons's `_build_exit_dataset_entry_label_h48cons_barrier` (position
    feature construction, adverse/giveback branches, terminal-window-near-barrier-end labeling)
    but the barrier end is found by SIMULATING the live ATR-adaptive barrier forward from each
    candidate (via `atr_eval._atr_pct`) instead of reading a precomputed CSV. Only candidates in
    `candidate_idx` are used (subsample -- see module docstring for why the full ~37K population
    is infeasible at this barrier's ~600-850 bar median duration)."""
    required = {"timestamp", "zigzag_action", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"live-ATR exit dataset missing columns: {missing}")
    if len(frame) != len(state):
        raise RuntimeError("live-ATR exit frame/state length mismatch")

    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    action = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    atr_pct = atr_eval._atr_pct(frame, int(atr_cfg["atr_window"]))
    route_cols = list(hard.ROUTE_COLS)
    route_values = frame[route_cols].to_numpy(dtype=np.float64)

    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    notional = float(omega.BASE_TEMPLATE["notional"])
    leverage = float(omega.BASE_TEMPLATE["leverage"])
    take_profit = float(omega.BASE_TEMPLATE["take_profit"])
    stop_loss = float(omega.BASE_TEMPLATE["stop_loss"])
    tw = max(int(terminal_window), 1)
    tp_mult, sl_mult = float(atr_cfg["tp_mult"]), float(atr_cfg["sl_mult"])
    min_tp, min_sl = float(atr_cfg["min_tp"]), float(atr_cfg["min_sl"])
    max_tp, max_sl = float(atr_cfg["max_tp"]), float(atr_cfg["max_sl"])
    n = len(frame)

    # Repeated OOM-shaped crashes (dev x2, server x1) all happened right at the start of this
    # loop with no python traceback -- consistent with peak memory from holding ~1.5M raw
    # `_position_feature_row` dicts (each ~290 freshly-allocated string keys, since f-string keys
    # are not auto-interned) in a single python list before ever converting to a DataFrame.
    # h48cons's 540K-row run (which succeeded) never had to hold more than that many at once;
    # this recipe's ~845-bar-avg candidates push the naive approach ~3x higher. Fix: flush to a
    # DataFrame chunk every CHUNK_SIZE rows so peak raw-dict memory stays bounded regardless of
    # --max-candidates, then pd.concat the chunks at the end (numpy-backed, not python-object
    # overhead). This only changes accumulation inside this NEW function -- `_position_feature_row`
    # itself (shared module) is untouched.
    CHUNK_SIZE = 20_000
    x_chunks: list[pd.DataFrame] = []
    f_chunks: list[pd.DataFrame] = []
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    frame_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    tb_barrier_reason_counts: dict[str, int] = {}
    used_candidates = 0
    skipped_candidates = 0
    skip_reasons: dict[str, int] = {}
    positive_count = 0
    total_rows_built = 0
    stop = False
    stopped_for_memory = False
    t_start = time.time()

    def _flush_chunk() -> None:
        nonlocal rows, frame_rows
        if not rows:
            return
        x_chunks.append(pd.DataFrame(rows))
        f_chunks.append(pd.DataFrame(frame_rows))
        rows = []
        frame_rows = []

    for candidate_num, i in enumerate(candidate_idx.tolist()):
        if stop:
            break
        if candidate_num > 0 and candidate_num % MEMORY_CHECK_EVERY_CANDIDATES == 0:
            avail_gb = _available_memory_gb()
            rss_gb = _process_rss_gb()
            print(
                f"  progress candidates_processed={candidate_num}/{len(candidate_idx)} "
                f"used_candidates={used_candidates} total_rows_built={total_rows_built} "
                f"rss_gb={rss_gb:.2f} available_gb={avail_gb:.2f} elapsed={time.time() - t_start:.1f}s",
                flush=True,
            )
            if avail_gb < MIN_AVAILABLE_MEMORY_GB or rss_gb > MAX_PROCESS_RSS_GB:
                print(
                    f"  MEMORY_SAFETY_STOP available_gb={avail_gb:.2f} (floor {MIN_AVAILABLE_MEMORY_GB}) "
                    f"rss_gb={rss_gb:.2f} (cap {MAX_PROCESS_RSS_GB}) -- stopping early with "
                    f"{total_rows_built} rows from {used_candidates} candidates rather than risk OOM",
                    flush=True,
                )
                stop = True
                stopped_for_memory = True
                break
        side_action = int(action[i])
        if side_action not in (1, 2):
            skipped_candidates += 1
            skip_reasons["not_action_bar"] = skip_reasons.get("not_action_bar", 0) + 1
            continue
        side = 1 if side_action == 1 else -1
        entry_i = min(i + 1, n - 1)

        filled, entry_price, _entry_fee, _route = omega._try_execution(
            arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff,
        )
        if not filled:
            skipped_candidates += 1
            skip_reasons["entry_not_filled"] = skip_reasons.get("entry_not_filled", 0) + 1
            continue

        a = float(atr_pct[i])
        tp_move = min(max(min_tp, a * tp_mult), max_tp)
        sl_move = min(max(min_sl, a * sl_mult), max_sl)
        if side > 0:
            tp_level = entry_price * (1.0 + tp_move)
            sl_level = entry_price * (1.0 - sl_move)
        else:
            tp_level = entry_price * (1.0 - tp_move)
            sl_level = entry_price * (1.0 + sl_move)
        end_bound = min(entry_i + int(max_horizon_bars), n - 1)
        tb_reason = "timeout"
        barrier_end_i = end_bound
        for row_i in range(entry_i, end_bound + 1):
            hi = arrays["high"][row_i]
            lo = arrays["low"][row_i]
            hit_sl = (lo <= sl_level) if side > 0 else (hi >= sl_level)
            hit_tp = (hi >= tp_level) if side > 0 else (lo <= tp_level)
            if hit_sl:
                tb_reason = "sl"
                barrier_end_i = row_i
                break
            if hit_tp:
                tb_reason = "tp"
                barrier_end_i = row_i
                break
        if barrier_end_i < entry_i:
            skipped_candidates += 1
            skip_reasons["empty_window"] = skip_reasons.get("empty_window", 0) + 1
            continue

        entry_state = state.iloc[int(i)]
        mfe = 0.0
        mae = 0.0
        candidate_rows = 0
        for row_i in range(entry_i, barrier_end_i + 1):
            px = float(arrays["close"][int(row_i)])
            raw = (
                (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1.0e-12)
                if side > 0
                else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1.0e-12)
            )
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            giveback = (mfe - unreal) / max(abs(mfe), 1.0e-8) if mfe > 0.0 else 0.0
            bars_to_barrier_end = int(barrier_end_i) - int(row_i)
            terminal = bars_to_barrier_end < tw
            adverse = unreal <= float(adverse_unreal)
            gave_back = mfe >= float(min_mfe_for_giveback) and giveback >= float(giveback_min) and unreal > 0.0
            if terminal:
                label = 1
                reason = "near_barrier_resolution_exit"
            elif adverse:
                label = 1
                reason = "adverse_unreal_exit"
            elif gave_back:
                label = 1
                reason = "mfe_giveback_exit"
            else:
                label = 0
                reason = "hold"
            row = exit_head._position_feature_row(
                state, entry_state, row_i=int(row_i), side=side, entry_price=float(entry_price),
                entry_i=int(entry_i), notional=notional, leverage=leverage, take_profit=take_profit,
                stop_loss=stop_loss, mfe=mfe, mae=mae, unreal=unreal,
            )
            rows.append(row)
            labels.append(label)
            positive_count += int(label)
            candidate_rows += 1
            total_rows_built += 1
            frame_rows.append(
                {
                    "timestamp": frame["timestamp"].iloc[row_i],
                    **{route_cols[j]: float(route_values[row_i, j]) for j in range(len(route_cols))},
                    "exit_path_candidate_signal_i": int(i),
                    "exit_path_entry_i": int(entry_i),
                    "exit_path_barrier_end_i": int(barrier_end_i),
                    "exit_path_side": int(side),
                    "exit_path_hold_bars": int(max(int(row_i) - int(entry_i), 0)),
                    "exit_liveatr_label": int(label),
                    "exit_liveatr_reason": reason,
                    "exit_path_mfe": float(mfe),
                    "exit_path_mae": float(mae),
                    "exit_path_unrealized": float(unreal),
                    "exit_path_giveback": float(giveback),
                    "exit_path_bars_to_barrier_end": int(bars_to_barrier_end),
                    "exit_tb_barrier_reason": str(tb_reason),
                }
            )
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if len(rows) >= CHUNK_SIZE:
                _flush_chunk()
            if max_rows > 0 and total_rows_built >= int(max_rows):
                stop = True
                break
        if candidate_rows > 0:
            used_candidates += 1
            tb_barrier_reason_counts[str(tb_reason)] = tb_barrier_reason_counts.get(str(tb_reason), 0) + 1
        else:
            skipped_candidates += 1
            skip_reasons["empty_candidate_window"] = skip_reasons.get("empty_candidate_window", 0) + 1
        if stop:
            break

    _flush_chunk()
    if not x_chunks:
        raise RuntimeError("empty live-ATR barrier Exit Head dataset")
    x = pd.concat(x_chunks, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(labels, dtype=np.int64)
    f = pd.concat(f_chunks, ignore_index=True)
    print(
        f"  build loop done: total_rows_built={total_rows_built} used_candidates={used_candidates} "
        f"stopped_for_memory={stopped_for_memory} rss_gb={_process_rss_gb():.2f} "
        f"available_gb={_available_memory_gb():.2f} elapsed={time.time() - t_start:.1f}s",
        flush=True,
    )
    return x, y, f, {
        "stopped_for_memory": bool(stopped_for_memory),
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "positive_count": int(positive_count),
        "negative_count": int(len(y) - positive_count),
        "continued_exit_reasons": reason_counts,
        "tb_barrier_reason_counts_at_used_candidates": tb_barrier_reason_counts,
        "used_candidates": int(used_candidates),
        "skipped_candidates": int(skipped_candidates),
        "skip_reasons": skip_reasons,
        "risk_template": {"notional": notional, "leverage": leverage, "take_profit": take_profit, "stop_loss": stop_loss},
        "label_mode": "entry_label_live_atr_barrier_subsampled_action_bars",
        "atr_cfg": dict(atr_cfg),
        "max_horizon_bars": int(max_horizon_bars),
        "terminal_window": int(tw),
        "adverse_unreal": float(adverse_unreal),
        "min_mfe_for_giveback": float(min_mfe_for_giveback),
        "giveback_min": float(giveback_min),
        "candidates_sampled": int(len(candidate_idx)),
    }


def _retrain_component_exit_head_liveatr(
    component: str,
    x_exit_raw: pd.DataFrame,
    y_exit: np.ndarray,
    frame_exit: pd.DataFrame,
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    """Same freeze-encoder/direction/quality, retrain-exit-head-only pattern as
    h48cons._retrain_component_exit_head (which itself reuses
    pricemove_retrain._fit_exit_head_only unchanged) -- reimplemented locally only so the saved
    model filenames say 'liveatr' instead of 'h48cons' and don't get confused with the prior
    (failed) recipe's artifacts."""
    baseline_bundle_path = h48cons.sweep.COMPONENTS[component]["bundle"]
    bundle = torch.load(baseline_bundle_path, map_location=device, weights_only=False)
    baseline_models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)

    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        model_path = out_dir / component / "models" / f"{expert}_3head_tabm_exit_liveatr.pt"
        payload = pricemove_retrain._fit_exit_head_only(
            baseline_models[expert], x_exit, y_exit, frame_exit,
            expert_idx=idx, seed=int(seed), epochs=int(epochs), device=device, model_path=model_path,
        )
        models[expert] = payload
        summaries[expert] = {
            "model": str(model_path),
            "exit_epochs_ran": int(payload["exit_epochs_ran"]),
            "best_exit_validation_loss": float(payload["best_exit_validation_loss"]),
        }

    bundle_path = out_dir / component / "true_3head_tabm_bundle.pt"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": parent.CFG.__dict__, "model_id": MODEL_ID},
        bundle_path,
    )
    return {"baseline_bundle": str(baseline_bundle_path), "new_bundle": str(bundle_path), "summaries": summaries}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["checkpoint_only", "full"], default="full")
    ap.add_argument("--max-candidates", type=int, default=2000)
    ap.add_argument("--max-horizon-bars", type=int, default=6000)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = no extra cap beyond --max-candidates")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260813)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = torch.device("cpu")
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=prepare_frames", flush=True)
    t0 = time.time()
    frames = omega4._prepare_frames(
        disable_tp_sl=False, direction_label_dir=DIRECTION_LABEL_DIR,
        quality_mode="same_as_direction", quality_label_dir=None,
        quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()
    print(f"  train_df rows={len(frames['train_df'])} elapsed={time.time() - t0:.1f}s", flush=True)

    print("stage=timescale_checkpoint", flush=True)
    t0 = time.time()
    tc = _fast_timescale_checkpoint(frames["train_df"], atr_cfg=LIVE_ATR_CFG, max_horizon_bars=int(args.max_horizon_bars))
    checkpoint = {
        "stage": "0_live_atr_timescale_pretraining_gate",
        "predecessor_doc": "docs/experiments/eth_omega461_live_exit_head_h48cons_relabel_20260813.md",
        "predecessor_h48cons_median_bars": {"long": 10, "short": 9},
        "baseline_avg_hold_bars": BASELINE_AVG_HOLD_BARS,
        "new_recipe_bars_stats": {"long": tc["long_bars_stats"], "short": tc["short_bars_stats"]},
        "reason_counts": tc["reason_counts"],
        "used_candidates_full_population": tc["used_candidates"],
        "truncated_at_horizon": tc["truncated_at_horizon"],
        "truncated_rate": tc["truncated_rate"],
        "max_horizon_bars": tc["max_horizon_bars"],
        "atr_cfg": tc["atr_cfg"],
        "elapsed_sec": time.time() - t0,
    }
    long_median = tc["long_bars_stats"].get("median", 0.0)
    short_median = tc["short_bars_stats"].get("median", 0.0)
    gate_pass = bool(long_median >= TIMESCALE_GATE_MIN_MEDIAN_BARS and short_median >= TIMESCALE_GATE_MIN_MEDIAN_BARS)
    checkpoint["gate_pass"] = gate_pass
    checkpoint["gate_rule"] = f"median bars-to-resolution (both long and short) >= {TIMESCALE_GATE_MIN_MEDIAN_BARS}"
    (out_dir / "stage0_timescale_checkpoint.json").write_text(
        json.dumps({k: v for k, v in checkpoint.items()}, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    print(json.dumps(checkpoint, ensure_ascii=False, indent=2, default=_json_default), flush=True)

    if not gate_pass:
        print("stage=ABORT gate_pass=False -- new barrier still resolves too fast, not proceeding to training", flush=True)
        return 1
    if str(args.stage) == "checkpoint_only":
        print("stage=done (checkpoint_only)", flush=True)
        return 0

    rng = np.random.default_rng(int(args.seed))
    valid_idx = np.asarray(tc["valid_candidate_idx"], dtype=np.int64)
    n_sample = min(int(args.max_candidates), len(valid_idx)) if int(args.max_candidates) > 0 else len(valid_idx)
    candidate_idx = np.sort(rng.choice(valid_idx, size=n_sample, replace=False))
    print(f"stage=build_live_atr_barrier_exit_dataset candidates_sampled={len(candidate_idx)}/{len(valid_idx)}", flush=True)
    t0 = time.time()
    x_exit_raw, y_exit, frame_exit, exit_diag = _build_exit_dataset_entry_label_live_atr_barrier(
        frames["train_df"], frames["s_train_label"],
        candidate_idx=candidate_idx, fee=fee, slip=slip, cost_mult=float(args.cost_mult),
        atr_cfg=LIVE_ATR_CFG, max_horizon_bars=int(args.max_horizon_bars), max_rows=int(args.max_rows),
    )
    build_elapsed = time.time() - t0
    print(f"  rows={exit_diag['rows']} used_candidates={exit_diag['used_candidates']} positive_rate={exit_diag['positive_rate']:.4f} elapsed={build_elapsed:.1f}s", flush=True)
    exit_diag["build_elapsed_sec"] = build_elapsed

    results: dict[str, Any] = {"checkpoint": checkpoint, "dataset": exit_diag, "components": {}}
    for component in ("h48qual", "zig075"):
        print(f"stage=retrain_exit_head component={component}", flush=True)
        t0 = time.time()
        retrain_info = _retrain_component_exit_head_liveatr(
            component, x_exit_raw, y_exit, frame_exit,
            seed=int(args.seed), epochs=int(args.epochs), device=device, out_dir=out_dir,
        )
        print(f"  {component} retrain elapsed={time.time() - t0:.1f}s", flush=True)

        print(f"stage=evaluate_val component={component}", flush=True)
        val_metrics = h48cons._evaluate_val(component, Path(retrain_info["new_bundle"]))
        print(json.dumps({"component": component, "val": val_metrics}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
        results["components"][component] = {"retrain": retrain_info, "val_metrics": val_metrics}

    report = {
        "model_id": MODEL_ID,
        "predecessor": "eth_omega461_exit_head_h48cons_relabel_20260813 (FAILED -- see docs/experiments/eth_omega461_live_exit_head_h48cons_relabel_20260813.md)",
        "design": (
            "Same dense every-zigzag_action-bar candidate structure as the h48cons predecessor, "
            "but the barrier is now the LIVE ATR-adaptive SLTP (atr_eval._atr_pct / same formula "
            "as trading_bot_modules/omega4_6_1_live.py's _ComponentConfig defaults) simulated "
            "forward per candidate, instead of the h48_conservative CSV. Candidates are a seeded "
            "random subsample (not the full ~37K population) because the live-ATR barrier's "
            "~600-850 bar median duration makes the full population's ~30M implied rows infeasible."
        ),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "exit_threshold_held_fixed_at": h48cons.sweep.BASELINE_EXIT_THRESHOLD,
        "val_window": [h48cons.sweep.VAL_START, h48cons.sweep.VAL_END],
        "oos_opened": False,
        **results,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"report={out_dir / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
