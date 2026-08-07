#!/usr/bin/env python3
"""RESEARCH ONLY -- ETH live Omega4.6.1 exit logic, round 15: VWMA(N) adverse-touch + causal
rolling VPVR (HVN/LVN) filter, PARTIAL stop-loss only (reduce side, no pyramid/add side -- an
older model line, eval_hf_v13_v49_profit_state_pyramid_v56.py, already found pyramiding harmful
via its own grid search, so the favorable-touch add-on side is explicitly out of scope here).

This is a genuinely new mechanism CATEGORY vs rounds 1-14 (see project memory
`project-eth-omega461-exit-logic-experiments-20260721` and today's rounds 11-14 in
tmp/research_20260722/ and tmp/research_20260723/): a classic technical-indicator combo
(volume-weighted trend filter + volume-at-price support/resistance), not an ML classifier or a
market-wide anomaly detector.

Design:
  1. VWMA(N): causal/trailing volume-weighted moving average,
     sum(close*volume, window=N) / sum(volume, window=N), pandas .rolling(N, min_periods=N)
     (no center=True anywhere). N in {100, 288}.
  2. Causal rolling VPVR: for each bar i, using ONLY the trailing vpvr_window bars ending at
     i-1 (current bar i's own volume is NOT part of its own histogram -- avoids any
     same-bar-volume leakage into the classification of that same bar's price), bin the
     [rolling-low-min, rolling-high-max] price range of that window into N_BINS=24 bins.
     APPROXIMATION USED: each historical bar's full volume is assigned to a single bin using
     that bar's typical price (high+low+close)/3, NOT a full intrabar volume distribution across
     every bin its own [low,high] range spans. Chosen because it fully vectorizes with
     numpy.lib.stride_tricks.sliding_window_view + a single global np.bincount call (no
     per-row Python loop over the window), which is what makes the full VAL+OOS+fresh dataset at
     window=576 tractable; full intrabar distribution would need weighting each bar into several
     bins by overlap fraction, which does not vectorize the same way and would be materially
     slower for 26k+ rows x 576-bar windows. From the resulting per-bar histogram: POC = argmax
     bin, Value Area = contiguous bins built by expanding outward from POC (whichever open
     neighbor bin has more volume) until >= 70% of total window volume is covered (fixed
     value_area_pct=0.70, not gridded). Current bar i's OWN close price is then binned into the
     SAME row's histogram's bin edges and classified HVN (inside Value Area) or LVN (outside).
  3. Trigger (checked once per bar with an open position, AFTER take_profit/stop_loss, BEFORE
     exit_head -- TP/SL are much larger full-exit moves so they take priority; exit_head is the
     baseline's own mechanism and is left as the last-resort catch-all):
       - adverse-side-of-VWMA: long + close < vwma, or short + close > vwma
       - penetration >= penetration_depth * ATR(same atr_window as the component's own TP/SL)
       - CONFIRMED for >= CONFIRM_BARS=2 consecutive bars (fixed choice per task spec, not
         gridded -- noise filter)
       - current bar classified HVN (not LVN -- HVN is the noise filter per the design brief)
       - not inside cooldown (cooldown bars after the PREVIOUS trigger on this open trade)
     On trigger: reduce the REMAINING open notional by reduce_frac (compounds if it fires more
     than once on the same trade), realize that slice's PnL/fee into cash immediately, arm a
     cooldown, and continue holding the (now smaller) remainder. This is a size reduction, not a
     full exit -- the trade only counts as "closed" (ledger row / trades count) when TP, full SL,
     exit_head, or forced-end at window end eventually fires on the remaining notional.

  Grid: vwma_window(2) x vpvr_window(2) x penetration_depth(2) x reduce_frac(2) x cooldown(2)
  = 32 configs x 2 components = 64 VAL runs (full grid, not cut).

  Mandatory sanity check: a config where the trigger can never fire (penetration_depth set
  absurdly high) must reproduce the static exit_threshold=0.95 baseline bit-for-bit.

Windows: VAL = 2025-10-01..2025-12-31 (same VAL-window note as rounds 1-14: this model's OOF
predictions don't exist before 2025-10-01, the canonical 2025-09-01 start would leak into the
parent's own TRAIN split). OOS = 2026-01-01..2026-03-31, single touch, only if a VAL config beats
baseline on BOTH PnL and MDD for a component. Fresh window = 2026-04-01..2026-07-12 (bounded by
extended prediction CSV coverage, same bound used in round 14 v2), diagnostic-only, NOT
selection-influencing.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false. All computations
(VWMA, VPVR histogram, ATR, adverse-streak, cooldown) are strictly trailing/causal.
"""

from __future__ import annotations

import gc
import json
import sys
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

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260724/vwma_vpvr_stoploss_20260724"
BASELINE_EXIT_THRESHOLD = sweep.BASELINE_EXIT_THRESHOLD  # 0.95
DEVICE = sweep.DEVICE
COST_MULT = sweep.COST_MULT
FRESH_START, FRESH_END = "2026-04-01", "2026-07-12"

N_BINS = 24
VALUE_AREA_PCT = 0.70
CONFIRM_BARS = 2

VWMA_WINDOWS = [100, 288]
VPVR_WINDOWS = [288, 576]
PENETRATION_DEPTHS = [0.5, 1.0]
REDUCE_FRACS = [0.25, 0.5]
COOLDOWNS = [50, 100]


# ---------------------------------------------------------------------------
# Signal computation (causal, computed once per component per split, reused across the grid)
# ---------------------------------------------------------------------------

def compute_vwma(frame: pd.DataFrame, window: int) -> np.ndarray:
    """Causal VWMA(N) = trailing sum(close*volume) / trailing sum(volume), min_periods=N."""
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    volume = pd.to_numeric(frame["volume"], errors="raise").to_numpy(dtype=np.float64)
    cv = pd.Series(close * volume).rolling(window=window, min_periods=window).sum()
    vv = pd.Series(volume).rolling(window=window, min_periods=window).sum()
    vv_np = vv.to_numpy(dtype=np.float64)
    vwma = np.where(vv_np > 1.0e-12, (cv.to_numpy(dtype=np.float64) / np.where(vv_np > 1.0e-12, vv_np, 1.0)), np.nan)
    return vwma


def compute_vpvr_hvn(frame: pd.DataFrame, window: int, *, n_bins: int = N_BINS,
                      value_area_pct: float = VALUE_AREA_PCT) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling VPVR HVN/LVN classification. For bar i, histogram is built ONLY from bars
    [i-window, i-1] (current bar i's own volume excluded from its own histogram); bar i's own
    close price is then classified against that histogram. Typical-price binning approximation
    (see module docstring). Returns (is_hvn bool array, valid bool array) both length len(frame);
    valid[i]=False for i < window (insufficient trailing history) and is_hvn defaults False there.
    """
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    volume = pd.to_numeric(frame["volume"], errors="raise").to_numpy(dtype=np.float64)
    typical = (high + low + close) / 3.0
    n = len(close)

    is_hvn = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    if n <= window + 1:
        return is_hvn, valid

    # windows over the trailing history [0, n-2] (i.e. excludes the very last bar), each of
    # length `window`; sliding_window_view row k covers original indices k..k+window-1, which is
    # the trailing window ending at (and including) bar i-1 for target bar i = k+window.
    sw = np.lib.stride_tricks.sliding_window_view
    tp_windows = sw(typical[:-1], window)
    vol_windows = sw(volume[:-1], window)
    low_windows = sw(low[:-1], window)
    high_windows = sw(high[:-1], window)

    lo_min = low_windows.min(axis=1)
    hi_max = high_windows.max(axis=1)
    rng = np.maximum(hi_max - lo_min, 1.0e-9)

    bin_idx = np.clip(((tp_windows - lo_min[:, None]) / rng[:, None] * n_bins).astype(np.int64), 0, n_bins - 1)
    n_rows = tp_windows.shape[0]
    row_ids = np.repeat(np.arange(n_rows, dtype=np.int64), window)
    flat_idx = row_ids * n_bins + bin_idx.ravel()
    hist_flat = np.bincount(flat_idx, weights=vol_windows.ravel(), minlength=n_rows * n_bins)
    hist = hist_flat.reshape(n_rows, n_bins)
    del tp_windows, vol_windows, low_windows, high_windows, bin_idx, row_ids, flat_idx, hist_flat
    gc.collect()

    total = hist.sum(axis=1)
    poc = hist.argmax(axis=1)
    rows = np.arange(n_rows)
    va_lo = poc.copy()
    va_hi = poc.copy()
    cum = hist[rows, poc].astype(np.float64)
    target = value_area_pct * total

    for _ in range(n_bins):
        done = cum >= target
        no_more = (va_lo <= 0) & (va_hi >= n_bins - 1)
        active = ~done & ~no_more
        if not active.any():
            break
        below_idx = np.clip(va_lo - 1, 0, n_bins - 1)
        above_idx = np.clip(va_hi + 1, 0, n_bins - 1)
        below_val = np.where(va_lo > 0, hist[rows, below_idx], -1.0)
        above_val = np.where(va_hi < n_bins - 1, hist[rows, above_idx], -1.0)
        take_above = active & (above_val >= below_val) & (above_val >= 0)
        take_below = active & ~take_above & (below_val >= 0)
        va_hi = np.where(take_above, va_hi + 1, va_hi)
        va_lo = np.where(take_below, va_lo - 1, va_lo)
        cum = cum + np.where(take_above, above_val, 0.0) + np.where(take_below, below_val, 0.0)

    target_price = close[window:window + n_rows]
    tbin = np.clip(((target_price - lo_min) / rng * n_bins).astype(np.int64), 0, n_bins - 1)
    row_is_hvn = (tbin >= va_lo) & (tbin <= va_hi)

    is_hvn[window:window + n_rows] = row_is_hvn
    valid[window:window + n_rows] = True
    return is_hvn, valid


# ---------------------------------------------------------------------------
# Causal bar-by-bar replay with VWMA/VPVR partial stop-loss layered on top of TP/full-SL/exit_head
# ---------------------------------------------------------------------------

@torch.no_grad()
def replay_vwma_vpvr_variant(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple],
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    vwma: np.ndarray,
    is_hvn: np.ndarray,
    vpvr_valid: np.ndarray,
    atr_pct: np.ndarray,
    penetration_depth: float,
    reduce_frac: float,
    cooldown: int,
    exit_threshold: float,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    entry_signal_i = 0
    notional = 0.0
    remaining_notional = 0.0
    leverage = 1.0
    margin_fraction = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    adverse_streak = 0
    cooldown_until = -1
    partial_triggers = 0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    route = hard._route_id(frame)
    from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _try_execution as omega_try_execution, _fill_price as omega_fill_price

    base_np, exit_runtime, pos_idx = rs._prepare_exit_runtime(base_x, loaded_models)

    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            mfe = max(mfe, move)
            mae = min(mae, move)
        else:
            move = 0.0

        if pos != 0:
            reason = ""
            exit_prob = 0.0
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"

            if not reason:
                close_i = float(arrays["close"][i])
                vwma_i = float(vwma[i])
                adverse = False
                if np.isfinite(vwma_i):
                    atr_price = float(atr_pct[i]) * close_i
                    if pos > 0 and close_i < vwma_i:
                        adverse = (vwma_i - close_i) >= penetration_depth * atr_price
                    elif pos < 0 and close_i > vwma_i:
                        adverse = (close_i - vwma_i) >= penetration_depth * atr_price
                adverse_streak = adverse_streak + 1 if adverse else 0
                confirmed = adverse_streak >= CONFIRM_BARS
                can_trigger = (
                    confirmed and bool(vpvr_valid[i]) and bool(is_hvn[i])
                    and i >= cooldown_until and remaining_notional > 1.0e-9
                )
                if can_trigger:
                    filled, exit_px, exit_fee, _route = omega_try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                    if filled:
                        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                        reduce_notional = remaining_notional * float(reduce_frac)
                        cash = cash * (1.0 + raw_exit * reduce_notional)
                        cash -= cash * exit_fee * reduce_notional
                        remaining_notional -= reduce_notional
                        cooldown_until = int(i) + int(cooldown)
                        partial_triggers += 1
                        adverse_streak = 0
                        if remaining_notional <= 1.0e-9:
                            reason = "partial_drained"

            if not reason and remaining_notional > 1.0e-9:
                hold = max(int(i) - int(entry_i), 0)
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                prob = rs._predict_exit_prob_one(
                    base_np, exit_runtime, pos_idx, row_i=int(i), expert=expert,
                    pos_values=[
                        float(pos), float(hold), float(move), float(mfe), float(mae),
                        float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move), float(move + abs(stop_loss)),
                        float(remaining_notional), float(leverage), float(remaining_notional * leverage), float(take_profit), float(stop_loss),
                    ],
                    device=device,
                )
                exit_prob = float(prob)
                if prob >= float(exit_threshold):
                    reason = "exit_head"

            if reason:
                filled, exit_px, exit_fee, _route = omega_try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * remaining_notional)
                cash -= before * exit_fee * remaining_notional
                trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
                trades += 1
                win = int(cash > entry_equity)
                wins += win
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({
                    "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(i),
                    "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                    "exit_timestamp": str(frame["timestamp"].iloc[int(i)]), "side": int(pos), "reason": reason,
                    "win": int(win), "raw_exit_price_move": float(raw_exit), "mfe_price_move": float(mfe),
                    "mae_price_move": float(mae), "trade_return": float(trade_return),
                    "net_per_notional": float(trade_return / max(notional, 1.0e-12)), "notional": float(notional),
                    "remaining_notional_at_close": float(remaining_notional), "partial_triggers": int(partial_triggers),
                    "margin_fraction": float(margin_fraction), "leverage": float(leverage),
                    "exit_prob": float(exit_prob), "take_profit": float(take_profit), "stop_loss": float(stop_loss),
                })
                pos = 0
                partial_triggers = 0
                continue

        eq = cash if pos == 0 else cash * (1.0 + move * remaining_notional)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
        if pos != 0 or not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, fee_paid, _route = omega_try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        row_leverage = float(risk_leverage[int(i)])
        row_margin = float(risk_margin_fraction[int(i)])
        row_notional = row_margin * row_leverage
        if row_notional <= 0.0:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        leverage = row_leverage
        margin_fraction = row_margin
        notional = row_notional
        remaining_notional = row_notional
        base_tp = float(row.get("take_profit", 0.0) or 0.0)
        base_sl = float(row.get("stop_loss", 0.0) or 0.0)
        if bool(notional_scaled_sltp):
            take_profit = base_tp * row_notional
            stop_loss = base_sl * row_notional
        else:
            take_profit = base_tp
            stop_loss = base_sl
        cash -= cash * fee_paid * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        mfe = 0.0
        mae = 0.0
        adverse_streak = 0
        cooldown_until = -1
        partial_triggers = 0

    if pos != 0:
        exit_px = omega_fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * remaining_notional)
        cash -= before * fee_eff * remaining_notional
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
        trades += 1
        win = int(cash > entry_equity)
        wins += win
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append({
            "entry_signal_i": int(entry_signal_i), "entry_i": int(entry_i), "exit_i": int(len(frame) - 1),
            "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]), "exit_timestamp": str(frame["timestamp"].iloc[-1]),
            "side": int(pos), "reason": "forced_end", "win": int(win), "raw_exit_price_move": float(raw_exit),
            "mfe_price_move": float(mfe), "mae_price_move": float(mae), "trade_return": float(trade_return),
            "net_per_notional": float(trade_return / max(notional, 1.0e-12)), "notional": float(notional),
            "remaining_notional_at_close": float(remaining_notional), "partial_triggers": int(partial_triggers),
            "margin_fraction": float(margin_fraction), "leverage": float(leverage), "exit_prob": 0.0,
            "take_profit": float(take_profit), "stop_loss": float(stop_loss),
        })

    n_entries = max(long_entries + short_entries, 1)
    ledger = pd.DataFrame(rows)
    hold_bars = (ledger["exit_i"] - ledger["entry_i"]).clip(lower=0) if len(ledger) else pd.Series(dtype=float)
    total_partial_triggers = int(ledger["partial_triggers"].sum()) if len(ledger) and "partial_triggers" in ledger else 0
    return (
        {
            "pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades),
            "wr": float(wins / trades) if trades else 0.0, "trades_per_day": float(trades / rs._duration_days(frame)),
            "avg_notional": float(notional_sum / n_entries), "avg_leverage": float(leverage_sum / n_entries),
            "avg_hold_bars": float(hold_bars.mean()) if len(hold_bars) else 0.0,
            "max_trade_pnl": float(ledger["trade_return"].max() * 100.0) if len(ledger) else 0.0,
            "p95_trade_pnl": float(ledger["trade_return"].quantile(0.95) * 100.0) if len(ledger) else 0.0,
            "long_entries": int(long_entries), "short_entries": int(short_entries), "exit_reasons": reasons,
            "total_partial_triggers": total_partial_triggers,
        },
        ledger,
    )


def run_one(name: str, p: dict[str, Any], *, vwma_window: int, vpvr_window: int, penetration_depth: float,
            reduce_frac: float, cooldown: int, vwma_arr: np.ndarray, hvn_arr: np.ndarray, valid_arr: np.ndarray,
            atr_pct: np.ndarray) -> dict[str, Any]:
    m, _ledger = replay_vwma_vpvr_variant(
        p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
        vwma=vwma_arr, is_hvn=hvn_arr, vpvr_valid=valid_arr, atr_pct=atr_pct,
        penetration_depth=penetration_depth, reduce_frac=reduce_frac, cooldown=cooldown,
        exit_threshold=BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=COST_MULT,
        notional_scaled_sltp=p["notional_scaled_sltp"], device=DEVICE,
    )
    return {"component": name, "vwma_window": vwma_window, "vpvr_window": vpvr_window,
            "penetration_depth": penetration_depth, "reduce_frac": reduce_frac, "cooldown": cooldown,
            **m, "exit_reasons": json.dumps(m["exit_reasons"])}


def prep_all(frame: pd.DataFrame, pred_dir_key: str, *, oof: bool) -> dict[str, dict[str, Any]]:
    prepped: dict[str, dict[str, Any]] = {}
    for name, cfg in sweep.COMPONENTS.items():
        pred_csv = sweep.EXT_PRED_DIR / name / f"{pred_dir_key}_{cfg['q_tag']}.csv"
        prepped[name] = sweep.prep_component(name, cfg, frame, pred_csv, oof=oof)
    return prepped


def baseline_metrics(prepped: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, p in prepped.items():
        m, _ = sweep.replay_exit_variant(
            p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
            exit_threshold=BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=COST_MULT,
            notional_scaled_sltp=p["notional_scaled_sltp"], device=DEVICE,
        )
        out[name] = m
    return out


def compute_signals_for_split(prepped: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Precompute VWMA (per vwma_window) and VPVR HVN/valid (per vpvr_window) once per
    component/split, plus ATR (component's own atr_window) -- all independent of the
    penetration_depth/reduce_frac/cooldown grid dims, so reused across the whole grid."""
    out: dict[str, dict[str, Any]] = {}
    for name, p in prepped.items():
        f = p["frame"]
        cfg = sweep.COMPONENTS[name]
        vwma_by_window = {w: compute_vwma(f, w) for w in VWMA_WINDOWS}
        vpvr_by_window = {w: compute_vpvr_hvn(f, w) for w in VPVR_WINDOWS}
        atr_pct = atr_eval._atr_pct(f, cfg["atr_window"])
        out[name] = {"vwma": vwma_by_window, "vpvr": vpvr_by_window, "atr_pct": atr_pct}
        gc.collect()
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    print(f"VAL frame rows={len(val_frame)} range=[{val_frame['timestamp'].min()}, {val_frame['timestamp'].max()}]", flush=True)
    val_prepped = prep_all(val_frame, "validation_predictions", oof=True)
    print("stage=val_signals", flush=True)
    val_signals = compute_signals_for_split(val_prepped)

    baseline_val = baseline_metrics(val_prepped)
    print("baseline VAL:", {k: {"pnl": v["pnl"], "mdd": v["mdd"], "trades": v["trades"]} for k, v in baseline_val.items()}, flush=True)

    # --- Sanity check: penetration_depth absurdly high -> trigger can never fire -> must
    # reproduce baseline bit-for-bit. ---
    print("stage=sanity", flush=True)
    sanity_rows = []
    for name, p in val_prepped.items():
        b = baseline_val[name]
        sig = val_signals[name]
        r = run_one(name, p, vwma_window=VWMA_WINDOWS[0], vpvr_window=VPVR_WINDOWS[0],
                    penetration_depth=1.0e9, reduce_frac=0.5, cooldown=50,
                    vwma_arr=sig["vwma"][VWMA_WINDOWS[0]], hvn_arr=sig["vpvr"][VPVR_WINDOWS[0]][0],
                    valid_arr=sig["vpvr"][VPVR_WINDOWS[0]][1], atr_pct=sig["atr_pct"])
        sanity_rows.append({"component": name, "baseline_pnl": b["pnl"], "variant_pnl": r["pnl"],
                             "baseline_mdd": b["mdd"], "variant_mdd": r["mdd"], "baseline_trades": b["trades"],
                             "variant_trades": r["trades"], "variant_partial_triggers": r["total_partial_triggers"]})
    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv(OUT_DIR / "sanity_checks_VAL.csv", index=False)
    print(sanity_df.to_string(index=False), flush=True)
    for row in sanity_rows:
        if (abs(row["baseline_pnl"] - row["variant_pnl"]) > 0.01 or abs(row["baseline_mdd"] - row["variant_mdd"]) > 0.01
                or row["baseline_trades"] != row["variant_trades"] or row["variant_partial_triggers"] != 0):
            print(f"SANITY CHECK FAILED: {row}", flush=True)
            return 1
    print("sanity check PASSED (unreachable penetration_depth reproduces baseline, 0 partial triggers)", flush=True)

    # --- Full VAL grid: vwma_window x vpvr_window x penetration_depth x reduce_frac x cooldown ---
    print("stage=val_grid", flush=True)
    grid_rows = []
    for name, p in val_prepped.items():
        sig = val_signals[name]
        for vw in VWMA_WINDOWS:
            vwma_arr = sig["vwma"][vw]
            for pw in VPVR_WINDOWS:
                hvn_arr, valid_arr = sig["vpvr"][pw]
                for pd_ in PENETRATION_DEPTHS:
                    for rf in REDUCE_FRACS:
                        for cd in COOLDOWNS:
                            grid_rows.append(run_one(
                                name, p, vwma_window=vw, vpvr_window=pw, penetration_depth=pd_,
                                reduce_frac=rf, cooldown=cd, vwma_arr=vwma_arr, hvn_arr=hvn_arr,
                                valid_arr=valid_arr, atr_pct=sig["atr_pct"],
                            ))
    val_grid = pd.DataFrame(grid_rows)
    val_grid.to_csv(OUT_DIR / "vwma_vpvr_grid_VAL.csv", index=False)
    print(val_grid[["component", "vwma_window", "vpvr_window", "penetration_depth", "reduce_frac", "cooldown",
                     "pnl", "mdd", "trades", "wr", "total_partial_triggers"]].to_string(index=False), flush=True)

    winners = []
    for _, r in val_grid.iterrows():
        b = baseline_val[r["component"]]
        if r["pnl"] > b["pnl"] and r["mdd"] > b["mdd"]:  # mdd is negative; "beats" means less negative
            winners.append(r.to_dict())
    winners_df = pd.DataFrame(winners)
    winners_df.to_csv(OUT_DIR / "val_winners.csv", index=False)
    print(f"VAL winners (beat baseline on BOTH pnl and mdd): {len(winners)}", flush=True)
    if len(winners):
        print(winners_df[["component", "vwma_window", "vpvr_window", "penetration_depth", "reduce_frac", "cooldown",
                           "pnl", "mdd", "trades"]].to_string(index=False), flush=True)

    print(f"total_partial_triggers summary (grid-wide): min={val_grid['total_partial_triggers'].min()} "
          f"max={val_grid['total_partial_triggers'].max()} mean={val_grid['total_partial_triggers'].mean():.2f}", flush=True)

    if not len(winners):
        print("stage=done no_val_winners -- skipping OOS run per established discipline (round 4/8/11/14 precedent)", flush=True)
        return 0

    # --- Single OOS touch, only for VAL-winning configs (best pnl per component). ---
    print("stage=oos_confirm", flush=True)
    oos_frame = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"OOS frame rows={len(oos_frame)} range=[{oos_frame['timestamp'].min()}, {oos_frame['timestamp'].max()}]", flush=True)
    oos_prepped = prep_all(oos_frame, "oos_predictions", oof=False)
    baseline_oos = baseline_metrics(oos_prepped)
    oos_signals = compute_signals_for_split(oos_prepped)

    best_by_component: dict[str, dict[str, Any]] = {}
    for w in winners:
        comp = w["component"]
        if comp not in best_by_component or w["pnl"] > best_by_component[comp]["pnl"]:
            best_by_component[comp] = w

    oos_rows = []
    fresh_rows = []
    fresh_frame = None
    fresh_prepped = None
    fresh_signals = None
    for comp, w in best_by_component.items():
        p = oos_prepped[comp]
        sig = oos_signals[comp]
        vw, pw = int(w["vwma_window"]), int(w["vpvr_window"])
        oos_m = run_one(comp, p, vwma_window=vw, vpvr_window=pw, penetration_depth=float(w["penetration_depth"]),
                         reduce_frac=float(w["reduce_frac"]), cooldown=int(w["cooldown"]),
                         vwma_arr=sig["vwma"][vw], hvn_arr=sig["vpvr"][pw][0], valid_arr=sig["vpvr"][pw][1],
                         atr_pct=sig["atr_pct"])
        b = baseline_oos[comp]
        oos_rows.append({**oos_m, "baseline_pnl": b["pnl"], "baseline_mdd": b["mdd"], "baseline_trades": b["trades"],
                          "beats_baseline_both": bool(oos_m["pnl"] > b["pnl"] and oos_m["mdd"] > b["mdd"])})

        # --- Fresh window (2026-04-01..07-12), diagnostic-only, NOT selection-influencing ---
        if fresh_prepped is None:
            fresh_frame = sweep.load_frame(FRESH_START, FRESH_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
            print(f"fresh frame rows={len(fresh_frame)} range=[{fresh_frame['timestamp'].min()}, {fresh_frame['timestamp'].max()}]", flush=True)
            fresh_prepped = prep_all(fresh_frame, "oos_predictions", oof=False)
            fresh_signals = compute_signals_for_split(fresh_prepped)
        pf = fresh_prepped[comp]
        sigf = fresh_signals[comp]
        fresh_m = run_one(comp, pf, vwma_window=vw, vpvr_window=pw, penetration_depth=float(w["penetration_depth"]),
                           reduce_frac=float(w["reduce_frac"]), cooldown=int(w["cooldown"]),
                           vwma_arr=sigf["vwma"][vw], hvn_arr=sigf["vpvr"][pw][0], valid_arr=sigf["vpvr"][pw][1],
                           atr_pct=sigf["atr_pct"])
        baseline_fresh_m, _ = sweep.replay_exit_variant(
            pf["frame"], pf["x"], pf["dec"], pf["loaded"], risk_margin_fraction=pf["margin"], risk_leverage=pf["leverage"],
            exit_threshold=BASELINE_EXIT_THRESHOLD, fee=pf["fee"], slip=pf["slip"], cost_mult=COST_MULT,
            notional_scaled_sltp=pf["notional_scaled_sltp"], device=DEVICE,
        )
        fresh_rows.append({**fresh_m, "baseline_pnl": baseline_fresh_m["pnl"], "baseline_mdd": baseline_fresh_m["mdd"],
                            "baseline_trades": baseline_fresh_m["trades"]})

    oos_df = pd.DataFrame(oos_rows)
    oos_df.to_csv(OUT_DIR / "oos_confirm.csv", index=False)
    print(oos_df.to_string(index=False), flush=True)

    fresh_df = pd.DataFrame(fresh_rows)
    fresh_df.to_csv(OUT_DIR / "fresh_window_diagnostic.csv", index=False)
    print(f"fresh window diagnostic ({FRESH_START}..{FRESH_END}, NOT selection-influencing):", flush=True)
    print(fresh_df.to_string(index=False), flush=True)

    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
