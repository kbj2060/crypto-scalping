#!/usr/bin/env python3
"""RESEARCH ONLY -- Ilias subproject, 1st research question: shared primitives for the
direction-quality-reactive exit-signal baseline.

Implements docs/experiments/ilias_eth_adaptive_exit_direction_quality_signal_design_20260817.md's
method-3 candidate-signal design and docs/experiments/ilias_eth_exit_head_passivity_root_cause_
20260817.md's diagnosis/recommendation (labels must be redefined as "does this trade eventually hit
stop_loss vs take_profit", feature plumbing (pos_* / POS_COLS) needs no rebuild).

Two primitives, both used by the labels/train/arm-eval scripts in this line
(research_ilias_eth_adaptive_exit_signal_labels_20260817.py,
research_ilias_eth_adaptive_exit_signal_train_20260817.py,
research_ilias_eth_adaptive_exit_signal_arm_eval_20260817.py):

1. `simulate_private_barrier_trades` -- causal, counterfactual, single-component-private (no
   portfolio slot sharing) TP/SL-barrier simulation. Resolves the circular-logic trap flagged in the
   design doc's pitfall #4 ("what does the new signal's own label do with trades exit_head already
   cut short?") by throwing away exit_head/guard/trailing-stop entirely and walking every entry all
   the way to its REAL terminal barrier (take_profit vs stop_loss), ignoring how the deployed system
   would actually have exited it. This is standard triple-barrier OFFLINE LABEL construction (it
   reads future price bars to build a training TARGET) -- it is not a live/causal DECISION and does
   not violate the Fresh-Forward rule (.claude/CLAUDE.md), which governs decisions, not label
   construction. The position-holding formulas (move/mfe/mae/giveback/dist_to_tp/dist_to_sl/
   notional/leverage/exposure) and flat-entry sizing (SCALE_MAP/LEVERAGE_CAP/NOTIONAL_CAP) are copied
   verbatim from research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.
   greedy_replay_entry_veto's position-holding/flat-entry blocks (that module is imported read-only,
   never edited) so a feature row built here is numerically identical to what the real replay's
   `pos_values` would compute for the same trade at the same bar.

2. `greedy_replay_new_exit_signal` -- a documented copy of research_eth_omega461_zig075_short_entry_
   veto_sustained_uptrend_20260814.greedy_replay_entry_veto (itself a documented copy of
   research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.
   greedy_replay_regime_aware_exit_guard, itself a copy of replay_omega4_6_1_greedy_router_20260706.
   greedy_replay -- the repo's established "documented renamed copy" convention for adding ONE new
   rule to a locked replay loop without touching the original). The ONLY change vs
   greedy_replay_entry_veto: if a component carries a `new_exit_model` (a small frozen classifier
   bundle, see research_ilias_eth_adaptive_exit_signal_train_20260817.py), that component's
   exit-decision branch (normally exit_head, or the h48qual regime-aware guard switching between two
   exit_head weight sets) is replaced with the new classifier scored on the SAME causal pos_values
   used everywhere else in this loop, plus the position's entry-time quality_for_action (tracked as a
   new `entry_quality` scalar, populated at entry from `comp["dec"]["quality_score"]`, mirroring how
   `take_profit`/`stop_loss` are already tracked as entry-time scalars in the original). No
   `new_exit_model` attached -> byte-identical to greedy_replay_entry_veto's own behaviour (verified
   by the calling scripts' own G0 fidelity checks, not asserted here).

fresh_forward_bar_by_bar=true for `greedy_replay_new_exit_signal` (a live/causal replay -- decisions
at bar i depend only on bar i and already-closed history). `simulate_private_barrier_trades` is label
construction, not a live decision -- see its own docstring for the Fresh-Forward-rule distinction.
trade_ledgers_used_as_input=false (ledgers are write-only outputs of both functions).
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false (both functions only ever
open a NEW position using that same bar's OWN causal signal -- future price bars are consumed only to
resolve an ALREADY-OPEN position's own terminal barrier, standard triple-barrier practice, never to
decide whether/when to enter).

Does NOT touch trading_bot.py / trading_bot_modules/* / runtime_config.py / .env. Does NOT modify any
imported module. No retraining of the underlying TabM/GBM models, no GPU (DEVICE=cpu), conda env
quant_ai.
"""
from __future__ import annotations

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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as veto_mod  # noqa: E402

COMPONENT = "h48qual"

# Feature order used EVERYWHERE in this line (label CSV columns, classifier training columns, and the
# live per-bar scoring vector inside greedy_replay_new_exit_signal) -- the first 13 are POS_COLS
# (trading_bot_modules/odyssey_tabm_core.py:45-59) in their canonical order, byte-identical values to
# what the real exit_head's own pos_values list computes; the 14th is the one new derived feature the
# root-cause diagnosis flagged as missing (entry-time quality_for_action, held constant for the whole
# trade) -- no other new raw feature is introduced (contract Shared Feature Contract: "신규 피처를
# 도입하지 않는다").
FEATURE_COLUMNS = [
    "pos_side", "pos_hold_bars", "pos_unrealized", "pos_mfe", "pos_mae", "pos_giveback",
    "pos_dist_to_tp", "pos_dist_to_sl", "pos_notional", "pos_leverage", "pos_exposure",
    "pos_tp", "pos_sl", "entry_quality_for_action",
]
NEW_EXIT_THRESHOLD_DEFAULT = 0.5  # pre-registered fixed decision threshold, chosen before viewing
# any of the 6 evaluation windows (see training script docstring) -- never tuned per-window.


def log(prefix: str, msg: str) -> None:
    print(f"[{prefix}] {msg}", flush=True)


# =====================================================================================================
# (1) Counterfactual TP/SL-barrier label/feature simulation.
# =====================================================================================================
def simulate_private_barrier_trades(
    frame: pd.DataFrame, comp: dict[str, Any], *, fee: float, slip: float, cost_mult: float,
    component_scale_prefix: str = COMPONENT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """See module docstring item 1. `comp` must carry `dec` (action/side/quality_score/take_profit/
    stop_loss, e.g. from replay_omega4_6_1_greedy_router_20260706.prepare_component /
    research_eth_omega461_exit_head_portfolio_asymmetric_20260813._prepare_component_val /
    research_eth_odyssey4_random_direction_risk_management_ablation_20260817.
    prepare_component_direction_override -- any of the three, unmodified) and `margin`/`leverage`
    (same source). Returns (feature_df, diag) where feature_df has one row per (trade_id, bar_i) with
    FEATURE_COLUMNS plus trade_id/bar_i/terminal_reason/label_sl, restricted to bars where a real
    exit-decision would actually have been evaluated (take_profit/stop_loss did not already trigger
    at that bar's own close -- mirrors the real loop's `if not reason:` gate) AND whose trade resolved
    (dropped: trades still open when the frame ends, count in diag).
    """
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(frame)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    dec = comp["dec"]
    side_arr = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    active_arr = np.asarray(omega._active(dec), dtype=bool)
    quality_arr = pd.to_numeric(dec["quality_score"], errors="raise").to_numpy(dtype=np.float64)
    tp_arr = pd.to_numeric(dec["take_profit"], errors="raise").to_numpy(dtype=np.float64)
    sl_arr = pd.to_numeric(dec["stop_loss"], errors="raise").to_numpy(dtype=np.float64)
    margin_arr = np.asarray(comp["margin"], dtype=np.float64)
    leverage_arr = np.asarray(comp["leverage"], dtype=np.float64)

    pos = 0
    entry_price = 0.0
    entry_i = 0
    notional = leverage_v = 0.0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    entry_quality = 0.0
    trade_id = -1
    n_trades = 0
    rows: list[dict[str, Any]] = []
    trade_terminal: dict[int, str] = {}

    for i in range(0, n - 2):
        if pos != 0:
            move = (
                (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price
                if pos > 0
                else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
            )
            mfe, mae = max(mfe, move), min(mae, move)
            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            if not reason:
                hold = max(i - entry_i, 0)
                giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
                giveback_clipped = float(np.clip(giveback, 0.0, 10.0))
                rows.append({
                    "trade_id": trade_id, "bar_i": i,
                    "pos_side": float(pos), "pos_hold_bars": float(hold), "pos_unrealized": float(move),
                    "pos_mfe": float(mfe), "pos_mae": float(mae), "pos_giveback": giveback_clipped,
                    "pos_dist_to_tp": float(take_profit - move), "pos_dist_to_sl": float(move + abs(stop_loss)),
                    "pos_notional": float(notional), "pos_leverage": float(leverage_v),
                    "pos_exposure": float(notional * leverage_v), "pos_tp": float(take_profit),
                    "pos_sl": float(stop_loss), "entry_quality_for_action": float(entry_quality),
                })
            else:
                trade_terminal[trade_id] = reason
                pos = 0
            continue
        side = int(side_arr[i])
        if side == 0 or not bool(active_arr[i]):
            continue
        row_margin, row_leverage = float(margin_arr[i]), float(leverage_arr[i])
        if row_margin <= 0.0:
            continue
        scale = greedy.SCALE_MAP.get(f"{component_scale_prefix}_{'L' if side > 0 else 'S'}", 1.0)
        row_leverage = min(row_leverage * scale, greedy.LEVERAGE_CAP)
        row_notional = min(row_margin * row_leverage, greedy.NOTIONAL_CAP)
        row_leverage = row_notional / max(row_margin, 1e-12)
        if row_notional <= 0.0:
            continue
        tp, sl = float(tp_arr[i]), float(sl_arr[i])
        if tp <= 0.0 or sl <= 0.0:
            continue
        entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip_eff if side > 0 else 1 - slip_eff)
        pos = side
        entry_price = float(entry_px)
        entry_i = min(i + 1, n - 1)
        notional, leverage_v = row_notional, row_leverage
        take_profit, stop_loss = tp, sl
        mfe = mae = 0.0
        entry_quality = float(quality_arr[i])
        trade_id += 1
        n_trades += 1

    feat_df = pd.DataFrame(rows)
    diag: dict[str, Any] = {
        "n_trades_total": n_trades, "n_trades_resolved": len(trade_terminal),
        "n_trades_truncated_open_at_frame_end": n_trades - len(trade_terminal),
        "n_rows_raw": int(len(feat_df)),
    }
    if feat_df.empty:
        diag["n_rows_labeled"] = 0
        feat_df["terminal_reason"] = pd.Series(dtype=object)
        feat_df["label_sl"] = pd.Series(dtype=int)
        return feat_df, diag

    feat_df["terminal_reason"] = feat_df["trade_id"].map(trade_terminal)
    n_dropped_rows = int(feat_df["terminal_reason"].isna().sum())
    feat_df = feat_df.dropna(subset=["terminal_reason"]).reset_index(drop=True)
    feat_df["label_sl"] = (feat_df["terminal_reason"] == "stop_loss").astype(int)
    diag["n_rows_truncated_dropped"] = n_dropped_rows
    diag["n_rows_labeled"] = int(len(feat_df))
    diag["label_positive_rate"] = float(feat_df["label_sl"].mean()) if len(feat_df) else 0.0
    return feat_df, diag


def score_new_exit_signal(new_exit_bundle: dict[str, Any], pos_values: list[float], entry_quality: float) -> float:
    x = np.asarray([list(pos_values) + [float(entry_quality)]], dtype=np.float64)
    proba = new_exit_bundle["model"].predict_proba(x)[0]
    return float(proba[1])


# =====================================================================================================
# (2) Documented copy of veto_mod.greedy_replay_entry_veto -- ONE inserted block (marked below).
# =====================================================================================================
@torch.no_grad()
def greedy_replay_new_exit_signal(
    frame: pd.DataFrame,
    components: dict,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    guard_component: str = "h48qual",
) -> tuple[dict, pd.DataFrame]:
    """Identical to veto_mod.greedy_replay_entry_veto (zig075 SHORT entry veto, h48qual regime-aware
    exit guard, TP/SL priority, all preserved), except: if `components[active_comp]` carries a
    `new_exit_model`, that component's ENTIRE exit-decision branch (guard included) is replaced by
    scoring `new_exit_model` on this bar's pos_values + the position's own entry-time quality_for_
    action -- reason is tagged "new_exit_signal" (not "exit_head", so exit-reason-distribution
    diagnostics can distinguish the two) if the classifier's P(label_sl) crosses
    `components[active_comp].get("new_exit_threshold", NEW_EXIT_THRESHOLD_DEFAULT)`."""
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(frame)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    active_comp = None
    entry_price = entry_equity = 1.0
    entry_i = entry_signal_i = 0
    notional = leverage_v = margin_fraction = 0.0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    entry_quality = 0.0
    rows: list[dict] = []
    reasons: dict[str, int] = {}
    guard_hold_bars = 0
    veto_bars = 0
    veto_events: list[dict] = []

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
            if active_comp == guard_component:
                guard_hold_bars += 1
            move = (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
            unreal = move * notional
            mfe, mae = max(mfe, move), min(mae, move)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            if not reason:
                hold = max(i - entry_i, 0)
                giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
                giveback_clipped = float(np.clip(giveback, 0.0, 10.0))
                pos_values = [float(pos), float(hold), float(move), float(mfe), float(mae),
                              giveback_clipped, float(take_profit - move),
                              float(move + abs(stop_loss)), float(notional), float(leverage_v),
                              float(notional * leverage_v), float(take_profit), float(stop_loss)]
                # --- new-exit-signal: only new logic vs veto_mod.greedy_replay_entry_veto ---
                new_model = comp.get("new_exit_model")
                if new_model is not None:
                    prob = score_new_exit_signal(new_model, pos_values, entry_quality)
                    active_threshold = float(comp.get("new_exit_threshold", NEW_EXIT_THRESHOLD_DEFAULT))
                    fired_reason = "new_exit_signal"
                else:
                    expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                    use_guard = False
                    mask = comp.get("sustained_uptrend_mask")
                    if active_comp == guard_component and mask is not None and bool(mask[i]):
                        use_guard = True
                    if use_guard:
                        prob = rs._predict_exit_prob_one(
                            comp["guard_base_np"], comp["guard_exit_runtime"], comp["guard_pos_idx"], row_i=int(i),
                            expert=expert, pos_values=pos_values, device=device,
                        )
                        active_threshold = float(comp.get("guard_exit_threshold", comp["exit_threshold"]))
                    else:
                        prob = rs._predict_exit_prob_one(
                            comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                            pos_values=pos_values, device=device,
                        )
                        active_threshold = float(comp["exit_threshold"])
                    fired_reason = "exit_head"
                if prob >= active_threshold:
                    reason = fired_reason
                # --- end new-exit-signal block ---
            if reason:
                exit_px = arrays["close"][i] * (1 - slip_eff if pos > 0 else 1 + slip_eff)
                raw_exit = (exit_px - entry_price) / entry_price if pos > 0 else (entry_price - exit_px) / entry_price
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee_eff * notional
                trade_return = cash / max(entry_equity, 1e-12) - 1.0
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({"entry_signal_i": entry_signal_i, "entry_i": entry_i, "exit_i": i,
                             "entry_timestamp": str(frame["timestamp"].iloc[entry_signal_i]),
                             "exit_timestamp": str(frame["timestamp"].iloc[i]), "side": int(pos),
                             "source_component": active_comp, "reason": reason,
                             "win": int(cash > entry_equity), "trade_return": float(trade_return),
                             "notional": float(notional), "margin_fraction": float(margin_fraction),
                             "leverage": float(leverage_v)})
                pos, active_comp = 0, None
                continue
            continue

        for name in greedy.PRIORITY:
            if name not in components:
                continue
            comp = components[name]
            side = int(comp["dec"]["side"].iloc[i])
            if side == 0 or not bool(omega._active(comp["dec"]).iloc[i] if hasattr(omega._active(comp["dec"]), "iloc") else omega._active(comp["dec"])[i]):
                continue
            veto_mask = comp.get("short_entry_veto_mask")
            if veto_mask is not None and side < 0 and bool(veto_mask[i]):
                veto_bars += 1
                veto_events.append({"i": int(i), "timestamp": str(frame["timestamp"].iloc[i]), "component": name})
                continue
            row_margin, row_leverage = float(comp["margin"][i]), float(comp["leverage"][i])
            if row_margin <= 0.0:
                continue
            scale = greedy.SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
            row_leverage = min(row_leverage * scale, greedy.LEVERAGE_CAP)
            row_notional = min(row_margin * row_leverage, greedy.NOTIONAL_CAP)
            row_leverage = row_notional / max(row_margin, 1e-12)
            if row_notional <= 0.0:
                continue
            entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip_eff if side > 0 else 1 - slip_eff)
            pos, active_comp = side, name
            entry_price, entry_equity = float(entry_px), cash
            entry_i, entry_signal_i = min(i + 1, n - 1), i
            margin_fraction, leverage_v, notional = row_margin, row_leverage, row_notional
            take_profit = float(comp["dec"]["take_profit"].iloc[i])
            stop_loss = float(comp["dec"]["stop_loss"].iloc[i])
            entry_quality = float(comp["dec"]["quality_score"].iloc[i])
            cash -= cash * fee_eff * notional
            mfe = mae = 0.0
            break

    diag = {
        "reason_counts": reasons,
        f"{guard_component}_hold_bars": guard_hold_bars,
        "veto_bars": veto_bars,
        "veto_events": veto_events,
    }
    return diag, pd.DataFrame(rows)
