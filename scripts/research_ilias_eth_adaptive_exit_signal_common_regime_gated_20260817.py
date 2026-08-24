#!/usr/bin/env python3
"""RESEARCH ONLY -- Ilias regime-gated hybrid session (2026-08-17, follow-up to the side-blind
correction in research_ilias_eth_adaptive_exit_signal_common_sideblind_20260817.py).

=== User question this answers ===
docs/experiments/eth_zig075_veto_ranging_misfire_fix_candidate_20260817.md's "추가" section found that
h48qual's DEPLOYED regime-aware exit guard (research_eth_omega461_regime_aware_exit_head_uptrend_
guard_20260814: ON -> h48qual's ORIGINAL frozen exit head threshold=0.95, OFF -> the live-ATR-relabeled
exit head) is causally INERT against its EXISTING exit_head OFF-branch -- real_g0 PnL is byte-identical
across NONE/V1/V3 detector masks in all 6 windows
(scripts/research_eth_odyssey4_h48qual_exit_guard_ranging_misfire_test_20260817.py). That inertness was
measured against the OLD (liveATR-relabeled) OFF-branch model. Ilias's contract
(docs/model_contracts/ilias_eth_human_direction_risk_management_contract_20260817.md, Layer Contracts
L9 row) inherits this guard UNCHANGED. This module asks: does the SAME detector, gating the SAME
ON-branch (h48qual original exit_head, threshold=0.95, byte-identical to the deployed guard), but with
the OFF-branch REPLACED by Ilias's own side-blind new exit signal (research_ilias_eth_adaptive_exit_
signal_common_sideblind_20260817.py, threshold=0.5) instead of the liveATR-relabeled exit head, still
behave inertly -- or does it do better than the side-blind signal running unguarded everywhere (the
result already on file in arm_eval_report_sideblind.json)?

=== Design (agreed in-session with the user, applied without re-asking) ===
Zero new free variables -- detector, threshold, both exit models, and their respective decision
thresholds are ALL pre-existing/already-validated, none re-tuned or newly introduced here:
  - ON branch  (detector==sustained-uptrend, `sustained_uptrend_mask[i]` True): h48qual's ORIGINAL
    (pre-liveATR-relabel) exit-head weights, threshold=0.95 -- exactly the deployed guard's own ON
    branch, unchanged (`comp["guard_base_np"]`/`comp["guard_exit_runtime"]`/`comp["guard_pos_idx"]`/
    `comp["guard_exit_threshold"]`, attached by research_eth_odyssey4_random_direction_risk_
    management_ablation_20260817.build_ablation_components exactly as it already is for the deployed
    guard -- not reconstructed here).
  - OFF branch (detector inactive): Ilias's side-blind new exit signal
    (tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/
    new_exit_signal_bundle_sideblind.pkl, threshold=0.5) -- replaces the spot the liveATR-relabeled
    exit head used to occupy in the deployed guard, via `comp["new_exit_model"]`/
    `comp["new_exit_threshold"]` (same attachment convention research_ilias_eth_adaptive_exit_signal_
    common_20260817.greedy_replay_new_exit_signal already uses for the UNGATED side-blind signal).
  - TP/SL always takes priority (unchanged, checked before either branch). zig075's own SHORT entry
    veto is completely untouched (same `short_entry_veto_mask` plumbing every script in this lineage
    already carries -- this module never looks at it).
  - Any component OTHER than `guard_component` (i.e. zig075), or an h48qual component that happens to
    carry NEITHER a mask NOR a `new_exit_model` (used only by this module's own G0-identity check, see
    the arm-eval script), falls back to that component's own plain exit_head
    (`comp["base_np"]`/`comp["exit_runtime"]`/`comp["pos_idx"]`/`comp["exit_threshold"]`) -- byte-
    identical to every other replay function in this lineage's own fallback behaviour.

Documented copy of research_ilias_eth_adaptive_exit_signal_common_sideblind_20260817.
greedy_replay_new_exit_signal_sideblind -- the ONLY logic change is the position-holding "not reason"
block: instead of "new_exit_model present -> always use it" (sideblind's ungated behaviour), this
version checks `sustained_uptrend_mask[i]` FIRST (mirroring research_eth_omega461_regime_aware_exit_
head_uptrend_guard_20260814.greedy_replay_regime_aware_exit_guard's own mask check) and only falls
through to the new_exit_model when the mask is inactive. `simulate_private_barrier_trades`,
`score_new_exit_signal`, `POS_VALUE_NAMES`, `FEATURE_COLUMNS`, `NEW_EXIT_THRESHOLD_DEFAULT` are all
re-exported UNCHANGED from the side-blind module (single source of truth -- criterion 1 in the arm-eval
script does not depend on gating at all, see that script's own docstring for why its numbers are
reused rather than recomputed here).

fresh_forward_bar_by_bar=true for `greedy_replay_new_exit_signal_regime_gated` (a live/causal replay --
decisions at bar i depend only on bar i, the pre-computed rolling detector mask up to bar i [itself a
plain backward-.rolling(), no negative shift -- see the guard module's own docstring], and already-
closed history). trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/* / runtime_config.py / .env. Does NOT modify any
imported module (including both non-gated ilias common modules and the regime-aware-exit-guard module
-- all read-only imports). No retraining, no GPU (DEVICE=cpu), conda env quant_ai.
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

import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import research_ilias_eth_adaptive_exit_signal_common_sideblind_20260817 as common_sb  # noqa: E402

COMPONENT = common_sb.COMPONENT
FEATURE_COLUMNS = common_sb.FEATURE_COLUMNS  # side-blind 10-feature list, unchanged
NEW_EXIT_THRESHOLD_DEFAULT = common_sb.NEW_EXIT_THRESHOLD_DEFAULT  # 0.5, unchanged
POS_VALUE_NAMES = common_sb.POS_VALUE_NAMES

# Re-exported unchanged -- see module docstring.
simulate_private_barrier_trades = common_sb.simulate_private_barrier_trades
score_new_exit_signal = common_sb.score_new_exit_signal


def log(msg: str) -> None:
    print(f"[ilias_regime_gated] {msg}", flush=True)


# =====================================================================================================
# Documented copy of common_sb.greedy_replay_new_exit_signal_sideblind. ONE change vs that function:
# the position-holding "not reason" block now checks `sustained_uptrend_mask[i]` FIRST (ON ->
# comp['guard_*'], h48qual's original exit head, threshold=0.95) and only falls through to
# `comp['new_exit_model']` (OFF -> Ilias side-blind signal, threshold=0.5) when the mask is inactive
# (or absent). Everything else (TP/SL priority, zig075 SHORT entry veto, flat-entry sizing/priority,
# ledger bookkeeping) is byte-identical to the sideblind original.
# =====================================================================================================
@torch.no_grad()
def greedy_replay_new_exit_signal_regime_gated(
    frame: pd.DataFrame,
    components: dict,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    guard_component: str = "h48qual",
) -> tuple[dict, pd.DataFrame]:
    """While `guard_component` (h48qual) holds an open position, the exit-decision branch is chosen
    per bar as: `components[guard_component]['sustained_uptrend_mask'][i]` True -> ORIGINAL (pre-
    relabel) exit head via `comp['guard_base_np']`/`comp['guard_exit_runtime']`/`comp['guard_pos_idx']`/
    `comp['guard_exit_threshold']` (0.95); else, if `comp['new_exit_model']` is attached -> Ilias
    side-blind classifier via `score_new_exit_signal` at `comp.get('new_exit_threshold',
    NEW_EXIT_THRESHOLD_DEFAULT)` (0.5); else (no mask and no new_exit_model -- other components, or an
    identity-check config) -> that component's own plain exit head (`comp['base_np']`/
    `comp['exit_runtime']`/`comp['pos_idx']`/`comp['exit_threshold']`), tagged reason "exit_head" for
    parity with every other replay function in this lineage. Reason strings are tagged distinctly
    ("exit_head_regime_on" / "new_exit_signal_regime_off" / "exit_head") so `reason_counts` can
    distinguish a trivial pass (new signal never actually fired) from real engagement -- see the
    arm-eval script's own trivial-pass check.
    """
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
                pos_values_by_name = dict(zip(POS_VALUE_NAMES, [
                    float(pos), float(hold), float(move), float(mfe), float(mae),
                    giveback_clipped, float(take_profit - move),
                    float(move + abs(stop_loss)), float(notional), float(leverage_v),
                    float(notional * leverage_v), float(take_profit), float(stop_loss),
                ]))
                # --- regime-gated hybrid: only new logic vs greedy_replay_new_exit_signal_sideblind ---
                mask = comp.get("sustained_uptrend_mask")
                detector_on = bool(active_comp == guard_component and mask is not None and bool(mask[i]))
                if detector_on:
                    expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                    pos_values = [pos_values_by_name[c] for c in POS_VALUE_NAMES]
                    prob = rs._predict_exit_prob_one(
                        comp["guard_base_np"], comp["guard_exit_runtime"], comp["guard_pos_idx"], row_i=int(i),
                        expert=expert, pos_values=pos_values, device=device,
                    )
                    active_threshold = float(comp.get("guard_exit_threshold", comp["exit_threshold"]))
                    fired_reason = "exit_head_regime_on"
                else:
                    new_model = comp.get("new_exit_model")
                    if new_model is not None:
                        prob = score_new_exit_signal(new_model, pos_values_by_name, entry_quality)
                        active_threshold = float(comp.get("new_exit_threshold", NEW_EXIT_THRESHOLD_DEFAULT))
                        fired_reason = "new_exit_signal_regime_off"
                    else:
                        expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                        pos_values = [pos_values_by_name[c] for c in POS_VALUE_NAMES]
                        prob = rs._predict_exit_prob_one(
                            comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                            pos_values=pos_values, device=device,
                        )
                        active_threshold = float(comp["exit_threshold"])
                        fired_reason = "exit_head"
                if prob >= active_threshold:
                    reason = fired_reason
                # --- end regime-gated hybrid block ---
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
