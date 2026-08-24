#!/usr/bin/env python3
"""RESEARCH ONLY -- Ilias correction session (2026-08-17, same day as the baseline this corrects):
side-blind variant of research_ilias_eth_adaptive_exit_signal_common_20260817.py.

=== Why this file exists ===
Opening the baseline session's frozen model (`new_exit_signal_bundle.pkl`, LogisticRegression,
StandardScaler pipeline) and reading its standardized coefficients directly showed quasi-separation on
4 of its 14 features: pos_side (-27.14), pos_leverage (-25.52), pos_exposure (-22.65), pos_notional
(+21.32) -- all |coef| an order of magnitude above every other feature (next largest: pos_tp -3.67).
pos_side is literally the always_long/always_short arm-defining variable (a constant +1 or -1 for the
whole arm) and the TRAIN label window (2025-01-01..2025-09-30) was itself a SHORT-favoring downtrend
regime -- so the plausible read is that the classifier partly memorized "this trade's structural side
(+ sizing outputs correlated with it) predicts the TRAIN-period winning direction", not "this trade's
own unrealized/dist_to_sl trajectory predicts its own SL/TP outcome" (pos_unrealized coef -0.0011,
pos_dist_to_sl coef -0.0011, both ~0). That would make the baseline's "6/6 windows pass criterion 1"
result partly a rediscovery of the always_long/always_short arm LABEL itself, not evidence of a
direction-quality-reactive post-entry signal.

This module removes the 4 direction/sizing-exposing columns (pos_side, pos_leverage, pos_notional,
pos_exposure -- sizing outputs are downstream of side via SCALE_MAP's L/S split and therefore also leak
direction) from the feature set and re-verifies the finding with a side-blind classifier trained on the
SAME label CSV. The remaining 10 features (pos_hold_bars, pos_unrealized, pos_mfe, pos_mae,
pos_giveback, pos_dist_to_tp, pos_dist_to_sl, pos_tp, pos_sl, entry_quality_for_action) are all
path-dependent/side-normalized quantities that do not directly expose which side the position is on.

Documented copy of research_ilias_eth_adaptive_exit_signal_common_20260817.py -- only
`score_new_exit_signal` and `greedy_replay_new_exit_signal` change (both generalized to select
bundle["feature_columns"] BY NAME from a name->value dict, instead of assuming the original's fixed
14-column pos_values list order -- necessary because a bundle trained on a feature SUBSET/reordering
would otherwise be scored with misaligned columns, or raise a sklearn n_features mismatch).
`simulate_private_barrier_trades` is imported UNCHANGED and re-exported: its output already contains
ALL raw pos_*/entry_quality_for_action/label_sl columns regardless of which subset a downstream
classifier trains on, so no duplicate implementation is needed or wanted (single source of truth for
label/feature construction).

fresh_forward_bar_by_bar=true for `greedy_replay_new_exit_signal_sideblind` (unchanged from the
original -- a live/causal replay). `simulate_private_barrier_trades` (re-exported) is offline label
construction, not a live decision -- see its own docstring in the original module.
trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/* / runtime_config.py / .env. Does NOT modify
research_ilias_eth_adaptive_exit_signal_common_20260817.py (imported read-only, original baseline
run's outputs are preserved untouched for the research record). No GPU (DEVICE=cpu), conda env
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

import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import research_ilias_eth_adaptive_exit_signal_common_20260817 as common  # noqa: E402

COMPONENT = common.COMPONENT

# Direction/sizing-exposing columns excluded -- see module docstring quasi-separation diagnosis.
# pos_side: literal arm-defining direction constant. pos_leverage/pos_notional/pos_exposure: sizing
# outputs that are downstream of side (SCALE_MAP has separate "_L"/"_S" entries), so they also encode
# direction indirectly.
EXCLUDED_DIRECTION_SIZING_COLUMNS = ["pos_side", "pos_leverage", "pos_notional", "pos_exposure"]
FEATURE_COLUMNS = [c for c in common.FEATURE_COLUMNS if c not in EXCLUDED_DIRECTION_SIZING_COLUMNS]
NEW_EXIT_THRESHOLD_DEFAULT = common.NEW_EXIT_THRESHOLD_DEFAULT  # unchanged, still pre-registered 0.5

# Re-exported unchanged -- label/feature construction does not depend on which FEATURE_COLUMNS subset a
# downstream classifier trains on (see module docstring).
simulate_private_barrier_trades = common.simulate_private_barrier_trades


def log(msg: str) -> None:
    common.log("ilias_sideblind", msg)


# Position-holding value NAMES in the fixed order greedy_replay_new_exit_signal_sideblind builds them
# below -- matches common.FEATURE_COLUMNS[:13] (POS_COLS canonical order). Used to build a name->value
# dict so score_new_exit_signal (below) can select an ARBITRARY bundle["feature_columns"]
# subset/order, instead of assuming the fixed 14-column layout the ORIGINAL score_new_exit_signal
# hardcodes (which would silently misalign columns, or raise a sklearn shape-mismatch error, if scored
# against a bundle trained on a different subset -- exactly this module's use case).
POS_VALUE_NAMES = common.FEATURE_COLUMNS[:13]


def score_new_exit_signal(new_exit_bundle: dict[str, Any], pos_values_by_name: dict[str, float], entry_quality: float) -> float:
    """Generalization of common.score_new_exit_signal: selects bundle['feature_columns'] BY NAME from a
    name->value dict, instead of assuming the fixed 14-column pos_values list order. Works correctly
    for both the side-blind (10-feature) bundle and, for robustness, any full-feature bundle too."""
    values_by_name = dict(pos_values_by_name)
    values_by_name["entry_quality_for_action"] = float(entry_quality)
    x = np.asarray([[float(values_by_name[c]) for c in new_exit_bundle["feature_columns"]]], dtype=np.float64)
    proba = new_exit_bundle["model"].predict_proba(x)[0]
    return float(proba[1])


# =====================================================================================================
# Documented copy of common.greedy_replay_new_exit_signal (itself a documented copy of
# veto_mod.greedy_replay_entry_veto) -- ONE inserted-block change vs the original: pos_values is built
# as a NAME->value dict (not a fixed-order list) and scored via this module's feature-name-generic
# score_new_exit_signal instead of common.score_new_exit_signal's fixed-14-column version. Everything
# else (TP/SL priority, zig075 SHORT entry veto, regime-aware exit guard fallback when no
# new_exit_model is attached, sizing, ledger bookkeeping) is byte-identical to the original.
# =====================================================================================================
@torch.no_grad()
def greedy_replay_new_exit_signal_sideblind(
    frame: pd.DataFrame,
    components: dict,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    guard_component: str = "h48qual",
) -> tuple[dict, pd.DataFrame]:
    """Identical to common.greedy_replay_new_exit_signal, except: if `components[active_comp]` carries a
    `new_exit_model`, that component's exit-decision branch is scored via this module's
    score_new_exit_signal (feature-name-generic) on a NAME->value dict of this bar's pos_values plus the
    position's own entry-time quality_for_action -- reason is tagged "new_exit_signal" if the
    classifier's P(label_sl) crosses `components[active_comp].get("new_exit_threshold",
    NEW_EXIT_THRESHOLD_DEFAULT)`. No `new_exit_model` attached -> byte-identical behaviour to
    common.greedy_replay_new_exit_signal / veto_mod.greedy_replay_entry_veto."""
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
                # --- new-exit-signal (side-blind generalization): only change vs
                # common.greedy_replay_new_exit_signal -- name->value dict instead of a fixed-order
                # list, so an arbitrary feature SUBSET/order can be selected by score_new_exit_signal.
                pos_values_by_name = dict(zip(POS_VALUE_NAMES, [
                    float(pos), float(hold), float(move), float(mfe), float(mae),
                    giveback_clipped, float(take_profit - move),
                    float(move + abs(stop_loss)), float(notional), float(leverage_v),
                    float(notional * leverage_v), float(take_profit), float(stop_loss),
                ]))
                new_model = comp.get("new_exit_model")
                if new_model is not None:
                    prob = score_new_exit_signal(new_model, pos_values_by_name, entry_quality)
                    active_threshold = float(comp.get("new_exit_threshold", NEW_EXIT_THRESHOLD_DEFAULT))
                    fired_reason = "new_exit_signal"
                else:
                    pos_values = [pos_values_by_name[c] for c in POS_VALUE_NAMES]
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
