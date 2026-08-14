#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 post-entry literature scouting (#6) rank-5 candidate, DEV-side
evaluation. Compares the Gittins retirement-value h48qual exit_head trained by
scripts/train_eval_omega461_gittins_retirement_exit_head_20260814.py (server, GPU; see that
script's docstring for the paper mechanism -- Dhankhar/Mishra/Bodas arXiv:2405.01157 -- and this
project's reformulation) against the CURRENT Odyssey2 baseline (h48qual TabM live-ATR-relabel exit
head, asymmetric_tabm_liveatr) at both the component level (h48qual standalone ledger,
research_eth_omega461_exit_sweep_20260721.replay_exit_variant-shaped) and the portfolio level
(h48qual+zig075 single-account greedy router, replay_omega4_6_1_greedy_router_20260706.greedy_
replay-shaped). zig075 / direction_head / quality_head / encoder are never touched -- only
h48qual's exit_head DECISION is swapped (probability-threshold classifier -> retirement-value-
threshold regressor).

=== G0 (runs first, unconditionally) ===
Component-level: reuses research_eth_omega461_exit_head_h48cons_relabel_20260813._evaluate_val
unchanged (reproduces component_baseline_original 5.45/-11.62/29 and component_tabm_liveatr
9.23/-7.59/63). Portfolio-level: reuses eth_omega461_multiwindow_confirmation_gate_20260814.
run_portfolio_variant + its own REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR constant (VAL no_gate
46.59/-21.70/35, with_gate 77.31/-21.76/26; OOS-Q1 no_gate 93.27/-15.48/24, with_gate 67.25/-15.48/
19 -- the exact numbers the coordinator specified). If G0 fails, this script aborts before trusting
any Gittins number (same discipline as every prior Odyssey2 candidate script).

=== Gittins injection ===
The trained retirement-value network is injected at the exact per-bar call site every exit_head
variant in this lineage shares (train_eval_omega4_2_risk_sidecar_20260622._prepare_exit_runtime for
column/pos_idx bookkeeping, reused unchanged) via a NEW prediction function
(_predict_retirement_value_one, mirroring _predict_exit_prob_one's exact signature/standardization
but reading the network's OWN diagonal Q(x,x) instead of a softmax-ed classifier logit) and two
RENAMED, targeted-block-only copies of the shared replay harnesses (replay_exit_variant_gittins /
greedy_replay_gittins -- same pattern as GBDT's duck-typed wrapper (#4) and TCN's windowed copies
(#5): the ONLY change from the originals is the exit-head decision block, which compares a
retirement value M(x) against a threshold (M(x) <= retirement_threshold -> exit) instead of a
classifier probability against EXIT_THRESHOLD=0.95. A raw probability-vs-threshold duck-type was
deliberately NOT used here (unlike GBDT's softmax(log(p))==p trick) because M(x) is an unbounded
real-valued regression output, not a probability -- force-fitting it through a fake sigmoid
calibration would add opaque free parameters with no decision-rule benefit over thresholding M(x)
directly (see the companion doc's design section).

=== Threshold calibration ===
retirement_threshold is swept on VAL over a small grid: {0.0 (a priori anchor -- "continuing has
non-positive expected marginal value"), plus the 10th/25th/50th percentile of M(x) observed at
h48qual's own held bars on VAL under a component-only, never-triggers diagnostic pass}, exactly the
same "sweep a handful of thresholds on VAL" discipline as every other Odyssey2 exit-threshold
candidate (queue-pressure's 3-point grid, zig075 recalibration's 8-point grid, etc.).

=== Promotion gate (per the task's explicit instructions) ===
VAL: passes if EITHER (a) the original 4-metric gate (portfolio no_gate PnL/MDD AND with_gate
PnL/MDD all non-worse than asymmetric_tabm_liveatr) OR (b) the relaxed gate (docs/experiments/
eth_omega461_relaxed_gate_rescoring_20260814.md: with_gate PnL improved, with_gate MDD within 3pp)
-- AND, independently of which of (a)/(b) passes, the component guardrail established by GBDT/TCN
(#4/#5): h48qual-standalone PnL must not flip sign or degrade more than 50% relative to the
component_tabm_liveatr baseline (9.23%). NOTE: docs/experiments/eth_omega461_relaxed_gate_
rescoring_20260814.md's own table lists a stale baseline with_gate value (54.88%/-31.11%) that
Odyssey2 log entry #8 discovered was actually the baseline_both_original ledger, not asymmetric_
tabm_liveatr -- this script uses the CORRECT asymmetric_tabm_liveatr with_gate baseline (77.31%/
-21.76%/26, matching both the coordinator's stated G0 and eth_omega461_multiwindow_confirmation_
gate_20260814.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR) for the relaxed-gate comparison too.
OOS: only opened if a VAL winner exists, then a SINGLE TOUCH over OOS-Q1+OOS-Q2 via eth_omega461_
multiwindow_confirmation_gate_20260814.summarize_multiwindow (both windows must pass together, no
sequential/iterative peeking) -- this project's established post-08-14 standard.

fresh_forward_bar_by_bar=true (every replay is a single causal bar-by-bar forward pass, i
increasing, only bar i and already-closed history used at bar i). trade_ledgers_used_as_input=false
(ledgers are written-only outputs). saved_parent_exit_timestamps_used=false. future_rows_used_for_
entry=false. Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py,
runtime_config.py, .env. Does NOT touch zig075.
"""
from __future__ import annotations

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

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_head_h48cons_relabel_20260813 as h48cons  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import train_eval_omega461_gittins_retirement_exit_head_20260814 as gittins_train  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_gittins_index_exit_head_20260814"
GITTINS_BUNDLE = gittins_train.OUT_DIR / "h48qual" / "gittins_retirement_bundle.pt"
PRIORITY = ("h48qual", "zig075")  # local copy of replay_omega4_6_1_greedy_router_20260706.PRIORITY's default value -- not imported-and-mutated, avoids depending on shared global state.

# Published in docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md, reused
# verbatim by every exit_head candidate's G0 in this lineage (GBDT #4 / TCN #5).
G0_REFERENCE = {
    "component_baseline_original": {"pnl": 5.45, "mdd": -11.62, "trades": 29},
    "component_tabm_liveatr": {"pnl": 9.23, "mdd": -7.59, "trades": 63},
}
RETIREMENT_MDD_SLACK_PP = 3.0  # relaxed-gate MDD allowance, matches eth_omega461_relaxed_gate_rescoring_20260814.md
GUARDRAIL_MAX_RELATIVE_DEGRADATION = 0.50  # component PnL must retain >=50% of baseline component PnL, no sign flip
DIAGNOSTIC_NEVER_TRIGGER_THRESHOLD = -1.0e9  # for the M(x) distribution pass: a threshold no real M(x) will ever cross


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


# =====================================================================================================
# Gittins model loading (mirrors GBDT's _load_gbdt_bundle / _gbdt_loaded_models / _inject_gbdt_exit_
# runtime, adapted for a torch state_dict bundle with REAL mean/std standardization instead of a
# pickle + identity scaler -- see train_eval_omega461_gittins_retirement_exit_head_20260814.py's
# DGN class / bundle schema).
# =====================================================================================================
def _load_gittins_bundle(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _gittins_models_from_bundle(bundle: dict[str, Any], device: torch.device) -> dict[str, gittins_train.DGN]:
    models: dict[str, gittins_train.DGN] = {}
    for expert in hard.EXPERT_NAMES:
        m = gittins_train.DGN(int(bundle["arch"]["state_dim"]), hidden=tuple(bundle["arch"]["hidden"]))
        m.load_state_dict(bundle["models"][expert])
        m.eval()
        m.to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        models[expert] = m
    return models


def _gittins_loaded_models(base_cols: list[str], gittins_models: dict[str, Any], bundle: dict[str, Any]) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Shape-compatible with parent._load_payloads' return value (loaded_models[expert] = (model,
    scaler_dict)), for harnesses (replay_exit_variant_gittins) that call rs._prepare_exit_runtime
    themselves. Uses the REAL mean/std this network was standardized with during training (unlike
    GBDT's identity scaler -- a neural net, unlike a GBDT, needs real standardization)."""
    cols = list(base_cols) + list(parent.POS_COLS)
    if cols != list(bundle["all_cols"]):
        raise RuntimeError("Gittins bundle column contract mismatch")
    scaler = {"columns": cols, "mean": np.asarray(bundle["scaler"]["mean"], dtype=np.float32), "std": np.asarray(bundle["scaler"]["std"], dtype=np.float32)}
    return {expert: (gittins_models[expert], scaler) for expert in hard.EXPERT_NAMES}


def _inject_gittins_exit_runtime_portfolio(
    prepped: dict[str, Any], gittins_models: dict[str, Any], bundle: dict[str, Any], base_cols: list[str], retirement_threshold: float,
) -> dict[str, Any]:
    """Shape-compatible override for greedy_replay_gittins (via prepare_component / _prepare_
    component_val, which already built `exit_runtime`) -- replaces only that key plus two new marker
    keys (`is_gittins`, `retirement_threshold`) that greedy_replay_gittins's per-bar branch reads;
    everything else (dec/atr/margin/leverage/route, all exit-head-independent) untouched. Returns a
    new dict (does not mutate `prepped`)."""
    cols = list(base_cols) + list(parent.POS_COLS)
    if cols != list(bundle["all_cols"]):
        raise RuntimeError("Gittins bundle column contract mismatch (portfolio injection)")
    n = int(prepped["base_np"].shape[1])
    if n != len(cols):
        raise RuntimeError(f"Gittins injection column count mismatch: base_np width={n} vs base_cols+POS_COLS={len(cols)}")
    mean = np.asarray(bundle["scaler"]["mean"], dtype=np.float32)
    std = np.asarray(bundle["scaler"]["std"], dtype=np.float32)
    out = dict(prepped)
    out["exit_runtime"] = {expert: (gittins_models[expert], mean, std) for expert in hard.EXPERT_NAMES}
    out["is_gittins"] = True
    out["retirement_threshold"] = float(retirement_threshold)
    return out


# =====================================================================================================
# Prediction + renamed replay copies (targeted block swap only -- see module docstring "Gittins
# injection" section for why a duck-typed probability wrapper was not used here).
# =====================================================================================================
@torch.no_grad()
def _predict_retirement_value_one(
    base_np: np.ndarray,
    runtime: dict[str, tuple[Any, np.ndarray, np.ndarray]],
    pos_idx: list[int],
    *,
    row_i: int,
    expert: str,
    pos_values: list[float],
    device: torch.device,
) -> float:
    """Mirrors train_eval_omega4_2_risk_sidecar_20260622._predict_exit_prob_one's exact signature
    and standardization, but reads the Gittins network's own diagonal M(x)=Q(x,x) instead of a
    softmax-ed TabM/GBDT classifier logit."""
    model, mean, std = runtime[expert]
    row = base_np[int(row_i)].copy()
    row[np.asarray(pos_idx, dtype=np.int64)] = np.asarray(pos_values, dtype=np.float32)
    x = ((row - mean) / std).reshape(1, -1).astype(np.float32)
    xt = torch.from_numpy(x).to(device)
    m = model.diagonal(xt)
    return float(m.detach().cpu().numpy()[0])


@torch.no_grad()
def replay_exit_variant_gittins(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple],
    *,
    risk_margin_fraction: np.ndarray,
    risk_leverage: np.ndarray,
    retirement_threshold: float,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    device: torch.device,
    collect_diagnostics: list[float] | None = None,
    trailing_activate_frac: float | None = None,
    trailing_retain_frac: float | None = None,
    trailing_trail_frac: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Renamed copy of research_eth_omega461_exit_sweep_20260721.replay_exit_variant -- IDENTICAL
    body (TP/SL order, fill/cost model, ledger schema, trailing-stop option kept unused for
    structural fidelity) except the exit-head block: instead of a classifier probability compared
    against exit_threshold, this queries _predict_retirement_value_one and exits when the retirement
    value M(x) falls TO OR BELOW retirement_threshold. If `collect_diagnostics` is passed a list,
    every computed M(x) (not just triggering ones) is appended -- used by the VAL-threshold-grid
    diagnostic pass below, so the grid is chosen from real observed M(x) values, not guessed."""
    trailing_enabled = trailing_activate_frac is not None and (trailing_retain_frac is not None or trailing_trail_frac is not None)
    if trailing_retain_frac is not None and trailing_trail_frac is not None:
        raise ValueError("pass either trailing_retain_frac (proportional) or trailing_trail_frac (fixed distance)")
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
    leverage = 1.0
    margin_fraction = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    armed = False
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    margin_sum = 0.0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    route = hard._route_id(frame)
    from train_eval_omega1_2_tabm_diffusion_risk_20260603 import _try_execution as omega_try_execution, _fill_price as omega_fill_price
    import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit

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
            retirement_value = 0.0
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            elif trailing_enabled and (not armed) and mfe >= float(trailing_activate_frac) * take_profit and take_profit > 0.0:
                armed = True
            if not reason and trailing_enabled and armed and mfe > 0.0:
                if trailing_retain_frac is not None:
                    if move <= float(trailing_retain_frac) * mfe:
                        reason = "trailing_stop"
                elif move <= mfe - float(trailing_trail_frac) * abs(stop_loss):
                    reason = "trailing_stop"
            if not reason:
                hold = max(int(i) - int(entry_i), 0)
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                mval = _predict_retirement_value_one(
                    base_np, exit_runtime, pos_idx, row_i=int(i), expert=expert,
                    pos_values=[
                        float(pos), float(hold), float(move), float(mfe), float(mae),
                        float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move), float(move + abs(stop_loss)),
                        float(notional), float(leverage), float(notional * leverage), float(take_profit), float(stop_loss),
                    ],
                    device=device,
                )
                retirement_value = float(mval)
                if collect_diagnostics is not None:
                    collect_diagnostics.append(float(mval))
                if mval <= float(retirement_threshold):
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _route = omega_try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
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
                    "margin_fraction": float(margin_fraction), "leverage": float(leverage),
                    "retirement_value": float(retirement_value), "take_profit": float(take_profit), "stop_loss": float(stop_loss),
                })
                pos = 0
                armed = False
                continue
        eq = cash if pos == 0 else cash * (1.0 + move * notional)
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
        margin_sum += margin_fraction
        mfe = 0.0
        mae = 0.0
        armed = False

    if pos != 0:
        exit_px = omega_fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
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
            "margin_fraction": float(margin_fraction), "leverage": float(leverage), "retirement_value": 0.0,
            "take_profit": float(take_profit), "stop_loss": float(stop_loss),
        })

    n_entries = max(long_entries + short_entries, 1)
    ledger = pd.DataFrame(rows)
    hold_bars = (ledger["exit_i"] - ledger["entry_i"]).clip(lower=0) if len(ledger) else pd.Series(dtype=float)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades),
            "wr": float(wins / trades) if trades else 0.0, "trades_per_day": float(trades / rs._duration_days(frame)),
            "avg_notional": float(notional_sum / n_entries), "avg_leverage": float(leverage_sum / n_entries),
            "avg_hold_bars": float(hold_bars.mean()) if len(hold_bars) else 0.0,
            "max_trade_pnl": float(ledger["trade_return"].max() * 100.0) if len(ledger) else 0.0,
            "p95_trade_pnl": float(ledger["trade_return"].quantile(0.95) * 100.0) if len(ledger) else 0.0,
            "long_entries": int(long_entries), "short_entries": int(short_entries), "exit_reasons": reasons,
        },
        ledger,
    )


@torch.no_grad()
def greedy_replay_gittins(
    frame: pd.DataFrame, components: dict, *, fee: float, slip: float, cost_mult: float, device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    """Renamed copy of replay_omega4_6_1_greedy_router_20260706.greedy_replay -- IDENTICAL body
    except the exit-head block branches per-component on `comp.get("is_gittins")` (same dynamic-
    branch-by-marker pattern TCN (#5) used for its IS_WINDOWED components): h48qual (is_gittins=True)
    uses _predict_retirement_value_one + comp["retirement_threshold"]; zig075 (unmarked) uses the
    ORIGINAL rs._predict_exit_prob_one + comp["exit_threshold"], completely untouched."""
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
    armed = False
    rows: list[dict] = []
    reasons: dict[str, int] = {}

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
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
                expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                pos_values = [float(pos), float(hold), float(move), float(mfe), float(mae),
                              float(np.clip(giveback, 0.0, 10.0)), float(take_profit - move),
                              float(move + abs(stop_loss)), float(notional), float(leverage_v),
                              float(notional * leverage_v), float(take_profit), float(stop_loss)]
                if comp.get("is_gittins", False):
                    mval = _predict_retirement_value_one(
                        comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                        pos_values=pos_values, device=device,
                    )
                    if mval <= float(comp["retirement_threshold"]):
                        reason = "exit_head"
                else:
                    prob = rs._predict_exit_prob_one(
                        comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                        pos_values=pos_values, device=device,
                    )
                    if prob >= comp["exit_threshold"]:
                        reason = "exit_head"
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

        for name in PRIORITY:
            if name not in components:
                continue
            comp = components[name]
            side = int(comp["dec"]["side"].iloc[i])
            if side == 0 or not bool(omega._active(comp["dec"]).iloc[i] if hasattr(omega._active(comp["dec"]), "iloc") else omega._active(comp["dec"])[i]):
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
            cash -= cash * fee_eff * notional
            mfe = mae = 0.0
            armed = False
            break

    return {"reason_counts": reasons}, pd.DataFrame(rows)


# =====================================================================================================
# Component / portfolio evaluation helpers
# =====================================================================================================
def _evaluate_component_gittins(
    gittins_models: dict[str, Any], bundle: dict[str, Any], retirement_threshold: float, *, collect_diagnostics: list[float] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    val_pred = sweep.EXT_PRED_DIR / "h48qual" / f"validation_predictions_{cfg['q_tag']}.csv"
    prepped = sweep.prep_component("h48qual", cfg, val_frame, val_pred, oof=True)
    base_cols = list(torch.load(cfg["bundle"], map_location="cpu", weights_only=False)["base_cols"])
    gittins_loaded = _gittins_loaded_models(base_cols, gittins_models, bundle)
    metrics, ledger = replay_exit_variant_gittins(
        prepped["frame"], prepped["x"], prepped["dec"], gittins_loaded,
        risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
        retirement_threshold=retirement_threshold, fee=prepped["fee"], slip=prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"], device=sweep.DEVICE,
        collect_diagnostics=collect_diagnostics,
    )
    return metrics, ledger


def _gittins_portfolio_variant(
    window_name: str, windows: dict[str, Any], gittins_models: dict[str, Any], bundle: dict[str, Any],
    retirement_threshold: float, *, fee: float, slip: float, device: torch.device, variant_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    w = windows[window_name]
    split = gate.WINDOW_DEFS[window_name]["split"]
    q_tags = {"h48qual": sweep.COMPONENTS["h48qual"]["q_tag"], "zig075": sweep.COMPONENTS["zig075"]["q_tag"]}
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(w["frame"], q_tags, split, OUT_DIR)
    h48qual_cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)
    zig075_cfg = portfolio._component_cfg("zig075")
    base_cols = list(torch.load(h48qual_cfg["bundle"], map_location="cpu", weights_only=False)["base_cols"])
    prep_fn = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component
    h48qual_prepped = prep_fn(aligned_frame, aligned_paths["h48qual"], h48qual_cfg, device)
    h48qual_gittins = _inject_gittins_exit_runtime_portfolio(h48qual_prepped, gittins_models, bundle, base_cols, retirement_threshold)
    zig075_prepped = prep_fn(aligned_frame, aligned_paths["zig075"], zig075_cfg, device)
    components = {"h48qual": h48qual_gittins, "zig075": zig075_prepped}
    _diag, ledger = greedy_replay_gittins(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
    ledger_path = OUT_DIR / f"portfolio_ledger_{window_name}_{variant_label}.csv"
    ledger.to_csv(ledger_path, index=False)
    no_gate = portfolio._ledger_metrics(ledger)
    with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
    return no_gate, with_gate


def _guardrail_pass(component_pnl: float) -> bool:
    baseline_pnl = float(G0_REFERENCE["component_tabm_liveatr"]["pnl"])
    if component_pnl <= 0.0:
        return False  # sign flip (or zero) vs a positive baseline
    relative_degradation = (baseline_pnl - component_pnl) / abs(baseline_pnl)
    return bool(relative_degradation <= GUARDRAIL_MAX_RELATIVE_DEGRADATION)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")  # dev has no GPU; the bundle was trained on the server's GPU but saved as plain CPU state_dicts.
    fee, slip = omega._load_fee_slip()

    print("=== stage=load_windows ===", flush=True)
    windows = gate.load_all_windows()

    print("=== stage=G0_component ===", flush=True)
    g0_component = h48cons._evaluate_val("h48qual", portfolio.NEW_H48QUAL_BUNDLE)
    g0_ok_component_baseline = gate._close(g0_component["baseline"], G0_REFERENCE["component_baseline_original"])
    g0_ok_component_tabm = gate._close(g0_component["h48cons_relabel"], G0_REFERENCE["component_tabm_liveatr"])
    print(f"  component_baseline_original match={g0_ok_component_baseline} actual={g0_component['baseline']}", flush=True)
    print(f"  component_tabm_liveatr match={g0_ok_component_tabm} actual={g0_component['h48cons_relabel']}", flush=True)

    print("=== stage=G0_portfolio (val + oos_q1, via gate.run_portfolio_variant) ===", flush=True)
    g0_portfolio: dict[str, Any] = {}
    for wname in ("val", "oos_q1"):
        result = gate.run_portfolio_variant(wname, windows, gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="asymmetric_tabm_liveatr")
        ref_ng, ref_wg = gate.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR[wname]
        ok_ng, ok_wg = gate._close(result["no_gate"], ref_ng), gate._close(result["with_gate"], ref_wg)
        g0_portfolio[wname] = {
            "no_gate": {"actual": result["no_gate"], "reference": ref_ng, "match": ok_ng},
            "with_gate": {"actual": result["with_gate"], "reference": ref_wg, "match": ok_wg},
        }
        print(f"  {wname}: no_gate match={ok_ng} {result['no_gate']['pnl']:.2f}%/{result['no_gate']['mdd']:.2f}%/{result['no_gate']['trades']}  "
              f"with_gate match={ok_wg} {result['with_gate']['pnl']:.2f}%/{result['with_gate']['mdd']:.2f}%/{result['with_gate']['trades']}", flush=True)

    g0_pass = bool(g0_ok_component_baseline and g0_ok_component_tabm and all(g0_portfolio[w]["no_gate"]["match"] and g0_portfolio[w]["with_gate"]["match"] for w in ("val", "oos_q1")))
    print(f"stage=G0_result pass={g0_pass}", flush=True)

    report: dict[str, Any] = {
        "paper_citation": "Dhankhar, Mishra, Bodas, arXiv:2405.01157 (retirement formulation, QGI/DGN)",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "g0_component": {
            "component_baseline_original": {"actual": g0_component["baseline"], "reference": G0_REFERENCE["component_baseline_original"], "match": g0_ok_component_baseline},
            "component_tabm_liveatr": {"actual": g0_component["h48cons_relabel"], "reference": G0_REFERENCE["component_tabm_liveatr"], "match": g0_ok_component_tabm},
        },
        "g0_portfolio": g0_portfolio,
        "gate_pass_g0": g0_pass,
    }
    if not g0_pass:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 failed -- this harness does not reproduce already-published reference numbers. Aborting before evaluating the Gittins model."
        (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
        print("stage=ABORT G0 failed", flush=True)
        return 1

    print("=== stage=load_gittins_bundle ===", flush=True)
    if not GITTINS_BUNDLE.exists():
        raise FileNotFoundError(f"Gittins bundle not found, run train_eval_omega461_gittins_retirement_exit_head_20260814.py on the server first: {GITTINS_BUNDLE}")
    bundle = _load_gittins_bundle(GITTINS_BUNDLE)
    gittins_models = _gittins_models_from_bundle(bundle, device)
    print(f"  bundle loaded: gamma={bundle.get('gamma')} arch={bundle.get('arch')}", flush=True)

    print("=== stage=m_distribution_diagnostic (component-only, never-trigger pass) ===", flush=True)
    diag_values: list[float] = []
    _evaluate_component_gittins(gittins_models, bundle, DIAGNOSTIC_NEVER_TRIGGER_THRESHOLD, collect_diagnostics=diag_values)
    arr = np.asarray(diag_values, dtype=np.float64)
    percentiles = {p: float(np.percentile(arr, p)) for p in (10, 25, 50)} if len(arr) else {10: 0.0, 25: 0.0, 50: 0.0}
    grid = sorted({0.0, round(percentiles[10], 6), round(percentiles[25], 6), round(percentiles[50], 6)})
    print(f"  n_observations={len(arr)} percentiles={percentiles} grid={grid}", flush=True)
    report["m_distribution_diagnostic"] = {
        "n_observations": int(len(arr)), "min": float(arr.min()) if len(arr) else None, "max": float(arr.max()) if len(arr) else None,
        "mean": float(arr.mean()) if len(arr) else None, "percentiles": percentiles, "val_threshold_grid": grid,
    }

    print("=== stage=val_sweep ===", flush=True)
    val_candidates: dict[str, Any] = {}
    baseline_val_no_gate = g0_portfolio["val"]["no_gate"]["actual"]
    baseline_val_with_gate = g0_portfolio["val"]["with_gate"]["actual"]
    for thr in grid:
        key = f"{thr:.6f}"
        component_metrics, _ = _evaluate_component_gittins(gittins_models, bundle, thr)
        no_gate, with_gate = _gittins_portfolio_variant("val", windows, gittins_models, bundle, thr, fee=fee, slip=slip, device=device, variant_label=f"val_thr{key}")
        original_gate_pass = bool(
            no_gate["pnl"] >= baseline_val_no_gate["pnl"] and no_gate["mdd"] >= baseline_val_no_gate["mdd"]
            and with_gate["pnl"] >= baseline_val_with_gate["pnl"] and with_gate["mdd"] >= baseline_val_with_gate["mdd"]
        )
        relaxed_gate_pass = bool(with_gate["pnl"] > baseline_val_with_gate["pnl"] and (with_gate["mdd"] - baseline_val_with_gate["mdd"]) >= -abs(RETIREMENT_MDD_SLACK_PP))
        guardrail_pass = _guardrail_pass(float(component_metrics["pnl"]))
        candidate_pass = bool((original_gate_pass or relaxed_gate_pass) and guardrail_pass)
        val_candidates[key] = {
            "threshold": thr, "component": component_metrics, "portfolio_no_gate": no_gate, "portfolio_with_gate": with_gate,
            "original_gate_pass": original_gate_pass, "relaxed_gate_pass": relaxed_gate_pass, "guardrail_pass": guardrail_pass,
            "candidate_pass": candidate_pass,
        }
        print(f"  thr={key}: component pnl={component_metrics['pnl']:.2f}% guardrail={guardrail_pass}  "
              f"portfolio no_gate={no_gate['pnl']:.2f}%/{no_gate['mdd']:.2f}%/{no_gate['trades']} with_gate={with_gate['pnl']:.2f}%/{with_gate['mdd']:.2f}%/{with_gate['trades']}  "
              f"original_gate={original_gate_pass} relaxed_gate={relaxed_gate_pass} candidate_pass={candidate_pass}", flush=True)

    report["val_reference"] = {"no_gate": baseline_val_no_gate, "with_gate": baseline_val_with_gate, "component_tabm_liveatr": G0_REFERENCE["component_tabm_liveatr"]}
    report["val_candidates"] = val_candidates

    passing = {k: v for k, v in val_candidates.items() if v["candidate_pass"]}
    val_winner_key = max(passing, key=lambda k: passing[k]["portfolio_with_gate"]["pnl"]) if passing else None
    val_winner_threshold = val_candidates[val_winner_key]["threshold"] if val_winner_key else None
    report["val_winner_key"] = val_winner_key
    report["val_winner_threshold"] = val_winner_threshold
    print(f"stage=val_sweep_result val_winner_key={val_winner_key} threshold={val_winner_threshold}", flush=True)

    if val_winner_key is None:
        report["oos_opened"] = False
        report["final_verdict"] = "REJECTED_VAL_GATE"
        report["stage_reached"] = "val_sweep"
        report["gate_pass"] = False
        (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
        print(f"report={OUT_DIR / 'report.json'}", flush=True)
        print("stage=done final_verdict=REJECTED_VAL_GATE (no VAL candidate passed gate+guardrail -- OOS not opened)", flush=True)
        return 0

    thr = float(val_winner_threshold)
    print(f"=== stage=oos_confirm (single touch OOS-Q1+OOS-Q2, threshold={thr}) ===", flush=True)
    baseline_tuples: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    candidate_tuples: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    component_by_window: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        b = gate.run_portfolio_variant(wname, windows, gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="asymmetric_tabm_liveatr_gittinsrun")
        baseline_tuples[wname] = (b["no_gate"], b["with_gate"])
        c_ng, c_wg = _gittins_portfolio_variant(wname, windows, gittins_models, bundle, thr, fee=fee, slip=slip, device=device, variant_label=f"gittins_winner_thr{thr:.6f}")
        candidate_tuples[wname] = (c_ng, c_wg)
        print(f"  {wname}: baseline no_gate={b['no_gate']['pnl']:.2f}%/{b['no_gate']['mdd']:.2f}%/{b['no_gate']['trades']} with_gate={b['with_gate']['pnl']:.2f}%/{b['with_gate']['mdd']:.2f}%/{b['with_gate']['trades']}  |  "
              f"candidate no_gate={c_ng['pnl']:.2f}%/{c_ng['mdd']:.2f}%/{c_ng['trades']} with_gate={c_wg['pnl']:.2f}%/{c_wg['mdd']:.2f}%/{c_wg['trades']}", flush=True)

    summary_strict = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=0.0)
    summary_relaxed = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=RETIREMENT_MDD_SLACK_PP)
    print(f"multiwindow verdict: strict={summary_strict['final_verdict']} relaxed_mdd{RETIREMENT_MDD_SLACK_PP}pp={summary_relaxed['final_verdict']}", flush=True)

    final_verdict = "CONFIRMED" if (summary_strict["final_verdict"] == "CONFIRMED" or summary_relaxed["final_verdict"] == "CONFIRMED") else "REJECTED_SIGN_MISMATCH"
    report["oos_opened"] = True
    report["oos_winner_threshold"] = thr
    report["multiwindow_by_window"] = {wname: {"baseline_no_gate": baseline_tuples[wname][0], "baseline_with_gate": baseline_tuples[wname][1], "candidate_no_gate": candidate_tuples[wname][0], "candidate_with_gate": candidate_tuples[wname][1]} for wname in gate.ALL_WINDOWS}
    report["multiwindow_strict_mdd0pp"] = summary_strict
    report["multiwindow_relaxed_mdd3pp"] = summary_relaxed
    report["final_verdict"] = final_verdict
    report["gate_pass"] = True
    report["stage_reached"] = "oos_confirm"

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)
    print(f"stage=done final_verdict={final_verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
