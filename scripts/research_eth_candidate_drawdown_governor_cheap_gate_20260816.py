#!/usr/bin/env python3
"""RESEARCH ONLY -- ETH 드로다운 거버너 cheap_gate: before building the full drawdown-budget state machine
(docs/model_contracts/eth_candidate_drawdown_budget_governor_contract_20260815.md), check whether a
trivially cheap risk cut already captures the MDD improvement. Two candidates, VAL window only
(the confirmation-gate discipline this repo uses reserves OOS for a config chosen on VAL -- this
script does not open OOS):

1. NOTIONAL_CAP sweep -- just lower the existing fixed constant (currently 1.8) and see how much
   MDD improves for how much PnL given up. No new state, no new logic.
2. Daily-loss-halt -- the simplest possible piece of the full governor design (account/daily DD
   caps + loss-streak cap + 3 intrabar circuit breakers): track a daily equity peak and skip new
   entries for the rest of that calendar day once realized loss from that peak exceeds a threshold.

Both reuse research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814's causal replay
machinery unmodified (import only). Candidate 1 needs no new function (monkeypatches the module
constant the replay loop already reads). Candidate 2 is a renamed copy of that script's
greedy_replay_entry_veto with ONE new block added (day-boundary tracking + a halt check in the flat
entry loop) -- same "copy + one block" pattern Odyssey4 itself used on Odyssey3's guard replay.

Baseline for comparison = Odyssey4 G0 (h48qual regime-aware exit guard + zig075 SHORT sustained-
uptrend entry veto, VAL window, with_gate = duration-OU-halflife gate applied): PnL 77.31%,
MDD -21.76%, trades 26 (docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md
G0 table). This script first reproduces that number via the unmodified baseline path before trusting
anything else.

fresh_forward_bar_by_bar=true (same causal replay loop, i increasing, day-boundary/halt state built
only from bars already seen). trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=
false. future_rows_used_for_entry=false. VAL window only -- OOS-Q1/OOS-Q2 not opened by this script.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py /
trading_bot_modules/runtime_config.py / .env. Does NOT modify any imported module. No retraining,
no GPU (DEVICE=cpu).
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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as o4  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_drawdown_governor_cheap_gate_20260816"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05
WINDOW = "val"

# Odyssey4 G0 baseline for VAL, with_gate column, copied verbatim from
# docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md G0 table.
G0_ODYSSEY4_VAL_WITH_GATE = {"pnl": 77.31, "mdd": -21.76, "trades": 26}
G0_ODYSSEY4_VAL_NO_GATE = {"pnl": 41.13, "mdd": -21.70, "trades": 35}

NOTIONAL_CAP_GRID = (1.8, 1.5, 1.2, 0.9, 0.6)
DAILY_LOSS_HALT_GRID = (0.05, 0.08, 0.12)


def log(msg: str) -> None:
    print(f"[candidate_drawdown_governor_cheap_gate] {msg}", flush=True)


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
        and int(actual["trades"]) == int(expected["trades"])
    )


def _metrics_pair(ledger: pd.DataFrame, aligned_frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    no_gate = portfolio._ledger_metrics(ledger)
    with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
    return no_gate, with_gate


# =====================================================================================================
# Candidate 2: renamed copy of
# research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.greedy_replay_entry_veto
# plus ONE new block: a per-day realized-equity peak and a halt on new entries once that day's
# drawdown from the peak exceeds daily_loss_halt. Every other line is unchanged (marked below).
# =====================================================================================================


@torch.no_grad()
def greedy_replay_entry_veto_daily_halt(
    frame: pd.DataFrame,
    components: dict,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    daily_loss_halt: float,
    guard_component: str = "h48qual",
) -> tuple[dict, pd.DataFrame]:
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
    guard_hold_bars = 0
    guard_active_bars = 0
    guard_decision_differs_bars = 0
    veto_bars = 0
    veto_events: list[dict] = []
    # --- daily-loss-halt: new state vs the unmodified copy ---
    day_key: str | None = None
    daily_peak = 1.0
    halt_bars = 0
    halt_events: list[dict] = []
    # --- end new state ---

    for i in range(0, n - 2):
        # --- daily-loss-halt: day-boundary peak reset, evaluated every bar (mirrors BTC
        # train_eval_clean_base_deep_drawdown_min_v4.py's daily_peak bookkeeping) ---
        key = pd.Timestamp(frame["timestamp"].iloc[i]).date().isoformat() if "timestamp" in frame.columns else str(i // 288)
        if key != day_key:
            day_key = key
            daily_peak = max(cash, 1e-12)
        daily_peak = max(daily_peak, cash)
        daily_dd = max(0.0, 1.0 - cash / max(daily_peak, 1e-12))
        # --- end daily-loss-halt bookkeeping ---
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
                expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                pos_values = [float(pos), float(hold), float(move), float(mfe), float(mae),
                              giveback_clipped, float(take_profit - move),
                              float(move + abs(stop_loss)), float(notional), float(leverage_v),
                              float(notional * leverage_v), float(take_profit), float(stop_loss)]
                use_guard = False
                mask = comp.get("sustained_uptrend_mask")
                if active_comp == guard_component and mask is not None and bool(mask[i]):
                    use_guard = True
                if use_guard:
                    guard_active_bars += 1
                    prob = rs._predict_exit_prob_one(
                        comp["guard_base_np"], comp["guard_exit_runtime"], comp["guard_pos_idx"], row_i=int(i),
                        expert=expert, pos_values=pos_values, device=device,
                    )
                    active_threshold = float(comp.get("guard_exit_threshold", comp["exit_threshold"]))
                    default_prob = rs._predict_exit_prob_one(
                        comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                        pos_values=pos_values, device=device,
                    )
                    if (prob >= active_threshold) != (default_prob >= float(comp["exit_threshold"])):
                        guard_decision_differs_bars += 1
                else:
                    prob = rs._predict_exit_prob_one(
                        comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                        pos_values=pos_values, device=device,
                    )
                    active_threshold = float(comp["exit_threshold"])
                if prob >= active_threshold:
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

        # --- daily-loss-halt: skip all new entries once today's realized drawdown breaches the
        # threshold. Checked in the flat-state loop only -- never touches an already-open position,
        # so it can only remove entries, never change side/TP/SL/exit of a held trade. ---
        if daily_dd >= float(daily_loss_halt):
            halt_bars += 1
            halt_events.append({"i": int(i), "timestamp": str(frame["timestamp"].iloc[i]), "daily_dd": float(daily_dd)})
            continue
        # --- end daily-loss-halt check ---

        # flat: try priority order
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
            cash -= cash * fee_eff * notional
            mfe = mae = 0.0
            armed = False
            break

    diag = {
        "reason_counts": reasons,
        f"{guard_component}_hold_bars": guard_hold_bars,
        f"{guard_component}_guard_active_bars": guard_active_bars,
        f"{guard_component}_guard_decision_differs_bars": guard_decision_differs_bars,
        "veto_bars": veto_bars,
        "veto_events": veto_events,
        "halt_bars": halt_bars,
        "halt_events": halt_events,
    }
    return diag, pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()
    report: dict[str, Any] = {
        "design": "ETH 드로다운 거버너 cheap_gate -- NOTIONAL_CAP sweep + daily-loss-halt, VAL only, vs Odyssey4 G0.",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "window": WINDOW,
    }

    log("=== stage=detector_build ===")
    score_by_base, robustness_thresholds, threshold = guard.build_detector()
    log(f"  primary(p90)={threshold:.10f}")

    log("=== stage=prepare_val ===")
    aligned_frame, components, prep_diag = guard.prepare_regime_aware_components(WINDOW, gate.load_all_windows(), score_by_base, threshold, OUT_DIR, device)
    mask, _ = guard._detector_mask_for_frame(aligned_frame, WINDOW, score_by_base, threshold)
    veto_components = o4._attach_veto_mask(components, mask)

    # =================================================================================================
    # stage=G0 -- reproduce the Odyssey4 VAL baseline via the unmodified imported function before
    # trusting any sweep. NOTIONAL_CAP left at its live default (greedy.NOTIONAL_CAP, 1.8).
    # =================================================================================================
    log("=== stage=G0_reproduce ===")
    assert float(greedy.NOTIONAL_CAP) == 1.8, f"expected live NOTIONAL_CAP=1.8, found {greedy.NOTIONAL_CAP} -- aborting, config drift"
    diag0, ledger0 = o4.greedy_replay_entry_veto(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
    no_gate0, with_gate0 = _metrics_pair(ledger0, aligned_frame)
    g0_ok = _close(no_gate0, G0_ODYSSEY4_VAL_NO_GATE) and _close(with_gate0, G0_ODYSSEY4_VAL_WITH_GATE)
    report["g0_reproduce"] = {"no_gate": no_gate0, "with_gate": with_gate0, "reference_no_gate": G0_ODYSSEY4_VAL_NO_GATE, "reference_with_gate": G0_ODYSSEY4_VAL_WITH_GATE, "pass": g0_ok}
    log(f"  no_gate={no_gate0['pnl']:.2f}%/{no_gate0['mdd']:.2f}%/{no_gate0['trades']}  with_gate={with_gate0['pnl']:.2f}%/{with_gate0['mdd']:.2f}%/{with_gate0['trades']}  match={g0_ok}")
    if not g0_ok:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 reproduction failed -- aborting before trusting any cheap_gate sweep."
        _write_report(report)
        log("stage=ABORT G0 failed")
        return 1

    # =================================================================================================
    # stage=notional_cap_sweep -- candidate 1: monkeypatch greedy.NOTIONAL_CAP, no new logic.
    # =================================================================================================
    log("=== stage=notional_cap_sweep ===")
    cap_rows: list[dict[str, Any]] = []
    original_cap = float(greedy.NOTIONAL_CAP)
    try:
        for cap in NOTIONAL_CAP_GRID:
            greedy.NOTIONAL_CAP = float(cap)
            diag_c, ledger_c = o4.greedy_replay_entry_veto(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
            no_gate_c, with_gate_c = _metrics_pair(ledger_c, aligned_frame)
            row = {"notional_cap": cap, "no_gate": no_gate_c, "with_gate": with_gate_c}
            cap_rows.append(row)
            log(f"  cap={cap:.2f}  no_gate={no_gate_c['pnl']:7.2f}%/{no_gate_c['mdd']:7.2f}%/{no_gate_c['trades']:3d}  with_gate={with_gate_c['pnl']:7.2f}%/{with_gate_c['mdd']:7.2f}%/{with_gate_c['trades']:3d}")
    finally:
        greedy.NOTIONAL_CAP = original_cap
    report["notional_cap_sweep"] = cap_rows

    # =================================================================================================
    # stage=daily_loss_halt_sweep -- candidate 2: new function, day-boundary tracking + halt only.
    # =================================================================================================
    log("=== stage=daily_loss_halt_sweep ===")
    halt_rows: list[dict[str, Any]] = []
    for thr in DAILY_LOSS_HALT_GRID:
        diag_h, ledger_h = greedy_replay_entry_veto_daily_halt(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device, daily_loss_halt=thr)
        no_gate_h, with_gate_h = _metrics_pair(ledger_h, aligned_frame)
        row = {"daily_loss_halt": thr, "no_gate": no_gate_h, "with_gate": with_gate_h, "halt_bars": diag_h["halt_bars"]}
        halt_rows.append(row)
        log(f"  halt={thr:.2f}  no_gate={no_gate_h['pnl']:7.2f}%/{no_gate_h['mdd']:7.2f}%/{no_gate_h['trades']:3d}  with_gate={with_gate_h['pnl']:7.2f}%/{with_gate_h['mdd']:7.2f}%/{with_gate_h['trades']:3d}  halt_bars={diag_h['halt_bars']}")
    report["daily_loss_halt_sweep"] = halt_rows

    report["stage_reached"] = "done"
    report["gate_pass"] = True
    _write_report(report)
    log("stage=done")
    return 0


def _write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
