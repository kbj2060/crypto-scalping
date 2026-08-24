#!/usr/bin/env python3
"""RESEARCH ONLY -- ETH 드로다운 거버너, step 2: implement the intrabar circuit breakers the cheap_gate step
(docs/experiments/eth_candidate_drawdown_governor_cheap_gate_20260816.md) recommended building FIRST,
ahead of the pre-entry notional/daily-DD/loss-streak caps. Cheap_gate found that pre-entry throttles
can't help because Odyssey's harness only updates portfolio peak/MDD while a position is *held* --
this script targets exactly that: three circuit breakers evaluated every bar on an OPEN position,
ported from BTC clean_base_deep_gated_drawdown_budget_v5 (docs/model_contracts/
eth_candidate_drawdown_budget_governor_contract_20260815.md "이식 원본 메커니즘"):

- equity_mdd_budget_stop: portfolio running mark-to-market peak vs current equity (the harness
  already computes `peak`/`eq` every held bar for MDD reporting -- this just acts on it).
- hard_loss_stop: unrealized account-PnL floor for the CURRENT trade (unreal <= -hard_loss).
- profit_trailing_lock: giveback of unrealized PnL from ITS OWN best-seen unrealized value,
  independent of the existing TP-relative `trailing_stop` (which only arms once MFE clears a
  fraction of take_profit).

Design decisions carried over from the contract's Open Issues:
- Small PRE-REGISTERED grid, ablated one mechanism at a time (not a full cross product) --
  Open Issue 5 flagged VAL-overfitting risk from copying BTC's much larger grid onto Odyssey's
  ~26-35-trade VAL window.
- Order: equity_mdd_budget_stop and hard_loss_stop checked BEFORE the existing take_profit/
  stop_loss/trailing_stop/exit_head chain (they are portfolio-level risk-budget overrides, matching
  BTC's design where they take priority over the frozen policy's own planned exit).
  profit_trailing_lock is checked alongside the existing trailing_stop check, but as an
  INDEPENDENT mechanism (does not require TP to be armed).
- VAL ONLY. OOS-Q1/OOS-Q2 are not opened by this script -- config selection happens on VAL, single-
  touch OOS confirmation is a separate later step per this repo's confirmation-gate discipline.

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=
false. future_rows_used_for_entry=false.

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

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_drawdown_governor_intrabar_stops_20260816"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05
WINDOW = "val"

# Odyssey4 G0 baseline, ALL 6 windows, no_gate/with_gate -- copied verbatim from
# docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md G0 table.
G0_ODYSSEY4 = {
    "2025q1": ({"pnl": 97.70, "mdd": -20.62, "trades": 28}, {"pnl": 44.98, "mdd": -20.62, "trades": 20}),
    "2025q2": ({"pnl": 65.83, "mdd": -14.17, "trades": 31}, {"pnl": 5.62, "mdd": -23.59, "trades": 19}),
    "2025q3": ({"pnl": -10.63, "mdd": -29.66, "trades": 23}, {"pnl": 20.17, "mdd": -19.72, "trades": 17}),
    "val": ({"pnl": 41.13, "mdd": -21.70, "trades": 35}, {"pnl": 77.31, "mdd": -21.76, "trades": 26}),
    "oos_q1": ({"pnl": 93.27, "mdd": -15.48, "trades": 24}, {"pnl": 67.25, "mdd": -15.48, "trades": 19}),
    "oos_q2": ({"pnl": -9.55, "mdd": -20.76, "trades": 13}, {"pnl": -12.69, "mdd": -20.76, "trades": 10}),
}

# Pre-registered ablation grid -- ONE mechanism varied at a time, others held off (None = disabled).
# Small on purpose (Open Issue 5): 3 + 3 + 2 = 8 replay runs total, not a 3x3x2=18 cross product.
EQUITY_MDD_STOP_GRID = (0.12, 0.16, 0.20)
HARD_LOSS_GRID = (0.03, 0.05, 0.07)
TRAIL_LOCK_GRID = ((0.03, 0.02), (0.05, 0.03))


def log(msg: str) -> None:
    print(f"[candidate_drawdown_governor_intrabar_stops] {msg}", flush=True)


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
# Renamed copy of
# research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.greedy_replay_entry_veto
# plus THREE new checks in the held-position branch (marked below). Every other line is unchanged.
# =====================================================================================================


@torch.no_grad()
def greedy_replay_entry_veto_intrabar_governor(
    frame: pd.DataFrame,
    components: dict,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    equity_mdd_stop: float | None,
    hard_loss: float | None,
    trail_activation: float | None,
    trail_gap: float | None,
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
    # NOTE: the source greedy_replay_entry_veto also has an optional TP-relative `trailing_stop`
    # branch (trailing_activate_frac/trailing_trail_frac), but Odyssey4's own G0 never passes those
    # args, so it is permanently disabled in the G0 baseline this script reproduces -- omitted here
    # rather than carried as dead code.
    rows: list[dict] = []
    reasons: dict[str, int] = {}
    guard_hold_bars = 0
    guard_active_bars = 0
    guard_decision_differs_bars = 0
    veto_bars = 0
    veto_events: list[dict] = []
    # --- intrabar governor: new counters vs the unmodified copy ---
    governor_stop_bars = 0
    # --- end new counters ---

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
            equity_dd = 1.0 - eq / max(peak, 1e-12)
            # unrealized-PnL peak in equity-fraction units. `move` is already side-normalized
            # (positive = favorable for both long and short), so mfe (running max of move) times
            # notional is the best unrealized equity fraction seen so far for either side.
            best_unreal = mfe * notional

            reason = ""
            # --- intrabar governor, checked FIRST: portfolio-level risk-budget overrides ---
            if equity_mdd_stop is not None and equity_dd >= float(equity_mdd_stop):
                reason = "equity_mdd_budget_stop"
            elif hard_loss is not None and unreal <= -abs(float(hard_loss)):
                reason = "hard_loss_stop"
            elif (
                trail_activation is not None
                and trail_gap is not None
                and best_unreal >= float(trail_activation)
                and unreal <= best_unreal - abs(float(trail_gap))
            ):
                reason = "profit_trailing_lock"
            if reason:
                governor_stop_bars += 1
            # --- end intrabar governor ---
            if not reason and take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif not reason and stop_loss > 0.0 and move <= -abs(stop_loss):
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

        # flat: try priority order (unchanged from greedy_replay_entry_veto)
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
            break

    diag = {
        "reason_counts": reasons,
        f"{guard_component}_hold_bars": guard_hold_bars,
        f"{guard_component}_guard_active_bars": guard_active_bars,
        f"{guard_component}_guard_decision_differs_bars": guard_decision_differs_bars,
        "veto_bars": veto_bars,
        "veto_events": veto_events,
        "governor_stop_bars": governor_stop_bars,
    }
    return diag, pd.DataFrame(rows)


def _prep(wname: str, windows: dict, score_by_base, threshold: float, device: torch.device):
    aligned_frame, components, prep_diag = guard.prepare_regime_aware_components(wname, windows, score_by_base, threshold, OUT_DIR, device)
    mask, _ = guard._detector_mask_for_frame(aligned_frame, wname, score_by_base, threshold)
    veto_components = o4._attach_veto_mask(components, mask)
    return aligned_frame, veto_components


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()
    report: dict[str, Any] = {
        "design": "ETH 드로다운 거버너 intrabar governor -- equity_mdd_budget_stop / hard_loss_stop / profit_trailing_lock, VAL-only ablation grid vs Odyssey4 G0.",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "window_used_for_grid": WINDOW,
    }

    log("=== stage=detector_build ===")
    score_by_base, robustness_thresholds, threshold = guard.build_detector()
    windows = gate.load_all_windows()

    # =================================================================================================
    # stage=G0_regression -- all 3 governor params off, ALL 6 windows must reproduce Odyssey4 G0
    # exactly (copy-fidelity proof, same discipline as Odyssey4's own G0b stage).
    # =================================================================================================
    log("=== stage=G0_regression (all 6 windows, governor off) ===")
    g0: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        aligned_frame, veto_components = _prep(wname, windows, score_by_base, threshold, device)
        diag, ledger = greedy_replay_entry_veto_intrabar_governor(
            aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
            equity_mdd_stop=None, hard_loss=None, trail_activation=None, trail_gap=None,
        )
        no_gate, with_gate = _metrics_pair(ledger, aligned_frame)
        ref_ng, ref_wg = G0_ODYSSEY4[wname]
        ok_ng, ok_wg = _close(no_gate, ref_ng), _close(with_gate, ref_wg)
        g0[wname] = {"no_gate": {"actual": no_gate, "reference": ref_ng, "match": ok_ng},
                     "with_gate": {"actual": with_gate, "reference": ref_wg, "match": ok_wg}}
        log(f"  {wname:8s} no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d} match={ok_ng}  "
            f"with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d} match={ok_wg}")
    g0_pass = all(g0[w]["no_gate"]["match"] and g0[w]["with_gate"]["match"] for w in gate.ALL_WINDOWS)
    report["g0_regression"] = {"windows": g0, "pass": g0_pass}
    log(f"stage=G0_regression_result pass={g0_pass}")
    if not g0_pass:
        report["stage_reached"] = "G0_regression"
        report["gate_pass"] = False
        report["note"] = "G0 regression failed -- aborting before trusting any ablation sweep."
        _write_report(report)
        log("stage=ABORT G0 regression failed")
        return 1

    val_frame, val_components = _prep(WINDOW, windows, score_by_base, threshold, device)

    def _run(equity_mdd_stop, hard_loss, trail_activation, trail_gap):
        diag, ledger = greedy_replay_entry_veto_intrabar_governor(
            val_frame, val_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
            equity_mdd_stop=equity_mdd_stop, hard_loss=hard_loss, trail_activation=trail_activation, trail_gap=trail_gap,
        )
        no_gate, with_gate = _metrics_pair(ledger, val_frame)
        return no_gate, with_gate, diag

    # =================================================================================================
    # stage=ablation_A -- equity_mdd_budget_stop alone
    # =================================================================================================
    log("=== stage=ablation_A_equity_mdd_stop ===")
    rows_a = []
    for v in EQUITY_MDD_STOP_GRID:
        ng, wg, diag = _run(v, None, None, None)
        rows_a.append({"equity_mdd_stop": v, "no_gate": ng, "with_gate": wg, "governor_stop_bars": diag["governor_stop_bars"], "stop_fired": diag["reason_counts"].get("equity_mdd_budget_stop", 0)})
        log(f"  eq_mdd_stop={v:.2f}  no_gate={ng['pnl']:7.2f}%/{ng['mdd']:7.2f}%/{ng['trades']:3d}  with_gate={wg['pnl']:7.2f}%/{wg['mdd']:7.2f}%/{wg['trades']:3d}  fired={diag['reason_counts'].get('equity_mdd_budget_stop', 0)}")
    report["ablation_A_equity_mdd_stop"] = rows_a

    # =================================================================================================
    # stage=ablation_B -- hard_loss_stop alone
    # =================================================================================================
    log("=== stage=ablation_B_hard_loss_stop ===")
    rows_b = []
    for v in HARD_LOSS_GRID:
        ng, wg, diag = _run(None, v, None, None)
        rows_b.append({"hard_loss": v, "no_gate": ng, "with_gate": wg, "stop_fired": diag["reason_counts"].get("hard_loss_stop", 0)})
        log(f"  hard_loss={v:.2f}  no_gate={ng['pnl']:7.2f}%/{ng['mdd']:7.2f}%/{ng['trades']:3d}  with_gate={wg['pnl']:7.2f}%/{wg['mdd']:7.2f}%/{wg['trades']:3d}  fired={diag['reason_counts'].get('hard_loss_stop', 0)}")
    report["ablation_B_hard_loss_stop"] = rows_b

    # =================================================================================================
    # stage=ablation_C -- profit_trailing_lock alone
    # =================================================================================================
    log("=== stage=ablation_C_profit_trailing_lock ===")
    rows_c = []
    for act, gap in TRAIL_LOCK_GRID:
        ng, wg, diag = _run(None, None, act, gap)
        rows_c.append({"trail_activation": act, "trail_gap": gap, "no_gate": ng, "with_gate": wg, "stop_fired": diag["reason_counts"].get("profit_trailing_lock", 0)})
        log(f"  trail={act:.2f}/{gap:.2f}  no_gate={ng['pnl']:7.2f}%/{ng['mdd']:7.2f}%/{ng['trades']:3d}  with_gate={wg['pnl']:7.2f}%/{wg['mdd']:7.2f}%/{wg['trades']:3d}  fired={diag['reason_counts'].get('profit_trailing_lock', 0)}")
    report["ablation_C_profit_trailing_lock"] = rows_c

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
