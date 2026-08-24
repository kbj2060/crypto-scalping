#!/usr/bin/env python3
"""RESEARCH ONLY -- light cheap_gate for a candidate idea: cap h48qual's holding time so the shared
slot frees up sooner for zig075. Motivated by two prior findings this session (2026-08-16):
- 15.8% of zig075 signal episodes (765/4844 pooled) are fully blocked because h48qual is already
  holding the shared slot -- concentrated in 2025-Q3 (38.7%) and VAL (23.4%).
- Reordering PRIORITY does NOT recover this (docs/experiments/
  eth_candidate_priority_swap_cheap_gate_20260816.md) -- PRIORITY only matters when both components
  are flat simultaneously, which is rare (0.43% of VAL bars); it cannot touch an already-open
  position. Shortening h48qual's OWN hold time is the other lever identified for this same
  opportunity cost, tested here.

h48qual's VAL hold-time distribution (from the real G0 ledger): p10=79, p25=138, p50=188, p75=405,
p90=882 bars. exit_head already resolves 9/14 h48qual VAL exits -- unlike zig075 (0/86), h48qual's
exit_head is NOT dead. A max-hold cap would specifically clip the long tail (388-1316 bars) where
exit_head hasn't (yet) fired. Grid chosen around/below the median and mid-tail: {150, 250, 400} bars.

Implementation: a renamed copy of research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_
20260814.greedy_replay_entry_veto with ONE new check (h48qual-only, checked after TP/SL and before
the regime-guard/exit_head branch -- a realized TP/SL takes priority; the cap only fires when neither
TP/SL nor exit_head has resolved the trade in time).

VAL ONLY. OOS-Q1/OOS-Q2 not opened by this script.

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py /
trading_bot_modules/runtime_config.py / .env. Does NOT modify any imported module. No GPU.
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

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_h48qual_max_hold_cheap_gate_20260816"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05
WINDOW = "val"
GUARD_COMPONENT = "h48qual"

G0_ODYSSEY4_VAL_WITH_GATE = {"pnl": 77.31, "mdd": -21.76, "trades": 26}
G0_ODYSSEY4_VAL_NO_GATE = {"pnl": 41.13, "mdd": -21.70, "trades": 35}

MAX_HOLD_GRID = (150, 250, 400)


def log(msg: str) -> None:
    print(f"[h48qual_max_hold_cheap_gate] {msg}", flush=True)


def _close(actual, expected, *, tol_pp: float = G0_TOLERANCE_PP) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
        and int(actual["trades"]) == int(expected["trades"])
    )


def _metrics_pair(ledger, aligned_frame):
    no_gate = portfolio._ledger_metrics(ledger)
    with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
    return no_gate, with_gate


# =====================================================================================================
# Renamed copy of greedy_replay_entry_veto plus ONE new check: h48qual_max_hold (None disables it).
# =====================================================================================================


@torch.no_grad()
def greedy_replay_entry_veto_h48qual_max_hold(
    frame: pd.DataFrame,
    components: dict,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    h48qual_max_hold: int | None,
    guard_component: str = GUARD_COMPONENT,
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
    rows: list[dict] = []
    reasons: dict[str, int] = {}
    guard_hold_bars = 0
    guard_active_bars = 0
    guard_decision_differs_bars = 0
    veto_bars = 0
    veto_events: list[dict] = []
    max_hold_stops = 0

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
            # --- h48qual max-hold cap: checked after TP/SL (realized outcome wins), before the
            # regime-guard/exit_head branch (give up waiting rather than let exit_head decide late).
            if not reason and active_comp == guard_component and h48qual_max_hold is not None:
                if (i - entry_i) >= int(h48qual_max_hold):
                    reason = "max_hold_cap"
                    max_hold_stops += 1
            # --- end max-hold cap ---
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
        "max_hold_stops": max_hold_stops,
    }
    return diag, pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()
    report: dict[str, Any] = {
        "design": "light cheap_gate -- cap h48qual holding time so the shared slot frees sooner for zig075, VAL only.",
        "window": WINDOW,
        "max_hold_grid": list(MAX_HOLD_GRID),
    }

    log("=== stage=detector_build ===")
    score_by_base, robustness_thresholds, threshold = guard.build_detector()

    log("=== stage=prepare_val ===")
    aligned_frame, components, prep_diag = guard.prepare_regime_aware_components(WINDOW, gate.load_all_windows(), score_by_base, threshold, OUT_DIR, device)
    mask, _ = guard._detector_mask_for_frame(aligned_frame, WINDOW, score_by_base, threshold)
    veto_components = o4._attach_veto_mask(components, mask)

    log("=== stage=G0_reproduce (max_hold=None) ===")
    diag0, ledger0 = greedy_replay_entry_veto_h48qual_max_hold(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device, h48qual_max_hold=None)
    no_gate0, with_gate0 = _metrics_pair(ledger0, aligned_frame)
    g0_ok = _close(no_gate0, G0_ODYSSEY4_VAL_NO_GATE) and _close(with_gate0, G0_ODYSSEY4_VAL_WITH_GATE)
    report["g0_reproduce"] = {"no_gate": no_gate0, "with_gate": with_gate0, "pass": g0_ok}
    log(f"  no_gate={no_gate0['pnl']:.2f}%/{no_gate0['mdd']:.2f}%/{no_gate0['trades']}  with_gate={with_gate0['pnl']:.2f}%/{with_gate0['mdd']:.2f}%/{with_gate0['trades']}  match={g0_ok}")
    if not g0_ok:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        _write_report(report)
        log("stage=ABORT G0 failed")
        return 1

    log("=== stage=max_hold_sweep ===")
    rows = []
    for cap in MAX_HOLD_GRID:
        diag_c, ledger_c = greedy_replay_entry_veto_h48qual_max_hold(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device, h48qual_max_hold=cap)
        no_gate_c, with_gate_c = _metrics_pair(ledger_c, aligned_frame)
        h48 = ledger_c[ledger_c["source_component"] == "h48qual"]
        zig = ledger_c[ledger_c["source_component"] == "zig075"]
        row = {
            "max_hold": cap, "no_gate": no_gate_c, "with_gate": with_gate_c,
            "max_hold_stops": diag_c["max_hold_stops"],
            "h48qual_trades": int(len(h48)), "h48qual_wr": float((h48["win"] == 1).mean()) if len(h48) else None,
            "zig075_trades": int(len(zig)),
        }
        rows.append(row)
        log(f"  cap={cap:4d}  no_gate={no_gate_c['pnl']:7.2f}%/{no_gate_c['mdd']:7.2f}%/{no_gate_c['trades']:3d}  "
            f"with_gate={with_gate_c['pnl']:7.2f}%/{with_gate_c['mdd']:7.2f}%/{with_gate_c['trades']:3d}  "
            f"max_hold_stops={diag_c['max_hold_stops']}  h48qual_trades={len(h48)}(wr={row['h48qual_wr']})  zig075_trades={len(zig)}")
    report["max_hold_sweep"] = rows

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
