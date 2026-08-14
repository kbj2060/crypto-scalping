#!/usr/bin/env python3
"""RESEARCH ONLY -- follow-up to docs/experiments/eth_omega4_6_1_atr_tpsl_floor_binding_investigation_20260812.md
(which found the live "ATR-adaptive" TP/SL is a misnomer: tp_mult=12.0/sl_mult=6.0 are so small
relative to ETH 5m ATR% that min_tp=0.075/min_sl=0.040 bind 95-98.5% of the time and max_tp=0.22/
max_sl=0.12 NEVER bind in 2025-2026 -- i.e. it is effectively a fixed 7.5%/4.0% target, not ATR-
adaptive at all). That doc left "bug vs intended design" as an open, unresolved question requiring
user judgment. This script does NOT resolve that judgment call -- it empirically tests whether
RECALIBRATING tp_mult/sl_mult so the ATR term actually engages (crossing the floor near the median
bar instead of only in the extreme tail) changes VAL/OOS performance, so the judgment call can be
made with evidence instead of just theory.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env, or
any live deployed threshold/bundle. Reuses the SAME frozen h48qual/zig075 parent bundles + risk
sidecars + already-generated OOF/held-out prediction CSVs as every other 2026-08-13 SLTP experiment
(research_eth_omega461_exit_sweep_20260721.prep_component/replay_exit_variant for component-level,
replay_omega4_6_1_greedy_router_20260706.greedy_replay for the priority-combined portfolio level --
the actual live combination mechanism -- via research_eth_omega461_live_sltp_mfe_width_20260813's
_as_router_component/_ledger_stats/_duration_gated helpers). No retraining -- tp_mult/sl_mult are
runtime execution constants (eval_omega4_1_atr_safety_sltp_20260622._apply_atr_safety_sltp), not
learned model weights, so this is a pure deterministic backtest replay. Because nothing is retrained,
there is no seed-diversity dimension to this specific test (unlike the model-retrain tracks earlier
tonight) -- a single deterministic VAL sweep is as decisive as this mechanism can be; what still
requires single-touch discipline is OOS, exactly as everywhere else in this sub-project.

Design: keep the tp_mult:sl_mult ratio fixed at the live value (12:6 = 2:1) and sweep the overall
scale, so this isolates "how strongly ATR-adaptive" the barrier is without changing the TP:SL
risk-reward shape. VAL atr_pct (window=192) percentiles computed directly from data (2025-10-01..
12-31): p50=0.2696%, p90=0.4256%, p99=0.6685%, max=0.9468%. At the baseline tp_mult=12, the floor
(0.075) crosses only above ~p97-98 (matches the parent investigation's 95-98.5% floor-bind finding).
A tp_mult around 28 would put the floor-crossing point near the MEDIAN instead. Grid below spans
from baseline to beyond that point.

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false. VAL window = 2025-10-01..2025-12-31 (base_sweep's window, same as
every sibling SLTP experiment tonight). OOS window = 2026-01-01..2026-03-31, single touch, only
opened if a candidate clears the VAL bar below.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as base_sweep  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as helpers  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as router  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_atr_tpsl_recalibration_pilot_20260813"

# (tp_mult, sl_mult) -- ratio fixed at the live 2:1. Index 0 is the exact live baseline (must
# reproduce known baseline numbers exactly -- self-consistency check before trusting anything else).
CANDIDATE_GRID: list[tuple[float, float]] = [
    (12.0, 6.0),   # baseline (live, unmodified)
    (16.0, 8.0),
    (22.0, 11.0),
    (28.0, 14.0),  # ~ puts the floor-crossing point near the VAL median atr_pct
]

BASELINE_EXIT_THRESHOLD = 0.95  # unchanged -- isolate the TP/SL axis only, exactly like the
# sibling SLTP-width/asymmetric-TP/SL experiments did.


def log(msg: str) -> None:
    print(f"[atr_recal] {msg}", flush=True)


def _floor_bind_rate(frame: pd.DataFrame, cfg: dict, dec: pd.DataFrame) -> dict[str, float]:
    """% of ACTIVE (side != 0) bars where the ATR-scaled raw TP/SL would sit at-or-below the floor
    (i.e. the floor is the binding constraint), computed the same way the parent investigation did."""
    atr_pct = base_sweep.atr_eval._atr_pct(frame, cfg["atr_window"])
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    active = side != 0
    if not active.any():
        return {"tp_floor_bind_rate": float("nan"), "sl_floor_bind_rate": float("nan"), "n_active": 0}
    tp_raw = atr_pct[active] * float(cfg["tp_mult"])
    sl_raw = atr_pct[active] * float(cfg["sl_mult"])
    return {
        "tp_floor_bind_rate": float((tp_raw <= cfg["min_tp"]).mean()),
        "sl_floor_bind_rate": float((sl_raw <= cfg["min_sl"]).mean()),
        "n_active": int(active.sum()),
    }


def run_split(split_name: str, frame: pd.DataFrame, *, oof: bool, grid: list[tuple[float, float]]) -> dict[str, Any]:
    log(f"=== split={split_name} rows={len(frame)} range=[{frame['timestamp'].min()}, {frame['timestamp'].max()}] ===")
    component_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []

    for tp_mult, sl_mult in grid:
        label = f"tp{tp_mult:g}_sl{sl_mult:g}"
        prepped: dict[str, dict[str, Any]] = {}
        for name, base_cfg in base_sweep.COMPONENTS.items():
            cfg = dict(base_cfg)
            cfg["tp_mult"], cfg["sl_mult"] = float(tp_mult), float(sl_mult)
            pred_csv = base_sweep.EXT_PRED_DIR / name / (f"validation_predictions_{cfg['q_tag']}.csv" if oof
                                                           else f"oos_predictions_{cfg['q_tag']}.csv")
            p = base_sweep.prep_component(name, cfg, frame, pred_csv, oof=oof)
            prepped[name] = p

            floor_diag = _floor_bind_rate(p["frame"], cfg, p["dec"])
            m, _ledger = base_sweep.replay_exit_variant(
                p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
                exit_threshold=BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=base_sweep.COST_MULT,
                notional_scaled_sltp=p["notional_scaled_sltp"], device=base_sweep.DEVICE,
            )
            component_rows.append({
                "split": split_name, "candidate": label, "tp_mult": tp_mult, "sl_mult": sl_mult, "component": name,
                **{k: v for k, v in m.items() if k != "exit_reasons"}, "exit_reasons": json.dumps(m["exit_reasons"]),
                **floor_diag,
            })
            log(f"  [{label}] component={name} pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']} "
                f"avg_hold={m['avg_hold_bars']:.1f} tp_floor_bind={floor_diag['tp_floor_bind_rate']*100:.1f}%")

        router_components = {name: helpers._as_router_component(p, exit_threshold=BASELINE_EXIT_THRESHOLD)
                              for name, p in prepped.items()}
        fee0, slip0 = prepped["h48qual"]["fee"], prepped["h48qual"]["slip"]
        _, ledger_combined = router.greedy_replay(frame, router_components, fee=fee0, slip=slip0,
                                                   cost_mult=base_sweep.COST_MULT, device=base_sweep.DEVICE)
        no_gate = helpers._ledger_stats(ledger_combined, frame)
        with_gate = helpers._duration_gated(ledger_combined, frame, router.DURATION_THRESHOLD)
        src_counts = ledger_combined["source_component"].value_counts().to_dict() if len(ledger_combined) else {}
        portfolio_rows.append({"split": split_name, "candidate": label, "tp_mult": tp_mult, "sl_mult": sl_mult,
                                "no_gate": no_gate, "with_gate": with_gate, "source_component_counts": src_counts})
        log(f"  [{label}] PORTFOLIO no_gate pnl={no_gate['pnl']:.2f}% mdd={no_gate['mdd']:.2f}% trades={no_gate['trades']} | "
            f"with_gate pnl={with_gate['pnl']:.2f}% mdd={with_gate['mdd']:.2f}% trades={with_gate['trades']}")

    return {"component_rows": component_rows, "portfolio_rows": portfolio_rows}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    val_frame = base_sweep.load_frame(base_sweep.VAL_START, base_sweep.VAL_END,
                                       base_csv=base_sweep.BASE_2025, wide24_csv=base_sweep.WIDE24_2025)
    val_result = run_split("VAL", val_frame, oof=True, grid=CANDIDATE_GRID)

    val_df = pd.DataFrame(val_result["component_rows"])
    val_df.to_csv(OUT_DIR / "component_val.csv", index=False)
    val_portfolio = pd.DataFrame([
        {"split": r["split"], "candidate": r["candidate"], "tp_mult": r["tp_mult"], "sl_mult": r["sl_mult"],
         "no_gate_pnl": r["no_gate"]["pnl"], "no_gate_mdd": r["no_gate"]["mdd"], "no_gate_trades": r["no_gate"]["trades"],
         "with_gate_pnl": r["with_gate"]["pnl"], "with_gate_mdd": r["with_gate"]["mdd"], "with_gate_trades": r["with_gate"]["trades"]}
        for r in val_result["portfolio_rows"]
    ])
    val_portfolio.to_csv(OUT_DIR / "portfolio_val.csv", index=False)

    # --- G0: self-consistency check -- baseline candidate (index 0) must reproduce the known live
    # baseline exactly (portfolio no_gate PnL +36.82% / MDD -24.34% / 29 trades). If this fails, the
    # harness itself is broken and nothing below should be trusted.
    baseline_row = val_result["portfolio_rows"][0]
    assert abs(baseline_row["tp_mult"] - 12.0) < 1e-9 and abs(baseline_row["sl_mult"] - 6.0) < 1e-9
    g0 = baseline_row["no_gate"]
    g0_ok = (abs(g0["pnl"] - 36.82) < 0.5) and (abs(g0["mdd"] - (-24.34)) < 0.5) and (g0["trades"] == 29)
    log(f"G0 self-consistency check: pnl={g0['pnl']:.2f} (expect 36.82) mdd={g0['mdd']:.2f} (expect -24.34) "
        f"trades={g0['trades']} (expect 29) -> {'PASS' if g0_ok else 'FAIL'}")

    # --- Decision: does any non-baseline candidate beat baseline on BOTH pnl and mdd, on BOTH
    # no_gate and with_gate views? (Same bar every other 08-13 candidate was held to.)
    baseline_no_gate, baseline_with_gate = val_result["portfolio_rows"][0]["no_gate"], val_result["portfolio_rows"][0]["with_gate"]
    qualifiers = []
    for r in val_result["portfolio_rows"][1:]:
        ng, wg = r["no_gate"], r["with_gate"]
        beats = (ng["pnl"] >= baseline_no_gate["pnl"] and ng["mdd"] >= baseline_no_gate["mdd"] and
                 wg["pnl"] >= baseline_with_gate["pnl"] and wg["mdd"] >= baseline_with_gate["mdd"])
        if beats:
            qualifiers.append(r)
    log(f"VAL qualifiers (beat baseline on pnl AND mdd, no_gate AND with_gate): "
        f"{[q['candidate'] for q in qualifiers] if qualifiers else 'NONE'}")

    result: dict[str, Any] = {
        "g0_self_consistency": {"ok": bool(g0_ok), "measured": g0, "expected": {"pnl": 36.82, "mdd": -24.34, "trades": 29}},
        "val_candidate_grid": CANDIDATE_GRID,
        "val_qualifiers": [q["candidate"] for q in qualifiers],
        "oos_run": False,
    }

    if not g0_ok:
        log("ABORTING before OOS: G0 self-consistency check failed, harness does not reproduce the known baseline.")
        (OUT_DIR / "report.json").write_text(json.dumps(result, indent=2, default=str))
        return 1

    if not qualifiers:
        log("No candidate cleared the VAL bar (pnl+mdd non-degraded, no_gate+with_gate). OOS NOT opened -- negative pilot result.")
        (OUT_DIR / "report.json").write_text(json.dumps(result, indent=2, default=str))
        return 0

    # Pick the single best VAL qualifier by no_gate pnl for the one-shot OOS look.
    best = max(qualifiers, key=lambda r: r["no_gate"]["pnl"])
    best_idx = [r["candidate"] for r in val_result["portfolio_rows"]].index(best["candidate"])
    best_tp, best_sl = CANDIDATE_GRID[best_idx]
    log(f"Best VAL qualifier: {best['candidate']} (tp_mult={best_tp}, sl_mult={best_sl}) -- opening SINGLE-TOUCH OOS now.")

    oos_frame = base_sweep.load_frame(base_sweep.OOS_START, base_sweep.OOS_END,
                                       base_csv=base_sweep.BASE_2026, wide24_csv=base_sweep.WIDE24_2026)
    # Re-run ONLY baseline + the one chosen candidate on OOS (not the full grid -- avoids any
    # appearance of OOS-based candidate selection).
    oos_grid = [(12.0, 6.0), (best_tp, best_sl)]
    oos_result = run_split("OOS", oos_frame, oof=False, grid=oos_grid)
    oos_df = pd.DataFrame(oos_result["component_rows"])
    oos_df.to_csv(OUT_DIR / "component_oos.csv", index=False)
    oos_portfolio = pd.DataFrame([
        {"split": r["split"], "candidate": r["candidate"], "tp_mult": r["tp_mult"], "sl_mult": r["sl_mult"],
         "no_gate_pnl": r["no_gate"]["pnl"], "no_gate_mdd": r["no_gate"]["mdd"], "no_gate_trades": r["no_gate"]["trades"],
         "with_gate_pnl": r["with_gate"]["pnl"], "with_gate_mdd": r["with_gate"]["mdd"], "with_gate_trades": r["with_gate"]["trades"]}
        for r in oos_result["portfolio_rows"]
    ])
    oos_portfolio.to_csv(OUT_DIR / "portfolio_oos.csv", index=False)

    oos_baseline, oos_candidate = oos_result["portfolio_rows"][0], oos_result["portfolio_rows"][1]
    oos_survives = (oos_candidate["no_gate"]["pnl"] >= oos_baseline["no_gate"]["pnl"] and
                    oos_candidate["no_gate"]["mdd"] >= oos_baseline["no_gate"]["mdd"])
    result.update({
        "oos_run": True,
        "oos_window": [base_sweep.OOS_START, base_sweep.OOS_END],
        "chosen_candidate": best["candidate"], "chosen_tp_mult": best_tp, "chosen_sl_mult": best_sl,
        "oos_baseline_no_gate": oos_baseline["no_gate"], "oos_candidate_no_gate": oos_candidate["no_gate"],
        "oos_baseline_with_gate": oos_baseline["with_gate"], "oos_candidate_with_gate": oos_candidate["with_gate"],
        "oos_survives_no_gate_pnl_and_mdd": bool(oos_survives),
    })
    log(f"OOS result: baseline pnl={oos_baseline['no_gate']['pnl']:.2f}% mdd={oos_baseline['no_gate']['mdd']:.2f}% | "
        f"candidate pnl={oos_candidate['no_gate']['pnl']:.2f}% mdd={oos_candidate['no_gate']['mdd']:.2f}% -> "
        f"{'SURVIVES' if oos_survives else 'REVERSES'}")

    (OUT_DIR / "report.json").write_text(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
