#!/usr/bin/env python3
"""RESEARCH ONLY -- follow-up to docs/experiments/eth_omega4_6_1_atr_tpsl_floor_binding_investigation_20260812.md
and docs/experiments/eth_omega461_atr_tpsl_recalibration_pilot_20260813.md.

The 08-12 investigation found the live "ATR-adaptive" TP/SL (min_tp=0.075/min_sl=0.040,
tp_mult=12/sl_mult=6) is effectively a FIXED 7.5%/4.0% target -- the floor binds 95-98.5% of the
time on ETH 5m. The 08-13 pilot tried recalibrating by RAISING tp_mult/sl_mult (ratio fixed at the
live 2:1) so the ATR term would actually engage more often; all 3 candidates ((16,8),(22,11),(28,14))
were flat-or-worse than baseline on VAL (portfolio no_gate PnL flipped sign to -7.25% at (16,8) vs
+36.82% baseline) -- that pilot's own "next steps" section flagged two axes as explicitly untried:
(a) changing the floor/cap ABSOLUTE values independently (not ratio-preserving), and (b) per-
COMPONENT recalibration (h48qual/zig075 have independently trained heads). This script does (a)+(b)
together in their smallest useful form: LOWER the floor alone (tp_mult/sl_mult, caps untouched,
per CLAUDE.md scope note -- caps never bind, 0.00% in all 2025-2026 data, not in scope) for ONE
component at a time, leaving the other component at the exact live floor.

Direction chosen: NARROWING (lower floor), not widening. The 08-13 pilot only tested widening (via
raised mult, which pushes the ALWAYS-BOUND value up since floor binds ~98% of the time) and found a
monotonic degradation as width grew (29->28->22->17 trades, longer holds, worse pnl/mdd at every
step). Since floor binds almost always, the floor value IS effectively the trade width; testing the
opposite direction (narrower fixed width) is the one untried direction with any real information
content -- widening again would only replicate the already-decisive 08-13 result.

Candidate floors are anchored to the VAL-window (2025-10-01..2025-12-31) atr_pct(window=192)
percentile crossing points of the UNCHANGED baseline tp_mult=12/sl_mult=6 formula (atr_pct*mult),
not arbitrary round numbers -- computed directly from data (n=26,209 bars, matches the 08-13 pilot's
independently-measured p50/p90/p99/max exactly):
  p25=0.2101%  p50=0.2696%  p75=0.3486%  p90=0.4256%  p99=0.6685%  max=0.9468%
At baseline mult, floor(7.5%/4.0%) crosses raw(=atr_pct*mult) between p98-p99 (raw_tp@p98=7.11%,
raw_tp@p99=8.02%) -- i.e. floor binds ~98% of the time, matching the 08-12 investigation's measured
95-98.5%. Candidates below move that crossing point down to p75/p50/p25 respectively, i.e. lower the
floor to sit exactly at raw(atr_pct*mult) evaluated at that percentile, keeping tp:sl floor ratio
at the same 2:1 as tp_mult:sl_mult (mult itself unchanged at 12/6 -- ONLY min_tp/min_sl move):
  C1 (p75-cross): min_tp=0.0418 (4.18%), min_sl=0.0209 (2.09%)  -- floor binds ~75% of the time
  C2 (p50-cross): min_tp=0.0324 (3.24%), min_sl=0.0162 (1.62%)  -- floor binds ~50% of the time
  C3 (p25-cross): min_tp=0.0252 (2.52%), min_sl=0.0126 (1.26%)  -- floor binds ~25% of the time
3 candidates x 2 components (each tested ALONE, other component at exact live floor) = 6 VAL cells,
deliberately small given researcher-d.o.f. risk noted for this exploratory thread and the adjacent
ratio-preserving lever's decisive VAL rejection.

Reuses the SAME frozen h48qual/zig075 parent bundles + risk sidecars + already-generated OOF/held-
out prediction CSVs as every sibling 2026-08-13 SLTP experiment (no retraining -- min_tp/min_sl are
runtime execution constants, not learned weights, so this is a pure deterministic backtest replay;
no seed-diversity dimension applies here, same as the 08-13 pilot). Does NOT touch
trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env, or any live
deployed threshold/bundle.

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false. VAL window = 2025-10-01..2025-12-31 (same as every sibling SLTP
experiment in this sub-project). OOS window = 2026-01-01..2026-03-31, single touch, opened only if
a VAL candidate clears the bar below.
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

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_atr_tpsl_floor_independent_percomponent_20260815"

# (label, component_to_change, min_tp, min_sl). component_to_change is None for the baseline row
# (both components at live floor). For non-baseline rows, ONLY component_to_change's min_tp/min_sl
# are overridden; the other component stays at its exact live floor (0.075/0.040). tp_mult/sl_mult
# and max_tp/max_sl are untouched everywhere (caps never bind -- out of scope per task).
CANDIDATE_GRID: list[tuple[str, str | None, float, float]] = [
    ("baseline", None, 0.075, 0.040),
    ("h48qual_p75", "h48qual", 0.0418, 0.0209),
    ("h48qual_p50", "h48qual", 0.0324, 0.0162),
    ("h48qual_p25", "h48qual", 0.0252, 0.0126),
    ("zig075_p75", "zig075", 0.0418, 0.0209),
    ("zig075_p50", "zig075", 0.0324, 0.0162),
    ("zig075_p25", "zig075", 0.0252, 0.0126),
]

BASELINE_EXIT_THRESHOLD = 0.95  # unchanged -- isolate the floor axis only.


def log(msg: str) -> None:
    print(f"[atr_floor_pc] {msg}", flush=True)


def _floor_bind_rate(frame: pd.DataFrame, cfg: dict, dec: pd.DataFrame) -> dict[str, float]:
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


def run_split(split_name: str, frame: pd.DataFrame, *, oof: bool,
              grid: list[tuple[str, str | None, float, float]]) -> dict[str, Any]:
    log(f"=== split={split_name} rows={len(frame)} range=[{frame['timestamp'].min()}, {frame['timestamp'].max()}] ===")
    component_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []

    for label, changed_component, min_tp, min_sl in grid:
        prepped: dict[str, dict[str, Any]] = {}
        for name, base_cfg in base_sweep.COMPONENTS.items():
            cfg = dict(base_cfg)
            if name == changed_component:
                cfg["min_tp"], cfg["min_sl"] = float(min_tp), float(min_sl)
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
                "split": split_name, "candidate": label, "changed_component": changed_component or "none",
                "component": name, "min_tp": cfg["min_tp"], "min_sl": cfg["min_sl"],
                **{k: v for k, v in m.items() if k != "exit_reasons"}, "exit_reasons": json.dumps(m["exit_reasons"]),
                **floor_diag,
            })
            log(f"  [{label}] component={name} min_tp={cfg['min_tp']:.4f} min_sl={cfg['min_sl']:.4f} "
                f"pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']} avg_hold={m['avg_hold_bars']:.1f} "
                f"tp_floor_bind={floor_diag['tp_floor_bind_rate']*100:.1f}%")

        router_components = {name: helpers._as_router_component(p, exit_threshold=BASELINE_EXIT_THRESHOLD)
                              for name, p in prepped.items()}
        fee0, slip0 = prepped["h48qual"]["fee"], prepped["h48qual"]["slip"]
        _, ledger_combined = router.greedy_replay(frame, router_components, fee=fee0, slip=slip0,
                                                   cost_mult=base_sweep.COST_MULT, device=base_sweep.DEVICE)
        no_gate = helpers._ledger_stats(ledger_combined, frame)
        with_gate = helpers._duration_gated(ledger_combined, frame, router.DURATION_THRESHOLD)
        src_counts = ledger_combined["source_component"].value_counts().to_dict() if len(ledger_combined) else {}
        portfolio_rows.append({"split": split_name, "candidate": label, "changed_component": changed_component or "none",
                                "min_tp": min_tp, "min_sl": min_sl, "no_gate": no_gate, "with_gate": with_gate,
                                "source_component_counts": src_counts})
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
        {"split": r["split"], "candidate": r["candidate"], "changed_component": r["changed_component"],
         "min_tp": r["min_tp"], "min_sl": r["min_sl"],
         "no_gate_pnl": r["no_gate"]["pnl"], "no_gate_mdd": r["no_gate"]["mdd"], "no_gate_trades": r["no_gate"]["trades"],
         "with_gate_pnl": r["with_gate"]["pnl"], "with_gate_mdd": r["with_gate"]["mdd"], "with_gate_trades": r["with_gate"]["trades"]}
        for r in val_result["portfolio_rows"]
    ])
    val_portfolio.to_csv(OUT_DIR / "portfolio_val.csv", index=False)

    # --- G0: self-consistency check -- baseline row must reproduce the known live baseline exactly
    # (portfolio no_gate PnL +36.82% / MDD -24.34% / 29 trades).
    baseline_row = val_result["portfolio_rows"][0]
    assert baseline_row["candidate"] == "baseline"
    g0 = baseline_row["no_gate"]
    g0_ok = (abs(g0["pnl"] - 36.82) < 0.5) and (abs(g0["mdd"] - (-24.34)) < 0.5) and (g0["trades"] == 29)
    log(f"G0 self-consistency check: pnl={g0['pnl']:.2f} (expect 36.82) mdd={g0['mdd']:.2f} (expect -24.34) "
        f"trades={g0['trades']} (expect 29) -> {'PASS' if g0_ok else 'FAIL'}")

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

    best = max(qualifiers, key=lambda r: r["no_gate"]["pnl"])
    best_idx = [r["candidate"] for r in val_result["portfolio_rows"]].index(best["candidate"])
    best_label, best_component, best_min_tp, best_min_sl = CANDIDATE_GRID[best_idx]
    log(f"Best VAL qualifier: {best_label} (component={best_component}, min_tp={best_min_tp}, min_sl={best_min_sl}) "
        f"-- opening SINGLE-TOUCH OOS now.")

    oos_frame = base_sweep.load_frame(base_sweep.OOS_START, base_sweep.OOS_END,
                                       base_csv=base_sweep.BASE_2026, wide24_csv=base_sweep.WIDE24_2026)
    # Known pre-existing WIDE24_2026 coverage gap (95 rows / 0.37%, 2026-02-28 16:05..23:55) leaves
    # Regime3 route probabilities non-finite for those bars -- hard._route_id raises on them.
    # Same fix as the precedent in research_eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_
    # 20260813.py: drop those bars before replay (causally faithful -- a live system cannot route a
    # bar it has no regime probabilities for either). Unrelated to this experiment's floor change.
    route_finite = np.isfinite(oos_frame[base_sweep.hard.ROUTE_COLS].to_numpy(dtype=np.float64)).all(axis=1)
    n_route_bad = int((~route_finite).sum())
    if n_route_bad:
        oos_frame = oos_frame[route_finite].reset_index(drop=True)
        log(f"dropped {n_route_bad} OOS bars with non-finite Regime3 route probabilities (WIDE24_2026 coverage gap)")
    oos_grid = [("baseline", None, 0.075, 0.040), (best_label, best_component, best_min_tp, best_min_sl)]
    oos_result = run_split("OOS", oos_frame, oof=False, grid=oos_grid)
    oos_df = pd.DataFrame(oos_result["component_rows"])
    oos_df.to_csv(OUT_DIR / "component_oos.csv", index=False)
    oos_portfolio = pd.DataFrame([
        {"split": r["split"], "candidate": r["candidate"], "changed_component": r["changed_component"],
         "min_tp": r["min_tp"], "min_sl": r["min_sl"],
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
        "chosen_candidate": best_label, "chosen_component": best_component,
        "chosen_min_tp": best_min_tp, "chosen_min_sl": best_min_sl,
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
