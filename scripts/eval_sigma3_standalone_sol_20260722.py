#!/usr/bin/env python3
"""Isolate whether the raw Sigma3 (1h HGB trend-scanning classifier) signal has edge on SOL,
separate from the Sigma6 composition (Sigma3 + Sigma5 ATR trailing-stop execution + Regime3
not_chop filter), which already failed OOS for SOL today (VAL +29.1%/OOS -15.9%, a VAL->OOS
collapse; see tmp/causal_regen_20260516/sigma6_regime_trend_sol_20260715/).

Evaluation methodology ported from scripts/replay_sigma3_1h_gates_20260705.py (ETH's original
standalone Sigma3 evaluation) -- fixed ATR-scaled TP/SL barriers + fixed max-hold time-stop, no
regime filter, no trailing stop. Uses replay_omega6_v2_variants_20260704.run_variant/cost_stress
directly against the SOL tape.

Tape: tmp/causal_regen_20260516/sigma3_1h_hgb_sol_20260715/tape_ensemble_sol.parquet
  Date range: 2025-06-25 00:00 .. 2026-07-12 16:00 (verified before this run).

Windows (matching the SOL Sigma-family convention established in
scripts/run_sigma6_regime_trend_sol_20260715.py / eval_sigma6_regime_trend_sol_oos_20260715.py):
  VAL: 2025-07-01 .. 2025-12-31
  OOS: 2026-01-01 .. 2026-03-31 (one-shot; only the frozen winner touches this)
  FRESH: 2026-04-01 .. 2026-07-12 (tape's actual end; requested 07-21 not available in this tape)

Grid (kept simple per task instructions): quality_threshold {0.60, 0.70} x fixed_leverage {3, 4}
x sl_atr_mult {1.5, 2.5}, tp_atr_mult fixed at 2.5, margin 0.30, cooldown 3, max_hold 48 (2 days),
persistence_bars 0, use_fallback=False (primary signal only, matching the ETH original).

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false -- this is a fixed-exit bar-by-bar backtest computed causally
from the precomputed decision tape (which itself was built causally); no ledger/replay reuse.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402

TAPE_PATH = ROOT / "tmp/causal_regen_20260516/sigma3_1h_hgb_sol_20260715/tape_ensemble_sol.parquet"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_standalone_sol_20260722"

VAL_START, VAL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31 23:59:59")
FRESH_START, FRESH_END = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-07-12 23:59:59")

THRESHOLDS = [0.60, 0.70]
LEVERAGES = [3.0, 4.0]
SL_MULTS = [1.5, 2.5]
TP_MULT = 2.5
COOLDOWN = 3
MAX_HOLD = 48
MARGIN = 0.30


def load_tape() -> pd.DataFrame:
    t = pd.read_parquet(TAPE_PATH)
    t["timestamp"] = pd.to_datetime(t["timestamp"])
    return t.sort_values("i").reset_index(drop=True)


def passes_gates_6mo(result: dict) -> bool:
    c1, c3 = result["cost1"], result["cost3"]
    return (
        c1["pnl"] > 0 and c3["pnl"] > 0
        and c1["mdd"] >= -20.0 and c3["mdd"] >= -20.0
        and c1["trades"] >= 20
        and len(c1["trades_by_month"]) >= 3
    )


def run_grid(tape: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    tapes = {thr: v2.apply_quality_threshold(tape, thr) for thr in THRESHOLDS}
    rows = []
    for thr, lev, sl in itertools.product(THRESHOLDS, LEVERAGES, SL_MULTS):
        cfg = v2.VariantConfig(
            name=f"sol_s3_qt{thr}_lev{lev}_sl{sl}",
            tp_mode="atr_scaled", tp_atr_mult=TP_MULT, sl_atr_mult=sl,
            sizing_mode="fixed", fixed_margin=MARGIN, fixed_leverage=lev,
            cooldown_bars=COOLDOWN, quality_threshold=thr, persistence_bars=0,
            max_hold_bars=MAX_HOLD, use_fallback=False,
        )
        r = v2.cost_stress(tapes[thr], cfg, start=start, end=end)
        row = {
            "quality_threshold": thr, "leverage": lev, "sl_mult": sl, "tp_mult": TP_MULT,
            "cost1_pnl": r["cost1"]["pnl"], "cost1_mdd": r["cost1"]["mdd"],
            "cost1_trades": r["cost1"]["trades"], "cost1_wr": r["cost1"]["wr"],
            "cost3_pnl": r["cost3"]["pnl"], "cost3_mdd": r["cost3"]["mdd"],
            "cost3_trades": r["cost3"]["trades"],
            "months": len(r["cost1"]["trades_by_month"]),
            "gate_pass": passes_gates_6mo(r),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return pd.DataFrame(rows)


def run_single(tape: pd.DataFrame, thr: float, lev: float, sl: float, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    tt = v2.apply_quality_threshold(tape, thr)
    cfg = v2.VariantConfig(
        name=f"sol_s3_qt{thr}_lev{lev}_sl{sl}_FROZEN",
        tp_mode="atr_scaled", tp_atr_mult=TP_MULT, sl_atr_mult=sl,
        sizing_mode="fixed", fixed_margin=MARGIN, fixed_leverage=lev,
        cooldown_bars=COOLDOWN, quality_threshold=thr, persistence_bars=0,
        max_hold_bars=MAX_HOLD, use_fallback=False,
    )
    return v2.cost_stress(tt, cfg, start=start, end=end)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tape = load_tape()
    print(f"tape range: {tape['timestamp'].min()} .. {tape['timestamp'].max()}, n={len(tape)}", flush=True)

    print("\n=== VAL grid (2025-07-01..12-31) ===", flush=True)
    grid = run_grid(tape, VAL_START, VAL_END)
    grid.to_csv(OUT_DIR / "val_grid.csv", index=False)
    grid_sorted = grid.sort_values("cost3_pnl", ascending=False)
    print(grid_sorted.to_string(index=False), flush=True)

    passing = grid[grid["gate_pass"]]
    if len(passing):
        winner = passing.sort_values("cost3_pnl", ascending=False).iloc[0]
        verdict_val = "gate_pass"
    else:
        winner = grid_sorted.iloc[0]
        verdict_val = "no_gate_pass_best_by_cost3pnl"
    thr, lev, sl = float(winner["quality_threshold"]), float(winner["leverage"]), float(winner["sl_mult"])
    print(f"\nFrozen winner ({verdict_val}): qt={thr} lev={lev} sl_atr={sl} tp_atr={TP_MULT}", flush=True)

    print("\n=== OOS one-shot touch (2026-01-01..03-31) ===", flush=True)
    oos = run_single(tape, thr, lev, sl, OOS_START, OOS_END)
    print(json.dumps({k: v for k, v in oos["cost1"].items()}, default=str), flush=True)
    print(json.dumps({k: v for k, v in oos["cost3"].items()}, default=str), flush=True)

    print("\n=== FRESH window (2026-04-01..07-12, tape end) ===", flush=True)
    fresh = run_single(tape, thr, lev, sl, FRESH_START, min(FRESH_END, tape["timestamp"].max()))
    print(json.dumps({k: v for k, v in fresh["cost1"].items()}, default=str), flush=True)
    print(json.dumps({k: v for k, v in fresh["cost3"].items()}, default=str), flush=True)

    report = {
        "asset": "SOL",
        "signal": "sigma3_1h_hgb_standalone",
        "methodology": "fixed_atr_tp_sl_time_stop_no_regime_filter_no_trailing_stop",
        "tape_path": str(TAPE_PATH),
        "tape_range": [str(tape["timestamp"].min()), str(tape["timestamp"].max())],
        "grid": {"threshold": THRESHOLDS, "leverage": LEVERAGES, "sl_atr_mult": SL_MULTS, "tp_atr_mult": TP_MULT,
                  "cooldown_bars": COOLDOWN, "max_hold_bars": MAX_HOLD, "margin": MARGIN},
        "val_window": [str(VAL_START), str(VAL_END)],
        "oos_window": [str(OOS_START), str(OOS_END)],
        "fresh_window": [str(FRESH_START), str(min(FRESH_END, tape['timestamp'].max()))],
        "val_verdict": verdict_val,
        "frozen_config": {"quality_threshold": thr, "leverage": lev, "sl_atr_mult": sl, "tp_atr_mult": TP_MULT},
        "oos_result": oos,
        "fresh_result": fresh,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    with open(OUT_DIR / "sigma3_standalone_sol_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved: {OUT_DIR / 'sigma3_standalone_sol_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
