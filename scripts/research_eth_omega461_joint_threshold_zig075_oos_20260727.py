#!/usr/bin/env python3
"""RESEARCH ONLY -- the missing zig075 OOS confirmation for round 12
(research_eth_omega461_joint_threshold_retrain_20260722.py).

Round 12 evaluated 12 retrained exit-head bundles x 4 lowered thresholds on VAL, then spent its
single OOS touch on the h48qual VAL-best config (gb075/gb085 @0.70), which collapsed. The zig075
VAL winners were never OOS-confirmed. This script closes exactly that gap and nothing else.

zig075 VAL baseline (frozen bundle @ EXIT_THRESHOLD=0.95): PnL +40.311%, MDD -13.0657%, 29 trades.
VAL winners carried forward (beat baseline on PnL, MDD tied to ~1e-13):
  - zig075_tw08  @0.75 -> VAL PnL +56.776%, MDD -13.0657%, 37 trades
  - zig075_gb055 @0.75 -> VAL PnL +45.003%, MDD -13.0657%, 31 trades

ONLY these two (variant, threshold) pairs are scored on OOS -- no threshold grid is re-swept on
OOS, no other variant is touched, so this is a two-point confirmation of a VAL-completed
selection, not an OOS search.

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
No retraining is performed (reuses the 20260721 checkpoints already on disk). Research artifact
only -- no promotion-gate claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as base  # noqa: E402
import research_eth_omega461_joint_threshold_retrain_20260722 as r12  # noqa: E402

# (variant label, threshold) pairs selected on VAL by round 12.
CONFIRM = [
    ("zig075_tw08", 0.75),
    ("zig075_gb055", 0.75),
]

OUT_DIR = ROOT / "tmp/research_20260727/joint_threshold_zig075_oos_20260727"


def score(name: str, threshold: float, *, split: str) -> dict:
    """Single (variant, threshold, split) replay. Mirrors r12.evaluate_variant but scores one
    threshold instead of the grid, so the OOS touch stays limited to the VAL-selected point."""
    overrides = r12.VARIANTS[name]
    cname = "zig075"
    cfg = dict(base.COMPONENTS[cname])
    cfg["bundle"] = overrides[cname]

    if split == "VAL":
        frame = base.load_frame(base.VAL_START, base.VAL_END, base_csv=base.BASE_2025, wide24_csv=base.WIDE24_2025)
    else:
        frame = base.load_frame(base.OOS_START, base.OOS_END, base_csv=base.BASE_2026, wide24_csv=base.WIDE24_2026)
    pred = base.EXT_PRED_DIR / cname / f"{'validation' if split == 'VAL' else 'oos'}_predictions_{cfg['q_tag']}.csv"
    prepped = base.prep_component(cname, cfg, frame, pred, oof=(split == "VAL"))
    m, _ledger = base.replay_exit_variant(
        prepped["frame"], prepped["x"], prepped["dec"], prepped["loaded"],
        risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
        exit_threshold=threshold, fee=prepped["fee"], slip=prepped["slip"],
        cost_mult=base.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"], device=base.DEVICE,
    )
    return {"variant": name, "component": cname, "split": split, "exit_threshold": threshold,
            **m, "exit_reasons": json.dumps(m["exit_reasons"])}


def baseline_control(split: str) -> dict:
    """Frozen live bundle at EXIT_THRESHOLD=0.95 -- the control every confirmation number is
    compared against. Run on the same split in the same process so the comparison can't drift."""
    cname = "zig075"
    cfg = dict(base.COMPONENTS[cname])
    if split == "VAL":
        frame = base.load_frame(base.VAL_START, base.VAL_END, base_csv=base.BASE_2025, wide24_csv=base.WIDE24_2025)
    else:
        frame = base.load_frame(base.OOS_START, base.OOS_END, base_csv=base.BASE_2026, wide24_csv=base.WIDE24_2026)
    pred = base.EXT_PRED_DIR / cname / f"{'validation' if split == 'VAL' else 'oos'}_predictions_{cfg['q_tag']}.csv"
    prepped = base.prep_component(cname, cfg, frame, pred, oof=(split == "VAL"))
    m, _ledger = base.replay_exit_variant(
        prepped["frame"], prepped["x"], prepped["dec"], prepped["loaded"],
        risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
        exit_threshold=base.BASELINE_EXIT_THRESHOLD, fee=prepped["fee"], slip=prepped["slip"],
        cost_mult=base.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"], device=base.DEVICE,
    )
    return {"variant": "baseline_frozen", "component": cname, "split": split,
            "exit_threshold": base.BASELINE_EXIT_THRESHOLD, **m, "exit_reasons": json.dumps(m["exit_reasons"])}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["variant", "split", "exit_threshold", "pnl", "mdd", "trades", "wr", "avg_hold_bars", "exit_reasons"]

    # 1) Sanity: reproduce the VAL baseline and the two VAL winners before spending the OOS touch.
    print("stage=sanity_val_reproduce", flush=True)
    san = [baseline_control("VAL")] + [score(n, t, split="VAL") for n, t in CONFIRM]
    san_df = pd.DataFrame(san)
    print(san_df[cols].to_string(index=False), flush=True)
    san_df.to_csv(OUT_DIR / "sanity_val_reproduce.csv", index=False)

    # 2) The confirmation itself.
    print("stage=oos_confirm", flush=True)
    oos = [baseline_control("OOS")] + [score(n, t, split="OOS") for n, t in CONFIRM]
    oos_df = pd.DataFrame(oos)
    print(oos_df[cols].to_string(index=False), flush=True)
    oos_df.to_csv(OUT_DIR / "zig075_oos_confirm.csv", index=False)

    ctrl = oos[0]
    verdict = []
    for row in oos[1:]:
        verdict.append({
            "variant": row["variant"], "exit_threshold": row["exit_threshold"],
            "oos_pnl": row["pnl"], "oos_mdd": row["mdd"], "oos_trades": row["trades"],
            "ctrl_pnl": ctrl["pnl"], "ctrl_mdd": ctrl["mdd"], "ctrl_trades": ctrl["trades"],
            "pnl_beats": bool(row["pnl"] > ctrl["pnl"]),
            "mdd_not_worse": bool(row["mdd"] >= ctrl["mdd"] - 1e-9),
            "confirmed": bool(row["pnl"] > ctrl["pnl"] and row["mdd"] >= ctrl["mdd"] - 1e-9),
        })
    v_df = pd.DataFrame(verdict)
    print(v_df.to_string(index=False), flush=True)
    v_df.to_csv(OUT_DIR / "zig075_oos_verdict.csv", index=False)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
