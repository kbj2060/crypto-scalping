#!/usr/bin/env python3
"""RESEARCH ONLY -- round 12 of the ETH Omega4.6.1 exit-head investigation.

Rounds 1-11 (2026-07-21/22, see memory) each varied ONE axis at a time:
  - Round 1: exit_threshold varied, but with the EXISTING head trained at threshold=0.95
    (train_eval_omega4_2_risk_sidecar_20260622.py's label is threshold-independent --
    exit_threshold is purely an eval-time probability gate applied on top of a fixed label).
  - "exit head retrain" follow-up: retrained the head with 7 giveback_min/terminal_window
    label variants, but ALWAYS evaluated at the fixed BASELINE_EXIT_THRESHOLD=0.95 -- so a
    head trained to be more sensitive (lower giveback_min) never got to demonstrate it at an
    operating point where it could actually fire.

This script is the untested combo: evaluate every ALREADY-TRAINED retrain variant (6 label
variants x 2 components = 12 bundles, from research_eth_omega461_exit_head_retrain_eval_20260721.py)
at each of 4 genuinely-lower thresholds {0.70, 0.75, 0.80, 0.85}, jointly, on VAL. This directly
targets the deployment operating point instead of training for 0.95 and testing at a mismatched bar.

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Read-only w.r.t. all existing tmp/research_2026072*/tmp/causal_regen_20260516/*_retrain_20260721_*
artifacts -- no retraining is performed by this script (it reuses the 12 checkpoints already on
disk). Research artifact only -- no promotion-gate claim.
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

def _bundle(suffix: str) -> Path:
    return ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_research_exit_head_retrain_20260721_{suffix}/true_3head_tabm_bundle.pt"


VARIANTS: dict[str, dict[str, Path]] = {
    "h48qual_gb045": {"h48qual": _bundle("h48qual_gb045")},
    "h48qual_gb055": {"h48qual": _bundle("h48qual_gb055")},
    "h48qual_gb065_control": {"h48qual": _bundle("h48qual_control_edge0020")},
    "h48qual_gb075": {"h48qual": _bundle("h48qual_gb075")},
    "h48qual_gb085": {"h48qual": _bundle("h48qual_gb085")},
    "h48qual_tw08": {"h48qual": _bundle("h48qual_tw08")},
    "zig075_gb045": {"zig075": _bundle("zig075_gb045")},
    "zig075_gb055": {"zig075": _bundle("zig075_gb055")},
    "zig075_gb065_control": {"zig075": _bundle("zig075_gb065_control")},
    "zig075_gb075": {"zig075": _bundle("zig075_gb075")},
    "zig075_gb085": {"zig075": _bundle("zig075_gb085")},
    "zig075_tw08": {"zig075": _bundle("zig075_tw08")},
}

THRESH_GRID = [0.70, 0.75, 0.80, 0.85]

OUT_DIR = ROOT / "tmp/research_20260722/joint_threshold_retrain_20260722"


def evaluate_variant(name: str, bundle_overrides: dict[str, Path], *, split: str) -> pd.DataFrame:
    components = {}
    for cname, cfg in base.COMPONENTS.items():
        cfg2 = dict(cfg)
        if cname in bundle_overrides:
            cfg2["bundle"] = bundle_overrides[cname]
        components[cname] = cfg2

    if split == "VAL":
        frame = base.load_frame(base.VAL_START, base.VAL_END, base_csv=base.BASE_2025, wide24_csv=base.WIDE24_2025)
    else:
        frame = base.load_frame(base.OOS_START, base.OOS_END, base_csv=base.BASE_2026, wide24_csv=base.WIDE24_2026)

    rows = []
    for cname, cfg in components.items():
        if cname not in bundle_overrides:
            continue
        pred = base.EXT_PRED_DIR / cname / f"{'validation' if split == 'VAL' else 'oos'}_predictions_{cfg['q_tag']}.csv"
        prepped = base.prep_component(cname, cfg, frame, pred, oof=(split == "VAL"))
        for et in THRESH_GRID:
            m, _ledger = base.replay_exit_variant(
                prepped["frame"], prepped["x"], prepped["dec"], prepped["loaded"],
                risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
                exit_threshold=et, fee=prepped["fee"], slip=prepped["slip"],
                cost_mult=base.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"], device=base.DEVICE,
            )
            rows.append({"variant": name, "component": cname, "split": split, "exit_threshold": et,
                         **m, "exit_reasons": json.dumps(m["exit_reasons"])})
    return pd.DataFrame(rows)


def sanity_check() -> pd.DataFrame:
    """No-op check: variant bundle == frozen live bundle path, threshold=0.95 (BASELINE) must
    reproduce the established live baseline numbers exactly (VAL h48qual PnL +5.45%/MDD -11.62%,
    VAL zig075 PnL +40.31%/MDD -13.07%)."""
    val_frame = base.load_frame(base.VAL_START, base.VAL_END, base_csv=base.BASE_2025, wide24_csv=base.WIDE24_2025)
    rows = []
    for cname, cfg in base.COMPONENTS.items():
        pred = base.EXT_PRED_DIR / cname / f"validation_predictions_{cfg['q_tag']}.csv"
        prepped = base.prep_component(cname, cfg, val_frame, pred, oof=True)
        m, _ledger = base.replay_exit_variant(
            prepped["frame"], prepped["x"], prepped["dec"], prepped["loaded"],
            risk_margin_fraction=prepped["margin"], risk_leverage=prepped["leverage"],
            exit_threshold=base.BASELINE_EXIT_THRESHOLD, fee=prepped["fee"], slip=prepped["slip"],
            cost_mult=base.COST_MULT, notional_scaled_sltp=prepped["notional_scaled_sltp"], device=base.DEVICE,
        )
        rows.append({"component": cname, "exit_threshold": base.BASELINE_EXIT_THRESHOLD, **m,
                     "exit_reasons": json.dumps(m["exit_reasons"])})
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("stage=sanity_check_baseline_reproduce", flush=True)
    san = sanity_check()
    print(san[["component", "exit_threshold", "pnl", "mdd", "trades", "wr"]].to_string(index=False), flush=True)
    san.to_csv(OUT_DIR / "sanity_check_baseline_reproduce.csv", index=False)

    print("stage=val_grid", flush=True)
    all_rows = []
    for name, overrides in VARIANTS.items():
        print(f"stage=evaluate variant={name} split=VAL", flush=True)
        df = evaluate_variant(name, overrides, split="VAL")
        print(df[["variant", "component", "exit_threshold", "pnl", "mdd", "trades", "wr", "avg_hold_bars"]].to_string(index=False), flush=True)
        all_rows.append(df)
    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(OUT_DIR / "joint_threshold_retrain_VAL.csv", index=False)
    print("stage=done_val", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
