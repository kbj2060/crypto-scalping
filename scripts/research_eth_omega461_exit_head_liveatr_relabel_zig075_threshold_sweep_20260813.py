#!/usr/bin/env python3
"""RESEARCH ONLY -- EXIT_THRESHOLD grid sweep for zig075's liveatr-relabel exit_head bundle.

Follow-up to `research_eth_omega461_exit_head_liveatr_relabel_20260813.py`
(see `docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md`): that recipe
IMPROVED h48qual's exit head (adopted) but REGRESSED zig075's (VAL PnL +40.31% -> +0.70%,
trades 29 -> 65, WR 48.3% -> 29.2%, MDD -13.07% -> -19.91%) at the shared, never-recalibrated
`EXIT_THRESHOLD=0.95`. A byte-identical-recipe audit ruled out ATR/feature/data mismatches
between the two components, leaving EXIT_THRESHOLD as the one remaining un-recalibrated
variable specific to zig075's retrained exit head. This script does NOT retrain anything -- it
reuses the already-certified `research_eth_omega461_exit_sweep_20260721.py` grid-sweep
mechanism (`prep_component` / `replay_exit_variant` / `run_grid`, all imported unchanged) against
zig075's already-trained liveatr-relabel bundle, sweeping only EXIT_THRESHOLD, all other
weights/config held fixed at `COMPONENTS["zig075"]` (q075, ATR-safety SLTP unchanged).

Bundle used (pulled read-only from the server via `handoff.sh pull`, no retraining):
`tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/zig075/true_3head_tabm_bundle.pt`
(model_id=eth_omega461_exit_head_liveatr_relabel_20260813, experts=bull/bear/chop, verified via
torch.load before this script was written).

VAL window: 2025-10-01..2025-12-31 -- inherited from `research_eth_omega461_exit_sweep_20260721`'s
own VAL_START/VAL_END (that script's docstring flags this is one month short of the canonical
CLAUDE.md 2025-09-01 start because the frozen OOF prediction CSVs only exist from 2025-10-01
onward; not re-derived or silently changed here).

fresh_forward_bar_by_bar=true (VAL replay is `replay_exit_variant`'s single causal forward pass,
unchanged). trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false. direction_head/quality_head/ATR-safety-SLTP unchanged --
only EXIT_THRESHOLD varies. No training. No server access (bundle already local).

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

NEW_BUNDLE = ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/zig075/true_3head_tabm_bundle.pt"
EXIT_GRID = [0.999, 0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.70]
OUT_DIR = ROOT / "tmp/research_20260813_zig075_liveatr_exit_threshold_sweep"


def main() -> int:
    if not NEW_BUNDLE.exists():
        raise FileNotFoundError(f"zig075 liveatr-relabel bundle not found: {NEW_BUNDLE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"stage=load_val_frame window=[{sweep.VAL_START},{sweep.VAL_END}]", flush=True)
    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    print(f"  rows={len(val_frame)} range=[{val_frame['timestamp'].min()}, {val_frame['timestamp'].max()}]", flush=True)

    cfg = dict(sweep.COMPONENTS["zig075"])
    val_pred = sweep.EXT_PRED_DIR / "zig075" / f"validation_predictions_{cfg['q_tag']}.csv"

    print("stage=prep_baseline (original/currently-deployed zig075 bundle)", flush=True)
    baseline_prepped = sweep.prep_component("zig075", cfg, val_frame, val_pred, oof=True)

    print("stage=prep_liveatr_relabel (new bundle, same config otherwise)", flush=True)
    cfg_new = dict(cfg)
    cfg_new["bundle"] = NEW_BUNDLE
    new_prepped = sweep.prep_component("zig075", cfg_new, val_frame, val_pred, oof=True)

    prepped = {
        "zig075_baseline_original": baseline_prepped,
        "zig075_liveatr_relabel": new_prepped,
    }
    print(f"stage=run_grid exit_thresholds={EXIT_GRID}", flush=True)
    val_grid = sweep.run_grid(prepped, exit_thresholds=EXIT_GRID)
    val_grid.to_csv(OUT_DIR / "val_exit_threshold_grid.csv", index=False)
    print(val_grid[["component", "exit_threshold", "pnl", "mdd", "trades", "wr", "avg_hold_bars", "exit_reasons"]].to_string(index=False), flush=True)
    print(f"stage=done out={OUT_DIR / 'val_exit_threshold_grid.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
