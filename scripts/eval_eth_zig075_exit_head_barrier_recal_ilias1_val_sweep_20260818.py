#!/usr/bin/env python3
"""VAL exit_threshold sweep replay of Ilias 1's zig075 barrier-recal exit_head -- directly
comparable to the original (pos_tp/pos_sl-bug-contaminated) attempt's table in
docs/experiments/eth_zig075_exit_head_barrier_recal_20260818.md ("frozen 결과" section).

Reuses research_eth_omega461_exit_sweep_20260721.py's prep_component/replay_exit_variant/
run_grid UNMODIFIED (import only, no reimplementation). exit_sweep.py's own VAL frame loader
(BASE_2025/WIDE24_2025) is already the canonical data source -- confirmed earlier this session
(train_eval_omega4_3head_parent72_eth_canonicaldata_posfix_20260818.py's own comments: WIDE24_
2025/2026 == REGIME3_CURRENT_2025/2026 canonical paths, direct diff) -- so unlike the training
wrappers, no data-source override is needed here.

Only 2 things point at Ilias 1 instead of the original zig075:
  1. sweep.COMPONENTS["zig075"]["bundle"/"sidecar_pkl"/"q_tag"/"quality_threshold"] overridden to
     Ilias 1's barrier-recal bundle (pos_tp/pos_sl bug fixed, canonical data, base_cols pinned to
     the original 102, exit_head retrained with adverse_unreal=-0.02/min_mfe_for_giveback=0.015/
     giveback_min=0.45) + its own newly-trained pinned102 risk sidecar.
  2. The VAL prediction CSV is generated FRESH from that bundle (reusing eval_eth_odyssey4_
     posfix_canonicaldata_freshforward_20260818.generate_predictions, the same proven mechanism
     validated earlier this session against published reference numbers) instead of reading the
     ORIGINAL zig075 bundle's stale EXT_PRED_DIR CSV -- direction/quality heads are frozen and
     byte-identical to the plain pinned102 bundle (only exit_head was retrained), so this is
     exact, not an approximation.

fresh_forward_bar_by_bar=true (replay_exit_variant is a single causal forward pass, i increasing,
only row i + already-closed history used at bar i). No stored trade ledger used as input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818 as ev  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_zig075_exit_head_barrier_recal_20260818_ilias1_encoder"
ILIAS1_ZIG075_BARRIER_RECAL_BUNDLE = OUT_DIR / "true_3head_tabm_bundle.pt"
ILIAS1_ZIG075_SIDECAR = ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_zig075_pinned102_q080_20260818/risk_sidecar.pkl"

# exit_threshold=1.01 disables exit_head entirely (predicted prob is a sigmoid output in [0,1],
# so `prob >= 1.01` never fires) -- this is the "baseline(라이브, exit_head 미사용)" row from the
# original comparison table, not a special code path.
EXIT_GRID = [1.01, 0.99, 0.95, 0.90, 0.80, 0.70]


def main() -> int:
    sweep.COMPONENTS["zig075"] = {
        **sweep.COMPONENTS["zig075"],
        "bundle": ILIAS1_ZIG075_BARRIER_RECAL_BUNDLE,
        "sidecar_pkl": ILIAS1_ZIG075_SIDECAR,
        "q_tag": "q080",
        "quality_threshold": 0.80,
    }
    cfg = sweep.COMPONENTS["zig075"]

    val_frame = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    print(f"VAL frame rows={len(val_frame)} range=[{val_frame['timestamp'].min()}, {val_frame['timestamp'].max()}]", flush=True)

    pred_cfg = {"bundle": cfg["bundle"], "threshold": cfg["quality_threshold"]}
    fresh_pred = ev.generate_predictions("zig075", pred_cfg, val_frame, oof=True)
    pred_csv = OUT_DIR / "validation_predictions_q080_fresh_ilias1_barrier_recal.csv"
    fresh_pred.to_csv(pred_csv, index=False)
    print(f"fresh VAL predictions written: {pred_csv} rows={len(fresh_pred)}", flush=True)

    prepped = {"zig075": sweep.prep_component("zig075", cfg, val_frame, pred_csv, oof=True)}

    print(f"stage=exit_threshold_sweep grid={EXIT_GRID}", flush=True)
    result = sweep.run_grid(prepped, exit_thresholds=EXIT_GRID)
    result["exit_head_fired"] = result["exit_reasons"].apply(lambda s: json.loads(s).get("exit_head", 0))
    result["baseline_no_exit_head"] = result["exit_threshold"] >= 1.0
    out_csv = OUT_DIR / "val_exit_threshold_sweep_ilias1_barrier_recal.csv"
    result.to_csv(out_csv, index=False)
    print(result[["exit_threshold", "pnl", "mdd", "trades", "wr", "exit_head_fired"]].to_string(index=False), flush=True)
    print(f"report={out_csv}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
