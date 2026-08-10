"""For each already-trained quality_threshold (0.40-0.60) per JM component, compute nonzero_side
activation rate via the REAL prepare_component pipeline (ATR safety, sidecar risk model, etc.), to
find which threshold's activation rate best matches the ORIGINAL wide24 rate (h48qual 0.4%,
zig075 6.1%) -- testing the hypothesis that matching activation frequency (not VAL PnL) is the
right selection criterion, since the router-level failure was traced to JM causing WAY more frequent
firing than wide24 at the VAL-PnL-selected thresholds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import prepare_component  # noqa: E402

START, END = "2026-01-01", "2026-02-28"
JM_H48QUAL_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_regime_jmlam4_20260809"
JM_ZIG075_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zig075_regime_jmlam4_20260809"
JM_REGIME3_2026 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2026_maskedname.csv"
OUT_DIR = ROOT / "tmp/eth_jmlam4_activation_scan_20260809"

TARGET = {"h48qual": 0.004, "zig075": 0.061}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = retest.DEVICE
    retest.WIDE24_2026 = JM_REGIME3_2026

    frame_all = retest.load_frame_current(START, END)

    for name, base_dir, cfg_key, thresholds in (
        ("h48qual", JM_H48QUAL_DIR, "h48qual", [40, 45, 50, 55, 60]),
        ("zig075", JM_ZIG075_DIR, "zig075", [40, 45, 50, 55, 60]),
    ):
        cfg = dict(retest.COMPONENTS[cfg_key])
        cfg["bundle"] = base_dir / "true_3head_tabm_bundle.pt"
        # sidecar_pkl is threshold-specific only via precomputed predictions, not the bundle itself;
        # reuse the VAL-tuned sidecar since risk mapping doesn't change entry side/quality decisions
        print(f"== {name} (target activation rate {TARGET[name]:.4f}) ==")
        for q in thresholds:
            pred_csv = base_dir / f"oos_predictions_q0{q}.csv"
            pred = pd.read_csv(pred_csv)
            pred["timestamp"] = pd.to_datetime(pred["timestamp"])
            common = set(frame_all["timestamp"]) & set(pred["timestamp"])
            frame = frame_all[frame_all["timestamp"].isin(common)].sort_values("timestamp").reset_index(drop=True)
            pred = pred[pred["timestamp"].isin(common)].sort_values("timestamp").reset_index(drop=True)
            tmp = OUT_DIR / f"_scan_{name}_q0{q}.csv"
            pred.to_csv(tmp, index=False)
            cfg_i = dict(cfg)
            cfg_i["sidecar_pkl"] = retest.COMPONENTS[cfg_key]["sidecar_pkl"]
            comp = prepare_component(frame, tmp, cfg_i, device)
            rate = float((comp["dec"]["side"] != 0).mean())
            print(f"  q0{q}: nonzero_side={rate:.4f}  (delta from target: {rate - TARGET[name]:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
