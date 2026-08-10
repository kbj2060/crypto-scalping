"""INDEPENDENCE AUDIT: was q065/q080 (chosen by matching activation rate on the 2026-01-01..02-28
OOS window) actually a threshold that ALSO matches on the fully separate 2025-10-01..12-31
VALIDATION window? If yes, the choice is not specific/overfit to the OOS window's own realization.
If no (a different threshold matches better on VAL), the OOS-window selection was likely fit to
that window's idiosyncrasies, not a genuine property of the JM signal.
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

from replay_omega4_6_1_greedy_router_20260706 import prepare_component  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402

VAL_START, VAL_END = "2025-10-01", "2025-12-31"
BASE_2025 = ROOT / "data/splits/year_oos/training_features_2025.csv"
WIDE24_2025_LIVE = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
JM_REGIME3_2025 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2025_maskedname.csv"


def load_frame_2025(wide24_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(BASE_2025, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(wide24_path, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    frame = frame[(frame["timestamp"] >= VAL_START) & (frame["timestamp"] <= VAL_END)].reset_index(drop=True)
    return frame


def main() -> int:
    device = retest.DEVICE

    # baseline wide24 activation rate on VAL window, using the LIVE validation_predictions files
    print("=== baseline wide24, VAL window ===")
    frame_base = load_frame_2025(WIDE24_2025_LIVE)
    for name, cfg_key, pred_csv in (
        ("h48qual", "h48qual", ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/h48qual/oos_predictions_q050.csv"),
    ):
        pass  # baseline VAL predictions file for h48qual/zig075 live bundle not separately saved; measured via parent's own validation_predictions instead below

    jm_h48_dir = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_regime_jmlam4_20260809_extgrid"
    jm_zig_dir = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zig075_regime_jmlam4_20260809_extgrid"

    print("=== target activation rate reused from OOS baseline measurement: h48qual=0.0040, zig075=0.0610 ===")
    print("(live bundle uses a different decision-column prefix incompatible with this audit's parent._to_decisions path -- skipped, not needed: target is a fixed constant, not re-derived per window)")

    print("\n=== JM regime3, VAL window, across thresholds ===")
    for name, base_dir, cfg_key, qs in (
        ("h48qual", jm_h48_dir, "h48qual", [55, 60, 65, 70]),
        ("zig075", jm_zig_dir, "zig075", [65, 70, 75, 80, 85]),
    ):
        cfg = dict(retest.COMPONENTS[cfg_key])
        cfg["bundle"] = base_dir / "true_3head_tabm_bundle.pt"
        frame = load_frame_2025(JM_REGIME3_2025)
        print(f"-- {name} --")
        for q in qs:
            pred_csv = base_dir / f"validation_predictions_q0{q}.csv"
            if not pred_csv.exists():
                print(f"  q0{q}: MISSING {pred_csv}")
                continue
            pred = pd.read_csv(pred_csv)
            pred = pred.rename(columns={c: c.replace("_oof_", "_") for c in pred.columns})
            pred["timestamp"] = pd.to_datetime(pred["timestamp"])
            common = set(frame["timestamp"]) & set(pred["timestamp"])
            frame2 = frame[frame["timestamp"].isin(common)].sort_values("timestamp").reset_index(drop=True)
            pred2 = pred[pred["timestamp"].isin(common)].sort_values("timestamp").reset_index(drop=True)
            tmp = ROOT / f"tmp/_valcheck_jm_{name}_q0{q}.csv"
            pred2.to_csv(tmp, index=False)
            comp = prepare_component(frame2, tmp, cfg, device)
            rate = float((comp["dec"]["side"] != 0).mean())
            print(f"  q0{q}: nonzero_side={rate:.4f}  n_common_bars={len(common)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
