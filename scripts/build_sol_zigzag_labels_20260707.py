"""Phase A pilot (SOL), step 5: build zigzag_action labels for SOL using the IDENTICAL builder
(build_zigzag_action_labels_no_max_horizon_conservative_20260620.py) and default params already
used for ETH/zig075, applied to SOL's regime3-merged feature frames (2024/2025/2026)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_zigzag_action_labels_no_max_horizon_conservative_20260620 as zz  # noqa: E402

PARAMS = dict(atr_window=48, tp_atr_mult=1.05, sl_atr_mult=0.80, tp_min=0.0045, tp_max=0.012,
              sl_min=0.0038, sl_max=0.009, min_utility=0.0009, time_penalty=0.000015,
              adverse_penalty=0.25, transition_buffer=2)

SPLITS = ROOT / "data/splits/year_oos"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sol_zigzag_action_labels_20260707"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in (2024, 2025, 2026):
        path = SPLITS / f"sol_features_{year}.csv"
        frame = pd.read_csv(path, low_memory=False)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        labels = zz.build_no_max_horizon_labels(frame, **PARAMS)
        out = OUT_DIR / f"sol_zigzag_action_labels_{year}.csv"
        labels.to_csv(out, index=False)
        counts = labels["zigzag_action"].value_counts().sort_index().to_dict()
        print(f"{year}: {len(labels)} rows, label_counts={counts} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
