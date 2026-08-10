"""Re-run the 2026-07-12 3-asset portfolio sweep on CURRENT data.

WHY: ETH live runs `FINAL_GOVERNOR_OMEGA4_6_1_ETH_DURATION_GATE_OFF=True` and
`ETH_NOTIONAL_MULTIPLIER=1.5`. Both flags rest on
docs/model_contracts/portfolio_concurrent_3asset_CURRENT_BASELINE_20260712.md, whose numbers come
from the same pre-revision era as ETH's +145.34 (which no longer reproduces -- actual +82.53
gated / +77.11 no-gate). The two claims being re-checked are:
  1. "disabling the duration gate STRICTLY DOMINATES the gate-on frontier at every cap level
     (higher PnL AND lower MDD simultaneously)"
  2. "1.5x is the safe point below the ~2.0-2.5x where validation PnL inverts sign"

WHY A WRAPPER: replay_portfolio_rl_gate_2action_native_20260708._eth_components aligns frame vs
prediction timestamps in its `validation` branch but NOT in its `oos` branch, which passes raw
inputs to prepare_component's exact-equality check. That worked on 2026-07-12; it fails now purely
because the ETH prediction CSVs were later EXTENDED to 2026-07-12 09:00 (55,405 rows) while the
frame still cuts at 2026-06-30 (51,841 rows). Measured: `in frame not pred = 0`, so the frame is a
strict SUBSET of the predictions -- this is a coverage difference, NOT data drift, and subsetting
the predictions to the frame is exactly what the validation branch already does. No numbers change
as a result of this fix; it only makes the oos branch runnable.

STATUS: verification of already-wired live flags. Not a selection, not a promotion. If a claim
fails to reproduce, the response is a pre-registered re-selection contract, NOT flipping the flag.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import replay_portfolio_rl_gate_2action_native_20260708 as native  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as eth_greedy  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as eth_retest  # noqa: E402
import sweep_portfolio_concurrent_3asset_v4_20260712 as sweep  # noqa: E402

_orig = native._eth_components
ALIGN_DIR = ROOT / "tmp/portfolio_sweep_recheck_20260808/_aligned"


def _eth_components_aligned(split: str, device: torch.device) -> tuple[pd.DataFrame, dict[str, Any], tuple[float, float]]:
    if split == "validation":
        return _orig(split, device)
    ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    frame = eth_retest.load_frame_current("2026-01-01", "2026-06-30")
    components: dict[str, Any] = {}
    for name, cfg in eth_retest.COMPONENTS.items():
        pred = pd.read_csv(eth_greedy.OUT_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv")
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        # same rule as the validation branch: intersect, frame first
        common = frame["timestamp"].isin(pred["timestamp"])
        frame_c = frame[common].reset_index(drop=True) if not common.all() else frame
        pred = pred[pred["timestamp"].isin(frame_c["timestamp"])].reset_index(drop=True)
        if len(pred) != len(frame_c):
            raise RuntimeError(f"oos/{name}: aligned {len(pred)} != frame {len(frame_c)}")
        tmp = ALIGN_DIR / f"_eth_oos_{name}_aligned.csv"
        pred.to_csv(tmp, index=False)
        components[name] = eth_greedy.prepare_component(frame_c, tmp, cfg, device)
        components[name]["sidecar"] = eth_greedy.sidecar
        components[name]["long_scale"] = eth_greedy.SCALE_MAP[f"{name}_L"]
        components[name]["short_scale"] = eth_greedy.SCALE_MAP[f"{name}_S"]
        frame = frame_c
    print(f"[aligned] eth oos rows={len(frame)}", flush=True)
    fee, slip = eth_greedy.omega._load_fee_slip()
    return frame, components, (float(fee), float(slip))


native._eth_components = _eth_components_aligned

if __name__ == "__main__":
    sys.argv = ["sweep", "--only", "all", "--out-dir", "tmp/portfolio_sweep_recheck_20260808"]
    raise SystemExit(sweep.main())
