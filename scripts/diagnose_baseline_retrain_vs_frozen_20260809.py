"""Diagnose the 6-month baseline divergence: is it retrain instability (expected, well-documented in
this project) or a genuine regenerated-label problem specific to March-June? Runs the TRUE FROZEN
live predictions (no retrain at all, bundle = actual live weights) through the exact same greedy
router methodology, for both the Jan-Feb window (compare to the already-trusted retrained-baseline
number) and the full Jan-June window (compare to the suspect retrained-baseline number). If the
frozen-vs-retrained gap is already large in Jan-Feb too, that's retrain noise, not a March-June-
specific label problem.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import (  # noqa: E402
    DURATION_THRESHOLD, greedy_replay, prepare_component,
)

OUT_DIR = ROOT / "tmp/diagnose_baseline_retrain_vs_frozen_20260809"
FROZEN_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"


def curve_metrics(returns: np.ndarray) -> dict:
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {"pnl": round(float((curve[-1] - 1.0) * 100.0), 4),
            "mdd": round(float(dd.min() * 100.0), 4),
            "trades": int(len(returns)),
            "wr": round(float((returns > 0).mean()), 4) if len(returns) else 0.0}


def run_frozen(tag: str, start: str, end: str) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = retest.DEVICE
    frame = retest.load_frame_current(start, end)
    fee, slip = omega._load_fee_slip()

    components = {}
    for name, tag_q in (("h48qual", "q050"), ("zig075", "q075")):
        pred_csv = FROZEN_DIR / name / f"oos_predictions_{tag_q}.csv"
        pred = pd.read_csv(pred_csv)
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        common = set(frame["timestamp"]) & set(pred["timestamp"])
        frame_i = frame[frame["timestamp"].isin(common)].sort_values("timestamp").reset_index(drop=True)
        pred_i = pred[pred["timestamp"].isin(common)].sort_values("timestamp").reset_index(drop=True)
        tmp = OUT_DIR / f"_aligned_{tag}_{name}.csv"
        pred_i.to_csv(tmp, index=False)
        components[name] = prepare_component(frame_i, tmp, retest.COMPONENTS[name], device)
        frame = frame_i  # shrink to common set progressively (both components must align to same frame)

    print(f"[{tag}] final rows={len(frame)}", flush=True)
    _, ledger = greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    returns = ledger["trade_return"].to_numpy(dtype=float)
    no_gate = curve_metrics(returns)
    ledger.to_csv(OUT_DIR / f"ledger_frozen_{tag}.csv", index=False)
    return {"no_gate": no_gate, "source_component_counts": ledger["source_component"].value_counts().to_dict()}


def main() -> int:
    out = {}
    out["frozen_jan_feb"] = run_frozen("janfeb", "2026-01-01", "2026-02-28")
    out["frozen_jan_june"] = run_frozen("janjune", "2026-01-01", "2026-06-29")
    (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
