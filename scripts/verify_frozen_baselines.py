#!/usr/bin/env python3
"""P0-3 of docs/pipeline_integrity_and_research_redesign_20260730.md: periodic frozen-baseline
reproduction check.

Re-runs each registered "frozen" backtest against CURRENT data/code and compares the result to its
recorded reference numbers within tolerance. A frozen baseline that no longer reproduces is exactly
what happened to the Omega4.6.1 07-06 greedy-router baseline (+145.34%/-10.13%/24trades ->
+82.53%/-15.48%/31trades, root cause: upstream Binance metrics zips retroactively revised -- see
project memory project-omega461-baseline-drift-bisection-20260730) and to the Sigma6 1h tape
(project-selection-stats-instrument-20260726). Both were previously only discovered by manual,
one-off investigation; this script exists so that keeps happening automatically instead.

Registry: docs/model_contracts/FROZEN_BASELINE_REGISTRY.json. Each entry optionally declares
`features_path`/`features_sha256_at_freeze`; when both are present this script also reports
whether the exact input file has drifted since the baseline was frozen (using
data/splits/DATASET_MANIFEST.json, see scripts/dataset_snapshot.py). When
features_sha256_at_freeze is null (the baseline predates that manifest, as with Omega4.6.1's
07-06 baseline), that comparison is reported as unavailable rather than guessed.

Usage:
  python scripts/verify_frozen_baselines.py            # run all wired baselines
  python scripts/verify_frozen_baselines.py --id omega461_eth_greedy_router_20260706

Exit code: 0 only if every WIRED baseline reproduces within tolerance. Unwired baselines
(`wired: false`) are reported but do not affect the exit code -- they are a documented backlog
item, not a pass. Run before any promotion decision and roughly weekly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

REGISTRY_PATH = ROOT / "docs/model_contracts/FROZEN_BASELINE_REGISTRY.json"
DATASET_MANIFEST_PATH = ROOT / "data/splits/DATASET_MANIFEST.json"


def _load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _load_dataset_manifest() -> dict[str, Any]:
    if not DATASET_MANIFEST_PATH.exists():
        return {"files": {}}
    return json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))


def _run_omega461_eth_greedy_router() -> dict[str, float]:
    """Fresh recompute of the Omega4.6.1 ETH greedy-router replay against whatever
    data/splits/year_oos/training_features_2026_rebuilt.csv currently contains. Reuses the frozen
    replay/prepare_component logic unmodified -- only the input data can differ from 2026-07-06."""
    import replay_omega4_6_1_greedy_router_20260706 as router
    import retest_omega4_6_1_extended_oos_20260706 as retest
    import numpy as np
    import pandas as pd

    device = retest.DEVICE
    frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    fee, slip = router.omega._load_fee_slip()

    # The frozen oos_predictions_*.csv files were later silently extended past their original
    # 2026-06-30 endpoint (now run to 07-12) by an unrelated regen -- prepare_component requires
    # an EXACT timestamp match against `frame`, so truncate a scratch copy to frame's own range.
    # Matches the same fix already used in
    # scripts/research_eth_omega461_tpsl_floor_portfolio_check_20260728.py.
    scratch_dir = ROOT / "tmp/research_20260730"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    components = {}
    for name, cfg in retest.COMPONENTS.items():
        pred_csv = router.OUT_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        pred = pd.read_csv(pred_csv)
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred[pred["timestamp"] <= frame["timestamp"].max()].reset_index(drop=True)
        if not pred["timestamp"].equals(frame["timestamp"]):
            raise RuntimeError(f"{name}: truncated prediction timestamps still mismatch frame")
        truncated_path = scratch_dir / f"_verify_truncated_{name}_{cfg['q_tag']}.csv"
        pred.to_csv(truncated_path, index=False)
        components[name] = router.prepare_component(frame, truncated_path, cfg, device)

    _, ledger = router.greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    if len(ledger) == 0:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0}

    market = frame[["timestamp", "ou_halflife"]]
    ledger = ledger.copy()
    ledger["entry_timestamp_dt"] = pd.to_datetime(ledger["entry_timestamp"])
    ledger = ledger.merge(market.rename(columns={"timestamp": "entry_timestamp_dt"}), on="entry_timestamp_dt", how="left")
    hit = ledger["ou_halflife"] <= router.DURATION_THRESHOLD
    gated_returns = np.where(hit, 0.0, ledger["trade_return"])
    curve = np.concatenate([[1.0], np.cumprod(1.0 + gated_returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {
        "pnl": float((curve[-1] - 1.0) * 100.0),
        "mdd": float(dd.min() * 100.0),
        "trades": int((~hit).sum()),
    }


RUNNERS: dict[str, Callable[[], dict[str, float]]] = {
    "omega461_eth_greedy_router_20260706": _run_omega461_eth_greedy_router,
}


def _check_dataset_drift(entry: dict[str, Any]) -> str:
    features_path = entry.get("features_path")
    frozen_sha = entry.get("features_sha256_at_freeze")
    if not features_path:
        return "no features_path recorded for this baseline"
    manifest = _load_dataset_manifest().get("files", {})
    current_entry = manifest.get(features_path)
    if current_entry is None:
        return f"{features_path} not registered in {DATASET_MANIFEST_PATH.relative_to(ROOT)}"
    current_sha = current_entry["sha256"]
    if frozen_sha is None:
        return (
            f"{features_path}: no hash was recorded when this baseline was frozen (predates "
            f"scripts/dataset_snapshot.py) -- current sha256={current_sha[:16]}..., cannot say "
            f"whether it matches what the baseline used"
        )
    if frozen_sha == current_sha:
        return f"{features_path}: unchanged since freeze (sha256={current_sha[:16]}...)"
    return f"{features_path}: DRIFTED since freeze (frozen={frozen_sha[:16]}... current={current_sha[:16]}...)"


def verify_one(entry: dict[str, Any]) -> tuple[bool, str]:
    baseline_id = entry["id"]
    if not entry.get("wired", False) or baseline_id not in RUNNERS:
        return True, "NOT WIRED (backlog item, does not count toward pass/fail) -- " + entry.get("notes", "")

    runner = RUNNERS[baseline_id]
    actual = runner()
    reference = entry["reference"]
    tolerance = entry["tolerance"]

    failures = []
    for metric in ("pnl", "mdd", "trades"):
        ref_v = reference.get(metric)
        tol_v = tolerance.get(metric)
        act_v = actual.get(metric)
        if ref_v is None or tol_v is None:
            continue
        if abs(act_v - ref_v) > tol_v:
            failures.append(f"{metric}: actual={act_v!r} reference={ref_v!r} tolerance={tol_v!r}")

    drift_note = _check_dataset_drift(entry)
    detail = f"actual={actual} reference={reference} | dataset: {drift_note}"
    if failures:
        return False, "FAILED (" + "; ".join(failures) + f") | {detail}"
    return True, "REPRODUCED | " + detail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default=None, help="only run this baseline id")
    args = ap.parse_args()

    registry = _load_registry()
    baselines = registry["baselines"]
    if args.id:
        baselines = [b for b in baselines if b["id"] == args.id]
        if not baselines:
            print(f"no baseline with id={args.id!r} in {REGISTRY_PATH}")
            return 2

    overall_ok = True
    for entry in baselines:
        ok, detail = verify_one(entry)
        wired = entry.get("wired", False)
        tag = "SKIP" if not wired else ("PASS" if ok else "FAIL")
        print(f"[{tag}] {entry['id']}")
        print(f"       {detail}")
        if wired and not ok:
            overall_ok = False

    print(f"\noverall: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
