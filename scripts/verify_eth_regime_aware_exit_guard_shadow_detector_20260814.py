#!/usr/bin/env python3
"""OFFLINE self-verification, run BEFORE deploying scripts/live_eth_regime_aware_exit_guard_shadow_
20260814.py: confirms that script's SustainedUptrendDetector (a live, O(1)-per-bar, incremental
rolling computation) produces EXACTLY the same per-bar score/active series as
research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814._rolling_dual_momentum_score
(the vectorized, whole-CSV pandas .rolling(2016, min_periods=2016).mean() the backtest actually used
and was gated on). Both functions are imported verbatim from their real, already-deployed/
already-validated source modules -- nothing is reimplemented or re-derived here, including
DETECTOR_THRESHOLD (read off the live shadow script's own hardcoded constant, itself copied verbatim
from the backtest's report.json).

Three checks, over BOTH base CSVs the backtest itself used (2025 full year + 2026 rebuilt):
  1. Full-file causal replay: feed dual_momentum row-by-row (timestamp order, matching the batch's
     own sort+dedup discipline) into a fresh SustainedUptrendDetector, starting empty exactly as the
     batch computation starts from the first row of the file. Compare every row's (score, active)
     against the batch series.
  2. Seed/resume replay: split the file at several arbitrary cut points, .seed() a fresh detector
     with the prefix (mimicking scripts/live_eth_regime_aware_exit_guard_shadow_20260814.seed_detector
     priming a detector from already-collected buffer history after a process restart), then
     .update() the remaining suffix one row at a time -- must reproduce the SAME per-row results as
     check 1 (proving seed()+update() is equivalent to a from-scratch update()-only replay, which is
     the actual live cold-start/resume code path).
  3. Activation-rate summary per file, for a human sanity cross-check against the backtest's own
     reported per-window activation rates (docs/experiments/eth_omega461_regime_aware_exit_head_
     uptrend_guard_20260814.md's per-window activation table) -- not a pass/fail gate on its own,
     informative only.

Exits 0 and prints "VERIFICATION PASSED" only if checks 1+2 match EXACTLY (score equality to 1e-9,
NaN-vs-None alignment exact, active-flag equality exact) on both files. Read-only: does not modify
any CSV/model/live file, no retraining, CPU-only, matching this whole script lineage.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
from live_eth_regime_aware_exit_guard_shadow_20260814 import (  # noqa: E402
    DETECTOR_THRESHOLD,
    DETECTOR_WEEK_BARS,
    SustainedUptrendDetector,
)

OUT_PATH = ROOT / "tmp/causal_regen_20260516/eth_omega461_regime_aware_exit_guard_shadow_20260814/detector_verification_report.json"
SEED_CUT_FRACTIONS = (0.1, 0.35, 0.6, 0.85)  # arbitrary cut points to exercise seed()+update()


def log(msg: str) -> None:
    print(f"[verify_detector] {msg}", flush=True)


def _load_dual_momentum(base_csv: Path) -> pd.DataFrame:
    """Identical read/sort/dedup discipline to guard._rolling_dual_momentum_score's own frame load
    (byte-for-byte the same call), so row order/count matches exactly."""
    frame = pd.read_csv(base_csv, low_memory=False, usecols=["timestamp", "dual_momentum"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return frame


def check_full_replay(base_csv: Path) -> dict[str, Any]:
    log(f"=== full-file causal replay: {base_csv.name} ===")
    frame = _load_dual_momentum(base_csv)
    batch = guard._rolling_dual_momentum_score(base_csv)
    assert batch["timestamp"].equals(frame["timestamp"]), "batch/frame timestamp mismatch"
    dm = pd.to_numeric(frame["dual_momentum"], errors="raise").to_numpy(dtype=float)
    batch_score = batch["sustained_uptrend_score"].to_numpy(dtype=float)  # NaN where warm-up

    det = SustainedUptrendDetector(threshold=DETECTOR_THRESHOLD, week_bars=DETECTOR_WEEK_BARS)
    n = len(dm)
    live_score = np.full(n, np.nan, dtype=float)
    live_active = np.zeros(n, dtype=bool)
    for i in range(n):
        score, active = det.update(dm[i])
        if score is not None:
            live_score[i] = score
            live_active[i] = active

    batch_nan = np.isnan(batch_score)
    live_nan = np.isnan(live_score)
    nan_mismatch = int((batch_nan != live_nan).sum())
    both_finite = ~batch_nan & ~live_nan
    max_abs_diff = float(np.max(np.abs(batch_score[both_finite] - live_score[both_finite]))) if both_finite.any() else 0.0
    score_mismatch = int((np.abs(batch_score[both_finite] - live_score[both_finite]) > 1e-9).sum())
    batch_active = (batch_score > DETECTOR_THRESHOLD) & ~batch_nan
    active_mismatch = int((batch_active != live_active).sum())

    result = {
        "base_csv": str(base_csv), "n_bars": int(n),
        "nan_bars_batch": int(batch_nan.sum()), "nan_bars_live": int(live_nan.sum()),
        "nan_alignment_mismatch_bars": nan_mismatch,
        "score_mismatch_bars_gt_1e9": score_mismatch, "max_abs_score_diff": max_abs_diff,
        "active_mismatch_bars": active_mismatch,
        "batch_active_frac": float(batch_active.mean()), "live_active_frac": float(live_active.mean()),
        "pass": bool(nan_mismatch == 0 and score_mismatch == 0 and active_mismatch == 0),
    }
    log(f"  n_bars={n} nan_batch={result['nan_bars_batch']} nan_live={result['nan_bars_live']} "
        f"nan_mismatch={nan_mismatch} score_mismatch={score_mismatch} max_abs_diff={max_abs_diff:.3e} "
        f"active_mismatch={active_mismatch} batch_active_frac={result['batch_active_frac']*100:.2f}% "
        f"live_active_frac={result['live_active_frac']*100:.2f}% pass={result['pass']}")
    return result


def check_seed_resume(base_csv: Path) -> dict[str, Any]:
    log(f"=== seed()+update() resume replay: {base_csv.name} ===")
    frame = _load_dual_momentum(base_csv)
    batch = guard._rolling_dual_momentum_score(base_csv)
    dm = pd.to_numeric(frame["dual_momentum"], errors="raise").to_numpy(dtype=float)
    batch_score = batch["sustained_uptrend_score"].to_numpy(dtype=float)
    n = len(dm)

    per_cut: list[dict[str, Any]] = []
    all_pass = True
    for frac in SEED_CUT_FRACTIONS:
        cut = int(n * frac)
        det = SustainedUptrendDetector(threshold=DETECTOR_THRESHOLD, week_bars=DETECTOR_WEEK_BARS)
        # Mimics live_eth_regime_aware_exit_guard_shadow_20260814.seed_detector: seed with everything
        # up to (and including) the cut row, then update() the remainder one at a time -- this is
        # exactly the resume code path (seed from buffer history, then process new bars live).
        det.seed(dm[: cut + 1])
        mismatches = 0
        checked = 0
        for i in range(cut + 1, n):
            score, active = det.update(dm[i])
            checked += 1
            b = batch_score[i]
            if np.isnan(b):
                if score is not None:
                    mismatches += 1
                continue
            if score is None or abs(score - b) > 1e-9 or active != bool(b > DETECTOR_THRESHOLD):
                mismatches += 1
        ok = mismatches == 0
        all_pass = all_pass and ok
        per_cut.append({"cut_fraction": frac, "cut_row": cut, "rows_checked_after_cut": checked, "mismatches": mismatches, "pass": ok})
        log(f"  cut_frac={frac} cut_row={cut} rows_checked={checked} mismatches={mismatches} pass={ok}")
    return {"base_csv": str(base_csv), "per_cut": per_cut, "pass": bool(all_pass)}


def main() -> int:
    t0 = time.time()
    log(f"DETECTOR_THRESHOLD={DETECTOR_THRESHOLD!r} DETECTOR_WEEK_BARS={DETECTOR_WEEK_BARS!r} "
        f"(both copied verbatim from live_eth_regime_aware_exit_guard_shadow_20260814.py's own "
        f"module-level constants -- not recomputed here)")
    base_csvs = [guard.sweep.BASE_2025, guard.sweep.BASE_2026]
    for p in base_csvs:
        if not p.exists():
            log(f"FATAL: base csv missing: {p}")
            return 2

    full_results = [check_full_replay(p) for p in base_csvs]
    seed_results = [check_seed_resume(p) for p in base_csvs]
    overall_pass = all(r["pass"] for r in full_results) and all(r["pass"] for r in seed_results)

    report = {
        "detector_threshold": DETECTOR_THRESHOLD, "detector_week_bars": DETECTOR_WEEK_BARS,
        "full_replay": full_results, "seed_resume_replay": seed_results,
        "overall_pass": overall_pass, "elapsed_seconds": time.time() - t0,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report written to {OUT_PATH}")
    if overall_pass:
        log(f"VERIFICATION PASSED (elapsed={report['elapsed_seconds']:.1f}s)")
        return 0
    log("VERIFICATION FAILED -- see mismatches above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
