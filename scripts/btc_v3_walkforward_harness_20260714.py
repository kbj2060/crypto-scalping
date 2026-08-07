#!/usr/bin/env python3
"""Reusable purged walk-forward harness for BTC v3 research (Stage 0 of the BTC v3 upgrade plan,
see docs/model_contracts/btc_v1_deep_analysis_20260714.md and
docs/model_contracts/btc_v3_holdout_policy_20260714.md).

Generalizes the one-off scratchpad walk-forward driver used for
docs/model_contracts/btc_v2_walkforward_evaluation_20260714.md and
docs/model_contracts/btc_v2_trendscan_threshold_sweep_20260714.md into a versioned, reusable
script: configurable fold width/step/embargo, enforces the frozen holdout boundary in code (no
fold's test window may reach HOLDOUT_START), and writes an immutable manifest (git hash, config,
per-fold + aggregate results) to its own timestamped run directory rather than overwriting a
shared path in place.

Reuses train_eval_btc_v2_regime_trendscan_20260714.py's pure functions unmodified via
monkeypatching its module-level TRAIN_END / HOURLY_DIR globals (same technique used throughout
this project's session for causal replay scripts) -- this file makes NO code changes to that
script or to any other existing script.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_btc_v2_regime_trendscan_20260714 as btc_v2  # noqa: E402

HOLDOUT_START = pd.Timestamp("2026-07-14 00:00:00")
DEFAULT_OUT_ROOT = ROOT / "tmp/causal_regen_20260516/btc_v3_walkforward_runs"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _generate_folds(
    *, start: pd.Timestamp, end: pd.Timestamp, fold_months: int, step_months: int,
    embargo_days: int, expanding: bool,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Yields (fold_id, train_end, test_start, test_end). test_start is train_end + embargo."""
    folds: list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    idx = 0
    while True:
        train_end = cursor if expanding else cursor
        test_start = (train_end + pd.Timedelta(days=embargo_days)).normalize()
        test_end = (test_start + pd.DateOffset(months=fold_months)) - pd.Timedelta(minutes=5)
        if test_end > end:
            break
        if test_end >= HOLDOUT_START:
            raise RuntimeError(
                f"fold test_end={test_end} would reach or cross the frozen holdout boundary "
                f"HOLDOUT_START={HOLDOUT_START} -- refusing to run (see "
                f"docs/model_contracts/btc_v3_holdout_policy_20260714.md)"
            )
        fold_id = chr(ord("A") + idx) if idx < 26 else f"F{idx}"
        folds.append((fold_id, train_end, test_start, test_end))
        cursor = cursor + pd.DateOffset(months=step_months)
        idx += 1
    return folds


def run(
    *, fold_months: int, step_months: int, embargo_days: int, expanding: bool,
    quality_threshold: float, regime_threshold: float | None,
    hourly_dir: Path | None, history_start: pd.Timestamp, history_end: pd.Timestamp,
    out_root: Path, run_label: str,
) -> dict[str, Any]:
    if hourly_dir is not None:
        btc_v2.HOURLY_DIR = hourly_dir
    orig_train_end = btc_v2.TRAIN_END

    print("stage=load_hourly_btc_features", flush=True)
    hourly, feature_columns = btc_v2._read_hourly()
    print("stage=load_5m_execution_tape", flush=True)
    five_minute = btc_v2._read_five_minute()

    folds = _generate_folds(
        start=history_start, end=history_end, fold_months=fold_months,
        step_months=step_months, embargo_days=embargo_days, expanding=expanding,
    )
    if not folds:
        raise RuntimeError("no folds generated -- check history_start/history_end/fold_months/step_months")

    fold_rows = []
    for fold_id, train_end, test_start, test_end in folds:
        btc_v2.TRAIN_END = train_end
        models, signal, parent_report = btc_v2._fit_parent(hourly, feature_columns)
        execution = btc_v2._merge_signal(five_minute, signal)
        side = btc_v2._candidate_side(execution, quality_threshold=quality_threshold, regime_threshold=regime_threshold)
        metrics, ledger, equity, frame = btc_v2._period(execution, side, test_start, test_end)
        row = {
            "fold": fold_id, "train_end": train_end, "test_start": test_start, "test_end": test_end,
            "train_rows": parent_report["train_rows"],
            "pnl": metrics["pnl"], "mdd": metrics["mdd"], "trades": metrics["trades"], "wr": metrics["wr"],
        }
        fold_rows.append(row)
        print(f"fold={fold_id} train_end={train_end.date()} test=[{test_start.date()}..{test_end.date()}] "
              f"pnl={metrics['pnl']:8.2f}% mdd={metrics['mdd']:7.2f}% trades={metrics['trades']:3d} wr={metrics['wr']:.1%}", flush=True)

    btc_v2.TRAIN_END = orig_train_end

    df = pd.DataFrame(fold_rows)
    summary = {
        "positive_folds": int((df["pnl"] > 0).sum()),
        "total_folds": len(df),
        "mean_pnl": float(df["pnl"].mean()),
        "std_pnl": float(df["pnl"].std()) if len(df) > 1 else 0.0,
        "mean_mdd": float(df["mdd"].mean()),
        "worst_mdd": float(df["mdd"].min()),
        "total_trades": int(df["trades"].sum()),
    }

    run_id = f"{pd.Timestamp.now().strftime('%Y%m%dT%H%M%S')}_{run_label}"
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=False)  # immutable: never overwrite a prior run

    manifest = {
        "run_id": run_id,
        "git_hash": _git_hash(),
        "created_at": pd.Timestamp.now().isoformat(),
        "config": {
            "fold_months": fold_months, "step_months": step_months, "embargo_days": embargo_days,
            "expanding": expanding, "quality_threshold": quality_threshold,
            "regime_threshold": regime_threshold, "hourly_dir": str(hourly_dir or btc_v2.HOURLY_DIR),
            "history_start": history_start, "history_end": history_end,
        },
        "holdout_start": HOLDOUT_START,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "promotion_grade": False,
        "summary": summary,
        "folds": fold_rows,
    }
    (out_dir / "report.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    df.to_csv(out_dir / "fold_results.csv", index=False)
    print(f"\nsaved immutable run to {out_dir}", flush=True)
    print(f"summary: {summary}", flush=True)
    return manifest


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold-months", type=int, default=3)
    ap.add_argument("--step-months", type=int, default=3)
    ap.add_argument("--embargo-days", type=int, default=1)
    ap.add_argument("--expanding", action="store_true", default=True)
    ap.add_argument("--quality-threshold", type=float, default=0.55)
    ap.add_argument("--regime-threshold", type=float, default=0.50)
    ap.add_argument("--hourly-dir", type=Path, default=None)
    ap.add_argument("--history-start", type=str, default="2024-12-31")
    ap.add_argument("--history-end", type=str, default="2026-07-12")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--run-label", type=str, default="btc_v3_stage0")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    run(
        fold_months=args.fold_months, step_months=args.step_months, embargo_days=args.embargo_days,
        expanding=args.expanding, quality_threshold=args.quality_threshold,
        regime_threshold=args.regime_threshold, hourly_dir=args.hourly_dir,
        history_start=pd.Timestamp(args.history_start), history_end=pd.Timestamp(args.history_end),
        out_root=args.out_root, run_label=args.run_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
