#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import backtest_alpha3_runtime_native_20260515 as runtime_native  # noqa: E402


DEFAULT_REPORT = ROOT / "data/ensemble/reports/alpha3_csv_live_unified_jan1m_20260516.json"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/alpha3_csv_live_unified_jan1m_20260516_ledger.csv"


def _date_range_to_indices(eval_csv: Path, start: str, end: str) -> tuple[int, int]:
    df = pd.read_csv(eval_csv, usecols=["timestamp"])
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (ts >= start_ts) & (ts < end_ts)
    idx = df.index[mask].to_numpy()
    if idx.size == 0:
        raise ValueError(f"empty date range: {start}..{end}")
    # The runtime contract decides on bar i and executes on i+1. Keep execution
    # inside the requested CSV date range by stopping one bar before range end.
    return int(idx[0]), int(max(idx[0], idx[-1] - 1))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Canonical Alpha3 CSV/live parity backtest. CSV historical runs and "
            "live-shadow backtests both call trading_bot.FinalGovernorRuntime.decide()."
        )
    )
    p.add_argument("--eval-csv", type=Path, default=runtime_native.DEFAULT_EVAL_CSV)
    p.add_argument("--start", type=str, default="2026-01-01")
    p.add_argument("--end", type=str, default="2026-02-01")
    p.add_argument("--window-bars", type=int, default=1200)
    p.add_argument("--progress", type=int, default=1000)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--ledger-out", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--max-bars", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    eval_csv = Path(args.eval_csv).resolve()
    start_index, end_index = _date_range_to_indices(eval_csv, args.start, args.end)

    os.environ["FINAL_GOVERNOR_ALPHA3_CANONICAL_DECISION_ENABLE"] = "1"
    os.environ["FINAL_GOVERNOR_V31_ENABLE"] = "1"
    os.environ["FINAL_GOVERNOR_V31_REQUIRED"] = "1"
    os.environ["FINAL_GOVERNOR_V31_DEEP_NOTIONAL"] = "2.0"
    os.environ["FINAL_GOVERNOR_WINDOW_BARS"] = str(int(args.window_bars))

    ns = argparse.Namespace(
        eval_csv=eval_csv,
        report_out=Path(args.report_out).resolve(),
        ledger_out=Path(args.ledger_out).resolve(),
        start_index=int(start_index),
        end_index=int(end_index),
        max_bars=args.max_bars,
        with_m7=False,
        progress=int(args.progress),
        v31_config_json="",
        v31_name="",
        accelerated_cache=True,
    )
    report = runtime_native.run(ns)
    report["model_id"] = "alpha3_csv_live_unified_20260516"
    report["unified_contract"] = {
        "csv_backtest": "FinalGovernorRuntime.decide",
        "live_backtest": "FinalGovernorRuntime.decide",
        "decision_path_identical": True,
        "legacy_direct_csv_backtest": "non_canonical_research_only",
        "date_range": {"start": args.start, "end": args.end},
        "index_range": {"start_index": int(start_index), "end_index": int(end_index)},
        "window_bars": int(args.window_bars),
        "alpha3_canonical_decision": True,
        "v31_deep_notional": 2.0,
    }
    ns.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=runtime_native._json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(ns.report_out),
                "ledger": str(ns.ledger_out),
                "metrics": report["metrics"],
                "unified_contract": report["unified_contract"],
            },
            ensure_ascii=False,
            default=runtime_native._json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
