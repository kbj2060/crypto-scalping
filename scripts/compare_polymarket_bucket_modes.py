#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_polymarket_news_overlay import (
    KST,
    OverlayConfig,
    _fetch_binance_1m,
    _load_duckdb_features,
    _load_trades,
    run_backtest as run_hard_backtest,
)
from scripts.backtest_polymarket_soft_overlay import SoftConfig, run_backtest as run_soft_backtest

BEST_HARD_CFG = OverlayConfig(
    entry_gap_th=0.0030,
    exit_gap_th=0.0035,
    shock_th=0.18,
    tail_th=0.52,
    aftershock_cap=0.45,
    toxicity_cap=0.80,
    neutral_gap_th=0.0015,
    entropy_cap=0.78,
)

BEST_SOFT_CFG = SoftConfig(
    veto_gap_th=0.0030,
    veto_tail_th=0.55,
    veto_entropy_cap=0.80,
    size_gap_th=0.0030,
    size_tail_th=0.52,
    shock_th=0.18,
    aftershock_cap=0.45,
    toxicity_cap=0.80,
    adverse_mult=0.80,
    neutral_mult=0.75,
)


def _summarize_mode(
    mode: str,
    tilted_alpha: float,
    events_path: str,
    fee: float,
    slip: float,
) -> dict:
    feat_kst, start_utc, end_utc = _load_duckdb_features(bucket_mode=mode, tilted_alpha=tilted_alpha)
    trades = _load_trades(events_path, start_utc=start_utc, end_utc=end_utc)
    if not trades:
        raise RuntimeError(f"No overlapping trades found for mode={mode}.")
    px_utc = _fetch_binance_1m(start_utc - pd.Timedelta(minutes=5), end_utc + pd.Timedelta(minutes=5))

    hard_result = run_hard_backtest(feat_kst, px_utc, trades, BEST_HARD_CFG, fee=fee, slip=slip)
    soft_result = run_soft_backtest(feat_kst, trades, BEST_SOFT_CFG)

    return {
        "bucket_mode": mode,
        "tilted_alpha": tilted_alpha,
        "window": {
            "duckdb_start_kst": str(start_utc.tz_convert(KST)),
            "duckdb_end_kst": str(end_utc.tz_convert(KST)),
        },
        "trade_count": len(trades),
        "baseline_sum_pct": float(hard_result["baseline_sum_pct"]),
        "hard_result": hard_result,
        "soft_result": soft_result,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare polymarket bucket valuation modes.")
    ap.add_argument("--events-path", default="data/live/dashboard_events.jsonl")
    ap.add_argument("--out-json", default="data/ensemble/reports/compare_polymarket_bucket_modes_20260424.json")
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--tilted-alpha", type=float, default=0.75)
    args = ap.parse_args()

    summaries = []
    for mode in ("mid", "tilted_upper", "upper"):
        alpha = args.tilted_alpha if mode == "tilted_upper" else 0.75
        summaries.append(
            _summarize_mode(
                mode=mode,
                tilted_alpha=alpha,
                events_path=args.events_path,
                fee=args.fee,
                slip=args.slip,
            )
        )

    best_soft = max(summaries, key=lambda x: x["soft_result"]["overlay_sum_pct"])
    best_hard = max(summaries, key=lambda x: x["hard_result"]["overlay_sum_pct"])
    report = {
        "modes": summaries,
        "best_soft_mode": {
            "bucket_mode": best_soft["bucket_mode"],
            "result": best_soft["soft_result"],
        },
        "best_hard_mode": {
            "bucket_mode": best_hard["bucket_mode"],
            "result": best_hard["hard_result"],
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Compare Polymarket Bucket Modes ===")
    for row in summaries:
        print(
            f"{row['bucket_mode']}: baseline={row['baseline_sum_pct']:+.4f}% "
            f"soft={row['soft_result']['overlay_sum_pct']:+.4f}% "
            f"hard={row['hard_result']['overlay_sum_pct']:+.4f}%"
        )
    print(
        f"best_soft_mode={report['best_soft_mode']['bucket_mode']} "
        f"best_soft={report['best_soft_mode']['result']['overlay_sum_pct']:+.4f}%"
    )
    print(
        f"best_hard_mode={report['best_hard_mode']['bucket_mode']} "
        f"best_hard={report['best_hard_mode']['result']['overlay_sum_pct']:+.4f}%"
    )
    print(f"report={out_path}")


if __name__ == "__main__":
    main()
