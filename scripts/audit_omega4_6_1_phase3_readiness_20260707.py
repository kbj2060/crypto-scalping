"""Phase 3 of the improvement roadmap (docs/model_contracts/omega4_6_1_improvement_roadmap_20260707.md)
is deliberately GATED on data/trade-count that doesn't exist yet -- its whole point is to avoid
repeating the mistake of Candidates 1-7 (searching for a new rule/model with too few trades to
separate structure from noise). There is nothing legitimate to "do" for items 7-9 until their gates
are met. What IS legitimate right now is a repeatable readiness checkpoint that reports how far
each gate is from being met, so a future session (or a scheduled rerun of this script) can tell at
a glance whether it's time to revisit Phase 3, without re-deriving the whole investigation.

Gates checked:
  7. Microstructure/orderbook data must span >=4-6 contiguous months before the correlation screen
     is worth redoing (last screen at 13,832 samples found |corr|<0.015 across every feature).
  8/9. Live+OOS effective completed-trade count should roughly double (OOS baseline 24 trades ->
     target 48) before any new learned component or second-sleeve diversifier is trusted on a
     VAL/OOS split.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MICROSTRUCTURE_DB = ROOT / "data/live/microstructure.duckdb"
MIN_CONTIGUOUS_MONTHS = 4.0
OOS_BASELINE_TRADES = 24
TARGET_TRADE_MULTIPLIER = 2.0
LIVE_DRIFT_REPORT = ROOT / "data/ensemble/omega4_6_1_live_drift_report.json"
OUT_PATH = ROOT / "data/ensemble/omega4_6_1_phase3_readiness_report.json"


def check_microstructure_gate() -> dict:
    if not MICROSTRUCTURE_DB.exists():
        return {"ready": False, "note": "microstructure.duckdb not found"}
    con = duckdb.connect(str(MICROSTRUCTURE_DB), read_only=True)
    row = con.execute("select min(ts), max(ts), count(*) from microstructure_1m").fetchone()
    min_ts, max_ts, n = row
    span_days = (max_ts - min_ts).total_seconds() / 86400.0
    # find largest contiguous run (no gap > 6 hours) to estimate USABLE contiguous span, not just
    # raw min/max span (which is misleading if there are multi-day outages in the middle)
    gaps = con.execute("""
        with ordered as (select ts, lag(ts) over (order by ts) as prev_ts from microstructure_1m)
        select prev_ts, ts, datediff('minute', prev_ts, ts) as gap_min from ordered where prev_ts is not null
    """).fetchdf()
    gaps["is_break"] = gaps["gap_min"] > 360  # >6h gap = new contiguous segment
    gaps["segment_id"] = gaps["is_break"].cumsum()
    seg_bounds = gaps.groupby("segment_id").agg(seg_start=("prev_ts", "min"), seg_end=("ts", "max"))
    seg_bounds["span_days"] = (seg_bounds["seg_end"] - seg_bounds["seg_start"]).dt.total_seconds() / 86400.0
    longest_contiguous_days = float(seg_bounds["span_days"].max()) if not seg_bounds.empty else span_days

    months_span = span_days / 30.44
    months_contiguous = longest_contiguous_days / 30.44
    ready = months_contiguous >= MIN_CONTIGUOUS_MONTHS
    return {
        "ready": ready, "rows": int(n), "range": [str(min_ts), str(max_ts)],
        "total_span_months": round(months_span, 2), "longest_contiguous_months": round(months_contiguous, 2),
        "target_contiguous_months": MIN_CONTIGUOUS_MONTHS,
        "note": (f"ready for correlation re-screen (longest contiguous segment {months_contiguous:.1f} "
                 f"months >= {MIN_CONTIGUOUS_MONTHS} target)") if ready else
                (f"not ready: longest contiguous segment is {months_contiguous:.1f} months, "
                 f"need {MIN_CONTIGUOUS_MONTHS}; total span {months_span:.1f} months but has gaps"),
    }


def check_trade_count_gate() -> dict:
    live_closed = 0
    if LIVE_DRIFT_REPORT.exists():
        try:
            live_closed = int(json.loads(LIVE_DRIFT_REPORT.read_text())["live"]["closed_trades"])
        except Exception:
            live_closed = 0
    effective_oos_trades = OOS_BASELINE_TRADES + live_closed
    target = int(OOS_BASELINE_TRADES * TARGET_TRADE_MULTIPLIER)
    ready = effective_oos_trades >= target
    return {
        "ready": ready, "oos_baseline_trades": OOS_BASELINE_TRADES, "live_closed_trades_since": live_closed,
        "effective_trade_count": effective_oos_trades, "target_trade_count": target,
        "note": (f"ready: {effective_oos_trades} >= target {target}") if ready else
                (f"not ready: {effective_oos_trades}/{target} trades (need {target - effective_oos_trades} more "
                 f"live closed trades; run scripts/monitor_omega4_6_1_live_drift_20260707.py to refresh count)"),
    }


def main() -> int:
    print("=== Phase 3 gate 1: microstructure/orderbook data contiguous coverage ===", flush=True)
    micro = check_microstructure_gate()
    print(f"  {micro['note']}", flush=True)

    print("\n=== Phase 3 gate 2: effective trade count (OOS baseline + live closed since) ===", flush=True)
    trades = check_trade_count_gate()
    print(f"  {trades['note']}", flush=True)

    print("\n=== Overall Phase 3 readiness ===", flush=True)
    if not micro["ready"]:
        print("  Item 7 (microstructure): NOT READY -- do not re-run the correlation screen yet.", flush=True)
    else:
        print("  Item 7 (microstructure): READY -- rerun the correlation screen before considering entry/execution use.", flush=True)
    if not trades["ready"]:
        print("  Items 8/9 (diversification, new architecture search): NOT READY -- do not open a new candidate yet.", flush=True)
    else:
        print("  Items 8/9 (diversification, new architecture search): READY -- effective trade count has roughly doubled.", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "checked_at": pd.Timestamp.now().isoformat(),
        "microstructure_gate": micro, "trade_count_gate": trades,
    }, indent=2, default=str))
    print(f"\nWrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
