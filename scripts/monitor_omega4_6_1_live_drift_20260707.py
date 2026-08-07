"""Phase 2 item 4 of the improvement roadmap: live monitoring / drift alerting for Omega4.6.1.

No automated process previously existed to flag whether REALIZED live performance is falling
outside the range already observed in backtests. This script:

1. Builds (and caches) a reference distribution from every window this project has scored this
   session (2025-Q1/Q2/Q3, VAL 2025-10..12, OOS 2026-01..06) -- per-trade returns, per-window
   pnl/mdd/wr, and the worst historical MDD (2025-Q3, per the Phase 1 audit).
2. Parses `data/live/trade_journal.jsonl` for Omega4.6.1 ENTER/EXIT events, pairs them by
   `trade_id`, and computes the live equity curve, live MDD, live win rate, live trade count.
3. Flags (WARN, not a hard gate -- this is monitoring, not an automated kill-switch) if:
   - live trade count is large enough to be meaningful (>= MIN_TRADES_FOR_STATS) AND live win rate
     falls below the worst historical window's win rate minus a margin,
   - live MDD is worse than the worst historical MDD (2025-Q3's -44.37%) by a margin,
   - any single live trade's return falls outside the historical min/max trade-return range (a
     sign of a broken barrier/duration-gate rather than normal variance).

Read-only against `data/live/trade_journal.jsonl`; does not touch trading_bot.py or any live
state. Intended to be rerun periodically (cron/supervisor hook).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
from test_omega4_6_1_drop_h48qual_20260706 import _metrics  # noqa: E402
from audit_omega4_6_1_phase1_robustness_20260707 import load_2025_quarter_components, load_val_components, load_oos_components  # noqa: E402

REFERENCE_PATH = ROOT / "data/ensemble/omega4_6_1_reference_distribution.json"
REPORT_PATH = ROOT / "data/ensemble/omega4_6_1_live_drift_report.json"
JOURNAL_PATH = ROOT / "data/live/trade_journal.jsonl"
MIN_TRADES_FOR_STATS = 8
WIN_RATE_MARGIN = 0.10       # WARN if live wr < worst_historical_wr - margin
MDD_MARGIN_PCT = 5.0         # WARN if live mdd% < worst_historical_mdd% - margin (more negative)
TRADE_RETURN_MARGIN = 0.02   # WARN if a single trade return is outside [min,max] by this much


def build_reference_distribution(force: bool = False) -> dict:
    if REFERENCE_PATH.exists() and not force:
        return json.loads(REFERENCE_PATH.read_text())

    fee, slip = omega._load_fee_slip()
    windows = []
    for start, end, label in [("2025-01-01", "2025-03-31 23:59:59", "2025-Q1"),
                              ("2025-04-01", "2025-06-30 23:59:59", "2025-Q2"),
                              ("2025-07-01", "2025-09-30 23:59:59", "2025-Q3")]:
        frame, components = load_2025_quarter_components(start, end)
        windows.append((label, frame, components))
    val_frame, val_components = load_val_components()
    windows.append(("VAL", val_frame, val_components))
    oos_frame, oos_components = load_oos_components()
    windows.append(("OOS", oos_frame, oos_components))

    all_returns: list[float] = []
    per_window = {}
    for label, frame, components in windows:
        greedy.PRIORITY = ("h48qual", "zig075")
        _, lg = greedy.greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=1.0, device=retest.DEVICE)
        m = _metrics(lg, frame, apply_gate=True)
        per_window[label] = m
        if not lg.empty:
            all_returns.extend(lg["trade_return"].astype(float).tolist())
        print(f"  reference window {label}: {m}", flush=True)

    ref = {
        "built_at": pd.Timestamp.now().isoformat(),
        "per_window_metrics": per_window,
        "worst_wr": min(m["wr"] for m in per_window.values()),
        "worst_mdd_pct": min(m["mdd"] for m in per_window.values()),
        "min_trade_return": min(all_returns) if all_returns else None,
        "max_trade_return": max(all_returns) if all_returns else None,
        "n_reference_trades": len(all_returns),
    }
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(json.dumps(ref, indent=2))
    print(f"Wrote reference distribution to {REFERENCE_PATH}", flush=True)
    return ref


def load_live_omega4_6_1_trades() -> pd.DataFrame:
    if not JOURNAL_PATH.exists():
        return pd.DataFrame(columns=["trade_id", "event", "ts", "pnl_frac", "reason"])
    rows = []
    with open(JOURNAL_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sleeve = str(d.get("model_sleeve") or d.get("open_model_sleeve") or "")
            source = str(d.get("source") or d.get("open_source") or "")
            if "omega4_6_1" not in sleeve and "omega4_6_1" not in source:
                continue
            rows.append(d)
    if not rows:
        return pd.DataFrame(columns=["trade_id", "event", "ts", "pnl_frac", "reason"])
    df = pd.DataFrame(rows)
    return df


def analyze_live(df: pd.DataFrame) -> dict:
    exits = df[df["event"].astype(str).str.upper().str.startswith("EXIT")].copy() if not df.empty else df
    if exits.empty:
        return {"closed_trades": 0, "returns": [], "pnl_pct": 0.0, "mdd_pct": 0.0, "wr": 0.0}
    returns = pd.to_numeric(exits["pnl_frac"], errors="coerce").dropna().tolist()
    curve = np.concatenate([[1.0], np.cumprod([1.0 + r for r in returns])])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {"closed_trades": len(returns), "returns": returns,
            "pnl_pct": float((curve[-1] - 1.0) * 100.0), "mdd_pct": float(dd.min() * 100.0),
            "wr": float(np.mean([r > 0 for r in returns])) if returns else 0.0}


def main() -> int:
    print("=== Building/loading reference distribution ===", flush=True)
    ref = build_reference_distribution()
    print(f"reference: worst_wr={ref['worst_wr']:.3f} worst_mdd={ref['worst_mdd_pct']:.2f}% "
          f"trade_return_range=[{ref['min_trade_return']:+.4f}, {ref['max_trade_return']:+.4f}] "
          f"n_reference_trades={ref['n_reference_trades']}", flush=True)

    print("\n=== Analyzing live Omega4.6.1 trades ===", flush=True)
    live_df = load_live_omega4_6_1_trades()
    live = analyze_live(live_df)
    print(f"live closed trades: {live['closed_trades']}  pnl={live['pnl_pct']:+.2f}%  "
          f"mdd={live['mdd_pct']:+.2f}%  wr={live['wr']:.3f}", flush=True)

    flags = []
    if live["closed_trades"] < MIN_TRADES_FOR_STATS:
        print(f"\nINFO: only {live['closed_trades']}/{MIN_TRADES_FOR_STATS} live trades closed -- "
              f"insufficient for statistical comparison yet, monitoring in accumulation mode.", flush=True)
    else:
        if live["wr"] < ref["worst_wr"] - WIN_RATE_MARGIN:
            flags.append(f"WARN win_rate {live['wr']:.3f} is below worst historical window "
                          f"({ref['worst_wr']:.3f}) minus margin {WIN_RATE_MARGIN}")
        if live["mdd_pct"] < ref["worst_mdd_pct"] - MDD_MARGIN_PCT:
            flags.append(f"WARN live MDD {live['mdd_pct']:.2f}% is worse than worst historical MDD "
                         f"({ref['worst_mdd_pct']:.2f}%) minus margin {MDD_MARGIN_PCT}pp")
        for r in live["returns"]:
            if r < ref["min_trade_return"] - TRADE_RETURN_MARGIN or r > ref["max_trade_return"] + TRADE_RETURN_MARGIN:
                flags.append(f"WARN a live trade return {r:+.4f} falls outside the historical "
                             f"range [{ref['min_trade_return']:+.4f}, {ref['max_trade_return']:+.4f}] "
                             f"+/- {TRADE_RETURN_MARGIN} -- check for a broken barrier/duration gate")

    if flags:
        print("\n=== FLAGS ===", flush=True)
        for f in flags:
            print(f"  {f}", flush=True)
    else:
        print("\n=== No drift flags raised ===", flush=True)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "checked_at": pd.Timestamp.now().isoformat(), "reference": ref, "live": live, "flags": flags,
    }, indent=2))
    print(f"\nWrote {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
