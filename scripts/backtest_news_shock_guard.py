#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.news_shock_guard import NewsShockGuardConfig, compute_news_shock_guard
from scripts.backtest_polymarket_news_overlay import KST, _load_duckdb_features, _load_trades, _mdd, _sharpe

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _tail_mean_worst(returns_pct: list[float], worst_frac: float = 0.2) -> float:
    if not returns_pct:
        return 0.0
    arr = np.sort(np.asarray(returns_pct, dtype=np.float64))
    k = max(1, int(np.ceil(len(arr) * worst_frac)))
    return float(arr[:k].mean())


def run_backtest(feat_kst: pd.DataFrame, trades, cfg: NewsShockGuardConfig) -> dict:
    feat = feat_kst.copy()
    feat["ts_utc"] = feat["ts"].dt.tz_convert("UTC")
    feat = feat.set_index("ts_utc")

    base_rets = [float(t.realized_pct) for t in trades]
    base_sum = float(sum(base_rets))
    base_eq = [1.0]
    for pnl in base_rets:
        base_eq.append(base_eq[-1] * (1.0 + pnl / 100.0))

    rets: list[float] = []
    eq = [1.0]
    trigger_count = 0
    severe_count = 0
    reduced_count = 0
    improved = 0
    worsened = 0

    for tr in trades:
        open_slice = feat.loc[: tr.open_ts]
        if len(open_slice) == 0:
            pnl = float(tr.realized_pct)
            rets.append(pnl)
            eq.append(eq[-1] * (1.0 + pnl / 100.0))
            continue

        path = feat.loc[(feat.index >= tr.open_ts) & (feat.index <= tr.close_ts)]
        applied_mult = 1.0
        triggered = False
        severe = False
        for _, row in path.iterrows():
            guard = compute_news_shock_guard(tr.side, row, cfg=cfg)
            if bool(guard["trigger"]):
                applied_mult = min(applied_mult, float(guard["reduce_mult"]))
                triggered = True
                severe = severe or bool(guard["severe"])
                break

        pnl = float(tr.realized_pct) * applied_mult
        rets.append(pnl)
        eq.append(eq[-1] * (1.0 + pnl / 100.0))

        if triggered:
            trigger_count += 1
            reduced_count += 1
        if severe:
            severe_count += 1
        if pnl > tr.realized_pct:
            improved += 1
        elif pnl < tr.realized_pct:
            worsened += 1

    return {
        "config": asdict(cfg),
        "name": cfg.name,
        "trades": len(trades),
        "baseline_sum_pct": base_sum,
        "guard_sum_pct": float(sum(rets)),
        "delta_pct": float(sum(rets)) - base_sum,
        "baseline_mdd_pct": _mdd(base_eq),
        "guard_mdd_pct": _mdd(eq),
        "baseline_tail20_pct": _tail_mean_worst(base_rets, worst_frac=0.2),
        "guard_tail20_pct": _tail_mean_worst(rets, worst_frac=0.2),
        "baseline_worst_trade_pct": float(min(base_rets) if base_rets else 0.0),
        "guard_worst_trade_pct": float(min(rets) if rets else 0.0),
        "guard_sharpe": _sharpe(rets),
        "trigger_count": trigger_count,
        "severe_count": severe_count,
        "reduced_count": reduced_count,
        "improved_trades": improved,
        "worsened_trades": worsened,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest conservative news shock guard against live trades.")
    ap.add_argument("--events-path", default="data/live/dashboard_events.jsonl")
    ap.add_argument("--out-json", default="data/ensemble/reports/news_shock_guard_backtest_20260424.json")
    args = ap.parse_args()

    feat_kst, start_utc, end_utc = _load_duckdb_features()
    trades = _load_trades(args.events_path, start_utc=start_utc, end_utc=end_utc)
    if not trades:
        raise SystemExit("No overlapping trades found for duckdb window.")

    grid = [
        NewsShockGuardConfig(
            shock_trigger_th=0.68,
            aftershock_trigger_th=0.58,
            toxicity_trigger_th=0.50,
            queue_trigger_th=0.58,
            poly_momentum_trigger=0.005,
            poly_gap_trigger=0.010,
            reduce_mult=0.35,
            severe_reduce_mult=0.0,
            cooldown_bars=6,
            severe_cooldown_bars=10,
        ),
        NewsShockGuardConfig(
            shock_trigger_th=0.72,
            aftershock_trigger_th=0.62,
            toxicity_trigger_th=0.55,
            queue_trigger_th=0.62,
            poly_momentum_trigger=0.006,
            poly_gap_trigger=0.012,
            reduce_mult=0.50,
            severe_reduce_mult=0.20,
            cooldown_bars=6,
            severe_cooldown_bars=10,
        ),
        NewsShockGuardConfig(
            shock_trigger_th=0.82,
            aftershock_trigger_th=0.74,
            toxicity_trigger_th=0.68,
            queue_trigger_th=0.74,
            poly_momentum_trigger=0.009,
            poly_gap_trigger=0.016,
            reduce_mult=0.50,
            severe_reduce_mult=0.20,
            cooldown_bars=6,
            severe_cooldown_bars=10,
        ),
        NewsShockGuardConfig(
            shock_trigger_th=0.86,
            aftershock_trigger_th=0.78,
            toxicity_trigger_th=0.72,
            queue_trigger_th=0.78,
            poly_momentum_trigger=0.010,
            poly_gap_trigger=0.018,
            reduce_mult=0.35,
            severe_reduce_mult=0.0,
            cooldown_bars=6,
            severe_cooldown_bars=10,
        ),
        NewsShockGuardConfig(
            shock_trigger_th=0.90,
            aftershock_trigger_th=0.82,
            toxicity_trigger_th=0.76,
            queue_trigger_th=0.82,
            poly_momentum_trigger=0.012,
            poly_gap_trigger=0.020,
            reduce_mult=0.50,
            severe_reduce_mult=0.0,
            cooldown_bars=8,
            severe_cooldown_bars=12,
        ),
    ]

    results = []
    iterator = grid
    if tqdm is not None:
        iterator = tqdm(grid, desc="news-shock-guard-grid", ncols=100)
    for cfg in iterator:
        results.append(run_backtest(feat_kst, trades, cfg))

    results.sort(
        key=lambda x: (
            x["guard_tail20_pct"] - x["baseline_tail20_pct"],
            x["guard_worst_trade_pct"] - x["baseline_worst_trade_pct"],
            -abs(x["delta_pct"]),
            x["delta_pct"],
        ),
        reverse=True,
    )

    summary = {
        "window": {
            "duckdb_start_kst": str(start_utc.tz_convert(KST)),
            "duckdb_end_kst": str(end_utc.tz_convert(KST)),
        },
        "trade_count": len(trades),
        "top3": results[:3],
        "all_results": results,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    best = results[0]
    print("=== News Shock Guard Backtest ===")
    print(f"window={summary['window']['duckdb_start_kst']} -> {summary['window']['duckdb_end_kst']}")
    print(
        f"best={best['name']} guard_sum={best['guard_sum_pct']:+.4f}% "
        f"delta={best['delta_pct']:+.4f}%p tail20={best['guard_tail20_pct']:+.4f}% "
        f"worst={best['guard_worst_trade_pct']:+.4f}% triggers={best['trigger_count']}"
    )
    print(f"report={out_path}")


if __name__ == "__main__":
    main()
