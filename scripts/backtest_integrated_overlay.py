#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.integrated_overlay import IntegratedOverlayConfig, compute_integrated_overlay
from scripts.backtest_polymarket_news_overlay import KST, _load_duckdb_features, _load_trades, _mdd, _sharpe

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def run_backtest(feat_kst: pd.DataFrame, trades, cfg: IntegratedOverlayConfig) -> dict:
    feat = feat_kst.copy()
    feat["ts_utc"] = feat["ts"].dt.tz_convert("UTC")
    feat = feat.set_index("ts_utc")

    base_sum = float(sum(float(t.realized_pct) for t in trades))
    base_wins = int(sum(1 for t in trades if float(t.realized_pct) > 0.0))

    rets: list[float] = []
    eq = [1.0]
    skip_count = 0
    exit_count = 0
    downsized = 0
    improved = 0
    worsened = 0

    for tr in trades:
        open_slice = feat.loc[: tr.open_ts]
        if len(open_slice) == 0:
            pnl = float(tr.realized_pct)
            rets.append(pnl)
            eq.append(eq[-1] * (1.0 + pnl / 100.0))
            continue

        open_state = open_slice.iloc[-1]
        entry_decision = compute_integrated_overlay(side=tr.side, row=open_state, cfg=cfg, dsac_strength=1.0)

        size_mult = float(entry_decision["size_mult"])
        if size_mult < 0.999:
            downsized += 1

        pnl = float(tr.realized_pct) * size_mult

        rets.append(float(pnl))
        if pnl > tr.realized_pct:
            improved += 1
        elif pnl < tr.realized_pct:
            worsened += 1
        eq.append(eq[-1] * (1.0 + pnl / 100.0))

    total = float(sum(rets))
    wins = int(sum(1 for x in rets if x > 0.0))
    return {
        "config": asdict(cfg),
        "name": cfg.name,
        "trades": len(trades),
        "baseline_sum_pct": base_sum,
        "overlay_sum_pct": total,
        "delta_pct": total - base_sum,
        "baseline_wr": 100.0 * base_wins / max(len(trades), 1),
        "overlay_wr": 100.0 * wins / max(len(trades), 1),
        "skip_count": skip_count,
        "exit_count": exit_count,
        "downsized": downsized,
        "improved_trades": improved,
        "worsened_trades": worsened,
        "overlay_mdd_pct": _mdd(eq),
        "overlay_sharpe": _sharpe(rets),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest integrated overlay skeleton against existing live trades.")
    ap.add_argument("--events-path", default="data/live/dashboard_events.jsonl")
    ap.add_argument("--out-json", default="data/ensemble/reports/integrated_overlay_backtest_20260424.json")
    args = ap.parse_args()

    feat_kst, start_utc, end_utc = _load_duckdb_features()
    trades = _load_trades(args.events_path, start_utc=start_utc, end_utc=end_utc)
    if not trades:
        raise SystemExit("No overlapping trades found for duckdb window.")

    grid = [
        IntegratedOverlayConfig(
            entry_score_th=-0.60,
            risk_block_th=0.96,
            risk_exit_th=0.88,
            micro_toxicity_block_th=1.10,
            tail_aftershock_block_th=0.94,
            tail_aftershock_exit_th=0.82,
            poly_adverse_gap_th=0.0025,
            poly_severe_gap_th=0.0050,
            poly_conf_low_th=0.06,
            max_size_mult=1.04,
            min_size_mult=0.95,
            cooldown_med_bars=3,
            cooldown_high_bars=6,
        ),
        IntegratedOverlayConfig(
            entry_score_th=-0.50,
            risk_block_th=0.92,
            risk_exit_th=0.88,
            micro_toxicity_block_th=1.10,
            tail_aftershock_block_th=0.90,
            tail_aftershock_exit_th=0.82,
            poly_adverse_gap_th=0.0030,
            poly_severe_gap_th=0.0060,
            poly_conf_low_th=0.06,
            max_size_mult=1.06,
            min_size_mult=0.96,
            cooldown_med_bars=3,
            cooldown_high_bars=6,
        ),
        IntegratedOverlayConfig(
            entry_score_th=-0.55,
            risk_block_th=0.98,
            risk_exit_th=0.82,
            micro_toxicity_block_th=1.10,
            tail_aftershock_block_th=0.96,
            tail_aftershock_exit_th=0.82,
            poly_adverse_gap_th=0.0030,
            poly_severe_gap_th=0.0050,
            poly_conf_low_th=0.06,
            max_size_mult=1.05,
            min_size_mult=0.97,
            cooldown_med_bars=3,
            cooldown_high_bars=6,
        ),
        IntegratedOverlayConfig(
            entry_score_th=-0.55,
            risk_block_th=0.96,
            risk_exit_th=0.82,
            micro_toxicity_block_th=1.10,
            tail_aftershock_block_th=0.94,
            tail_aftershock_exit_th=0.82,
            poly_adverse_gap_th=0.0030,
            poly_severe_gap_th=0.0050,
            poly_conf_low_th=0.06,
            max_size_mult=1.06,
            min_size_mult=0.98,
            cooldown_med_bars=3,
            cooldown_high_bars=6,
        ),
        IntegratedOverlayConfig(
            entry_score_th=-0.55,
            risk_block_th=0.96,
            risk_exit_th=0.82,
            micro_toxicity_block_th=1.10,
            tail_aftershock_block_th=0.94,
            tail_aftershock_exit_th=0.82,
            poly_adverse_gap_th=0.0035,
            poly_severe_gap_th=0.0055,
            poly_conf_low_th=0.06,
            max_size_mult=1.05,
            min_size_mult=0.99,
            cooldown_med_bars=3,
            cooldown_high_bars=6,
        ),
        IntegratedOverlayConfig(
            entry_score_th=-0.58,
            risk_block_th=0.96,
            risk_exit_th=0.88,
            micro_toxicity_block_th=1.10,
            tail_aftershock_block_th=0.94,
            tail_aftershock_exit_th=0.82,
            poly_adverse_gap_th=0.0030,
            poly_severe_gap_th=0.0050,
            poly_conf_low_th=0.05,
            max_size_mult=1.07,
            min_size_mult=0.96,
            cooldown_med_bars=3,
            cooldown_high_bars=6,
        ),
        IntegratedOverlayConfig(
            entry_score_th=-0.52,
            risk_block_th=0.96,
            risk_exit_th=0.88,
            micro_toxicity_block_th=1.10,
            tail_aftershock_block_th=0.94,
            tail_aftershock_exit_th=0.82,
            poly_adverse_gap_th=0.0030,
            poly_severe_gap_th=0.0050,
            poly_conf_low_th=0.07,
            max_size_mult=1.04,
            min_size_mult=0.95,
            cooldown_med_bars=3,
            cooldown_high_bars=6,
        ),
    ]

    results = []
    iterator = grid
    if tqdm is not None:
        iterator = tqdm(grid, desc="integrated-overlay-grid", ncols=100)
    for cfg in iterator:
        results.append(run_backtest(feat_kst, trades, cfg))
    results.sort(key=lambda x: (x["delta_pct"], x["overlay_sum_pct"], -x["overlay_mdd_pct"]), reverse=True)

    summary = {
        "window": {
            "duckdb_start_kst": str(start_utc.tz_convert(KST)),
            "duckdb_end_kst": str(end_utc.tz_convert(KST)),
        },
        "trade_count": len(trades),
        "baseline_sum_pct": results[0]["baseline_sum_pct"],
        "top5": results[:5],
        "all_results": results,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    best = results[0]
    print("=== Integrated Overlay Backtest ===")
    print(f"window={summary['window']['duckdb_start_kst']} -> {summary['window']['duckdb_end_kst']}")
    print(f"trades={len(trades)} baseline_sum={summary['baseline_sum_pct']:+.4f}%")
    print(
        "best="
        f"{best['name']} overlay_sum={best['overlay_sum_pct']:+.4f}% "
        f"delta={best['delta_pct']:+.4f}%p wr={best['overlay_wr']:.2f}% "
        f"mdd={best['overlay_mdd_pct']:.2f}% skips={best['skip_count']} "
        f"downsized={best['downsized']} exits={best['exit_count']}"
    )
    print(f"report={out_path}")


if __name__ == "__main__":
    main()
