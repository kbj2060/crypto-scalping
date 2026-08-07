#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.playbook_meta_controller import PlaybookMetaConfig, compute_playbook_meta_controller
from scripts.backtest_polymarket_news_overlay import (
    KST,
    _est_lev,
    _load_duckdb_features,
    _load_trades,
    _mdd,
    _net_frac,
    _sharpe,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


FEE = 0.0005
SLIP = 0.0002


def _tail_mean_worst(returns_pct: list[float], worst_frac: float = 0.2) -> float:
    if not returns_pct:
        return 0.0
    arr = np.sort(np.asarray(returns_pct, dtype=np.float64))
    k = max(1, int(np.ceil(len(arr) * worst_frac)))
    return float(arr[:k].mean())


def _find_exit(path: pd.DataFrame, entry_idx: int, exit_limit_idx: int, side: str, cfg: PlaybookMetaConfig) -> tuple[int, str, float]:
    last_idx = max(entry_idx, exit_limit_idx)
    trigger_idx = last_idx
    reason = "BASE_CLOSE"
    peak_pnl = -999.0
    for j in range(entry_idx + 1, last_idx + 1):
        row = path.iloc[j]
        ctl = compute_playbook_meta_controller(side, row, cfg=cfg)
        danger = float(ctl["exit_danger"])
        trigger = float(ctl["exit_trigger"])
        if j == entry_idx + 1:
            peak_pnl = danger
        else:
            peak_pnl = max(peak_pnl, danger)
        if danger >= trigger:
            trigger_idx = j
            reason = f"PLAYBOOK_EXIT:{ctl['playbook']}"
            break
    return trigger_idx, reason, peak_pnl


def run_backtest(feat_kst: pd.DataFrame, trades, cfg: PlaybookMetaConfig) -> dict:
    feat = feat_kst.copy()
    feat["ts_utc"] = feat["ts"].dt.tz_convert("UTC")
    feat = feat.set_index("ts_utc", drop=False)

    base_rets = [float(t.realized_pct) for t in trades]
    base_sum = float(sum(base_rets))
    base_eq = [1.0]
    for pnl in base_rets:
        base_eq.append(base_eq[-1] * (1.0 + pnl / 100.0))

    rets: list[float] = []
    eq = [1.0]
    skipped = 0
    delayed = 0
    early_exits = 0
    hold_capped = 0
    boosted = 0
    reduced = 0
    improved = 0
    worsened = 0
    playbook_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}

    for tr in trades:
        path = feat.loc[(feat.index >= tr.open_ts.floor("min")) & (feat.index <= tr.close_ts.ceil("min"))].copy()
        if len(path) == 0:
            pnl = float(tr.realized_pct)
            rets.append(pnl)
            eq.append(eq[-1] * (1.0 + pnl / 100.0))
            continue

        path = path.sort_index().reset_index(drop=True)
        open_row = path.iloc[0]
        ctl = compute_playbook_meta_controller(tr.side, open_row, cfg=cfg)
        playbook_counts[ctl["playbook"]] = playbook_counts.get(ctl["playbook"], 0) + 1
        mode_counts[ctl["mode"]] = mode_counts.get(ctl["mode"], 0) + 1

        if bool(ctl["skip_entry"]):
            pnl = 0.0
            skipped += 1
            worsened += 1 if tr.realized_pct > 0 else 0
            rets.append(pnl)
            eq.append(eq[-1] * (1.0 + pnl / 100.0))
            continue

        entry_idx = min(int(ctl["delay_bars"]), len(path) - 1)
        if entry_idx > 0:
            delayed += 1
        entry_ts = pd.Timestamp(path.iloc[entry_idx]["ts_utc"])
        if entry_ts >= tr.close_ts:
            pnl = 0.0
            skipped += 1
            worsened += 1 if tr.realized_pct > 0 else 0
            rets.append(pnl)
            eq.append(eq[-1] * (1.0 + pnl / 100.0))
            continue

        lev = _est_lev(tr, fee=FEE, slip=SLIP)
        adj_lev = max(0.0, lev * float(ctl["size_mult"]))
        if ctl["size_mult"] > 1.001:
            boosted += 1
        elif ctl["size_mult"] < 0.999:
            reduced += 1

        entry_price = float(path.iloc[entry_idx]["close"])
        close_limit_ts = min(tr.close_ts, entry_ts + pd.Timedelta(minutes=int(ctl["max_hold_bars"])))
        exit_limit_idx = int(path.index[path["ts_utc"] <= close_limit_ts].max()) if (path["ts_utc"] <= close_limit_ts).any() else len(path) - 1
        if close_limit_ts < tr.close_ts:
            hold_capped += 1

        exit_idx, exit_reason, _ = _find_exit(path, entry_idx, exit_limit_idx, tr.side, cfg)
        if exit_reason != "BASE_CLOSE":
            early_exits += 1
        exit_price = float(path.iloc[exit_idx]["close"])
        pnl = _net_frac(tr.side, entry_price, exit_price, adj_lev, FEE, SLIP) * 100.0

        rets.append(float(pnl))
        eq.append(eq[-1] * (1.0 + pnl / 100.0))
        if pnl > tr.realized_pct:
            improved += 1
        elif pnl < tr.realized_pct:
            worsened += 1

    overlay_sum = float(sum(rets))
    overlay_tail20 = _tail_mean_worst(rets, 0.2)
    overlay_mdd = _mdd(eq)
    objective = (
        (overlay_sum - base_sum)
        + 0.45 * (overlay_tail20 - _tail_mean_worst(base_rets, 0.2))
        + 0.20 * (overlay_mdd - _mdd(base_eq))
    )
    return {
        "config": asdict(cfg),
        "name": cfg.name,
        "trades": len(trades),
        "baseline_sum_pct": base_sum,
        "overlay_sum_pct": overlay_sum,
        "delta_pct": overlay_sum - base_sum,
        "baseline_mdd_pct": _mdd(base_eq),
        "overlay_mdd_pct": overlay_mdd,
        "baseline_tail20_pct": _tail_mean_worst(base_rets, 0.2),
        "overlay_tail20_pct": overlay_tail20,
        "baseline_worst_trade_pct": float(min(base_rets) if base_rets else 0.0),
        "overlay_worst_trade_pct": float(min(rets) if rets else 0.0),
        "overlay_sharpe": _sharpe(rets),
        "objective": float(objective),
        "skipped": skipped,
        "delayed": delayed,
        "early_exits": early_exits,
        "hold_capped": hold_capped,
        "boosted": boosted,
        "reduced": reduced,
        "improved_trades": improved,
        "worsened_trades": worsened,
        "playbook_counts": playbook_counts,
        "mode_counts": mode_counts,
    }


def _coarse_grid() -> list[PlaybookMetaConfig]:
    return [
        PlaybookMetaConfig(event_k=0.95, hazard_k=1.10, continuation_k=0.90, pullback_k=0.90, size_boost=0.10, delay_scale=1.5, hold_scale=0.20, exit_aggr=0.90, skip_hazard_th=0.88),
        PlaybookMetaConfig(event_k=1.05, hazard_k=1.15, continuation_k=1.00, pullback_k=0.95, size_boost=0.12, delay_scale=2.0, hold_scale=0.25, exit_aggr=1.00, skip_hazard_th=0.86),
        PlaybookMetaConfig(event_k=1.10, hazard_k=1.20, continuation_k=1.05, pullback_k=1.00, size_boost=0.14, delay_scale=2.0, hold_scale=0.30, exit_aggr=1.05, skip_hazard_th=0.84),
        PlaybookMetaConfig(event_k=1.20, hazard_k=1.30, continuation_k=1.10, pullback_k=1.00, size_boost=0.12, delay_scale=2.5, hold_scale=0.30, exit_aggr=1.10, skip_hazard_th=0.82),
        PlaybookMetaConfig(event_k=1.00, hazard_k=1.25, continuation_k=0.95, pullback_k=1.05, size_boost=0.10, delay_scale=2.5, hold_scale=0.15, exit_aggr=1.15, skip_hazard_th=0.80),
        PlaybookMetaConfig(event_k=1.15, hazard_k=1.05, continuation_k=1.10, pullback_k=0.85, size_boost=0.16, delay_scale=1.5, hold_scale=0.35, exit_aggr=0.90, skip_hazard_th=0.90),
    ]


def _refine_grid(best: PlaybookMetaConfig) -> list[PlaybookMetaConfig]:
    out: list[PlaybookMetaConfig] = []
    for de in (-0.08, 0.0, 0.08):
        for dh in (-0.08, 0.0, 0.08):
            for ds in (-0.02, 0.0, 0.02):
                for dd in (-0.5, 0.0, 0.5):
                    cfg = replace(
                        best,
                        event_k=round(max(0.70, best.event_k + de), 3),
                        hazard_k=round(max(0.80, best.hazard_k + dh), 3),
                        size_boost=round(_bounded(best.size_boost + ds, 0.06, 0.20), 3),
                        delay_scale=round(_bounded(best.delay_scale + dd, 1.0, 3.5), 3),
                    )
                    out.append(cfg)
    uniq: dict[str, PlaybookMetaConfig] = {cfg.name: cfg for cfg in out}
    return list(uniq.values())


def _bounded(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))


def _run_grid(feat_kst: pd.DataFrame, trades, grid: list[PlaybookMetaConfig], desc: str) -> list[dict]:
    results: list[dict] = []
    iterator = grid
    if tqdm is not None:
        iterator = tqdm(grid, desc=desc, ncols=110)
    for cfg in iterator:
        results.append(run_backtest(feat_kst, trades, cfg))
    results.sort(
        key=lambda x: (
            x["objective"],
            x["delta_pct"],
            x["overlay_sum_pct"],
            x["overlay_tail20_pct"],
            x["overlay_mdd_pct"],
        ),
        reverse=True,
    )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest playbook meta-controller using live trade history + duckdb features.")
    ap.add_argument("--events-path", default="data/live/dashboard_events.jsonl")
    ap.add_argument("--out-json", default="data/ensemble/reports/playbook_meta_controller_backtest_20260425.json")
    args = ap.parse_args()

    feat_kst, start_utc, end_utc = _load_duckdb_features()
    trades = _load_trades(args.events_path, start_utc=start_utc, end_utc=end_utc)
    if not trades:
        raise SystemExit("No overlapping trades found for duckdb window.")

    coarse = _run_grid(feat_kst, trades, _coarse_grid(), desc="playbook-coarse")
    refine = _run_grid(feat_kst, trades, _refine_grid(PlaybookMetaConfig(**coarse[0]["config"])), desc="playbook-refine")

    summary = {
        "window": {
            "duckdb_start_kst": str(start_utc.tz_convert(KST)),
            "duckdb_end_kst": str(end_utc.tz_convert(KST)),
        },
        "trade_count": len(trades),
        "coarse_top3": coarse[:3],
        "refine_top5": refine[:5],
        "best_overall": refine[0],
        "all_coarse": coarse,
        "all_refine": refine,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    best = refine[0]
    print("=== Playbook Meta Controller Backtest ===")
    print(f"window={summary['window']['duckdb_start_kst']} -> {summary['window']['duckdb_end_kst']}")
    print(
        f"best={best['name']} overlay_sum={best['overlay_sum_pct']:+.4f}% "
        f"delta={best['delta_pct']:+.4f}%p objective={best['objective']:+.4f} "
        f"tail20={best['overlay_tail20_pct']:+.4f}% mdd={best['overlay_mdd_pct']:+.4f}%"
    )


if __name__ == "__main__":
    main()
