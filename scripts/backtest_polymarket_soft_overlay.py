#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_polymarket_news_overlay import KST, _est_lev, _load_duckdb_features, _load_trades, _mdd, _sharpe


@dataclass(frozen=True)
class SoftConfig:
    veto_gap_th: float
    veto_tail_th: float
    veto_entropy_cap: float
    size_gap_th: float
    size_tail_th: float
    shock_th: float
    aftershock_cap: float
    toxicity_cap: float
    adverse_mult: float
    neutral_mult: float

    @property
    def name(self) -> str:
        return (
            f"vg{self.veto_gap_th:.4f}_sg{self.size_gap_th:.4f}_sh{self.shock_th:.2f}"
            f"_am{self.adverse_mult:.2f}_nm{self.neutral_mult:.2f}"
        )


def _entry_veto(row: pd.Series, side: str, cfg: SoftConfig) -> tuple[bool, str]:
    gap = float(row["target_gap"])
    entropy = float(row["entropy"])
    tail_up = float(row["tail_up_prob"])
    tail_down = float(row["tail_down_prob"])
    shock = float(row["shock_score"])
    if side == "LONG" and gap <= -cfg.veto_gap_th and tail_down >= cfg.veto_tail_th and entropy >= cfg.veto_entropy_cap:
        return True, "poly_veto_long"
    if side == "SHORT" and gap >= cfg.veto_gap_th and tail_up >= cfg.veto_tail_th and entropy >= cfg.veto_entropy_cap:
        return True, "poly_veto_short"
    if shock >= max(cfg.shock_th, 0.28) and abs(gap) <= cfg.veto_gap_th * 0.5 and entropy >= cfg.veto_entropy_cap:
        return True, "poly_veto_uncertain"
    return False, ""


def _size_multiplier(row: pd.Series, side: str, cfg: SoftConfig) -> tuple[float, str]:
    gap = float(row["target_gap"])
    tail_up = float(row["tail_up_prob"])
    tail_down = float(row["tail_down_prob"])
    shock = float(row["shock_score"])
    aftershock = float(row["aftershock"])
    toxicity = float(row["toxicity"])
    entropy = float(row["entropy"])
    adverse = (
        side == "LONG" and gap <= -cfg.size_gap_th and tail_down >= cfg.size_tail_th
    ) or (
        side == "SHORT" and gap >= cfg.size_gap_th and tail_up >= cfg.size_tail_th
    )
    if adverse and shock >= cfg.shock_th and (aftershock >= cfg.aftershock_cap or toxicity >= cfg.toxicity_cap):
        return cfg.adverse_mult, "poly_adverse_scale"
    if shock >= cfg.shock_th and abs(gap) <= cfg.size_gap_th * 0.6 and entropy >= 0.80:
        return cfg.neutral_mult, "poly_neutral_scale"
    return 1.0, ""


def run_backtest(feat_kst: pd.DataFrame, trades, cfg: SoftConfig) -> dict:
    feat = feat_kst.copy()
    feat["ts_utc"] = feat["ts"].dt.tz_convert("UTC")
    feat = feat.set_index("ts_utc")
    base_sum = float(sum(float(t.realized_pct) for t in trades))
    base_wins = int(sum(1 for t in trades if float(t.realized_pct) > 0.0))

    rets: list[float] = []
    eq = [1.0]
    vetoes = 0
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
        veto, _ = _entry_veto(open_state, tr.side, cfg)
        if veto:
            pnl = 0.0
            vetoes += 1
        else:
            path = feat.loc[(feat.index > tr.open_ts) & (feat.index <= tr.close_ts)]
            mult = 1.0
            for _, row in path.iterrows():
                mult, reason = _size_multiplier(row, tr.side, cfg)
                if mult < 0.999:
                    downsized += 1
                    break
            lev = _est_lev(tr, fee=0.0005, slip=0.0002)
            pnl = float(tr.realized_pct) * mult
            if lev > 0 and mult < 1.0:
                pnl = float(tr.realized_pct) * mult
        rets.append(pnl)
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
        "vetoes": vetoes,
        "downsized": downsized,
        "improved_trades": improved,
        "worsened_trades": worsened,
        "overlay_mdd_pct": _mdd(eq),
        "overlay_sharpe": _sharpe(rets),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest polymarket soft overlay: entry veto + size-down.")
    ap.add_argument("--events-path", default="data/live/dashboard_events.jsonl")
    ap.add_argument("--out-json", default="data/ensemble/reports/polymarket_soft_overlay_backtest_20260424.json")
    args = ap.parse_args()

    feat_kst, start_utc, end_utc = _load_duckdb_features()
    trades = _load_trades(args.events_path, start_utc=start_utc, end_utc=end_utc)
    if not trades:
        raise SystemExit("No overlapping trades found for duckdb window.")

    grid = [
        SoftConfig(*vals)
        for vals in product(
            [0.0030, 0.0045, 0.0060],
            [0.55, 0.62],
            [0.80, 0.88],
            [0.0030, 0.0045, 0.0060],
            [0.52, 0.58],
            [0.18, 0.24, 0.30],
            [0.45, 0.60],
            [0.80, 1.00],
            [0.50, 0.65, 0.80],
            [0.75, 0.85, 0.95],
        )
    ]

    results = []
    iterator = grid
    if tqdm is not None:
        iterator = tqdm(grid, desc="soft-overlay-grid", ncols=100)
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
    print("=== Polymarket Soft Overlay Backtest ===")
    print(f"window={summary['window']['duckdb_start_kst']} -> {summary['window']['duckdb_end_kst']}")
    print(f"trades={len(trades)} baseline_sum={summary['baseline_sum_pct']:+.4f}%")
    print(
        "best="
        f"{best['name']} overlay_sum={best['overlay_sum_pct']:+.4f}% "
        f"delta={best['delta_pct']:+.4f}%p wr={best['overlay_wr']:.2f}% "
        f"mdd={best['overlay_mdd_pct']:.2f}% vetoes={best['vetoes']} downsized={best['downsized']}"
    )
    print(f"report={out_path}")


if __name__ == "__main__":
    main()
