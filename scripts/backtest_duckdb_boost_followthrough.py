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

from scripts.backtest_duckdb_dsac_synergy_overlay import _prepare_features
from scripts.backtest_polymarket_news_overlay import _asof_price, _est_lev, _mdd, _net_frac, _sharpe


@dataclass(frozen=True)
class FollowthroughConfig:
    consensus_th: float
    risk_cap: float
    mtm_gate: float
    min_elapsed_min: int
    confirm_bars: int
    boost_mult: float
    revert_risk: float
    revert_consensus: float

    @property
    def name(self) -> str:
        return (
            f"c{self.consensus_th:.2f}_r{self.risk_cap:.2f}_m{self.mtm_gate:.4f}"
            f"_e{self.min_elapsed_min}_k{self.confirm_bars}_b{self.boost_mult:.2f}"
        )


def _build_followthrough_features(feat: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    out["cons_long"] = (
        0.30 * np.tanh(out["target_gap"] / 0.0035)
        + 0.16 * out["tail_bias"]
        + 0.14 * np.tanh(out["taker_edge"] / 0.45)
        + 0.12 * np.tanh(out["obi_norm"] / 0.45)
        + 0.11 * np.tanh(out["whale_flow"] / 0.45)
        + 0.09 * np.tanh(out["liq_imbalance"] / 0.45)
        + 0.08 * np.tanh(out["absorption"] / 0.70)
        + 0.05 * np.tanh(out["mode_spread"] / 0.18)
    ).clip(-1.0, 1.0)
    out["cons_short"] = (
        0.30 * np.tanh(-out["target_gap"] / 0.0035)
        + 0.16 * (-out["tail_bias"])
        + 0.14 * np.tanh(-out["taker_edge"] / 0.45)
        + 0.12 * np.tanh(-out["obi_norm"] / 0.45)
        + 0.11 * np.tanh(-out["whale_flow"] / 0.45)
        + 0.09 * np.tanh(-out["liq_imbalance"] / 0.45)
        + 0.08 * np.tanh(out["absorption"] / 0.70)
        + 0.05 * np.tanh(out["mode_spread"] / 0.18)
    ).clip(-1.0, 1.0)
    out["risk_drag"] = (
        0.42 * np.tanh(out["toxicity"] / 0.75)
        + 0.32 * np.tanh(out["aftershock"] / 0.55)
        + 0.14 * np.tanh(out["queue_penalty"] / 0.80)
        + 0.12 * (1.0 - out["mode_prob"].clip(0.0, 1.0))
    ).clip(0.0, 1.5)
    return out


def _segment_return(side: str, start_px: float, end_px: float, lev: float, mult: float) -> float:
    return _net_frac(side, start_px, end_px, lev * mult, fee=0.0005, slip=0.0002) * 100.0


def run_backtest(feat: pd.DataFrame, px_utc: pd.DataFrame, trades, cfg: FollowthroughConfig) -> dict:
    baseline = float(sum(float(t.realized_pct) for t in trades))
    base_wins = int(sum(1 for t in trades if float(t.realized_pct) > 0.0))
    rets: list[float] = []
    eq = [1.0]
    boosted = 0
    reverted = 0
    improved = 0
    worsened = 0

    for tr in trades:
        lev = _est_lev(tr, fee=0.0005, slip=0.0002)
        pnl = float(tr.realized_pct)
        path = feat.loc[(feat.index > tr.open_ts) & (feat.index <= tr.close_ts)]
        cons_col = "cons_long" if tr.side == "LONG" else "cons_short"
        boost_ts = None
        revert_ts = None
        streak = 0

        for ts, row in path.iterrows():
            elapsed = int((ts - tr.open_ts).total_seconds() // 60)
            trigger_px = _asof_price(px_utc, ts)
            if trigger_px is None:
                continue
            mtm = ((trigger_px - tr.open_price) / tr.open_price) if tr.side == "LONG" else ((tr.open_price - trigger_px) / tr.open_price)
            consensus = float(row[cons_col])
            risk_drag = float(row["risk_drag"])
            mode_prob = float(row["mode_prob"])

            if boost_ts is None:
                if (
                    elapsed >= cfg.min_elapsed_min
                    and mtm >= cfg.mtm_gate
                    and consensus >= cfg.consensus_th
                    and risk_drag <= cfg.risk_cap
                    and mode_prob >= 0.36
                ):
                    streak += 1
                else:
                    streak = 0
                if streak >= cfg.confirm_bars:
                    boost_ts = ts
                    boosted += 1
                    continue
            else:
                if risk_drag >= cfg.revert_risk or consensus <= cfg.revert_consensus:
                    revert_ts = ts
                    reverted += 1
                    break

        if boost_ts is not None:
            boost_px = _asof_price(px_utc, boost_ts)
            if boost_px is not None:
                first = _segment_return(tr.side, tr.open_price, boost_px, lev, 1.0)
                if revert_ts is not None:
                    revert_px = _asof_price(px_utc, revert_ts)
                    if revert_px is not None:
                        second = _segment_return(tr.side, boost_px, revert_px, lev, cfg.boost_mult)
                        third = _segment_return(tr.side, revert_px, tr.close_price, lev, 1.0)
                        pnl = first + second + third
                    else:
                        second = _segment_return(tr.side, boost_px, tr.close_price, lev, cfg.boost_mult)
                        pnl = first + second
                else:
                    second = _segment_return(tr.side, boost_px, tr.close_price, lev, cfg.boost_mult)
                    pnl = first + second

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
        "baseline_sum_pct": baseline,
        "overlay_sum_pct": total,
        "delta_pct": total - baseline,
        "baseline_wr": 100.0 * base_wins / max(len(trades), 1),
        "overlay_wr": 100.0 * wins / max(len(trades), 1),
        "boosted": boosted,
        "reverted": reverted,
        "improved_trades": improved,
        "worsened_trades": worsened,
        "overlay_mdd_pct": _mdd(eq),
        "overlay_sharpe": _sharpe(rets),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest followthrough boost overlay using duckdb consensus features.")
    ap.add_argument("--out-json", default="data/ensemble/reports/duckdb_boost_followthrough_20260424.json")
    args = ap.parse_args()

    feat, px_utc, trades = _prepare_features()
    feat = _build_followthrough_features(feat)
    if not trades:
        raise SystemExit("No overlapping live trades found.")

    grid = [
        FollowthroughConfig(*vals)
        for vals in product(
            [0.34, 0.42],
            [0.30, 0.40],
            [0.0005, 0.0010],
            [10, 15],
            [2, 3],
            [1.10, 1.20],
            [0.45, 0.55],
            [0.18, 0.24],
        )
    ]

    results = []
    iterator = grid
    if tqdm is not None:
        iterator = tqdm(grid, desc="followthrough-grid", ncols=100)
    for cfg in iterator:
        results.append(run_backtest(feat, px_utc, trades, cfg))
    results.sort(key=lambda x: (x["overlay_sum_pct"], x["delta_pct"], -x["overlay_mdd_pct"]), reverse=True)

    out = {
        "trade_count": len(trades),
        "baseline_sum_pct": results[0]["baseline_sum_pct"],
        "top10": results[:10],
        "all_results": results,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    best = results[0]
    print("=== DuckDB Boost Followthrough ===")
    print(f"trades={len(trades)} baseline_sum={out['baseline_sum_pct']:+.4f}%")
    print(
        f"best={best['name']} overlay_sum={best['overlay_sum_pct']:+.4f}% "
        f"delta={best['delta_pct']:+.4f}%p wr={best['overlay_wr']:.2f}% "
        f"mdd={best['overlay_mdd_pct']:.2f}% boosted={best['boosted']} reverted={best['reverted']}"
    )
    print(f"report={out_path}")


if __name__ == "__main__":
    main()
