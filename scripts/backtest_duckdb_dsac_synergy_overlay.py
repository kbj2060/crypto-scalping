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

from scripts.backtest_polymarket_news_overlay import (
    KST,
    _asof_price,
    _est_lev,
    _fetch_binance_1m,
    _load_duckdb_features,
    _load_trades,
    _mdd,
    _net_frac,
    _sharpe,
)


@dataclass(frozen=True)
class SynergyConfig:
    veto_th: float
    adverse_th: float
    favorable_th: float
    shock_th: float
    up_mult: float
    down_mult: float
    exit_loss_guard: float
    aftershock_cap: float
    toxicity_cap: float
    min_mode_prob: float

    @property
    def name(self) -> str:
        return (
            f"vt{self.veto_th:.2f}_at{self.adverse_th:.2f}_ft{self.favorable_th:.2f}"
            f"_up{self.up_mult:.2f}_dn{self.down_mult:.2f}"
        )


def _prepare_features() -> tuple[pd.DataFrame, pd.DataFrame, list]:
    feat_kst, start_utc, end_utc = _load_duckdb_features()
    trades = _load_trades("data/live/dashboard_events.jsonl", start_utc=start_utc, end_utc=end_utc)
    px_utc = _fetch_binance_1m(start_utc - pd.Timedelta(minutes=5), end_utc + pd.Timedelta(minutes=5))

    feat = feat_kst.copy()
    feat["liq_imbalance"] = (
        (pd.to_numeric(feat["short_usd_1m"], errors="coerce").fillna(0.0) - pd.to_numeric(feat["long_usd_1m"], errors="coerce").fillna(0.0))
        / (
            pd.to_numeric(feat["short_usd_1m"], errors="coerce").fillna(0.0)
            + pd.to_numeric(feat["long_usd_1m"], errors="coerce").fillna(0.0)
            + 1e-9
        )
    ).clip(-1.0, 1.0)
    feat["whale_flow"] = np.tanh(pd.to_numeric(feat["nif_whale"], errors="coerce").fillna(0.0) / 0.35)
    feat["taker_edge"] = (pd.to_numeric(feat["taker_buy_ratio"], errors="coerce").fillna(0.5) - 0.5) * 2.0
    feat["queue_penalty"] = pd.to_numeric(feat["shadow_queue_collapse"], errors="coerce").fillna(0.0).clip(0.0, 1.5)
    feat["absorption"] = pd.to_numeric(feat["shadow_absorption_score"], errors="coerce").fillna(0.0).clip(0.0, 1.5)
    feat["toxicity"] = pd.to_numeric(feat["shadow_toxicity_score"], errors="coerce").fillna(0.0).clip(0.0, 1.5)
    feat["aftershock"] = pd.to_numeric(feat["shadow_aftershock_prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    feat["regime_conf"] = pd.to_numeric(feat["shadow_regime_conf"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    feat["mode_prob"] = pd.to_numeric(feat["mode_prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    feat["signal_bias_norm"] = pd.to_numeric(feat["signal_bias"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)

    def _stack_scores(df: pd.DataFrame, side: str) -> tuple[pd.Series, pd.Series]:
        sgn = 1.0 if side == "LONG" else -1.0
        fav = (
            0.24 * np.tanh((sgn * df["target_gap"]) / 0.0040)
            + 0.15 * (sgn * df["tail_bias"])
            + 0.14 * np.tanh((sgn * df["taker_edge"]) / 0.55)
            + 0.12 * np.tanh((sgn * df["obi_norm"]) / 0.55)
            + 0.11 * np.tanh((sgn * df["whale_flow"]) / 0.55)
            + 0.08 * np.tanh((sgn * df["liq_imbalance"]) / 0.55)
            + 0.07 * np.tanh(df["absorption"] / 0.8)
            + 0.05 * np.tanh((sgn * df["signal_bias_norm"]) / 0.8)
            + 0.04 * np.tanh(df["mode_spread"] / 0.20)
            - 0.08 * np.tanh(df["toxicity"] / 0.8)
            - 0.06 * np.tanh(df["aftershock"] / 0.6)
            - 0.04 * np.tanh(df["queue_penalty"] / 0.8)
        )
        adv = (
            0.24 * np.tanh((-sgn * df["target_gap"]) / 0.0040)
            + 0.15 * (-sgn * df["tail_bias"])
            + 0.12 * np.tanh((-sgn * df["taker_edge"]) / 0.55)
            + 0.10 * np.tanh((-sgn * df["obi_norm"]) / 0.55)
            + 0.10 * np.tanh((-sgn * df["whale_flow"]) / 0.55)
            + 0.08 * np.tanh((-sgn * df["liq_imbalance"]) / 0.55)
            + 0.08 * np.tanh(df["toxicity"] / 0.8)
            + 0.07 * np.tanh(df["aftershock"] / 0.6)
            + 0.04 * np.tanh(df["queue_penalty"] / 0.8)
            - 0.04 * np.tanh(df["absorption"] / 0.8)
        )
        return fav.clip(-1.0, 1.0), adv.clip(-1.0, 1.0)

    feat["fav_long"], feat["adv_long"] = _stack_scores(feat, "LONG")
    feat["fav_short"], feat["adv_short"] = _stack_scores(feat, "SHORT")
    feat["ts_utc"] = feat["ts"].dt.tz_convert("UTC")
    return feat.set_index("ts_utc"), px_utc, trades


def _segment_return(side: str, start_px: float, end_px: float, lev: float, exposure_mult: float) -> float:
    return _net_frac(side, start_px, end_px, lev * exposure_mult, fee=0.0005, slip=0.0002) * 100.0


def _mark_to_trigger(side: str, entry_px: float, trigger_px: float) -> float:
    if side == "LONG":
        return (trigger_px - entry_px) / max(entry_px, 1e-12)
    return (entry_px - trigger_px) / max(entry_px, 1e-12)


def run_backtest(feat: pd.DataFrame, px_utc: pd.DataFrame, trades, cfg: SynergyConfig) -> dict:
    baseline = float(sum(float(t.realized_pct) for t in trades))
    base_wins = int(sum(1 for t in trades if float(t.realized_pct) > 0.0))
    rets: list[float] = []
    eq = [1.0]
    vetoes = 0
    scaled_up = 0
    scaled_down = 0
    emergency_exits = 0
    improved = 0
    worsened = 0

    for tr in trades:
        lev = _est_lev(tr, fee=0.0005, slip=0.0002)
        open_slice = feat.loc[: tr.open_ts]
        if len(open_slice) == 0:
            pnl = float(tr.realized_pct)
            rets.append(pnl)
            eq.append(eq[-1] * (1.0 + pnl / 100.0))
            continue
        open_row = open_slice.iloc[-1]
        fav_col = "fav_long" if tr.side == "LONG" else "fav_short"
        adv_col = "adv_long" if tr.side == "LONG" else "adv_short"
        if (
            float(open_row[adv_col]) >= cfg.veto_th
            and float(open_row["shock_score"]) >= cfg.shock_th
            and float(open_row["mode_prob"]) >= cfg.min_mode_prob
        ):
            pnl = 0.0
            vetoes += 1
        else:
            pnl = float(tr.realized_pct)
            path = feat.loc[(feat.index > tr.open_ts) & (feat.index <= tr.close_ts)]
            changed = False
            for ts, row in path.iterrows():
                trigger_px = _asof_price(px_utc, ts)
                if trigger_px is None:
                    continue
                mtm = _mark_to_trigger(tr.side, tr.open_price, trigger_px)
                adverse = (
                    float(row[adv_col]) >= cfg.adverse_th
                    and float(row["shock_score"]) >= cfg.shock_th
                    and (float(row["aftershock"]) >= cfg.aftershock_cap or float(row["toxicity"]) >= cfg.toxicity_cap)
                    and float(row["mode_prob"]) >= cfg.min_mode_prob
                )
                favorable = (
                    float(row[fav_col]) >= cfg.favorable_th
                    and float(row["shock_score"]) <= max(cfg.shock_th + 0.10, 0.32)
                    and float(row["mode_prob"]) >= cfg.min_mode_prob
                    and float(row["aftershock"]) <= cfg.aftershock_cap
                    and float(row["toxicity"]) <= cfg.toxicity_cap
                )
                if adverse and mtm <= cfg.exit_loss_guard:
                    pnl = _segment_return(tr.side, tr.open_price, trigger_px, lev, 1.0)
                    emergency_exits += 1
                    changed = True
                    break
                if adverse and mtm <= 0.0010:
                    first = _segment_return(tr.side, tr.open_price, trigger_px, lev, 1.0)
                    second = _segment_return(tr.side, trigger_px, tr.close_price, lev, cfg.down_mult)
                    pnl = first + second
                    scaled_down += 1
                    changed = True
                    break
                if favorable and mtm >= -0.0005:
                    first = _segment_return(tr.side, tr.open_price, trigger_px, lev, 1.0)
                    second = _segment_return(tr.side, trigger_px, tr.close_price, lev, cfg.up_mult)
                    pnl = first + second
                    scaled_up += 1
                    changed = True
                    break
            if not changed:
                pnl = float(tr.realized_pct)
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
        "vetoes": vetoes,
        "scaled_up": scaled_up,
        "scaled_down": scaled_down,
        "emergency_exits": emergency_exits,
        "improved_trades": improved,
        "worsened_trades": worsened,
        "overlay_mdd_pct": _mdd(eq),
        "overlay_sharpe": _sharpe(rets),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Explore duckdb synergy overlay for DSAC live trades.")
    ap.add_argument("--out-json", default="data/ensemble/reports/duckdb_dsac_synergy_overlay_20260424.json")
    args = ap.parse_args()

    feat, px_utc, trades = _prepare_features()
    if not trades:
        raise SystemExit("No overlapping live trades found.")

    grid = [
        SynergyConfig(*vals)
        for vals in product(
            [0.28, 0.36],
            [0.30, 0.38],
            [0.26, 0.34],
            [0.18, 0.24],
            [1.10, 1.25],
            [0.60, 0.80],
            [-0.0010, 0.0],
            [0.45, 0.60],
            [0.80, 1.00],
            [0.36, 0.48],
        )
    ]

    results = []
    iterator = grid
    if tqdm is not None:
        iterator = tqdm(grid, desc="synergy-grid", ncols=100)
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
    print("=== DuckDB DSAC Synergy Overlay ===")
    print(f"trades={len(trades)} baseline_sum={out['baseline_sum_pct']:+.4f}%")
    print(
        f"best={best['name']} overlay_sum={best['overlay_sum_pct']:+.4f}% "
        f"delta={best['delta_pct']:+.4f}%p wr={best['overlay_wr']:.2f}% "
        f"mdd={best['overlay_mdd_pct']:.2f}% up={best['scaled_up']} down={best['scaled_down']} "
        f"exit={best['emergency_exits']} veto={best['vetoes']}"
    )
    print(f"report={out_path}")


if __name__ == "__main__":
    main()
