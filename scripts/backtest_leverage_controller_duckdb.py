#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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


@dataclass
class Result:
    profile: str
    pnl_pct: float
    mdd_pct: float
    trades: int
    wr_pct: float
    avg_leverage: float
    boosted: int
    reverted: int


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _add_features(feat: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    out["cons_long"] = (
        0.28 * np.tanh(out["target_gap"] / 0.0038)
        + 0.16 * out["tail_bias"]
        + 0.14 * np.tanh(out["taker_edge"] / 0.45)
        + 0.12 * np.tanh(out["obi_norm"] / 0.45)
        + 0.10 * np.tanh(out["whale_flow"] / 0.45)
        + 0.10 * np.tanh(out["liq_imbalance"] / 0.45)
        + 0.10 * np.tanh(out["absorption"] / 0.70)
    ).clip(-1.0, 1.0)
    out["cons_short"] = (
        0.28 * np.tanh(-out["target_gap"] / 0.0038)
        + 0.16 * (-out["tail_bias"])
        + 0.14 * np.tanh(-out["taker_edge"] / 0.45)
        + 0.12 * np.tanh(-out["obi_norm"] / 0.45)
        + 0.10 * np.tanh(-out["whale_flow"] / 0.45)
        + 0.10 * np.tanh(-out["liq_imbalance"] / 0.45)
        + 0.10 * np.tanh(out["absorption"] / 0.70)
    ).clip(-1.0, 1.0)
    out["risk_drag"] = (
        0.36 * np.tanh(out["toxicity"] / 0.75)
        + 0.28 * np.tanh(out["aftershock"] / 0.55)
        + 0.18 * np.tanh(out["queue_penalty"] / 0.80)
        + 0.10 * (1.0 - out["mode_prob"].clip(0.0, 1.0))
        + 0.08 * (out["shadow_regime_tag"].astype(str).str.contains("whipsaw|chop", case=False, na=False)).astype(float)
    ).clip(0.0, 1.5)
    return out


def _controller_convex_like(row: pd.Series, side: str, dd: float) -> float:
    cons = float(row["cons_long"] if side == "LONG" else row["cons_short"])
    conf = float(row.get("mode_prob", 0.0) or 0.0)
    risk_drag = float(row["risk_drag"])
    raw = 1.15 * cons + 0.65 * conf - 0.90 * risk_drag - 0.85 * min(dd / 0.05, 1.5)
    lev = 1.0 + 1.10 * _sigmoid(raw + 0.05)
    lev = min(2.0, lev)
    if dd >= 0.03:
        lev = min(lev, 1.40)
    if risk_drag >= 0.85:
        lev = min(lev, 1.08)
    return float(np.clip(lev, 1.0, 2.0))


def _controller_duckdb(row: pd.Series, side: str, dd: float) -> float:
    cons = float(row["cons_long"] if side == "LONG" else row["cons_short"])
    risk_drag = float(row["risk_drag"])
    aftershock = float(row["aftershock"])
    toxicity = float(row["toxicity"])
    mode_prob = float(row["mode_prob"])
    bonus = 0.0
    if cons >= 0.42 and mode_prob >= 0.44 and aftershock <= 0.35 and toxicity <= 0.45:
        bonus += 0.12
    if cons <= 0.10 and risk_drag >= 0.65:
        bonus -= 0.18
    raw = 1.25 * cons + 0.70 * mode_prob - 1.05 * risk_drag - 0.95 * min(dd / 0.05, 1.5) + bonus
    lev = 1.0 + 1.12 * _sigmoid(raw)
    lev = min(2.0, lev)
    if dd >= 0.025:
        lev = min(lev, 1.35)
    if toxicity >= 0.75 or aftershock >= 0.70:
        lev = 1.0
    return float(np.clip(lev, 1.0, 2.0))


def _simulate(feat: pd.DataFrame, px_utc: pd.DataFrame, trades, profile: str) -> dict:
    balance = 1.0
    eq = [1.0]
    pnls: list[float] = []
    levs: list[float] = []
    boosted = 0
    reverted = 0
    wins = 0
    peak = 1.0

    iterator = trades
    if tqdm is not None:
        iterator = tqdm(trades, desc=f"{profile}-trades", ncols=100)
    for tr in iterator:
        entry_slice = feat.loc[: tr.open_ts]
        if len(entry_slice) == 0:
            pnl = float(tr.realized_pct)
            levs.append(_est_lev(tr, 0.0005, 0.0002))
            pnls.append(pnl)
            wins += int(pnl > 0)
            balance *= 1.0 + pnl / 100.0
            eq.append(balance)
            peak = max(peak, balance)
            continue

        entry_row = entry_slice.iloc[-1]
        dd = 1.0 - balance / max(peak, 1e-8)
        if profile == "controller_convex_live":
            target_lev = _controller_convex_like(entry_row, tr.side, dd)
        elif profile == "controller_convex_duckdb":
            target_lev = _controller_duckdb(entry_row, tr.side, dd)
        else:
            raise ValueError(profile)

        base_lev = _est_lev(tr, 0.0005, 0.0002)
        applied_lev = max(0.0, base_lev * target_lev)
        if target_lev > 1.02:
            boosted += 1

        pnl = float(tr.realized_pct)
        path = feat.loc[(feat.index > tr.open_ts) & (feat.index <= tr.close_ts)]
        reverted_here = False
        if len(path):
            for ts, row in path.iterrows():
                trigger_px = _asof_price(px_utc, ts)
                if trigger_px is None:
                    continue
                cons = float(row["cons_long"] if tr.side == "LONG" else row["cons_short"])
                risk_drag = float(row["risk_drag"])
                if risk_drag >= 0.85 or cons <= 0.05:
                    first = _net_frac(tr.side, tr.open_price, trigger_px, applied_lev, 0.0005, 0.0002) * 100.0
                    second = _net_frac(tr.side, trigger_px, tr.close_price, base_lev, 0.0005, 0.0002) * 100.0
                    pnl = first + second
                    reverted += 1
                    reverted_here = True
                    break
        if not reverted_here:
            pnl = _net_frac(tr.side, tr.open_price, tr.close_price, applied_lev, 0.0005, 0.0002) * 100.0

        levs.append(applied_lev)
        pnls.append(pnl)
        wins += int(pnl > 0)
        balance *= 1.0 + pnl / 100.0
        eq.append(balance)
        peak = max(peak, balance)

    res = Result(
        profile=profile,
        pnl_pct=float(sum(pnls)),
        mdd_pct=_mdd(eq),
        trades=len(trades),
        wr_pct=float(100.0 * wins / max(len(trades), 1)),
        avg_leverage=float(np.mean(levs) if levs else 0.0),
        boosted=boosted,
        reverted=reverted,
    )
    return asdict(res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="/home/kbj20/crypto-scalping/data/ensemble/reports/backtest_leverage_controller_duckdb_20260424.json")
    args = ap.parse_args()

    feat, px_utc, trades = _prepare_features()
    feat = _add_features(feat)

    results = [
        _simulate(feat, px_utc, trades, "controller_convex_live"),
        _simulate(feat, px_utc, trades, "controller_convex_duckdb"),
    ]

    payload = {
        "trade_count": len(trades),
        "results": results,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
