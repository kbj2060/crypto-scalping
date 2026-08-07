#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, os.path.join(_ROOT_DIR, "ensemble")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scripts.eval_2026_dsac_limit as base


CSV_PATH = "data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv"
CKPT = "data/ensemble/ckpt/best_dsac_agents.pth"
OUT_JSON = "data/ensemble/reports/eval_2026_dsac_partial_limit_best_currentsplit.json"


@dataclass
class PartialLimitConfig:
    name: str
    market_share: float
    limit_share: float
    entry_scale: float
    volatility_mult: float
    liquidity_mult: float
    model_anchor: float
    entry_ttl: int
    fallback_conf: float
    fallback_edge: float


def simulate_partial_limit(df, actor, device: str, cfg: PartialLimitConfig) -> dict:
    numeric_cols = [c for c in df.columns if c != "timestamp"]
    values = df[numeric_cols].to_numpy(dtype=np.float64)
    open_np = df["open"].to_numpy(dtype=np.float64)
    high_np = df["high"].to_numpy(dtype=np.float64)
    low_np = df["low"].to_numpy(dtype=np.float64)
    close_np = df["close"].to_numpy(dtype=np.float64)

    router = base.DSACRouter(actor, device=device)
    balance = 1.0
    pos: str | None = None
    market_entry_price = 0.0
    market_lev = 0.0
    limit_entry_price = 0.0
    limit_lev = 0.0
    hold_count = 0
    trades = wins = 0
    maker_entries = fallback_entries = market_entries = missed_entries = 0
    eq_curve = [1.0]
    pending: dict | None = None

    def _entry_fee_total() -> float:
        fee = 0.0
        if market_lev > 0.0:
            fee += base.TAKER_FEE * market_lev
        if limit_lev > 0.0:
            fee += base.MAKER_FEE * limit_lev
        return float(fee)

    def _unrealized(cp: float) -> float:
        total = 0.0
        if pos is None:
            return 0.0
        if market_lev > 0.0 and market_entry_price > 0.0:
            if pos == "LONG":
                gross = (cp * (1.0 - base.TAKER_SLIP) - market_entry_price) / market_entry_price
            else:
                gross = (market_entry_price - cp * (1.0 + base.TAKER_SLIP)) / market_entry_price
            total += gross * market_lev
        if limit_lev > 0.0 and limit_entry_price > 0.0:
            if pos == "LONG":
                gross = (cp - limit_entry_price) / limit_entry_price
            else:
                gross = (limit_entry_price - cp) / limit_entry_price
            total += gross * limit_lev
        total -= _entry_fee_total()
        total -= base.MAKER_FEE * limit_lev
        total -= base.TAKER_FEE * market_lev
        return float(total)

    def _close_realized(side: str, exit_price_market: float) -> float:
        total = 0.0
        if market_lev > 0.0 and market_entry_price > 0.0:
            total += base._realized_return(
                side,
                market_entry_price,
                exit_price_market,
                market_lev,
                base.TAKER_FEE,
                base.TAKER_FEE,
            )
        if limit_lev > 0.0 and limit_entry_price > 0.0:
            total += base._realized_return(
                side,
                limit_entry_price,
                exit_price_market,
                limit_lev,
                base.MAKER_FEE,
                base.TAKER_FEE,
            )
        return float(total)

    for i in range(1, len(df) - 1):
        bar_open = float(open_np[i])
        bar_high = float(high_np[i])
        bar_low = float(low_np[i])
        bar_close = float(close_np[i])

        if pos is not None:
            hold_count += 1

        if pending is not None and limit_lev <= 0.0:
            should_fill = (pending["side"] == "LONG" and bar_low <= pending["price"]) or (
                pending["side"] == "SHORT" and bar_high >= pending["price"]
            )
            if should_fill:
                limit_entry_price = float(pending["price"])
                limit_lev = float(pending["lev"])
                balance -= balance * base.MAKER_FEE * limit_lev
                maker_entries += 1
                pending = None
            elif i > pending["expire_idx"]:
                if pending.get("allow_fallback", False):
                    limit_entry_price = (
                        bar_open * (1.0 + base.TAKER_SLIP)
                        if pending["side"] == "LONG"
                        else bar_open * (1.0 - base.TAKER_SLIP)
                    )
                    limit_lev = float(pending["lev"])
                    balance -= balance * base.TAKER_FEE * limit_lev
                    fallback_entries += 1
                else:
                    missed_entries += 1
                pending = None

        cp = float(close_np[i - 1])
        pos_dict = {
            "type": pos,
            "entry_price": float(market_entry_price or limit_entry_price or 0.0),
            "unrealized": float(_unrealized(cp)),
            "mdd": 0.0,
            "hold_norm": float(min(hold_count / 96.0, 1.0)),
            "margin_usage": float((market_lev + limit_lev) if pos else 0.0),
            "hold_count": float(hold_count),
        }
        features = {k: float(v) for k, v in zip(numeric_cols, values[i - 1])}
        action_int, lev, _ = router.decide(features, pos_dict)
        lev = float(np.clip(lev, 0.0, 1.0))
        row = df.iloc[i - 1]

        if pos is None and pending is None:
            if action_int in (1, 2) and lev > 0.0:
                side = "LONG" if action_int == 1 else "SHORT"
                pos = side
                hold_count = 0
                market_lev = float(lev * cfg.market_share)
                limit_target_lev = float(lev * cfg.limit_share)
                if market_lev > 0.0:
                    market_entry_price = (
                        bar_open * (1.0 + base.TAKER_SLIP)
                        if side == "LONG"
                        else bar_open * (1.0 - base.TAKER_SLIP)
                    )
                    balance -= balance * base.TAKER_FEE * market_lev
                    market_entries += 1
                else:
                    market_entry_price = 0.0
                limit_entry_price = 0.0
                limit_lev = 0.0
                if limit_target_lev > 0.0:
                    plan = base._compute_entry_plan(
                        row,
                        side,
                        cp,
                        limit_target_lev,
                        base.LimitConfig(
                            name=cfg.name,
                            entry_scale=cfg.entry_scale,
                            volatility_mult=cfg.volatility_mult,
                            liquidity_mult=cfg.liquidity_mult,
                            model_anchor=cfg.model_anchor,
                            exit_buffer_bps=0.0,
                            entry_ttl=cfg.entry_ttl,
                            exit_ttl=0,
                            fallback_conf=cfg.fallback_conf,
                            fallback_edge=cfg.fallback_edge,
                        ),
                    )
                    pending = {
                        "kind": "entry",
                        "side": side,
                        "price": float(plan["limit_price"]),
                        "expire_idx": i - 1 + int(plan["ttl"]),
                        "lev": limit_target_lev,
                        "allow_fallback": bool(plan["allow_fallback"]),
                    }
        elif pos is not None:
            should_close = action_int == 0 or (action_int == 1 and pos == "SHORT") or (action_int == 2 and pos == "LONG")
            if should_close:
                exit_price = bar_open * (1.0 - base.TAKER_SLIP) if pos == "LONG" else bar_open * (1.0 + base.TAKER_SLIP)
                realized = _close_realized(pos, exit_price)
                balance *= 1.0 + realized
                trades += 1
                wins += int(realized > 0.0)
                pos = None
                market_entry_price = 0.0
                market_lev = 0.0
                limit_entry_price = 0.0
                limit_lev = 0.0
                pending = None
                hold_count = 0

        eq_curve.append(max(balance * (1.0 + _unrealized(bar_close)) if pos else balance, 1e-8))

    if pos is not None:
        last_close = float(close_np[-1])
        exit_price = last_close * (1.0 - base.TAKER_SLIP) if pos == "LONG" else last_close * (1.0 + base.TAKER_SLIP)
        realized = _close_realized(pos, exit_price)
        balance *= 1.0 + realized
        trades += 1
        wins += int(realized > 0.0)

    return {
        "method": "partial_limit",
        "config": asdict(cfg),
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "wr_pct": round((100.0 * wins / trades) if trades else 0.0, 2),
        "trades": trades,
        "sharpe": round(base._sharpe(eq_curve), 4),
        "mdd_pct": round(base._mdd(eq_curve), 4),
        "market_entries": int(market_entries),
        "maker_entries": int(maker_entries),
        "fallback_entries": int(fallback_entries),
        "missed_entries": int(missed_entries),
        "maker_entry_ratio": round(maker_entries / max(maker_entries + fallback_entries, 1), 4),
    }


def main() -> None:
    base.CSV_PATH = CSV_PATH
    base.CKPT = CKPT
    device = "cuda" if base.torch.cuda.is_available() else "cpu"
    df = base._load_df()
    actor = base._build_actor(device)

    baseline = base.simulate_market(df, base.DSACRouter(copy.deepcopy(actor), device=device))
    print("[BASELINE]", baseline)

    configs = [
        PartialLimitConfig("market90_limit10", 0.90, 0.10, 0.40, 0.25, 0.20, 0.88, 1, 0.65, 0.03),
        PartialLimitConfig("market85_limit15", 0.85, 0.15, 0.35, 0.25, 0.20, 0.90, 1, 0.62, 0.03),
        PartialLimitConfig("market80_limit20", 0.80, 0.20, 0.30, 0.20, 0.15, 0.92, 1, 0.60, 0.02),
        PartialLimitConfig("market70_limit30", 0.70, 0.30, 0.28, 0.20, 0.15, 0.92, 1, 0.58, 0.02),
    ]

    results = []
    for cfg in configs:
        result = simulate_partial_limit(df, copy.deepcopy(actor), device, cfg)
        result["delta_vs_market_pct"] = round(result["pnl_pct"] - baseline["pnl_pct"], 4)
        results.append(result)
        print(
            "[PARTIAL]",
            cfg.name,
            f"pnl={result['pnl_pct']:.2f}%",
            f"delta={result['delta_vs_market_pct']:+.2f}%",
            f"wr={result['wr_pct']:.2f}%",
            f"trades={result['trades']}",
            f"maker_entries={result['maker_entries']}",
            f"fallback_entries={result['fallback_entries']}",
            f"missed={result['missed_entries']}",
        )

    best = max(results, key=lambda x: (x["pnl_pct"], -abs(x["mdd_pct"])))
    report = {
        "checkpoint": CKPT,
        "data_period": f"{df['timestamp'].min()} -> {df['timestamp'].max()}",
        "data_rows": int(len(df)),
        "baseline_market": baseline,
        "tested_configs": results,
        "best_partial_limit": best,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[BEST]", best["config"]["name"], best)
    print("[SAVED]", OUT_JSON)


if __name__ == "__main__":
    main()
