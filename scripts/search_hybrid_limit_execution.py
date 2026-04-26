#!/usr/bin/env python3
from __future__ import annotations

import copy
import itertools
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, os.path.join(_ROOT_DIR, "ensemble")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scripts.eval_2026_dsac_limit as base


CSV_PATH = "data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv"
CKPT = "data/ensemble/ckpt/best_dsac_agents.pth"
OUT_JSON = "data/ensemble/reports/search_hybrid_limit_execution_2026.json"


@dataclass
class HybridConfig:
    name: str
    market_share: float
    maker_share: float
    offset_bps: float
    ttl_bars: int
    conf_th: float
    qwidth_max: float
    edge_th: float
    flow_th: float
    tox_max: float
    queue_max: float
    aftershock_max: float


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _load_baseline_trades(df, actor, device: str) -> tuple[dict, list[dict]]:
    numeric_cols = [c for c in df.columns if c != "timestamp"]
    values = df[numeric_cols].to_numpy(dtype=np.float64)
    open_np = df["open"].to_numpy(dtype=np.float64)
    close_np = df["close"].to_numpy(dtype=np.float64)

    router = base.DSACRouter(copy.deepcopy(actor), device=device)
    balance = 1.0
    pos: str | None = None
    entry_price = 0.0
    cur_lev = 0.0
    hold_count = 0
    trades: list[dict] = []
    wins = 0

    def _unrealized(current_price: float) -> float:
        if pos is None or entry_price <= 0.0 or cur_lev <= 0.0:
            return 0.0
        gross = (
            (current_price * (1.0 - base.TAKER_SLIP) - entry_price) / entry_price
            if pos == "LONG"
            else (entry_price - current_price * (1.0 + base.TAKER_SLIP)) / entry_price
        )
        return float(gross * cur_lev - (2.0 * base.TAKER_FEE * cur_lev))

    for i in range(len(df) - 1):
        cp = float(close_np[i])
        next_open = float(open_np[i + 1])
        if pos is not None:
            hold_count += 1

        pos_dict = {
            "type": pos,
            "entry_price": float(entry_price),
            "unrealized": float(_unrealized(cp)),
            "mdd": 0.0,
            "hold_norm": float(min(hold_count / 96.0, 1.0)),
            "margin_usage": float(cur_lev if pos else 0.0),
            "hold_count": float(hold_count),
        }
        features = {k: float(v) for k, v in zip(numeric_cols, values[i])}
        action_int, lev, _ = router.decide(features, pos_dict)
        lev = float(np.clip(lev, 0.0, 1.0))

        if pos is None:
            if action_int == 1 and lev > 0.0:
                pos = "LONG"
                entry_price = next_open * (1.0 + base.TAKER_SLIP)
                cur_lev = lev
                hold_count = 0
                signal_idx = i
                entry_idx = i + 1
                balance -= balance * base.TAKER_FEE * cur_lev
            elif action_int == 2 and lev > 0.0:
                pos = "SHORT"
                entry_price = next_open * (1.0 - base.TAKER_SLIP)
                cur_lev = lev
                hold_count = 0
                signal_idx = i
                entry_idx = i + 1
                balance -= balance * base.TAKER_FEE * cur_lev
        else:
            should_close = action_int == 0 or (action_int == 1 and pos == "SHORT") or (action_int == 2 and pos == "LONG")
            if should_close:
                exit_idx = i + 1
                exit_price = next_open * (1.0 - base.TAKER_SLIP) if pos == "LONG" else next_open * (1.0 + base.TAKER_SLIP)
                realized = base._realized_return(pos, entry_price, exit_price, cur_lev, base.TAKER_FEE, base.TAKER_FEE)
                balance *= 1.0 + realized
                wins += int(realized > 0.0)
                trades.append(
                    {
                        "side": pos,
                        "signal_idx": int(signal_idx),
                        "entry_idx": int(entry_idx),
                        "exit_idx": int(exit_idx),
                        "lev": float(cur_lev),
                        "baseline_entry": float(entry_price),
                        "baseline_exit": float(exit_price),
                        "baseline_realized": float(realized),
                    }
                )
                pos = None
                entry_price = 0.0
                cur_lev = 0.0
                hold_count = 0

    if pos is not None:
        exit_idx = len(df) - 1
        exit_price = float(close_np[-1]) * (1.0 - base.TAKER_SLIP) if pos == "LONG" else float(close_np[-1]) * (1.0 + base.TAKER_SLIP)
        realized = base._realized_return(pos, entry_price, exit_price, cur_lev, base.TAKER_FEE, base.TAKER_FEE)
        balance *= 1.0 + realized
        wins += int(realized > 0.0)
        trades.append(
            {
                "side": pos,
                "signal_idx": int(signal_idx),
                "entry_idx": int(entry_idx),
                "exit_idx": int(exit_idx),
                "lev": float(cur_lev),
                "baseline_entry": float(entry_price),
                "baseline_exit": float(exit_price),
                "baseline_realized": float(realized),
            }
        )

    baseline = {
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "trades": int(len(trades)),
        "wr_pct": round((100.0 * wins / max(len(trades), 1)), 2),
    }
    return baseline, trades


def _eligible(df, trade: dict, cfg: HybridConfig) -> bool:
    row = df.iloc[int(trade["signal_idx"])]
    side = str(trade["side"])
    conf = float(np.clip(_safe_float(row.get("m7_confidence", 0.5), 0.5), 0.0, 1.0))
    qwidth = max(_safe_float(row.get("m7_qwidth", 0.01), 0.01), 1e-4)
    p_dn, _, p_up = base._prob_triplet(row)
    edge = (p_up - p_dn) if side == "LONG" else (p_dn - p_up)
    flow = base._flow_alignment(row, side)
    toxicity = max(_safe_float(row.get("shadow_toxicity_score", 0.0), 0.0), 0.0)
    queue = max(_safe_float(row.get("shadow_queue_collapse", 0.0), 0.0), 0.0)
    aftershock = max(_safe_float(row.get("shadow_aftershock_prob", 0.0), 0.0), 0.0)
    return bool(
        conf >= cfg.conf_th
        and qwidth <= cfg.qwidth_max
        and edge >= cfg.edge_th
        and flow >= cfg.flow_th
        and toxicity <= cfg.tox_max
        and queue <= cfg.queue_max
        and aftershock <= cfg.aftershock_max
    )


def _simulate_execution(df, trades: list[dict], cfg: HybridConfig) -> dict:
    open_np = df["open"].to_numpy(dtype=np.float64)
    high_np = df["high"].to_numpy(dtype=np.float64)
    low_np = df["low"].to_numpy(dtype=np.float64)
    close_np = df["close"].to_numpy(dtype=np.float64)

    realizeds: list[float] = []
    maker_count = 0
    fallback_count = 0
    eligible_count = 0
    wins = 0

    for trade in trades:
        side = str(trade["side"])
        lev = float(trade["lev"])
        mkt_lev = lev * cfg.market_share
        mk_lev = lev * cfg.maker_share if _eligible(df, trade, cfg) else 0.0
        eligible_count += int(mk_lev > 0.0)

        market_entry = float(trade["baseline_entry"])
        market_ret = base._realized_return(side, market_entry, float(trade["baseline_exit"]), mkt_lev, base.TAKER_FEE, base.TAKER_FEE) if mkt_lev > 0.0 else 0.0

        maker_ret = 0.0
        if mk_lev > 0.0:
            signal_close = float(close_np[int(trade["signal_idx"])])
            limit_price = signal_close * (1.0 - cfg.offset_bps / 10000.0) if side == "LONG" else signal_close * (1.0 + cfg.offset_bps / 10000.0)
            fill = False
            fill_end = min(int(trade["entry_idx"]) + cfg.ttl_bars - 1, int(trade["exit_idx"]))
            for j in range(int(trade["entry_idx"]), fill_end + 1):
                if (side == "LONG" and low_np[j] <= limit_price) or (side == "SHORT" and high_np[j] >= limit_price):
                    fill = True
                    break
            if fill:
                maker_ret = base._realized_return(side, limit_price, float(trade["baseline_exit"]), mk_lev, base.MAKER_FEE, base.TAKER_FEE)
                maker_count += 1
            else:
                fallback_idx = min(int(trade["entry_idx"]) + cfg.ttl_bars, int(trade["exit_idx"]))
                fallback_entry = open_np[fallback_idx] * (1.0 + base.TAKER_SLIP) if side == "LONG" else open_np[fallback_idx] * (1.0 - base.TAKER_SLIP)
                maker_ret = base._realized_return(side, fallback_entry, float(trade["baseline_exit"]), mk_lev, base.TAKER_FEE, base.TAKER_FEE)
                fallback_count += 1

        total_ret = float(market_ret + maker_ret)
        wins += int(total_ret > 0.0)
        realizeds.append(total_ret)

    balance = float(np.prod([1.0 + r for r in realizeds])) if realizeds else 1.0
    return {
        "config": asdict(cfg),
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "trades": int(len(trades)),
        "wr_pct": round((100.0 * wins / max(len(trades), 1)), 2),
        "maker_entries": int(maker_count),
        "fallback_entries": int(fallback_count),
        "eligible_entries": int(eligible_count),
        "maker_fill_ratio": round(maker_count / max(eligible_count, 1), 4),
    }


def main() -> None:
    base.CSV_PATH = CSV_PATH
    base.CKPT = CKPT
    device = "cuda" if base.torch.cuda.is_available() else "cpu"
    df = base._load_df()
    actor = base._build_actor(device)
    baseline, trades = _load_baseline_trades(df, actor, device)

    configs: list[HybridConfig] = []
    for market_share, offset_bps, conf_th, qwidth_max, edge_th, flow_th in itertools.product(
        [0.99, 0.98, 0.97, 0.95, 0.90],
        [0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
        [0.45, 0.50, 0.55, 0.60],
        [0.008, 0.010, 0.012, 0.015],
        [0.00, 0.02, 0.04],
        [-0.05, 0.00, 0.05],
    ):
        maker_share = round(1.0 - market_share, 2)
        name = f"m{market_share:.2f}_mk{maker_share:.2f}_obps{offset_bps:.1f}_c{conf_th:.2f}_qw{qwidth_max:.3f}_e{edge_th:.2f}_f{flow_th:.2f}"
        configs.append(
            HybridConfig(
                name=name,
                market_share=float(market_share),
                maker_share=float(maker_share),
                offset_bps=float(offset_bps),
                ttl_bars=1,
                conf_th=float(conf_th),
                qwidth_max=float(qwidth_max),
                edge_th=float(edge_th),
                flow_th=float(flow_th),
                tox_max=0.45,
                queue_max=0.40,
                aftershock_max=0.40,
            )
        )

    results = []
    best: dict | None = None
    for cfg in tqdm(configs, desc="search_hybrid_limit", unit="cfg"):
        result = _simulate_execution(df, trades, cfg)
        result["delta_vs_baseline_pct"] = round(result["pnl_pct"] - baseline["pnl_pct"], 4)
        results.append(result)
        if best is None or result["delta_vs_baseline_pct"] > best["delta_vs_baseline_pct"]:
            best = result
            tqdm.write(
                f"best={cfg.name} pnl={result['pnl_pct']:.4f}% delta={result['delta_vs_baseline_pct']:+.4f}% "
                f"maker={result['maker_entries']} fallback={result['fallback_entries']} eligible={result['eligible_entries']}"
            )

    results.sort(key=lambda x: (x["delta_vs_baseline_pct"], x["pnl_pct"]), reverse=True)
    report = {
        "checkpoint": CKPT,
        "csv_path": CSV_PATH,
        "data_period": f"{df['timestamp'].min()} -> {df['timestamp'].max()}",
        "data_rows": int(len(df)),
        "baseline_execution": baseline,
        "search_space_size": int(len(configs)),
        "best_hybrid": best,
        "top10": results[:10],
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[BASELINE]", baseline)
    print("[BEST]", json.dumps(best, ensure_ascii=False))
    print("[SAVED]", OUT_JSON)


if __name__ == "__main__":
    main()
