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
from scripts.eval_2026_formula_limit_closedloop import BEST_CFG, _formula_limit_price


CSV_PATH = "data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv"
CKPT = "data/ensemble/ckpt/best_dsac_agents.pth"
OUT_JSON = "data/ensemble/reports/eval_2026_formula_limit_gated_closedloop.json"


@dataclass
class GateConfig:
    name: str
    trend_take_th: float
    pullback_min: float
    release_th: float
    toxicity_max: float
    queue_max: float
    aftershock_max: float
    ttl_bars: int


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _sgn_tanh(v: float, scale: float) -> float:
    return float(np.tanh(float(v) / max(float(scale), 1e-8)))


def _entry_regime_scores(row, side: str) -> tuple[float, float, dict]:
    sign = 1.0 if side == "LONG" else -1.0
    p_up = _safe_float(row.get("m7_trend_xgb_up", row.get("prob_up", 1.0 / 3.0)), 1.0 / 3.0)
    p_dn = _safe_float(row.get("m7_trend_xgb_dn", row.get("prob_dn", 1.0 / 3.0)), 1.0 / 3.0)
    conf = float(np.clip(_safe_float(row.get("m7_confidence", 0.5), 0.5), 0.0, 1.0))
    quality = float(np.clip(_safe_float(row.get("m7_quality_pred", 0.0), 0.0), -0.05, 0.05))
    qwidth = max(_safe_float(row.get("m7_qwidth", 0.01), 0.01), 1e-4)
    tail_risk = float(np.clip(_safe_float(row.get("m7_tail_risk", 0.0), 0.0), 0.0, 1.0))
    trend_edge = sign * (p_up - p_dn)
    trend_xgb_edge = sign * (p_up - p_dn)
    quant_edge = sign * (
        _safe_float(row.get("m7_quant_up", 0.0), 0.0)
        - _safe_float(row.get("m7_quant_dn", 0.0), 0.0)
    )
    flow = sign * np.tanh(_safe_float(row.get("smart_money_flow", 0.0), 0.0) / 0.0035)
    micro = sign * np.tanh(_safe_float(row.get("ofi_acceleration", 0.0), 0.0) / 0.18)
    crowd = sign * np.tanh(_safe_float(row.get("cvp_volume_imbalance", 0.0), 0.0) / 0.55)
    ret1 = _safe_float(row.get("close", 0.0), 0.0)
    prev_close = _safe_float(row.get("open", ret1), ret1)
    ret1 = (ret1 / max(prev_close, 1e-12)) - 1.0
    vwap_gap = _safe_float(row.get("cvp_poc_dist", 0.0), 0.0)
    regime_push = 0.0
    continuation = (
        0.22 * _sgn_tanh(trend_edge, 0.30)
        + 0.16 * _sgn_tanh(trend_xgb_edge, 0.25)
        + 0.10 * _sgn_tanh(quant_edge, 0.25)
        + 0.14 * _sgn_tanh(flow, 0.20)
        + 0.10 * _sgn_tanh(crowd, 0.15)
        + 0.10 * _sgn_tanh(micro, 0.20)
        + 0.05 * _sgn_tanh(ret1, 0.0020)
        + 0.05 * _sgn_tanh(vwap_gap, 0.0020)
        + 0.04 * regime_push
        + 0.04 * conf
        + 0.03 * _sgn_tanh(quality, 0.006)
        - 0.06 * _sgn_tanh(qwidth, 0.012)
        - 0.08 * tail_risk
    )
    toxicity = max(_safe_float(row.get("shadow_toxicity_score", 0.0), 0.0), 0.0)
    queue_collapse = max(_safe_float(row.get("shadow_queue_collapse", 0.0), 0.0), 0.0)
    spoof_support = 0.0
    pullback = (
        0.26 * _sgn_tanh(-ret1, 0.0020)
        + 0.18 * _sgn_tanh(-flow, 0.20)
        + 0.12 * _sgn_tanh(-micro, 0.20)
        + 0.10 * _sgn_tanh(qwidth, 0.012)
        + 0.10 * _sgn_tanh(queue_collapse, 0.50)
        + 0.10 * _sgn_tanh(toxicity, 0.50)
        - 0.10 * conf
        - 0.08 * _sgn_tanh(quality, 0.006)
        - 0.06 * regime_push
        - 0.04 * spoof_support
    )
    return float(continuation), float(pullback), {
        "continuation": round(float(continuation), 6),
        "pullback": round(float(pullback), 6),
        "toxicity": round(float(toxicity), 6),
        "queue_collapse": round(float(queue_collapse), 6),
        "aftershock": round(float(_safe_float(row.get("shadow_aftershock_prob", 0.0), 0.0)), 6),
    }


def simulate_gated_formula(df, actor, device: str, gate_cfg: GateConfig) -> dict:
    numeric_cols = [c for c in df.columns if c != "timestamp"]
    values = df[numeric_cols].to_numpy(dtype=np.float64)
    open_np = df["open"].to_numpy(dtype=np.float64)
    high_np = df["high"].to_numpy(dtype=np.float64)
    low_np = df["low"].to_numpy(dtype=np.float64)
    close_np = df["close"].to_numpy(dtype=np.float64)

    router = base.DSACRouter(copy.deepcopy(actor), device=device)
    balance = 1.0
    pos: str | None = None
    entry_price = 0.0
    entry_fee = 0.0
    cur_lev = 0.0
    hold_count = 0
    trades = 0
    wins = 0
    maker_entries = 0
    fallback_entries = 0
    market_entries = 0
    resize_count = 0
    eq_curve = [1.0]
    pending: dict | None = None
    debug_samples: list[dict] = []

    def _unrealized(current_price: float) -> float:
        if pos is None or entry_price <= 0.0 or cur_lev <= 0.0:
            return 0.0
        gross = ((current_price - entry_price) / entry_price) if pos == "LONG" else ((entry_price - current_price) / entry_price)
        return float(gross * cur_lev - ((entry_fee + base.MAKER_FEE) * cur_lev))

    for i in range(1, len(df) - 1):
        bar_open = float(open_np[i])
        bar_high = float(high_np[i])
        bar_low = float(low_np[i])
        bar_close = float(close_np[i])

        if pos is not None:
            hold_count += 1

        if pending is not None:
            fill = (pending["side"] == "LONG" and bar_low <= pending["price"]) or (pending["side"] == "SHORT" and bar_high >= pending["price"])
            if fill:
                pos = pending["side"]
                entry_price = float(pending["price"])
                entry_fee = base.MAKER_FEE
                cur_lev = float(pending["lev"])
                hold_count = 0
                balance -= balance * base.MAKER_FEE * cur_lev
                maker_entries += 1
                if len(debug_samples) < 8:
                    debug_samples.append({"entry_mode": "maker", "signal_idx": int(pending["signal_idx"]), "side": str(pos), "entry_price": round(entry_price, 6), **pending["debug"]})
                pending = None
            elif i > pending["expire_idx"]:
                pos = pending["side"]
                entry_price = bar_open * (1.0 + base.TAKER_SLIP) if pos == "LONG" else bar_open * (1.0 - base.TAKER_SLIP)
                entry_fee = base.TAKER_FEE
                cur_lev = float(pending["lev"])
                hold_count = 0
                balance -= balance * base.TAKER_FEE * cur_lev
                fallback_entries += 1
                if len(debug_samples) < 8:
                    debug_samples.append({"entry_mode": "fallback_market", "signal_idx": int(pending["signal_idx"]), "side": str(pos), "entry_price": round(entry_price, 6), **pending["debug"]})
                pending = None

        if pending is None:
            cp = float(close_np[i - 1])
            pos_dict = {
                "type": pos,
                "entry_price": float(entry_price),
                "unrealized": float(_unrealized(cp)),
                "mdd": 0.0,
                "hold_norm": float(min(hold_count / 96.0, 1.0)),
                "margin_usage": float(cur_lev if pos else 0.0),
                "hold_count": float(hold_count),
            }
            features = {k: float(v) for k, v in zip(numeric_cols, values[i - 1])}
            action_int, lev, _ = router.decide(features, pos_dict)
            lev = float(np.clip(lev, 0.0, 1.0))

            if pos is None:
                if action_int in (1, 2) and lev > 0.0:
                    side = "LONG" if action_int == 1 else "SHORT"
                    row = df.iloc[i - 1]
                    continuation, pullback, reg_dbg = _entry_regime_scores(row, side)
                    toxicity = _safe_float(row.get("shadow_toxicity_score", 0.0), 0.0)
                    queue = _safe_float(row.get("shadow_queue_collapse", 0.0), 0.0)
                    aftershock = _safe_float(row.get("shadow_aftershock_prob", 0.0), 0.0)
                    calm_ok = toxicity <= gate_cfg.toxicity_max and queue <= gate_cfg.queue_max and aftershock <= gate_cfg.aftershock_max
                    use_formula = continuation < gate_cfg.trend_take_th and continuation < gate_cfg.release_th and pullback >= gate_cfg.pullback_min and calm_ok
                    if use_formula:
                        price, fdbg = _formula_limit_price(df, i - 1, side, BEST_CFG)
                        pending = {
                            "side": side,
                            "price": float(price),
                            "expire_idx": i - 1 + gate_cfg.ttl_bars,
                            "lev": lev,
                            "signal_idx": int(i - 1),
                            "debug": {**reg_dbg, **fdbg},
                        }
                    else:
                        pos = side
                        entry_price = bar_open * (1.0 + base.TAKER_SLIP) if side == "LONG" else bar_open * (1.0 - base.TAKER_SLIP)
                        entry_fee = base.TAKER_FEE
                        cur_lev = lev
                        hold_count = 0
                        balance -= balance * base.TAKER_FEE * cur_lev
                        market_entries += 1
                        if len(debug_samples) < 8:
                            debug_samples.append({"entry_mode": "market", "signal_idx": int(i - 1), "side": str(pos), "entry_price": round(entry_price, 6), **reg_dbg})
            else:
                should_close = action_int == 0 or (action_int == 1 and pos == "SHORT") or (action_int == 2 and pos == "LONG")
                if should_close:
                    exit_price = bar_open * (1.0 - base.TAKER_SLIP) if pos == "LONG" else bar_open * (1.0 + base.TAKER_SLIP)
                    realized = base._realized_return(pos, entry_price, exit_price, cur_lev, entry_fee, base.TAKER_FEE)
                    balance *= 1.0 + realized
                    trades += 1
                    wins += int(realized > 0.0)
                    pos = None
                    entry_price = 0.0
                    entry_fee = 0.0
                    cur_lev = 0.0
                    hold_count = 0
                elif abs(lev - cur_lev) > 0.05:
                    balance -= balance * base.TAKER_FEE * abs(lev - cur_lev)
                    cur_lev = lev
                    resize_count += 1

        eq_curve.append(max(balance * (1.0 + _unrealized(bar_close)) if pos else balance, 1e-8))

    if pos is not None and entry_price > 0.0:
        exit_price = float(close_np[-1]) * (1.0 - base.TAKER_SLIP) if pos == "LONG" else float(close_np[-1]) * (1.0 + base.TAKER_SLIP)
        realized = base._realized_return(pos, entry_price, exit_price, cur_lev, entry_fee, base.TAKER_FEE)
        balance *= 1.0 + realized
        trades += 1
        wins += int(realized > 0.0)

    return {
        "method": "gated_formula_limit_closed_loop",
        "gate_config": asdict(gate_cfg),
        "formula_config": asdict(BEST_CFG),
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "wr_pct": round((100.0 * wins / trades) if trades else 0.0, 2),
        "trades": int(trades),
        "sharpe": round(base._sharpe(eq_curve), 4),
        "mdd_pct": round(base._mdd(eq_curve), 4),
        "market_entries": int(market_entries),
        "maker_entries": int(maker_entries),
        "fallback_entries": int(fallback_entries),
        "resize_count": int(resize_count),
        "debug_samples": debug_samples,
    }


def main() -> None:
    base.CSV_PATH = CSV_PATH
    base.CKPT = CKPT
    device = "cuda" if base.torch.cuda.is_available() else "cpu"
    df = base._load_df()
    actor = base._build_actor(device)
    baseline = base.simulate_market(df, base.DSACRouter(copy.deepcopy(actor), device=device))

    gate_configs = [
        GateConfig(
            name=f"tt{tt:.2f}_pb{pb:.2f}_rl{rl:.2f}_tx{tx:.2f}_qc{qc:.2f}_af{af:.2f}",
            trend_take_th=tt,
            pullback_min=pb,
            release_th=rl,
            toxicity_max=tx,
            queue_max=qc,
            aftershock_max=af,
            ttl_bars=1,
        )
        for tt, pb, rl, tx, qc, af in itertools.product(
            [0.15, 0.18],
            [0.20, 0.24],
            [0.05],
            [0.35, 0.45],
            [0.35],
            [0.35],
        )
    ]

    results = []
    best = None
    for cfg in tqdm(gate_configs, desc="gate_search", unit="cfg"):
        res = simulate_gated_formula(df, actor, device, cfg)
        res["delta_vs_market_pct"] = round(res["pnl_pct"] - baseline["pnl_pct"], 4)
        results.append(res)
        if best is None or res["delta_vs_market_pct"] > best["delta_vs_market_pct"]:
            best = res
            tqdm.write(
                f"best={cfg.name} pnl={res['pnl_pct']:.4f}% delta={res['delta_vs_market_pct']:+.4f}% "
                f"market={res['market_entries']} maker={res['maker_entries']} fb={res['fallback_entries']}"
            )

    results.sort(key=lambda x: (x["delta_vs_market_pct"], x["pnl_pct"]), reverse=True)
    report = {
        "checkpoint": CKPT,
        "csv_path": CSV_PATH,
        "data_period": f"{df['timestamp'].min()} -> {df['timestamp'].max()}",
        "data_rows": int(len(df)),
        "baseline_market": baseline,
        "best_gated_formula": best,
        "top10": results[:10],
        "search_space_size": int(len(gate_configs)),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[BASELINE]", baseline)
    print("[BEST]", json.dumps(best, ensure_ascii=False))
    print("[SAVED]", OUT_JSON)


if __name__ == "__main__":
    main()
