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
OUT_JSON = "data/ensemble/reports/backtest_formula_limit_execution_2026.json"


@dataclass
class FormulaConfig:
    name: str
    geom_k: float
    support_k: float
    align_k: float
    curve_k: float
    integral_k: float
    blend_anchor: float
    ttl_bars: int


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.zeros_like(v, dtype=np.float64)
    return (v / n).astype(np.float64)


def _signed(row, key: str, side: str, scale: float, default: float = 0.0) -> float:
    sign = 1.0 if side == "LONG" else -1.0
    return float(np.tanh(sign * _safe_float(row.get(key, default), default) / max(scale, 1e-8)))


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
    wins = 0
    trades: list[dict] = []

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

    return (
        {
            "pnl_pct": round((balance - 1.0) * 100.0, 4),
            "trades": int(len(trades)),
            "wr_pct": round(100.0 * wins / max(len(trades), 1), 2),
        },
        trades,
    )


def _formula_limit_price(df, trade: dict, cfg: FormulaConfig) -> tuple[float, dict]:
    idx = int(trade["signal_idx"])
    row = df.iloc[idx]
    prev1 = df.iloc[max(idx - 1, 0)]
    prev2 = df.iloc[max(idx - 2, 0)]
    side = str(trade["side"])
    sign = 1.0 if side == "LONG" else -1.0
    close_px = float(row["close"])

    p_dn, _, p_up = base._prob_triplet(row)
    edge = float(sign * (p_up - p_dn))
    xgb_edge = float(
        sign
        * (
            _safe_float(row.get("m7_trend_xgb_up", 0.0), 0.0)
            - _safe_float(row.get("m7_trend_xgb_dn", 0.0), 0.0)
        )
    )
    quant_edge = float(
        sign
        * (
            _safe_float(row.get("m7_quant_up", 0.0), 0.0)
            - _safe_float(row.get("m7_quant_dn", 0.0), 0.0)
        )
    )
    conf = float(np.clip(_safe_float(row.get("m7_confidence", 0.5), 0.5), 0.0, 1.0))
    qwidth = max(_safe_float(row.get("m7_qwidth", 0.01), 0.01), 1e-4)
    quality = float(np.clip(_safe_float(row.get("m7_quality_pred", 0.0), 0.0), -0.05, 0.05))
    expected_ret = float(np.clip(_safe_float(row.get("m7_expected_ret", 0.0), 0.0), -0.02, 0.02))
    tail_risk = float(np.clip(_safe_float(row.get("m7_tail_risk", 0.0), 0.0), 0.0, 1.0))
    toxicity = max(_safe_float(row.get("shadow_toxicity_score", 0.0), 0.0), 0.0)
    queue = max(_safe_float(row.get("shadow_queue_collapse", 0.0), 0.0), 0.0)
    absorption = max(_safe_float(row.get("shadow_absorption_score", 0.0), 0.0), 0.0)
    aftershock = max(_safe_float(row.get("shadow_aftershock_prob", 0.0), 0.0), 0.0)

    demand_vec = np.array(
        [
            np.tanh(edge / 0.15),
            np.tanh(xgb_edge / 0.15),
            np.tanh(quant_edge / 0.15),
            np.tanh(expected_ret / 0.003),
            np.tanh((conf - 0.5) / 0.18),
        ],
        dtype=np.float64,
    )
    micro_vec = np.array(
        [
            _signed(row, "smart_money_flow", side, 0.0035),
            _signed(row, "ofi_acceleration", side, 0.18),
            _signed(row, "cvp_volume_imbalance", side, 0.55),
            _signed(row, "obi", side, 0.90),
            np.tanh(absorption / 0.45),
        ],
        dtype=np.float64,
    )
    risk_vec = np.array(
        [
            np.tanh(qwidth / 0.012),
            np.tanh(toxicity / 0.55),
            np.tanh(queue / 0.55),
            np.tanh(aftershock / 0.45),
            np.tanh(tail_risk / 0.45),
        ],
        dtype=np.float64,
    )
    diff1 = demand_vec - np.array(
        [
            np.tanh((sign * (_safe_float(prev1.get("m7_trend_xgb_up", 0.0), 0.0) - _safe_float(prev1.get("m7_trend_xgb_dn", 0.0), 0.0))) / 0.15),
            np.tanh((sign * (_safe_float(prev1.get("m7_quant_up", 0.0), 0.0) - _safe_float(prev1.get("m7_quant_dn", 0.0), 0.0))) / 0.15),
            np.tanh((sign * (_safe_float(prev1.get("m7_expected_ret", 0.0), 0.0))) / 0.003),
            np.tanh((_safe_float(prev1.get("m7_quality_pred", 0.0), 0.0)) / 0.012),
            np.tanh((_safe_float(prev1.get("m7_confidence", 0.5), 0.5) - 0.5) / 0.18),
        ],
        dtype=np.float64,
    )
    diff2 = diff1 - (
        np.array(
            [
                np.tanh((sign * (_safe_float(prev1.get("m7_trend_xgb_up", 0.0), 0.0) - _safe_float(prev1.get("m7_trend_xgb_dn", 0.0), 0.0))) / 0.15),
                np.tanh((sign * (_safe_float(prev1.get("m7_quant_up", 0.0), 0.0) - _safe_float(prev1.get("m7_quant_dn", 0.0), 0.0))) / 0.15),
                np.tanh((sign * (_safe_float(prev1.get("m7_expected_ret", 0.0), 0.0))) / 0.003),
                np.tanh((_safe_float(prev1.get("m7_quality_pred", 0.0), 0.0)) / 0.012),
                np.tanh((_safe_float(prev1.get("m7_confidence", 0.5), 0.5) - 0.5) / 0.18),
            ],
            dtype=np.float64,
        )
        - np.array(
            [
                np.tanh((sign * (_safe_float(prev2.get("m7_trend_xgb_up", 0.0), 0.0) - _safe_float(prev2.get("m7_trend_xgb_dn", 0.0), 0.0))) / 0.15),
                np.tanh((sign * (_safe_float(prev2.get("m7_quant_up", 0.0), 0.0) - _safe_float(prev2.get("m7_quant_dn", 0.0), 0.0))) / 0.15),
                np.tanh((sign * (_safe_float(prev2.get("m7_expected_ret", 0.0), 0.0))) / 0.003),
                np.tanh((_safe_float(prev2.get("m7_quality_pred", 0.0), 0.0)) / 0.012),
                np.tanh((_safe_float(prev2.get("m7_confidence", 0.5), 0.5) - 0.5) / 0.18),
            ],
            dtype=np.float64,
        )
    )

    alignment = float(np.dot(_unit(demand_vec), _unit(micro_vec)))
    support_mag = _norm(micro_vec)
    risk_mag = _norm(risk_vec)
    curvature = float(_norm(diff2) / max(_norm(diff1), 1e-6))
    signed_returns = np.array(
        [
            sign * _safe_float(row.get("smart_money_flow", 0.0), 0.0),
            sign * _safe_float(row.get("ofi_acceleration", 0.0), 0.0),
            sign * _safe_float(row.get("cvp_volume_imbalance", 0.0), 0.0),
        ],
        dtype=np.float64,
    )
    integral_pressure = float(np.trapz(np.tanh(signed_returns / np.array([0.0035, 0.18, 0.55], dtype=np.float64))))

    model_offset = abs(
        _safe_float(
            row.get("m7_entry_long_offset" if side == "LONG" else "m7_entry_short_offset", 0.0),
            0.0,
        )
    )
    anchor_price = _safe_float(
        row.get("m7_entry_long_price" if side == "LONG" else "m7_entry_short_price", 0.0),
        0.0,
    )
    if anchor_price <= 0.0:
        anchor_price = close_px

    geom_scale = float(
        np.exp(
            cfg.geom_k * risk_mag
            - cfg.support_k * support_mag
            - cfg.align_k * alignment
            + cfg.curve_k * curvature
            - cfg.integral_k * integral_pressure
        )
    )
    offset = model_offset * geom_scale
    offset += 0.00035 * np.tanh((qwidth - 0.008) / 0.004)
    offset += 0.00025 * np.tanh((toxicity + queue + aftershock) / 0.75)
    offset -= 0.00030 * np.tanh(absorption / 0.45)
    offset = float(np.clip(offset, 0.00015, 0.0120))

    geometric_price = close_px * (1.0 - offset) if side == "LONG" else close_px * (1.0 + offset)
    target = cfg.blend_anchor * anchor_price + (1.0 - cfg.blend_anchor) * geometric_price
    target = float(min(target, close_px) if side == "LONG" else max(target, close_px))

    return target, {
        "alignment": round(alignment, 6),
        "support_mag": round(support_mag, 6),
        "risk_mag": round(risk_mag, 6),
        "curvature": round(curvature, 6),
        "integral_pressure": round(integral_pressure, 6),
        "offset": round(offset, 6),
        "anchor_price": round(anchor_price, 6),
        "geom_scale": round(geom_scale, 6),
    }


def _simulate(df, trades: list[dict], cfg: FormulaConfig) -> dict:
    open_np = df["open"].to_numpy(dtype=np.float64)
    high_np = df["high"].to_numpy(dtype=np.float64)
    low_np = df["low"].to_numpy(dtype=np.float64)

    realizeds: list[float] = []
    maker_fills = 0
    fallbacks = 0
    sample_debug: list[dict] = []

    for k, trade in enumerate(trades):
        side = str(trade["side"])
        lev = float(trade["lev"])
        limit_price, dbg = _formula_limit_price(df, trade, cfg)

        fill = False
        fill_end = min(int(trade["entry_idx"]) + cfg.ttl_bars - 1, int(trade["exit_idx"]))
        for j in range(int(trade["entry_idx"]), fill_end + 1):
            if (side == "LONG" and low_np[j] <= limit_price) or (side == "SHORT" and high_np[j] >= limit_price):
                fill = True
                break

        if fill:
            entry_price = float(limit_price)
            realized = base._realized_return(side, entry_price, float(trade["baseline_exit"]), lev, base.MAKER_FEE, base.TAKER_FEE)
            maker_fills += 1
        else:
            fallback_idx = min(int(trade["entry_idx"]) + cfg.ttl_bars, int(trade["exit_idx"]))
            entry_price = open_np[fallback_idx] * (1.0 + base.TAKER_SLIP) if side == "LONG" else open_np[fallback_idx] * (1.0 - base.TAKER_SLIP)
            realized = base._realized_return(side, float(entry_price), float(trade["baseline_exit"]), lev, base.TAKER_FEE, base.TAKER_FEE)
            fallbacks += 1

        if k < 5:
            sample_debug.append(
                {
                    "side": side,
                    "signal_idx": int(trade["signal_idx"]),
                    "baseline_entry": round(float(trade["baseline_entry"]), 6),
                    "formula_limit": round(float(limit_price), 6),
                    "used_entry": round(float(entry_price), 6),
                    "filled": bool(fill),
                    **dbg,
                }
            )
        realizeds.append(float(realized))

    balance = float(np.prod([1.0 + r for r in realizeds])) if realizeds else 1.0
    wins = sum(1 for r in realizeds if r > 0.0)
    return {
        "config": asdict(cfg),
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "trades": int(len(trades)),
        "wr_pct": round(100.0 * wins / max(len(trades), 1), 2),
        "maker_fills": int(maker_fills),
        "fallbacks": int(fallbacks),
        "sample_debug": sample_debug,
    }


def main() -> None:
    base.CSV_PATH = CSV_PATH
    base.CKPT = CKPT
    device = "cuda" if base.torch.cuda.is_available() else "cpu"
    df = base._load_df()
    actor = base._build_actor(device)
    baseline, trades = _load_baseline_trades(df, actor, device)

    configs: list[FormulaConfig] = []
    for geom_k, support_k, align_k, curve_k, integral_k, blend_anchor in itertools.product(
        [0.10, 0.18, 0.26],
        [0.08, 0.14, 0.20],
        [0.04, 0.08, 0.12],
        [0.04, 0.08, 0.12],
        [0.02, 0.05, 0.08],
        [0.55, 0.70, 0.85],
    ):
        name = (
            f"g{geom_k:.2f}_s{support_k:.2f}_a{align_k:.2f}_"
            f"c{curve_k:.2f}_i{integral_k:.2f}_b{blend_anchor:.2f}"
        )
        configs.append(
            FormulaConfig(
                name=name,
                geom_k=float(geom_k),
                support_k=float(support_k),
                align_k=float(align_k),
                curve_k=float(curve_k),
                integral_k=float(integral_k),
                blend_anchor=float(blend_anchor),
                ttl_bars=1,
            )
        )

    results = []
    best: dict | None = None
    for cfg in tqdm(configs, desc="formula_limit_bt", unit="cfg"):
        result = _simulate(df, trades, cfg)
        result["delta_vs_baseline_pct"] = round(result["pnl_pct"] - baseline["pnl_pct"], 4)
        results.append(result)
        if best is None or result["delta_vs_baseline_pct"] > best["delta_vs_baseline_pct"]:
            best = result
            tqdm.write(
                f"best={cfg.name} pnl={result['pnl_pct']:.4f}% delta={result['delta_vs_baseline_pct']:+.4f}% "
                f"fills={result['maker_fills']} fallback={result['fallbacks']}"
            )

    results.sort(key=lambda x: (x["delta_vs_baseline_pct"], x["pnl_pct"]), reverse=True)
    report = {
        "checkpoint": CKPT,
        "csv_path": CSV_PATH,
        "data_period": f"{df['timestamp'].min()} -> {df['timestamp'].max()}",
        "data_rows": int(len(df)),
        "formula_family": "offset = model_offset * exp(g*risk - s*support - a*alignment + c*curvature - i*integral_pressure) + nonlinear corrections",
        "baseline_execution": baseline,
        "search_space_size": int(len(configs)),
        "best_formula": best,
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
