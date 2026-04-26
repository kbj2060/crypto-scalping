#!/usr/bin/env python3
from __future__ import annotations

import copy
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
OUT_JSON = "data/ensemble/reports/eval_2026_formula_limit_closedloop.json"


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


BEST_CFG = FormulaConfig(
    name="g0.26_s0.08_a0.12_c0.12_i0.02_b0.55",
    geom_k=0.26,
    support_k=0.08,
    align_k=0.12,
    curve_k=0.12,
    integral_k=0.02,
    blend_anchor=0.55,
    ttl_bars=1,
)


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


def _formula_limit_price(df, idx: int, side: str, cfg: FormulaConfig) -> tuple[float, dict]:
    row = df.iloc[idx]
    prev1 = df.iloc[max(idx - 1, 0)]
    prev2 = df.iloc[max(idx - 2, 0)]
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
    prev1_vec = np.array(
        [
            np.tanh((sign * (_safe_float(prev1.get("m7_trend_xgb_up", 0.0), 0.0) - _safe_float(prev1.get("m7_trend_xgb_dn", 0.0), 0.0))) / 0.15),
            np.tanh((sign * (_safe_float(prev1.get("m7_quant_up", 0.0), 0.0) - _safe_float(prev1.get("m7_quant_dn", 0.0), 0.0))) / 0.15),
            np.tanh((sign * _safe_float(prev1.get("m7_expected_ret", 0.0), 0.0)) / 0.003),
            np.tanh(_safe_float(prev1.get("m7_quality_pred", 0.0), 0.0) / 0.012),
            np.tanh((_safe_float(prev1.get("m7_confidence", 0.5), 0.5) - 0.5) / 0.18),
        ],
        dtype=np.float64,
    )
    prev2_vec = np.array(
        [
            np.tanh((sign * (_safe_float(prev2.get("m7_trend_xgb_up", 0.0), 0.0) - _safe_float(prev2.get("m7_trend_xgb_dn", 0.0), 0.0))) / 0.15),
            np.tanh((sign * (_safe_float(prev2.get("m7_quant_up", 0.0), 0.0) - _safe_float(prev2.get("m7_quant_dn", 0.0), 0.0))) / 0.15),
            np.tanh((sign * _safe_float(prev2.get("m7_expected_ret", 0.0), 0.0)) / 0.003),
            np.tanh(_safe_float(prev2.get("m7_quality_pred", 0.0), 0.0) / 0.012),
            np.tanh((_safe_float(prev2.get("m7_confidence", 0.5), 0.5) - 0.5) / 0.18),
        ],
        dtype=np.float64,
    )
    diff1 = demand_vec - prev1_vec
    diff2 = prev1_vec - prev2_vec

    alignment = float(np.dot(_unit(demand_vec), _unit(micro_vec)))
    support_mag = _norm(micro_vec)
    risk_mag = _norm(risk_vec)
    curvature = float(_norm(diff1 - diff2) / max(_norm(diff1), 1e-6))
    signed_returns = np.array(
        [
            sign * _safe_float(row.get("smart_money_flow", 0.0), 0.0),
            sign * _safe_float(row.get("ofi_acceleration", 0.0), 0.0),
            sign * _safe_float(row.get("cvp_volume_imbalance", 0.0), 0.0),
        ],
        dtype=np.float64,
    )
    integral_pressure = float(np.trapz(np.tanh(signed_returns / np.array([0.0035, 0.18, 0.55], dtype=np.float64))))

    model_offset = abs(_safe_float(row.get("m7_entry_long_offset" if side == "LONG" else "m7_entry_short_offset", 0.0), 0.0))
    anchor_price = _safe_float(row.get("m7_entry_long_price" if side == "LONG" else "m7_entry_short_price", 0.0), 0.0)
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


def simulate_formula_limit(df, actor, device: str, cfg: FormulaConfig) -> dict:
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
    missed_entries = 0
    resize_count = 0
    eq_curve = [1.0]
    pending: dict | None = None
    debug_samples: list[dict] = []

    def _unrealized(current_price: float) -> float:
        if pos is None or entry_price <= 0.0 or cur_lev <= 0.0:
            return 0.0
        gross = (
            (current_price - entry_price) / entry_price
            if pos == "LONG"
            else (entry_price - current_price) / entry_price
        )
        return float(gross * cur_lev - ((entry_fee + base.MAKER_FEE) * cur_lev))

    for i in tqdm(range(1, len(df) - 1), desc="formula_closedloop", unit="bar"):
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
                    debug_samples.append(
                        {
                            "signal_idx": int(pending["signal_idx"]),
                            "side": str(pos),
                            "entry_mode": "maker",
                            "entry_price": round(entry_price, 6),
                            **dict(pending.get("debug", {})),
                        }
                    )
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
                    debug_samples.append(
                        {
                            "signal_idx": int(pending["signal_idx"]),
                            "side": str(pos),
                            "entry_mode": "fallback_market",
                            "entry_price": round(entry_price, 6),
                            **dict(pending.get("debug", {})),
                        }
                    )
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
                    price, debug = _formula_limit_price(df, i - 1, side, cfg)
                    pending = {
                        "kind": "entry",
                        "side": side,
                        "price": float(price),
                        "expire_idx": i - 1 + int(cfg.ttl_bars),
                        "lev": lev,
                        "signal_idx": int(i - 1),
                        "debug": debug,
                    }
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

    if pending is not None:
        missed_entries += 1
    if pos is not None and entry_price > 0.0:
        exit_price = float(close_np[-1]) * (1.0 - base.TAKER_SLIP) if pos == "LONG" else float(close_np[-1]) * (1.0 + base.TAKER_SLIP)
        realized = base._realized_return(pos, entry_price, exit_price, cur_lev, entry_fee, base.TAKER_FEE)
        balance *= 1.0 + realized
        trades += 1
        wins += int(realized > 0.0)

    return {
        "method": "formula_limit_closed_loop",
        "config": asdict(cfg),
        "pnl_pct": round((balance - 1.0) * 100.0, 4),
        "wr_pct": round((100.0 * wins / trades) if trades else 0.0, 2),
        "trades": int(trades),
        "sharpe": round(base._sharpe(eq_curve), 4),
        "mdd_pct": round(base._mdd(eq_curve), 4),
        "maker_entries": int(maker_entries),
        "fallback_entries": int(fallback_entries),
        "missed_entries": int(missed_entries),
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
    formula = simulate_formula_limit(df, actor, device, BEST_CFG)
    formula["delta_vs_market_pct"] = round(formula["pnl_pct"] - baseline["pnl_pct"], 4)

    report = {
        "checkpoint": CKPT,
        "csv_path": CSV_PATH,
        "data_period": f"{df['timestamp'].min()} -> {df['timestamp'].max()}",
        "data_rows": int(len(df)),
        "baseline_market": baseline,
        "formula_limit": formula,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[BASELINE]", baseline)
    print("[FORMULA]", json.dumps(formula, ensure_ascii=False))
    print("[SAVED]", OUT_JSON)


if __name__ == "__main__":
    main()
