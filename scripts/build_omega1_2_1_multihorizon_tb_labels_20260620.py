#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_1_multihorizon_tb_daytrade_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"

HORIZONS = [12, 24, 48, 96, 192]
TP_ATR_MULTS = [1.0, 1.5, 2.0]
SL_ATR_MULTS = [0.8, 1.0, 1.2]
TP_BOUNDS = (0.008, 0.050)
SL_BOUNDS = (0.006, 0.035)
FEE_PER_SIDE = 0.0001 * 3.0
NOTIONAL = 0.15
HOLD_PENALTY = 0.000006
TAIL_PENALTY = 0.20


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _f(row: dict[str, str], col: str, default: float = 0.0) -> float:
    try:
        return float(row.get(col, default))
    except Exception:
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), float(lo)), float(hi))


def _atr_pct(rows: list[dict[str, str]], lookback: int = 48) -> list[float]:
    out: list[float] = []
    trs: list[float] = []
    prev_close = _f(rows[0], "close")
    for row in rows:
        high = _f(row, "high")
        low = _f(row, "low")
        close = _f(row, "close")
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        if len(trs) > lookback:
            trs.pop(0)
        atr = sum(trs) / len(trs)
        out.append(atr / close if close > 0.0 else 0.0)
        prev_close = close
    return out


def _rolling_ret(rows: list[dict[str, str]], i: int, bars: int) -> float:
    if i - bars < 0:
        return 0.0
    a = _f(rows[i - bars], "close")
    b = _f(rows[i], "close")
    return (b - a) / a if a > 0.0 else 0.0


def _future_windows(rows: list[dict[str, str]]) -> dict[int, list[dict[str, float]]]:
    windows: dict[int, list[dict[str, float]]] = {}
    for horizon in HORIZONS:
        vals: list[dict[str, float]] = []
        for i in range(len(rows)):
            entry_i = i + 1
            if entry_i >= len(rows):
                vals.append({"ok": 0.0})
                continue
            end_i = min(len(rows) - 1, entry_i + int(horizon))
            highs = [_f(rows[j], "high") for j in range(entry_i, end_i + 1)]
            lows = [_f(rows[j], "low") for j in range(entry_i, end_i + 1)]
            vals.append(
                {
                    "ok": 1.0,
                    "entry": _f(rows[entry_i], "open"),
                    "max_high": max(highs),
                    "min_low": min(lows),
                    "end_close": _f(rows[end_i], "close"),
                    "hold": float(end_i - entry_i),
                }
            )
        windows[horizon] = vals
    return windows


def _simulate_fast(
    win: dict[str, float],
    *,
    side: int,
    tp_price_move: float,
    sl_price_move: float,
) -> dict[str, Any]:
    if float(win.get("ok", 0.0)) <= 0.0:
        return {"utility": -999.0, "ret": 0.0, "hold": 0, "reason": "no_entry", "mfe": 0.0, "mae": 0.0}
    entry = float(win["entry"])
    if entry <= 0.0:
        return {"utility": -999.0, "ret": 0.0, "hold": 0, "reason": "bad_entry", "mfe": 0.0, "mae": 0.0}
    if side > 0:
        mfe = (float(win["max_high"]) - entry) / entry
        mae = (float(win["min_low"]) - entry) / entry
        end_raw = (float(win["end_close"]) - entry) / entry
    else:
        mfe = (entry - float(win["min_low"])) / entry
        mae = (entry - float(win["max_high"])) / entry
        end_raw = (entry - float(win["end_close"])) / entry
    hit_tp = mfe >= tp_price_move
    hit_sl = mae <= -abs(sl_price_move)
    if hit_tp and hit_sl:
        reason = "stop_loss"
        raw = -abs(sl_price_move)
    elif hit_tp:
        reason = "take_profit"
        raw = tp_price_move
    elif hit_sl:
        reason = "stop_loss"
        raw = -abs(sl_price_move)
    else:
        reason = "max_hold"
        raw = end_raw
    net = raw * NOTIONAL - (2.0 * FEE_PER_SIDE * NOTIONAL)
    hold = int(win["hold"])
    tail = TAIL_PENALTY * max(0.0, abs(mae) - abs(sl_price_move))
    utility = net - HOLD_PENALTY * hold - tail
    return {"utility": utility, "ret": net, "hold": hold, "reason": reason, "mfe": mfe, "mae": mae}


def _simulate(
    rows: list[dict[str, str]],
    i: int,
    *,
    side: int,
    horizon: int,
    tp_price_move: float,
    sl_price_move: float,
) -> dict[str, Any]:
    entry_i = i + 1
    if entry_i >= len(rows):
        return {"utility": -999.0, "ret": 0.0, "hold": 0, "reason": "no_entry", "mfe": 0.0, "mae": 0.0}
    entry = _f(rows[entry_i], "open")
    if entry <= 0.0:
        return {"utility": -999.0, "ret": 0.0, "hold": 0, "reason": "bad_entry", "mfe": 0.0, "mae": 0.0}
    end_i = min(len(rows) - 1, entry_i + int(horizon))
    exit_i = end_i
    exit_px = _f(rows[end_i], "close")
    reason = "max_hold"
    mfe = 0.0
    mae = 0.0
    for j in range(entry_i, end_i + 1):
        high = _f(rows[j], "high")
        low = _f(rows[j], "low")
        if side > 0:
            hi_raw = (high - entry) / entry
            lo_raw = (low - entry) / entry
        else:
            hi_raw = (entry - low) / entry
            lo_raw = (entry - high) / entry
        mfe = max(mfe, hi_raw)
        mae = min(mae, lo_raw)
        hit_tp = hi_raw >= tp_price_move
        hit_sl = lo_raw <= -abs(sl_price_move)
        if hit_tp or hit_sl:
            exit_i = j
            if hit_sl and hit_tp:
                reason = "stop_loss"
                exit_px = entry * (1.0 - side * sl_price_move)
            elif hit_tp:
                reason = "take_profit"
                exit_px = entry * (1.0 + side * tp_price_move)
            else:
                reason = "stop_loss"
                exit_px = entry * (1.0 - side * sl_price_move)
            break
    raw = (exit_px - entry) / entry if side > 0 else (entry - exit_px) / entry
    net = raw * NOTIONAL - (2.0 * FEE_PER_SIDE * NOTIONAL)
    hold = int(exit_i - entry_i)
    tail = TAIL_PENALTY * max(0.0, abs(mae) - abs(sl_price_move))
    utility = net - HOLD_PENALTY * hold - tail
    return {"utility": utility, "ret": net, "hold": hold, "reason": reason, "mfe": mfe, "mae": mae}


def _build(rows: list[dict[str, str]], split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    atr = _atr_pct(rows)
    windows = _future_windows(rows)
    labels: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    horizon_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for i, row in enumerate(rows):
        if i + max(HORIZONS) + 2 >= len(rows):
            continue
        atr_i = _clip(atr[i], 0.002, 0.040)
        best: dict[str, Any] = {
            "side": 0,
            "horizon": 0,
            "tp_price_move": 0.0,
            "sl_price_move": 0.0,
            "utility": 0.0,
            "ret": 0.0,
            "hold": 0,
            "reason": "cash",
            "mfe": 0.0,
            "mae": 0.0,
        }
        for side in (1, -1):
            for horizon in HORIZONS:
                for tp_mult in TP_ATR_MULTS:
                    for sl_mult in SL_ATR_MULTS:
                        tp = _clip(tp_mult * atr_i, *TP_BOUNDS)
                        sl = _clip(sl_mult * atr_i, *SL_BOUNDS)
                        if sl >= tp:
                            continue
                        sim = _simulate_fast(windows[horizon][i], side=side, tp_price_move=tp, sl_price_move=sl)
                        if float(sim["utility"]) > float(best["utility"]):
                            best = {
                                "side": side,
                                "horizon": horizon,
                                "tp_price_move": tp,
                                "sl_price_move": sl,
                                **sim,
                            }
        side_name = "LONG" if int(best["side"]) > 0 else "SHORT" if int(best["side"]) < 0 else "CASH"
        counts[side_name] += 1
        horizon_counts[str(best["horizon"])] += 1
        reason_counts[str(best["reason"])] += 1
        labels.append(
            {
                "timestamp": row["timestamp"],
                "split": split,
                "close": _f(row, "close"),
                "atr_pct_48": atr_i,
                "ret_12": _rolling_ret(rows, i, 12),
                "ret_48": _rolling_ret(rows, i, 48),
                "ret_96": _rolling_ret(rows, i, 96),
                "label_side": side_name,
                "label_side_id": int(best["side"]),
                "label_horizon": int(best["horizon"]),
                "label_tp_price_move": float(best["tp_price_move"]),
                "label_sl_price_move": float(best["sl_price_move"]),
                "label_utility": float(best["utility"]),
                "label_net_return": float(best["ret"]),
                "label_hold_bars": int(best["hold"]),
                "label_reason": str(best["reason"]),
                "label_mfe": float(best["mfe"]),
                "label_mae": float(best["mae"]),
            }
        )
    diag = {
        "split": split,
        "rows": len(labels),
        "side_counts": dict(counts),
        "horizon_counts": dict(horizon_counts),
        "reason_counts": dict(reason_counts),
        "horizons": HORIZONS,
        "tp_atr_mults": TP_ATR_MULTS,
        "sl_atr_mults": SL_ATR_MULTS,
        "notional_for_label_utility": NOTIONAL,
        "hold_penalty": HOLD_PENALTY,
        "tail_penalty": TAIL_PENALTY,
    }
    return labels, diag


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = _read_csv(TRAIN_CSV)
    eval_rows = _read_csv(EVAL_CSV)
    train_labels, train_diag = _build(train_rows, "train_2025")
    oos_labels, oos_diag = _build(eval_rows, "oos_2026")
    _write_csv(OUT_DIR / "train_2025_multihorizon_tb_labels.csv", train_labels)
    _write_csv(OUT_DIR / "oos_2026_multihorizon_tb_labels.csv", oos_labels)
    report = {
        "model_id": MODEL_ID,
        "status": "labels_built",
        "label_mode": "multi_horizon_atr_triple_barrier_first_touch",
        "risk_contract": {
            "tp_price_move": "clip(tp_atr_mult * atr_pct_48, tp_min, tp_max)",
            "sl_price_move": "clip(sl_atr_mult * atr_pct_48, sl_min, sl_max)",
            "notional": NOTIONAL,
            "pnl": "price_move * notional - fees",
            "max_hold": "selected horizon bucket",
        },
        "train_2025": train_diag,
        "oos_2026": oos_diag,
        "artifacts": {
            "train_labels": str((OUT_DIR / "train_2025_multihorizon_tb_labels.csv").relative_to(ROOT)),
            "oos_labels": str((OUT_DIR / "oos_2026_multihorizon_tb_labels.csv").relative_to(ROOT)),
        },
    }
    (OUT_DIR / "label_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
