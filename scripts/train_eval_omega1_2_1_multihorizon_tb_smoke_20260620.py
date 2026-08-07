#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_1_multihorizon_tb_daytrade_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
LABEL_2025 = OUT_DIR / "train_2025_multihorizon_tb_labels.csv"
PARENT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"

SPLIT_TS = "2025-10-01"
LEVERAGE = 2.0
FEE_PER_SIDE = 0.0001 * 3.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows: {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _f(row: dict[str, str], col: str, default: float = 0.0) -> float:
    try:
        return float(row.get(col, default))
    except Exception:
        return default


def _side(action: int) -> int:
    if action == 1:
        return 1
    if action == 2:
        return -1
    return 0


def _bin_signed(x: float, eps: float) -> str:
    if x > eps:
        return "pos"
    if x < -eps:
        return "neg"
    return "flat"


def _bin_quantile(x: float, q1: float, q2: float) -> str:
    if x <= q1:
        return "lo"
    if x <= q2:
        return "mid"
    return "hi"


def _hour_bin(ts: str) -> str:
    hour = int(ts[11:13])
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 16:
        return "europe"
    return "us"


def _rolling_ret(rows: list[dict[str, str]], i: int, bars: int) -> float:
    if i - bars < 0:
        return 0.0
    a = _f(rows[i - bars], "close")
    b = _f(rows[i], "close")
    return (b - a) / a if a > 0.0 else 0.0


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


def _feature_rows(rows: list[dict[str, str]], *, vol_q: tuple[float, float]) -> list[dict[str, Any]]:
    atr = _atr_pct(rows)
    feats: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        r12 = _rolling_ret(rows, i, 12)
        r48 = _rolling_ret(rows, i, 48)
        r96 = _rolling_ret(rows, i, 96)
        feats.append(
            {
                "timestamp": row["timestamp"],
                "open": _f(row, "open"),
                "high": _f(row, "high"),
                "low": _f(row, "low"),
                "close": _f(row, "close"),
                "atr_pct_48": atr[i],
                "vol_bin": _bin_quantile(atr[i], *vol_q),
                "ret12_bin": _bin_signed(r12, 0.0015),
                "ret48_bin": _bin_signed(r48, 0.0030),
                "ret96_bin": _bin_signed(r96, 0.0050),
                "hour_bin": _hour_bin(row["timestamp"]),
                "rsi_bin": _bin_quantile(_f(row, "rsi", 50.0), 43.0, 57.0),
                "breakout_bin": _bin_quantile(_f(row, "breakout_strength"), -0.2, 0.2),
                "chop_bin": _bin_quantile(_f(row, "chop_index"), 40.0, 60.0),
                "funding_bin": _bin_signed(_f(row, "last_funding_rate"), 0.00002),
            }
        )
    return feats


def _key(row: dict[str, Any], level: int) -> tuple[Any, ...]:
    if level == 0:
        return (row["vol_bin"], row["ret12_bin"], row["ret48_bin"], row["hour_bin"], row["rsi_bin"])
    if level == 1:
        return (row["vol_bin"], row["ret12_bin"], row["ret48_bin"])
    if level == 2:
        return (row["vol_bin"], row["ret48_bin"])
    return (row["ret48_bin"],)


def _load_parent(split: str) -> dict[str, dict[str, Any]]:
    if split == "validation":
        path = PARENT_DIR / "validation_predictions_2025_true3head.csv"
        prefix = "omega1_regime3_expertdq_oof_"
    elif split == "oos":
        path = PARENT_DIR / "oos_predictions_2026_true3head.csv"
        prefix = "omega1_regime3_expertdq_"
    else:
        return {}
    out = {}
    for r in _read_csv(path):
        out[r["timestamp"]] = {
            "parent_side": _side(int(_f(r, f"{prefix}dir_action"))),
            "parent_quality": _f(r, f"{prefix}quality_for_action"),
            "parent_trade_prob": _f(r, f"{prefix}dir_trade_prob"),
            "parent_expert": str(r.get(f"{prefix}router_expert", "")),
        }
    return out


def _fit_policy(rows: list[dict[str, Any]], labels: dict[str, dict[str, str]], cfg: dict[str, Any]) -> dict[tuple[int, tuple[Any, ...]], dict[str, Any]]:
    buckets: dict[tuple[int, tuple[Any, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lab = labels.get(row["timestamp"])
        if lab is None:
            continue
        side = int(float(lab["label_side_id"]))
        horizon = int(float(lab["label_horizon"]))
        utility = float(lab["label_utility"])
        if side == 0 or horizon <= 0 or horizon > int(cfg["max_horizon_train"]):
            continue
        if utility < float(cfg["min_label_utility"]):
            continue
        sample = {
            "side": side,
            "horizon": horizon,
            "tp": float(lab["label_tp_price_move"]),
            "sl": float(lab["label_sl_price_move"]),
            "utility": utility,
        }
        for level in range(4):
            buckets[(level, _key(row, level))].append(sample)
    model: dict[tuple[int, tuple[Any, ...]], dict[str, Any]] = {}
    for key, vals in buckets.items():
        if len(vals) < int(cfg["min_bucket_count"]):
            continue
        avg = sum(float(v["utility"]) for v in vals) / len(vals)
        if avg < float(cfg["min_bucket_utility"]):
            continue
        combos: Counter[tuple[int, int]] = Counter((int(v["side"]), int(v["horizon"])) for v in vals)
        side, horizon = combos.most_common(1)[0][0]
        filtered = [v for v in vals if int(v["side"]) == side and int(v["horizon"]) == horizon]
        if len(filtered) < max(3, int(cfg["min_bucket_count"]) // 3):
            filtered = vals
        model[key] = {
            "side": side,
            "horizon": horizon,
            "tp": median(float(v["tp"]) for v in filtered),
            "sl": median(float(v["sl"]) for v in filtered),
            "avg_utility": avg,
            "count": len(vals),
        }
    return model


def _predict(row: dict[str, Any], parent: dict[str, Any] | None, model: dict[tuple[int, tuple[Any, ...]], dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any] | None:
    pred = None
    for level in range(4):
        pred = model.get((level, _key(row, level)))
        if pred is not None:
            break
    if pred is None:
        return None
    if parent is not None and float(cfg["parent_min_quality"]) > 0.0:
        pside = int(parent.get("parent_side", 0))
        pq = float(parent.get("parent_quality", 0.0))
        if pq >= float(cfg["parent_min_quality"]) and pside != 0 and pside != int(pred["side"]):
            return None
    notional = float(cfg["margin_fraction"]) * LEVERAGE
    return {
        "side": int(pred["side"]),
        "horizon": int(pred["horizon"]),
        "tp": float(pred["tp"]),
        "sl": float(pred["sl"]),
        "notional": notional,
        "margin_fraction": float(cfg["margin_fraction"]),
        "model_count": int(pred["count"]),
        "model_avg_utility": float(pred["avg_utility"]),
    }


def _run(rows: list[dict[str, Any]], parents: dict[str, dict[str, Any]], model: dict[tuple[int, tuple[Any, ...]], dict[str, Any]], cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    ledger: list[dict[str, Any]] = []
    i = 0
    trade_id = 1
    while i < len(rows) - 2:
        sig = _predict(rows[i], parents.get(rows[i]["timestamp"]), model, cfg)
        if sig is None:
            i += 1
            continue
        entry_i = i + 1
        entry = float(rows[entry_i]["open"])
        if entry <= 0.0:
            i += 1
            continue
        side = int(sig["side"])
        horizon = int(sig["horizon"])
        end_i = min(len(rows) - 1, entry_i + horizon)
        exit_i = end_i
        exit_px = float(rows[end_i]["close"])
        reason = "max_hold"
        mfe = 0.0
        mae = 0.0
        for j in range(entry_i, end_i + 1):
            high = float(rows[j]["high"])
            low = float(rows[j]["low"])
            if side > 0:
                hi_raw = (high - entry) / entry
                lo_raw = (low - entry) / entry
            else:
                hi_raw = (entry - low) / entry
                lo_raw = (entry - high) / entry
            mfe = max(mfe, hi_raw)
            mae = min(mae, lo_raw)
            hit_tp = hi_raw >= float(sig["tp"])
            hit_sl = lo_raw <= -abs(float(sig["sl"]))
            if hit_tp or hit_sl:
                exit_i = j
                if hit_sl and hit_tp:
                    reason = "stop_loss"
                    exit_px = entry * (1.0 - side * float(sig["sl"]))
                elif hit_tp:
                    reason = "take_profit"
                    exit_px = entry * (1.0 + side * float(sig["tp"]))
                else:
                    reason = "stop_loss"
                    exit_px = entry * (1.0 - side * float(sig["sl"]))
                break
        cash_before_fee = cash
        cash -= cash * FEE_PER_SIDE * float(sig["notional"])
        raw = (exit_px - entry) / entry if side > 0 else (entry - exit_px) / entry
        pnl_frac = raw * float(sig["notional"]) - FEE_PER_SIDE * float(sig["notional"])
        before = cash
        cash *= 1.0 + pnl_frac
        peak = max(peak, cash)
        mdd = min(mdd, cash / peak - 1.0)
        ledger.append(
            {
                "trade_id": trade_id,
                "side": "LONG" if side > 0 else "SHORT",
                "entry_time": rows[entry_i]["timestamp"],
                "exit_time": rows[exit_i]["timestamp"],
                "entry_price": entry,
                "exit_price": exit_px,
                "horizon_bars": horizon,
                "hold_bars": exit_i - entry_i,
                "notional_exposure": sig["notional"],
                "margin_fraction": sig["margin_fraction"],
                "execution_leverage": LEVERAGE,
                "tp_price_move": sig["tp"],
                "sl_price_move": sig["sl"],
                "take_profit": float(sig["tp"]) * float(sig["notional"]),
                "stop_loss": float(sig["sl"]) * float(sig["notional"]),
                "gross_raw_ret": raw,
                "net_trade_return_pct": pnl_frac * 100.0,
                "cash_before": cash_before_fee,
                "cash_after": cash,
                "mfe_pct": mfe * 100.0,
                "mae_pct": mae * 100.0,
                "exit_reason": reason,
                "model_count": sig["model_count"],
                "model_avg_utility": sig["model_avg_utility"],
            }
        )
        trade_id += 1
        i = exit_i + 1
    wins = sum(1 for r in ledger if float(r["net_trade_return_pct"]) > 0.0)
    days = max(1.0, len(rows) * 5.0 / 1440.0)
    holds = [int(r["hold_bars"]) for r in ledger]
    reasons = Counter(str(r["exit_reason"]) for r in ledger)
    horizons = Counter(str(r["horizon_bars"]) for r in ledger)
    metrics = {
        "pnl_pct": (cash - 1.0) * 100.0,
        "mdd_pct": mdd * 100.0,
        "wr": wins / len(ledger) if ledger else 0.0,
        "trades": len(ledger),
        "trades_per_day": len(ledger) / days,
        "avg_hold_bars": sum(holds) / len(holds) if holds else 0.0,
        "median_hold_bars": median(holds) if holds else 0.0,
        "max_hold_bars": max(holds) if holds else 0,
        "exit_reasons": dict(reasons),
        "horizon_counts": dict(horizons),
    }
    return metrics, ledger


def _score(m: dict[str, Any]) -> float:
    tpd = float(m["trades_per_day"])
    pnl = float(m["pnl_pct"])
    mdd = abs(float(m["mdd_pct"]))
    avg_hold = float(m["avg_hold_bars"])
    return pnl + min(tpd, 2.0) * 5.0 - max(0.0, mdd - 12.0) * 6.0 - max(0.0, 1.0 - tpd) * 40.0 - avg_hold * 0.04


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read_csv(TRAIN_CSV)
    raw_oos = _read_csv(EVAL_CSV)
    labels = {r["timestamp"]: r for r in _read_csv(LABEL_2025)}
    train_raw = [r for r in raw_2025 if r["timestamp"] < SPLIT_TS]
    val_raw = [r for r in raw_2025 if r["timestamp"] >= SPLIT_TS]
    train_atr = sorted(_atr_pct(train_raw))
    vol_q = (train_atr[int(len(train_atr) * 0.33)], train_atr[int(len(train_atr) * 0.66)])
    train_rows = _feature_rows(train_raw, vol_q=vol_q)
    val_rows = _feature_rows(val_raw, vol_q=vol_q)
    oos_rows = _feature_rows(raw_oos, vol_q=vol_q)
    val_parents = _load_parent("validation")
    oos_parents = _load_parent("oos")
    ranking: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_model: dict[tuple[int, tuple[Any, ...]], dict[str, Any]] | None = None
    selected_val: dict[str, Any] | None = None
    grid = itertools.product(
        [64, 128],
        [0.00040, 0.00080],
        [0.00040, 0.00080],
        [48],
        [0.60, 0.65],
        [0.025, 0.050],
    )
    for min_bucket_count, min_bucket_utility, min_label_utility, max_horizon_train, parent_min_quality, margin_fraction in grid:
        cfg = {
            "min_bucket_count": min_bucket_count,
            "min_bucket_utility": min_bucket_utility,
            "min_label_utility": min_label_utility,
            "max_horizon_train": max_horizon_train,
            "parent_min_quality": parent_min_quality,
            "margin_fraction": margin_fraction,
        }
        model = _fit_policy(train_rows, labels, cfg)
        if not model:
            continue
        val_m, _ = _run(val_rows, val_parents, model, cfg)
        row = {**cfg, **{f"val_{k}": v for k, v in val_m.items() if not isinstance(v, dict)}}
        row["model_buckets"] = len(model)
        row["validation_score"] = _score(val_m)
        ranking.append(row)
        if selected is None or float(row["validation_score"]) > float(selected["validation_score"]):
            selected = row
            selected_model = model
            selected_val = val_m
    if selected is None or selected_model is None or selected_val is None:
        raise RuntimeError("no selectable smoke policy")
    ranking.sort(key=lambda r: (float(r["validation_score"]), float(r["val_pnl_pct"])), reverse=True)
    _write_csv(OUT_DIR / "smoke_validation_ranking.csv", ranking)
    selected_cfg = {k: selected[k] for k in ("min_bucket_count", "min_bucket_utility", "min_label_utility", "max_horizon_train", "parent_min_quality", "margin_fraction")}
    selected_cfg = {
        "min_bucket_count": int(selected_cfg["min_bucket_count"]),
        "min_bucket_utility": float(selected_cfg["min_bucket_utility"]),
        "min_label_utility": float(selected_cfg["min_label_utility"]),
        "max_horizon_train": int(selected_cfg["max_horizon_train"]),
        "parent_min_quality": float(selected_cfg["parent_min_quality"]),
        "margin_fraction": float(selected_cfg["margin_fraction"]),
    }
    val_m, val_ledger = _run(val_rows, val_parents, selected_model, selected_cfg)
    oos_m, oos_ledger = _run(oos_rows, oos_parents, selected_model, selected_cfg)
    _write_csv(OUT_DIR / "selected_smoke_validation_ledger.csv", val_ledger)
    _write_csv(OUT_DIR / "selected_smoke_oos_ledger.csv", oos_ledger)
    report = {
        "model_id": MODEL_ID,
        "status": "smoke_eval_complete",
        "design": "Pure-Python bucket policy trained on 2025 Jan-Sep multi-horizon triple-barrier labels. Validation selects policy; OOS is untouched for selection. Parent prediction is used only as an optional fixed disagreement filter because Jan-Sep parent OOF predictions are unavailable.",
        "selected": selected_cfg,
        "validation": val_m,
        "oos": oos_m,
        "risk_contract": {
            "notional": "margin_fraction * execution_leverage",
            "execution_leverage": LEVERAGE,
            "take_profit": "tp_price_move * notional",
            "stop_loss": "sl_price_move * notional",
        },
        "artifacts": {
            "ranking": str((OUT_DIR / "smoke_validation_ranking.csv").relative_to(ROOT)),
            "validation_ledger": str((OUT_DIR / "selected_smoke_validation_ledger.csv").relative_to(ROOT)),
            "oos_ledger": str((OUT_DIR / "selected_smoke_oos_ledger.csv").relative_to(ROOT)),
        },
        "redteam_notes": [
            "This is a smoke model, not promotion-ready.",
            "Labels use future OHLC only for supervised targets, not runtime features.",
            "Validation, not OOS, selects the configuration.",
            "Runtime-native fresh parent OOF predictions are needed before a TabM promotion pass.",
        ],
    }
    (OUT_DIR / "smoke_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
