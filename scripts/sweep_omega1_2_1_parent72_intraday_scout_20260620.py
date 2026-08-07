#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_1_parent72_intraday_scout_sweep_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
PARENT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"

SPLIT_TS = "2025-10-01"
CORE_THR = {"bull": 0.72, "bear": 0.64, "chop": 0.65, "chop_expert": 0.65}
CORE_BASE_NOTIONAL = {"bull": 0.2925, "bear": 0.405, "chop": 0.405, "chop_expert": 0.405}
CORE_SCALE = 2.0
CORE_CAP = 0.90
CORE_TP_ACCOUNT = 0.052
CORE_SL_ACCOUNT = 0.028
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
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _f(row: dict[str, str], col: str, default: float = 0.0) -> float:
    try:
        return float(row[col])
    except Exception:
        return default


def _load_split(split: str) -> list[dict[str, Any]]:
    if split == "validation":
        raw = [r for r in _read_csv(TRAIN_CSV) if r["timestamp"] >= SPLIT_TS]
        pred = _read_csv(PARENT_DIR / "validation_predictions_2025_true3head.csv")
        prefix = "omega1_regime3_expertdq_oof_"
    elif split == "oos":
        raw = _read_csv(EVAL_CSV)
        pred = _read_csv(PARENT_DIR / "oos_predictions_2026_true3head.csv")
        prefix = "omega1_regime3_expertdq_"
    else:
        raise RuntimeError(split)
    pmap = {r["timestamp"]: r for r in pred}
    rows: list[dict[str, Any]] = []
    for r in raw:
        p = pmap.get(r["timestamp"])
        if p is None:
            continue
        expert = str(p[f"{prefix}router_expert"]).replace("chop_expert", "chop")
        q = _f(p, f"{prefix}quality_for_action")
        direction = int(_f(p, f"{prefix}dir_action"))
        rows.append(
            {
                "timestamp": r["timestamp"],
                "open": _f(r, "open"),
                "high": _f(r, "high"),
                "low": _f(r, "low"),
                "close": _f(r, "close"),
                "expert": expert,
                "quality": q,
                "dir_action": direction,
            }
        )
    if len(rows) < 100:
        raise RuntimeError(f"{split}: too few aligned rows: {len(rows)}")
    return rows


def _side(action: int) -> int:
    if action == 1:
        return 1
    if action == 2:
        return -1
    return 0


def _core_signal(row: dict[str, Any]) -> dict[str, Any] | None:
    side = _side(int(row["dir_action"]))
    if side == 0:
        return None
    expert = str(row["expert"])
    if float(row["quality"]) < CORE_THR.get(expert, 0.65):
        return None
    base = CORE_BASE_NOTIONAL.get(expert, 0.405)
    notional = min(base * CORE_SCALE, CORE_CAP)
    return {
        "layer": "core",
        "side": side,
        "notional": notional,
        "margin_fraction": notional / LEVERAGE,
        "tp_price_move": CORE_TP_ACCOUNT / notional,
        "sl_price_move": CORE_SL_ACCOUNT / notional,
        "take_profit": CORE_TP_ACCOUNT,
        "stop_loss": CORE_SL_ACCOUNT,
        "max_hold": 0,
    }


def _scout_signal(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
    side = _side(int(row["dir_action"]))
    if side == 0:
        return None
    if float(row["quality"]) < float(cfg["scout_quality_threshold"]):
        return None
    notional = float(cfg["scout_notional"])
    tp_move = float(cfg["scout_tp_price_move"])
    sl_move = float(cfg["scout_sl_price_move"])
    return {
        "layer": "scout",
        "side": side,
        "notional": notional,
        "margin_fraction": notional / LEVERAGE,
        "tp_price_move": tp_move,
        "sl_price_move": sl_move,
        "take_profit": tp_move * notional,
        "stop_loss": sl_move * notional,
        "max_hold": int(cfg["scout_max_hold"]),
    }


def _run(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    ledger: list[dict[str, Any]] = []
    i = 0
    trade_id = 1
    while i < len(rows) - 2:
        sig = _core_signal(rows[i])
        if sig is None:
            sig = _scout_signal(rows[i], cfg)
        if sig is None:
            i += 1
            continue
        entry_i = i + 1
        entry = float(rows[entry_i]["open"])
        if entry <= 0.0:
            i += 1
            continue
        side = int(sig["side"])
        notional = float(sig["notional"])
        take_profit = float(sig["take_profit"])
        stop_loss = float(sig["stop_loss"])
        max_hold = int(sig["max_hold"])
        end_i = min(len(rows) - 1, entry_i + max_hold) if max_hold > 0 else len(rows) - 1
        cash -= cash * FEE_PER_SIDE * notional
        exit_i = end_i
        exit_reason = "max_hold" if max_hold > 0 else "forced_end"
        exit_px = float(rows[end_i]["close"])
        mfe = 0.0
        mae = 0.0
        for j in range(entry_i, end_i + 1):
            px = float(rows[j]["close"])
            raw = (px - entry) / entry if side > 0 else (entry - px) / entry
            pnl_frac = raw * notional
            mfe = max(mfe, pnl_frac)
            mae = min(mae, pnl_frac)
            if pnl_frac >= take_profit:
                exit_i = j
                exit_px = px
                exit_reason = "take_profit"
                break
            if pnl_frac <= -abs(stop_loss):
                exit_i = j
                exit_px = px
                exit_reason = "stop_loss"
                break
        raw = (exit_px - entry) / entry if side > 0 else (entry - exit_px) / entry
        pnl_frac = raw * notional - FEE_PER_SIDE * notional
        before = cash
        cash *= 1.0 + pnl_frac
        peak = max(peak, cash)
        mdd = min(mdd, cash / peak - 1.0)
        ledger.append(
            {
                "trade_id": trade_id,
                "layer": sig["layer"],
                "side": "LONG" if side > 0 else "SHORT",
                "entry_time": rows[entry_i]["timestamp"],
                "exit_time": rows[exit_i]["timestamp"],
                "entry_price": entry,
                "exit_price": exit_px,
                "notional_exposure": notional,
                "margin_fraction": sig["margin_fraction"],
                "execution_leverage": LEVERAGE,
                "tp_price_move": sig["tp_price_move"],
                "sl_price_move": sig["sl_price_move"],
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "max_hold_bars": max_hold,
                "gross_raw_ret": raw,
                "net_trade_return_pct": pnl_frac * 100.0,
                "cash_before": before,
                "cash_after": cash,
                "mfe_pct": mfe * 100.0,
                "mae_pct": mae * 100.0,
                "exit_reason": exit_reason,
            }
        )
        trade_id += 1
        i = exit_i + 1
    wins = sum(1 for r in ledger if float(r["net_trade_return_pct"]) > 0.0)
    days = max(1.0, len(rows) * 5.0 / 1440.0)
    core = sum(1 for r in ledger if r["layer"] == "core")
    scout = sum(1 for r in ledger if r["layer"] == "scout")
    metrics = {
        "pnl_pct": (cash - 1.0) * 100.0,
        "mdd_pct": mdd * 100.0,
        "wr": wins / len(ledger) if ledger else 0.0,
        "trades": len(ledger),
        "trades_per_day": len(ledger) / days,
        "core_trades": core,
        "scout_trades": scout,
        "first_time": rows[0]["timestamp"],
        "last_time": rows[-1]["timestamp"],
    }
    return metrics, ledger


def _score(val: dict[str, Any]) -> float:
    # Prefer day-trading frequency, but penalize blown-up drawdown.
    tpd = float(val["trades_per_day"])
    pnl = float(val["pnl_pct"])
    mdd = abs(float(val["mdd_pct"]))
    wr = float(val["wr"])
    freq_bonus = min(tpd, 3.0) * 12.0
    dd_penalty = max(0.0, mdd - 18.0) * 5.0
    low_freq_penalty = max(0.0, 1.0 - tpd) * 30.0
    return pnl + freq_bonus + wr * 10.0 - dd_penalty - low_freq_penalty


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_rows = _load_split("validation")
    oos_rows = _load_split("oos")
    grid = []
    diagnostic = []
    configs = []
    for scout_quality_threshold, scout_tp_price_move, scout_sl_price_move, scout_max_hold, scout_notional in itertools.product(
        [0.50, 0.55, 0.60, 0.65],
        [0.008, 0.012, 0.016, 0.020],
        [0.006, 0.008, 0.010, 0.012],
        [12, 24, 48, 96],
        [0.15, 0.25, 0.35],
    ):
        if scout_sl_price_move >= scout_tp_price_move:
            continue
        cfg = {
            "scout_quality_threshold": scout_quality_threshold,
            "scout_tp_price_move": scout_tp_price_move,
            "scout_sl_price_move": scout_sl_price_move,
            "scout_max_hold": scout_max_hold,
            "scout_notional": scout_notional,
        }
        val_m, _ = _run(val_rows, cfg)
        row = {**cfg, **{f"val_{k}": v for k, v in val_m.items() if k not in {"first_time", "last_time"}}}
        row["validation_score"] = _score(val_m)
        grid.append(row)
        oos_m, _ = _run(oos_rows, cfg)
        diagnostic.append(
            {
                **cfg,
                **{f"val_{k}": v for k, v in val_m.items() if k not in {"first_time", "last_time"}},
                **{f"oos_{k}": v for k, v in oos_m.items() if k not in {"first_time", "last_time"}},
                "validation_score": row["validation_score"],
            }
        )
        configs.append((cfg, val_m))
    grid.sort(key=lambda r: (float(r["validation_score"]), float(r["val_trades_per_day"]), float(r["val_pnl_pct"])), reverse=True)
    diagnostic.sort(key=lambda r: (float(r["oos_pnl_pct"]), float(r["oos_trades_per_day"]), float(r["validation_score"])), reverse=True)
    _write_csv(OUT_DIR / "validation_only_scout_ranking.csv", grid)
    _write_csv(OUT_DIR / "all_candidates_oos_diagnostic.csv", diagnostic)
    selected = {
        k: grid[0][k]
        for k in ("scout_quality_threshold", "scout_tp_price_move", "scout_sl_price_move", "scout_max_hold", "scout_notional")
    }
    val_m, val_ledger = _run(val_rows, selected)
    oos_m, oos_ledger = _run(oos_rows, selected)
    _write_csv(OUT_DIR / "selected_validation_ledger.csv", val_ledger)
    _write_csv(OUT_DIR / "selected_oos_ledger.csv", oos_ledger)
    report = {
        "model_id": MODEL_ID,
        "design": "Parent72 core preserved. Intraday scout is allowed only when the core parent signal is CASH, using lower notional, short price-move TP/SL, and max_hold.",
        "selection": {
            "rule": "validation_only max validation_score",
            "selected": selected,
            "oos_used_for_selection": False,
        },
        "risk_contract": {
            "notional": "margin_fraction * execution_leverage",
            "take_profit": "tp_price_move * notional",
            "stop_loss": "sl_price_move * notional",
            "execution_leverage": LEVERAGE,
        },
        "validation": val_m,
        "oos": oos_m,
        "artifacts": {
            "ranking": str((OUT_DIR / "validation_only_scout_ranking.csv").relative_to(ROOT)),
            "oos_diagnostic": str((OUT_DIR / "all_candidates_oos_diagnostic.csv").relative_to(ROOT)),
            "validation_ledger": str((OUT_DIR / "selected_validation_ledger.csv").relative_to(ROOT)),
            "oos_ledger": str((OUT_DIR / "selected_oos_ledger.csv").relative_to(ROOT)),
        },
        "redteam_notes": [
            "Screening replay uses existing parent predictions and OHLC frames; no fresh untouched post-2026-02 period is claimed.",
            "Scout layer is parent-CASH-only to preserve core decision ownership.",
            "Promotion requires runtime-native replay and forward/fresh OOS.",
        ],
    }
    _write_json(OUT_DIR / "report.json", report)
    print(json.dumps({"report": str((OUT_DIR / "report.json").relative_to(ROOT)), "selected": selected, "validation": val_m, "oos": oos_m}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
