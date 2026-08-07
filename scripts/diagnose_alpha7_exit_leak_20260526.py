#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.train_eval_alpha7_meta_fallback_cash_router_20260526 import (  # noqa: E402
    EVAL_CSV,
    TRAIN_CSV,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402


MODEL_ID = "alpha7_exit_leak_diagnosis_20260526"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_exit_leak_diagnosis_20260526"


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _open_price(df: pd.DataFrame, idx: int) -> float:
    row = df.iloc[int(np.clip(idx, 0, len(df) - 1))]
    px = _safe(row, "open", _safe(row, "close", 0.0))
    return float(px)


def _close_series(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(df["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != ACTION_CASH) & (side != 0)


def _simulate_with_ledger(
    df: pd.DataFrame,
    combo_dec: pd.DataFrame,
    primary_dec: pd.DataFrame,
    fallback_dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fee_base = float(fee) * float(cost_mult)
    slip_base = float(slip) * float(cost_mult)
    maker_fee = float(fee_base * 0.20)
    close = _close_series(df)

    primary_active = _active(primary_dec)
    fallback_active = _active(fallback_dec)

    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    notional = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    max_hold = 0
    cooldown = 0
    next_cooldown = 0

    trades = 0
    wins = 0
    exits: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None
    mfe = 0.0
    mae = 0.0

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        if pos > 0:
            raw = (px * (1.0 - slip_base) - entry_price) / max(entry_price, 1e-12)
        else:
            raw = (entry_price - px * (1.0 + slip_base)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = int(i - entry_idx)
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = "max_hold"
            if reason:
                fill_idx = min(int(i) + 1, len(df) - 1)
                exit_price = _open_price(df, fill_idx)  # next_open touch0 maker proxy
                if pos > 0:
                    raw = (exit_price - entry_price) / max(entry_price, 1e-12)
                else:
                    raw = (entry_price - exit_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * maker_fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if open_record is not None:
                    gross_return = raw * notional
                    net_return = cash / max(entry_equity, 1e-12) - 1.0
                    giveback = 0.0
                    if mfe > 1e-12:
                        giveback = float(np.clip((mfe - gross_return) / max(abs(mfe), 1e-12), 0.0, 5.0))
                    out = {
                        **open_record,
                        "exit_idx": int(i),
                        "exit_timestamp": str(df["timestamp"].iloc[i]) if "timestamp" in df.columns else "",
                        "exit_fill_idx": int(fill_idx),
                        "exit_fill_timestamp": str(df["timestamp"].iloc[fill_idx]) if "timestamp" in df.columns else "",
                        "exit_reason": reason,
                        "hold_bars": hold,
                        "mfe_frac": float(mfe),
                        "mae_frac": float(mae),
                        "gross_return_frac": float(gross_return),
                        "net_return_frac": float(net_return),
                        "giveback_ratio": float(giveback),
                        "cash_after": float(cash),
                    }
                    ledger.append(out)
                pos = 0
                cooldown = int(next_cooldown)
                next_cooldown = 0
                open_record = None
                mfe = 0.0
                mae = 0.0
                continue

        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue

        dec = combo_dec.iloc[i]
        if int(pd.to_numeric(dec.get("action", 0), errors="coerce")) == ACTION_CASH:
            continue
        side = int(pd.to_numeric(dec.get("side", 0), errors="coerce"))
        if side == 0:
            continue

        fill_idx = min(int(i) + 1, len(df) - 1)
        px = _open_price(df, fill_idx)  # next_open touch0 maker proxy
        pos = int(side)
        entry_price = float(px)
        entry_equity = float(cash)
        entry_idx = int(i)
        notional = float(np.clip(pd.to_numeric(dec.get("notional_exposure", 0.0), errors="coerce"), 0.0, 10.0))
        take_profit = float(max(pd.to_numeric(dec.get("take_profit", 0.0), errors="coerce"), 0.0))
        stop_loss = float(max(pd.to_numeric(dec.get("stop_loss", 0.0), errors="coerce"), 0.0))
        max_hold = int(max(pd.to_numeric(dec.get("max_hold_bars", 0), errors="coerce"), 0))
        next_cooldown = int(max(pd.to_numeric(dec.get("cooldown_bars", 0), errors="coerce"), 0))
        cash -= cash * maker_fee * notional
        mfe = 0.0
        mae = 0.0
        source = "primary"
        if (not primary_active[i]) and fallback_active[i]:
            source = "fallback"
        open_record = {
            "entry_idx": int(i),
            "entry_timestamp": str(df["timestamp"].iloc[i]) if "timestamp" in df.columns else "",
            "entry_fill_idx": int(fill_idx),
            "entry_fill_timestamp": str(df["timestamp"].iloc[fill_idx]) if "timestamp" in df.columns else "",
            "source": source,
            "side": "LONG" if side > 0 else "SHORT",
            "entry_price": float(entry_price),
            "notional_exposure": float(notional),
            "take_profit": float(take_profit),
            "stop_loss": float(stop_loss),
            "max_hold_bars": int(max_hold),
            "cooldown_bars": int(next_cooldown),
            "quality_score": float(pd.to_numeric(dec.get("quality_score", 0.0), errors="coerce")),
            "confidence": float(pd.to_numeric(dec.get("confidence", 0.0), errors="coerce")),
            "fee_entry_frac": float(maker_fee * notional),
            "fee_model": "maker_fee_mult_0p20",
        }

    if pos != 0 and open_record is not None:
        fill_idx = len(df) - 1
        px = _open_price(df, fill_idx)
        if pos > 0:
            raw = (px * (1.0 - slip_base) - entry_price) / max(entry_price, 1e-12)
        else:
            raw = (entry_price - px * (1.0 + slip_base)) / max(entry_price, 1e-12)
        gross_return = raw * notional
        before = cash
        cash = cash * (1.0 + gross_return)
        cash -= before * fee_base * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
        giveback = 0.0
        if mfe > 1e-12:
            giveback = float(np.clip((mfe - gross_return) / max(abs(mfe), 1e-12), 0.0, 5.0))
        ledger.append(
            {
                **open_record,
                "exit_idx": int(fill_idx),
                "exit_timestamp": str(df["timestamp"].iloc[fill_idx]) if "timestamp" in df.columns else "",
                "exit_fill_idx": int(fill_idx),
                "exit_fill_timestamp": str(df["timestamp"].iloc[fill_idx]) if "timestamp" in df.columns else "",
                "exit_reason": "forced_end",
                "hold_bars": int(fill_idx - entry_idx),
                "mfe_frac": float(mfe),
                "mae_frac": float(mae),
                "gross_return_frac": float(gross_return),
                "net_return_frac": float(cash / max(entry_equity, 1e-12) - 1.0),
                "giveback_ratio": float(giveback),
                "cash_after": float(cash),
            }
        )

    days = max((pd.to_datetime(df["timestamp"]).iloc[-1] - pd.to_datetime(df["timestamp"]).iloc[0]).total_seconds() / 86400.0, 1e-8)
    metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / days),
        "exits": exits,
    }
    return metrics, pd.DataFrame(ledger)


def _exit_leak_stats(ledger: pd.DataFrame, *, fee_frac: float) -> dict[str, Any]:
    if ledger.empty:
        return {"rows": 0}
    df = ledger.copy()
    df["hold_bars"] = pd.to_numeric(df["hold_bars"], errors="coerce").fillna(0).astype(int)
    df["mfe_frac"] = pd.to_numeric(df["mfe_frac"], errors="coerce").fillna(0.0)
    df["mae_frac"] = pd.to_numeric(df["mae_frac"], errors="coerce").fillna(0.0)
    df["take_profit"] = pd.to_numeric(df["take_profit"], errors="coerce").fillna(0.0)
    df["gross_return_frac"] = pd.to_numeric(df["gross_return_frac"], errors="coerce").fillna(0.0)
    df["net_return_frac"] = pd.to_numeric(df["net_return_frac"], errors="coerce").fillna(0.0)
    df["giveback_ratio"] = pd.to_numeric(df["giveback_ratio"], errors="coerce").fillna(0.0)
    df["exit_reason"] = df["exit_reason"].astype(str)

    early_sl = (df["exit_reason"] == "stop_loss") & (df["hold_bars"] <= 3)
    near_tp_but_miss = (df["mfe_frac"] >= 0.8 * df["take_profit"]) & (df["exit_reason"] != "take_profit") & (df["take_profit"] > 0)
    giveback_sl = (df["mfe_frac"] > (3.0 * fee_frac)) & (df["net_return_frac"] < 0.0)

    by_reason = df["exit_reason"].value_counts(dropna=False).to_dict()
    by_source_reason = (
        df.groupby(["source", "exit_reason"], dropna=False).size().rename("count").reset_index().to_dict(orient="records")
    )

    out = {
        "rows": int(len(df)),
        "exit_reason_counts": by_reason,
        "early_sl_ratio": float(early_sl.mean()),
        "near_tp_but_miss_ratio": float(near_tp_but_miss.mean()),
        "giveback_negative_ratio": float(giveback_sl.mean()),
        "avg_hold_bars": float(df["hold_bars"].mean()),
        "avg_mfe_frac": float(df["mfe_frac"].mean()),
        "avg_mae_frac": float(df["mae_frac"].mean()),
        "avg_giveback_ratio": float(df["giveback_ratio"].mean()),
        "avg_net_return_frac": float(df["net_return_frac"].mean()),
        "by_source_exit_reason": by_source_reason,
    }
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Diagnose Alpha7 baseline exit leak on val/OOS windows.")
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=EVAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline = get_live_baseline()

    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)

    primary_parent = joblib.load(baseline.primary_parent)
    fallback_parent = joblib.load(baseline.fallback_parent)
    primary_rt = _load_best_scale_runtime(baseline.primary_summary)
    fallback_rt = _load_best_scale_runtime(baseline.fallback_summary)

    primary_val = _predict_scaled(primary_parent, val_df, primary_rt)
    fallback_val = _predict_scaled(fallback_parent, val_df, fallback_rt)
    combo_val = _combine_primary_fallback(primary_val, fallback_val)

    primary_eval = _predict_scaled(primary_parent, eval_df, primary_rt)
    fallback_eval = _predict_scaled(fallback_parent, eval_df, fallback_rt)
    combo_eval = _combine_primary_fallback(primary_eval, fallback_eval)

    fee = float(primary_parent.get("config", {}).get("fee", 0.0004))
    slip = float(primary_parent.get("config", {}).get("slip", 0.00015))

    val_cost3, val_ledger = _simulate_with_ledger(
        val_df,
        combo_val,
        primary_val,
        fallback_val,
        fee=fee,
        slip=slip,
        cost_mult=3.0,
    )
    eval_cost3, eval_ledger = _simulate_with_ledger(
        eval_df,
        combo_eval,
        primary_eval,
        fallback_eval,
        fee=fee,
        slip=slip,
        cost_mult=3.0,
    )

    fee_frac_cost3 = float(fee * 3.0 * 0.20)
    val_diag = _exit_leak_stats(val_ledger, fee_frac=fee_frac_cost3)
    eval_diag = _exit_leak_stats(eval_ledger, fee_frac=fee_frac_cost3)

    val_ledger_path = args.out_dir / "val_2025q4_cost3_ledger.csv"
    eval_ledger_path = args.out_dir / "oos_2026_cost3_ledger.csv"
    val_ledger.to_csv(val_ledger_path, index=False)
    eval_ledger.to_csv(eval_ledger_path, index=False)

    report = {
        "model_id": MODEL_ID,
        "baseline_model_id": str(baseline.model_id),
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS",
        "cost_contract": {
            "cost_mult": 3.0,
            "entry_exit_fill": "next_open_limit_touch0_maker_proxy",
            "maker_fee_mult": 0.20,
        },
        "metrics": {
            "val_cost3": val_cost3,
            "oos_cost3": eval_cost3,
        },
        "diagnosis": {
            "val_cost3": val_diag,
            "oos_cost3": eval_diag,
        },
        "artifacts": {
            "val_ledger_csv": str(val_ledger_path),
            "oos_ledger_csv": str(eval_ledger_path),
        },
    }
    report_path = args.out_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "oos_cost3": eval_cost3,
                "oos_exit_reasons": eval_diag.get("exit_reason_counts", {}),
                "oos_early_sl_ratio": eval_diag.get("early_sl_ratio"),
                "oos_near_tp_but_miss_ratio": eval_diag.get("near_tp_but_miss_ratio"),
                "oos_giveback_negative_ratio": eval_diag.get("giveback_negative_ratio"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
