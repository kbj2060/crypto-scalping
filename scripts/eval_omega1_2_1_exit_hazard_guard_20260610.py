#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_exposure_selector_20260606 as exposure  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_exit_hazard_guard_20260610"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

BASE_TP = 0.026
BASE_SL = 0.014
COMPENSATED_SCALE = 2.0
MARGIN_CAP = 0.90
TRUE_LEVERAGE = 2.0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _apply_true_leverage_price_barrier(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = np.flatnonzero(omega._active(out))
    if len(active) == 0:
        return out

    base_notional = pd.to_numeric(out.loc[active, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    margin_notional = np.minimum(base_notional * COMPENSATED_SCALE, MARGIN_CAP)
    ratio = margin_notional / np.maximum(base_notional, 1e-12)
    effective_exposure = margin_notional * TRUE_LEVERAGE
    barrier_scale = ratio * TRUE_LEVERAGE

    out.loc[active, "notional_exposure"] = effective_exposure
    out.loc[active, "position_fraction"] = margin_notional
    out.loc[active, "leverage"] = TRUE_LEVERAGE
    out.loc[active, "take_profit"] = BASE_TP * barrier_scale
    out.loc[active, "stop_loss"] = BASE_SL * barrier_scale
    out.loc[active, "max_hold_bars"] = 0
    out.loc[active, "cooldown_bars"] = 0
    return out


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}


def _build_decisions() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    frames = exposure.th._prepare_frames(disable_tp_sl=False)
    val_frame, _val_src, val_dec, _val_prefix = exposure._build_split(frames, "validation")
    oos_frame, _oos_src, oos_dec, _oos_prefix = exposure._build_split(frames, "oos")
    return {
        "validation": (val_frame, _apply_true_leverage_price_barrier(val_dec)),
        "oos": (oos_frame, _apply_true_leverage_price_barrier(oos_dec)),
    }


def _metric_from_trades(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], long_entries: int, short_entries: int, notional_sum: float, leverage_sum: float) -> dict[str, Any]:
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) == 0:
        eq = np.asarray([1.0], dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12) - 1.0) * 100.0
    n_entries = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(dd.min()),
        "trades": int(len(trades)),
        "wr": float(np.mean(np.asarray(trades, dtype=np.float64) > 0.0)) if trades else 0.0,
        "avg_notional": float(notional_sum / n_entries),
        "avg_leverage": float(leverage_sum / n_entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": dict(reasons),
    }


def _simulate_exit_guard(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    lock_trigger: float,
    lock_floor: float,
    giveback_frac: float,
    emergency_loss: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = _arrays(frame)
    active = np.asarray(omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)

    cash = 1.0
    equity_curve: list[float] = [cash]
    trades: list[float] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}

    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_signal_i = 0
    entry_i = 0
    notional = 0.0
    leverage = 1.0
    margin_notional = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    floor_unreal = 0.0
    mfe = 0.0
    mae = 0.0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0

    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)

            if emergency_loss > 0.0:
                floor_unreal = max(floor_unreal, -abs(float(emergency_loss)))
            if lock_trigger > 0.0 and mfe >= float(lock_trigger):
                floor_unreal = max(floor_unreal, float(lock_floor))

            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif floor_unreal > -abs(stop_loss) and unreal <= floor_unreal:
                reason = "emergency_exit" if floor_unreal < 0.0 else "profit_lock_exit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif giveback_frac > 0.0 and mfe > 0.0 and unreal > 0.0 and mfe >= float(lock_trigger) and ((mfe - unreal) / max(mfe, 1e-12)) >= float(giveback_frac):
                reason = "giveback_model_exit"

            if reason:
                _filled, exit_px, exit_fee, exit_route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                net_pct = float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(
                    {
                        "trade_id": len(rows) + 1,
                        "side": "LONG" if pos > 0 else "SHORT",
                        "entry_signal_i": int(entry_signal_i),
                        "entry_i": int(entry_i),
                        "exit_i": int(i),
                        "entry_time": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                        "exit_time": str(frame["timestamp"].iloc[int(i)]),
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_px),
                        "margin_notional": float(margin_notional),
                        "leverage": float(leverage),
                        "effective_exposure": float(notional),
                        "tp_equity_ret": float(take_profit),
                        "sl_equity_ret": float(stop_loss),
                        "gross_raw_ret": float(raw_exit),
                        "net_trade_return_pct": net_pct,
                        "mfe_pct": float(mfe * 100.0),
                        "mae_pct": float(mae * 100.0),
                        "exit_reason": reason,
                        "exit_route": exit_route,
                        "cash_after": float(cash),
                    }
                )
                pos = 0
                equity_curve.append(cash)
                continue

            equity_curve.append(cash * (1.0 + unreal))
            continue

        equity_curve.append(cash)
        if not bool(active[i]):
            continue

        row = dec.iloc[int(i)]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue

        filled, px, entry_fee, _entry_route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue

        pos = side
        entry_signal_i = int(i)
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_price = float(px)
        entry_equity = cash
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        margin_notional = float(row.get("position_fraction", 0.0) or 0.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = abs(float(row.get("stop_loss", 0.0) or 0.0))
        floor_unreal = -abs(stop_loss)
        mfe = 0.0
        mae = 0.0
        cash -= cash * entry_fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage

    if pos != 0:
        exit_i = len(frame) - 1
        exit_px = omega._fill_price(arrays, exit_i, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        net_pct = float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(
            {
                "trade_id": len(rows) + 1,
                "side": "LONG" if pos > 0 else "SHORT",
                "entry_signal_i": int(entry_signal_i),
                "entry_i": int(entry_i),
                "exit_i": int(exit_i),
                "entry_time": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                "exit_time": str(frame["timestamp"].iloc[int(exit_i)]),
                "entry_price": float(entry_price),
                "exit_price": float(exit_px),
                "margin_notional": float(margin_notional),
                "leverage": float(leverage),
                "effective_exposure": float(notional),
                "tp_equity_ret": float(take_profit),
                "sl_equity_ret": float(stop_loss),
                "gross_raw_ret": float(raw_exit),
                "net_trade_return_pct": net_pct,
                "mfe_pct": float(mfe * 100.0),
                "mae_pct": float(mae * 100.0),
                "exit_reason": "forced_end",
                "exit_route": "forced_end",
                "cash_after": float(cash),
            }
        )
        equity_curve.append(cash)

    metrics = _metric_from_trades(cash, equity_curve, trades, reasons, long_entries, short_entries, notional_sum, leverage_sum)
    return metrics, pd.DataFrame(rows)


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def _entry_audit(base: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    base_entries = set(pd.to_numeric(base.get("entry_signal_i", pd.Series(dtype=int)), errors="coerce").dropna().astype(int).tolist())
    cand_entries = set(pd.to_numeric(candidate.get("entry_signal_i", pd.Series(dtype=int)), errors="coerce").dropna().astype(int).tolist())
    return {
        "base_entries": int(len(base_entries)),
        "candidate_entries": int(len(cand_entries)),
        "shared_entries": int(len(base_entries & cand_entries)),
        "added_entries_after_earlier_exit": int(len(cand_entries - base_entries)),
        "dropped_entries": int(len(base_entries - cand_entries)),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = omega._load_fee_slip()
    splits = _build_decisions()

    grid: list[dict[str, Any]] = []
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}

    variants: list[dict[str, float | str]] = [
        {"variant": "baseline_true_leverage_no_exit_guard", "lock_trigger": 999.0, "lock_floor": -999.0, "giveback_frac": 0.0, "emergency_loss": 999.0},
    ]
    for lock_trigger in (0.025, 0.040, 0.060, 0.080):
        for lock_floor in (0.0, 0.010, 0.020):
            variants.append({"variant": "breakeven_profit_lock", "lock_trigger": lock_trigger, "lock_floor": lock_floor, "giveback_frac": 0.0, "emergency_loss": 999.0})
    for lock_trigger in (0.025, 0.040, 0.060, 0.080):
        for giveback_frac in (0.35, 0.50, 0.65, 0.80):
            variants.append({"variant": "giveback_model_exit", "lock_trigger": lock_trigger, "lock_floor": -999.0, "giveback_frac": giveback_frac, "emergency_loss": 999.0})
    for emergency_loss in (0.025, 0.035, 0.045):
        variants.append({"variant": "emergency_loss_exit", "lock_trigger": 999.0, "lock_floor": -999.0, "giveback_frac": 0.0, "emergency_loss": emergency_loss})
    for lock_trigger in (0.040, 0.060):
        for lock_floor in (0.0, 0.010):
            for emergency_loss in (0.035, 0.045):
                variants.append({"variant": "profit_lock_plus_emergency", "lock_trigger": lock_trigger, "lock_floor": lock_floor, "giveback_frac": 0.0, "emergency_loss": emergency_loss})

    baseline_ledgers: dict[str, pd.DataFrame] = {}
    for variant_id, params in enumerate(variants):
        row: dict[str, Any] = {"variant_id": int(variant_id), **params}
        split_ledgers: dict[str, pd.DataFrame] = {}
        for split, (frame, dec) in splits.items():
            metrics, ledger = _simulate_exit_guard(
                frame,
                dec,
                fee=fee,
                slip=slip,
                cost_mult=3.0,
                lock_trigger=float(params["lock_trigger"]),
                lock_floor=float(params["lock_floor"]),
                giveback_frac=float(params["giveback_frac"]),
                emergency_loss=float(params["emergency_loss"]),
            )
            row.update(_row(split, metrics))
            split_ledgers[split] = ledger
            if variant_id == 0:
                baseline_ledgers[split] = ledger
            else:
                row[f"{split}_entry_audit"] = _entry_audit(baseline_ledgers[split], ledger)
        ledgers[str(variant_id)] = split_ledgers
        grid.append(row)

    ranking = pd.DataFrame(grid)
    ranking["score"] = ranking["validation_pnl"] + 0.50 * ranking["oos_pnl"] + 0.25 * ranking["validation_mdd"] + 0.25 * ranking["oos_mdd"]
    ranking = ranking.sort_values(["validation_pnl", "oos_pnl", "oos_mdd"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "exit_hazard_guard_grid.csv", index=False)

    top_ids = [int(x) for x in ranking["variant_id"].head(5).tolist()]
    for variant_id in [0, *top_ids]:
        for split, ledger in ledgers[str(variant_id)].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{variant_id}_trade_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "baseline_model": "omega1_2_1_true_leverage_price_barrier_scale200_cap090",
        "method": "Exit-only overlay. Omega1.2.1 parent decisions and true-leverage risk contract are frozen; only in-position exit timing may change. Earlier exits can admit later frozen parent signals.",
        "cost_accounting": {
            "fee": fee,
            "slip": slip,
            "cost_mult": 3.0,
            "notional_exposure": "effective account exposure",
            "position_fraction": "margin notional",
            "entry_exit_fee_base": "effective exposure",
        },
        "variants": grid,
        "ranking_top20": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "exit_hazard_guard_grid.csv"),
            "ledgers": "validation_variant{variant_id}_trade_ledger.csv / oos_variant{variant_id}_trade_ledger.csv for baseline and top candidates",
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(ranking.head(15).to_string(index=False))
    print(json.dumps(report["artifacts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
