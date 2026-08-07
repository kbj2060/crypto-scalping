#!/usr/bin/env python3
from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_1_tp_runner_20260610"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


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


def _metric(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], long_entries: int, short_entries: int) -> dict[str, Any]:
    eq = np.asarray(equity_curve if equity_curve else [1.0], dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12) - 1.0) * 100.0
    arr = np.asarray(trades, dtype=np.float64)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(dd.min()),
        "trades": int(len(trades)),
        "wr": float(np.mean(arr > 0.0)) if len(arr) else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": dict(reasons),
    }


def _runner_allowed(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    pos: base.Position,
    i: int,
    *,
    mode: str,
    quality_min: float,
) -> bool:
    if mode == "none":
        return True
    row = state.iloc[int(i)]
    quality = float(row.get("tabm_quality_for_action", 0.0))
    if quality < float(quality_min):
        return False
    close = pd.to_numeric(frame["close"], errors="raise")
    ret3 = float(close.pct_change(3).iloc[int(i)] if int(i) >= 3 else 0.0)
    ret6 = float(close.pct_change(6).iloc[int(i)] if int(i) >= 6 else 0.0)
    side_mom = ret3 * float(pos.side)
    side_mom6 = ret6 * float(pos.side)
    if mode == "mom3":
        return side_mom > 0.0
    if mode == "mom6":
        return side_mom6 > 0.0
    if mode == "mom3_quality":
        return side_mom > 0.0
    if mode == "strong_mom_quality":
        return side_mom > 0.0015 and side_mom6 > 0.0
    raise RuntimeError(f"unknown runner mode: {mode}")


def _ledger_row(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    pos: base.Position,
    exit_i: int,
    cash: float,
    net_pct: float,
    reason: str,
    runner_extensions: int,
) -> dict[str, Any]:
    return {
        "side": "LONG" if pos.side > 0 else "SHORT",
        "entry_signal_i": int(pos.entry_signal_i),
        "entry_i": int(pos.entry_i),
        "exit_i": int(exit_i),
        "entry_time": str(frame["timestamp"].iloc[int(pos.entry_signal_i)]),
        "exit_time": str(frame["timestamp"].iloc[int(exit_i)]),
        "entry_price": float(pos.entry_price),
        "exit_price": float(arrays["close"][int(exit_i)]),
        "effective_exposure": float(pos.notional),
        "margin_notional": float(pos.margin_notional),
        "leverage": float(pos.leverage),
        "tp_equity_ret": float(pos.take_profit),
        "sl_equity_ret": float(pos.stop_loss),
        "net_trade_return_pct": float(net_pct),
        "mfe_pct": float(pos.mfe * 100.0),
        "mae_pct": float(pos.mae * 100.0),
        "runner_extensions": int(runner_extensions),
        "exit_reason": str(reason),
        "cash_after": float(cash),
    }


def _simulate_tp_runner(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    mode: str,
    quality_min: float,
    extend_mult: float,
    floor_frac: float,
    max_extensions: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)

    cash = 1.0
    equity_curve: list[float] = [cash]
    trades: list[float] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    pos = base.Position()
    long_entries = 0
    short_entries = 0
    runner_extensions = 0
    base_tp = 0.0

    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            equity_curve.append(cash * (1.0 + unreal))

            reason = ""
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                can_extend = (
                    int(max_extensions) > 0
                    and runner_extensions < int(max_extensions)
                    and _runner_allowed(frame, state, pos, i, mode=mode, quality_min=float(quality_min))
                )
                if can_extend:
                    runner_extensions += 1
                    old_tp = float(pos.take_profit)
                    pos.floor_unreal = max(float(pos.floor_unreal), old_tp * float(floor_frac))
                    pos.take_profit = old_tp * float(extend_mult)
                    reason = ""
                else:
                    reason = "take_profit"
            elif pos.floor_unreal > -abs(pos.stop_loss) and unreal <= pos.floor_unreal:
                reason = "runner_profit_lock_exit"
            elif pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
                reason = "stop_loss"

            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(_ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason, runner_extensions))
                runner_extensions = 0
                base_tp = 0.0
            continue

        equity_curve.append(cash)
        if not bool(active[i]):
            continue
        before_side = int(dec.iloc[int(i)].get("side", 0) or 0)
        cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
        if entered:
            long_entries += int(before_side > 0)
            short_entries += int(before_side < 0)
            runner_extensions = 0
            base_tp = float(pos.take_profit)

    if pos.side != 0:
        close_pos = base.Position(**pos.__dict__)
        cash, pos, _ = base._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(_ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end", runner_extensions))

    return _metric(cash, equity_curve, trades, reasons, long_entries, short_entries), pd.DataFrame(rows)


def _build() -> dict[str, dict[str, Any]]:
    fee, slip = base.omega._load_fee_slip()
    splits = base._build_splits()
    out: dict[str, dict[str, Any]] = {}
    for split, payload in splits.items():
        dec = base._to_decisions(payload["src"], payload["prefix"], oof=payload["oof"], thresholds=base.HIGH_THRESHOLDS)
        state = base._state_base(payload["frame"], payload["src"], dec, payload["prefix"])
        out[split] = {"frame": payload["frame"], "dec": dec, "state": state, "fee": fee, "slip": slip}
    return out


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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _build()
    rows: list[dict[str, Any]] = []

    baseline_cfg = {"mode": "baseline", "quality_min": 0.0, "extend_mult": 1.0, "floor_frac": 0.0, "max_extensions": 0}
    configs = [baseline_cfg]
    for mode, quality_min, extend_mult, floor_frac, max_extensions in product(
        ("none", "mom3", "mom6", "mom3_quality", "strong_mom_quality"),
        (0.0, 0.62, 0.70),
        (1.20, 1.35, 1.50, 1.75, 2.00),
        (0.45, 0.60, 0.75, 0.90),
        (1, 2),
    ):
        if mode in {"none", "mom3", "mom6"} and quality_min != 0.0:
            continue
        if mode in {"mom3_quality", "strong_mom_quality"} and quality_min == 0.0:
            continue
        configs.append(
            {
                "mode": mode,
                "quality_min": float(quality_min),
                "extend_mult": float(extend_mult),
                "floor_frac": float(floor_frac),
                "max_extensions": int(max_extensions),
            }
        )

    for idx, cfg in enumerate(configs):
        result: dict[str, Any] = {"candidate_id": int(idx), **cfg}
        ledgers: dict[str, pd.DataFrame] = {}
        for split in ("validation", "oos"):
            payload = data[split]
            metrics, ledger = _simulate_tp_runner(
                payload["frame"],
                payload["dec"],
                payload["state"],
                fee=float(payload["fee"]),
                slip=float(payload["slip"]),
                cost_mult=3.0,
                mode="none" if cfg["mode"] == "baseline" else str(cfg["mode"]),
                quality_min=float(cfg["quality_min"]),
                extend_mult=float(cfg["extend_mult"]),
                floor_frac=float(cfg["floor_frac"]),
                max_extensions=int(cfg["max_extensions"]),
            )
            result.update(_row(split[:3], metrics))
            ledgers[split] = ledger
        rows.append(result)
        if idx == 0:
            ledgers["validation"].to_csv(OUT_DIR / "validation_baseline_ledger.csv", index=False)
            ledgers["oos"].to_csv(OUT_DIR / "oos_baseline_ledger.csv", index=False)

    ranking = pd.DataFrame(rows)
    base_row = ranking[ranking["mode"] == "baseline"].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(base_row["oos_pnl"])
    ranking["delta_val_pnl"] = ranking["val_pnl"] - float(base_row["val_pnl"])
    ranking["score"] = ranking["oos_pnl"] + 0.45 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.25 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "tp_runner_grid.csv", index=False)
    promotable = ranking[
        (ranking["mode"] != "baseline")
        & (ranking["oos_pnl"] > float(base_row["oos_pnl"]))
        & (ranking["val_pnl"] > float(base_row["val_pnl"]) * 0.80)
        & (ranking["oos_mdd"] >= float(base_row["oos_mdd"]) * 1.35)
    ].copy()
    promotable.to_csv(OUT_DIR / "tp_runner_promotable.csv", index=False)

    for rank, row in ranking.head(10).iterrows():
        if str(row["mode"]) == "baseline":
            continue
        tag = f"rank{rank+1:02d}_id{int(row['candidate_id'])}"
        cfg = row.to_dict()
        for split in ("validation", "oos"):
            payload = data[split]
            _metrics, ledger = _simulate_tp_runner(
                payload["frame"],
                payload["dec"],
                payload["state"],
                fee=float(payload["fee"]),
                slip=float(payload["slip"]),
                cost_mult=3.0,
                mode=str(cfg["mode"]),
                quality_min=float(cfg["quality_min"]),
                extend_mult=float(cfg["extend_mult"]),
                floor_frac=float(cfg["floor_frac"]),
                max_extensions=int(cfg["max_extensions"]),
            )
            ledger.to_csv(OUT_DIR / f"{split}_{tag}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Winner-extension overlay. Entry/risk owner remains omega1_2_1_true_leverage_price_barrier_scale200_cap090.",
        "baseline": base_row.to_dict(),
        "promotable_count": int(len(promotable)),
        "top": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "tp_runner_grid.csv"),
            "promotable": str(OUT_DIR / "tp_runner_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top5": ranking.head(5).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
