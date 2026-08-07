#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as legacy_runner  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402


MODEL_ID = "omega1_2_1_tp_runner_clean_repair_20260613"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LIVE_DIR = ROOT / "data/ensemble/supervised" / MODEL_ID


@dataclass(frozen=True)
class RunnerConfig:
    candidate_id: int
    mode: str
    quality_min: float
    extend_mult: float
    floor_frac: float
    max_extensions: int


@dataclass
class CleanPosition:
    side: int = 0
    entry_signal_i: int = 0
    entry_i: int = 0
    entry_price: float = 0.0
    entry_equity: float = 1.0
    notional: float = 0.0
    margin_notional: float = 0.0
    leverage: float = 1.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    floor_unreal: float = -1.0
    mfe: float = 0.0
    mae: float = 0.0
    extensions: int = 0


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


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64)
        for c in ("open", "high", "low", "close")
    }


def _exec_price(px: float, side: int, slip_eff: float, *, entry: bool) -> float:
    if side > 0:
        return float(px) * (1.0 + slip_eff if entry else 1.0 - slip_eff)
    return float(px) * (1.0 - slip_eff if entry else 1.0 + slip_eff)


def _unreal_at_price(exec_px: float, pos: CleanPosition) -> float:
    raw = (float(exec_px) - pos.entry_price) / max(pos.entry_price, 1e-12)
    if pos.side < 0:
        raw = -raw
    return float(raw * pos.notional)


def _close_unreal(arrays: dict[str, np.ndarray], pos: CleanPosition, i: int, slip_eff: float) -> float:
    px = _exec_price(float(arrays["close"][int(i)]), pos.side, slip_eff, entry=False)
    return _unreal_at_price(px, pos)


def _bar_best_worst(arrays: dict[str, np.ndarray], pos: CleanPosition, i: int, slip_eff: float) -> tuple[float, float]:
    high = float(arrays["high"][int(i)])
    low = float(arrays["low"][int(i)])
    if pos.side > 0:
        best_px = _exec_price(high, pos.side, slip_eff, entry=False)
        worst_px = _exec_price(low, pos.side, slip_eff, entry=False)
    else:
        best_px = _exec_price(low, pos.side, slip_eff, entry=False)
        worst_px = _exec_price(high, pos.side, slip_eff, entry=False)
    return _unreal_at_price(best_px, pos), _unreal_at_price(worst_px, pos)


def _exit_price_from_equity_return(pos: CleanPosition, target_unreal: float) -> float:
    raw = float(target_unreal) / max(pos.notional, 1e-12)
    if pos.side > 0:
        return float(pos.entry_price * (1.0 + raw))
    return float(pos.entry_price * (1.0 - raw))


def _runtime_close(cash: float, pos: CleanPosition, *, exit_px: float, fee_eff: float) -> tuple[float, float]:
    raw = (float(exit_px) - pos.entry_price) / max(pos.entry_price, 1e-12)
    if pos.side < 0:
        raw = -raw
    before = float(cash)
    new_cash = before * (1.0 + raw * pos.notional)
    new_cash -= before * float(fee_eff) * pos.notional
    net_pct = (new_cash / max(pos.entry_equity, 1e-12) - 1.0) * 100.0
    return float(new_cash), float(net_pct)


def _runner_allowed(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    pos: CleanPosition,
    i: int,
    cfg: RunnerConfig,
) -> bool:
    if cfg.mode == "baseline":
        return False
    if cfg.mode == "none":
        return True
    row = state.iloc[int(i)]
    quality = float(row.get("tabm_quality_for_action", 0.0))
    if quality < float(cfg.quality_min):
        return False
    close = pd.to_numeric(frame["close"], errors="raise")
    ret3 = float(close.pct_change(3).iloc[int(i)] if int(i) >= 3 else 0.0)
    ret6 = float(close.pct_change(6).iloc[int(i)] if int(i) >= 6 else 0.0)
    side_mom3 = ret3 * float(pos.side)
    side_mom6 = ret6 * float(pos.side)
    if cfg.mode == "mom3":
        return side_mom3 > 0.0
    if cfg.mode == "mom6":
        return side_mom6 > 0.0
    if cfg.mode == "mom3_quality":
        return side_mom3 > 0.0
    if cfg.mode == "strong_mom_quality":
        return side_mom3 > 0.0015 and side_mom6 > 0.0
    raise RuntimeError(f"unknown runner mode: {cfg.mode}")


def _metric(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], long_entries: int, short_entries: int) -> dict[str, Any]:
    eq = np.asarray(equity_curve if equity_curve else [1.0], dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12) - 1.0) * 100.0
    arr = np.asarray(trades, dtype=np.float64)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(dd.min()),
        "trades": int(len(arr)),
        "wr": float(np.mean(arr > 0.0)) if len(arr) else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": dict(reasons),
    }


def _ledger_row(frame: pd.DataFrame, pos: CleanPosition, exit_i: int, exit_px: float, cash: float, net_pct: float, reason: str) -> dict[str, Any]:
    return {
        "side": "LONG" if pos.side > 0 else "SHORT",
        "entry_signal_i": int(pos.entry_signal_i),
        "entry_i": int(pos.entry_i),
        "exit_i": int(exit_i),
        "entry_time": str(frame["timestamp"].iloc[int(pos.entry_signal_i)]),
        "exit_time": str(frame["timestamp"].iloc[int(exit_i)]),
        "entry_price": float(pos.entry_price),
        "exit_price": float(exit_px),
        "effective_exposure": float(pos.notional),
        "margin_notional": float(pos.margin_notional),
        "leverage": float(pos.leverage),
        "tp_equity_ret": float(pos.take_profit),
        "sl_equity_ret": float(pos.stop_loss),
        "net_trade_return_pct": float(net_pct),
        "mfe_pct": float(pos.mfe * 100.0),
        "mae_pct": float(pos.mae * 100.0),
        "runner_extensions": int(pos.extensions),
        "exit_reason": str(reason),
        "cash_after": float(cash),
    }


def _simulate_clean(payload: dict[str, Any], cfg: RunnerConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = payload["frame"]
    dec = payload["dec"]
    state = payload["state"]
    arrays = _arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(payload["fee"]) * 3.0
    slip_eff = float(payload["slip"]) * 3.0

    cash = 1.0
    equity_curve: list[float] = [cash]
    trades: list[float] = []
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    long_entries = 0
    short_entries = 0
    pos = CleanPosition()

    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            close_unreal = _close_unreal(arrays, pos, i, slip_eff)
            best_unreal, worst_unreal = _bar_best_worst(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, best_unreal, close_unreal)
            pos.mae = min(pos.mae, worst_unreal, close_unreal)
            equity_curve.append(cash * (1.0 + close_unreal))

            reason = ""
            target_unreal: float | None = None
            if pos.floor_unreal > -abs(pos.stop_loss) and worst_unreal <= pos.floor_unreal:
                reason = "runner_profit_lock_exit"
                target_unreal = float(pos.floor_unreal)
            elif pos.stop_loss > 0.0 and worst_unreal <= -abs(pos.stop_loss):
                reason = "stop_loss"
                target_unreal = -abs(float(pos.stop_loss))
            elif pos.take_profit > 0.0 and best_unreal >= pos.take_profit:
                can_extend = (
                    int(cfg.max_extensions) > 0
                    and pos.extensions < int(cfg.max_extensions)
                    and _runner_allowed(frame, state, pos, i, cfg)
                )
                if can_extend:
                    pos.extensions += 1
                    old_tp = float(pos.take_profit)
                    pos.floor_unreal = max(float(pos.floor_unreal), old_tp * float(cfg.floor_frac))
                    pos.take_profit = old_tp * float(cfg.extend_mult)
                else:
                    reason = "take_profit"
                    target_unreal = float(pos.take_profit)

            if reason and target_unreal is not None:
                exit_px = _exit_price_from_equity_return(pos, target_unreal)
                close_pos = CleanPosition(**pos.__dict__)
                cash, net_pct = _runtime_close(cash, close_pos, exit_px=exit_px, fee_eff=fee_eff)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(_ledger_row(frame, close_pos, i, exit_px, cash, net_pct, reason))
                pos = CleanPosition()
            continue

        equity_curve.append(cash)
        if not bool(active[i]):
            continue
        row = dec.iloc[int(i)]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        fill_i = min(int(i) + 1, len(frame) - 1)
        entry_px = _exec_price(float(arrays["open"][fill_i]), side, slip_eff, entry=True)
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        if notional <= 0.0:
            continue
        cash -= cash * fee_eff * notional
        pos = CleanPosition(
            side=side,
            entry_signal_i=int(i),
            entry_i=int(fill_i),
            entry_price=float(entry_px),
            entry_equity=float(cash),
            notional=float(notional),
            margin_notional=float(row.get("position_fraction", 0.0) or 0.0),
            leverage=float(row.get("leverage", 1.0) or 1.0),
            take_profit=float(row.get("take_profit", 0.0) or 0.0),
            stop_loss=abs(float(row.get("stop_loss", 0.0) or 0.0)),
            floor_unreal=-abs(float(row.get("stop_loss", 0.0) or 0.0)),
        )
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    if pos.side != 0:
        exit_i = len(frame) - 1
        exit_px = _exec_price(float(arrays["close"][exit_i]), pos.side, slip_eff, entry=False)
        close_pos = CleanPosition(**pos.__dict__)
        cash, net_pct = _runtime_close(cash, close_pos, exit_px=exit_px, fee_eff=fee_eff)
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(_ledger_row(frame, close_pos, exit_i, exit_px, cash, net_pct, "forced_end"))

    return _metric(cash, equity_curve, trades, reasons, long_entries, short_entries), pd.DataFrame(rows)


def _configs() -> list[RunnerConfig]:
    cfgs = [RunnerConfig(0, "baseline", 0.0, 1.0, 0.0, 0)]
    idx = 1
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
        cfgs.append(RunnerConfig(idx, str(mode), float(quality_min), float(extend_mult), float(floor_frac), int(max_extensions)))
        idx += 1
    return cfgs


def _row(prefix: str, m: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(m["pnl"]),
        f"{prefix}_mdd": float(m["mdd"]),
        f"{prefix}_wr": float(m["wr"]),
        f"{prefix}_trades": int(m["trades"]),
        f"{prefix}_long": int(m["long_entries"]),
        f"{prefix}_short": int(m["short_entries"]),
        f"{prefix}_reasons": m["exit_reasons"],
    }


def _score_validation(row: pd.Series, base_row: pd.Series) -> float:
    pnl = float(row["val_pnl"])
    mdd = float(row["val_mdd"])
    wr = float(row["val_wr"])
    trades = float(row["val_trades"])
    trade_penalty = 0.0 if trades >= max(10.0, float(base_row["val_trades"]) * 0.80) else -50.0
    return float(pnl + 0.40 * mdd + 15.0 * wr + trade_penalty)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = legacy_runner._build()
    rows: list[dict[str, Any]] = []
    ledgers: dict[tuple[str, int], pd.DataFrame] = {}

    for cfg in _configs():
        result: dict[str, Any] = {
            "candidate_id": int(cfg.candidate_id),
            "mode": cfg.mode,
            "quality_min": float(cfg.quality_min),
            "extend_mult": float(cfg.extend_mult),
            "floor_frac": float(cfg.floor_frac),
            "max_extensions": int(cfg.max_extensions),
        }
        for split in ("validation", "oos"):
            metrics, ledger = _simulate_clean(data[split], cfg)
            result.update(_row(split[:3], metrics))
            if split == "validation" or cfg.candidate_id == 0:
                ledgers[(split, cfg.candidate_id)] = ledger
        rows.append(result)

    ranking = pd.DataFrame(rows)
    base_row = ranking[ranking["candidate_id"] == 0].iloc[0].copy()
    ranking["delta_val_pnl"] = ranking["val_pnl"] - float(base_row["val_pnl"])
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(base_row["oos_pnl"])
    ranking["selection_score_val_only"] = ranking.apply(lambda r: _score_validation(r, base_row), axis=1)
    ranking = ranking.sort_values(["selection_score_val_only", "val_pnl", "val_mdd"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "validation_only_ranking.csv", index=False)

    selected = ranking.iloc[0].to_dict()
    selected_cfg = RunnerConfig(
        int(selected["candidate_id"]),
        str(selected["mode"]),
        float(selected["quality_min"]),
        float(selected["extend_mult"]),
        float(selected["floor_frac"]),
        int(selected["max_extensions"]),
    )

    artifacts: dict[str, str] = {}
    for split in ("validation", "oos"):
        metrics, ledger = _simulate_clean(data[split], selected_cfg)
        path = OUT_DIR / f"{split}_selected_ledger.csv"
        ledger.to_csv(path, index=False)
        artifacts[f"{split}_selected_ledger"] = str(path)
        selected.update(_row(f"{split}_selected", metrics))

    for split in ("validation", "oos"):
        path = OUT_DIR / f"{split}_baseline_ledger.csv"
        ledgers[(split, 0)].to_csv(path, index=False)
        artifacts[f"{split}_baseline_ledger"] = str(path)

    promotable = bool(
        float(selected["val_pnl"]) >= float(base_row["val_pnl"])
        and float(selected["val_mdd"]) >= float(base_row["val_mdd"]) * 1.25
        and int(selected["val_trades"]) >= max(10, int(float(base_row["val_trades"]) * 0.80))
    )

    manifest = {
        "model_id": MODEL_ID,
        "status": "clean_repair_candidate_shadow_required" if promotable else "clean_repair_research_only",
        "selection_policy": "validation_only_no_oos_selection",
        "accounting_policy": "next_open_taker_entry_intrabar_high_low_price_barrier_exit_taker_fee_cost3",
        "oos_usage": "reported_once_after_validation_selection",
        "selected_config": {
            "candidate_id": int(selected_cfg.candidate_id),
            "mode": selected_cfg.mode,
            "quality_min": float(selected_cfg.quality_min),
            "extend_mult": float(selected_cfg.extend_mult),
            "floor_frac": float(selected_cfg.floor_frac),
            "max_extensions": int(selected_cfg.max_extensions),
        },
        "validation": {
            "pnl_pct": float(selected["val_pnl"]),
            "mdd_pct": float(selected["val_mdd"]),
            "wr": float(selected["val_wr"]),
            "trades": int(selected["val_trades"]),
        },
        "oos": {
            "pnl_pct": float(selected["oos_pnl"]),
            "mdd_pct": float(selected["oos_mdd"]),
            "wr": float(selected["oos_wr"]),
            "trades": int(selected["oos_trades"]),
        },
        "baseline_clean_accounting": {
            "validation": {
                "pnl_pct": float(base_row["val_pnl"]),
                "mdd_pct": float(base_row["val_mdd"]),
                "wr": float(base_row["val_wr"]),
                "trades": int(base_row["val_trades"]),
            },
            "oos": {
                "pnl_pct": float(base_row["oos_pnl"]),
                "mdd_pct": float(base_row["oos_mdd"]),
                "wr": float(base_row["oos_wr"]),
                "trades": int(base_row["oos_trades"]),
            },
        },
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "validation_only_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
            **artifacts,
        },
        "promotion_blockers": [] if promotable else ["selected validation-only candidate did not beat clean-accounting baseline promotion thresholds"],
    }

    (OUT_DIR / "report.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DIR / "baseline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": manifest["selected_config"], "validation": manifest["validation"], "oos": manifest["oos"], "status": manifest["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
