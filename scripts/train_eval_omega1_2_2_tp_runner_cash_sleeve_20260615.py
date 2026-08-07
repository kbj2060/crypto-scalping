#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as legacy_runner  # noqa: E402
import repair_omega1_2_1_tp_runner_clean_baseline_20260613 as repair  # noqa: E402


MODEL_ID = "omega1_2_2_tp_runner_cash_sleeve_20260615"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASELINE_ID = "omega1_2_1_tp_runner_clean_repair_20260613"
BASELINE_REPORT = ROOT / "tmp/causal_regen_20260516/omega1_2_1_tp_runner_clean_repair_20260613/report.json"
FORBIDDEN_FEATURE_PREFIXES = (
    "teacher_",
    "regime4_pred_",
    "clean_regime4_",
    "clean_regime_2024_unsup_v4_",
)
FORBIDDEN_FEATURE_EXACT = {"tp_sl_action_score"}

ACTION_CASH = 0
ACTION_LONG = 1
ACTION_SHORT = 2


@dataclass(frozen=True)
class SleeveRisk:
    name: str
    take_profit: float
    stop_loss: float
    notional: float
    leverage: float
    max_hold_bars: int


@dataclass
class Position:
    sleeve: str = ""
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
    max_hold_bars: int = 0


RISKS = (
    SleeveRisk("micro_tp010_sl007_n030_h96", 0.010, 0.007, 0.30, 2.0, 96),
    SleeveRisk("base_tp026_sl014_n0405_h192", 0.026, 0.014, 0.405, 2.0, 192),
    SleeveRisk("mid_tp030_sl018_n055_h192", 0.030, 0.018, 0.55, 2.0, 192),
)


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


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="raise").to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    notional = pd.to_numeric(dec["notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    return (action != ACTION_CASH) & (side != 0) & (notional > 0.0)


def _features(payload: dict[str, Any]) -> pd.DataFrame:
    dec = payload["dec"].reset_index(drop=True)
    active = _active(dec)
    out = payload["state"].copy().reset_index(drop=True)
    out["primary_is_cash"] = (~active).astype(float)
    out["primary_active_roll_12"] = pd.Series(active.astype(float)).rolling(12, min_periods=1).mean().to_numpy(dtype=np.float64)
    out["primary_active_roll_48"] = pd.Series(active.astype(float)).rolling(48, min_periods=1).mean().to_numpy(dtype=np.float64)
    streak = np.zeros(len(out), dtype=np.float64)
    cur = 0
    for i, is_cash in enumerate(~active):
        cur = cur + 1 if bool(is_cash) else 0
        streak[i] = cur
    out["primary_cash_streak"] = np.tanh(streak / 144.0)
    bad = [c for c in out.columns if c in FORBIDDEN_FEATURE_EXACT or c.startswith(FORBIDDEN_FEATURE_PREFIXES)]
    if bad:
        raise RuntimeError(f"forbidden sleeve feature columns: {bad[:20]}")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _exec_price(px: float, side: int, slip_eff: float, *, entry: bool) -> float:
    if side > 0:
        return float(px) * (1.0 + slip_eff if entry else 1.0 - slip_eff)
    return float(px) * (1.0 - slip_eff if entry else 1.0 + slip_eff)


def _unreal_at_price(exec_px: float, pos: Position) -> float:
    raw = (float(exec_px) - pos.entry_price) / max(pos.entry_price, 1e-12)
    if pos.side < 0:
        raw = -raw
    return float(raw * pos.notional)


def _close_unreal(arrays: dict[str, np.ndarray], pos: Position, i: int, slip_eff: float) -> float:
    return _unreal_at_price(_exec_price(float(arrays["close"][int(i)]), pos.side, slip_eff, entry=False), pos)


def _bar_best_worst(arrays: dict[str, np.ndarray], pos: Position, i: int, slip_eff: float) -> tuple[float, float]:
    high = float(arrays["high"][int(i)])
    low = float(arrays["low"][int(i)])
    if pos.side > 0:
        return (
            _unreal_at_price(_exec_price(high, pos.side, slip_eff, entry=False), pos),
            _unreal_at_price(_exec_price(low, pos.side, slip_eff, entry=False), pos),
        )
    return (
        _unreal_at_price(_exec_price(low, pos.side, slip_eff, entry=False), pos),
        _unreal_at_price(_exec_price(high, pos.side, slip_eff, entry=False), pos),
    )


def _exit_price_from_unreal(pos: Position, target_unreal: float) -> float:
    raw = float(target_unreal) / max(pos.notional, 1e-12)
    if pos.side > 0:
        return float(pos.entry_price * (1.0 + raw))
    return float(pos.entry_price * (1.0 - raw))


def _runtime_close(cash: float, pos: Position, exit_px: float, fee_eff: float) -> tuple[float, float]:
    raw = (float(exit_px) - pos.entry_price) / max(pos.entry_price, 1e-12)
    if pos.side < 0:
        raw = -raw
    before = float(cash)
    cash = before * (1.0 + raw * pos.notional)
    cash -= before * float(fee_eff) * pos.notional
    net_pct = (cash / max(pos.entry_equity, 1e-12) - 1.0) * 100.0
    return float(cash), float(net_pct)


def _entry(cash: float, arrays: dict[str, np.ndarray], i: int, side: int, fee_eff: float, slip_eff: float, *, sleeve: str, row: pd.Series | None, risk: SleeveRisk | None) -> tuple[float, Position, bool]:
    fill_i = min(int(i) + 1, len(arrays["open"]) - 1)
    entry_px = _exec_price(float(arrays["open"][fill_i]), int(side), slip_eff, entry=True)
    if sleeve == "primary":
        assert row is not None
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        margin = float(row.get("position_fraction", 0.0) or 0.0)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = abs(float(row.get("stop_loss", 0.0) or 0.0))
        max_hold = int(row.get("max_hold_bars", 0) or 0)
    else:
        assert risk is not None
        notional = float(risk.notional)
        margin = float(risk.notional)
        leverage = float(risk.leverage)
        take_profit = float(risk.take_profit)
        stop_loss = abs(float(risk.stop_loss))
        max_hold = int(risk.max_hold_bars)
    if notional <= 0.0:
        return cash, Position(), False
    entry_equity = float(cash)
    cash -= cash * float(fee_eff) * notional
    pos = Position(
        sleeve=sleeve,
        side=int(side),
        entry_signal_i=int(i),
        entry_i=int(fill_i),
        entry_price=float(entry_px),
        entry_equity=entry_equity,
        notional=notional,
        margin_notional=margin,
        leverage=leverage,
        take_profit=take_profit,
        stop_loss=stop_loss,
        floor_unreal=-abs(stop_loss),
        max_hold_bars=max_hold,
    )
    return float(cash), pos, True


def _runner_allowed(frame: pd.DataFrame, state: pd.DataFrame, pos: Position, i: int, cfg: repair.RunnerConfig) -> bool:
    return repair._runner_allowed(frame, state, repair.CleanPosition(**{k: getattr(pos, k) for k in repair.CleanPosition().__dict__.keys()}), i, cfg)


def _metric(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], long_entries: int, short_entries: int, primary_entries: int, fallback_entries: int) -> dict[str, Any]:
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
        "primary_entries": int(primary_entries),
        "fallback_entries": int(fallback_entries),
        "exit_reasons": dict(reasons),
    }


def _simulate_combo(
    payload: dict[str, Any],
    cfg: repair.RunnerConfig,
    risk: SleeveRisk | None,
    fallback_action: np.ndarray | None,
    fallback_conf: np.ndarray | None,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = payload["frame"].reset_index(drop=True)
    dec = payload["dec"].reset_index(drop=True)
    state = payload["state"].reset_index(drop=True)
    arrays = repair._arrays(frame)
    active = _active(dec)
    fee_eff = float(payload["fee"]) * 3.0
    slip_eff = float(payload["slip"]) * 3.0
    cash = 1.0
    equity_curve: list[float] = [cash]
    trades: list[float] = []
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    pos = Position()
    long_entries = short_entries = primary_entries = fallback_entries = 0

    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            close_unreal = _close_unreal(arrays, pos, i, slip_eff)
            best_unreal, worst_unreal = _bar_best_worst(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, best_unreal, close_unreal)
            pos.mae = min(pos.mae, worst_unreal, close_unreal)
            equity_curve.append(cash * (1.0 + close_unreal))
            reason = ""
            target_unreal: float | None = None
            if pos.sleeve == "fallback" and bool(active[i]):
                reason = "fallback_primary_takeover"
                target_unreal = close_unreal
            elif pos.floor_unreal > -abs(pos.stop_loss) and worst_unreal <= pos.floor_unreal:
                reason = f"{pos.sleeve}_runner_profit_lock_exit"
                target_unreal = float(pos.floor_unreal)
            elif pos.stop_loss > 0.0 and worst_unreal <= -abs(pos.stop_loss):
                reason = f"{pos.sleeve}_stop_loss"
                target_unreal = -abs(float(pos.stop_loss))
            elif pos.take_profit > 0.0 and best_unreal >= pos.take_profit:
                if pos.sleeve == "primary":
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
                        reason = "primary_take_profit"
                        target_unreal = float(pos.take_profit)
                else:
                    reason = "fallback_take_profit"
                    target_unreal = float(pos.take_profit)
            elif pos.sleeve == "fallback" and pos.max_hold_bars > 0 and int(i) - int(pos.entry_i) >= int(pos.max_hold_bars):
                reason = "fallback_max_hold"
                target_unreal = close_unreal

            if reason and target_unreal is not None:
                exit_px = _exit_price_from_unreal(pos, target_unreal)
                close_pos = Position(**pos.__dict__)
                cash, net_pct = _runtime_close(cash, close_pos, exit_px, fee_eff)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(
                    {
                        "sleeve": close_pos.sleeve,
                        "side": "LONG" if close_pos.side > 0 else "SHORT",
                        "entry_i": int(close_pos.entry_i),
                        "exit_i": int(i),
                        "entry_time": str(frame["timestamp"].iloc[int(close_pos.entry_signal_i)]),
                        "exit_time": str(frame["timestamp"].iloc[int(i)]),
                        "net_trade_return_pct": float(net_pct),
                        "mfe_pct": float(close_pos.mfe * 100.0),
                        "mae_pct": float(close_pos.mae * 100.0),
                        "runner_extensions": int(close_pos.extensions),
                        "exit_reason": reason,
                        "cash_after": float(cash),
                    }
                )
                pos = Position()
            else:
                continue

        equity_curve.append(cash)
        if bool(active[i]):
            row = dec.iloc[int(i)]
            side = int(row.get("side", 0) or 0)
            cash, pos, entered = _entry(cash, arrays, i, side, fee_eff, slip_eff, sleeve="primary", row=row, risk=None)
            if entered:
                primary_entries += 1
                long_entries += int(side > 0)
                short_entries += int(side < 0)
            continue

        if risk is None or fallback_action is None or fallback_conf is None:
            continue
        action = int(fallback_action[int(i)]) if int(i) < len(fallback_action) else ACTION_CASH
        conf = float(fallback_conf[int(i)]) if int(i) < len(fallback_conf) else 0.0
        if action not in (ACTION_LONG, ACTION_SHORT) or conf < float(threshold):
            continue
        side = 1 if action == ACTION_LONG else -1
        cash, pos, entered = _entry(cash, arrays, i, side, fee_eff, slip_eff, sleeve="fallback", row=None, risk=risk)
        if entered:
            fallback_entries += 1
            long_entries += int(side > 0)
            short_entries += int(side < 0)

    if pos.side != 0:
        exit_px = _exec_price(float(arrays["close"][-1]), pos.side, slip_eff, entry=False)
        close_pos = Position(**pos.__dict__)
        cash, net_pct = _runtime_close(cash, close_pos, exit_px, fee_eff)
        trades.append(net_pct)
        reason = f"{close_pos.sleeve}_forced_end"
        reasons[reason] = reasons.get(reason, 0) + 1
        rows.append({"sleeve": close_pos.sleeve, "side": "LONG" if close_pos.side > 0 else "SHORT", "entry_i": int(close_pos.entry_i), "exit_i": int(len(frame) - 1), "entry_time": str(frame["timestamp"].iloc[int(close_pos.entry_signal_i)]), "exit_time": str(frame["timestamp"].iloc[-1]), "net_trade_return_pct": float(net_pct), "mfe_pct": float(close_pos.mfe * 100.0), "mae_pct": float(close_pos.mae * 100.0), "runner_extensions": int(close_pos.extensions), "exit_reason": reason, "cash_after": float(cash)})

    return _metric(cash, equity_curve, trades, reasons, long_entries, short_entries, primary_entries, fallback_entries), pd.DataFrame(rows)


def _simulate_label(frame: pd.DataFrame, arrays: dict[str, np.ndarray], i: int, side: int, risk: SleeveRisk, fee_eff: float, slip_eff: float) -> float:
    fill_i = min(int(i) + 1, len(frame) - 1)
    entry_px = _exec_price(float(arrays["open"][fill_i]), int(side), slip_eff, entry=True)
    pos = Position(sleeve="label", side=int(side), entry_signal_i=int(i), entry_i=int(fill_i), entry_price=entry_px, entry_equity=1.0, notional=float(risk.notional), margin_notional=float(risk.notional), leverage=float(risk.leverage), take_profit=float(risk.take_profit), stop_loss=abs(float(risk.stop_loss)), floor_unreal=-abs(float(risk.stop_loss)), max_hold_bars=int(risk.max_hold_bars))
    cash = 1.0 - float(fee_eff) * float(risk.notional)
    end_i = min(len(frame) - 2, fill_i + int(risk.max_hold_bars))
    target = 0.0
    for j in range(fill_i, end_i + 1):
        best, worst = _bar_best_worst(arrays, pos, j, slip_eff)
        close_unreal = _close_unreal(arrays, pos, j, slip_eff)
        target = close_unreal
        if worst <= -abs(float(risk.stop_loss)):
            target = -abs(float(risk.stop_loss))
            break
        if best >= float(risk.take_profit):
            target = float(risk.take_profit)
            break
    exit_px = _exit_price_from_unreal(pos, target)
    cash, _net_pct = _runtime_close(cash, pos, exit_px, fee_eff)
    return float(cash - 1.0)


def _labels(payload: dict[str, Any], risk: SleeveRisk, min_edge: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame = payload["frame"].reset_index(drop=True)
    dec = payload["dec"].reset_index(drop=True)
    arrays = repair._arrays(frame)
    active = _active(dec)
    cash_mask = ~active
    fee_eff = float(payload["fee"]) * 3.0
    slip_eff = float(payload["slip"]) * 3.0
    y = np.zeros(len(frame), dtype=np.int64)
    valid = np.zeros(len(frame), dtype=bool)
    best_net = []
    for i in np.flatnonzero(cash_mask):
        if i >= len(frame) - int(risk.max_hold_bars) - 3:
            continue
        valid[int(i)] = True
        long_net = _simulate_label(frame, arrays, int(i), 1, risk, fee_eff, slip_eff)
        short_net = _simulate_label(frame, arrays, int(i), -1, risk, fee_eff, slip_eff)
        if long_net >= short_net:
            best_side, net = ACTION_LONG, long_net
        else:
            best_side, net = ACTION_SHORT, short_net
        best_net.append(float(net))
        if net > float(min_edge):
            y[int(i)] = int(best_side)
    counts = {str(k): int(v) for k, v in pd.Series(y[valid]).value_counts().sort_index().items()}
    return y, valid, {"valid_cash_rows": int(valid.sum()), "label_counts": counts, "best_net_mean": float(np.mean(best_net)) if best_net else 0.0}


def _make_model(name: str, seed: int):
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.035, max_leaf_nodes=7, l2_regularization=2.0, random_state=int(seed))
    if name == "extra":
        return ExtraTreesClassifier(n_estimators=260, max_depth=5, min_samples_leaf=35, class_weight="balanced", random_state=int(seed), n_jobs=-1)
    raise RuntimeError(f"unknown model: {name}")


def _predict_oof(model_name: str, x: pd.DataFrame, y: np.ndarray, mask: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    idx = np.flatnonzero(mask)
    action = np.zeros(len(x), dtype=np.int64)
    conf = np.zeros(len(x), dtype=np.float64)
    folds = []
    n = len(idx)
    for fold_id, (train_frac, end_frac) in enumerate(((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00))):
        train_end = int(n * train_frac)
        val_end = int(n * end_frac)
        if train_end < 100 or val_end <= train_end:
            continue
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        if len(np.unique(y[train_idx])) < 2:
            folds.append({"fold": int(fold_id), "skipped": "single_class", "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx))})
            continue
        model = _make_model(model_name, seed + fold_id)
        model.fit(x.iloc[train_idx].to_numpy(dtype=np.float64), y[train_idx])
        proba = model.predict_proba(x.iloc[val_idx].to_numpy(dtype=np.float64))
        classes = np.asarray(model.classes_, dtype=np.int64)
        best = np.argmax(proba, axis=1)
        action[val_idx] = classes[best]
        conf[val_idx] = proba[np.arange(len(val_idx)), best]
        folds.append({"fold": int(fold_id), "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx)), "classes": classes.tolist()})
    return action, conf, {"folds": folds, "oof_rows": int(np.count_nonzero(conf > 0.0))}


def _fit_predict(model_name: str, x_train: pd.DataFrame, y_train: np.ndarray, mask: np.ndarray, x_eval: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(mask)
    if len(np.unique(y_train[idx])) < 2:
        return np.zeros(len(x_eval), dtype=np.int64), np.zeros(len(x_eval), dtype=np.float64)
    model = _make_model(model_name, seed)
    model.fit(x_train.iloc[idx].to_numpy(dtype=np.float64), y_train[idx])
    proba = model.predict_proba(x_eval.to_numpy(dtype=np.float64))
    classes = np.asarray(model.classes_, dtype=np.int64)
    best = np.argmax(proba, axis=1)
    return classes[best].astype(np.int64), proba[np.arange(len(x_eval)), best].astype(np.float64)


def _row(prefix: str, m: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(m["pnl"]),
        f"{prefix}_mdd": float(m["mdd"]),
        f"{prefix}_wr": float(m["wr"]),
        f"{prefix}_trades": int(m["trades"]),
        f"{prefix}_long": int(m["long_entries"]),
        f"{prefix}_short": int(m["short_entries"]),
        f"{prefix}_primary_entries": int(m["primary_entries"]),
        f"{prefix}_fallback_entries": int(m["fallback_entries"]),
        f"{prefix}_reasons": m["exit_reasons"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_report = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    selected_cfg = baseline_report["selected_config"]
    cfg = repair.RunnerConfig(
        int(selected_cfg["candidate_id"]),
        str(selected_cfg["mode"]),
        float(selected_cfg["quality_min"]),
        float(selected_cfg["extend_mult"]),
        float(selected_cfg["floor_frac"]),
        int(selected_cfg["max_extensions"]),
    )
    print(json.dumps({"stage": "load", "baseline": BASELINE_ID, "cfg": asdict(cfg)}, ensure_ascii=False), flush=True)
    data = legacy_runner._build()
    x_val = _features(data["validation"])
    x_oos = _features(data["oos"])
    base_val, base_val_ledger = _simulate_combo(data["validation"], cfg, None, None, None, 1.0)
    base_oos, base_oos_ledger = _simulate_combo(data["oos"], cfg, None, None, None, 1.0)
    base_val_ledger.to_csv(OUT_DIR / "validation_baseline_replay_ledger.csv", index=False)
    base_oos_ledger.to_csv(OUT_DIR / "oos_baseline_replay_ledger.csv", index=False)
    rows: list[dict[str, Any]] = [
        {"model": "baseline_tp_runner_clean_repair", "risk": "none", "min_edge": 0.0, "threshold": 1.0, **_row("val", base_val), **_row("oos", base_oos)}
    ]
    diagnostics: dict[str, Any] = {
        "baseline_report": str(BASELINE_REPORT),
        "selected_tp_runner_config": asdict(cfg),
        "feature_count": int(x_val.shape[1]),
        "features": list(x_val.columns),
        "forbidden_feature_exact": sorted(FORBIDDEN_FEATURE_EXACT),
        "forbidden_feature_prefixes": list(FORBIDDEN_FEATURE_PREFIXES),
        "validation_cash_rows": int(np.count_nonzero(~_active(data["validation"]["dec"]))),
        "oos_cash_rows": int(np.count_nonzero(~_active(data["oos"]["dec"]))),
        "baseline_replay": {"validation": base_val, "oos": base_oos},
    }
    best_ledger: pd.DataFrame | None = None
    best_name = "baseline"
    best_oos_pnl = float(base_oos["pnl"])

    for risk in RISKS:
        for min_edge in (0.002, 0.004, 0.006):
            print(json.dumps({"stage": "labels", "risk": risk.name, "min_edge": min_edge}, ensure_ascii=False), flush=True)
            y_val, valid_val, label_diag = _labels(data["validation"], risk, min_edge)
            train_mask = valid_val & (~_active(data["validation"]["dec"]))
            diagnostics[f"{risk.name}_edge{min_edge}"] = label_diag
            if int(train_mask.sum()) < 500 or len(np.unique(y_val[train_mask])) < 2:
                continue
            for model_name in ("hgb", "extra"):
                print(json.dumps({"stage": "fit_eval", "risk": risk.name, "min_edge": min_edge, "model": model_name}, ensure_ascii=False), flush=True)
                val_action, val_conf, oof_diag = _predict_oof(model_name, x_val, y_val, train_mask, seed=260615)
                oos_action, oos_conf = _fit_predict(model_name, x_val, y_val, train_mask, x_oos, seed=260615)
                diagnostics[f"{risk.name}_edge{min_edge}_{model_name}_oof"] = oof_diag
                for threshold in (0.45, 0.55, 0.65, 0.75, 0.85, 0.90, 0.95):
                    val_m, _val_ledger = _simulate_combo(data["validation"], cfg, risk, val_action, val_conf, threshold)
                    oos_m, oos_ledger = _simulate_combo(data["oos"], cfg, risk, oos_action, oos_conf, threshold)
                    row = {"model": model_name, "risk": risk.name, "min_edge": float(min_edge), "threshold": float(threshold)}
                    row.update(_row("val", val_m))
                    row.update(_row("oos", oos_m))
                    rows.append(row)
                    if best_ledger is None or float(oos_m["pnl"]) > best_oos_pnl:
                        best_ledger = oos_ledger
                        best_name = f"{model_name}_{risk.name}_edge{min_edge}_thr{threshold}"
                        best_oos_pnl = float(oos_m["pnl"])

    ranking = pd.DataFrame(rows)
    ranking["val_delta_pnl"] = ranking["val_pnl"] - float(base_val["pnl"])
    ranking["oos_delta_pnl"] = ranking["oos_pnl"] - float(base_oos["pnl"])
    ranking["val_delta_mdd"] = ranking["val_mdd"] - float(base_val["mdd"])
    ranking["oos_delta_mdd"] = ranking["oos_mdd"] - float(base_oos["mdd"])
    ranking["score_val_only"] = ranking["val_pnl"] + 0.40 * ranking["val_mdd"] + 15.0 * ranking["val_wr"]
    ranking = ranking.sort_values(["score_val_only", "val_pnl", "val_mdd"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "validation_only_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos_diagnostic = ranking.sort_values(["oos_pnl", "oos_mdd", "oos_wr"], ascending=False).iloc[0].to_dict()
    if best_ledger is not None:
        best_ledger.to_csv(OUT_DIR / f"oos_best_oos_ledger_{best_name}.csv", index=False)
    feature_names = list(x_val.columns)
    forbidden_features = [c for c in feature_names if c in FORBIDDEN_FEATURE_EXACT or c.startswith(FORBIDDEN_FEATURE_PREFIXES)]
    redteam_blockers: list[str] = []
    if len(x_val) != len(data["validation"]["dec"]):
        redteam_blockers.append("validation feature/decision row count mismatch")
    if len(x_oos) != len(data["oos"]["dec"]):
        redteam_blockers.append("oos feature/decision row count mismatch")
    if feature_names != list(x_oos.columns):
        redteam_blockers.append("validation/oos feature columns mismatch")
    if forbidden_features:
        redteam_blockers.append(f"forbidden feature columns present: {forbidden_features[:20]}")
    if int(diagnostics["validation_cash_rows"]) <= 0:
        redteam_blockers.append("no validation primary-CASH rows available for sleeve training")
    if int(diagnostics["oos_cash_rows"]) <= 0:
        redteam_blockers.append("no oos primary-CASH rows available for sleeve testing")
    if len(ranking) <= 1:
        redteam_blockers.append("no trained sleeve candidates were produced")
    if str(selected["model"]) == "baseline_tp_runner_clean_repair":
        redteam_blockers.append("validation selection returned baseline instead of a trained sleeve candidate")
    redteam_pass = len(redteam_blockers) == 0
    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_shadow_candidate" if redteam_pass else "redteam_fail",
        "baseline_model_id": BASELINE_ID,
        "method": "Alpha-style cash sleeve transfer: preserve Omega1.2.1 TP-runner clean-repair primary; train fallback only on validation rows where primary is CASH; OOS reported after validation-only ranking.",
        "selection_policy": "validation_only_no_oos_selection",
        "redteam_policy": "PnL and OOS lift are diagnostics only. FAIL is limited to logical defects, data/feature contract violations, forbidden feature leakage, missing train/test cash rows, or failed sleeve candidate generation.",
        "accounting_policy": "Cost3 next-open entry and intrabar high/low barrier exits; primary TP-runner config frozen from clean repair.",
        "diagnostics": diagnostics,
        "baseline": {"validation": base_val, "oos": base_oos},
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos_diagnostic,
        "top10": ranking.head(10).to_dict(orient="records"),
        "redteam_pass": redteam_pass,
        "redteam_blockers": redteam_blockers,
        "promotion_blockers": redteam_blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "validation_only_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
            "validation_baseline_replay_ledger": str(OUT_DIR / "validation_baseline_replay_ledger.csv"),
            "oos_baseline_replay_ledger": str(OUT_DIR / "oos_baseline_replay_ledger.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "baseline": report["baseline"], "selected": selected}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
