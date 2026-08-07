from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


MODEL_ID = "certified_teacher_dual_side_execution_router_v3"
CONTRACTS = {
    "MICRO_SCALP": {"max_hold_bars": 6, "stop_loss": 0.0045, "take_profit": 0.0080, "trailing_stop": 0.0040},
    "PULLBACK_REVERT": {"max_hold_bars": 18, "stop_loss": 0.0080, "take_profit": 0.0180, "trailing_stop": 0.0070},
    "BREAKOUT_FOLLOW": {"max_hold_bars": 48, "stop_loss": 0.0140, "take_profit": 0.0420, "trailing_stop": 0.0120},
    "TAIL_REBOUND": {"max_hold_bars": 24, "stop_loss": 0.0100, "take_profit": 0.0280, "trailing_stop": 0.0090},
    "CALM_DRIFT": {"max_hold_bars": 36, "stop_loss": 0.0090, "take_profit": 0.0240, "trailing_stop": 0.0080},
}


@dataclass(frozen=True)
class RouterConfig:
    min_edge_pct: float
    catastrophic_q10_pct: float
    max_notional: float
    min_notional: float
    leverage: float
    cooldown_bars: int
    candidate_stride: int
    target_trades_day: float


@dataclass
class Position:
    side: int
    family: str
    signal_idx: int
    entry_idx: int
    entry_price: float
    notional: float
    leverage: float
    expected_pct: float
    q10_pct: float
    peak_raw: float = 0.0


def runtime_grid() -> list[RouterConfig]:
    out: list[RouterConfig] = []
    for min_edge in (-0.04, 0.00, 0.03):
        for q10 in (-1.20, -0.80, -0.50):
            for max_n in (0.8, 1.4):
                out.append(RouterConfig(min_edge, q10, max_n, 0.10, 5.0, 1, 2, 3.7))
    return out


def matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) for c in cols}, index=frame.index)


def feature_cols(frames: list[pd.DataFrame], clean_prefix: str) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    banned = {"timestamp", "open", "high", "low", "close", "regime_bull", "regime_bear", "regime_chop", "regime_whipsaw", "regime_normal", "regime_trending", "cvp_regime"}
    cols: list[str] = []
    for col in sorted(common):
        lower = col.lower()
        if col in banned or lower.startswith("_") or "future" in lower or "target" in lower or "label" in lower or "realized" in lower or "cash_after" in lower:
            continue
        if ("regime" in lower and not lower.startswith(clean_prefix)) or "hdb" in lower or lower.startswith("hmm_") or "legacy" in lower:
            continue
        if any(pd.to_numeric(frame[col], errors="coerce").notna().any() for frame in frames):
            cols.append(col)
    return cols


def replay(frame: pd.DataFrame, idx: int, side: int, contract: dict[str, float], *, fee: float, slip: float) -> tuple[float, float, str]:
    if idx + 1 >= len(frame):
        return -999.0, 0.0, "no_next_bar"
    entry = float(frame.iloc[idx + 1]["open"])
    if entry <= 0.0:
        return -999.0, 0.0, "bad_entry"
    peak_raw = -999.0
    adverse = 0.0
    exit_idx = min(idx + int(contract["max_hold_bars"]), len(frame) - 1)
    reason = "max_hold"
    for j in range(idx + 1, min(idx + int(contract["max_hold_bars"]) + 1, len(frame))):
        raw = _raw(side, entry, float(frame.iloc[j]["close"]))
        peak_raw = max(peak_raw, raw)
        adverse = max(adverse, max(0.0, -raw))
        if raw <= -float(contract["stop_loss"]):
            exit_idx, reason = min(j + 1, len(frame) - 1), "stop_loss"
            break
        if raw >= float(contract["take_profit"]):
            exit_idx, reason = min(j + 1, len(frame) - 1), "take_profit"
            break
        if peak_raw >= float(contract["trailing_stop"]) * 1.15 and raw <= peak_raw - float(contract["trailing_stop"]):
            exit_idx, reason = min(j + 1, len(frame) - 1), "trailing_stop"
            break
    px = float(frame.iloc[exit_idx]["open"] if exit_idx < len(frame) - 1 else frame.iloc[exit_idx]["close"])
    return float((_raw(side, entry, px) - 2.0 * (fee + slip)) * 100.0), float(adverse * 100.0), reason


def build_side_candidates(frame: pd.DataFrame, cols: list[str], side: int, *, fee: float, slip: float, label: bool, row_stride: int = 1) -> pd.DataFrame:
    idx_values = np.arange(0, len(frame), max(1, int(row_stride)), dtype=np.int32)
    base = matrix(frame, cols).iloc[idx_values].reset_index(drop=True)
    rows = []
    for family, contract in CONTRACTS.items():
        part = base.copy()
        part["_idx"] = idx_values
        part["cand_family"] = family
        for fam in CONTRACTS:
            part[f"cand_family_{fam.lower()}"] = 1.0 if fam == family else 0.0
        part["cand_max_hold_bars"] = float(contract["max_hold_bars"])
        part["cand_stop_loss"] = float(contract["stop_loss"])
        part["cand_take_profit"] = float(contract["take_profit"])
        part["cand_trailing_stop"] = float(contract["trailing_stop"])
        if label:
            vals = [replay(frame, int(i), side, contract, fee=fee, slip=slip) for i in idx_values]
            part["target_net_pct"] = [v[0] for v in vals]
            part["target_adverse_pct"] = [v[1] for v in vals]
            part["target_exit_reason"] = [v[2] for v in vals]
        rows.append(part)
    return pd.concat(rows, axis=0, ignore_index=True)


def model_cols(cands: pd.DataFrame) -> list[str]:
    return [c for c in cands.columns if c not in {"_idx", "cand_family", "target_net_pct", "target_adverse_pct", "target_exit_reason"}]


def fit_side_ranker(cands: pd.DataFrame, cols: list[str], *, seed: int, max_rows: int) -> dict[str, Any]:
    train = cands
    if len(train) > max_rows:
        train = train.sample(max_rows, random_state=seed)
    x = matrix(train, cols)
    y = pd.to_numeric(train["target_net_pct"], errors="coerce").fillna(-999.0).to_numpy(dtype=float)
    params = dict(max_iter=220, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.10, min_samples_leaf=25, early_stopping=False)
    mean = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(loss="squared_error", random_state=seed, **params))
    q10 = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(loss="quantile", quantile=0.10, random_state=seed + 1, **params))
    mean.fit(x, y)
    q10.fit(x, y)
    return {"mean": mean, "q10": q10, "cols": cols}


def predict_side(model: dict[str, Any], cands: pd.DataFrame) -> pd.DataFrame:
    out = cands.copy()
    x = matrix(out, list(model["cols"]))
    out["pred_net_pct"] = np.asarray(model["mean"].predict(x), dtype=float)
    out["pred_q10_pct"] = np.asarray(model["q10"].predict(x), dtype=float)
    out["rank_score"] = out["pred_net_pct"] + 0.25 * out["pred_q10_pct"]
    return out


def backtest(frame: pd.DataFrame, long_pred: pd.DataFrame, short_pred: pd.DataFrame, cfg: RouterConfig, *, fee: float, slip: float) -> dict[str, Any]:
    long_by = {int(k): g for k, g in long_pred.groupby("_idx", sort=False)}
    short_by = {int(k): g for k, g in short_pred.groupby("_idx", sort=False)}
    equity = 1.0
    peak = 1.0
    min_eq = 1.0
    pos: Position | None = None
    last_exit = -100000
    ledger = []
    blocks: dict[str, int] = {}
    for i in range(0, len(frame) - 1):
        next_open = float(frame.iloc[i + 1]["open"])
        if pos is not None:
            contract = CONTRACTS[pos.family]
            raw = _raw(pos.side, pos.entry_price, float(frame.iloc[i]["close"]))
            pos.peak_raw = max(pos.peak_raw, raw)
            mark = equity * max(0.0, 1.0 + pos.notional * raw)
            peak = max(peak, mark)
            min_eq = min(min_eq, mark)
            reason = ""
            if raw <= -float(contract["stop_loss"]):
                reason = "stop_loss"
            elif raw >= float(contract["take_profit"]):
                reason = "take_profit"
            elif pos.peak_raw >= float(contract["trailing_stop"]) * 1.15 and raw <= pos.peak_raw - float(contract["trailing_stop"]):
                reason = "trailing_stop"
            elif i - pos.entry_idx >= int(contract["max_hold_bars"]):
                reason = "max_hold"
            if reason:
                realized = _raw(pos.side, pos.entry_price, next_open)
                cost = pos.notional * (fee + slip)
                gross = pos.notional * realized
                equity *= max(0.0, 1.0 + gross - cost)
                peak = max(peak, equity)
                min_eq = min(min_eq, equity)
                ledger.append(_row(frame, pos, i + 1, next_open, realized, gross, cost, equity, reason, len(ledger)))
                pos = None
                last_exit = i + 1
                continue
        if pos is not None or i <= last_exit + cfg.cooldown_bars or i % max(1, cfg.candidate_stride) != 0:
            continue
        choices = []
        if i in long_by:
            b = long_by[i].sort_values("rank_score", ascending=False).iloc[0]
            choices.append((1, b))
        if i in short_by:
            b = short_by[i].sort_values("rank_score", ascending=False).iloc[0]
            choices.append((-1, b))
        if not choices:
            continue
        side, best = max(choices, key=lambda x: float(x[1]["rank_score"]))
        if float(best["pred_q10_pct"]) < cfg.catastrophic_q10_pct:
            blocks["catastrophic_q10"] = blocks.get("catastrophic_q10", 0) + 1
            continue
        edge = float(best["pred_net_pct"])
        q10 = float(best["pred_q10_pct"])
        transition = float(frame.iloc[i].get("clean_regime_2024_unsup_v4_transition_risk", 0.0) or 0.0)
        edge_scale = np.clip((edge - cfg.min_edge_pct + 0.20) / 0.70, 0.10, 1.25)
        down_scale = np.clip((q10 - cfg.catastrophic_q10_pct) / max(abs(cfg.catastrophic_q10_pct), 0.10), 0.15, 1.0)
        t_scale = np.clip(1.0 - 0.35 * transition, 0.30, 1.0)
        notional = float(np.clip(cfg.min_notional + (cfg.max_notional - cfg.min_notional) * edge_scale * down_scale * t_scale, cfg.min_notional, cfg.max_notional))
        if edge < cfg.min_edge_pct:
            blocks["low_edge_sized_down"] = blocks.get("low_edge_sized_down", 0) + 1
        equity *= max(0.0, 1.0 - notional * (fee + slip))
        min_eq = min(min_eq, equity)
        pos = Position(side, str(best["cand_family"]), i, i + 1, next_open, notional, cfg.leverage, edge, q10)
    if pos is not None:
        i = len(frame) - 1
        px = float(frame.iloc[i]["close"])
        realized = _raw(pos.side, pos.entry_price, px)
        cost = pos.notional * (fee + slip)
        gross = pos.notional * realized
        equity *= max(0.0, 1.0 + gross - cost)
        min_eq = min(min_eq, equity)
        ledger.append(_row(frame, pos, i, px, realized, gross, cost, equity, "end", len(ledger)))
    ledger.append({"trade_id": -1, "timestamp": str(frame.iloc[-1]["timestamp"]), "action": "coverage_end", "side": "COVERAGE", "cash_after": float(equity), "stop_reason": "coverage_end"})
    trades = [r for r in ledger if r.get("action") == "trade"]
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    days = max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0, 1e-12)
    wins = sum(float(r["trade_pnl_pct"]) > 0.0 for r in trades)
    return {
        "pnl": float((equity - 1.0) * 100.0),
        "mdd": float((min_eq / max(peak, 1e-12) - 1.0) * 100.0),
        "trades": int(len(trades)),
        "trades_per_day": float(len(trades) / days),
        "wr": float(wins / len(trades)) if trades else 0.0,
        "avg_notional": float(np.mean([float(r["notional"]) for r in trades])) if trades else 0.0,
        "max_margin_fraction": float(np.max([float(r["margin_fraction"]) for r in trades])) if trades else 0.0,
        "final_equity": float(equity),
        "coverage_start": str(frame.iloc[0]["timestamp"]),
        "coverage_end": str(frame.iloc[-1]["timestamp"]),
        "block_reason_counts": blocks,
        "ledger": ledger,
    }


def score(result: dict[str, Any], cost2: dict[str, Any], target_trades_day: float) -> float:
    tpd = float(result["trades_per_day"])
    if tpd < target_trades_day or int(result["trades"]) < 25:
        return -1e9 + float(result["pnl"]) - 1000.0 * max(0.0, target_trades_day - tpd)
    pnl = float(result["pnl"])
    mdd = abs(float(result["mdd"]))
    cost_drop = max(0.0, pnl - float(cost2["pnl"]))
    return float(pnl - 1.1 * mdd - 0.35 * cost_drop + 0.15 * min(tpd, 8.0))


def _raw(side: int, entry: float, price: float) -> float:
    if entry <= 0.0 or price <= 0.0:
        return 0.0
    return float(side) * (float(price) / float(entry) - 1.0)


def _row(frame: pd.DataFrame, pos: Position, exit_idx: int, exit_price: float, realized: float, gross: float, cost: float, equity: float, reason: str, trade_id: int) -> dict[str, Any]:
    return {
        "trade_id": int(trade_id),
        "timestamp": str(frame.iloc[pos.signal_idx]["timestamp"]),
        "entry_time": str(frame.iloc[pos.entry_idx]["timestamp"]),
        "exit_time": str(frame.iloc[exit_idx]["timestamp"]),
        "entry_idx": int(pos.entry_idx),
        "exit_idx": int(exit_idx),
        "side": "LONG" if pos.side > 0 else "SHORT",
        "contract_family": pos.family,
        "action": "trade",
        "sleeve": MODEL_ID,
        "entry_price": float(pos.entry_price),
        "exit_price": float(exit_price),
        "notional": float(pos.notional),
        "leverage": float(pos.leverage),
        "margin_fraction": float(pos.notional / max(pos.leverage, 1e-12)),
        "expected_net_pct": float(pos.expected_pct),
        "q10_pct": float(pos.q10_pct),
        "realized_raw": float(realized),
        "exit_fee_cash": float(cost),
        "trade_pnl_pct": float((gross - cost) * 100.0),
        "cash_after": float(equity),
        "blocked": False,
        "stop_reason": reason,
    }


def save_bundle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)

