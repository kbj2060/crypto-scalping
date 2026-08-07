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


MODEL_ID = "certified_teacher_utility_ranker_v2"
CONTRACTS = {
    "SCALP": {"max_hold_bars": 8, "stop_loss": 0.006, "take_profit": 0.012, "trailing_stop": 0.006},
    "REBOUND": {"max_hold_bars": 20, "stop_loss": 0.009, "take_profit": 0.022, "trailing_stop": 0.008},
    "TREND": {"max_hold_bars": 48, "stop_loss": 0.014, "take_profit": 0.040, "trailing_stop": 0.012},
}
SIDES = {"LONG": 1, "SHORT": -1}


@dataclass(frozen=True)
class RankerConfig:
    min_expected_pct: float
    q10_floor_pct: float
    max_notional: float
    min_notional: float
    leverage: float
    cooldown_bars: int
    candidate_stride: int
    transition_size_penalty: float


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


def runtime_grid() -> list[RankerConfig]:
    rows: list[RankerConfig] = []
    for min_edge in (0.02, 0.04, 0.07, 0.10):
        for q10 in (-0.55, -0.35, -0.20):
            for max_n in (0.8, 1.4, 2.2):
                rows.append(RankerConfig(min_edge, q10, max_n, 0.15, 5.0, 1, 3, 0.45))
    return rows


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) for c in cols}, index=frame.index)


def candidate_feature_cols(frames: list[pd.DataFrame], clean_prefix: str) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    out: list[str] = []
    banned_exact = {"timestamp", "open", "high", "low", "close", "regime_bull", "regime_bear", "regime_chop", "regime_whipsaw", "regime_normal", "regime_trending", "cvp_regime"}
    for col in sorted(common):
        lower = col.lower()
        if col in banned_exact or lower.startswith("_") or "future" in lower or "target" in lower or "label" in lower or "realized" in lower or "cash_after" in lower:
            continue
        if ("regime" in lower and not lower.startswith(clean_prefix)) or "hdb" in lower or lower.startswith("hmm_") or "legacy" in lower:
            continue
        if any(pd.to_numeric(frame[col], errors="coerce").notna().any() for frame in frames):
            out.append(col)
    return out


def replay_contract(frame: pd.DataFrame, idx: int, side: int, contract: dict[str, float], *, fee: float, slip: float) -> tuple[float, float, int, str]:
    n = len(frame)
    if idx + 1 >= n:
        return -999.0, 0.0, 0, "no_next_bar"
    entry = float(frame.iloc[idx + 1]["open"])
    if entry <= 0.0:
        return -999.0, 0.0, 0, "bad_entry"
    max_hold = int(contract["max_hold_bars"])
    stop_loss = float(contract["stop_loss"])
    take_profit = float(contract["take_profit"])
    trailing_stop = float(contract["trailing_stop"])
    peak_raw = -999.0
    adverse = 0.0
    exit_idx = min(idx + max_hold, n - 1)
    reason = "max_hold"
    for j in range(idx + 1, min(idx + max_hold + 1, n)):
        close = float(frame.iloc[j]["close"])
        raw = _raw_ret(side, entry, close)
        peak_raw = max(peak_raw, raw)
        adverse = max(adverse, max(0.0, -raw))
        if raw <= -stop_loss:
            exit_idx, reason = min(j + 1, n - 1), "stop_loss"
            break
        if raw >= take_profit:
            exit_idx, reason = min(j + 1, n - 1), "take_profit"
            break
        if peak_raw >= trailing_stop * 1.15 and raw <= peak_raw - trailing_stop:
            exit_idx, reason = min(j + 1, n - 1), "trailing_stop"
            break
    exit_price = float(frame.iloc[exit_idx]["open"] if exit_idx < n - 1 else frame.iloc[exit_idx]["close"])
    realized = _raw_ret(side, entry, exit_price)
    net_pct = (realized - 2.0 * (fee + slip)) * 100.0
    return float(net_pct), float(adverse * 100.0), int(exit_idx - (idx + 1)), reason


def build_candidate_table(
    frame: pd.DataFrame,
    base_cols: list[str],
    *,
    fee: float,
    slip: float,
    label: bool,
    row_stride: int = 1,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base = matrix(frame, base_cols)
    idx_values = np.arange(0, len(frame), max(1, int(row_stride)), dtype=np.int32)
    base = base.iloc[idx_values].reset_index(drop=True)
    for family, contract in CONTRACTS.items():
        for side_name, side in SIDES.items():
            part = base.copy()
            part["_idx"] = idx_values
            part["cand_side"] = float(side)
            for name in SIDES:
                part[f"cand_side_{name.lower()}"] = 1.0 if name == side_name else 0.0
            for fam in CONTRACTS:
                part[f"cand_family_{fam.lower()}"] = 1.0 if fam == family else 0.0
            part["cand_max_hold_bars"] = float(contract["max_hold_bars"])
            part["cand_stop_loss"] = float(contract["stop_loss"])
            part["cand_take_profit"] = float(contract["take_profit"])
            part["cand_trailing_stop"] = float(contract["trailing_stop"])
            part["cand_family"] = family
            part["cand_side_name"] = side_name
            if label:
                vals = [replay_contract(frame, int(i), side, contract, fee=fee, slip=slip) for i in idx_values]
                part["target_net_pct"] = [v[0] for v in vals]
                part["target_adverse_pct"] = [v[1] for v in vals]
                part["target_hold_bars"] = [v[2] for v in vals]
                part["target_exit_reason"] = [v[3] for v in vals]
            rows.append(part)
    out = pd.concat(rows, axis=0, ignore_index=True)
    if "target_net_pct" not in out.columns:
        return out.reset_index(drop=True)
    valid = np.isfinite(pd.to_numeric(out["target_net_pct"], errors="coerce").fillna(-999.0))
    return out[valid].reset_index(drop=True)


def candidate_model_cols(candidate: pd.DataFrame) -> list[str]:
    return [c for c in candidate.columns if c not in {"_idx", "cand_family", "cand_side_name", "target_net_pct", "target_adverse_pct", "target_hold_bars", "target_exit_reason"}]


def fit_ranker(candidates: pd.DataFrame, cols: list[str], *, max_train_rows: int = 300000) -> dict[str, Any]:
    train = candidates
    if len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=913)
    x = matrix(train, cols)
    y = pd.to_numeric(train["target_net_pct"], errors="coerce").fillna(-999.0).to_numpy(dtype=float)
    params = dict(max_iter=240, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.10, min_samples_leaf=30, early_stopping=False, random_state=913)
    mean = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(loss="squared_error", **params))
    mean.fit(x, y)
    q10 = _fit_quantile(x, y, 0.10, {**params, "random_state": 914})
    q50 = _fit_quantile(x, y, 0.50, {**params, "random_state": 915})
    q90 = _fit_quantile(x, y, 0.90, {**params, "random_state": 916})
    return {"mean": mean, "q10": q10, "q50": q50, "q90": q90, "model_cols": cols}


def _fit_quantile(x: pd.DataFrame, y: np.ndarray, q: float, params: dict[str, Any]) -> Any:
    try:
        model = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(loss="quantile", quantile=q, **params))
        model.fit(x, y)
        return model
    except TypeError:
        model = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(loss="squared_error", **params))
        model.fit(x, y)
        return model


def predict_candidates(bundle: dict[str, Any], candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    x = matrix(out, list(bundle["model_cols"]))
    out["pred_net_pct"] = np.asarray(bundle["mean"].predict(x), dtype=float)
    out["pred_q10_pct"] = np.asarray(bundle["q10"].predict(x), dtype=float)
    out["pred_q50_pct"] = np.asarray(bundle["q50"].predict(x), dtype=float)
    out["pred_q90_pct"] = np.asarray(bundle["q90"].predict(x), dtype=float)
    out["rank_score"] = out["pred_net_pct"] + 0.35 * out["pred_q10_pct"] + 0.10 * (out["pred_q90_pct"] - out["pred_q10_pct"])
    return out


def backtest_ranker(frame: pd.DataFrame, predicted: pd.DataFrame, cfg: RankerConfig, *, fee: float, slip: float) -> dict[str, Any]:
    by_idx = {int(k): g for k, g in predicted.groupby("_idx", sort=False)}
    equity = 1.0
    peak = 1.0
    min_equity = 1.0
    pos: Position | None = None
    last_exit = -100000
    ledger: list[dict[str, Any]] = []
    block_counts: dict[str, int] = {}
    for i in range(0, len(frame) - 1):
        close = float(frame.iloc[i]["close"])
        next_open = float(frame.iloc[i + 1]["open"])
        if pos is not None:
            contract = CONTRACTS[pos.family]
            raw = _raw_ret(pos.side, pos.entry_price, close)
            pos.peak_raw = max(pos.peak_raw, raw)
            mark = equity * max(0.0, 1.0 + pos.notional * raw)
            peak = max(peak, mark)
            min_equity = min(min_equity, mark)
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
                realized = _raw_ret(pos.side, pos.entry_price, next_open)
                cost = pos.notional * (fee + slip)
                gross = pos.notional * realized
                equity *= max(0.0, 1.0 + gross - cost)
                peak = max(peak, equity)
                min_equity = min(min_equity, equity)
                ledger.append(_ledger_row(frame, pos, i + 1, next_open, realized, gross, cost, equity, reason, len(ledger)))
                pos = None
                last_exit = i + 1
                continue
        if pos is not None or i <= last_exit + cfg.cooldown_bars or i % max(1, cfg.candidate_stride) != 0:
            continue
        cands = by_idx.get(i)
        if cands is None or cands.empty:
            continue
        best = cands.sort_values("rank_score", ascending=False).iloc[0]
        if float(best["pred_net_pct"]) < cfg.min_expected_pct:
            block_counts["expected_edge_floor"] = block_counts.get("expected_edge_floor", 0) + 1
            continue
        if float(best["pred_q10_pct"]) < cfg.q10_floor_pct:
            block_counts["q10_floor"] = block_counts.get("q10_floor", 0) + 1
            continue
        transition = float(frame.iloc[i].get("clean_regime_2024_unsup_v4_transition_risk", 0.0) or 0.0)
        edge_scale = np.clip(float(best["pred_net_pct"]) / 0.50, 0.0, 1.4)
        q_scale = np.clip((float(best["pred_q10_pct"]) - cfg.q10_floor_pct) / max(abs(cfg.q10_floor_pct), 0.10), 0.15, 1.0)
        t_scale = np.clip(1.0 - cfg.transition_size_penalty * transition, 0.25, 1.0)
        notional = float(np.clip(cfg.min_notional + (cfg.max_notional - cfg.min_notional) * edge_scale * q_scale * t_scale, cfg.min_notional, cfg.max_notional))
        entry_cost = notional * (fee + slip)
        equity *= max(0.0, 1.0 - entry_cost)
        min_equity = min(min_equity, equity)
        pos = Position(int(best["cand_side"]), str(best["cand_family"]), i, i + 1, next_open, notional, cfg.leverage, float(best["pred_net_pct"]), float(best["pred_q10_pct"]))
    if pos is not None:
        i = len(frame) - 1
        exit_price = float(frame.iloc[i]["close"])
        realized = _raw_ret(pos.side, pos.entry_price, exit_price)
        cost = pos.notional * (fee + slip)
        gross = pos.notional * realized
        equity *= max(0.0, 1.0 + gross - cost)
        min_equity = min(min_equity, equity)
        ledger.append(_ledger_row(frame, pos, i, exit_price, realized, gross, cost, equity, "end", len(ledger)))
    ledger.append({"trade_id": -1, "timestamp": str(frame.iloc[-1]["timestamp"]), "action": "coverage_end", "side": "COVERAGE", "cash_after": float(equity), "stop_reason": "coverage_end"})
    trades = [r for r in ledger if r.get("action") == "trade"]
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    days = max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0, 1e-12)
    wins = sum(float(r["trade_pnl_pct"]) > 0.0 for r in trades)
    return {
        "pnl": float((equity - 1.0) * 100.0),
        "mdd": float((min_equity / max(peak, 1e-12) - 1.0) * 100.0),
        "trades": int(len(trades)),
        "trades_per_day": float(len(trades) / days),
        "wr": float(wins / len(trades)) if trades else 0.0,
        "avg_notional": float(np.mean([float(r["notional"]) for r in trades])) if trades else 0.0,
        "max_margin_fraction": float(np.max([float(r["margin_fraction"]) for r in trades])) if trades else 0.0,
        "final_equity": float(equity),
        "coverage_start": str(frame.iloc[0]["timestamp"]),
        "coverage_end": str(frame.iloc[-1]["timestamp"]),
        "block_reason_counts": block_counts,
        "ledger": ledger,
    }


def score(result: dict[str, Any], cost2: dict[str, Any]) -> float:
    pnl = float(result["pnl"])
    mdd = abs(float(result["mdd"]))
    trades_day = float(result["trades_per_day"])
    trades = int(result["trades"])
    if trades < 25 or trades_day < 1.0:
        return -1e9 + pnl - 1000.0 * max(0.0, 1.0 - trades_day)
    cost_drop = max(0.0, pnl - float(cost2["pnl"]))
    return float(pnl - 1.2 * mdd - 0.5 * cost_drop + 0.15 * min(trades_day, 8.0))


def _raw_ret(side: int, entry: float, price: float) -> float:
    if entry <= 0.0 or price <= 0.0:
        return 0.0
    return float(side) * (float(price) / float(entry) - 1.0)


def _ledger_row(frame: pd.DataFrame, pos: Position, exit_idx: int, exit_price: float, realized: float, gross: float, exit_cost: float, equity: float, reason: str, trade_id: int) -> dict[str, Any]:
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
        "exit_fee_cash": float(exit_cost),
        "trade_pnl_pct": float((gross - exit_cost) * 100.0),
        "cash_after": float(equity),
        "blocked": False,
        "stop_reason": reason,
    }


def save_bundle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
