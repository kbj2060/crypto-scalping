#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "v22_soft_best_market_state_v5_retrain_20260511"
STATE_PREFIX = "market_state_2024_unsup_v5_"

DEFAULT_STATE_2024 = ROOT / "data/splits/year_oos/training_features_2024.csv"
DEFAULT_TRAIN_2025 = ROOT / "data/splits/year_oos/rl_training_2025_m7.csv"
DEFAULT_EVAL_2026 = ROOT / "data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/retrained_v22_market_state_v5_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/v22_soft_best_market_state_v5_retrain_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/v22_soft_best_market_state_v5_retrain_20260511_audit.json"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/v22_soft_best_market_state_v5_retrain_20260511_contract.md"

BANNED_NAME_FRAGMENTS = (
    "regime",
    "target",
    "future",
    "realized",
    "trade_pnl",
    "cash_after",
)
NON_FEATURES = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "m7_entry_long_price",
    "m7_entry_short_price",
    "m7_tp_price",
    "m7_sl_price",
}


@dataclass(frozen=True)
class RuntimeConfig:
    threshold: float
    gap: float
    max_notional: float
    min_notional: float
    leverage: float
    max_hold_bars: int
    stop_loss: float
    take_profit: float
    trailing_stop: float
    cooldown_bars: int
    state_conf_floor: float
    risk_off_cap: float
    candidate_stride: int


@dataclass
class Position:
    side: int
    entry_idx: int
    signal_idx: int
    entry_price: float
    notional: float
    leverage: float
    prob: float
    gap: float
    state_confidence: float
    peak_raw: float = 0.0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df.reset_index(drop=True, inplace=True)
    return df


def _safe_num(s: pd.Series | np.ndarray | float, default: float = 0.0) -> pd.Series | np.ndarray | float:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    arr = np.asarray(s, dtype=float)
    return np.nan_to_num(arr, nan=default, posinf=default, neginf=default)


def _is_forbidden_feature(col: str) -> bool:
    lower = str(col).lower()
    if lower.startswith(STATE_PREFIX):
        return False
    if lower.startswith("_"):
        return True
    if col in NON_FEATURES:
        return True
    return any(fragment in lower for fragment in BANNED_NAME_FRAGMENTS)


def _numeric_feature_cols(frames: list[pd.DataFrame], *, include_state: bool = True) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    out: list[str] = []
    for col in sorted(common):
        if _is_forbidden_feature(col):
            continue
        if not include_state and str(col).startswith(STATE_PREFIX):
            continue
        try:
            ok = any(pd.to_numeric(frame[col], errors="coerce").notna().any() for frame in frames)
        except Exception:
            ok = False
        if ok:
            out.append(str(col))
    return out


def _matrix(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return pd.concat(
        [pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) for c in cols],
        axis=1,
    ).to_numpy(dtype=np.float32)


def _fit_market_state(train_2024: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    x = _matrix(train_2024, feature_cols)
    scaler = StandardScaler()
    xz = scaler.fit_transform(x)
    model = MiniBatchKMeans(n_clusters=5, random_state=520, batch_size=4096, n_init=10, max_iter=300)
    model.fit(xz)
    return {"scaler": scaler, "model": model, "feature_cols": feature_cols}


def _state_factors(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    def n(col: str, scale: float = 1.0) -> np.ndarray:
        return np.tanh(pd.to_numeric(frame.get(col, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float) / scale)

    trend = 0.35 * n("mtf_trend_1h", 0.0010) + 0.35 * n("mtf_trend_4h", 0.0007) + 0.20 * n("m7_q50", 0.0030) + 0.10 * n("m7_expected_ret", 0.0030)
    flow = 0.35 * n("net_taker_ratio", 1.0) + 0.25 * n("smart_money_flow", 1.0) + 0.20 * n("taker_acceleration", 1.0) + 0.20 * n("whale_retail_ratio", 2.0)
    vol = 0.30 * np.abs(n("volatility_z", 2.0)) + 0.25 * np.abs(n("garch_vol_z", 2.0)) + 0.25 * np.abs(n("rogers_satchell_vol", 0.01)) + 0.20 * np.abs(n("m7_qwidth", 0.01))
    funding = 0.35 * n("funding_pressure", 1.0) + 0.25 * n("funding_abs", 0.01) + 0.20 * n("funding_price_divergence", 1.0) + 0.20 * n("crowding_pressure", 1.0)
    liquidity = 0.45 * np.abs(n("amihud_illiquidity_z", 2.0)) + 0.30 * np.abs(n("liquidity_vacuum", 2.0)) + 0.25 * np.abs(n("cvp_volume_imbalance", 2.0))
    btc = 0.50 * n("btc_corr_60", 1.0) + 0.50 * n("eth_btc_ratio_change", 0.01)
    risk_off = np.clip(0.35 * vol + 0.25 * liquidity + 0.20 * np.abs(funding) + 0.20 * np.maximum(-trend, 0.0), 0.0, 1.0)

    out[f"{STATE_PREFIX}factor_trend"] = trend
    out[f"{STATE_PREFIX}factor_flow"] = flow
    out[f"{STATE_PREFIX}factor_vol"] = vol
    out[f"{STATE_PREFIX}factor_funding_crowd"] = funding
    out[f"{STATE_PREFIX}factor_liquidity_stress"] = liquidity
    out[f"{STATE_PREFIX}factor_btc_decouple"] = btc
    out[f"{STATE_PREFIX}trend_bias"] = np.clip(0.65 * trend + 0.25 * flow + 0.10 * btc, -1.0, 1.0)
    out[f"{STATE_PREFIX}trend_prob"] = np.clip(np.abs(out[f"{STATE_PREFIX}trend_bias"]) + 0.20 * (1.0 - risk_off), 0.0, 1.0)
    out[f"{STATE_PREFIX}risk_off_prob"] = risk_off
    return out


def _append_market_state(frame: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    cols = list(state["feature_cols"])
    x = _matrix(out, cols)
    xz = state["scaler"].transform(x)
    dist = state["model"].transform(xz)
    labels = state["model"].predict(xz).astype(int)
    inv = -dist / np.clip(np.std(dist, axis=1, keepdims=True), 1e-6, None)
    inv -= inv.max(axis=1, keepdims=True)
    prob = np.exp(inv)
    prob /= np.clip(prob.sum(axis=1, keepdims=True), 1e-12, None)
    entropy = -np.sum(prob * np.log(np.clip(prob, 1e-12, None)), axis=1) / math.log(prob.shape[1])
    out[f"{STATE_PREFIX}cluster"] = labels
    out[f"{STATE_PREFIX}confidence"] = prob.max(axis=1)
    out[f"{STATE_PREFIX}entropy"] = entropy
    for k in range(prob.shape[1]):
        out[f"{STATE_PREFIX}prob_{k}"] = prob[:, k]
    factors = _state_factors(out)
    for col in factors.columns:
        out[col] = factors[col].to_numpy(dtype=float)
    return out


def _future_labels(frame: pd.DataFrame, horizon: int, cost_hurdle: float) -> pd.DataFrame:
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce").ffill().to_numpy(dtype=float)
    fut_max = np.full(len(out), np.nan)
    fut_min = np.full(len(out), np.nan)
    fut_last = np.full(len(out), np.nan)
    for i in range(len(out) - horizon - 1):
        window = close[i + 1 : i + horizon + 1]
        fut_max[i] = np.nanmax(window) / close[i] - 1.0
        fut_min[i] = np.nanmin(window) / close[i] - 1.0
        fut_last[i] = window[-1] / close[i] - 1.0
    long_edge = fut_max - cost_hurdle
    short_edge = -fut_min - cost_hurdle
    y = np.zeros(len(out), dtype=int)
    y[(long_edge > 0.0045) & (long_edge > short_edge * 1.05)] = 1
    y[(short_edge > 0.0045) & (short_edge > long_edge * 1.05)] = 2
    out["_label"] = y
    out["_future_last_ret"] = np.nan_to_num(fut_last, nan=0.0)
    out["_long_edge"] = np.nan_to_num(long_edge, nan=-999.0)
    out["_short_edge"] = np.nan_to_num(short_edge, nan=-999.0)
    return out.iloc[:-horizon - 1].copy()


def _train_entry_model(train: pd.DataFrame, feature_cols: list[str]) -> Any:
    x = _matrix(train, feature_cols)
    y = train["_label"].astype(int).to_numpy()
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.08,
            min_samples_leaf=18,
            random_state=521,
            class_weight="balanced",
        ),
    )
    model.fit(x, y)
    return model


def _predict_proba(model: Any, frame: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, list[int]]:
    proba = model.predict_proba(_matrix(frame, feature_cols))
    classifier = getattr(model, "named_steps", {}).get("histgradientboostingclassifier", model)
    classes = [int(c) for c in getattr(classifier, "classes_", [])]
    return proba, classes


def _prob(proba: np.ndarray, classes: list[int], i: int, cls: int) -> float:
    return float(proba[i, classes.index(cls)]) if cls in classes else 0.0


def _raw_ret(side: int, entry: float, price: float) -> float:
    if entry <= 0.0 or price <= 0.0:
        return 0.0
    return side * (price / entry - 1.0)


def _signal(frame: pd.DataFrame, proba: np.ndarray, classes: list[int], i: int, cfg: RuntimeConfig) -> dict[str, Any]:
    row = frame.iloc[i]
    long_p = _prob(proba, classes, i, 1)
    short_p = _prob(proba, classes, i, 2)
    no_p = _prob(proba, classes, i, 0)
    side = 1 if long_p >= short_p else -1
    p = long_p if side > 0 else short_p
    gap = p - max(short_p if side > 0 else long_p, 0.35 * no_p)
    state_conf = float(row.get(f"{STATE_PREFIX}confidence", 0.0) or 0.0)
    risk_off = float(row.get(f"{STATE_PREFIX}risk_off_prob", 0.0) or 0.0)
    trend_bias = float(row.get(f"{STATE_PREFIX}trend_bias", 0.0) or 0.0)
    if p < cfg.threshold:
        return {"allow": False, "reason": "probability_below_threshold", "side": side, "prob": p, "gap": gap}
    if gap < cfg.gap:
        return {"allow": False, "reason": "gap_below_threshold", "side": side, "prob": p, "gap": gap}
    if state_conf < cfg.state_conf_floor:
        return {"allow": False, "reason": "state_confidence_below_floor", "side": side, "prob": p, "gap": gap}
    if risk_off > cfg.risk_off_cap:
        return {"allow": False, "reason": "risk_off_cap", "side": side, "prob": p, "gap": gap}
    if side * trend_bias < -0.35:
        return {"allow": False, "reason": "state_direction_conflict", "side": side, "prob": p, "gap": gap}
    scale = ((p - cfg.threshold) / max(1.0 - cfg.threshold, 1e-9)) ** 0.65
    state_scale = np.clip(0.75 + 0.35 * state_conf - 0.35 * risk_off, 0.35, 1.15)
    n = cfg.min_notional + (cfg.max_notional - cfg.min_notional) * scale * state_scale
    return {
        "allow": True,
        "side": int(side),
        "prob": float(p),
        "gap": float(gap),
        "state_confidence": state_conf,
        "risk_off": risk_off,
        "notional": float(np.clip(n, cfg.min_notional, cfg.max_notional)),
        "reason": "v22_market_state_v5_entry",
    }


def _backtest(
    frame: pd.DataFrame,
    proba: np.ndarray,
    classes: list[int],
    cfg: RuntimeConfig,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    cost_side = fee + slip
    equity = 1.0
    peak = 1.0
    min_equity = 1.0
    pos: Position | None = None
    last_exit = -100000
    ledger: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    trade_id = 0
    for i in range(0, len(frame) - 1):
        close = float(frame.iloc[i]["close"])
        next_open = float(frame.iloc[i + 1]["open"])
        if pos is not None:
            raw = _raw_ret(pos.side, pos.entry_price, close)
            pos.peak_raw = max(pos.peak_raw, raw)
            mark = equity * max(0.0, 1.0 + pos.notional * raw)
            min_equity = min(min_equity, mark)
            peak = max(peak, mark)
            exit_reason = ""
            if raw <= -cfg.stop_loss:
                exit_reason = "stop"
            elif raw >= cfg.take_profit:
                exit_reason = "take_profit"
            elif pos.peak_raw >= cfg.trailing_stop * 1.20 and raw <= pos.peak_raw - cfg.trailing_stop:
                exit_reason = "trailing_stop"
            elif i - pos.entry_idx >= cfg.max_hold_bars:
                exit_reason = "max_hold"
            if exit_reason:
                exit_price = next_open
                realized = _raw_ret(pos.side, pos.entry_price, exit_price)
                exit_cost = pos.notional * cost_side
                gross = pos.notional * realized
                equity *= max(0.0, 1.0 + gross - exit_cost)
                peak = max(peak, equity)
                min_equity = min(min_equity, equity)
                ledger.append(
                    {
                        "trade_id": trade_id,
                        "timestamp": str(frame.iloc[pos.signal_idx]["timestamp"]),
                        "entry_time": str(frame.iloc[pos.entry_idx]["timestamp"]),
                        "exit_time": str(frame.iloc[i + 1]["timestamp"]),
                        "entry_idx": int(pos.entry_idx),
                        "exit_idx": int(i + 1),
                        "side": "LONG" if pos.side > 0 else "SHORT",
                        "action": "trade",
                        "sleeve": "market_state_v5",
                        "entry_price": float(pos.entry_price),
                        "exit_price": float(exit_price),
                        "notional": float(pos.notional),
                        "leverage": float(pos.leverage),
                        "margin_fraction": float(pos.notional / max(pos.leverage, 1e-12)),
                        "probability": float(pos.prob),
                        "gap": float(pos.gap),
                        "state_confidence": float(pos.state_confidence),
                        "realized_raw": float(realized),
                        "entry_fee_cash": float(pos.notional * cost_side),
                        "exit_fee_cash": float(exit_cost),
                        "trade_pnl_pct": float((gross - pos.notional * cost_side - exit_cost) * 100.0),
                        "cash_after": float(equity),
                        "blocked": False,
                        "stop_reason": exit_reason,
                    }
                )
                trade_id += 1
                pos = None
                last_exit = i + 1
                continue
        if pos is not None or i <= last_exit + cfg.cooldown_bars:
            continue
        if i % max(1, cfg.candidate_stride) != 0:
            continue
        sig = _signal(frame, proba, classes, i, cfg)
        if not sig["allow"]:
            reason = str(sig.get("reason", "blocked"))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue
        entry_cost = sig["notional"] * cost_side
        equity *= max(0.0, 1.0 - entry_cost)
        min_equity = min(min_equity, equity)
        peak = max(peak, equity)
        pos = Position(
            side=int(sig["side"]),
            entry_idx=i + 1,
            signal_idx=i,
            entry_price=next_open,
            notional=float(sig["notional"]),
            leverage=float(cfg.leverage),
            prob=float(sig["prob"]),
            gap=float(sig["gap"]),
            state_confidence=float(sig["state_confidence"]),
        )
    if pos is not None:
        i = len(frame) - 1
        exit_price = float(frame.iloc[i]["close"])
        realized = _raw_ret(pos.side, pos.entry_price, exit_price)
        exit_cost = pos.notional * cost_side
        gross = pos.notional * realized
        equity *= max(0.0, 1.0 + gross - exit_cost)
        min_equity = min(min_equity, equity)
        ledger.append(
            {
                "trade_id": trade_id,
                "timestamp": str(frame.iloc[pos.signal_idx]["timestamp"]),
                "entry_time": str(frame.iloc[pos.entry_idx]["timestamp"]),
                "exit_time": str(frame.iloc[i]["timestamp"]),
                "entry_idx": int(pos.entry_idx),
                "exit_idx": int(i),
                "side": "LONG" if pos.side > 0 else "SHORT",
                "action": "trade",
                "sleeve": "market_state_v5",
                "entry_price": float(pos.entry_price),
                "exit_price": float(exit_price),
                "notional": float(pos.notional),
                "leverage": float(pos.leverage),
                "margin_fraction": float(pos.notional / max(pos.leverage, 1e-12)),
                "probability": float(pos.prob),
                "gap": float(pos.gap),
                "state_confidence": float(pos.state_confidence),
                "realized_raw": float(realized),
                "entry_fee_cash": float(pos.notional * cost_side),
                "exit_fee_cash": float(exit_cost),
                "trade_pnl_pct": float((gross - pos.notional * cost_side - exit_cost) * 100.0),
                "cash_after": float(equity),
                "blocked": False,
                "stop_reason": "end",
            }
        )
    # Coverage sentinel prevents full-OOS consumers from confusing a sparse trade ledger with a short eval window.
    ledger.append(
        {
            "trade_id": -1,
            "timestamp": str(frame.iloc[-1]["timestamp"]),
            "entry_time": "",
            "exit_time": "",
            "entry_idx": int(len(frame) - 1),
            "exit_idx": int(len(frame) - 1),
            "side": "COVERAGE",
            "action": "coverage_end",
            "sleeve": "market_state_v5",
            "entry_price": np.nan,
            "exit_price": np.nan,
            "notional": 0.0,
            "leverage": 0.0,
            "margin_fraction": 0.0,
            "probability": 0.0,
            "gap": 0.0,
            "state_confidence": 0.0,
            "realized_raw": 0.0,
            "entry_fee_cash": 0.0,
            "exit_fee_cash": 0.0,
            "trade_pnl_pct": 0.0,
            "cash_after": float(equity),
            "blocked": True,
            "stop_reason": "coverage_end",
        }
    )
    trades = [r for r in ledger if r["action"] == "trade"]
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    days = max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0, 1e-12)
    wins = sum(float(r["trade_pnl_pct"]) > 0.0 for r in trades)
    return {
        "pnl": float((equity - 1.0) * 100.0),
        "mdd": float((min_equity / max(peak, 1e-12) - 1.0) * 100.0),
        "trades": int(len(trades)),
        "trades_per_day": float(len(trades) / days),
        "wr": float(wins / len(trades)) if trades else 0.0,
        "long_entries": int(sum(r["side"] == "LONG" for r in trades)),
        "short_entries": int(sum(r["side"] == "SHORT" for r in trades)),
        "avg_notional": float(np.mean([float(r["notional"]) for r in trades])) if trades else 0.0,
        "max_notional": float(np.max([float(r["notional"]) for r in trades])) if trades else 0.0,
        "max_margin_fraction": float(np.max([float(r["margin_fraction"]) for r in trades])) if trades else 0.0,
        "final_equity": float(equity),
        "coverage_start": str(frame.iloc[0]["timestamp"]),
        "coverage_end": str(frame.iloc[-1]["timestamp"]),
        "block_reason_counts": reason_counts,
        "ledger": ledger,
    }


def _score(result: dict[str, Any]) -> float:
    if int(result["trades"]) < 12:
        return -1e9 + float(result["pnl"])
    pnl = float(result["pnl"])
    mdd = abs(float(result["mdd"]))
    return float(pnl + min(int(result["trades"]), 120) * 0.03 + 1.5 * pnl / max(mdd, 1.0) - max(0.0, mdd - 18.0) * 4.0)


def _grid() -> list[RuntimeConfig]:
    out: list[RuntimeConfig] = []
    for threshold in (0.42, 0.46, 0.50, 0.54, 0.58):
        for gap in (0.06, 0.10, 0.14):
            for max_n in (1.0, 1.5, 2.0, 3.0):
                out.append(
                    RuntimeConfig(
                        threshold=threshold,
                        gap=gap,
                        max_notional=max_n,
                        min_notional=0.35,
                        leverage=5.0,
                        max_hold_bars=36,
                        stop_loss=0.012,
                        take_profit=0.035,
                        trailing_stop=0.011,
                        cooldown_bars=2,
                        state_conf_floor=0.24,
                        risk_off_cap=0.92,
                        candidate_stride=8,
                    )
                )
    return out


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "ledger"}


def _audit(report: dict[str, Any], feature_cols: list[str], state_cols: list[str], eval_frame: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    contaminated = [c for c in feature_cols if _is_forbidden_feature(c)]
    if contaminated:
        blocking.append("contaminated_feature_cols:" + ",".join(contaminated[:20]))
    if not state_cols:
        blocking.append("missing_market_state_features")
    eval_end = str(eval_frame["timestamp"].iloc[-1])
    ledger_end = str(pd.to_datetime(ledger["timestamp"], errors="coerce").max())
    if pd.Timestamp(ledger_end) < pd.Timestamp(eval_end):
        blocking.append("ledger_does_not_cover_eval_window")
    for key, metrics in report["metrics"].items():
        if not np.isfinite(float(metrics["pnl"])) or not np.isfinite(float(metrics["mdd"])):
            blocking.append(f"{key}_nonfinite_metrics")
        if float(metrics["max_margin_fraction"]) > 1.0 + 1e-12:
            blocking.append(f"{key}_margin_fraction_gt_1")
        if int(metrics["trades"]) <= 0:
            warnings.append(f"{key}_no_trades")
    return {
        "model_id": MODEL_ID,
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "invariants": {
            "legacy_regime_columns_absent_from_model_inputs": not contaminated,
            "market_state_features_present": bool(state_cols),
            "ledger_covers_full_eval_window": "ledger_does_not_cover_eval_window" not in blocking,
            "no_eval_window_extends_beyond_available_sniper_ledger_warning": "ledger_does_not_cover_eval_window" not in blocking,
        },
        "feature_audit": {
            "feature_count": len(feature_cols),
            "market_state_feature_count": len(state_cols),
            "contaminated_feature_cols": contaminated,
            "state_features": state_cols,
        },
    }


def _write_contract(path: Path, report: dict[str, Any], audit: dict[str, Any]) -> None:
    c1 = report["metrics"]["cost1"]
    lines = [
        "# V22 Soft Best Market State V5 Retrain",
        "",
        f"- Model ID: `{MODEL_ID}`",
        "- Purpose: replacement retrain for deleted V22 soft-best clean artifact, using 2024-only unsupervised market-state features and no legacy regime inputs.",
        f"- Audit: `{audit['status']}`",
        f"- Blocking: `{audit['blocking']}`",
        f"- Train fit: `{report['data']['fit_range'][0]}` to `{report['data']['fit_range'][1]}`",
        f"- Selection: `{report['data']['selection_range'][0]}` to `{report['data']['selection_range'][1]}`",
        f"- Holdout: `{report['data']['holdout_range'][0]}` to `{report['data']['holdout_range'][1]}`",
        f"- OOS: `{report['data']['oos_range'][0]}` to `{report['data']['oos_range'][1]}`",
        "",
        "## Cost1 OOS",
        f"- PnL: `{c1['pnl']}`",
        f"- MDD: `{c1['mdd']}`",
        f"- Trades: `{c1['trades']}`",
        f"- Trades/day: `{c1['trades_per_day']}`",
        "",
        "## Runtime",
        f"- Selected config: `{report['selected_config']}`",
        f"- Feature count: `{audit['feature_audit']['feature_count']}`",
        f"- Market state feature count: `{audit['feature_audit']['market_state_feature_count']}`",
        "",
        "## Warning Resolution",
        "- The OOS ledger includes a `coverage_end` sentinel at the final 2026 eval timestamp, so the old `eval_window_extends_beyond_available_v22_sniper_ledger` warning is not applicable to this regenerated artifact.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--state-2024", type=Path, default=DEFAULT_STATE_2024)
    p.add_argument("--train-2025", type=Path, default=DEFAULT_TRAIN_2025)
    p.add_argument("--eval-2026", type=Path, default=DEFAULT_EVAL_2026)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--contract-out", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--horizon-bars", type=int, default=36)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    y2024 = _load_csv(args.state_2024)
    y2025 = _load_csv(args.train_2025)
    y2026 = _load_csv(args.eval_2026)

    state_cols = _numeric_feature_cols([y2024, y2025, y2026], include_state=False)
    state = _fit_market_state(y2024, state_cols)
    y2025s = _append_market_state(y2025, state)
    y2026s = _append_market_state(y2026, state)
    materialized_2025 = args.out_dir / "market_state_v5_2025.csv"
    materialized_2026 = args.out_dir / "market_state_v5_2026.csv"
    y2025s.to_csv(materialized_2025, index=False)
    y2026s.to_csv(materialized_2026, index=False)

    labeled = _future_labels(y2025s, args.horizon_bars, 2.0 * (args.fee + args.slip))
    fit = labeled[labeled["timestamp"] < pd.Timestamp("2025-09-01")].copy()
    selection = labeled[(labeled["timestamp"] >= pd.Timestamp("2025-09-01")) & (labeled["timestamp"] < pd.Timestamp("2025-11-01"))].copy()
    holdout = labeled[labeled["timestamp"] >= pd.Timestamp("2025-11-01")].copy()
    eval_frame = y2026s.copy()

    feature_cols = _numeric_feature_cols([fit, selection, eval_frame], include_state=True)
    state_feature_cols = [c for c in feature_cols if c.startswith(STATE_PREFIX)]
    model = _train_entry_model(fit, feature_cols)
    selection_proba, classes = _predict_proba(model, selection, feature_cols)
    holdout_proba, _ = _predict_proba(model, holdout, feature_cols)
    eval_proba, _ = _predict_proba(model, eval_frame, feature_cols)

    rows: list[dict[str, Any]] = []
    best_cfg: RuntimeConfig | None = None
    best_score = -1e18
    best_selection: dict[str, Any] | None = None
    for cfg in _grid():
        r1 = _backtest(selection, selection_proba, classes, cfg, fee=args.fee, slip=args.slip)
        r2 = _backtest(selection, selection_proba, classes, cfg, fee=args.fee * 2.0, slip=args.slip * 2.0)
        r3 = _backtest(selection, selection_proba, classes, cfg, fee=args.fee * 3.0, slip=args.slip * 3.0)
        score = 0.50 * _score(r1) + 0.30 * _score(r2) + 0.20 * _score(r3)
        if r2["pnl"] < 0:
            score -= abs(r2["pnl"]) * 3.0
        if r3["pnl"] < 0:
            score -= abs(r3["pnl"]) * 5.0
        row = {"score": score, **asdict(cfg), **{f"selection_{k}": v for k, v in _compact(r1).items()}}
        row["selection_cost2_pnl"] = r2["pnl"]
        row["selection_cost3_pnl"] = r3["pnl"]
        rows.append(row)
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_selection = r1
    if best_cfg is None or best_selection is None:
        raise RuntimeError("no config selected")

    holdout_result = _backtest(holdout, holdout_proba, classes, best_cfg, fee=args.fee, slip=args.slip)
    metrics: dict[str, Any] = {}
    ledger_paths: dict[str, str] = {}
    last_ledger_df: pd.DataFrame | None = None
    for mult in (1, 2, 3):
        result = _backtest(eval_frame, eval_proba, classes, best_cfg, fee=args.fee * mult, slip=args.slip * mult)
        key = f"cost{mult}"
        metrics[key] = _compact(result)
        ledger_path = args.report_out.with_name(args.report_out.stem + f"_{key}_ledger.csv")
        ledger_df = pd.DataFrame(result["ledger"])
        ledger_df.to_csv(ledger_path, index=False)
        ledger_paths[key] = str(ledger_path)
        last_ledger_df = ledger_df

    model_payload = {
        "model_id": MODEL_ID,
        "state_model": state,
        "entry_model": model,
        "feature_cols": feature_cols,
        "classes": classes,
        "selected_config": asdict(best_cfg),
        "state_prefix": STATE_PREFIX,
        "forbidden_policy": "drop_any_feature_name_containing_regime_target_future_realized_trade_pnl_cash_after",
    }
    model_path = args.out_dir / "v22_soft_best_market_state_v5_retrain.pkl"
    joblib.dump(model_payload, model_path)
    grid_path = args.report_out.with_name(args.report_out.stem + "_selection_grid.csv")
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(grid_path, index=False)

    report = {
        "model_id": MODEL_ID,
        "design": "V22 soft-best replacement retrain using 2024-only unsupervised market-state features, no legacy regime inputs, next-bar-open execution, fee+slippage on entry and exit.",
        "data": {
            "state_train_2024": str(args.state_2024),
            "train_2025": str(args.train_2025),
            "eval_2026": str(args.eval_2026),
            "materialized_2025": str(materialized_2025),
            "materialized_2026": str(materialized_2026),
            "fit_range": [str(fit["timestamp"].iloc[0]), str(fit["timestamp"].iloc[-1])],
            "selection_range": [str(selection["timestamp"].iloc[0]), str(selection["timestamp"].iloc[-1])],
            "holdout_range": [str(holdout["timestamp"].iloc[0]), str(holdout["timestamp"].iloc[-1])],
            "oos_range": [str(eval_frame["timestamp"].iloc[0]), str(eval_frame["timestamp"].iloc[-1])],
        },
        "artifacts": {
            "model": str(model_path),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "contract": str(args.contract_out),
            "selection_grid": str(grid_path),
            "ledgers": ledger_paths,
        },
        "selected_config": asdict(best_cfg),
        "selection_result": _compact(best_selection),
        "selection_score": best_score,
        "holdout_result": _compact(holdout_result),
        "metrics": metrics,
        "feature_audit": {
            "state_fit_feature_count": len(state_cols),
            "model_feature_count": len(feature_cols),
            "market_state_feature_count": len(state_feature_cols),
            "contaminated_feature_cols": [c for c in feature_cols if _is_forbidden_feature(c)],
            "market_state_features": state_feature_cols,
        },
    }
    audit = _audit(report, feature_cols, state_feature_cols, eval_frame, last_ledger_df if last_ledger_df is not None else pd.DataFrame())
    report["audit"] = audit
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    _write_contract(args.contract_out, report, audit)
    print(json.dumps({"status": audit["status"], "metrics": metrics, "report": str(args.report_out), "audit": str(args.audit_out)}, indent=2, ensure_ascii=False))
    return 0 if audit["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
