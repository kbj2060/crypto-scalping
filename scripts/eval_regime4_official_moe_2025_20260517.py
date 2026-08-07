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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.certified_teacher_regime_moe import label_frame, json_default  # noqa: E402


MODEL_ID = "regime4_official_moe_2025_v1"
CLEAN4_PREFIX = "clean_regime4_2024_unsup_v1_"
PRED4_PREFIX = "regime4_pred_"
REGIMES = ("bull", "bear", "chop", "whipsaw")

DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime4_official_moe_2025_v1_20260517"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime4_official_moe_2025_ablation_20260517.json"

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
    adverse_cap: float
    regime_conf_floor: float
    candidate_stride: int


@dataclass
class Position:
    side: int
    signal_idx: int
    entry_idx: int
    entry_price: float
    notional: float
    leverage: float
    prob: float
    gap: float
    predicted_adverse: float
    regime_state: str
    entry_cost: float
    peak_raw: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate official 4-class Regime4 features as MoE routing inputs on 2025.")
    p.add_argument("--base-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--ai-2025", type=Path, default=ROOT / "data/tmp/unified_build_ckpt/03_after_ai.csv")
    p.add_argument("--m7-2025", type=Path, default=ROOT / "data/splits/year_oos/rl_training_2025_m7.csv")
    p.add_argument("--clean4-2025", type=Path, default=ROOT / "data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/training_features_2025_clean_regime4_raw_state12_v1.csv")
    p.add_argument("--pred4-2025", type=Path, default=ROOT / "data/ensemble/supervised/regime4_pred_tft_vsn_h12_official_20260517/training_features_2025_regime4_pred_tft_vsn_selected.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--horizon-bars", type=int, default=36)
    p.add_argument("--max-features", type=int, default=112)
    p.add_argument("--max-grid", type=int, default=24)
    p.add_argument("--row-limit", type=int, default=0, help="debug only; tail limit after merging")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    return p.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def merge_by_timestamp(base: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in add.columns if c == "timestamp" or c not in base.columns]
    if len(cols) <= 1:
        return base.copy()
    out = base.merge(add[cols], on="timestamp", how="left")
    return out.sort_values("timestamp").reset_index(drop=True)


def merge_teacher_sources(base: pd.DataFrame, ai: pd.DataFrame, m7: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    ai_cols = [
        c for c in ai.columns
        if c == "timestamp"
        or c.startswith(("ai_", "patchtst_", "tide_", "timesnet_", "dlinear_"))
        or c in {"pred_patchtst", "conf_patchtst"}
    ]
    out = merge_by_timestamp(out, ai[ai_cols])
    m7_cols = [c for c in m7.columns if c == "timestamp" or c.startswith("m7_")]
    out = merge_by_timestamp(out, m7[m7_cols])
    return out


def forbidden_feature(col: str) -> bool:
    lower = col.lower()
    if col in NON_FEATURES:
        return True
    if lower.startswith("_"):
        return True
    if lower.startswith((CLEAN4_PREFIX, PRED4_PREFIX)):
        return False
    if lower.startswith("clean_regime_") or lower.startswith("regime_"):
        return True
    if lower.startswith(("future", "target", "label")):
        return True
    if "future" in lower or "realized" in lower:
        return True
    if lower in {"cvp_regime", "regime_bull", "regime_bear", "regime_chop", "regime_whipsaw", "regime_normal", "regime_trending", "regime_break"}:
        return True
    if "legacy" in lower or "regime_v2" in lower or "hdb" in lower or lower.startswith("hmm_"):
        return True
    if lower in {"trade_pnl_pct", "cash_after", "entry_fee_cash", "exit_fee_cash"}:
        return True
    return False


def matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) for c in cols},
        index=frame.index,
    )


def candidate_feature_cols(frames: list[pd.DataFrame]) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    cols: list[str] = []
    for col in sorted(common):
        if forbidden_feature(col):
            continue
        if any(pd.to_numeric(frame[col], errors="coerce").notna().any() for frame in frames):
            cols.append(col)
    return cols


def variant_features(all_cols: list[str], variant: str) -> tuple[list[str], str | None, list[str]]:
    current_cols = [c for c in all_cols if c.startswith(CLEAN4_PREFIX)]
    future_cols = [c for c in all_cols if c.startswith(PRED4_PREFIX)]
    if variant == "baseline":
        return [c for c in all_cols if not c.startswith((CLEAN4_PREFIX, PRED4_PREFIX))], None, []
    if variant == "regime4_current":
        return [c for c in all_cols if not c.startswith(PRED4_PREFIX)], CLEAN4_PREFIX, current_cols
    if variant == "regime4_future":
        return [c for c in all_cols if not c.startswith(CLEAN4_PREFIX)], PRED4_PREFIX, future_cols
    if variant == "regime4_both":
        return all_cols, PRED4_PREFIX, current_cols + future_cols
    raise ValueError(f"unknown variant {variant}")


def feature_analysis(fit: pd.DataFrame, cols: list[str], forced: list[str], max_features: int) -> tuple[list[str], list[dict[str, Any]]]:
    x = matrix(fit, cols)
    y = fit["_label"].astype(int).to_numpy()
    sample_n = min(len(x), 50000)
    if sample_n < len(x):
        rng = np.random.default_rng(417)
        idx = np.sort(rng.choice(len(x), size=sample_n, replace=False))
        xs = x.iloc[idx]
        ys = y[idx]
    else:
        xs, ys = x, y
    try:
        mi = mutual_info_classif(xs, ys, random_state=417, discrete_features=False)
    except Exception:
        mi = np.zeros(len(cols), dtype=float)
    rows: list[dict[str, Any]] = []
    selected: list[str] = []
    for col in forced:
        if col in cols and col not in selected:
            selected.append(col)
    for col, val in zip(cols, mi):
        family = (
            "regime4_current" if col.startswith(CLEAN4_PREFIX)
            else "regime4_future" if col.startswith(PRED4_PREFIX)
            else "m7" if col.startswith("m7_")
            else "ai" if col.startswith(("ai_", "patchtst_", "tide_", "timesnet_", "dlinear_")) or col in {"pred_patchtst", "conf_patchtst"}
            else "market"
        )
        rows.append({"feature": col, "mutual_info": float(val), "family": family})
    for row in sorted(rows, key=lambda r: r["mutual_info"], reverse=True):
        col = str(row["feature"])
        if col not in selected and float(row["mutual_info"]) > 0.0:
            selected.append(col)
        if len(selected) >= max_features:
            break
    return selected[:max_features], sorted(rows, key=lambda r: r["mutual_info"], reverse=True)


def _fit_classifier(fit: pd.DataFrame, cols: list[str], seed: int) -> Any:
    clf = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.042,
        max_leaf_nodes=31,
        l2_regularization=0.12,
        min_samples_leaf=22,
        early_stopping=False,
        random_state=seed,
    )
    model = make_pipeline(SimpleImputer(strategy="median"), clf)
    model.fit(matrix(fit, cols), fit["_label"].astype(int).to_numpy())
    return model


def _fit_regressor(fit: pd.DataFrame, cols: list[str], target: str, seed: int) -> Any:
    reg = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.046,
        max_leaf_nodes=31,
        l2_regularization=0.09,
        min_samples_leaf=22,
        early_stopping=False,
        random_state=seed,
    )
    model = make_pipeline(SimpleImputer(strategy="median"), reg)
    model.fit(matrix(fit, cols), pd.to_numeric(fit[target], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    return model


def _predict_proba_3(model: Any, frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(matrix(frame, cols)), dtype=float)
    clf = getattr(model, "named_steps", {}).get("histgradientboostingclassifier", model)
    classes = [int(c) for c in getattr(clf, "classes_", [])]
    out = np.zeros((len(frame), 3), dtype=float)
    for cls in (0, 1, 2):
        if cls in classes:
            out[:, cls] = raw[:, classes.index(cls)]
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out


def regime_prob_matrix(frame: pd.DataFrame, prefix: str | None) -> np.ndarray:
    if prefix is None:
        return np.zeros((len(frame), len(REGIMES)), dtype=float)
    mat = np.column_stack([
        pd.to_numeric(frame.get(f"{prefix}{name}_prob", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for name in REGIMES
    ])
    total = mat.sum(axis=1, keepdims=True)
    return mat / np.clip(total, 1e-12, None)


def train_moe(fit: pd.DataFrame, cols: list[str], gate_prefix: str | None) -> dict[str, Any]:
    global_model = _fit_classifier(fit, cols, 417)
    experts: dict[str, Any] = {}
    if gate_prefix is not None:
        probs = regime_prob_matrix(fit, gate_prefix)
        hard = probs.argmax(axis=1)
        for k, name in enumerate(REGIMES):
            sub = fit.iloc[np.flatnonzero(hard == k)]
            if len(sub) >= 1500 and sub["_label"].nunique() >= 2:
                experts[name] = _fit_classifier(sub, cols, 520 + k)
    risk_long = _fit_regressor(fit, cols, "_long_adverse", 601)
    risk_short = _fit_regressor(fit, cols, "_short_adverse", 602)
    return {
        "global_model": global_model,
        "experts": experts,
        "risk_long": risk_long,
        "risk_short": risk_short,
        "feature_cols": cols,
        "gate_prefix": gate_prefix,
    }


def predict_moe(bundle: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = list(bundle["feature_cols"])
    global_proba = _predict_proba_3(bundle["global_model"], frame, cols)
    gate_prefix = bundle.get("gate_prefix")
    experts = dict(bundle.get("experts", {}) or {})
    if gate_prefix is None or not experts:
        proba = global_proba
    else:
        gate = regime_prob_matrix(frame, str(gate_prefix))
        expert_mix = np.zeros_like(global_proba)
        covered = np.zeros(len(frame), dtype=float)
        for k, name in enumerate(REGIMES):
            if name not in experts:
                continue
            ep = _predict_proba_3(experts[name], frame, cols)
            expert_mix += gate[:, [k]] * ep
            covered += gate[:, k]
        fallback = global_proba * np.clip(1.0 - covered[:, None], 0.0, 1.0)
        expert_mix += fallback
        confidence = gate.max(axis=1)
        blend = np.clip(0.25 + 0.45 * confidence, 0.25, 0.70)
        proba = global_proba * (1.0 - blend[:, None]) + expert_mix * blend[:, None]
    proba /= np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    risk_long = np.asarray(bundle["risk_long"].predict(matrix(frame, cols)), dtype=float)
    risk_short = np.asarray(bundle["risk_short"].predict(matrix(frame, cols)), dtype=float)
    return proba, risk_long, risk_short


def runtime_grid() -> list[RuntimeConfig]:
    out: list[RuntimeConfig] = []
    for threshold in (0.42, 0.46, 0.50, 0.54):
        for gap in (0.04, 0.08, 0.12):
            for adverse in (0.010, 0.014, 0.020):
                for max_n in (1.0, 1.8, 2.6):
                    out.append(RuntimeConfig(threshold, gap, max_n, 0.25, 5.0, 36, 0.012, 0.034, 0.010, 2, adverse, 0.22, 6))
    return out


def _raw_ret(side: int, entry: float, price: float) -> float:
    if entry <= 0.0 or price <= 0.0:
        return 0.0
    return float(side) * (float(price) / float(entry) - 1.0)


def _regime_conf(row: pd.Series, gate_prefix: str | None) -> float:
    if gate_prefix is None:
        return 1.0
    return float(row.get(f"{gate_prefix}confidence", 0.0) or 0.0)


def _directional_bias(row: pd.Series, gate_prefix: str | None) -> float:
    for prefix in ([gate_prefix] if gate_prefix else []) + [PRED4_PREFIX, CLEAN4_PREFIX]:
        if prefix and f"{prefix}directional_bias" in row:
            return float(row.get(f"{prefix}directional_bias", 0.0) or 0.0)
    return 0.0


def _state_name(row: pd.Series, gate_prefix: str | None) -> str:
    if gate_prefix is None:
        return "global"
    probs = {name: float(row.get(f"{gate_prefix}{name}_prob", 0.0) or 0.0) for name in REGIMES}
    return max(probs, key=probs.get)


def _ledger_row(frame: pd.DataFrame, pos: Position, exit_idx: int, exit_price: float, realized: float, gross: float, exit_cost: float, equity: float, reason: str, trade_id: int) -> dict[str, Any]:
    return {
        "trade_id": int(trade_id),
        "timestamp": str(frame.iloc[pos.signal_idx]["timestamp"]),
        "entry_time": str(frame.iloc[pos.entry_idx]["timestamp"]),
        "exit_time": str(frame.iloc[exit_idx]["timestamp"]),
        "entry_idx": int(pos.entry_idx),
        "exit_idx": int(exit_idx),
        "side": "LONG" if pos.side > 0 else "SHORT",
        "action": "trade",
        "sleeve": MODEL_ID,
        "regime_state": pos.regime_state,
        "entry_price": float(pos.entry_price),
        "exit_price": float(exit_price),
        "notional": float(pos.notional),
        "leverage": float(pos.leverage),
        "margin_fraction": float(pos.notional / max(pos.leverage, 1e-12)),
        "probability": float(pos.prob),
        "gap": float(pos.gap),
        "predicted_adverse": float(pos.predicted_adverse),
        "realized_raw": float(realized),
        "entry_fee_cash": float(pos.entry_cost),
        "exit_fee_cash": float(exit_cost),
        "trade_pnl_pct": float((gross - pos.entry_cost - exit_cost) * 100.0),
        "cash_after": float(equity),
        "blocked": False,
        "stop_reason": reason,
    }


def backtest(frame: pd.DataFrame, proba: np.ndarray, risk_long: np.ndarray, risk_short: np.ndarray, cfg: RuntimeConfig, *, fee: float, slip: float, gate_prefix: str | None) -> dict[str, Any]:
    cost_side = float(fee) + float(slip)
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
            raw = _raw_ret(pos.side, pos.entry_price, close)
            pos.peak_raw = max(pos.peak_raw, raw)
            mark = equity * max(0.0, 1.0 + pos.notional * raw)
            peak = max(peak, mark)
            min_equity = min(min_equity, mark)
            reason = ""
            if raw <= -cfg.stop_loss:
                reason = "stop_loss"
            elif raw >= cfg.take_profit:
                reason = "take_profit"
            elif pos.peak_raw >= cfg.trailing_stop * 1.15 and raw <= pos.peak_raw - cfg.trailing_stop:
                reason = "trailing_stop"
            elif i - pos.entry_idx >= cfg.max_hold_bars:
                reason = "max_hold"
            if reason:
                realized = _raw_ret(pos.side, pos.entry_price, next_open)
                exit_cost = pos.notional * cost_side
                gross = pos.notional * realized
                equity *= max(0.0, 1.0 + gross - exit_cost)
                peak = max(peak, equity)
                min_equity = min(min_equity, equity)
                ledger.append(_ledger_row(frame, pos, i + 1, next_open, realized, gross, exit_cost, equity, reason, len(ledger)))
                pos = None
                last_exit = i + 1
                continue
        if pos is not None or i <= last_exit + cfg.cooldown_bars or i % max(1, cfg.candidate_stride) != 0:
            continue
        long_p, short_p, no_p = float(proba[i, 1]), float(proba[i, 2]), float(proba[i, 0])
        side = 1 if long_p >= short_p else -1
        p = long_p if side > 0 else short_p
        alt = short_p if side > 0 else long_p
        gap = p - max(alt, 0.35 * no_p)
        row = frame.iloc[i]
        regime_conf = _regime_conf(row, gate_prefix)
        directional_bias = _directional_bias(row, gate_prefix)
        adverse = float(risk_long[i] if side > 0 else risk_short[i])
        reason = ""
        if p < cfg.threshold:
            reason = "probability_below_threshold"
        elif gap < cfg.gap:
            reason = "gap_below_threshold"
        elif regime_conf < cfg.regime_conf_floor:
            reason = "regime_confidence_below_floor"
        elif adverse > cfg.adverse_cap:
            reason = "adverse_cap"
        elif side * directional_bias < -0.42:
            reason = "direction_state_conflict"
        if reason:
            block_counts[reason] = block_counts.get(reason, 0) + 1
            continue
        edge_scale = ((p - cfg.threshold) / max(1.0 - cfg.threshold, 1e-9)) ** 0.70
        state_scale = np.clip(0.80 + 0.40 * regime_conf, 0.25, 1.20)
        risk_scale = np.clip(1.0 - adverse / max(cfg.adverse_cap, 1e-9) * 0.55, 0.25, 1.0)
        notional = float(np.clip(cfg.min_notional + (cfg.max_notional - cfg.min_notional) * edge_scale * state_scale * risk_scale, cfg.min_notional, cfg.max_notional))
        equity *= max(0.0, 1.0 - notional * cost_side)
        min_equity = min(min_equity, equity)
        pos = Position(side, i, i + 1, next_open, notional, cfg.leverage, p, gap, adverse, _state_name(row, gate_prefix), notional * cost_side)
    if pos is not None:
        i = len(frame) - 1
        exit_price = float(frame.iloc[i]["close"])
        realized = _raw_ret(pos.side, pos.entry_price, exit_price)
        exit_cost = pos.notional * cost_side
        gross = pos.notional * realized
        equity *= max(0.0, 1.0 + gross - exit_cost)
        min_equity = min(min_equity, equity)
        ledger.append(_ledger_row(frame, pos, i, exit_price, realized, gross, exit_cost, equity, "end", len(ledger)))
    ledger.append({"trade_id": -1, "timestamp": str(frame.iloc[-1]["timestamp"]), "action": "coverage_end", "side": "COVERAGE", "cash_after": float(equity), "stop_reason": "coverage_end"})
    trades = [r for r in ledger if r.get("action") == "trade"]
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    days = max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0, 1e-12)
    trade_returns = np.asarray([float(r["trade_pnl_pct"]) / 100.0 for r in trades], dtype=float)
    wins = int(np.sum(trade_returns > 0.0)) if trade_returns.size else 0
    trade_sharpe = 0.0
    if trade_returns.size >= 2 and float(np.std(trade_returns)) > 1e-12:
        trade_sharpe = float(np.mean(trade_returns) / np.std(trade_returns) * math.sqrt(trade_returns.size))
    return {
        "pnl": float((equity - 1.0) * 100.0),
        "mdd": float((min_equity / max(peak, 1e-12) - 1.0) * 100.0),
        "trades": int(len(trades)),
        "trades_per_day": float(len(trades) / days),
        "wr": float(wins / len(trades)) if trades else 0.0,
        "trade_sharpe": float(trade_sharpe),
        "avg_notional": float(np.mean([float(r["notional"]) for r in trades])) if trades else 0.0,
        "max_margin_fraction": float(np.max([float(r["margin_fraction"]) for r in trades])) if trades else 0.0,
        "final_equity": float(equity),
        "coverage_start": str(frame.iloc[0]["timestamp"]),
        "coverage_end": str(frame.iloc[-1]["timestamp"]),
        "block_reason_counts": block_counts,
        "ledger": ledger,
    }


def score(result: dict[str, Any]) -> float:
    pnl = float(result["pnl"])
    mdd = abs(float(result["mdd"]))
    trades = int(result["trades"])
    if trades < 20:
        return -1e9 + pnl
    return float(pnl + 0.06 * min(trades, 240) + 1.8 * pnl / max(mdd, 1.0) - max(0.0, mdd - 16.0) * 4.0)


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "ledger"}


def class_metrics(frame: pd.DataFrame, proba: np.ndarray) -> dict[str, Any]:
    y = frame["_label"].astype(int).to_numpy()
    pred = proba.argmax(axis=1)
    out: dict[str, Any] = {"accuracy": float(np.mean(pred == y)), "rows": int(len(y))}
    for cls, name in ((0, "none"), (1, "long"), (2, "short")):
        mask = y == cls
        out[f"recall_{name}"] = float(np.mean(pred[mask] == cls)) if np.any(mask) else 0.0
    return out


def validate_regime_sidecars(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for prefix in (CLEAN4_PREFIX, PRED4_PREFIX):
        prob_cols = [f"{prefix}{name}_prob" for name in REGIMES]
        missing = [c for c in prob_cols if c not in frame.columns]
        if missing:
            out[prefix] = {"status": "fail", "missing": missing}
            continue
        sums = frame[prob_cols].sum(axis=1)
        out[prefix] = {
            "status": "pass",
            "prob_sum_min": float(sums.min()),
            "prob_sum_max": float(sums.max()),
            "nan_count": int(frame[prob_cols].isna().sum().sum()),
            "rows": int(len(frame)),
        }
    return out


def split_ranges(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit = frame[frame["timestamp"] < pd.Timestamp("2025-09-01")].copy()
    selection = frame[(frame["timestamp"] >= pd.Timestamp("2025-09-01")) & (frame["timestamp"] < pd.Timestamp("2025-11-01"))].copy()
    holdout = frame[frame["timestamp"] >= pd.Timestamp("2025-11-01")].copy()
    if fit.empty or selection.empty or holdout.empty:
        raise ValueError("empty 2025 fit/selection/holdout split")
    return fit, selection, holdout


def evaluate_variant(name: str, fit: pd.DataFrame, selection: pd.DataFrame, holdout: pd.DataFrame, all_cols: list[str], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    cols, gate_prefix, forced = variant_features(all_cols, name)
    selected, feature_rows = feature_analysis(fit, cols, forced, int(args.max_features))
    bundle = train_moe(fit, selected, gate_prefix)
    sel_proba, sel_risk_l, sel_risk_s = predict_moe(bundle, selection)
    hold_proba, hold_risk_l, hold_risk_s = predict_moe(bundle, holdout)
    grid = runtime_grid()
    if args.max_grid and args.max_grid > 0:
        grid = grid[: int(args.max_grid)]
    best_score = -1e18
    best_cfg: RuntimeConfig | None = None
    best_selection: dict[str, Any] | None = None
    grid_rows: list[dict[str, Any]] = []
    for idx, cfg in enumerate(grid, start=1):
        if idx == 1 or idx % 8 == 0 or idx == len(grid):
            print(f"[{MODEL_ID}] {name} grid {idx}/{len(grid)}", flush=True)
        cost1 = backtest(selection, sel_proba, sel_risk_l, sel_risk_s, cfg, fee=args.fee, slip=args.slip, gate_prefix=gate_prefix)
        cost2 = backtest(selection, sel_proba, sel_risk_l, sel_risk_s, cfg, fee=args.fee * 2.0, slip=args.slip * 2.0, gate_prefix=gate_prefix)
        cost3 = backtest(selection, sel_proba, sel_risk_l, sel_risk_s, cfg, fee=args.fee * 3.0, slip=args.slip * 3.0, gate_prefix=gate_prefix)
        s = 0.50 * score(cost1) + 0.30 * score(cost2) + 0.20 * score(cost3)
        if cost2["pnl"] < 0:
            s -= abs(float(cost2["pnl"])) * 2.0
        if cost3["pnl"] < 0:
            s -= abs(float(cost3["pnl"])) * 3.5
        row = {"variant": name, "score": float(s), **asdict(cfg)}
        row.update({f"selection_{k}": v for k, v in compact(cost1).items()})
        row.update({"selection_cost2_pnl": cost2["pnl"], "selection_cost3_pnl": cost3["pnl"]})
        grid_rows.append(row)
        if s > best_score:
            best_score = float(s)
            best_cfg = cfg
            best_selection = cost1
    if best_cfg is None or best_selection is None:
        raise RuntimeError(f"{name} produced no runtime config")
    holdout_costs: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        result = backtest(holdout, hold_proba, hold_risk_l, hold_risk_s, best_cfg, fee=args.fee * mult, slip=args.slip * mult, gate_prefix=gate_prefix)
        key = f"cost{mult}"
        holdout_costs[key] = compact(result)
        ledger_path = args.report_out.with_name(args.report_out.stem + f"_{name}_{key}_ledger.csv")
        pd.DataFrame(result["ledger"]).to_csv(ledger_path, index=False)
        ledgers[key] = str(ledger_path)
    model_path = args.out_dir / f"{name}_model.pkl"
    joblib.dump({"model_id": MODEL_ID, "variant": name, "bundle": bundle, "selected_config": asdict(best_cfg)}, model_path)
    feature_path = args.report_out.with_name(args.report_out.stem + f"_{name}_feature_analysis.csv")
    pd.DataFrame(feature_rows).to_csv(feature_path, index=False)
    grid_path = args.report_out.with_name(args.report_out.stem + f"_{name}_selection_grid.csv")
    pd.DataFrame(grid_rows).sort_values("score", ascending=False).to_csv(grid_path, index=False)
    result = {
        "variant": name,
        "gate_prefix": gate_prefix,
        "selected_features": selected,
        "selected_feature_count": int(len(selected)),
        "regime_feature_count": int(sum(c.startswith((CLEAN4_PREFIX, PRED4_PREFIX)) for c in selected)),
        "expert_count": int(len(bundle.get("experts", {}) or {})),
        "experts": sorted((bundle.get("experts", {}) or {}).keys()),
        "classification": {
            "selection": class_metrics(selection, sel_proba),
            "holdout": class_metrics(holdout, hold_proba),
        },
        "best_selection_score": float(best_score),
        "best_config": asdict(best_cfg),
        "selection_cost1": compact(best_selection),
        "holdout": holdout_costs,
        "artifacts": {
            "model": str(model_path),
            "feature_analysis": str(feature_path),
            "selection_grid": str(grid_path),
            "ledgers": ledgers,
        },
    }
    return result, {"variant": name, "grid_rows": grid_rows}


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    base = load_csv(args.base_2025)
    ai = load_csv(args.ai_2025)
    m7 = load_csv(args.m7_2025)
    clean4 = load_csv(args.clean4_2025)
    pred4 = load_csv(args.pred4_2025)

    frame = merge_teacher_sources(base, ai, m7)
    frame = merge_by_timestamp(frame, clean4)
    frame = merge_by_timestamp(frame, pred4)
    if args.row_limit and args.row_limit > 0 and len(frame) > args.row_limit:
        frame = frame.tail(int(args.row_limit)).reset_index(drop=True)
    sidecar_audit = validate_regime_sidecars(frame)
    if any(v.get("status") != "pass" for v in sidecar_audit.values()):
        raise ValueError(f"regime sidecar audit failed: {sidecar_audit}")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    labeled = label_frame(frame, int(args.horizon_bars), float(args.fee), float(args.slip))
    fit, selection, holdout = split_ranges(labeled)
    all_cols = candidate_feature_cols([fit, selection, holdout])

    variants = ["baseline", "regime4_current", "regime4_future", "regime4_both"]
    results: list[dict[str, Any]] = []
    for variant in variants:
        print(f"[{MODEL_ID}] training {variant}", flush=True)
        result, _ = evaluate_variant(variant, fit, selection, holdout, all_cols, args)
        results.append(result)

    ranking = sorted(
        [
            {
                "variant": r["variant"],
                "holdout_cost1_pnl": r["holdout"]["cost1"]["pnl"],
                "holdout_cost1_mdd": r["holdout"]["cost1"]["mdd"],
                "holdout_cost1_trade_sharpe": r["holdout"]["cost1"]["trade_sharpe"],
                "holdout_cost1_trades": r["holdout"]["cost1"]["trades"],
                "holdout_accuracy": r["classification"]["holdout"]["accuracy"],
                "expert_count": r["expert_count"],
            }
            for r in results
        ],
        key=lambda r: (float(r["holdout_cost1_pnl"]), float(r["holdout_cost1_trade_sharpe"])),
        reverse=True,
    )
    best = ranking[0]["variant"] if ranking else None
    report = {
        "model_id": MODEL_ID,
        "purpose": "Evaluate official 4-class current/future regime features as MoE expert gates and trading-model inputs.",
        "official_regime_contract": "4-class bull/bear/chop/whipsaw; no normal/risk_off/transition classes.",
        "data": {
            "base_2025": str(args.base_2025),
            "ai_2025": str(args.ai_2025),
            "m7_2025": str(args.m7_2025),
            "clean4_2025": str(args.clean4_2025),
            "pred4_2025": str(args.pred4_2025),
            "rows_merged": int(len(frame)),
            "rows_labeled": int(len(labeled)),
            "fit_range": [str(fit["timestamp"].iloc[0]), str(fit["timestamp"].iloc[-1])],
            "selection_range": [str(selection["timestamp"].iloc[0]), str(selection["timestamp"].iloc[-1])],
            "holdout_range": [str(holdout["timestamp"].iloc[0]), str(holdout["timestamp"].iloc[-1])],
            "horizon_bars": int(args.horizon_bars),
        },
        "audit": {
            "sidecars": sidecar_audit,
            "forbidden_feature_count": int(sum(forbidden_feature(c) for c in frame.columns)),
            "candidate_features": int(len(all_cols)),
            "selection_bias_warning": "2025 holdout is internal downstream validation, not a final 2026 OOS result.",
        },
        "variants": results,
        "ranking": ranking,
        "best_variant": best,
    }
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    compact_path = args.report_out.with_name(args.report_out.stem + "_compact.csv")
    pd.DataFrame(ranking).to_csv(compact_path, index=False)
    print(json.dumps({"report": str(args.report_out), "compact": str(compact_path), "best_variant": best, "ranking": ranking}, ensure_ascii=False, default=json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
