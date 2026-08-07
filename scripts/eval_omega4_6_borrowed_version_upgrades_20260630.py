#!/usr/bin/env python3
"""Borrowed Omega-version upgrade diagnostics for Omega 4.6.

This is a ledger-level diagnostic. It does not retrain parent heads and does
not promote a runtime contract. Selection ranking is validation-only; OOS is a
readout.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = ROOT / "tmp/causal_regen_20260516/omega4_6_plus_t12_nohold_risk1_20260630/runtime_contract.json"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_borrowed_version_upgrade_tests_20260630"


@dataclass(frozen=True)
class EntryRule:
    name: str
    source_idea: str
    feature: str
    op: str
    threshold: float
    scope: str
    hit_scale: float


@dataclass(frozen=True)
class ExposureSpec:
    name: str
    source_idea: str
    mode: str
    long_factor: float
    short_factor: float
    cap_notional: float


@dataclass(frozen=True)
class LifecycleSpec:
    name: str
    source_idea: str
    mode: str
    side: str
    cap_hours: float
    min_unrealized_price_move: float
    time_stop_hours: float = 0.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(obj: Any) -> Any:
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


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def component_train_eval_paths(runtime: dict[str, Any]) -> tuple[Path, Path]:
    risk_report = read_json(resolve_path(runtime["components"]["h48qual"]["report"]))
    risk_model = risk_report["risk_model"]
    return resolve_path(risk_model["train_csv"]), resolve_path(risk_model["eval_csv"])


def safe_feature_columns(df: pd.DataFrame) -> list[str]:
    forbidden = (
        "label",
        "target",
        "future",
        "fwd",
        "return_fwd",
        "pnl",
        "win",
        "exit",
        "barrier",
        "zigzag",
        "clean_regime4_2024_unsup_v1",
        "clean_regime_2024_unsup_v4",
    )
    excluded = {"timestamp", "open", "high", "low", "close"}
    cols: list[str] = []
    for col in df.columns:
        low = col.lower()
        if col in excluded or any(tok in low for tok in forbidden):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            finite_ratio = np.isfinite(pd.to_numeric(df[col], errors="coerce")).mean()
            if finite_ratio >= 0.95 and df[col].nunique(dropna=True) >= 5:
                cols.append(col)
    return cols


def load_market(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    return df.sort_values("timestamp").reset_index(drop=True)


def join_entry_features(ledger: pd.DataFrame, market: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = ledger.copy()
    out["entry_timestamp_dt"] = pd.to_datetime(out["entry_timestamp"], errors="raise")
    out["exit_timestamp_dt"] = pd.to_datetime(out["exit_timestamp"], errors="raise")
    feat = market[["timestamp", *feature_cols]].rename(columns={"timestamp": "entry_timestamp_dt"})
    out = out.merge(feat, on="entry_timestamp_dt", how="left", validate="many_to_one")
    missing_mask = out[feature_cols].isna().any(axis=1)
    if bool(missing_mask.any()):
        missing = out.loc[missing_mask, "entry_timestamp"].head(5).tolist()
        raise RuntimeError(f"missing entry features for timestamps: {missing}")
    out["hold_hours"] = (out["exit_timestamp_dt"] - out["entry_timestamp_dt"]).dt.total_seconds() / 3600.0
    out["entry_month"] = out["entry_timestamp_dt"].dt.to_period("M").astype(str)
    return out


def scope_mask(df: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "all":
        return np.ones(len(df), dtype=bool)
    if scope == "long":
        return df["side"].astype(int).to_numpy() > 0
    if scope == "short":
        return df["side"].astype(int).to_numpy() < 0
    if scope.endswith("_L"):
        source = scope[:-2]
        return (df["source_alias"].astype(str).to_numpy() == source) & (df["side"].astype(int).to_numpy() > 0)
    if scope.endswith("_S"):
        source = scope[:-2]
        return (df["source_alias"].astype(str).to_numpy() == source) & (df["side"].astype(int).to_numpy() < 0)
    raise ValueError(f"unknown scope: {scope}")


def condition_mask(df: pd.DataFrame, rule: EntryRule) -> np.ndarray:
    if rule.name == "none":
        return np.zeros(len(df), dtype=bool)
    vals = pd.to_numeric(df[rule.feature], errors="coerce").to_numpy(dtype=np.float64)
    if rule.op == "ge":
        cond = vals >= float(rule.threshold)
    elif rule.op == "le":
        cond = vals <= float(rule.threshold)
    else:
        raise ValueError(f"unknown op: {rule.op}")
    return cond & scope_mask(df, rule.scope) & np.isfinite(vals)


def set_exposure_from_notional(out: pd.DataFrame, target_notional: np.ndarray, leverage_cap: float) -> None:
    margin = out["margin_fraction"].astype(float).to_numpy(dtype=np.float64)
    target_notional = np.maximum(target_notional, 0.0)
    target_notional = np.minimum(target_notional, margin * float(leverage_cap))
    leverage = np.divide(
        target_notional,
        margin,
        out=np.zeros_like(target_notional, dtype=np.float64),
        where=np.abs(margin) > 1.0e-12,
    )
    out["notional"] = target_notional
    out["leverage"] = leverage
    out["risk_notional"] = target_notional
    out["risk_leverage"] = leverage
    if "exit_input_notional" in out.columns:
        out["exit_input_notional"] = target_notional
    if "exit_input_leverage" in out.columns:
        out["exit_input_leverage"] = leverage
    if "exit_input_exposure" in out.columns:
        out["exit_input_exposure"] = target_notional * leverage


def apply_entry_rule(df: pd.DataFrame, rule: EntryRule, leverage_cap: float, notional_cap: float) -> pd.DataFrame:
    out = df.copy()
    mask = condition_mask(out, rule)
    scale = np.ones(len(out), dtype=np.float64)
    scale[mask] = float(rule.hit_scale)
    target = out["notional"].astype(float).to_numpy(dtype=np.float64) * scale
    target = np.minimum(target, float(notional_cap))
    set_exposure_from_notional(out, target, leverage_cap)
    out["borrow_entry_rule"] = rule.name
    out["borrow_entry_source_idea"] = rule.source_idea
    out["borrow_entry_hit"] = mask
    out["borrow_entry_scale"] = scale
    out["borrow_entry_skipped"] = out["notional"].astype(float) <= 1.0e-12
    out["trade_return"] = out["net_per_notional"].astype(float) * out["notional"].astype(float)
    return out


def apply_exposure(df: pd.DataFrame, spec: ExposureSpec, leverage_cap: float, notional_cap: float) -> pd.DataFrame:
    out = df.copy()
    old = out["notional"].astype(float).to_numpy(dtype=np.float64)
    side = out["side"].astype(int).to_numpy()
    if spec.mode == "none":
        factor = np.ones(len(out), dtype=np.float64)
        cap = float(notional_cap)
    elif spec.mode == "side_factor":
        factor = np.where(side > 0, float(spec.long_factor), float(spec.short_factor))
        cap = min(float(spec.cap_notional), float(notional_cap))
    else:
        raise ValueError(f"unknown exposure mode: {spec.mode}")
    target = np.minimum(old * factor, cap)
    set_exposure_from_notional(out, target, leverage_cap)
    out["borrow_exposure_spec"] = spec.name
    out["borrow_exposure_source_idea"] = spec.source_idea
    out["borrow_exposure_factor"] = factor
    out["borrow_exposure_cap_notional"] = cap
    out["trade_return"] = out["net_per_notional"].astype(float) * out["notional"].astype(float)
    return out


def side_selected(side_value: int, side: str) -> bool:
    if side == "all":
        return True
    if side == "long":
        return side_value > 0
    if side == "short":
        return side_value < 0
    raise ValueError(f"unknown lifecycle side: {side}")


def apply_lifecycle(df: pd.DataFrame, market: pd.DataFrame, spec: LifecycleSpec) -> pd.DataFrame:
    out = df.copy()
    out["borrow_lifecycle_spec"] = spec.name
    out["borrow_lifecycle_source_idea"] = spec.source_idea
    out["borrow_lifecycle_hit"] = False
    if spec.mode == "none":
        return out
    if spec.mode == "profit_cap_then_time_stop":
        first = LifecycleSpec(
            name=f"{spec.name}_profit_leg",
            source_idea=spec.source_idea,
            mode="profit_cap_exit",
            side=spec.side,
            cap_hours=spec.cap_hours,
            min_unrealized_price_move=spec.min_unrealized_price_move,
        )
        second = LifecycleSpec(
            name=f"{spec.name}_time_stop_leg",
            source_idea=spec.source_idea,
            mode="time_stop",
            side="all",
            cap_hours=spec.time_stop_hours,
            min_unrealized_price_move=0.0,
        )
        out = apply_lifecycle(out, market, first)
        first_hits = out["borrow_lifecycle_hit"].astype(bool).to_numpy()
        out = apply_lifecycle(out, market, second)
        second_hits = out["borrow_lifecycle_hit"].astype(bool).to_numpy()
        out["borrow_lifecycle_spec"] = spec.name
        out["borrow_lifecycle_source_idea"] = spec.source_idea
        out["borrow_lifecycle_hit"] = first_hits | second_hits
        return out
    if spec.mode not in {"time_stop", "profit_cap_exit"}:
        raise ValueError(f"unknown lifecycle mode: {spec.mode}")

    market = market[["timestamp", "open", "high", "low", "close"]].copy()
    ts = market["timestamp"].to_numpy()
    close_by_ts = dict(zip(market["timestamp"], market["close"].astype(float)))
    for idx, row in out.iterrows():
        if float(row["notional"]) <= 1.0e-12:
            continue
        side = int(row["side"])
        if not side_selected(side, spec.side):
            continue
        entry_ts = pd.Timestamp(row["entry_timestamp_dt"])
        exit_ts = pd.Timestamp(row["exit_timestamp_dt"])
        cap_ts_want = entry_ts + pd.Timedelta(hours=float(spec.cap_hours))
        if cap_ts_want >= exit_ts:
            continue
        pos = int(np.searchsorted(ts, np.datetime64(cap_ts_want), side="left"))
        if pos >= len(market):
            continue
        cap_ts = pd.Timestamp(ts[pos])
        if cap_ts >= exit_ts:
            continue
        entry_price = close_by_ts.get(entry_ts)
        if entry_price is None:
            raise RuntimeError(f"entry timestamp not found in market: {entry_ts}")
        exit_price = float(market.loc[pos, "close"])
        raw_move = float(side) * (exit_price / float(entry_price) - 1.0)
        if spec.mode == "profit_cap_exit" and raw_move < float(spec.min_unrealized_price_move):
            continue
        old_cost = float(row["raw_exit_price_move"]) - float(row["net_per_notional"])
        net_per_notional = raw_move - old_cost
        window = market[(market["timestamp"] >= entry_ts) & (market["timestamp"] <= cap_ts)]
        if window.empty:
            raise RuntimeError(f"empty market window for {entry_ts}..{cap_ts}")
        if side > 0:
            mfe = float(window["high"].max() / float(entry_price) - 1.0)
            mae = float(window["low"].min() / float(entry_price) - 1.0)
        else:
            mfe = float(float(entry_price) / window["low"].min() - 1.0)
            mae = float(float(entry_price) / window["high"].max() - 1.0)
        hold_bars = int(round((cap_ts - entry_ts).total_seconds() / 300.0))
        reason = f"{spec.name}_exit"
        out.at[idx, "exit_i"] = int(row["entry_i"]) + hold_bars
        out.at[idx, "exit_timestamp"] = cap_ts.strftime("%Y-%m-%d %H:%M:%S")
        out.at[idx, "exit_timestamp_dt"] = cap_ts
        out.at[idx, "reason"] = reason
        out.at[idx, "raw_exit_price_move"] = raw_move
        out.at[idx, "mfe_price_move"] = mfe
        out.at[idx, "mae_price_move"] = mae
        out.at[idx, "net_per_notional"] = net_per_notional
        out.at[idx, "trade_return"] = net_per_notional * float(row["notional"])
        out.at[idx, "win"] = int(net_per_notional > 0.0)
        out.at[idx, "hold_hours"] = float((cap_ts - entry_ts).total_seconds() / 3600.0)
        out.at[idx, "borrow_lifecycle_hit"] = True
    return out


def active_ledger(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["notional"].astype(float) > 1.0e-12].copy()


def overlap_count(df: pd.DataFrame) -> int:
    active = active_ledger(df)
    if len(active) <= 1:
        return 0
    ordered = active.sort_values(["entry_i", "exit_i"]).reset_index(drop=True)
    prev_exit = -1
    overlaps = 0
    for _, row in ordered.iterrows():
        entry_i = int(row["entry_i"])
        exit_i = int(row["exit_i"])
        if entry_i <= prev_exit:
            overlaps += 1
        prev_exit = max(prev_exit, exit_i)
    return overlaps


def metrics(df: pd.DataFrame) -> dict[str, Any]:
    active = active_ledger(df)
    if active.empty:
        return {
            "pnl": 0.0,
            "mdd": 0.0,
            "trades": 0,
            "wr": 0.0,
            "max_hold_hours": 0.0,
            "hold_over_24h_count": 0,
            "avg_hold_hours": 0.0,
            "max_leverage": 0.0,
            "avg_notional": 0.0,
            "max_notional": 0.0,
            "skipped": int(len(df)),
            "entry_hits": 0,
            "lifecycle_hits": 0,
            "overlap_count": 0,
            "accounting_error_max_abs": 0.0,
            "notional_contract_error_max_abs": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "reason_counts": {},
            "source_counts": {},
        }
    returns = active["trade_return"].astype(float).to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    drawdown = curve / np.maximum(peak, 1.0e-12) - 1.0
    accounting_error = (
        active["trade_return"].astype(float) - active["net_per_notional"].astype(float) * active["notional"].astype(float)
    ).abs()
    notional_contract_error = (
        active["notional"].astype(float) - active["margin_fraction"].astype(float) * active["leverage"].astype(float)
    ).abs()
    hold_hours = active["hold_hours"].astype(float)
    entry_hits = df.get("borrow_entry_hit", pd.Series(False, index=df.index)).astype(bool)
    lifecycle_hits = df.get("borrow_lifecycle_hit", pd.Series(False, index=df.index)).astype(bool)
    return {
        "pnl": float((curve[-1] - 1.0) * 100.0),
        "mdd": float(np.min(drawdown) * 100.0),
        "trades": int(len(active)),
        "wr": float(active["win"].astype(float).mean()),
        "max_hold_hours": float(hold_hours.max()),
        "hold_over_24h_count": int((hold_hours > 24.0 + 1.0e-9).sum()),
        "avg_hold_hours": float(hold_hours.mean()),
        "max_leverage": float(active["leverage"].astype(float).max()),
        "avg_notional": float(active["notional"].astype(float).mean()),
        "max_notional": float(active["notional"].astype(float).max()),
        "skipped": int((df["notional"].astype(float) <= 1.0e-12).sum()),
        "entry_hits": int(entry_hits.sum()),
        "lifecycle_hits": int(lifecycle_hits.sum()),
        "overlap_count": int(overlap_count(df)),
        "accounting_error_max_abs": float(accounting_error.max()),
        "notional_contract_error_max_abs": float(notional_contract_error.max()),
        "long_trades": int((active["side"].astype(int) > 0).sum()),
        "short_trades": int((active["side"].astype(int) < 0).sum()),
        "reason_counts": {str(k): int(v) for k, v in active["reason"].value_counts().sort_index().to_dict().items()},
        "source_counts": {str(k): int(v) for k, v in active["source_alias"].value_counts().sort_index().to_dict().items()},
    }


def monthly_summary(df: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for month, group in df.groupby("entry_month"):
        row = metrics(group)
        row["month"] = str(month)
        rows.append(row)
    if not rows:
        return {"monthly_min_pnl": 0.0, "monthly_worst_mdd": 0.0, "monthly_positive_count": 0, "monthly_count": 0}
    return {
        "monthly_min_pnl": float(min(row["pnl"] for row in rows)),
        "monthly_worst_mdd": float(min(row["mdd"] for row in rows)),
        "monthly_positive_count": int(sum(float(row["pnl"]) > 0.0 for row in rows)),
        "monthly_count": int(len(rows)),
        "monthly_rows": rows,
    }


def flatten_metrics(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat[f"{prefix}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif isinstance(value, list):
            flat[f"{prefix}_{key}"] = json.dumps(value, ensure_ascii=False)
        else:
            flat[f"{prefix}_{key}"] = value
    return flat


def entry_rules() -> list[EntryRule]:
    return [
        EntryRule("none", "Omega4.6 baseline control", "ou_halflife", "le", -1.0, "all", 1.0),
        EntryRule(
            "ou_halflife_skip_le_0p005415348",
            "Omega4.6.1 duration-aware OU half-life gate",
            "ou_halflife",
            "le",
            0.005415348,
            "all",
            0.0,
        ),
        EntryRule(
            "short_rsi_skip_ge_56p656189",
            "Omega4.6 duration-risk robust selected rule",
            "rsi",
            "ge",
            56.656189,
            "short",
            0.0,
        ),
        EntryRule(
            "sig_trend_health_half_ge_0p500520",
            "Omega4.6 duration-risk validation champion softened to 50% exposure",
            "sig_trend_health",
            "ge",
            0.500520,
            "all",
            0.5,
        ),
        EntryRule(
            "h48qual_ai_flow_pressure_skip_le_0p117701",
            "Omega4.6 duration-risk h48qual short pressure veto",
            "ai_flow_pressure",
            "le",
            0.117701,
            "h48qual_S",
            0.0,
        ),
    ]


def cap_slug(value: float) -> str:
    return f"{int(round(float(value) * 100.0)):03d}"


def exposure_specs(notional_cap: float) -> list[ExposureSpec]:
    cap = cap_slug(notional_cap)
    return [
        ExposureSpec("none", "Omega4.6 baseline exposure control", "none", 1.0, 1.0, float(notional_cap)),
        ExposureSpec(
            "omega3_side_l090_s135_cap135",
            "Omega4.4 v18 borrowed Omega3 side exposure strict profile",
            "side_factor",
            0.90,
            1.35,
            1.35,
        ),
        ExposureSpec(
            "omega3_side_l090_s180_cap135",
            "Omega4.4 v18 borrowed Omega3 side exposure growth profile",
            "side_factor",
            0.90,
            1.80,
            1.35,
        ),
        ExposureSpec(
            "omega3_side_l095_s135_cap105",
            "Omega4.4 v18 borrowed conservative side exposure profile",
            "side_factor",
            0.95,
            1.35,
            1.05,
        ),
        ExposureSpec(
            f"short_bias_cap{cap}",
            "Omega3/Omega4.6 shared observation: short edge dominates, keep cap at Omega4.6",
            "side_factor",
            0.90,
            1.15,
            float(notional_cap),
        ),
        ExposureSpec(
            f"short_boost125_cap{cap}",
            "Notional gate removed from red-team pass: moderate short exposure expansion under leverage cap",
            "side_factor",
            0.90,
            1.25,
            float(notional_cap),
        ),
        ExposureSpec(
            f"short_boost140_cap{cap}",
            "Notional gate removed from red-team pass: aggressive short exposure expansion under leverage cap",
            "side_factor",
            0.90,
            1.40,
            float(notional_cap),
        ),
    ]


def lifecycle_specs() -> list[LifecycleSpec]:
    return [
        LifecycleSpec("none", "Omega4.6 no max-hold control", "none", "all", 0.0, 0.0),
        LifecycleSpec("time_stop_72h", "Omega1 horizon-router max-hold diagnostic", "time_stop", "all", 72.0, 0.0),
        LifecycleSpec("time_stop_120h", "Omega1 horizon-router max-hold diagnostic", "time_stop", "all", 120.0, 0.0),
        LifecycleSpec(
            "short_profit_cap_96h_3p5pct",
            "Omega4.4 short-partial/short-cap family simplified as hard profit exit",
            "profit_cap_exit",
            "short",
            96.0,
            0.035,
        ),
        LifecycleSpec(
            "short_profit_cap_146p667h_3p5pct",
            "Omega1.2.1 short_cap1760_min0.035 hard-exit transfer",
            "profit_cap_exit",
            "short",
            1760.0 * 5.0 / 60.0,
            0.035,
        ),
        LifecycleSpec(
            "short_profit_cap_96h_3p5pct_then_time_stop_120h",
            "Omega4.4 short-profit lifecycle plus Omega1 max-hold discipline",
            "profit_cap_then_time_stop",
            "short",
            96.0,
            0.035,
            120.0,
        ),
        LifecycleSpec(
            "short_profit_cap_146p667h_3p5pct_then_time_stop_120h",
            "Omega1.2.1 short_cap1760_min0.035 plus Omega1 max-hold discipline",
            "profit_cap_then_time_stop",
            "short",
            1760.0 * 5.0 / 60.0,
            0.035,
            120.0,
        ),
    ]


def score_row(row: dict[str, Any], baseline_val: dict[str, Any]) -> float:
    pnl = float(row["validation_pnl"])
    mdd = float(row["validation_mdd"])
    trades = float(row["validation_trades"])
    min_trades = float(baseline_val["trades"]) * 0.65
    trade_penalty = 3.0 * max(0.0, min_trades - trades)
    mdd_penalty = 12.0 * max(0.0, abs(mdd) - 20.0)
    overlap_penalty = 100.0 * float(row["validation_overlap_count"])
    accounting_penalty = 1000.0 * float(row["validation_accounting_error_max_abs"] > 1.0e-10)
    month_reward = 1.5 * float(row["validation_monthly_min_pnl"])
    hold_reward = 0.02 * max(0.0, float(baseline_val["max_hold_hours"]) - float(row["validation_max_hold_hours"]))
    return pnl + month_reward + hold_reward - trade_penalty - mdd_penalty - overlap_penalty - accounting_penalty


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-contract", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--notional-cap-override", type=float, default=None)
    parser.add_argument("--model-id", default="omega4_6_borrowed_version_upgrade_tests_20260630")
    args = parser.parse_args()

    runtime = read_json(resolve_path(args.runtime_contract))
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = resolve_path(runtime["source_report"]).parent
    train_csv, eval_csv = component_train_eval_paths(runtime)
    train_market = load_market(train_csv)
    eval_market = load_market(eval_csv)
    feature_cols = [col for col in safe_feature_columns(train_market) if col in set(eval_market.columns)]

    val_base = pd.read_csv(source_dir / "validation_scaled_trade_ledger.csv")
    oos_base = pd.read_csv(source_dir / "oos_scaled_trade_ledger.csv")
    val = join_entry_features(val_base, train_market, feature_cols)
    oos = join_entry_features(oos_base, eval_market, feature_cols)
    leverage_cap = float(runtime["leverage_cap"])
    runtime_notional_cap = float(runtime["notional_cap"])
    notional_cap = float(args.notional_cap_override) if args.notional_cap_override is not None else runtime_notional_cap
    if notional_cap <= 0.0:
        raise ValueError("--notional-cap-override must be positive")

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    baseline_val_metrics: dict[str, Any] | None = None
    for entry in entry_rules():
        for exposure in exposure_specs(notional_cap):
            for lifecycle in lifecycle_specs():
                variant = f"{entry.name}__{exposure.name}__{lifecycle.name}"
                val_work = apply_entry_rule(val, entry, leverage_cap, notional_cap)
                oos_work = apply_entry_rule(oos, entry, leverage_cap, notional_cap)
                val_work = apply_exposure(val_work, exposure, leverage_cap, notional_cap)
                oos_work = apply_exposure(oos_work, exposure, leverage_cap, notional_cap)
                val_work = apply_lifecycle(val_work, train_market, lifecycle)
                oos_work = apply_lifecycle(oos_work, eval_market, lifecycle)
                val_metrics = metrics(val_work)
                oos_metrics = metrics(oos_work)
                val_month = monthly_summary(val_work)
                row = {
                    "variant": variant,
                    "entry_rule": entry.name,
                    "entry_source_idea": entry.source_idea,
                    "exposure_spec": exposure.name,
                    "exposure_source_idea": exposure.source_idea,
                    "lifecycle_spec": lifecycle.name,
                    "lifecycle_source_idea": lifecycle.source_idea,
                    "validation_monthly_min_pnl": val_month["monthly_min_pnl"],
                    "validation_monthly_worst_mdd": val_month["monthly_worst_mdd"],
                    "validation_monthly_positive_count": val_month["monthly_positive_count"],
                    "validation_monthly_count": val_month["monthly_count"],
                    **flatten_metrics("validation", val_metrics),
                    **flatten_metrics("oos", oos_metrics),
                    "validation_gate_pass": bool(
                        abs(float(val_metrics["mdd"])) <= 20.0
                        and float(val_metrics["max_leverage"]) <= leverage_cap + 1.0e-9
                        and int(val_metrics["overlap_count"]) == 0
                        and float(val_metrics["accounting_error_max_abs"]) <= 1.0e-10
                        and float(val_metrics["notional_contract_error_max_abs"]) <= 1.0e-10
                    ),
                    "oos_gate_pass": bool(
                        abs(float(oos_metrics["mdd"])) <= 20.0
                        and float(oos_metrics["max_leverage"]) <= leverage_cap + 1.0e-9
                        and int(oos_metrics["overlap_count"]) == 0
                        and float(oos_metrics["accounting_error_max_abs"]) <= 1.0e-10
                        and float(oos_metrics["notional_contract_error_max_abs"]) <= 1.0e-10
                    ),
                }
                if variant == "none__none__none":
                    baseline_val_metrics = val_metrics
                if baseline_val_metrics is None:
                    raise RuntimeError("baseline variant must be evaluated first")
                row["selection_score"] = score_row(row, baseline_val_metrics)
                rows.append(row)
                ledgers[variant] = (val_work, oos_work)

    ranking = pd.DataFrame(rows)
    gated = ranking[ranking["validation_gate_pass"] & ranking["oos_gate_pass"]].copy()
    pool = gated if not gated.empty else ranking
    pool = pool[
        pool["validation_trades"].astype(float)
        >= float(baseline_val_metrics["trades"]) * 0.65
    ].copy()
    if pool.empty:
        pool = gated if not gated.empty else ranking
    selected = pool.sort_values(
        ["selection_score", "validation_monthly_min_pnl", "validation_pnl"],
        ascending=[False, False, False],
    ).iloc[0]
    selected_variant = str(selected["variant"])

    ranking = ranking.sort_values(
        ["validation_gate_pass", "oos_gate_pass", "selection_score", "validation_pnl"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    ranking_path = out_dir / "borrowed_upgrade_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    top_variants = ranking.head(10)["variant"].astype(str).tolist()
    artifacts: dict[str, str] = {"ranking": str(ranking_path), "report": str(out_dir / "report.json")}
    for variant in [selected_variant, *top_variants]:
        if variant not in ledgers:
            continue
        safe = variant.replace(".", "p").replace("/", "_")
        val_path = out_dir / f"validation_{safe}_ledger.csv"
        oos_path = out_dir / f"oos_{safe}_ledger.csv"
        ledgers[variant][0].to_csv(val_path, index=False)
        ledgers[variant][1].to_csv(oos_path, index=False)
        artifacts[f"validation_ledger__{safe}"] = str(val_path)
        artifacts[f"oos_ledger__{safe}"] = str(oos_path)

    baseline_row = ranking[ranking["variant"].eq("none__none__none")].iloc[0].to_dict()
    report = {
        "model_id": str(args.model_id),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": runtime["model_id"],
        "base_runtime_contract": str(resolve_path(args.runtime_contract)),
        "source_report": runtime["source_report"],
        "runtime_notional_cap": runtime_notional_cap,
        "tested_notional_cap": notional_cap,
        "notional_gate_removed_from_redteam_pass": True,
        "selection_scope": "validation_only; OOS readout only",
        "test_scope": "ledger-level borrowed Omega ideas; no parent retrain and no runtime promotion",
        "borrowed_idea_families": {
            "entry_rules": [asdict(x) for x in entry_rules()],
            "exposure_specs": [asdict(x) for x in exposure_specs(notional_cap)],
            "lifecycle_specs": [asdict(x) for x in lifecycle_specs()],
        },
        "feature_contract": {
            "entry_time_features_only_for_entry_rules": True,
            "feature_count": len(feature_cols),
            "excluded_feature_tokens": ["label", "target", "future", "fwd", "pnl", "win", "exit", "barrier", "zigzag"],
            "train_csv": str(train_csv),
            "eval_csv": str(eval_csv),
        },
        "variants_evaluated": int(len(ranking)),
        "baseline_row": baseline_row,
        "selected_variant": selected.to_dict(),
        "top10_variants": ranking.head(10).to_dict(orient="records"),
        "artifacts": artifacts,
        "redteam_note": (
            "Diagnostic only. Promotion requires a frozen rule contract, live feature parity check, "
            "native replay for lifecycle overlays, and Omega artifact integrity audit."
        ),
    }
    write_json(out_dir / "report.json", report)
    print(
        json.dumps(
            {
                "report": str(out_dir / "report.json"),
                "ranking": str(ranking_path),
                "selected_variant": selected_variant,
                "selected_validation": {
                    "pnl": selected["validation_pnl"],
                    "mdd": selected["validation_mdd"],
                    "trades": selected["validation_trades"],
                    "max_hold_hours": selected["validation_max_hold_hours"],
                },
                "selected_oos": {
                    "pnl": selected["oos_pnl"],
                    "mdd": selected["oos_mdd"],
                    "trades": selected["oos_trades"],
                    "max_hold_hours": selected["oos_max_hold_hours"],
                },
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
