#!/usr/bin/env python3
"""Validation-only duration-aware entry risk layer for Omega 4.6.

The layer changes entry-time exposure only. It does not alter exits and does
not use OOS for selection.
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
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_duration_aware_risk_layer_20260630"


@dataclass(frozen=True)
class Rule:
    name: str
    feature: str
    op: str
    threshold: float
    scope: str
    hit_scale: float


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def component_train_eval_paths(runtime: dict[str, Any]) -> tuple[Path, Path]:
    risk_report = read_json(resolve_path(runtime["components"]["h48qual"]["report"]))
    risk_model = risk_report["risk_model"]
    return resolve_path(risk_model["train_csv"]), resolve_path(risk_model["eval_csv"])


def source_dir(runtime: dict[str, Any]) -> Path:
    return resolve_path(runtime["source_report"]).parent


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


def load_market_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    return df.sort_values("timestamp").reset_index(drop=True)


def join_entry_features(ledger: pd.DataFrame, market: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    work = ledger.copy()
    work["entry_timestamp_dt"] = pd.to_datetime(work["entry_timestamp"], errors="raise")
    feat = market[["timestamp", *feature_cols]].rename(columns={"timestamp": "entry_timestamp_dt"})
    out = work.merge(feat, on="entry_timestamp_dt", how="left", validate="many_to_one")
    if out[feature_cols].isna().any().any():
        missing = out.loc[out[feature_cols].isna().any(axis=1), "entry_timestamp"].head(5).tolist()
        raise RuntimeError(f"missing entry features for timestamps: {missing}")
    out["hold_hours"] = (
        pd.to_datetime(out["exit_timestamp"], errors="raise") - out["entry_timestamp_dt"]
    ).dt.total_seconds() / 3600.0
    out["entry_month"] = out["entry_timestamp_dt"].dt.to_period("M").astype(str)
    out["entry_hour"] = out["entry_timestamp_dt"].dt.hour.astype(float)
    out["entry_dow"] = out["entry_timestamp_dt"].dt.dayofweek.astype(float)
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


def condition_mask(df: pd.DataFrame, rule: Rule) -> np.ndarray:
    vals = pd.to_numeric(df[rule.feature], errors="coerce").to_numpy(dtype=float)
    if rule.op == "ge":
        cond = vals >= float(rule.threshold)
    elif rule.op == "le":
        cond = vals <= float(rule.threshold)
    else:
        raise ValueError(f"unknown op: {rule.op}")
    return cond & scope_mask(df, rule.scope) & np.isfinite(vals)


def apply_rule(df: pd.DataFrame, rule: Rule, *, leverage_cap: float, notional_cap: float) -> pd.DataFrame:
    out = df.copy()
    mask = condition_mask(out, rule)
    scale = np.ones(len(out), dtype=float)
    scale[mask] = float(rule.hit_scale)
    old_notional = out["notional"].astype(float).to_numpy()
    margin = out["margin_fraction"].astype(float).to_numpy()
    target_notional = old_notional * scale
    target_notional = np.minimum(target_notional, float(notional_cap))
    target_notional = np.minimum(target_notional, margin * float(leverage_cap))
    leverage = np.divide(
        target_notional,
        margin,
        out=np.zeros_like(target_notional, dtype=float),
        where=np.abs(margin) > 1.0e-12,
    )
    out["duration_risk_rule"] = rule.name
    out["duration_risk_hit"] = mask
    out["duration_risk_scale"] = scale
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
    out["trade_return"] = out["net_per_notional"].astype(float) * out["notional"].astype(float)
    out["duration_risk_skipped"] = out["notional"].astype(float) <= 1.0e-12
    return out


def active_ledger(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["notional"].astype(float) > 1.0e-12].copy()


def metrics(df: pd.DataFrame) -> dict[str, Any]:
    active = active_ledger(df)
    if active.empty:
        return {
            "pnl": 0.0,
            "mdd": 0.0,
            "trades": 0,
            "wr": 0.0,
            "avg_hold_hours": 0.0,
            "max_hold_hours": 0.0,
            "hold_over_24h_count": 0,
            "max_leverage": 0.0,
            "max_notional": 0.0,
            "avg_notional": 0.0,
            "skipped": int(len(df)),
            "hit_count": int(df.get("duration_risk_hit", pd.Series(False, index=df.index)).astype(bool).sum()),
        }
    returns = active["trade_return"].astype(float).to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    drawdown = curve / np.maximum(peak, 1.0e-12) - 1.0
    hold_hours = active["hold_hours"].astype(float)
    hit_col = df.get("duration_risk_hit", pd.Series(False, index=df.index))
    return {
        "pnl": float((curve[-1] - 1.0) * 100.0),
        "mdd": float(np.min(drawdown) * 100.0),
        "trades": int(len(active)),
        "wr": float(active["win"].astype(float).mean()),
        "avg_hold_hours": float(hold_hours.mean()),
        "max_hold_hours": float(hold_hours.max()),
        "hold_over_24h_count": int((hold_hours > 24.0 + 1.0e-9).sum()),
        "max_leverage": float(active["leverage"].astype(float).max()),
        "max_notional": float(active["notional"].astype(float).max()),
        "avg_notional": float(active["notional"].astype(float).mean()),
        "skipped": int(df["duration_risk_skipped"].astype(bool).sum()) if "duration_risk_skipped" in df.columns else 0,
        "hit_count": int(hit_col.astype(bool).sum()),
    }


def selection_score(val: dict[str, Any], baseline: dict[str, Any]) -> float:
    pnl = float(val["pnl"])
    mdd_penalty = 12.0 * max(0.0, abs(float(val["mdd"])) - 20.0)
    trade_penalty = 0.8 * max(0.0, float(baseline["trades"]) * 0.65 - float(val["trades"]))
    pnl_loss = max(0.0, float(baseline["pnl"]) - pnl)
    avg_hold_gain = max(0.0, float(baseline["avg_hold_hours"]) - float(val["avg_hold_hours"]))
    max_hold_gain = max(0.0, float(baseline["max_hold_hours"]) - float(val["max_hold_hours"]))
    return pnl - 0.10 * pnl_loss + 0.05 * avg_hold_gain + 0.01 * max_hold_gain - mdd_penalty - trade_penalty


def monthly_summary(df: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for month, group in df.groupby("entry_month"):
        row = metrics(group)
        row["month"] = str(month)
        rows.append(row)
    if not rows:
        return {
            "monthly_min_pnl": 0.0,
            "monthly_worst_mdd": 0.0,
            "monthly_positive_count": 0,
            "monthly_count": 0,
            "monthly_rows": [],
        }
    pnls = [float(row["pnl"]) for row in rows]
    mdds = [float(row["mdd"]) for row in rows]
    return {
        "monthly_min_pnl": float(min(pnls)),
        "monthly_worst_mdd": float(min(mdds)),
        "monthly_positive_count": int(sum(p > 0.0 for p in pnls)),
        "monthly_count": int(len(rows)),
        "monthly_rows": rows,
    }


def robust_selection_score(val: dict[str, Any], month: dict[str, Any], baseline: dict[str, Any]) -> float:
    trade_penalty = 2.0 * max(0.0, float(baseline["trades"]) * 0.65 - float(val["trades"]))
    month_mdd_penalty = 10.0 * max(0.0, abs(float(month["monthly_worst_mdd"])) - 20.0)
    max_hold_gain = max(0.0, float(baseline["max_hold_hours"]) - float(val["max_hold_hours"]))
    return (
        2.0 * float(month["monthly_min_pnl"])
        + 0.12 * float(val["pnl"])
        + 0.05 * max_hold_gain
        - month_mdd_penalty
        - trade_penalty
    )


def duration_priority_score(val: dict[str, Any], month: dict[str, Any], baseline: dict[str, Any]) -> float:
    max_hold_gain = max(0.0, float(baseline["max_hold_hours"]) - float(val["max_hold_hours"]))
    avg_hold_gain = float(baseline["avg_hold_hours"]) - float(val["avg_hold_hours"])
    return (
        2.0 * float(month["monthly_min_pnl"])
        + 0.10 * float(val["pnl"])
        + 0.10 * max_hold_gain
        + 0.03 * avg_hold_gain
    )


def candidate_rules(val_df: pd.DataFrame, feature_cols: list[str]) -> list[Rule]:
    scopes = ["all", "long", "short", "h48qual_L", "h48qual_S", "zig075_L", "zig075_S"]
    scales = [0.0, 0.50, 0.75, 1.25, 1.50]
    rules: list[Rule] = [Rule("baseline", "entry_hour", "ge", -1.0, "all", 1.0)]
    search_cols = [*feature_cols, "entry_hour", "entry_dow"]
    for feature in search_cols:
        vals_all = pd.to_numeric(val_df[feature], errors="coerce")
        if vals_all.nunique(dropna=True) < 4:
            continue
        for scope in scopes:
            smask = scope_mask(val_df, scope)
            vals = vals_all[smask]
            if vals.notna().sum() < 4:
                continue
            thresholds = sorted(set(float(x) for x in vals.quantile([0.25, 0.5, 0.75]).dropna()))
            for threshold in thresholds:
                for op in ("ge", "le"):
                    base = Rule("tmp", feature, op, threshold, scope, 1.0)
                    hits = int(condition_mask(val_df, base).sum())
                    if hits < 2 or hits > len(val_df) - 2:
                        continue
                    for scale in scales:
                        name = f"{scope}_{feature}_{op}_{threshold:.6g}_x{scale:.2f}"
                        rules.append(Rule(name, feature, op, threshold, scope, scale))
    return rules


def rank_feature_columns(val_df: pd.DataFrame, feature_cols: list[str], limit: int) -> list[str]:
    if limit <= 0 or len(feature_cols) <= limit:
        return feature_cols
    scores: list[tuple[float, str]] = []
    y_ret = pd.to_numeric(val_df["trade_return"], errors="coerce")
    y_hold = pd.to_numeric(val_df["hold_hours"], errors="coerce")
    for col in feature_cols:
        x = pd.to_numeric(val_df[col], errors="coerce")
        if x.nunique(dropna=True) < 4:
            continue
        ret_corr = abs(float(x.corr(y_ret, method="spearman"))) if x.notna().sum() >= 4 else 0.0
        hold_corr = abs(float(x.corr(y_hold, method="spearman"))) if x.notna().sum() >= 4 else 0.0
        score = np.nan_to_num(ret_corr, nan=0.0) + 0.5 * np.nan_to_num(hold_corr, nan=0.0)
        scores.append((float(score), col))
    scores.sort(reverse=True)
    return [col for _, col in scores[:limit]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-contract", type=Path, default=DEFAULT_RUNTIME)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-rules", type=int, default=0, help="0 means evaluate all generated rules")
    ap.add_argument("--feature-limit", type=int, default=32)
    args = ap.parse_args()

    runtime = read_json(resolve_path(args.runtime_contract))
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv, eval_csv = component_train_eval_paths(runtime)
    train_market = load_market_features(train_csv)
    eval_market = load_market_features(eval_csv)
    feature_cols = [col for col in safe_feature_columns(train_market) if col in set(eval_market.columns)]
    src = source_dir(runtime)
    val_ledger = pd.read_csv(src / "validation_scaled_trade_ledger.csv")
    oos_ledger = pd.read_csv(src / "oos_scaled_trade_ledger.csv")
    val_df = join_entry_features(val_ledger, train_market, feature_cols)
    oos_df = join_entry_features(oos_ledger, eval_market, feature_cols)
    feature_cols = rank_feature_columns(val_df, feature_cols, int(args.feature_limit))

    leverage_cap = float(runtime["leverage_cap"])
    notional_cap = float(runtime["notional_cap"])
    rules = candidate_rules(val_df, feature_cols)
    if args.max_rules and args.max_rules > 0:
        rules = rules[: args.max_rules]

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    baseline_metrics: dict[str, Any] | None = None
    for rule in rules:
        val_scaled = apply_rule(val_df, rule, leverage_cap=leverage_cap, notional_cap=notional_cap)
        oos_scaled = apply_rule(oos_df, rule, leverage_cap=leverage_cap, notional_cap=notional_cap)
        val = metrics(val_scaled)
        oos = metrics(oos_scaled)
        val_month = monthly_summary(val_scaled)
        if rule.name == "baseline":
            baseline_metrics = val
        if baseline_metrics is None:
            raise RuntimeError("baseline rule must be first")
        row = {
            **{f"rule_{k}": v for k, v in asdict(rule).items()},
            "selection_score": selection_score(val, baseline_metrics),
            "robust_selection_score": robust_selection_score(val, val_month, baseline_metrics),
            "duration_priority_score": duration_priority_score(val, val_month, baseline_metrics),
            "validation_gate_pass": bool(
                float(val["mdd"]) >= -20.0
                and int(val["trades"]) >= int(max(1, np.floor(float(baseline_metrics["trades"]) * 0.65)))
                and float(val["max_leverage"]) <= leverage_cap + 1.0e-9
                and float(val["max_notional"]) <= notional_cap + 1.0e-9
            ),
            "validation_monthly_min_pnl": val_month["monthly_min_pnl"],
            "validation_monthly_worst_mdd": val_month["monthly_worst_mdd"],
            "validation_monthly_positive_count": val_month["monthly_positive_count"],
            "validation_monthly_count": val_month["monthly_count"],
            **{f"validation_{k}": v for k, v in val.items()},
            **{f"oos_{k}": v for k, v in oos.items()},
        }
        rows.append(row)
        ledgers[rule.name] = (val_scaled, oos_scaled)

    ranking = pd.DataFrame(rows)
    gated = ranking[ranking["validation_gate_pass"]].copy()
    if gated.empty:
        selected = ranking.sort_values(["selection_score"], ascending=False).iloc[0]
        robust_selected = selected
        duration_selected = selected
    else:
        selected = gated.sort_values(["selection_score", "validation_pnl"], ascending=[False, False]).iloc[0]
        robust_selected = gated.sort_values(
            ["robust_selection_score", "validation_monthly_min_pnl", "validation_pnl"],
            ascending=[False, False, False],
        ).iloc[0]
        duration_pool = gated[
            (gated["validation_pnl"] > float(baseline_metrics["pnl"]))
            & (gated["validation_max_hold_hours"] <= float(baseline_metrics["max_hold_hours"]) * 0.75)
        ]
        duration_selected = (
            duration_pool.sort_values(
                ["duration_priority_score", "validation_monthly_min_pnl", "validation_pnl"],
                ascending=[False, False, False],
            ).iloc[0]
            if not duration_pool.empty
            else robust_selected
        )
    selected_name = str(selected["rule_name"])
    robust_selected_name = str(robust_selected["rule_name"])
    duration_selected_name = str(duration_selected["rule_name"])

    ranking = ranking.sort_values(
        ["validation_gate_pass", "selection_score", "validation_pnl"],
        ascending=[False, False, False],
    )
    ranking_path = out_dir / "duration_risk_rule_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    best_val, best_oos = ledgers[selected_name]
    best_val.to_csv(out_dir / "validation_selected_duration_risk_ledger.csv", index=False)
    best_oos.to_csv(out_dir / "oos_selected_duration_risk_ledger.csv", index=False)
    robust_val, robust_oos = ledgers[robust_selected_name]
    robust_val.to_csv(out_dir / "validation_robust_duration_risk_ledger.csv", index=False)
    robust_oos.to_csv(out_dir / "oos_robust_duration_risk_ledger.csv", index=False)
    duration_val, duration_oos = ledgers[duration_selected_name]
    duration_val.to_csv(out_dir / "validation_duration_priority_risk_ledger.csv", index=False)
    duration_oos.to_csv(out_dir / "oos_duration_priority_risk_ledger.csv", index=False)

    baseline_row = ranking[ranking["rule_name"] == "baseline"].iloc[0].to_dict()
    report = {
        "model_id": "omega4_6_duration_aware_risk_layer_20260630",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_runtime_contract": str(resolve_path(args.runtime_contract)),
        "source_report": runtime["source_report"],
        "components": runtime["components"],
        "selection_scope": "validation_only; OOS is readout only",
        "feature_contract": {
            "entry_time_features_only": True,
            "feature_count": len(feature_cols),
            "feature_limit": int(args.feature_limit),
            "selected_features": feature_cols,
            "excluded_feature_tokens": ["label", "target", "future", "fwd", "pnl", "win", "exit", "barrier", "zigzag"],
            "train_csv": str(train_csv),
            "eval_csv": str(eval_csv),
        },
        "rules_evaluated": int(len(rules)),
        "selected_rule": selected.to_dict(),
        "robust_selected_rule": robust_selected.to_dict(),
        "duration_priority_selected_rule": duration_selected.to_dict(),
        "baseline_row": baseline_row,
        "promotion_recommendation": (
            "candidate_duration_priority_rule_requires_redteam"
            if duration_selected_name != "baseline"
            and float(duration_selected["validation_pnl"]) > float(baseline_row["validation_pnl"])
            else "do_not_promote_baseline_wins"
        ),
        "artifacts": {
            "ranking": str(ranking_path),
            "validation_selected_duration_risk_ledger": str(out_dir / "validation_selected_duration_risk_ledger.csv"),
            "oos_selected_duration_risk_ledger": str(out_dir / "oos_selected_duration_risk_ledger.csv"),
            "validation_robust_duration_risk_ledger": str(out_dir / "validation_robust_duration_risk_ledger.csv"),
            "oos_robust_duration_risk_ledger": str(out_dir / "oos_robust_duration_risk_ledger.csv"),
            "validation_duration_priority_risk_ledger": str(out_dir / "validation_duration_priority_risk_ledger.csv"),
            "oos_duration_priority_risk_ledger": str(out_dir / "oos_duration_priority_risk_ledger.csv"),
        },
        "redteam_note": "Diagnostic risk-layer search. Promotion requires a frozen rule contract and Omega artifact integrity audit.",
    }
    write_json(out_dir / "report.json", report)
    print(
        json.dumps(
            {
                "report": str(out_dir / "report.json"),
                "selected_rule": selected_name,
                "robust_selected_rule": robust_selected_name,
                "duration_priority_selected_rule": duration_selected_name,
                "selected": selected.to_dict(),
                "robust_selected": robust_selected.to_dict(),
                "duration_priority_selected": duration_selected.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
