#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_current_rank1_baseline_safety_2026 import _decision_audit  # noqa: E402
from scripts.run_clean_scope_muzero_az_reaudit_2026 import realistic_ledger_replay  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import (  # noqa: E402
    _base_frame,
    _compact,
    backtest_no_limit_exit,
)


DEFAULT_POLICY = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_v4_clean_train_to_2025_10.pkl"
DEFAULT_EXIT = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_no_limit_exit_clean_train_to_2025_10.pkl"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/clean_scope_muzero_az_reaudit_2026.json"
DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_exposure_sweep_2026.json"
DEFAULT_GRID_CSV = ROOT / "data/ensemble/reports/clean_base_exposure_sweep_2026.csv"

BASE_REFERENCE = {
    "pnl": 177.3298088749005,
    "mdd": -17.75966486035323,
    "trades": 363,
    "trades_per_day": 6.1875,
    "cost_2x_pnl": 92.25487780535948,
    "cost_3x_pnl": -7.969394502459748,
}


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _split_train_validation(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required for clean validation split")
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    split = pd.Timestamp(split_date)
    train = df.loc[ts < split].reset_index(drop=True)
    val = df.loc[ts >= split].reset_index(drop=True)
    return train, val


def _range(df: pd.DataFrame) -> list[str]:
    if "timestamp" not in df.columns or df.empty:
        return ["", ""]
    return [str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1])]


def _scale_leverage(
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    leverage_mult: float,
    leverage_cap: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    feat, dec, close, fill = precomputed
    out = dec.copy()
    active = (
        (pd.to_numeric(out.get("action", 0), errors="coerce").fillna(0).astype(int) != 0)
        & (pd.to_numeric(out.get("side", 0), errors="coerce").fillna(0).astype(int) != 0)
        & (pd.to_numeric(out.get("notional_exposure", 0.0), errors="coerce").fillna(0.0) > 0.0)
    )
    lev = pd.to_numeric(out.loc[active, "leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    lev = np.clip(lev * float(leverage_mult), 1.0, float(leverage_cap))
    out.loc[active, "leverage"] = lev
    notional = pd.to_numeric(out.loc[active, "notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out.loc[active, "position_fraction"] = notional / np.maximum(lev, 1e-12)
    return feat, out, close, fill


def _run(
    df: pd.DataFrame,
    policy: dict[str, Any],
    exit_model: Any,
    entry_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    return _compact(
        backtest_no_limit_exit(
            df,
            policy,
            exit_model,
            entry_config=entry_cfg,
            risk_config=risk_cfg,
            exit_threshold=float(exit_cfg["exit_threshold"]),
            min_exit_age=int(exit_cfg["min_exit_age"]),
            fee=float(fee),
            slip=float(slip),
            precomputed=precomputed,
        )
    )


def _score(metrics: dict[str, Any]) -> float:
    pnl = float(metrics.get("pnl", -1e9))
    mdd = float(metrics.get("mdd", -1e9))
    tpd = float(metrics.get("trades_per_day", 0.0))
    coverage_penalty = max(0.0, 5.5 - tpd) * 20.0
    return pnl + 4.0 * mdd - coverage_penalty


def _candidate_name(notional_mult: float, max_notional: float, leverage_mult: float, leverage_cap: float) -> str:
    return f"nm{notional_mult:.2f}_maxn{max_notional:.1f}_levm{leverage_mult:.2f}_levcap{leverage_cap:.1f}"


def _select_validation_candidates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    selected["base_reference"] = min(
        rows,
        key=lambda r: (
            abs(float(r["notional_mult"]) - 1.5)
            + abs(float(r["max_notional"]) - 3.6)
            + abs(float(r["leverage_mult"]) - 1.0)
        ),
    )
    selected["validation_max_pnl"] = max(rows, key=lambda r: float(r["validation"]["pnl"]))
    selected["validation_balanced_score"] = max(rows, key=lambda r: float(r["validation_score"]))
    constrained = [
        r
        for r in rows
        if float(r["validation"]["mdd"]) >= -25.0
        and float(r["validation"]["trades_per_day"]) >= 5.5
        and float(r["validation"]["pnl"]) >= BASE_REFERENCE["pnl"]
    ]
    if constrained:
        selected["validation_redteam_constrained"] = max(constrained, key=lambda r: float(r["validation"]["pnl"]))
    else:
        selected["validation_redteam_constrained"] = max(rows, key=lambda r: float(r["validation_score"]))
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validation-selected exposure sweep for the clean base policy.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--split-date", default="2025-11-01")
    p.add_argument("--notional-mults", default="1.5,2.0,2.5,3.0,3.5,4.0,5.0")
    p.add_argument("--max-notionals", default="3.6,5.0,7.5,10.0,12.5")
    p.add_argument("--leverage-mults", default="1.0,1.5,2.0,3.0")
    p.add_argument("--leverage-cap", type=float, default=10.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-csv-out", type=Path, default=DEFAULT_GRID_CSV)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy = joblib.load(args.policy)
    exit_payload = joblib.load(args.exit_model)
    exit_model = exit_payload["model"] if isinstance(exit_payload, dict) and "model" in exit_payload else exit_payload
    audit = json.load(args.audit_report.open("r", encoding="utf-8"))
    controls = audit["control_selection"]["selected"]
    base_entry_cfg = dict(controls["entry_config"])
    base_risk_cfg = dict(controls["risk_config"])
    exit_cfg = dict(controls["exit_config"])

    train_full = _read(args.train_csv)
    _train_df, val_df = _split_train_validation(train_full, args.split_date)
    eval_df = _read(args.eval_csv)

    notional_mults = _parse_floats(args.notional_mults)
    max_notionals = _parse_floats(args.max_notionals)
    leverage_mults = _parse_floats(args.leverage_mults)

    cache: dict[tuple[str, float, float], tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]] = {}

    def precompute(split: str, df: pd.DataFrame, entry_cfg: dict[str, Any], leverage_mult: float) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        key = (split, float(entry_cfg["notional_mult"]), float(entry_cfg["max_notional"]))
        if key not in cache:
            cache[key] = _base_frame(df, policy, entry_cfg)
        return _scale_leverage(cache[key], leverage_mult=leverage_mult, leverage_cap=float(args.leverage_cap))

    rows: list[dict[str, Any]] = []
    for notional_mult in notional_mults:
        for max_notional in max_notionals:
            for leverage_mult in leverage_mults:
                entry_cfg = dict(base_entry_cfg)
                risk_cfg = dict(base_risk_cfg)
                entry_cfg["notional_mult"] = float(notional_mult)
                entry_cfg["max_notional"] = float(max_notional)
                risk_cfg["max_notional"] = float(max_notional)
                pre_val = precompute("validation", val_df, entry_cfg, leverage_mult)
                val = _run(
                    val_df,
                    policy,
                    exit_model,
                    entry_cfg,
                    risk_cfg,
                    exit_cfg,
                    pre_val,
                    fee=float(args.fee),
                    slip=float(args.slip),
                )
                rows.append(
                    {
                        "name": _candidate_name(notional_mult, max_notional, leverage_mult, float(args.leverage_cap)),
                        "notional_mult": float(notional_mult),
                        "max_notional": float(max_notional),
                        "leverage_mult": float(leverage_mult),
                        "leverage_cap": float(args.leverage_cap),
                        "entry_config": entry_cfg,
                        "risk_config": risk_cfg,
                        "exit_config": exit_cfg,
                        "validation": val,
                        "validation_score": _score(val),
                    }
                )

    selected = _select_validation_candidates(rows)
    selected_names = {v["name"] for v in selected.values()}
    top_validation = sorted(rows, key=lambda r: float(r["validation"]["pnl"]), reverse=True)[:20]
    top_balanced = sorted(rows, key=lambda r: float(r["validation_score"]), reverse=True)[:20]
    eval_rows_by_name = {r["name"]: r for r in [*selected.values(), *top_validation, *top_balanced]}

    eval_results: dict[str, Any] = {}
    for name, row in eval_rows_by_name.items():
        entry_cfg = dict(row["entry_config"])
        risk_cfg = dict(row["risk_config"])
        leverage_mult = float(row["leverage_mult"])
        pre_eval = precompute("eval", eval_df, entry_cfg, leverage_mult)
        cost_stress = {
            f"cost_{mult:g}x": _run(
                eval_df,
                policy,
                exit_model,
                entry_cfg,
                risk_cfg,
                exit_cfg,
                pre_eval,
                fee=float(args.fee) * mult,
                slip=float(args.slip) * mult,
            )
            for mult in (1.0, 2.0, 3.0)
        }
        _feat, dec, _close, _fill = pre_eval
        invariant = _decision_audit(
            dec,
            max_notional=float(risk_cfg.get("max_notional", entry_cfg.get("max_notional", 3.6))),
            leverage_cap=float(args.leverage_cap),
        )
        eval_results[name] = {
            "candidate": {k: row[k] for k in ("name", "notional_mult", "max_notional", "leverage_mult", "leverage_cap")},
            "validation": row["validation"],
            "validation_score": row["validation_score"],
            "oos": cost_stress["cost_1x"],
            "cost_stress": cost_stress,
            "decision_invariant_audit": invariant,
            "selected_by": [label for label, selected_row in selected.items() if selected_row["name"] == name],
        }

    selected_eval = {label: eval_results[row["name"]] for label, row in selected.items()}
    best_eval_diagnostic = max(eval_results.values(), key=lambda r: float(r["oos"]["pnl"]))
    realistic_candidates = {
        label: result
        for label, result in selected_eval.items()
        if label in {"validation_max_pnl", "validation_balanced_score", "validation_redteam_constrained"}
    }
    for label, result in realistic_candidates.items():
        cand = result["candidate"]
        entry_cfg = dict(base_entry_cfg)
        risk_cfg = dict(base_risk_cfg)
        entry_cfg["notional_mult"] = float(cand["notional_mult"])
        entry_cfg["max_notional"] = float(cand["max_notional"])
        risk_cfg["max_notional"] = float(cand["max_notional"])
        pre_eval = precompute("eval", eval_df, entry_cfg, float(cand["leverage_mult"]))
        realistic = realistic_ledger_replay(
            eval_df,
            exit_model,
            risk_cfg,
            exit_cfg,
            pre_eval,
            fee=float(args.fee),
            slip=float(args.slip),
            funding_mult=1.0,
            impact_per_notional=0.00008,
            partial_fill_ratio=0.96,
            maintenance_margin=0.006,
            liquidation_fee=0.002,
        )
        result["realistic_replay"] = realistic["eval"]

    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "notional_mult",
                "max_notional",
                "leverage_mult",
                "leverage_cap",
                "val_pnl",
                "val_mdd",
                "val_trades",
                "val_trades_per_day",
                "val_avg_notional",
                "val_avg_leverage",
                "validation_score",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda r: float(r["validation"]["pnl"]), reverse=True):
            v = row["validation"]
            writer.writerow(
                {
                    "name": row["name"],
                    "notional_mult": row["notional_mult"],
                    "max_notional": row["max_notional"],
                    "leverage_mult": row["leverage_mult"],
                    "leverage_cap": row["leverage_cap"],
                    "val_pnl": v["pnl"],
                    "val_mdd": v["mdd"],
                    "val_trades": v["trades"],
                    "val_trades_per_day": v["trades_per_day"],
                    "val_avg_notional": v["avg_notional"],
                    "val_avg_leverage": v["avg_leverage"],
                    "validation_score": row["validation_score"],
                }
            )

    report = {
        "type": "clean_base_exposure_sweep_2026",
        "note": "Clean base policy is frozen; validation selects exposure-only variants before one-shot 2026 OOS replay. Leverage affects exit-model state and margin metadata; canonical PnL is primarily driven by notional_exposure.",
        "base_reference": BASE_REFERENCE,
        "data": {
            "validation_range": _range(val_df),
            "validation_rows": int(len(val_df)),
            "eval_range": _range(eval_df),
            "eval_rows": int(len(eval_df)),
        },
        "grid": {
            "notional_mults": notional_mults,
            "max_notionals": max_notionals,
            "leverage_mults": leverage_mults,
            "leverage_cap": float(args.leverage_cap),
            "candidates": int(len(rows)),
            "csv": str(args.grid_csv_out),
        },
        "selected_by_validation": {label: row["name"] for label, row in selected.items()},
        "selected_eval": selected_eval,
        "top_validation_pnl": [
            {"name": r["name"], "validation": r["validation"], "validation_score": r["validation_score"]}
            for r in top_validation[:10]
        ],
        "top_balanced_validation": [
            {"name": r["name"], "validation": r["validation"], "validation_score": r["validation_score"]}
            for r in top_balanced[:10]
        ],
        "best_eval_diagnostic_not_promotable": best_eval_diagnostic,
        "promotion_rules": {
            "promotable_selection_source": "validation only",
            "minimum_oos_pnl": BASE_REFERENCE["pnl"],
            "minimum_oos_mdd": BASE_REFERENCE["mdd"],
            "minimum_trades_per_day": 5.5,
            "cost_1x_2x_required_positive": True,
            "cost_3x_must_be_reported": True,
            "invariant_audit_required": True,
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "grid_csv": str(args.grid_csv_out), "selected": report["selected_by_validation"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

