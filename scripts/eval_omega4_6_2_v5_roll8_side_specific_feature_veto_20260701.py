#!/usr/bin/env python3
"""Single-feature short-entry veto for Omega 4.6.2 v5 roll8 PnL-tilt branch."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_HELPER_PATH = ROOT / "scripts/audit_omega4_6_2_loss_cluster_governor_redteam_20260701.py"
PNL_TILT_PATH = ROOT / "scripts/eval_omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll8_side_specific_feature_veto_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll8_side_specific_pnl_tilt_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll8_side_specific_feature_veto_20260701.md"
REFERENCE_CAP = 4.20
EPS = 1.0e-12
MAX_VALIDATION_SHORT_VETO_FRACTION = 0.25
QUANTILES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.75, 0.80, 0.85, 0.90, 0.95)
LOOKAHEAD_EXCLUDE_RE = re.compile(
    r"(timestamp|exit_|raw_exit|mfe|mae|trade_return|net_per_notional|log_return|win$|"
    r"reason|hold|entry_i|exit_i|segment|roundtrip|tp_move|sl_move|source|ledger|report|"
    r"notional|leverage|margin|return|pnl|profit|loss|stop|take_profit|stop_loss|"
    r"paper_|borrow_|roll8_|roll24_|cum|peak|dd)",
    re.IGNORECASE,
)
PROTECTED_REFRESH_COLUMNS = {
    "timestamp",
    "entry_timestamp",
    "exit_timestamp",
    "entry_i",
    "exit_i",
    "side",
    "notional",
    "leverage",
    "margin_fraction",
    "risk_notional",
    "risk_leverage",
    "risk_margin_fraction",
    "exit_input_notional",
    "exit_input_leverage",
    "exit_input_exposure",
    "trade_return",
    "net_per_notional",
    "win",
    "reason",
    "feature_veto_spec",
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def flatten(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            out[f"{prefix}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[f"{prefix}_{key}"] = value
    return out


def load_market_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    pnl = load_module("omega462_pnl_tilt_for_feature_refresh", PNL_TILT_PATH)
    roll8 = pnl.load_module("omega462_roll8_for_feature_refresh", pnl.ROLL8_PATH)
    base = roll8.load_module("omega462_roll9_base_for_feature_refresh", roll8.BASE_PATH)
    source = base.load_module("omega462_roll16_source_for_feature_refresh", base.SOURCE_MODULE_PATH)
    v1 = source.load_module("omega462_loss_cluster_v1_for_feature_refresh", source.V1_MODULE_PATH)
    stop_mod = v1.load_module("omega462_stop_mod_for_feature_refresh", v1.STOP_MODULE_PATH)
    runtime = stop_mod.read_json(stop_mod.resolve_path(v1.DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    return pd.read_csv(train_csv), pd.read_csv(eval_csv)


def refresh_entry_features(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Replace segment ledger feature columns with source values at segment entry timestamp."""
    if "entry_timestamp" not in df.columns:
        raise RuntimeError("entry_timestamp column is required for entry feature refresh")
    if "timestamp" not in market.columns:
        raise RuntimeError("market timestamp column is required for entry feature refresh")
    shared = [
        col
        for col in market.columns
        if col in df.columns and col not in PROTECTED_REFRESH_COLUMNS
    ]
    if not shared:
        raise RuntimeError("no shared market feature columns available for entry feature refresh")
    out = df.copy()
    key_col = "__entry_timestamp_key"
    out[key_col] = pd.to_datetime(out["entry_timestamp"], errors="coerce").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    market_work = market[["timestamp", *shared]].copy()
    market_work[key_col] = pd.to_datetime(
        market_work["timestamp"], errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    if bool(market_work[key_col].duplicated().any()):
        raise RuntimeError("market timestamps are not unique for entry feature refresh")
    market_work = market_work.set_index(key_col)
    missing = sorted(set(out[key_col].dropna()) - set(market_work.index))
    if missing:
        raise RuntimeError(
            f"entry feature refresh failed; {len(missing)} entry timestamps missing from market"
        )
    refreshed = market_work.reindex(out[key_col])[shared].reset_index(drop=True)
    refreshed.index = out.index
    out.loc[:, shared] = refreshed
    out = out.drop(columns=[key_col])
    return out


def candidate_features(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        if col == "side" or LOOKAHEAD_EXCLUDE_RE.search(col):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().mean() < 0.85:
            continue
        if series.nunique(dropna=True) < 5:
            continue
        out.append(col)
    return out


def apply_veto(df: pd.DataFrame, feature: str, op: str, threshold: float, spec_name: str) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    if "feature_veto_spec" not in out.columns:
        out["feature_veto_spec"] = "none"
    active = out["notional"].astype(float) > EPS
    is_short = out["side"].astype(int) < 0
    values = pd.to_numeric(out[feature], errors="coerce")
    if op == "<=":
        condition = values <= float(threshold)
    elif op == ">=":
        condition = values >= float(threshold)
    else:
        raise ValueError(f"unsupported op: {op}")
    mask = active & is_short & condition.fillna(False)
    if not bool(mask.any()):
        return out, 0
    for col in [
        "notional",
        "leverage",
        "margin_fraction",
        "risk_notional",
        "risk_leverage",
        "risk_margin_fraction",
        "exit_input_notional",
        "exit_input_leverage",
        "exit_input_exposure",
        "trade_return",
    ]:
        if col in out.columns:
            out.loc[mask, col] = 0.0
    out.loc[mask, "reason"] = "feature_veto"
    out.loc[mask, "feature_veto_spec"] = spec_name
    return out, int(mask.sum())


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 v5 Roll8 Side-Specific Feature Veto - 2026-07-01

## Method

This branch starts from `{REFERENCE_MODEL_ID}` and applies one path-causal short-entry veto based on a single entry-time feature threshold. Selection is validation-primary; OOS is a safety gate and is not an ordering key.

Lookahead-like fields are excluded by name before threshold search.

## Result

- Status: `{report["status"]}`
- Reference model: `{report["reference_model_id"]}`
- Parent model: `{report["parent_model_id"]}`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `{reference["validation_pnl"]:.4f}` | `{selected["validation_pnl"]:.4f}` | `{reference["oos_pnl"]:.4f}` | `{selected["oos_pnl"]:.4f}` |
| MDD % | `{reference["validation_mdd"]:.4f}` | `{selected["validation_mdd"]:.4f}` | `{reference["oos_mdd"]:.4f}` | `{selected["oos_mdd"]:.4f}` |
| Trades | `{reference["validation_trades"]}` | `{selected["validation_trades"]}` | `{reference["oos_trades"]}` | `{selected["oos_trades"]}` |
| Avg hold h | `{reference["validation_avg_hold_hours"]:.4f}` | `{selected["validation_avg_hold_hours"]:.4f}` | `{reference["oos_avg_hold_hours"]:.4f}` | `{selected["oos_avg_hold_hours"]:.4f}` |
| Max hold h | `{reference["validation_max_hold_hours"]:.4f}` | `{selected["validation_max_hold_hours"]:.4f}` | `{reference["oos_max_hold_hours"]:.4f}` | `{selected["oos_max_hold_hours"]:.4f}` |

## Selected Candidate

- Feature: `{selected["feature_name"]}`
- Rule: `{selected["feature_name"]} {selected["feature_op"]} {selected["feature_threshold"]:.8g}`
- Quantile: `{selected["feature_quantile"]}`
- Validation/OOS vetoed shorts: `{selected["validation_vetoed"]}` / `{selected["oos_vetoed"]}`
- Research gate pass: `{selected["research_feature_veto_gate_pass"]}`

## Artifacts

- Ranking: `{report["artifacts"]["ranking"]}`
- Top 20: `{report["artifacts"]["top20"]}`
- Report: `{report["artifacts"]["report"]}`
- Validation ledger: `{report["artifacts"]["selected_validation_ledger"]}`
- OOS ledger: `{report["artifacts"]["selected_oos_ledger"]}`
"""
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    helper = load_module("omega462_feature_veto_audit_helper", AUDIT_HELPER_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_report = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))
    reference = reference_report["selected_variant"]
    val = pd.read_csv(reference_report["artifacts"]["selected_validation_ledger"])
    oos = pd.read_csv(reference_report["artifacts"]["selected_oos_ledger"])
    train_market, eval_market = load_market_frames()
    val = refresh_entry_features(val, train_market)
    oos = refresh_entry_features(oos, eval_market)
    active_short = val[(val["notional"].astype(float) > EPS) & (val["side"].astype(int) < 0)]
    max_validation_vetoed = max(2, int(len(active_short) * MAX_VALIDATION_SHORT_VETO_FRACTION))

    rows: list[dict[str, Any]] = []
    for feature in candidate_features(val):
        feature_values = (
            pd.to_numeric(active_short[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(feature_values) < 20:
            continue
        for quantile in QUANTILES:
            threshold = float(feature_values.quantile(quantile))
            op = "<=" if quantile < 0.5 else ">="
            spec_name = f"{feature}_{op}_{threshold:.8g}"
            val_work, val_vetoed = apply_veto(val, feature, op, threshold, spec_name)
            if val_vetoed < 2 or val_vetoed > max_validation_vetoed:
                continue
            oos_work, oos_vetoed = apply_veto(oos, feature, op, threshold, spec_name)
            val_metrics = helper.metrics(val_work)
            oos_metrics = helper.metrics(oos_work)
            row = {
                "feature_name": feature,
                "feature_op": op,
                "feature_threshold": threshold,
                "feature_quantile": quantile,
                "validation_vetoed": val_vetoed,
                "oos_vetoed": oos_vetoed,
                "max_validation_short_veto_fraction": MAX_VALIDATION_SHORT_VETO_FRACTION,
                **flatten("validation", val_metrics),
                **flatten("oos", oos_metrics),
            }
            validation_gate = bool(
                float(row["validation_pnl"]) > float(reference["validation_pnl"])
                and float(row["validation_mdd"]) >= -20.0
                and float(row["validation_max_hold_hours"]) <= float(reference["validation_max_hold_hours"]) + 1.0e-9
                and float(row["validation_avg_hold_hours"]) < float(reference["validation_avg_hold_hours"])
                and int(row["validation_overlap_count"]) == 0
                and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
                and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
                and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
            )
            oos_gate = bool(
                float(row["oos_pnl"]) >= float(reference["oos_pnl"])
                and float(row["oos_mdd"]) >= float(reference["oos_mdd"])
                and float(row["oos_max_hold_hours"]) <= float(reference["oos_max_hold_hours"]) + 1.0e-9
                and float(row["oos_avg_hold_hours"]) <= float(reference["oos_avg_hold_hours"]) + 1.0e-9
                and int(row["oos_overlap_count"]) == 0
                and float(row["oos_max_notional"]) <= 5.0 + 1.0e-9
                and float(row["oos_accounting_error_max_abs"]) <= 1.0e-10
                and float(row["oos_notional_contract_error_max_abs"]) <= 1.0e-10
            )
            row["validation_feature_veto_gate_pass"] = validation_gate
            row["oos_safety_gate_pass"] = oos_gate
            row["research_feature_veto_gate_pass"] = validation_gate and oos_gate
            rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        [
            "research_feature_veto_gate_pass",
            "validation_pnl",
            "validation_avg_hold_hours",
            "validation_vetoed",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    if ranking.empty:
        raise RuntimeError("no feature-veto variants evaluated")
    selected = ranking.iloc[0].to_dict()
    spec_name = f"{selected['feature_name']}_{selected['feature_op']}_{float(selected['feature_threshold']):.8g}"
    selected_val, _ = apply_veto(
        val,
        str(selected["feature_name"]),
        str(selected["feature_op"]),
        float(selected["feature_threshold"]),
        spec_name,
    )
    selected_oos, _ = apply_veto(
        oos,
        str(selected["feature_name"]),
        str(selected["feature_op"]),
        float(selected["feature_threshold"]),
        spec_name,
    )
    status = (
        "RESEARCH_ROLL8_SIDE_SPECIFIC_FEATURE_VETO_PASS"
        if bool(selected["research_feature_veto_gate_pass"])
        else "NO_ROLL8_SIDE_SPECIFIC_FEATURE_VETO_PASSING_CANDIDATE"
    )
    safe = spec_name.replace(".", "p").replace("/", "_").replace("<=", "le").replace(">=", "ge")
    ranking_path = OUT_DIR / "roll8_side_specific_feature_veto_ranking.csv"
    top20_path = OUT_DIR / "roll8_side_specific_feature_veto_top20.csv"
    val_out = OUT_DIR / f"validation_{safe}_ledger.csv"
    oos_out = OUT_DIR / f"oos_{safe}_ledger.csv"
    report_path = OUT_DIR / "report.json"
    ranking.to_csv(ranking_path, index=False)
    ranking.head(20).to_csv(top20_path, index=False)
    selected_val.to_csv(val_out, index=False)
    selected_oos.to_csv(oos_out, index=False)
    report = {
        "model_id": MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "validation_primary_single_entry_feature_short_veto_with_oos_reference_safety_gate; fresh_holdout_required",
        "selection_rule": "search single non-lookahead-named numeric entry feature thresholds; among research-gated variants sort by validation_pnl, validation_avg_hold, validation_vetoed; OOS is not an ordering key",
        "lookahead_exclude_regex": LOOKAHEAD_EXCLUDE_RE.pattern,
        "features_evaluated": len(candidate_features(val)),
        "variants_evaluated": int(len(ranking)),
        "reference_variant": reference,
        "selected_variant": selected,
        "top20": ranking.head(20).to_dict(orient="records"),
        "status": status,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(ranking_path),
            "top20": str(top20_path),
            "selected_validation_ledger": str(val_out),
            "selected_oos_ledger": str(oos_out),
            "report": str(report_path),
            "audit_md": str(AUDIT_MD),
        },
    }
    write_json(report_path, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": status,
                "selected_feature": selected["feature_name"],
                "selected_rule": f"{selected['feature_name']} {selected['feature_op']} {float(selected['feature_threshold']):.8g}",
                "selected_validation": {
                    "pnl": selected["validation_pnl"],
                    "mdd": selected["validation_mdd"],
                    "avg_hold_hours": selected["validation_avg_hold_hours"],
                    "max_hold_hours": selected["validation_max_hold_hours"],
                    "vetoed": selected["validation_vetoed"],
                    "gate": bool(selected["validation_feature_veto_gate_pass"]),
                },
                "selected_oos": {
                    "pnl": selected["oos_pnl"],
                    "mdd": selected["oos_mdd"],
                    "avg_hold_hours": selected["oos_avg_hold_hours"],
                    "max_hold_hours": selected["oos_max_hold_hours"],
                    "vetoed": selected["oos_vetoed"],
                    "gate": bool(selected["oos_safety_gate_pass"]),
                },
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
