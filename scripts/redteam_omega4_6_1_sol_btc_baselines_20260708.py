#!/usr/bin/env python3
"""Red-team audit for active SOL/BTC Omega4.6.1 research baselines."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_sol_btc_baseline_redteam_20260708"
AUDIT_MD = ROOT / "docs/audits/omega4_6_1_sol_btc_baseline_redteam_20260708.md"

BASELINES: dict[str, dict[str, Any]] = {
    "sol": {
        "component": "zig075",
        "tag": "q070",
        "quality_threshold": 0.70,
        "duration_threshold": 0.0055208323,
        "report": ROOT / "tmp/causal_regen_20260516/sol_final_scale_map_20260707/report.json",
        "validation_ledger": ROOT / "tmp/causal_regen_20260516/sol_final_scale_map_20260707/validation_ledger.csv",
        "oos_ledger": ROOT / "tmp/causal_regen_20260516/sol_final_scale_map_20260707/oos_ledger.csv",
        "features_2025": ROOT / "data/splits/year_oos/sol_features_2025.csv",
        "features_2026": ROOT / "data/splits/year_oos/sol_features_2026.csv",
        "parent_dir": ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_zig075_20260707",
        "risk_dir": ROOT / "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_zig075_q070_20260707",
        "expected": {
            "validation": {"pnl": 56.751511299446754, "mdd": -15.865518178523164, "trades": 28, "wr": 0.42857142857142855},
            "oos_extended": {"pnl": 13.922356640229516, "mdd": -29.378701782015327, "trades": 39, "wr": 0.38461538461538464},
            "oos_frozen_q1_2026": {"pnl": 41.979021886908455, "mdd": -21.033233004014562, "trades": 20, "wr": 0.5},
        },
    },
    "btc": {
        "component": "h48qual",
        "tag": "q055",
        "quality_threshold": 0.55,
        "duration_threshold": 0.00541154875,
        "report": ROOT / "tmp/causal_regen_20260516/btc_final_scale_map_20260708/report.json",
        "validation_ledger": ROOT / "tmp/causal_regen_20260516/btc_final_scale_map_20260708/validation_ledger.csv",
        "oos_ledger": ROOT / "tmp/causal_regen_20260516/btc_final_scale_map_20260708/oos_ledger.csv",
        "features_2025": ROOT / "data/splits/year_oos/btc_features_2025.csv",
        "features_2026": ROOT / "data/splits/year_oos/btc_features_2026.csv",
        "parent_dir": ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_20260708",
        "risk_dir": ROOT / "tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260708",
        "expected": {
            "validation": {"pnl": 12.394013383530167, "mdd": -6.491917072790132, "trades": 10, "wr": 0.4},
            "oos_extended": {"pnl": 29.23304318876616, "mdd": -10.654749514772488, "trades": 24, "wr": 0.4166666666666667},
            "oos_frozen_q1_2026": {"pnl": 10.170892511666851, "mdd": -10.654749514772488, "trades": 16, "wr": 0.375},
        },
    },
}

FORBIDDEN_PREFIXES = ("clean_regime4_", "regime4_pred_", "regime3_pred_", "teacher_", "teacher_oof_", "a5dir_")
FORBIDDEN_TOKENS = ("target", "future", "label", "pnl", "zigzag", "wave3", "tp_sl_action_score")
ALLOWED_REASONS = {"take_profit", "stop_loss", "exit_head", "forced_end"}


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


def _compound_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    for ret in ledger["trade_return"].to_numpy(dtype=np.float64):
        cash *= 1.0 + float(ret)
        wins += int(ret > 0.0)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    return {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(len(ledger)), "wr": float(wins / len(ledger))}


def _check_close(name: str, got: float, expected: float, tol: float, issues: list[dict[str, Any]]) -> None:
    if abs(float(got) - float(expected)) > tol:
        issues.append({"severity": "P1", "check": name, "message": f"expected {expected}, got {got}"})


def _load_gated_ledger(ledger_path: Path, features_path: Path, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger = pd.read_csv(ledger_path, parse_dates=["entry_timestamp", "exit_timestamp"])
    feats = pd.read_csv(features_path, usecols=["timestamp", "ou_halflife"], parse_dates=["timestamp"])
    merged = ledger.merge(feats.rename(columns={"timestamp": "entry_timestamp"}), on="entry_timestamp", how="left", validate="one_to_one")
    return ledger, merged.loc[merged["ou_halflife"] > float(threshold)].reset_index(drop=True)


def _audit_ledger(asset: str, split: str, ledger: pd.DataFrame, gated: pd.DataFrame, issues: list[dict[str, Any]]) -> dict[str, Any]:
    diag: dict[str, Any] = {"rows": int(len(ledger))}
    required = {
        "entry_signal_i", "entry_i", "exit_i", "entry_timestamp", "exit_timestamp", "side", "reason",
        "trade_return", "net_per_notional", "notional", "margin_fraction", "leverage",
        "take_profit", "stop_loss",
    }
    missing = sorted(required - set(ledger.columns))
    if missing:
        issues.append({"severity": "P0", "check": f"{asset}_{split}_ledger_schema", "message": f"missing columns {missing}"})
        return diag
    if ledger.empty:
        issues.append({"severity": "P1", "check": f"{asset}_{split}_empty_ledger", "message": "ledger has no trades"})
        return diag
    missing_duration = int(gated["ou_halflife"].isna().sum()) if "ou_halflife" in gated.columns else int(len(gated))
    diag["gated_rows"] = int(len(gated))
    diag["missing_duration_rows_after_gate"] = missing_duration
    if missing_duration:
        issues.append({"severity": "P0", "check": f"{asset}_{split}_duration_merge", "message": f"{missing_duration} gated rows missing ou_halflife"})

    entry_i = pd.to_numeric(ledger["entry_i"], errors="raise").to_numpy(dtype=np.int64)
    exit_i = pd.to_numeric(ledger["exit_i"], errors="raise").to_numpy(dtype=np.int64)
    if not np.all(exit_i >= entry_i):
        issues.append({"severity": "P0", "check": f"{asset}_{split}_exit_before_entry", "message": "some exit_i < entry_i"})
    order = np.argsort(entry_i)
    overlaps = int(np.sum(entry_i[order][1:] <= exit_i[order][:-1]))
    diag["overlap_count"] = overlaps
    if overlaps:
        issues.append({"severity": "P0", "check": f"{asset}_{split}_position_overlap", "message": f"{overlaps} overlapping intervals"})

    notional = pd.to_numeric(ledger["notional"], errors="raise").to_numpy(dtype=np.float64)
    margin = pd.to_numeric(ledger["margin_fraction"], errors="raise").to_numpy(dtype=np.float64)
    leverage = pd.to_numeric(ledger["leverage"], errors="raise").to_numpy(dtype=np.float64)
    trade_return = pd.to_numeric(ledger["trade_return"], errors="raise").to_numpy(dtype=np.float64)
    net_per_notional = pd.to_numeric(ledger["net_per_notional"], errors="raise").to_numpy(dtype=np.float64)
    notional_err = np.abs(notional - margin * leverage)
    return_err = np.abs(trade_return - net_per_notional * notional)
    diag["max_notional_identity_error"] = float(notional_err.max())
    diag["max_return_identity_error"] = float(return_err.max())
    if float(notional_err.max()) > 1e-9:
        issues.append({"severity": "P0", "check": f"{asset}_{split}_notional_identity", "message": f"max error {notional_err.max()}"})
    if float(return_err.max()) > 1e-9:
        issues.append({"severity": "P0", "check": f"{asset}_{split}_return_identity", "message": f"max error {return_err.max()}"})
    if float(np.nanmax(leverage)) > 5.0 + 1e-9:
        issues.append({"severity": "P0", "check": f"{asset}_{split}_leverage_cap", "message": f"max leverage {np.nanmax(leverage)}"})
    if float(np.nanmax(notional)) > 1.8 + 1e-9:
        issues.append({"severity": "P0", "check": f"{asset}_{split}_notional_cap", "message": f"max notional {np.nanmax(notional)}"})
    if bool(np.any(trade_return <= -1.0)):
        issues.append({"severity": "P0", "check": f"{asset}_{split}_bankruptcy_return", "message": "trade_return <= -100%"})
    reasons = set(ledger["reason"].astype(str))
    bad_reasons = sorted(reasons - ALLOWED_REASONS)
    if bad_reasons:
        issues.append({"severity": "P1", "check": f"{asset}_{split}_exit_reasons", "message": f"unexpected reasons {bad_reasons}"})
    sides = set(ledger["side"].astype(str))
    bad_sides = sorted(sides - {"long", "short", "-1", "1"})
    if bad_sides:
        issues.append({"severity": "P0", "check": f"{asset}_{split}_sides", "message": f"unexpected sides {bad_sides}"})
    diag.update(
        {
            "max_leverage": float(np.nanmax(leverage)),
            "max_notional": float(np.nanmax(notional)),
            "min_trade_return": float(np.nanmin(trade_return)),
            "max_trade_return": float(np.nanmax(trade_return)),
            "reason_counts": {str(k): int(v) for k, v in ledger["reason"].value_counts().items()},
        }
    )
    return diag


def _audit_artifacts(asset: str, cfg: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    diag: dict[str, Any] = {}
    parent_dir = cfg["parent_dir"]
    risk_dir = cfg["risk_dir"]
    expected_files = [
        cfg["report"],
        cfg["validation_ledger"],
        cfg["oos_ledger"],
        parent_dir / "true_3head_tabm_bundle.pt",
        risk_dir / "risk_sidecar.pkl",
        parent_dir / f"train_predictions_{cfg['tag']}.csv",
        parent_dir / f"validation_predictions_{cfg['tag']}.csv",
        parent_dir / f"oos_predictions_{cfg['tag']}.csv",
    ]
    missing = [str(p) for p in expected_files if not Path(p).exists()]
    diag["missing_files"] = missing
    if missing:
        issues.append({"severity": "P0", "check": f"{asset}_artifact_presence", "message": f"missing files: {missing}"})

    if not missing:
        bundle = torch.load(parent_dir / "true_3head_tabm_bundle.pt", map_location="cpu", weights_only=False)
        base_cols = list(bundle["base_cols"])
        forbidden = [c for c in base_cols if c.startswith(FORBIDDEN_PREFIXES) or any(tok in c.lower() for tok in FORBIDDEN_TOKENS)]
        diag["base_feature_count"] = len(base_cols)
        diag["forbidden_base_features"] = forbidden
        if forbidden:
            issues.append({"severity": "P0", "check": f"{asset}_forbidden_features", "message": f"forbidden base cols {forbidden[:20]}"})
        with open(risk_dir / "risk_sidecar.pkl", "rb") as f:
            pkl = pickle.load(f)
        diag["risk_feature_mode"] = pkl.get("risk_feature_mode")
        diag["risk_side_split_model"] = bool(pkl.get("side_split_model"))
        diag["risk_dynamic_leverage"] = bool(pkl.get("dynamic_leverage"))
        if pkl.get("risk_feature_mode") != "parent_outputs":
            issues.append({"severity": "P1", "check": f"{asset}_risk_feature_mode", "message": f"risk_feature_mode={pkl.get('risk_feature_mode')}"})
        if not pkl.get("side_split_model"):
            issues.append({"severity": "P1", "check": f"{asset}_side_split", "message": "risk sidecar is not side-split"})
        if not pkl.get("dynamic_leverage"):
            issues.append({"severity": "P1", "check": f"{asset}_dynamic_leverage", "message": "risk sidecar does not use dynamic leverage"})
    return diag


def _audit_report_contract(asset: str, cfg: dict[str, Any], report: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    diag: dict[str, Any] = {"keys": sorted(report.keys())}
    explicit = {
        "component": report.get("component"),
        "quality_threshold": report.get("quality_threshold"),
        "precomputed_prediction_tag": report.get("precomputed_prediction_tag"),
        "long_scale": report.get("long_scale"),
        "short_scale": report.get("short_scale"),
    }
    diag.update(explicit)
    if explicit["component"] is not None and explicit["component"] != cfg["component"]:
        issues.append({"severity": "P0", "check": f"{asset}_report_component", "message": f"report component {explicit['component']} != {cfg['component']}"})
    if explicit["quality_threshold"] is not None and abs(float(explicit["quality_threshold"]) - float(cfg["quality_threshold"])) > 1e-12:
        issues.append({"severity": "P0", "check": f"{asset}_report_quality_threshold", "message": f"report quality_threshold {explicit['quality_threshold']} != {cfg['quality_threshold']}"})
    if explicit["precomputed_prediction_tag"] is not None and explicit["precomputed_prediction_tag"] != cfg["tag"]:
        issues.append({"severity": "P0", "check": f"{asset}_report_prediction_tag", "message": f"report tag {explicit['precomputed_prediction_tag']} != {cfg['tag']}"})
    selected = report.get("duration_gate", {}).get("selected", {}) if isinstance(report.get("duration_gate"), dict) else {}
    if selected:
        diag["duration_gate_selected_threshold"] = selected.get("threshold")
        if abs(float(selected.get("threshold", np.nan)) - float(cfg["duration_threshold"])) > 1e-12:
            issues.append({"severity": "P0", "check": f"{asset}_duration_threshold", "message": f"report duration threshold {selected.get('threshold')} != {cfg['duration_threshold']}"})
    elif asset == "sol":
        issues.append({"severity": "P2", "check": "sol_report_missing_duration_gate_object", "message": "SOL final report stores older grid format without explicit selected duration_gate object"})
    return diag


def audit_asset(asset: str, cfg: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    diag: dict[str, Any] = {"asset": asset, "component": cfg["component"], "tag": cfg["tag"], "duration_threshold": cfg["duration_threshold"]}
    diag["artifacts"] = _audit_artifacts(asset, cfg, issues)

    val_raw, val_gated = _load_gated_ledger(cfg["validation_ledger"], cfg["features_2025"], cfg["duration_threshold"])
    oos_raw, oos_gated = _load_gated_ledger(cfg["oos_ledger"], cfg["features_2026"], cfg["duration_threshold"])
    oos_q1 = oos_gated.loc[oos_gated["entry_timestamp"] < pd.Timestamp("2026-04-01")].reset_index(drop=True)
    diag["ledgers"] = {
        "validation": _audit_ledger(asset, "validation", val_raw, val_gated, issues),
        "oos": _audit_ledger(asset, "oos", oos_raw, oos_gated, issues),
    }
    metrics = {"validation": _compound_metrics(val_gated), "oos_extended": _compound_metrics(oos_gated), "oos_frozen_q1_2026": _compound_metrics(oos_q1)}
    diag["recomputed_gated_metrics"] = metrics
    for split, exp in cfg["expected"].items():
        got = metrics[split]
        for k, exp_v in exp.items():
            _check_close(f"{asset}_{split}_{k}", got[k], exp_v, 1e-6, issues)

    if metrics["validation"]["trades"] < 20:
        issues.append({"severity": "P2", "check": f"{asset}_thin_validation_trades", "message": f"validation gated trades={metrics['validation']['trades']}"})
    if metrics["oos_frozen_q1_2026"]["trades"] < 20:
        issues.append({"severity": "P2", "check": f"{asset}_thin_q1_trades", "message": f"Q1 gated trades={metrics['oos_frozen_q1_2026']['trades']}"})
    if metrics["oos_extended"]["mdd"] < -25.0:
        issues.append({"severity": "P1", "check": f"{asset}_oos_mdd_high", "message": f"OOS MDD={metrics['oos_extended']['mdd']:.2f}%"})

    # Report flag audit.
    report = json.loads(Path(cfg["report"]).read_text(encoding="utf-8"))
    diag["report_contract"] = _audit_report_contract(asset, cfg, report, issues)
    for key in ("fresh_forward_bar_by_bar", "trade_ledgers_used_as_input", "saved_parent_exit_timestamps_used", "future_rows_used_for_entry"):
        if key in report:
            diag[key] = report[key]
    if asset == "sol":
        issues.append({"severity": "P2", "check": "sol_report_missing_replay_flags", "message": "SOL final report predates explicit fresh-forward flags; audit recomputed ledgers instead"})

    blockers = [x for x in issues if x["severity"] in {"P0", "P1"}]
    diag["issues"] = issues
    diag["blocker_count"] = len(blockers)
    diag["promotion_pass"] = len(blockers) == 0
    return diag


def write_md(report: dict[str, Any]) -> None:
    lines = ["# Omega4.6.1 SOL/BTC Baseline Red-Team Audit - 2026-07-08", "", f"Overall pass: `{report['overall_pass']}`", ""]
    for asset, diag in report["assets"].items():
        lines.extend([f"## {asset.upper()}", "", f"Promotion pass: `{diag['promotion_pass']}`", ""])
        m = diag["recomputed_gated_metrics"]
        lines.extend([
            "| split | PnL | MDD | trades | WR |",
            "|---|---:|---:|---:|---:|",
        ])
        for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
            x = m[split]
            lines.append(f"| {split} | {x['pnl']:.2f}% | {x['mdd']:.2f}% | {x['trades']} | {x['wr']:.2%} |")
        lines.extend(["", "Issues:", ""])
        if diag["issues"]:
            for issue in diag["issues"]:
                lines.append(f"- `{issue['severity']}` `{issue['check']}`: {issue['message']}")
        else:
            lines.append("- None")
        lines.append("")
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assets = {asset: audit_asset(asset, cfg) for asset, cfg in BASELINES.items()}
    report = {
        "audit_id": "omega4_6_1_sol_btc_baseline_redteam_20260708",
        "assets": assets,
        "overall_pass": all(diag["promotion_pass"] for diag in assets.values()),
    }
    (OUT_DIR / "redteam_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    write_md(report)
    print(json.dumps({"audit": str(OUT_DIR / "redteam_audit.json"), "md": str(AUDIT_MD), "overall_pass": report["overall_pass"], "asset_pass": {k: v["promotion_pass"] for k, v in assets.items()}}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
