#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.certified_teacher_regime_moe import (  # noqa: E402
    CLEAN_PREFIX,
    MODEL_ID,
    append_clean_regime,
    backtest,
    candidate_feature_cols,
    feature_analysis,
    fit_clean_regime_predictor,
    json_default,
    label_frame,
    load_csv,
    merge_teacher_sources,
    predict_moe,
    runtime_grid,
    save_bundle,
    score,
    train_moe,
)
from pipeline.certified_feature_audit import audit_ai_contracts, audit_frame_contract  # noqa: E402


DEFAULT_DIR = ROOT / "data/ensemble/supervised/certified_teacher_regime_moe_v1"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/certified_teacher_regime_moe_v1_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/certified_teacher_regime_moe_v1_audit.json"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/certified_teacher_regime_moe_v1_contract.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate AI+M7+clean-regime teacher MoE on 2025 and fixed 2026 OOS.")
    p.add_argument("--state-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--base-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--ai-2025", type=Path, default=ROOT / "data/tmp/unified_build_ckpt/03_after_ai.csv")
    p.add_argument("--m7-2025", type=Path, default=ROOT / "data/splits/year_oos/rl_training_2025_m7.csv")
    p.add_argument("--base-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--ai-2026", type=Path, default=ROOT / "data/tmp/unified_build_ckpt_2026/03_after_ai.csv")
    p.add_argument("--m7-2026", type=Path, default=ROOT / "data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--contract-out", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--horizon-bars", type=int, default=36)
    p.add_argument("--max-features", type=int, default=112)
    p.add_argument("--max-grid", type=int, default=24, help="Number of runtime configs to evaluate; use 0 for full grid.")
    p.add_argument("--row-limit", type=int, default=0, help="debug only; tail limit for fast smoke tests")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    return p.parse_args()


def _maybe_tail(frame: pd.DataFrame, row_limit: int) -> pd.DataFrame:
    if row_limit and row_limit > 0 and len(frame) > row_limit:
        return frame.tail(row_limit).reset_index(drop=True)
    return frame


def _overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    ta = pd.to_datetime(a["timestamp"], errors="coerce").dropna().astype("int64")
    tb = pd.to_datetime(b["timestamp"], errors="coerce").dropna().astype("int64")
    return int(len(set(ta.tolist()) & set(tb.tolist())))


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "ledger"}


def _write_contract(path: Path, report: dict[str, Any], audit: dict[str, Any]) -> None:
    c1 = report["metrics"]["cost1"]
    lines = [
        "# Certified Teacher Regime MoE V1",
        "",
        f"- Model ID: `{MODEL_ID}`",
        "- 2025 is the training/selection/holdout year. 2026 is fixed OOS.",
        "- Inputs: certified AI outputs, M7 outputs, clean_regime_2024_unsup_v4_* features, causal market/microstructure features.",
        "- Forbidden: legacy regime-v2/HDB/HMM, raw future/target/label/accounting columns, uncertified regime columns.",
        f"- Audit status: `{audit['status']}`",
        f"- Blocking: `{audit['blocking']}`",
        "",
        "## Splits",
        f"- Fit: `{report['data']['fit_range'][0]}` to `{report['data']['fit_range'][1]}`",
        f"- Selection: `{report['data']['selection_range'][0]}` to `{report['data']['selection_range'][1]}`",
        f"- Holdout: `{report['data']['holdout_range'][0]}` to `{report['data']['holdout_range'][1]}`",
        f"- OOS: `{report['data']['oos_range'][0]}` to `{report['data']['oos_range'][1]}`",
        "",
        "## Cost1 OOS",
        f"- PnL: `{c1['pnl']}`",
        f"- MDD: `{c1['mdd']}`",
        f"- Trades/day: `{c1['trades_per_day']}`",
        "",
        "## Selected Feature Families",
        f"- Selected features: `{len(report['feature_contract']['selected_features'])}`",
        f"- Clean regime features: `{len([c for c in report['feature_contract']['selected_features'] if c.startswith(CLEAN_PREFIX)])}`",
        f"- AI features: `{len([c for c in report['feature_contract']['selected_features'] if c.startswith(('ai_', 'patchtst_', 'tide_', 'timesnet_', 'dlinear_')) or c in {'pred_patchtst', 'conf_patchtst'}])}`",
        f"- M7 features: `{len([c for c in report['feature_contract']['selected_features'] if c.startswith('m7_')])}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ai_audit = audit_ai_contracts()

    y2024 = load_csv(args.state_2024)
    base2025 = _maybe_tail(load_csv(args.base_2025), args.row_limit)
    ai2025 = load_csv(args.ai_2025)
    m72025 = load_csv(args.m7_2025)
    base2026 = _maybe_tail(load_csv(args.base_2026), args.row_limit)
    ai2026 = load_csv(args.ai_2026)
    m72026 = load_csv(args.m7_2026)

    # 2024 clean-regime predictor is fitted only on 2024 market data. AI/M7 teacher columns are
    # used in 2025/2026 decision learning, not for fitting the clean regime predictor.
    regime = fit_clean_regime_predictor(y2024)
    train2025 = append_clean_regime(merge_teacher_sources(base2025, ai2025, m72025), regime)
    eval2026 = append_clean_regime(merge_teacher_sources(base2026, ai2026, m72026), regime)
    train2025.to_csv(args.out_dir / "features_2025.csv", index=False)
    eval2026.to_csv(args.out_dir / "features_2026.csv", index=False)

    labeled = label_frame(train2025, int(args.horizon_bars), float(args.fee), float(args.slip))
    fit = labeled[labeled["timestamp"] < pd.Timestamp("2025-09-01")].copy()
    selection = labeled[(labeled["timestamp"] >= pd.Timestamp("2025-09-01")) & (labeled["timestamp"] < pd.Timestamp("2025-11-01"))].copy()
    holdout = labeled[labeled["timestamp"] >= pd.Timestamp("2025-11-01")].copy()
    if fit.empty or selection.empty or holdout.empty:
        raise ValueError("empty 2025 fit/selection/holdout split")

    candidates = candidate_feature_cols([fit, selection, holdout, eval2026])
    selected, feature_rows = feature_analysis(fit, candidates, int(args.max_features))
    feature_audit = audit_frame_contract(train2025, feature_cols=selected, clean_prefix=CLEAN_PREFIX)
    if feature_audit["status"] != "pass":
        raise ValueError("feature contract failed: " + json.dumps(feature_audit, ensure_ascii=False))

    model = train_moe(fit, selected)
    sel_proba, sel_risk_l, sel_risk_s = predict_moe(model, selection)
    hold_proba, hold_risk_l, hold_risk_s = predict_moe(model, holdout)
    oos_proba, oos_risk_l, oos_risk_s = predict_moe(model, eval2026)

    grid = runtime_grid()
    if args.max_grid and args.max_grid > 0:
        grid = grid[: int(args.max_grid)]
    rows: list[dict[str, Any]] = []
    best_cfg = None
    best_score = -1e18
    best_sel = None
    for idx, cfg in enumerate(grid, start=1):
        if idx == 1 or idx % 8 == 0 or idx == len(grid):
            print(f"[{MODEL_ID}] selection grid {idx}/{len(grid)}", flush=True)
        r1 = backtest(selection, sel_proba, sel_risk_l, sel_risk_s, cfg, fee=args.fee, slip=args.slip)
        r2 = backtest(selection, sel_proba, sel_risk_l, sel_risk_s, cfg, fee=args.fee * 2.0, slip=args.slip * 2.0)
        r3 = backtest(selection, sel_proba, sel_risk_l, sel_risk_s, cfg, fee=args.fee * 3.0, slip=args.slip * 3.0)
        s = 0.50 * score(r1) + 0.30 * score(r2) + 0.20 * score(r3)
        if r2["pnl"] < 0:
            s -= abs(float(r2["pnl"])) * 2.0
        if r3["pnl"] < 0:
            s -= abs(float(r3["pnl"])) * 3.5
        rows.append({"score": float(s), **asdict(cfg), **{f"selection_{k}": v for k, v in _compact(r1).items()}, "selection_cost2_pnl": r2["pnl"], "selection_cost3_pnl": r3["pnl"]})
        if s > best_score:
            best_score = float(s)
            best_cfg = cfg
            best_sel = r1
    if best_cfg is None or best_sel is None:
        raise RuntimeError("no selected runtime config")

    holdout_result = backtest(holdout, hold_proba, hold_risk_l, hold_risk_s, best_cfg, fee=args.fee, slip=args.slip)
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    last_ledger = pd.DataFrame()
    for mult in (1, 2, 3):
        result = backtest(eval2026, oos_proba, oos_risk_l, oos_risk_s, best_cfg, fee=args.fee * mult, slip=args.slip * mult)
        key = f"cost{mult}"
        metrics[key] = _compact(result)
        ledger_path = args.report_out.with_name(args.report_out.stem + f"_{key}_ledger.csv")
        ledger = pd.DataFrame(result["ledger"])
        ledger.to_csv(ledger_path, index=False)
        ledgers[key] = str(ledger_path)
        last_ledger = ledger

    model_path = args.out_dir / "model.pkl"
    save_bundle(
        model_path,
        {
            "model_id": MODEL_ID,
            "clean_prefix": CLEAN_PREFIX,
            "regime": regime,
            "model": model,
            "feature_cols": selected,
            "selected_config": asdict(best_cfg),
            "split_policy": "2025 train/selection/holdout, 2026 fixed OOS",
        },
    )
    grid_path = args.report_out.with_name(args.report_out.stem + "_selection_grid.csv")
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(grid_path, index=False)
    feature_path = args.report_out.with_name(args.report_out.stem + "_feature_analysis.csv")
    pd.DataFrame(feature_rows).to_csv(feature_path, index=False)

    report = {
        "model_id": MODEL_ID,
        "design": "AI/M7/clean_regime teacher-fusion encoder + clean-regime conditioned MoE + adverse-risk critic + cost-stressed selection.",
        "data": {
            "state_2024": str(args.state_2024),
            "base_2025": str(args.base_2025),
            "ai_2025": str(args.ai_2025),
            "m7_2025": str(args.m7_2025),
            "base_2026": str(args.base_2026),
            "ai_2026": str(args.ai_2026),
            "m7_2026": str(args.m7_2026),
            "fit_range": [str(fit["timestamp"].iloc[0]), str(fit["timestamp"].iloc[-1])],
            "selection_range": [str(selection["timestamp"].iloc[0]), str(selection["timestamp"].iloc[-1])],
            "holdout_range": [str(holdout["timestamp"].iloc[0]), str(holdout["timestamp"].iloc[-1])],
            "oos_range": [str(eval2026["timestamp"].iloc[0]), str(eval2026["timestamp"].iloc[-1])],
        },
        "data_audit": {
            "ai_contracts": ai_audit,
            "feature_contract": feature_audit,
            "fit_rows": int(len(fit)),
            "selection_rows": int(len(selection)),
            "holdout_rows": int(len(holdout)),
            "eval_rows": int(len(eval2026)),
            "train_eval_overlap": _overlap(fit, eval2026) + _overlap(selection, eval2026) + _overlap(holdout, eval2026),
            "m7_provenance_status": "uncertified_existing_artifacts_used_as_teacher_inputs",
        },
        "artifacts": {
            "model": str(model_path),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "contract": str(args.contract_out),
            "selection_grid": str(grid_path),
            "feature_analysis": str(feature_path),
            "ledgers": ledgers,
        },
        "feature_contract": {
            "candidate_feature_count": len(candidates),
            "selected_features": selected,
            "top_feature_analysis": feature_rows[:50],
        },
        "selected_config": asdict(best_cfg),
        "selection_score": best_score,
        "selection_result": _compact(best_sel),
        "holdout_result": _compact(holdout_result),
        "metrics": metrics,
    }
    blocking = []
    warnings = []
    if ai_audit["status"] != "pass":
        blocking.extend(ai_audit["blocking"])
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    if report["data_audit"]["train_eval_overlap"] != 0:
        blocking.append("train_eval_timestamp_overlap")
    if last_ledger.empty or pd.to_datetime(last_ledger["timestamp"], errors="coerce").max() < pd.to_datetime(eval2026["timestamp"], errors="coerce").max():
        blocking.append("ledger_does_not_cover_eval_window")
    warnings.append("m7_existing_artifacts_have_no_embedded_training_provenance; treat as teacher-input caveat until M7 is force-retrained with provenance writer")
    audit = {
        "model_id": MODEL_ID,
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "invariants": {
            "2025_declared_as_training_year": True,
            "2026_fixed_oos_no_selection": True,
            "ai_contracts_2024_certified": ai_audit["status"] == "pass",
            "legacy_regime_v2_quarantined": not feature_audit["forbidden_feature_cols"],
            "next_bar_open_execution": True,
            "entry_and_exit_cost_charged": True,
            "cost_1x_2x_3x_reported": all(k in metrics for k in ("cost1", "cost2", "cost3")),
        },
    }
    report["audit"] = audit
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    _write_contract(args.contract_out, report, audit)
    print(json.dumps({"status": audit["status"], "metrics": metrics, "report": str(args.report_out), "audit": str(args.audit_out)}, indent=2, ensure_ascii=False, default=json_default))
    return 0 if audit["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
