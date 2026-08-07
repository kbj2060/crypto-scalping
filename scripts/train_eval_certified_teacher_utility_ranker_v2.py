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
    append_clean_regime,
    fit_clean_regime_predictor,
    load_csv,
    merge_teacher_sources,
)
from ensemble.certified_teacher_utility_ranker import (  # noqa: E402
    MODEL_ID,
    backtest_ranker,
    build_candidate_table,
    candidate_feature_cols,
    candidate_model_cols,
    fit_ranker,
    json_default,
    predict_candidates,
    runtime_grid,
    save_bundle,
    score,
)
from pipeline.certified_feature_audit import audit_ai_contracts, audit_frame_contract  # noqa: E402
from pipeline.certified_teacher_meta_features import append_teacher_meta_features  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/certified_teacher_utility_ranker_v2"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/certified_teacher_utility_ranker_v2_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/certified_teacher_utility_ranker_v2_audit.json"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/certified_teacher_utility_ranker_v2_contract.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate execution-utility ranker v2.")
    p.add_argument("--state-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--base-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--ai-2025", type=Path, default=ROOT / "data/tmp/unified_build_ckpt/03_after_ai.csv")
    p.add_argument("--m7-2025", type=Path, default=ROOT / "data/splits/year_oos/rl_training_2025_m7.csv")
    p.add_argument("--base-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--ai-2026", type=Path, default=ROOT / "data/tmp/unified_build_ckpt_2026/03_after_ai.csv")
    p.add_argument("--m7-2026", type=Path, default=ROOT / "data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--contract-out", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--max-features", type=int, default=96)
    p.add_argument("--max-train-candidates", type=int, default=120000)
    p.add_argument("--max-grid", type=int, default=3)
    p.add_argument("--train-label-stride", type=int, default=48)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    return p.parse_args()


def _overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    ta = pd.to_datetime(a["timestamp"], errors="coerce").dropna().astype("int64")
    tb = pd.to_datetime(b["timestamp"], errors="coerce").dropna().astype("int64")
    return int(len(set(ta.tolist()) & set(tb.tolist())))


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "ledger"}


def _write_contract(path: Path, report: dict[str, Any], audit: dict[str, Any]) -> None:
    c1 = report["metrics"]["cost1"]
    lines = [
        "# Certified Teacher Utility Ranker V2",
        "",
        f"- Model ID: `{MODEL_ID}`",
        "- Architecture: teacher meta encoder + execution replay utility labels + candidate utility ranker + adaptive contract family.",
        "- 2025 is model train/selection/holdout. 2026 is fixed OOS and is not used for selection.",
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
        "## Output Contract",
        "- `side`, `contract_family`, `expected_net_pct`, `q10_pct`, `notional`, `leverage`, `SL/TP/trailing/max_hold` from selected contract family.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    y2024 = load_csv(args.state_2024)
    base2025 = load_csv(args.base_2025)
    ai2025 = load_csv(args.ai_2025)
    m72025 = load_csv(args.m7_2025)
    base2026 = load_csv(args.base_2026)
    ai2026 = load_csv(args.ai_2026)
    m72026 = load_csv(args.m7_2026)
    regime = fit_clean_regime_predictor(y2024)
    train2025 = append_teacher_meta_features(append_clean_regime(merge_teacher_sources(base2025, ai2025, m72025), regime))
    eval2026 = append_teacher_meta_features(append_clean_regime(merge_teacher_sources(base2026, ai2026, m72026), regime))
    return train2025, eval2026, regime


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ai_audit = audit_ai_contracts()
    train2025, eval2026, regime = _build_frame(args)
    train2025.to_csv(args.out_dir / "features_2025.csv", index=False)
    eval2026.to_csv(args.out_dir / "features_2026.csv", index=False)

    fit = train2025[train2025["timestamp"] < pd.Timestamp("2025-09-01")].copy()
    selection = train2025[(train2025["timestamp"] >= pd.Timestamp("2025-09-01")) & (train2025["timestamp"] < pd.Timestamp("2025-11-01"))].copy()
    holdout = train2025[train2025["timestamp"] >= pd.Timestamp("2025-11-01")].copy()
    if fit.empty or selection.empty or holdout.empty:
        raise ValueError("empty fit/selection/holdout split")

    base_cols = candidate_feature_cols([fit, selection, holdout, eval2026], CLEAN_PREFIX)
    family_priority = [c for c in base_cols if c.startswith(("teacher_", CLEAN_PREFIX, "m7_", "ai_", "patchtst_", "tide_", "timesnet_", "dlinear_")) or c in {"pred_patchtst", "conf_patchtst"}]
    market_rest = [c for c in base_cols if c not in family_priority]
    base_cols = (family_priority + market_rest)[: int(args.max_features)]

    feature_audit = audit_frame_contract(train2025, feature_cols=base_cols, clean_prefix=CLEAN_PREFIX)
    if feature_audit["status"] != "pass":
        raise ValueError("feature audit failed: " + json.dumps(feature_audit, ensure_ascii=False))

    print(f"[{MODEL_ID}] building train candidates", flush=True)
    train_candidates = build_candidate_table(fit, base_cols, fee=args.fee, slip=args.slip, label=True, row_stride=int(args.train_label_stride))
    model_cols = candidate_model_cols(train_candidates)
    print(f"[{MODEL_ID}] fitting ranker rows={len(train_candidates)} cols={len(model_cols)}", flush=True)
    ranker = fit_ranker(train_candidates, model_cols, max_train_rows=int(args.max_train_candidates))

    print(f"[{MODEL_ID}] predicting candidates", flush=True)
    selection_pred = predict_candidates(ranker, build_candidate_table(selection, base_cols, fee=args.fee, slip=args.slip, label=False))
    holdout_pred = predict_candidates(ranker, build_candidate_table(holdout, base_cols, fee=args.fee, slip=args.slip, label=False))
    oos_pred = predict_candidates(ranker, build_candidate_table(eval2026, base_cols, fee=args.fee, slip=args.slip, label=False))

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
        r1 = backtest_ranker(selection, selection_pred, cfg, fee=args.fee, slip=args.slip)
        r2 = backtest_ranker(selection, selection_pred, cfg, fee=args.fee * 2.0, slip=args.slip * 2.0)
        s = score(r1, r2)
        rows.append({"score": s, **asdict(cfg), **{f"selection_{k}": v for k, v in _compact(r1).items()}, "selection_cost2_pnl": r2["pnl"]})
        if s > best_score:
            best_score = float(s)
            best_cfg = cfg
            best_sel = r1
    if best_cfg is None or best_sel is None:
        raise RuntimeError("no selected config")

    holdout_result = backtest_ranker(holdout, holdout_pred, best_cfg, fee=args.fee, slip=args.slip)
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    last_ledger = pd.DataFrame()
    for mult in (1, 2, 3):
        result = backtest_ranker(eval2026, oos_pred, best_cfg, fee=args.fee * mult, slip=args.slip * mult)
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
            "regime": regime,
            "ranker": ranker,
            "base_feature_cols": base_cols,
            "candidate_model_cols": model_cols,
            "selected_config": asdict(best_cfg),
            "split_policy": "2025 train/selection/holdout; 2026 fixed OOS",
        },
    )
    grid_path = args.report_out.with_name(args.report_out.stem + "_selection_grid.csv")
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(grid_path, index=False)

    report = {
        "model_id": MODEL_ID,
        "design": "Execution replay net-utility labels, teacher meta compression, candidate utility ranker, adaptive SCALP/REBOUND/TREND contract selection.",
        "data": {
            "fit_range": [str(fit["timestamp"].iloc[0]), str(fit["timestamp"].iloc[-1])],
            "selection_range": [str(selection["timestamp"].iloc[0]), str(selection["timestamp"].iloc[-1])],
            "holdout_range": [str(holdout["timestamp"].iloc[0]), str(holdout["timestamp"].iloc[-1])],
            "oos_range": [str(eval2026["timestamp"].iloc[0]), str(eval2026["timestamp"].iloc[-1])],
        },
        "data_audit": {
            "ai_contracts": ai_audit,
            "feature_contract": feature_audit,
            "train_candidate_rows": int(len(train_candidates)),
            "fit_rows": int(len(fit)),
            "selection_rows": int(len(selection)),
            "holdout_rows": int(len(holdout)),
            "eval_rows": int(len(eval2026)),
            "train_eval_overlap": _overlap(fit, eval2026) + _overlap(selection, eval2026) + _overlap(holdout, eval2026),
            "m7_provenance_status": "uncertified_existing_artifacts_used_as_teacher_meta_inputs",
        },
        "artifacts": {
            "model": str(model_path),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "contract": str(args.contract_out),
            "selection_grid": str(grid_path),
            "ledgers": ledgers,
        },
        "feature_contract": {
            "base_feature_count": len(base_cols),
            "candidate_model_feature_count": len(model_cols),
            "base_features": base_cols,
            "candidate_model_cols": model_cols,
        },
        "selected_config": asdict(best_cfg),
        "selection_score": best_score,
        "selection_result": _compact(best_sel),
        "holdout_result": _compact(holdout_result),
        "metrics": metrics,
    }
    blocking: list[str] = []
    warnings = ["m7_existing_artifacts_have_no_embedded_training_provenance; v2 uses compressed M7 teacher meta with caveat"]
    if ai_audit["status"] != "pass":
        blocking.extend(ai_audit["blocking"])
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    if report["data_audit"]["train_eval_overlap"] != 0:
        blocking.append("train_eval_timestamp_overlap")
    if last_ledger.empty or pd.to_datetime(last_ledger["timestamp"], errors="coerce").max() < pd.to_datetime(eval2026["timestamp"], errors="coerce").max():
        blocking.append("ledger_does_not_cover_eval_window")
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
            "labels_are_execution_replay_net_utility": True,
            "adaptive_contract_family_used": True,
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
