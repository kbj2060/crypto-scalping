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

from ensemble.certified_teacher_dual_side_router import (  # noqa: E402
    MODEL_ID,
    backtest,
    build_side_candidates,
    feature_cols,
    fit_side_ranker,
    model_cols,
    predict_side,
    runtime_grid,
    save_bundle,
    score,
)
from ensemble.certified_teacher_regime_moe import CLEAN_PREFIX, append_clean_regime, fit_clean_regime_predictor, load_csv, merge_teacher_sources  # noqa: E402
from pipeline.certified_feature_audit import audit_ai_contracts, audit_frame_contract  # noqa: E402
from pipeline.teacher_meta_side_features import append_side_teacher_features  # noqa: E402


DEFAULT_OUT = ROOT / "data/ensemble/supervised/certified_teacher_dual_side_execution_router_v3"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/certified_teacher_dual_side_execution_router_v3_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/certified_teacher_dual_side_execution_router_v3_audit.json"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/certified_teacher_dual_side_execution_router_v3_contract.md"


def _json_default(obj: Any) -> Any:
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--state-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--base-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--ai-2025", type=Path, default=ROOT / "data/tmp/unified_build_ckpt/03_after_ai.csv")
    p.add_argument("--m7-2025", type=Path, default=ROOT / "data/splits/year_oos/rl_training_2025_m7.csv")
    p.add_argument("--base-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--ai-2026", type=Path, default=ROOT / "data/tmp/unified_build_ckpt_2026/03_after_ai.csv")
    p.add_argument("--m7-2026", type=Path, default=ROOT / "data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--contract-out", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--max-features", type=int, default=96)
    p.add_argument("--label-stride", type=int, default=48)
    p.add_argument("--max-train-rows", type=int, default=40000)
    p.add_argument("--max-grid", type=int, default=3)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    return p.parse_args()


def _build(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    y2024 = load_csv(args.state_2024)
    regime = fit_clean_regime_predictor(y2024)
    y2025 = append_side_teacher_features(append_clean_regime(merge_teacher_sources(load_csv(args.base_2025), load_csv(args.ai_2025), load_csv(args.m7_2025)), regime))
    y2026 = append_side_teacher_features(append_clean_regime(merge_teacher_sources(load_csv(args.base_2026), load_csv(args.ai_2026), load_csv(args.m7_2026)), regime))
    return y2025, y2026, regime


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "ledger"}


def _overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    ta = pd.to_datetime(a["timestamp"], errors="coerce").dropna().astype("int64")
    tb = pd.to_datetime(b["timestamp"], errors="coerce").dropna().astype("int64")
    return int(len(set(ta.tolist()) & set(tb.tolist())))


def _write_contract(path: Path, report: dict[str, Any], audit: dict[str, Any]) -> None:
    c1 = report["metrics"]["cost1"]
    lines = [
        "# Certified Teacher Dual Side Execution Router V3",
        "",
        f"- Model ID: `{MODEL_ID}`",
        "- LONG/SHORT rankers are separated.",
        "- q10/edge are used mostly as size/contract scalers; only catastrophic q10 is a hard veto.",
        "- 2025 is train/selection/holdout. 2026 is fixed OOS.",
        f"- Audit: `{audit['status']}`",
        f"- Blocking: `{audit['blocking']}`",
        "",
        "## OOS Cost1",
        f"- PnL: `{c1['pnl']}`",
        f"- MDD: `{c1['mdd']}`",
        f"- Trades/day: `{c1['trades_per_day']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ai_audit = audit_ai_contracts()
    y2025, y2026, regime = _build(args)
    y2025.to_csv(args.out_dir / "features_2025.csv", index=False)
    y2026.to_csv(args.out_dir / "features_2026.csv", index=False)
    fit = y2025[y2025["timestamp"] < pd.Timestamp("2025-09-01")].copy()
    selection = y2025[(y2025["timestamp"] >= pd.Timestamp("2025-09-01")) & (y2025["timestamp"] < pd.Timestamp("2025-11-01"))].copy()
    holdout = y2025[y2025["timestamp"] >= pd.Timestamp("2025-11-01")].copy()
    cols = feature_cols([fit, selection, holdout, y2026], CLEAN_PREFIX)
    priority = [c for c in cols if c.startswith(("teacher_", CLEAN_PREFIX, "m7_", "ai_", "patchtst_", "tide_", "timesnet_", "dlinear_")) or c in {"pred_patchtst", "conf_patchtst"}]
    cols = (priority + [c for c in cols if c not in priority])[: int(args.max_features)]
    feature_audit = audit_frame_contract(y2025, feature_cols=cols, clean_prefix=CLEAN_PREFIX)
    if feature_audit["status"] != "pass":
        raise ValueError(json.dumps(feature_audit, ensure_ascii=False))

    print(f"[{MODEL_ID}] building long/short labels", flush=True)
    long_train = build_side_candidates(fit, cols, 1, fee=args.fee, slip=args.slip, label=True, row_stride=args.label_stride)
    short_train = build_side_candidates(fit, cols, -1, fee=args.fee, slip=args.slip, label=True, row_stride=args.label_stride)
    long_cols = model_cols(long_train)
    short_cols = model_cols(short_train)
    print(f"[{MODEL_ID}] fitting heads long={len(long_train)} short={len(short_train)}", flush=True)
    long_model = fit_side_ranker(long_train, long_cols, seed=1011, max_rows=args.max_train_rows)
    short_model = fit_side_ranker(short_train, short_cols, seed=2011, max_rows=args.max_train_rows)

    def pred(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (
            predict_side(long_model, build_side_candidates(frame, cols, 1, fee=args.fee, slip=args.slip, label=False)),
            predict_side(short_model, build_side_candidates(frame, cols, -1, fee=args.fee, slip=args.slip, label=False)),
        )

    sel_l, sel_s = pred(selection)
    hold_l, hold_s = pred(holdout)
    oos_l, oos_s = pred(y2026)
    grid = runtime_grid()
    if args.max_grid > 0:
        grid = grid[: args.max_grid]
    best = None
    best_score = -1e18
    best_sel = None
    rows = []
    for idx, cfg in enumerate(grid, start=1):
        print(f"[{MODEL_ID}] grid {idx}/{len(grid)}", flush=True)
        r1 = backtest(selection, sel_l, sel_s, cfg, fee=args.fee, slip=args.slip)
        r2 = backtest(selection, sel_l, sel_s, cfg, fee=args.fee * 2, slip=args.slip * 2)
        s = score(r1, r2, cfg.target_trades_day)
        rows.append({"score": s, **asdict(cfg), **{f"selection_{k}": v for k, v in _compact(r1).items()}, "selection_cost2_pnl": r2["pnl"]})
        if s > best_score:
            best, best_score, best_sel = cfg, s, r1
    if best is None:
        raise RuntimeError("no config selected")
    holdout_result = backtest(holdout, hold_l, hold_s, best, fee=args.fee, slip=args.slip)
    metrics = {}
    ledgers = {}
    for mult in (1, 2, 3):
        result = backtest(y2026, oos_l, oos_s, best, fee=args.fee * mult, slip=args.slip * mult)
        key = f"cost{mult}"
        metrics[key] = _compact(result)
        lp = args.report_out.with_name(args.report_out.stem + f"_{key}_ledger.csv")
        pd.DataFrame(result["ledger"]).to_csv(lp, index=False)
        ledgers[key] = str(lp)
    model_path = args.out_dir / "model.pkl"
    save_bundle(model_path, {"model_id": MODEL_ID, "regime": regime, "long_model": long_model, "short_model": short_model, "feature_cols": cols, "selected_config": asdict(best)})
    grid_path = args.report_out.with_name(args.report_out.stem + "_selection_grid.csv")
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(grid_path, index=False)
    report = {
        "model_id": MODEL_ID,
        "data": {
            "fit_range": [str(fit["timestamp"].iloc[0]), str(fit["timestamp"].iloc[-1])],
            "selection_range": [str(selection["timestamp"].iloc[0]), str(selection["timestamp"].iloc[-1])],
            "holdout_range": [str(holdout["timestamp"].iloc[0]), str(holdout["timestamp"].iloc[-1])],
            "oos_range": [str(y2026["timestamp"].iloc[0]), str(y2026["timestamp"].iloc[-1])],
        },
        "data_audit": {"ai_contracts": ai_audit, "feature_contract": feature_audit, "train_eval_overlap": _overlap(fit, y2026) + _overlap(selection, y2026) + _overlap(holdout, y2026)},
        "artifacts": {"model": str(model_path), "selection_grid": str(grid_path), "ledgers": ledgers},
        "feature_count": len(cols),
        "selected_config": asdict(best),
        "selection_score": best_score,
        "selection_result": _compact(best_sel),
        "holdout_result": _compact(holdout_result),
        "metrics": metrics,
    }
    blocking = []
    if ai_audit["status"] != "pass":
        blocking += ai_audit["blocking"]
    if feature_audit["status"] != "pass":
        blocking += feature_audit["blocking"]
    if report["data_audit"]["train_eval_overlap"] != 0:
        blocking.append("train_eval_timestamp_overlap")
    audit = {"model_id": MODEL_ID, "status": "pass" if not blocking else "fail", "blocking": blocking, "warnings": ["M7 embedded provenance caveat remains"], "invariants": {"long_short_heads_separated": True, "2026_fixed_oos_no_selection": True, "cost_1x_2x_3x_reported": True}}
    report["audit"] = audit
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    _write_contract(args.contract_out, report, audit)
    print(json.dumps({"status": audit["status"], "metrics": metrics, "report": str(args.report_out), "audit": str(args.audit_out)}, indent=2, ensure_ascii=False, default=_json_default))
    return 0 if audit["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
