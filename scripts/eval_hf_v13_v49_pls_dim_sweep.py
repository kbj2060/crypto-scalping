#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from ensemble.fully_learned_governor_policy import build_training_set, predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_conservative_limit_sniper_v46 as v46  # noqa: E402
from scripts import eval_hf_v13_v40_6_full_v31_stack_retrain as v40_6  # noqa: E402
from scripts.eval_hf_v13_v31_rl_surrounding_v49_v50_v51 import (  # noqa: E402
    DEFAULT_EVAL,
    DEFAULT_JACKPOT,
    DEFAULT_PARENT,
    DEFAULT_PARENT_REPORT,
    DEFAULT_TRAIN,
    DEFAULT_V27,
    _feature_audit,
    _load_pickle,
    _run_v49_exit_rl,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_v49_pls_dim_sweep_20260513"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v49_pls_dim_sweep_20260513"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v49_pls_dim_sweep_20260513_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v49_pls_dim_sweep_20260513_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v49_pls_dim_sweep_20260513_grid.csv"


def _projection_targets(y: dict[str, np.ndarray]) -> np.ndarray:
    return v40_6._projection_targets(y)


def _build_pls_frames(
    *,
    args: argparse.Namespace,
    parent_report: dict[str, Any],
    train_all: pd.DataFrame,
    eval_df: pd.DataFrame,
    macro_dim: int,
    micro_dim: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_ts = pd.Timestamp("2025-10-01")
    train = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    report = json.loads(json.dumps(parent_report))
    report.setdefault("comparison", {})
    report["comparison"]["selected_projection_spec"] = {
        "macro_dim": int(macro_dim),
        "micro_dim": int(micro_dim),
        "drop_raw_micro": True,
    }
    feature_cols = _feature_cols(train_all, eval_df)
    cfg = v40_6._parent_cfg()
    x_train, y, _ = build_training_set(train, cfg=cfg, stride_bars=int(args.train_stride), batch_size=512, feature_cols=feature_cols)
    train_idx_sample = np.arange(0, max(0, len(train) - cfg.max_train_horizon_bars - 1), max(1, int(args.train_stride)), dtype=np.int64)
    if len(train_idx_sample) != len(x_train):
        raise RuntimeError(f"train sample mismatch for PLS projection: {len(train_idx_sample)} vs {len(x_train)}")
    proj_targets = _projection_targets(y)
    train_feat = prepare_features(train, side_hint=0, close=_close(train), feature_cols=feature_cols)
    val_feat = prepare_features(val, side_hint=0, close=_close(val), feature_cols=feature_cols)
    eval_feat = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    train_full, val_full, eval_full, meta = v40_6._build_v40_6_frames(
        args=args,
        parent_report=report,
        train_df=train,
        val_df=val,
        eval_df=eval_df,
        train_feat=train_feat,
        val_feat=val_feat,
        eval_feat=eval_feat,
        train_idx_sample=train_idx_sample,
        proj_targets=proj_targets,
    )
    meta["override_projection_spec"] = {"macro_dim": int(macro_dim), "micro_dim": int(micro_dim), "drop_raw_micro": True}
    return train_full, val_full, eval_full, meta


def _factor_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("macro_factor_") or c.startswith("micro_factor_")]


def _flatten_grid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        cfg = dict(row["config"])
        out.append(
            {
                "variant": row.get("feature_mode"),
                "seed": row.get("seed"),
                **{f"cfg_{k}": v for k, v in cfg.items()},
                "selection_score": row["selection_score"],
                "val_cost1_pnl": row["validation_cost1"]["pnl"],
                "val_cost1_mdd": row["validation_cost1"]["mdd"],
                "val_cost1_trades": row["validation_cost1"]["trades"],
                "val_cost2_pnl": row["validation_cost2"]["pnl"],
                "val_cost3_pnl": row["validation_cost3"]["pnl"],
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V49 Exit RL PLS-factor-only dimension sweep.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--parent-report", type=Path, default=DEFAULT_PARENT_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--dims", type=str, default="1x1,2x2,4x4,8x8")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=2049)
    p.add_argument("--train-stride", type=int, default=48)
    p.add_argument("--embed-batch", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    dim_specs: list[tuple[int, int]] = []
    for spec in args.dims.split(","):
        left, right = spec.lower().split("x", 1)
        dim_specs.append((int(left), int(right)))
    print(f"[{MODEL_ID}] loading frozen V31 stack dims={dim_specs} epochs={args.epochs} seed={args.seed}", flush=True)
    bundle = _load_pickle(args.parent_model)
    jackpot_payload = _load_pickle(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(args.v27_model)
    base = dict(bundle["config"])
    fee = float(base["fee"])
    slip = float(base["slip"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    train = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)

    print(f"[{MODEL_ID}] predicting parent decisions and V27 utilities once", flush=True)
    train_dec = predict_policy_frame(bundle, train, close=_close(train))
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    train_q = v31._predict_all(v27_model, train, v27_payload["seq_cols"], v27_payload["norm"])
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    baseline: dict[str, Any] = {}
    overlay = v46._base_overlay()
    for mult in (1, 2, 3):
        baseline[f"cost{mult}"] = v31.backtest(eval_df, bundle, jackpot_model, add_cfg, eval_q, overlay, fee=fee, slip=slip, cost_mult=float(mult), decisions=eval_dec)

    with args.parent_report.open(encoding="utf-8") as f:
        parent_report = json.load(f)
    parent_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))

    results: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    encoding_meta: dict[str, Any] = {}
    for macro_dim, micro_dim in dim_specs:
        variant = f"pls_m{macro_dim}_k{micro_dim}_only"
        print(f"[{MODEL_ID}] building {variant}", flush=True)
        tr, va, ev, meta = _build_pls_frames(
            args=args,
            parent_report=parent_report,
            train_all=train_all,
            eval_df=eval_df,
            macro_dim=macro_dim,
            micro_dim=micro_dim,
        )
        cols = _factor_cols(tr)
        if len(cols) != macro_dim + micro_dim:
            raise RuntimeError(f"{variant} factor col mismatch: {len(cols)} vs {macro_dim + micro_dim}")
        audit = _feature_audit(cols, pd.concat([tr, va], ignore_index=True), ev)
        if audit["status"] != "pass":
            raise RuntimeError(f"{variant} audit failed: {audit}")
        result = _run_v49_exit_rl(
            mode=variant,
            train=tr,
            val=va,
            eval_df=ev,
            bundle=bundle,
            jackpot_model=jackpot_model,
            add_cfg=add_cfg,
            train_q=train_q,
            val_q=val_q,
            eval_q=eval_q,
            train_dec=train_dec,
            val_dec=val_dec,
            eval_dec=eval_dec,
            feature_cols=cols,
            fee=fee,
            slip=slip,
            out_dir=args.out_dir,
            report_out=args.report_out,
            epochs=int(args.epochs),
            seed=int(args.seed),
        )
        result["pls_spec"] = {"macro_dim": macro_dim, "micro_dim": micro_dim, "factor_cols": cols}
        result["feature_audit"] = audit
        results.append(result)
        grid_rows.extend(result["grid"])
        encoding_meta[variant] = meta
        print(
            f"[{MODEL_ID}] {variant} cost1={result['metrics']['cost1']['pnl']:.2f} "
            f"cost2={result['metrics']['cost2']['pnl']:.2f} cost3={result['metrics']['cost3']['pnl']:.2f}",
            flush=True,
        )

    pd.DataFrame(_flatten_grid(grid_rows)).to_csv(args.grid_out, index=False)
    best = max(results, key=lambda r: float(r["metrics"]["cost1"]["pnl"]))
    blocking: list[str] = []
    warnings: list[str] = []
    if parent_audit["status"] != "pass":
        blocking.extend(parent_audit.get("blocking", []))
    warnings.extend(parent_audit.get("warnings", []))
    if best["metrics"]["cost1"]["pnl"] <= baseline["cost1"]["pnl"]:
        warnings.append("best_cost1_did_not_beat_v31")
    if best["metrics"]["cost2"]["pnl"] <= baseline["cost2"]["pnl"]:
        warnings.append("best_cost2_did_not_beat_v31")
    if best["metrics"]["cost3"]["pnl"] <= baseline["cost3"]["pnl"]:
        warnings.append("best_cost3_did_not_beat_v31")
    verdict = (
        "candidate_recheck"
        if not blocking
        and best["metrics"]["cost1"]["pnl"] > baseline["cost1"]["pnl"]
        and best["metrics"]["cost2"]["pnl"] > baseline["cost2"]["pnl"]
        and best["metrics"]["cost3"]["pnl"] > baseline["cost3"]["pnl"]
        else "iterate"
    )
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS after validation selection",
        "baseline_recomputed_v31": baseline,
        "parent_contract_audit": parent_audit,
        "best": {"feature_mode": best["feature_mode"], "metrics": best["metrics"], "selected_config": best["selected_config"], "model": best["artifacts"]["model"]},
    }
    report = {
        "model_id": MODEL_ID,
        "design": "V49 Exit RL standalone PLS factor sweep. Raw market feature columns are excluded from the overlay feature list; only Chronos/Kairos target-aware PLS factors are added to the mandatory V49 position/deep state.",
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "dim_specs": [{"macro_dim": m, "micro_dim": k} for m, k in dim_specs],
        "baseline_recomputed_v31": baseline,
        "results": results,
        "best": {"feature_mode": best["feature_mode"], "metrics": best["metrics"], "selected_config": best["selected_config"], "model": best["artifacts"]["model"]},
        "audit": audit,
        "encoding_meta": encoding_meta,
        "artifacts": {"report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "out_dir": str(args.out_dir)},
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "best": report["best"], "verdict": verdict}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
