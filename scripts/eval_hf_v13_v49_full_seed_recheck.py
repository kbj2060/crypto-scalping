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

from ensemble.fully_learned_governor_policy import predict_policy_frame  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_conservative_limit_sniper_v46 as v46  # noqa: E402
from scripts.eval_hf_v13_v31_rl_surrounding_v49_v50_v51 import (  # noqa: E402
    DEFAULT_EVAL,
    DEFAULT_JACKPOT,
    DEFAULT_PARENT,
    DEFAULT_TRAIN,
    DEFAULT_V27,
    _feature_audit,
    _load_pickle,
    _numeric_cols,
    _run_v49_exit_rl,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_v49_raw_all_full_seed_recheck_20260513"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v49_raw_all_full_seed_recheck_20260513"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v49_raw_all_full_seed_recheck_20260513_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v49_raw_all_full_seed_recheck_20260513_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v49_raw_all_full_seed_recheck_20260513_grid.csv"


def _mean_std(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "min": float(arr.min()), "max": float(arr.max())}


def _flatten_grid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        cfg = dict(row["config"])
        out.append(
            {
                "seed": row.get("seed"),
                "feature_mode": row.get("feature_mode"),
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
    p = argparse.ArgumentParser(description="Full epoch/seed recheck for V49 raw-all exit RL.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seeds", type=str, default="2049,2050,2051")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    print(f"[{MODEL_ID}] loading frozen V31 stack seeds={seeds} epochs={args.epochs}", flush=True)
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
    feature_cols = _numeric_cols(train_all, eval_df)
    feature_audit = _feature_audit(feature_cols, train_all, eval_df)
    parent_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    if feature_audit["status"] != "pass":
        raise RuntimeError(f"feature audit failed: {feature_audit}")

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

    results: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"[{MODEL_ID}] running V49 raw-all seed={seed}", flush=True)
        result = _run_v49_exit_rl(
            mode="raw_all_full",
            train=train,
            val=val,
            eval_df=eval_df,
            bundle=bundle,
            jackpot_model=jackpot_model,
            add_cfg=add_cfg,
            train_q=train_q,
            val_q=val_q,
            eval_q=eval_q,
            train_dec=train_dec,
            val_dec=val_dec,
            eval_dec=eval_dec,
            feature_cols=feature_cols,
            fee=fee,
            slip=slip,
            out_dir=args.out_dir,
            report_out=args.report_out,
            epochs=int(args.epochs),
            seed=int(seed),
        )
        results.append(result)
        grid_rows.extend(result["grid"])
        print(
            f"[{MODEL_ID}] seed={seed} cost1={result['metrics']['cost1']['pnl']:.2f} "
            f"cost2={result['metrics']['cost2']['pnl']:.2f} cost3={result['metrics']['cost3']['pnl']:.2f}",
            flush=True,
        )

    pd.DataFrame(_flatten_grid(grid_rows)).to_csv(args.grid_out, index=False)
    best = max(results, key=lambda r: float(r["metrics"]["cost1"]["pnl"]))
    pnl_stats = {f"cost{m}": _mean_std([float(r["metrics"][f"cost{m}"]["pnl"]) for r in results]) for m in (1, 2, 3)}
    mdd_stats = {f"cost{m}": _mean_std([float(r["metrics"][f"cost{m}"]["mdd"]) for r in results]) for m in (1, 2, 3)}
    blocking: list[str] = []
    warnings: list[str] = []
    if parent_audit["status"] != "pass":
        blocking.extend(parent_audit.get("blocking", []))
    warnings.extend(parent_audit.get("warnings", []))
    for m in (1, 2, 3):
        if pnl_stats[f"cost{m}"]["min"] <= 0.0:
            warnings.append(f"seed_min_cost{m}_not_survived")
    if pnl_stats["cost1"]["mean"] <= float(baseline["cost1"]["pnl"]):
        warnings.append("mean_cost1_did_not_beat_v31")
    if pnl_stats["cost2"]["mean"] <= float(baseline["cost2"]["pnl"]):
        warnings.append("mean_cost2_did_not_beat_v31")
    if pnl_stats["cost3"]["mean"] <= float(baseline["cost3"]["pnl"]):
        warnings.append("mean_cost3_did_not_beat_v31")
    seed_consistent = (
        pnl_stats["cost1"]["min"] > float(baseline["cost1"]["pnl"])
        and pnl_stats["cost2"]["min"] > float(baseline["cost2"]["pnl"])
        and pnl_stats["cost3"]["min"] > float(baseline["cost3"]["pnl"])
    )
    verdict = "promote_to_injection_audit" if not blocking and seed_consistent else "iterate"
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS after seed/config selection",
        "seed_consistent_beats_v31_all_costs": bool(seed_consistent),
        "feature_audit": feature_audit,
        "parent_contract_audit": parent_audit,
        "baseline_recomputed_v31": baseline,
        "pnl_stats": pnl_stats,
        "mdd_stats": mdd_stats,
        "best": {"seed": best["seed"], "metrics": best["metrics"], "selected_config": best["selected_config"], "model": best["artifacts"]["model"]},
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Full epoch multi-seed recheck of V49 raw-all. Frozen V31 parent/V27/V21.2 stack is preserved; only the V27 deep_alpha exit overlay is replaced by a discrete hold/close policy trained on raw-all non-forbidden features plus live position state.",
        "epochs": int(args.epochs),
        "seeds": seeds,
        "baseline_recomputed_v31": baseline,
        "results": results,
        "pnl_stats": pnl_stats,
        "mdd_stats": mdd_stats,
        "best": {"seed": best["seed"], "metrics": best["metrics"], "selected_config": best["selected_config"], "model": best["artifacts"]["model"]},
        "audit": audit,
        "artifacts": {"report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "out_dir": str(args.out_dir)},
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "best": report["best"], "verdict": verdict}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
