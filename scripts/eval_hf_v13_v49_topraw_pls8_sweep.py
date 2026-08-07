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
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_conservative_limit_sniper_v46 as v46  # noqa: E402
from scripts import train_eval_hf_v13_frozen_v27_offline_rl_exit_overlay_v33 as v33  # noqa: E402
from scripts.eval_hf_v13_v31_rl_surrounding_v49_v50_v51 import (  # noqa: E402
    DEFAULT_EVAL,
    DEFAULT_JACKPOT,
    DEFAULT_PARENT,
    DEFAULT_PARENT_REPORT,
    DEFAULT_TRAIN,
    DEFAULT_V27,
    _feature_audit,
    _load_pickle,
    _patch_v33_state,
    _run_v49_exit_rl,
)
from scripts.eval_hf_v13_v49_pls_dim_sweep import _build_pls_frames, _factor_cols  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_v49_topraw_pls8_sweep_20260513"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v49_topraw_pls8_sweep_20260513"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v49_topraw_pls8_sweep_20260513_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v49_topraw_pls8_sweep_20260513_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v49_topraw_pls8_sweep_20260513_grid.csv"
DEFAULT_RANKING = ROOT / "data/ensemble/reports/hf_v13_v49_topraw_pls8_sweep_20260513_feature_ranking.csv"

RAW_KEYS = (
    "vol",
    "bb_",
    "garch",
    "parkinson",
    "rogers",
    "taker",
    "ofi",
    "trade",
    "liquid",
    "amihud",
    "smart",
    "squeeze",
    "breakout",
    "whale",
    "m7_",
    "ai_",
    "patchtst",
    "tide",
    "timesnet",
    "dlinear",
    "funding",
    "oi_",
    "flow",
)
FORBIDDEN = (
    "future",
    "target",
    "label",
    "realized",
    "cash_after",
    "regime_v2",
    "legacy",
    "hdb",
    "hmm",
    "clean_regime_",
)


def _candidate_raw_cols(train_all: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    common = set(train_all.columns) & set(eval_df.columns)
    cols: list[str] = []
    for col in train_all.columns:
        name = col.lower()
        if col == "timestamp" or col not in common:
            continue
        if any(tok in name for tok in FORBIDDEN):
            continue
        if not any(tok in name for tok in RAW_KEYS):
            continue
        if pd.api.types.is_numeric_dtype(train_all[col]) or pd.api.types.is_numeric_dtype(eval_df[col]):
            cols.append(col)
    return cols


def _rank_raw_features(
    *,
    train: pd.DataFrame,
    train_dec: pd.DataFrame,
    train_q: np.ndarray,
    raw_cols: list[str],
    fee: float,
    slip: float,
    seed: int,
) -> tuple[pd.DataFrame, int, float]:
    base_cfg = v33.OverlayConfig("v49_rank_base", 0.010, 0.004, 1.2, 12, 0.60, 2, 0.045, 0.022, 48)
    with _patch_v33_state(raw_cols) as state_cols:
        x_train, y_train = v33._collect_reversal_training(train, train_dec, train_q, base_cfg, fee=fee, slip=slip)
    feat_cols = [f"feat__{c}" for c in raw_cols]
    x = x_train.loc[:, feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    x = SimpleImputer(strategy="median").fit_transform(x)
    y = np.asarray(y_train, dtype=np.int64)
    mi = mutual_info_classif(x, y, discrete_features=False, random_state=int(seed))
    corr = []
    y_float = y.astype(np.float64)
    y_std = float(np.std(y_float)) or 1.0
    for j in range(x.shape[1]):
        col = x[:, j]
        cstd = float(np.std(col))
        if cstd <= 1e-12:
            corr.append(0.0)
        else:
            corr.append(abs(float(np.mean((col - np.mean(col)) * (y_float - np.mean(y_float))) / (cstd * y_std))))
    out = pd.DataFrame({"feature": raw_cols, "mutual_info": mi, "abs_corr": corr})
    out["rank_score"] = out["mutual_info"].rank(pct=True) + out["abs_corr"].rank(pct=True)
    out = out.sort_values(["rank_score", "mutual_info", "abs_corr"], ascending=False).reset_index(drop=True)
    return out, int(len(y)), float(np.mean(y))


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.40 * c2["pnl"] + 0.24 * c3["pnl"] - 0.25 * abs(c1["mdd"]))


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
    p = argparse.ArgumentParser(description="V49 Exit RL top raw micro/vol/AI + PLS8x8 sweep.")
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
    p.add_argument("--ranking-out", type=Path, default=DEFAULT_RANKING)
    p.add_argument("--top-k", type=str, default="20,30,40")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=2049)
    p.add_argument("--train-stride", type=int, default=48)
    p.add_argument("--embed-batch", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    top_ks = [int(x.strip()) for x in args.top_k.split(",") if x.strip()]
    print(f"[{MODEL_ID}] loading frozen V31 stack top_k={top_ks} epochs={args.epochs} seed={args.seed}", flush=True)
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

    overlay = v46._base_overlay()
    baseline = {
        f"cost{mult}": v31.backtest(
            eval_df,
            bundle,
            jackpot_model,
            add_cfg,
            eval_q,
            overlay,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
            decisions=eval_dec,
        )
        for mult in (1, 2, 3)
    }

    raw_candidates = _candidate_raw_cols(train_all, eval_df)
    print(f"[{MODEL_ID}] ranking raw candidates={len(raw_candidates)}", flush=True)
    ranking, rank_rows, close_rate = _rank_raw_features(
        train=train,
        train_dec=train_dec,
        train_q=train_q,
        raw_cols=raw_candidates,
        fee=fee,
        slip=slip,
        seed=int(args.seed),
    )
    ranking.to_csv(args.ranking_out, index=False)

    with args.parent_report.open(encoding="utf-8") as f:
        parent_report = json.load(f)
    print(f"[{MODEL_ID}] building PLS 8x8 frames", flush=True)
    tr_pls, va_pls, ev_pls, encoding_meta = _build_pls_frames(
        args=args,
        parent_report=parent_report,
        train_all=train_all,
        eval_df=eval_df,
        macro_dim=8,
        micro_dim=8,
    )
    pls_cols = _factor_cols(tr_pls)
    if len(pls_cols) != 16:
        raise RuntimeError(f"PLS 8x8 factor mismatch: {len(pls_cols)}")

    parent_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    results: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    for k in top_ks:
        selected_raw = ranking["feature"].head(k).tolist()
        feature_cols = selected_raw + pls_cols
        mode = f"topraw{k}_pls8x8"
        audit = _feature_audit(feature_cols, pd.concat([tr_pls, va_pls], ignore_index=True), ev_pls)
        if audit["status"] != "pass":
            raise RuntimeError(f"{mode} feature audit failed: {audit}")
        print(f"[{MODEL_ID}] running {mode} features={len(feature_cols)}", flush=True)
        result = _run_v49_exit_rl(
            mode=mode,
            train=tr_pls,
            val=va_pls,
            eval_df=ev_pls,
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
            seed=int(args.seed),
        )
        result["selected_raw_features"] = selected_raw
        result["pls_factor_cols"] = pls_cols
        result["feature_audit"] = audit
        results.append(result)
        grid_rows.extend(result["grid"])
        print(
            f"[{MODEL_ID}] {mode} cost1={result['metrics']['cost1']['pnl']:.2f} "
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
        "ranking_rows": int(rank_rows),
        "ranking_close_rate": float(close_rate),
        "raw_candidate_count": int(len(raw_candidates)),
        "best": {
            "feature_mode": best["feature_mode"],
            "metrics": best["metrics"],
            "selected_config": best["selected_config"],
            "model": best["artifacts"]["model"],
        },
    }
    report = {
        "model_id": MODEL_ID,
        "design": "V49 Exit RL feature sweep using ranked top raw micro/vol/AI features plus Chronos/Kairos target-aware PLS 8x8 factors. Frozen V31 parent, V27 deep scout, and V21.2 jackpot runner are preserved.",
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "top_ks": top_ks,
        "ranking_method": "mutual_info_classif + abs correlation on V49 offline RL close/hold labels",
        "baseline_recomputed_v31": baseline,
        "results": results,
        "best": {
            "feature_mode": best["feature_mode"],
            "metrics": best["metrics"],
            "selected_config": best["selected_config"],
            "model": best["artifacts"]["model"],
        },
        "audit": audit,
        "encoding_meta": encoding_meta,
        "artifacts": {
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "grid": str(args.grid_out),
            "feature_ranking": str(args.ranking_out),
            "out_dir": str(args.out_dir),
        },
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "best": report["best"], "verdict": verdict}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
