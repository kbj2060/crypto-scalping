#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame, prepare_features
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2
from scripts import eval_alpha3_limit_close_fallback_20260514 as alpha3_close
from scripts import eval_alpha3_regime4_state24_v2_full_retrain_20260526 as alpha3_full
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27
from scripts import train_eval_hf_v13_jackpot_runner_v21_2 as v21
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read


MODEL_ID = "alpha3_regime4_state24_v2_sl_bucket_widen_20260527"
BASE_REPORT = Path("/home/llewyn/crypto-scalping/data/ensemble/reports/alpha3_regime4_state24_v2_full_retrain_20260526_summary.json")
GRID_OUT = Path(f"/home/llewyn/crypto-scalping/data/ensemble/reports/{MODEL_ID}_grid.csv")
REPORT_OUT = Path(f"/home/llewyn/crypto-scalping/data/ensemble/reports/{MODEL_ID}_summary.json")
AUDIT_OUT = Path(f"/home/llewyn/crypto-scalping/data/ensemble/reports/{MODEL_ID}_audit.json")


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _load_teacher(path: Path) -> tuple[Any, list[str], dict[str, Any], tuple[float, ...]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = alpha2._load_teacher_model(payload)
    cols = list(payload["feature_cols"])
    norm = dict(payload["train_meta"]["norm"])
    buckets = tuple(float(x) for x in payload["buckets"])
    return model, cols, norm, buckets


def _load_deep(path: Path) -> tuple[Any, list[str], dict[str, np.ndarray]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cols = list(payload["seq_cols"])
    norm = dict(payload["norm"])
    model = v27.DeepAlphaTCN(len(cols))
    model.load_state_dict(payload["state_dict"])
    return model.cpu().eval(), cols, norm


def _scaled_stop_loss_buckets(base: tuple[float, ...], scales: tuple[float, ...], cap: float) -> tuple[float, ...]:
    out = []
    for v, s in zip(base, scales):
        out.append(min(float(cap), float(v) * float(s)))
    return tuple(out)


def _variants(base: tuple[float, ...]) -> list[dict[str, Any]]:
    n = len(base)
    ones = tuple(1.0 for _ in range(n))
    return [
        {"name": "baseline", "scales": ones, "cap": max(base)},
        {"name": "slx1p15_all_cap090", "scales": tuple(1.15 for _ in range(n)), "cap": 0.090},
        {"name": "slx1p30_all_cap090", "scales": tuple(1.30 for _ in range(n)), "cap": 0.090},
        {"name": "sl_front_open_cap090", "scales": (1.45, 1.45, 1.35, 1.30, 1.20, 1.15, 1.10), "cap": 0.090},
        {"name": "sl_mid_open_cap090", "scales": (1.15, 1.20, 1.30, 1.30, 1.20, 1.15, 1.10), "cap": 0.090},
    ]


def _merge_state24(df: pd.DataFrame, sidecar_path: Path) -> pd.DataFrame:
    side = alpha3_full._rename_state24_sidecar(_read(sidecar_path))
    merged, _ = alpha3_full._merge_state24(df, side)
    return merged


def _metrics(
    df: pd.DataFrame,
    parent: dict[str, Any],
    runner: dict[str, Any],
    add_cfg: v21.CostRunnerConfig,
    q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    limit_cfg: Any,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    return alpha3_close._metrics_signal_limit_close(
        df,
        parent,
        runner,
        add_cfg,
        q,
        decisions,
        overlay,
        limit_cfg,
        fee=fee,
        slip=slip,
    )


def main() -> int:
    base = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    exp = dict(base["experiments"][-1])
    artifacts = dict(exp["artifacts"])

    parent_base = joblib.load(artifacts["parent"])
    runner_payload = joblib.load(artifacts["runner"])
    runner_model = runner_payload["cost_runner"]
    teacher_rt = alpha2.Alpha2Runtime(**dict(exp["selected_teacher_runtime"]))
    add_cfg = v21.CostRunnerConfig(**dict(exp["selected_runner_config"]))
    overlay = v31.OverlayConfig(**dict(exp["selected_overlay"]))
    limit_cfg = alpha3_full._canonical_limit_cfg()
    fee = float(parent_base["config"]["fee"])
    slip = float(parent_base["config"]["slip"])

    teacher_model, teacher_cols, teacher_norm, teacher_buckets = _load_teacher(Path(artifacts["teacher"]))
    deep_model, deep_cols, deep_norm = _load_deep(Path(artifacts["deep_scout"]))

    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_all = _merge_state24(train_all, alpha3_full.SIDE_CLEAN4_2025)
    eval_df = _merge_state24(eval_df, alpha3_full.SIDE_CLEAN4_2026)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)

    val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=teacher_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=teacher_cols)
    val_teacher = alpha2.teacher._predict_deep(teacher_model, val_features, teacher_cols, teacher_norm)
    eval_teacher = alpha2.teacher._predict_deep(teacher_model, eval_features, teacher_cols, teacher_norm)
    val_q = v27._predict_all(deep_model, val_df, deep_cols, deep_norm)
    eval_q = v27._predict_all(deep_model, eval_df, deep_cols, deep_norm)

    base_sl = tuple(float(x) for x in parent_base["config"]["stop_loss_buckets"])
    rows: list[dict[str, Any]] = []
    best_name = ""
    best_val_score = -1e18
    best_oos: dict[str, Any] | None = None

    for v in _variants(base_sl):
        parent = dict(parent_base)
        cfg = dict(parent["config"])
        cfg["stop_loss_buckets"] = _scaled_stop_loss_buckets(base_sl, tuple(v["scales"]), float(v["cap"]))
        parent["config"] = cfg

        val_parent = predict_policy_frame(parent, val_df, close=_close(val_df))
        eval_parent = predict_policy_frame(parent, eval_df, close=_close(eval_df))
        val_dec = alpha2._decisions(val_parent, val_teacher, teacher_buckets, teacher_rt)
        eval_dec = alpha2._decisions(eval_parent, eval_teacher, teacher_buckets, teacher_rt)

        val_metrics = _metrics(val_df, parent, runner_model, add_cfg, val_q, val_dec, overlay, limit_cfg, fee=fee, slip=slip)
        oos_metrics = _metrics(eval_df, parent, runner_model, add_cfg, eval_q, eval_dec, overlay, limit_cfg, fee=fee, slip=slip)
        val_score = _score(val_metrics)
        oos_score = _score(oos_metrics)
        c3 = oos_metrics["cost3"]
        row = {
            "variant": v["name"],
            "stop_loss_buckets": json.dumps(cfg["stop_loss_buckets"]),
            "val_score": float(val_score),
            "val_cost1_pnl": float(val_metrics["cost1"]["pnl"]),
            "val_cost1_mdd": float(val_metrics["cost1"]["mdd"]),
            "val_cost3_pnl": float(val_metrics["cost3"]["pnl"]),
            "oos_score": float(oos_score),
            "oos_cost3_pnl": float(c3["pnl"]),
            "oos_cost3_mdd": float(c3["mdd"]),
            "oos_cost3_wr": float(c3["wr"]),
            "oos_cost3_trades": int(c3["trades"]),
            "oos_cost3_v21_sl": int(c3["exits"].get("v21_2_stop_loss", 0)),
            "oos_cost3_deep_sl": int(c3["exits"].get("deep_alpha_stop_loss", 0)),
        }
        rows.append(row)
        if float(val_score) > best_val_score:
            best_val_score = float(val_score)
            best_name = str(v["name"])
            best_oos = {
                "variant": str(v["name"]),
                "stop_loss_buckets": tuple(float(x) for x in cfg["stop_loss_buckets"]),
                "metrics": oos_metrics,
                "score": float(oos_score),
                "selection_val_score": float(val_score),
            }

    if best_oos is None:
        raise RuntimeError("no variant evaluated")

    grid = pd.DataFrame(rows).sort_values("val_score", ascending=False).reset_index(drop=True)
    grid.to_csv(GRID_OUT, index=False)
    baseline_row = next(r for r in rows if r["variant"] == "baseline")
    best_row = next(r for r in rows if r["variant"] == best_name)
    delta = {
        "cost3_pnl": float(best_row["oos_cost3_pnl"] - baseline_row["oos_cost3_pnl"]),
        "cost3_mdd": float(best_row["oos_cost3_mdd"] - baseline_row["oos_cost3_mdd"]),
        "cost3_wr": float(best_row["oos_cost3_wr"] - baseline_row["oos_cost3_wr"]),
        "cost3_trades": int(best_row["oos_cost3_trades"] - baseline_row["oos_cost3_trades"]),
        "v21_stop_loss": int(best_row["oos_cost3_v21_sl"] - baseline_row["oos_cost3_v21_sl"]),
        "deep_stop_loss": int(best_row["oos_cost3_deep_sl"] - baseline_row["oos_cost3_deep_sl"]),
    }
    audit = {
        "status": "pass",
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS",
        "base_report": str(BASE_REPORT),
        "selected_variant": best_name,
        "baseline_variant": "baseline",
        "delta_vs_baseline": delta,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha3 state24_v2 candidate stop-loss bucket widening ablation. Parent stop_loss_buckets are widened by predefined scales, while teacher/runtime, v21 runner, deep scout, execution, and overlay remain fixed.",
        "selected_variant": best_oos,
        "baseline_oos": baseline_row,
        "grid": str(GRID_OUT),
        "audit": audit,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT), "selected_variant": best_name}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
