#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame, prepare_features
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2
from scripts import eval_alpha3_limit_close_fallback_20260514 as alpha3_close
from scripts import eval_alpha3_regime4_state24_v2_full_retrain_20260526 as alpha3_full
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27
from scripts import train_eval_hf_v13_jackpot_runner_v21_2 as v21
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read


MODEL_ID = "alpha3_regime4_state24_v2_entry_gate_20260527"
BASE_REPORT = Path("/home/llewyn/crypto-scalping/data/ensemble/reports/alpha3_regime4_state24_v2_full_retrain_20260526_summary.json")
SL_REPORT = Path("/home/llewyn/crypto-scalping/data/ensemble/reports/alpha3_regime4_state24_v2_sl_bucket_widen_20260527_summary.json")
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


def _suppress_row(out: pd.DataFrame, i: int) -> None:
    out.at[i, "action"] = ACTION_CASH
    out.at[i, "side"] = 0
    out.at[i, "notional_exposure"] = 0.0
    out.at[i, "position_fraction"] = 0.0
    out.at[i, "take_profit"] = 0.0
    out.at[i, "stop_loss"] = 0.0
    out.at[i, "max_hold_bars"] = 0
    out.at[i, "cooldown_bars"] = 0
    out.at[i, "leverage"] = 1.0


def _apply_entry_gate(
    decisions: pd.DataFrame,
    *,
    quality_min: float,
    confidence_min: float,
    same_side_cooldown: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = decisions.copy()
    blocked_quality = 0
    blocked_conf = 0
    blocked_cooldown = 0
    accepted = 0
    last_side = 0
    last_signal_idx = -10**9
    for i, row in out.iterrows():
        action = int(row.get("action", 0))
        side = int(row.get("side", 0))
        if action == ACTION_CASH or side == 0:
            continue
        q = float(row.get("quality_score", 0.0))
        c = float(row.get("confidence", 0.0))
        if q < float(quality_min):
            _suppress_row(out, int(i))
            blocked_quality += 1
            continue
        if c < float(confidence_min):
            _suppress_row(out, int(i))
            blocked_conf += 1
            continue
        if int(same_side_cooldown) > 0 and side == last_side and (int(i) - int(last_signal_idx)) <= int(same_side_cooldown):
            _suppress_row(out, int(i))
            blocked_cooldown += 1
            continue
        accepted += 1
        last_signal_idx = int(i)
        last_side = int(side)
    return out, {
        "accepted": int(accepted),
        "blocked_quality": int(blocked_quality),
        "blocked_confidence": int(blocked_conf),
        "blocked_cooldown": int(blocked_cooldown),
    }


def _gate_grid() -> list[dict[str, Any]]:
    return [
        {"name": "baseline_no_gate", "quality_min": -999.0, "confidence_min": 0.0, "same_side_cooldown": 0},
        {"name": "cd1_only", "quality_min": -999.0, "confidence_min": 0.0, "same_side_cooldown": 1},
        {"name": "cd2_only", "quality_min": -999.0, "confidence_min": 0.0, "same_side_cooldown": 2},
        {"name": "cd3_only", "quality_min": -999.0, "confidence_min": 0.0, "same_side_cooldown": 3},
        {"name": "cd4_only", "quality_min": -999.0, "confidence_min": 0.0, "same_side_cooldown": 4},
        {"name": "qmin_0p0015", "quality_min": 0.0015, "confidence_min": 0.0, "same_side_cooldown": 0},
        {"name": "qmin_0p0020", "quality_min": 0.0020, "confidence_min": 0.0, "same_side_cooldown": 0},
        {"name": "qmin_0p0020_cd2", "quality_min": 0.0020, "confidence_min": 0.0, "same_side_cooldown": 2},
        {"name": "qmin_0p0020_cd3", "quality_min": 0.0020, "confidence_min": 0.0, "same_side_cooldown": 3},
        {"name": "qmin_0p0020_cd6", "quality_min": 0.0020, "confidence_min": 0.0, "same_side_cooldown": 6},
        {"name": "qmin_0p0025_cd6", "quality_min": 0.0025, "confidence_min": 0.0, "same_side_cooldown": 6},
        {"name": "qmin_0p0025_cd12_conf058", "quality_min": 0.0025, "confidence_min": 0.58, "same_side_cooldown": 12},
        {"name": "qmin_0p0030_cd12_conf060", "quality_min": 0.0030, "confidence_min": 0.60, "same_side_cooldown": 12},
    ]


def main() -> int:
    base = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    sl = json.loads(SL_REPORT.read_text(encoding="utf-8"))
    artifacts = dict(base["experiments"][-1]["artifacts"])

    parent_base = joblib.load(artifacts["parent"])
    sl_buckets = tuple(float(x) for x in sl["selected_variant"]["stop_loss_buckets"])
    parent = dict(parent_base)
    cfg = dict(parent["config"])
    cfg["stop_loss_buckets"] = sl_buckets
    parent["config"] = cfg

    runner_payload = joblib.load(artifacts["runner"])
    runner_model = runner_payload["cost_runner"]
    teacher_rt = alpha2.Alpha2Runtime(**dict(base["experiments"][-1]["selected_teacher_runtime"]))
    add_cfg = v21.CostRunnerConfig(**dict(base["experiments"][-1]["selected_runner_config"]))
    overlay = v31.OverlayConfig(**dict(base["experiments"][-1]["selected_overlay"]))
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

    val_parent = predict_policy_frame(parent, val_df, close=_close(val_df))
    eval_parent = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    val_dec_base = alpha2._decisions(val_parent, val_teacher, teacher_buckets, teacher_rt)
    eval_dec_base = alpha2._decisions(eval_parent, eval_teacher, teacher_buckets, teacher_rt)

    rows: list[dict[str, Any]] = []
    best_name = ""
    best_val_score = -1e18
    best_oos: dict[str, Any] | None = None

    for g in _gate_grid():
        val_dec, val_gate_stats = _apply_entry_gate(
            val_dec_base,
            quality_min=float(g["quality_min"]),
            confidence_min=float(g["confidence_min"]),
            same_side_cooldown=int(g["same_side_cooldown"]),
        )
        eval_dec, eval_gate_stats = _apply_entry_gate(
            eval_dec_base,
            quality_min=float(g["quality_min"]),
            confidence_min=float(g["confidence_min"]),
            same_side_cooldown=int(g["same_side_cooldown"]),
        )
        val_metrics = _metrics(val_df, parent, runner_model, add_cfg, val_q, val_dec, overlay, limit_cfg, fee=fee, slip=slip)
        oos_metrics = _metrics(eval_df, parent, runner_model, add_cfg, eval_q, eval_dec, overlay, limit_cfg, fee=fee, slip=slip)
        val_score = _score(val_metrics)
        oos_score = _score(oos_metrics)
        c3 = oos_metrics["cost3"]
        row = {
            "gate": g["name"],
            "quality_min": float(g["quality_min"]),
            "confidence_min": float(g["confidence_min"]),
            "same_side_cooldown": int(g["same_side_cooldown"]),
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
            "eval_gate_accepted": int(eval_gate_stats["accepted"]),
            "eval_gate_blocked_quality": int(eval_gate_stats["blocked_quality"]),
            "eval_gate_blocked_confidence": int(eval_gate_stats["blocked_confidence"]),
            "eval_gate_blocked_cooldown": int(eval_gate_stats["blocked_cooldown"]),
        }
        rows.append(row)
        if float(val_score) > best_val_score:
            best_val_score = float(val_score)
            best_name = str(g["name"])
            best_oos = {
                "gate": dict(g),
                "metrics": oos_metrics,
                "score": float(oos_score),
                "selection_val_score": float(val_score),
                "eval_gate_stats": eval_gate_stats,
            }

    if best_oos is None:
        raise RuntimeError("no gate variant evaluated")

    grid = pd.DataFrame(rows).sort_values("val_score", ascending=False).reset_index(drop=True)
    grid.to_csv(GRID_OUT, index=False)
    baseline_row = next(r for r in rows if r["gate"] == "baseline_no_gate")
    best_row = next(r for r in rows if r["gate"] == best_name)
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
        "sl_report": str(SL_REPORT),
        "selected_gate": best_name,
        "baseline_gate": "baseline_no_gate",
        "delta_vs_baseline_gate": delta,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Entry suppression ablation over quality threshold, confidence threshold, and same-side cooldown on top of the selected stop-loss-bucket-widened Alpha3 stack.",
        "selected_gate_variant": best_oos,
        "baseline_oos": baseline_row,
        "grid": str(GRID_OUT),
        "audit": audit,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT), "selected_gate": best_name}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
