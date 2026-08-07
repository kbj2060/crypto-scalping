#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_alpha1_soft_execution_proxy_20260513 as soft  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_frozen_parent_layer_ablation_v45 as v45  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha1_4_teacher_deep_parent_soft_exec_20260514"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_4_teacher_deep_parent_soft_exec_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_4_teacher_deep_parent_soft_exec_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha1_4_teacher_deep_parent_soft_exec_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha1_4_teacher_deep_parent_soft_exec_20260514_grid.csv"


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.25 * c3["pnl"] - 0.25 * abs(c1["mdd"]))


def _metrics(
    df: pd.DataFrame,
    parent: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    q: np.ndarray,
    decisions: pd.DataFrame,
    variant: v45.LayerVariant,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    return {
        f"cost{mult}": v45.backtest_variant(
            df,
            parent,
            jackpot_model,
            add_cfg,
            q,
            variant,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
            decisions=decisions,
        )
        for mult in (1, 2, 3)
    }


def _teacher_decisions(
    raw_decisions: pd.DataFrame,
    pred: dict[str, np.ndarray],
    buckets: tuple[float, ...],
    runtime: teacher.Runtime,
) -> pd.DataFrame:
    return teacher._constrained_decisions(raw_decisions, pred, buckets, runtime)


def main() -> int:
    print(f"[{MODEL_ID}] loading Alpha1.4 stack", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base = dict(parent["config"])
    fee = float(base["fee"])
    slip = float(base["slip"])
    buckets = tuple(base.get("notional_buckets", (0.23, 0.368, 0.575, 0.8625, 1.2075, 1.6675, 2.3, 3.105, 4.14)))

    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))

    print(f"[{MODEL_ID}] parent decisions", flush=True)
    train_dec = predict_policy_frame(parent, train, close=_close(train))
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    print(f"[{MODEL_ID}] training teacher-constrained deep parent without interruption", flush=True)
    train_features = prepare_features(train, side_hint=0, close=_close(train), feature_cols=feature_cols)
    train_seq = teacher._seq_tensor(train_features, np.arange(len(train), dtype=np.int64), feature_cols)
    y_action = train_dec["action"].astype(int).to_numpy(dtype=np.int64)
    y_quality = pd.to_numeric(train_dec["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y_notional = teacher._bucket_labels(train_dec, buckets)
    model, train_meta = teacher._train_teacher_model(
        train_seq,
        y_action,
        y_quality,
        y_notional,
        n_buckets=len(buckets),
        epochs=35,
    )

    print(f"[{MODEL_ID}] predicting teacher and frozen V27", flush=True)
    val_features = prepare_features(val, side_hint=0, close=_close(val), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    val_pred = teacher._predict_deep(model, val_features, feature_cols, train_meta["norm"])
    eval_pred = teacher._predict_deep(model, eval_features, feature_cols, train_meta["norm"])
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    print(f"[{MODEL_ID}] selecting teacher runtime + soft execution variant on 2025Q4", flush=True)
    rows: list[dict[str, Any]] = []
    selected_runtime: teacher.Runtime | None = None
    selected_variant: v45.LayerVariant | None = None
    best_score = -1e18
    variants = soft._variants()
    baseline_variant = variants[0]
    for runtime in teacher._grid():
        dec = _teacher_decisions(val_dec, val_pred, buckets, runtime)
        for variant in variants:
            if variant.name == "alpha1_baseline_taker":
                continue
            vm = _metrics(val, parent, jackpot_model, add_cfg, val_q, dec, variant, fee=fee, slip=slip)
            score = _score(vm["cost1"], vm["cost2"], vm["cost3"])
            row = {
                "runtime": runtime.name,
                "variant": variant.name,
                "selection_score": score,
                "val_cost1_pnl": vm["cost1"]["pnl"],
                "val_cost1_mdd": vm["cost1"]["mdd"],
                "val_cost1_trades": vm["cost1"]["trades"],
                "val_cost2_pnl": vm["cost2"]["pnl"],
                "val_cost3_pnl": vm["cost3"]["pnl"],
                "runtime_config": asdict(runtime),
                "variant_config": asdict(variant),
            }
            rows.append(row)
            if score > best_score:
                best_score = score
                selected_runtime = runtime
                selected_variant = variant
                print(
                    f"[{MODEL_ID}] new best runtime={runtime.name} variant={variant.name} "
                    f"score={score:.2f} c1={vm['cost1']['pnl']:.2f} c2={vm['cost2']['pnl']:.2f} c3={vm['cost3']['pnl']:.2f}",
                    flush=True,
                )
    assert selected_runtime is not None and selected_variant is not None
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)

    print(f"[{MODEL_ID}] evaluating fixed 2026 OOS", flush=True)
    selected_eval_dec = _teacher_decisions(eval_dec, eval_pred, buckets, selected_runtime)
    experiments: list[dict[str, Any]] = []
    for name, decisions, variant in (
        ("alpha1_baseline_taker", eval_dec, baseline_variant),
        ("alpha1_4_soft_exec", eval_dec, selected_variant),
        (f"teacher_soft::{selected_runtime.name}::{selected_variant.name}", selected_eval_dec, selected_variant),
    ):
        metrics = _metrics(eval_df, parent, jackpot_model, add_cfg, eval_q, decisions, variant, fee=fee, slip=slip)
        experiments.append(
            {
                "name": name,
                "runtime": asdict(selected_runtime) if name.startswith("teacher_soft") else None,
                "variant": asdict(variant),
                "metrics": metrics,
                "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"]),
            }
        )
        print(
            f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} "
            f"cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}",
            flush=True,
        )

    model_path = OUT_DIR / "teacher_deep_parent_soft_exec.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "train_meta": train_meta,
            "selected_runtime": asdict(selected_runtime),
            "selected_variant": asdict(selected_variant),
            "buckets": buckets,
        },
        model_path,
    )
    best = max(experiments, key=lambda x: x["score"])
    alpha14 = next(e for e in experiments if e["name"] == "alpha1_4_soft_exec")
    blocking = list(parent_audit.get("blocking", []))
    warnings = list(parent_audit.get("warnings", []))
    warnings.append("soft_execution_proxy_uses_ohlcv_micro_proxy_not_live_l2_orderbook")
    if best["name"].startswith("teacher_soft"):
        if best["metrics"]["cost1"]["pnl"] <= alpha14["metrics"]["cost1"]["pnl"]:
            warnings.append("teacher_soft_did_not_beat_alpha1_4_cost1")
        if best["metrics"]["cost2"]["pnl"] <= alpha14["metrics"]["cost2"]["pnl"]:
            warnings.append("teacher_soft_did_not_beat_alpha1_4_cost2")
        if best["metrics"]["cost3"]["pnl"] <= alpha14["metrics"]["cost3"]["pnl"]:
            warnings.append("teacher_soft_did_not_beat_alpha1_4_cost3")
    else:
        warnings.append("selected_best_is_not_teacher_soft_combo")

    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote"
        if (
            not blocking
            and best["name"].startswith("teacher_soft")
            and best["metrics"]["cost1"]["pnl"] > alpha14["metrics"]["cost1"]["pnl"]
            and best["metrics"]["cost2"]["pnl"] > 0.0
            and best["metrics"]["cost3"]["pnl"] > 0.0
        )
        else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "train_window": "2025-01-01..2025-09-30",
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "parent_base_model": str(v31.DEFAULT_PARENT),
        "teacher_deep_parent_retrained": True,
        "v27_deep_scout_preserved": True,
        "v21_2_model_preserved": True,
        "v31_exit_preserved": True,
        "selected_runtime": asdict(selected_runtime),
        "selected_variant": asdict(selected_variant),
        "parent_audit": parent_audit,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Combination test: train a teacher-constrained sequence deep parent on 2025-01..09, preserve parent CASH bars so V27 deep scout remains active, then apply the selected Alpha1.4 soft execution proxy. Runtime and execution variant are selected only on 2025Q4 and evaluated on fixed 2026 OOS.",
        "experiments": experiments,
        "audit": audit,
        "artifacts": {
            "model": str(model_path),
            "report": str(REPORT_OUT),
            "audit": str(AUDIT_OUT),
            "grid": str(GRID_OUT),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "best": best["name"], "verdict": audit["verdict"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
