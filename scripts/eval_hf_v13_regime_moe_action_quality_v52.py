#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    FullyLearnedGovernorConfig,
    _classifier,
    _regressor,
    build_training_set,
    predict_policy_frame,
)
from ensemble.regime_moe_policy import EXPERT_NAMES, RegimeMoEActionModel, RegimeMoEQualityModel, regime_router_weights  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_conservative_limit_sniper_v46 as v46  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_regime_moe_action_quality_v52_20260513"
DEFAULT_PARENT = v31.DEFAULT_PARENT
DEFAULT_JACKPOT = v31.DEFAULT_JACKPOT
DEFAULT_V27 = v31.DEFAULT_V27
DEFAULT_TRAIN = v31.DEFAULT_TRAIN
DEFAULT_EVAL = v31.DEFAULT_EVAL
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_regime_moe_action_quality_v52_20260513"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_regime_moe_action_quality_v52_20260513_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_regime_moe_action_quality_v52_20260513_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_regime_moe_action_quality_v52_20260513_grid.csv"

FORBIDDEN_TOKENS = ("regime_v2", "hdb", "hmm", "legacy", "future", "target", "label", "realized", "cash_after")


@dataclass(frozen=True)
class MoEConfig:
    name: str
    temperature: float
    floor: float
    expert_sharpness: float
    cash_weight: float


def _load_pickle(path: Path) -> dict[str, Any]:
    try:
        obj = joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path} did not contain dict")
    return obj


def _grid() -> list[MoEConfig]:
    return [
        MoEConfig("v52_soft_t1_floor03", 1.0, 0.03, 1.0, 0.35),
        MoEConfig("v52_sharp_t2_floor03", 2.0, 0.03, 1.5, 0.35),
        MoEConfig("v52_sharp_t2_floor06", 2.0, 0.06, 1.5, 0.35),
        MoEConfig("v52_soft_lowcash", 1.0, 0.03, 1.0, 0.25),
        MoEConfig("v52_sharp_lowcash", 2.0, 0.03, 1.6, 0.25),
    ]


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 30:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.38 * c2["pnl"] + 0.22 * c3["pnl"] - 0.25 * abs(c1["mdd"]) + 0.03 * min(c1.get("deep_entries", 0), 120))


def _feature_audit(feature_cols: list[str], train_all: pd.DataFrame, eval_df: pd.DataFrame) -> dict[str, Any]:
    lower = {c: c.lower() for c in feature_cols}
    forbidden = [
        c
        for c, name in lower.items()
        if any(tok in name for tok in FORBIDDEN_TOKENS)
        and not c.startswith("clean_regime_2024_unsup_v4_")
    ]
    clean_cols = [c for c in feature_cols if c.startswith("clean_regime_2024_unsup_v4_")]
    overlap = int(len(set(train_all["timestamp"].astype("int64")) & set(eval_df["timestamp"].astype("int64"))))
    blocking: list[str] = []
    if forbidden:
        blocking.append("forbidden_feature_cols:" + ",".join(forbidden[:20]))
    if not clean_cols:
        blocking.append("clean_regime_features_missing")
    if overlap:
        blocking.append(f"train_eval_timestamp_overlap:{overlap}")
    return {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": [],
        "feature_count": len(feature_cols),
        "clean_regime_feature_count": len(clean_cols),
        "forbidden_feature_cols": forbidden,
        "train_eval_timestamp_overlap": overlap,
    }


def _fit_classifier(model: Any, x: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> Any:
    if np.unique(y).size < 2:
        raise RuntimeError("expert action labels have fewer than two classes")
    model.fit(x, y, histgradientboostingclassifier__sample_weight=weights)
    return model


def _fit_regressor(model: Any, x: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> Any:
    model.fit(x, y, histgradientboostingregressor__sample_weight=weights)
    return model


def _train_moe_bundle(
    *,
    base_bundle: dict[str, Any],
    x: pd.DataFrame,
    y: dict[str, np.ndarray],
    cfg: MoEConfig,
    random_state: int,
) -> dict[str, Any]:
    action = np.asarray(y["action"], dtype=np.int64)
    quality = np.asarray(y["quality"], dtype=np.float64)
    action_weights = np.where(action == ACTION_CASH, float(cfg.cash_weight), 1.0)
    quality_weights = np.clip(np.abs(quality), 0.03, 1.0)
    base_weights = np.maximum(action_weights, quality_weights)
    router = regime_router_weights(x, temperature=float(cfg.temperature), floor=float(cfg.floor))
    if abs(float(cfg.expert_sharpness) - 1.0) > 1e-9:
        train_router = np.power(np.clip(router, 1e-8, None), float(cfg.expert_sharpness))
        train_router = train_router / np.maximum(train_router.sum(axis=1, keepdims=True), 1e-12)
    else:
        train_router = router

    action_experts: dict[str, Any] = {}
    quality_experts: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {"expert_weight_sum": {}, "expert_action_distribution": {}}
    classes = np.asarray([0, 1, 2], dtype=int)
    for j, name in enumerate(EXPERT_NAMES):
        w = base_weights * np.clip(train_router[:, j], 0.0, None)
        w = np.maximum(w, 1e-6)
        action_experts[name] = _fit_classifier(_classifier(random_state + 11 + j), x, action, w)
        quality_experts[name] = _fit_regressor(_regressor(random_state + 31 + j), x, quality, w)
        diagnostics["expert_weight_sum"][name] = float(w.sum())
        diagnostics["expert_action_distribution"][name] = {
            str(k): float(w[action == k].sum()) for k in sorted(np.unique(action).tolist())
        }

    out = copy.deepcopy(base_bundle)
    out["model_id"] = MODEL_ID
    out["model_type"] = "regime_moe_action_quality_overlay_v52"
    out["base_model_id"] = str(base_bundle.get("model_id", ""))
    out["moe_config"] = asdict(cfg)
    out["moe_diagnostics"] = diagnostics
    out["action_model"] = RegimeMoEActionModel(action_experts, classes, temperature=float(cfg.temperature), floor=float(cfg.floor))
    out["quality_model"] = RegimeMoEQualityModel(quality_experts, temperature=float(cfg.temperature), floor=float(cfg.floor))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V52 clean-regime soft MoE action/quality parent overlay.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=2052)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{MODEL_ID}] loading frozen V31 stack", flush=True)
    base_bundle = _load_pickle(args.parent_model)
    jackpot_payload = _load_pickle(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(args.v27_model)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    train = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    feature_cols = list(base_bundle.get("feature_cols") or [])
    feature_audit = _feature_audit(feature_cols, train_all, eval_df)
    parent_audit = _audit_contract(train_all, eval_df, feature_cols)
    if feature_audit["status"] != "pass":
        raise RuntimeError(f"feature audit failed: {feature_audit}")

    print(f"[{MODEL_ID}] building action/quality labels train=2025 Jan-Sep stride={args.stride}", flush=True)
    cfg = FullyLearnedGovernorConfig(**dict(base_bundle.get("config", {})))
    x_train, y_train, training_meta = build_training_set(
        train,
        cfg=cfg,
        stride_bars=int(args.stride),
        batch_size=int(args.batch_size),
        feature_cols=feature_cols,
    )
    print(f"[{MODEL_ID}] predicting frozen V27 utilities", flush=True)
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    base_val_dec = predict_policy_frame(base_bundle, val, close=_close(val))
    base_eval_dec = predict_policy_frame(base_bundle, eval_df, close=_close(eval_df))
    fee = float(dict(base_bundle["config"])["fee"])
    slip = float(dict(base_bundle["config"])["slip"])
    overlay = v46._base_overlay()
    baseline: dict[str, Any] = {}
    for mult in (1, 2, 3):
        baseline[f"cost{mult}"] = v31.backtest(
            eval_df,
            base_bundle,
            jackpot_model,
            add_cfg,
            eval_q,
            overlay,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
            decisions=base_eval_dec,
        )

    rows: list[dict[str, Any]] = []
    bundles: dict[str, dict[str, Any]] = {}
    best: dict[str, Any] | None = None
    print(f"[{MODEL_ID}] training MoE variants", flush=True)
    for i, moe_cfg in enumerate(_grid()):
        print(f"[{MODEL_ID}] train/eval {moe_cfg.name}", flush=True)
        bundle = _train_moe_bundle(base_bundle=base_bundle, x=x_train, y=y_train, cfg=moe_cfg, random_state=int(args.seed + i * 101))
        bundles[moe_cfg.name] = bundle
        val_dec = predict_policy_frame(bundle, val, close=_close(val))
        eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
        v1 = v31.backtest(val, bundle, jackpot_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=1.0, decisions=val_dec)
        v2 = v31.backtest(val, bundle, jackpot_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=2.0, decisions=val_dec)
        v3 = v31.backtest(val, bundle, jackpot_model, add_cfg, val_q, overlay, fee=fee, slip=slip, cost_mult=3.0, decisions=val_dec)
        row = {"config": asdict(moe_cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    selected = MoEConfig(**best["config"])
    selected_bundle = bundles[selected.name]
    eval_dec = predict_policy_frame(selected_bundle, eval_df, close=_close(eval_df))
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = v31.backtest(
            eval_df,
            selected_bundle,
            jackpot_model,
            add_cfg,
            eval_q,
            overlay,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
            decisions=eval_dec,
            record=(mult == 1),
        )
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            ledger_path = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            ledger.to_csv(ledger_path, index=False)
            ledgers["cost1"] = str(ledger_path)
        metrics[f"cost{mult}"] = r

    selected_path = args.out_dir / "v52_regime_moe_action_quality_parent.pkl"
    joblib.dump(selected_bundle, selected_path)
    pd.DataFrame(
        [
            {
                **{f"cfg_{k}": v for k, v in row["config"].items()},
                "selection_score": row["selection_score"],
                "val_cost1_pnl": row["validation_cost1"]["pnl"],
                "val_cost1_mdd": row["validation_cost1"]["mdd"],
                "val_cost1_trades": row["validation_cost1"]["trades"],
                "val_cost2_pnl": row["validation_cost2"]["pnl"],
                "val_cost3_pnl": row["validation_cost3"]["pnl"],
            }
            for row in rows
        ]
    ).to_csv(args.grid_out, index=False)

    blocking: list[str] = []
    warnings: list[str] = []
    if parent_audit["status"] != "pass":
        blocking.extend(parent_audit.get("blocking", []))
    warnings.extend(parent_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= baseline["cost1"]["pnl"]:
        warnings.append("oos_cost1_did_not_beat_v31")
    if metrics["cost2"]["pnl"] <= baseline["cost2"]["pnl"]:
        warnings.append("oos_cost2_did_not_beat_v31")
    if metrics["cost3"]["pnl"] <= baseline["cost3"]["pnl"]:
        warnings.append("oos_cost3_did_not_beat_v31")
    verdict = (
        "candidate_recheck"
        if not blocking
        and metrics["cost1"]["pnl"] > baseline["cost1"]["pnl"]
        and metrics["cost2"]["pnl"] > baseline["cost2"]["pnl"]
        and metrics["cost3"]["pnl"] > baseline["cost3"]["pnl"]
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
        "feature_audit": feature_audit,
        "parent_contract_audit": parent_audit,
        "baseline_recomputed_v31": baseline,
        "selected_config": asdict(selected),
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Clean-regime soft MoE for parent action/quality only. Router uses clean_regime_2024_unsup_v4 probabilities; bull/bear/chop experts are sample-weighted specialists. Existing V31 parent bucket heads, V21.2 jackpot runner, and V27 deep scout are preserved.",
        "split_policy": "train=2025 Jan-Sep, selection=2025 Q4, OOS=2026 fixed",
        "training_meta": training_meta,
        "selected_config": asdict(selected),
        "selection_result": best,
        "baseline_recomputed_v31": baseline,
        "metrics": metrics,
        "audit": audit,
        "artifacts": {"model": str(selected_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers},
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(selected_path), "selected": asdict(selected), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
