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

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    FullyLearnedGovernorConfig,
    build_training_set,
    prepare_features,
    predict_policy_frame,
)
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3_exec  # noqa: E402
from scripts import eval_alpha3_limit_close_fallback_20260514 as alpha3_close  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.eval_hf_v13_deep_tabular_parent_mdd_20260514 import (  # noqa: E402
    FTTransformerParent,
    ParentDataset,
    RuntimeConfig,
    _decisions_from_outputs,
    _normalise_apply,
    _normalise_fit,
    _predict_outputs,
    _train_model,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha3_ft_transformer_mtl_parent_20260515"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_ft_transformer_mtl_parent_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_ft_transformer_mtl_parent_20260515_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_ft_transformer_mtl_parent_20260515_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_ft_transformer_mtl_parent_20260515_grid.csv"


def _load_teacher() -> tuple[Any, list[str], dict[str, Any], tuple[float, ...]]:
    payload = torch.load(alpha3_exec.TEACHER_MODEL, map_location="cpu", weights_only=False)
    model = alpha2._load_teacher_model(payload)
    return model, list(payload["feature_cols"]), dict(payload["train_meta"]["norm"]), tuple(float(x) for x in payload["buckets"])


def _selected_alpha3_runtime() -> alpha2.Alpha2Runtime:
    audit = json.loads(alpha3_exec.ALPHA2_AUDIT.read_text(encoding="utf-8"))
    runtime = dict(audit.get("selected_runtime", {}) or {})
    return alpha2.Alpha2Runtime(
        name=str(runtime.get("name", "noflip_c0.56_parent_scale1.10")),
        confidence=float(runtime.get("confidence", 0.56)),
        parent_notional_scale=float(runtime.get("parent_notional_scale", 1.10)),
        max_notional=float(runtime.get("max_notional", 2.75)),
    )


def _limit_cfg() -> alpha3_exec.ImmediateLimitConfig:
    return alpha3_exec.ImmediateLimitConfig(
        "next_open_limit_touch0_fee20",
        "next_open",
        0.0,
        0.0,
        0.0,
        0.20,
        entry_miss="skip",
        exit_miss="market_fallback",
    )


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _runtime_grid() -> list[RuntimeConfig]:
    out: list[RuntimeConfig] = []
    for conf in (0.30, 0.38, 0.46, 0.54, 0.62, 0.70):
        for q_floor in (-0.020, -0.010, 0.000, 0.010, 0.020):
            for scale, cap in ((0.70, 1.60), (0.90, 2.10), (1.00, 2.75), (1.10, 2.75)):
                for unc in (0.040, 0.070, 0.100):
                    out.append(
                        RuntimeConfig(
                            name=f"ft_mtl_replace_c{conf:.2f}_q{q_floor:.3f}_s{scale:.2f}_cap{cap:.2f}_u{unc:.3f}",
                            model_key="ft_transformer_mtl",
                            mode="replace",
                            confidence=float(conf),
                            quality_floor=float(q_floor),
                            notional_scale=float(scale),
                            max_notional=float(cap),
                            uncertainty_max=float(unc),
                        )
                    )
    return out


def _alpha3_metrics(
    *,
    df: pd.DataFrame,
    original_parent: dict[str, Any],
    decision_frame: pd.DataFrame,
    teacher_pred: dict[str, np.ndarray],
    teacher_buckets: tuple[float, ...],
    alpha3_runtime: alpha2.Alpha2Runtime,
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    q: np.ndarray,
    overlay: Any,
    limit_cfg: alpha3_exec.ImmediateLimitConfig,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    decisions = alpha2._decisions(decision_frame, teacher_pred, teacher_buckets, alpha3_runtime)
    return alpha3_close._metrics_signal_limit_close(
        df,
        original_parent,
        jackpot_model,
        add_cfg,
        q,
        decisions,
        overlay,
        limit_cfg,
        fee=fee,
        slip=slip,
    )


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Train FT-Transformer MTL parent replacement and backtest inside Alpha3 corrected stack.")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    torch.manual_seed(20260515)
    np.random.seed(20260515)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{MODEL_ID}] device={device} epochs={args.epochs} stride={args.stride}", flush=True)

    original_parent = joblib.load(v31.DEFAULT_PARENT)
    cfg = FullyLearnedGovernorConfig(**dict(original_parent["config"]))
    feature_cols = list(original_parent["feature_cols"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    audit_base = _audit_contract(train_all, eval_df, feature_cols)

    print(f"[{MODEL_ID}] building original MTL labels", flush=True)
    x_train, y_train, train_meta = build_training_set(train_df, cfg=cfg, stride_bars=int(args.stride), batch_size=512, feature_cols=feature_cols)
    x_val, y_val, val_meta = build_training_set(val_df, cfg=cfg, stride_bars=max(3, int(args.stride)), batch_size=512, feature_cols=feature_cols)
    x_train_norm, norm = _normalise_fit(x_train)
    x_val_norm = _normalise_apply(x_val, norm)

    model = FTTransformerParent(len(feature_cols), cfg, d_model=72, n_layers=3)
    train_ds = ParentDataset(x_train_norm.to_numpy(dtype=np.float32), y_train)
    val_ds = ParentDataset(x_val_norm.astype(np.float32), y_val)
    print(f"[{MODEL_ID}] training FT-Transformer MTL parent", flush=True)
    training = _train_model(
        "ft_transformer_mtl",
        model,
        train_ds,
        val_ds,
        epochs=int(args.epochs),
        device=device,
        batch_size=int(args.batch_size),
    )

    torch.save(
        {
            "model_id": MODEL_ID,
            "architecture": "FTTransformerParent(d_model=72,n_layers=3) multi-task heads: action, quality, notional, leverage, take_profit, stop_loss, max_hold, cooldown",
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "normalizer": norm,
            "config": dict(original_parent["config"]),
            "training": training,
        },
        OUT_DIR / "ft_transformer_mtl_parent.pt",
    )

    print(f"[{MODEL_ID}] loading fixed Alpha3 downstream layers", flush=True)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    _, v27_model = v31._load_v27(v31.DEFAULT_V27)
    v27_payload = torch.load(v31.DEFAULT_V27, map_location="cpu", weights_only=False)
    teacher_model, teacher_cols, teacher_norm, teacher_buckets = _load_teacher()
    alpha3_runtime = _selected_alpha3_runtime()
    overlay = next(v.overlay for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    limit_cfg = _limit_cfg()
    fee = float(dict(original_parent["config"])["fee"])
    slip = float(dict(original_parent["config"])["slip"])

    print(f"[{MODEL_ID}] predicting validation/eval outputs", flush=True)
    val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    val_x = _normalise_apply(val_features, norm)
    eval_x = _normalise_apply(eval_features, norm)
    val_out = _predict_outputs(model, val_x, None, device, int(args.batch_size), mc_passes=5)
    eval_out = _predict_outputs(model, eval_x, None, device, int(args.batch_size), mc_passes=5)
    val_teacher_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=teacher_cols)
    eval_teacher_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=teacher_cols)
    val_teacher_pred = teacher._predict_deep(teacher_model, val_teacher_features, teacher_cols, teacher_norm)
    eval_teacher_pred = teacher._predict_deep(teacher_model, eval_teacher_features, teacher_cols, teacher_norm)
    val_q = v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    print(f"[{MODEL_ID}] selecting runtime on 2025Q4", flush=True)
    rows: list[dict[str, Any]] = []
    best_rt: RuntimeConfig | None = None
    best_score = -1e18
    rt_grid = _runtime_grid()
    if args.quick:
        rt_grid = [r for r in rt_grid if r.confidence in (0.38, 0.54, 0.70) and r.quality_floor in (-0.01, 0.0) and r.uncertainty_max == 0.070]
    for rt in rt_grid:
        val_dec = _decisions_from_outputs(val_out, cfg, rt, val_df.index)
        metrics = _alpha3_metrics(
            df=val_df,
            original_parent=original_parent,
            decision_frame=val_dec,
            teacher_pred=val_teacher_pred,
            teacher_buckets=teacher_buckets,
            alpha3_runtime=alpha3_runtime,
            jackpot_model=jackpot_model,
            add_cfg=add_cfg,
            q=val_q,
            overlay=overlay,
            limit_cfg=limit_cfg,
            fee=fee,
            slip=slip,
        )
        score = _score(metrics)
        rows.append(
            {
                **asdict(rt),
                "score": score,
                "val_cost1_pnl": metrics["cost1"]["pnl"],
                "val_cost1_mdd": metrics["cost1"]["mdd"],
                "val_cost1_trades": metrics["cost1"]["trades"],
                "val_cost1_deep_entries": metrics["cost1"].get("deep_entries", 0),
                "val_cost2_pnl": metrics["cost2"]["pnl"],
                "val_cost3_pnl": metrics["cost3"]["pnl"],
            }
        )
        if score > best_score:
            best_score = float(score)
            best_rt = rt
            print(
                f"[{MODEL_ID}] new best {rt.name} score={score:.2f} "
                f"c1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f}",
                flush=True,
            )
    assert best_rt is not None
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)

    print(f"[{MODEL_ID}] fixed 2026 OOS", flush=True)
    eval_ft_dec = _decisions_from_outputs(eval_out, cfg, best_rt, eval_df.index)
    eval_hgb_dec = predict_policy_frame(original_parent, eval_df, close=_close(eval_df))
    experiments: list[dict[str, Any]] = []
    for name, dec in (
        ("alpha3_original_hgb_parent", eval_hgb_dec),
        (f"alpha3_ft_transformer_mtl_parent::{best_rt.name}", eval_ft_dec),
    ):
        metrics = _alpha3_metrics(
            df=eval_df,
            original_parent=original_parent,
            decision_frame=dec,
            teacher_pred=eval_teacher_pred,
            teacher_buckets=teacher_buckets,
            alpha3_runtime=alpha3_runtime,
            jackpot_model=jackpot_model,
            add_cfg=add_cfg,
            q=eval_q,
            overlay=overlay,
            limit_cfg=limit_cfg,
            fee=fee,
            slip=slip,
        )
        experiments.append({"name": name, "runtime": asdict(best_rt) if name.startswith("alpha3_ft") else None, "metrics": metrics, "score": _score(metrics)})
        print(
            f"[{MODEL_ID}] {name} c1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} "
            f"c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}",
            flush=True,
        )

    baseline = experiments[0]
    candidate = experiments[1]
    blocking = list(audit_base.get("blocking", []))
    warnings = list(audit_base.get("warnings", []))
    if candidate["score"] <= baseline["score"]:
        warnings.append("ft_transformer_mtl_parent_did_not_beat_alpha3_hgb_parent")
    if candidate["metrics"]["cost1"]["pnl"] <= 0:
        warnings.append("ft_transformer_mtl_parent_cost1_not_survived")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and candidate["score"] > baseline["score"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after runtime selection",
        "alpha3_execution_contract": asdict(limit_cfg),
        "alpha3_teacher_runtime": asdict(alpha3_runtime),
        "ft_selected_runtime": asdict(best_rt),
        "train_meta": train_meta,
        "val_meta": val_meta,
        "base_feature_audit": audit_base,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha3 parent replacement with FT-Transformer Feature Tokenizer backbone and MTL heads for action, quality, notional, leverage, TP, SL, max_hold, and cooldown. Downstream Alpha3 teacher gate, V27 scout, V21.2 runner, L2/V31 overlay, and corrected next_open_limit_touch0_fee20 execution are fixed.",
        "architecture": {
            "backbone": "FTTransformerParent",
            "tokenization": "each scalar feature becomes a learned token: x_j * W_j + b_j plus CLS token",
            "transformer": "3 encoder layers, d_model=72, 4 attention heads, GELU FFN",
            "heads": ["action", "quality", "notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"],
            "normalization": "train-only QuantileTransformer normal distribution fitted on 2025 Jan-Sep training candidates",
            "loss": "multi-task CE/SmoothL1 with active-trade bucket masking and homoscedastic balancing from existing parent experiment helper",
        },
        "training": training,
        "experiments": experiments,
        "audit": audit,
        "artifacts": {
            "model": str(OUT_DIR / "ft_transformer_mtl_parent.pt"),
            "report": str(REPORT_OUT),
            "audit": str(AUDIT_OUT),
            "grid": str(GRID_OUT),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "candidate": candidate}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
