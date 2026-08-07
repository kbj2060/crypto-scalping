#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
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
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_deep_entry_parent_lite_v38 import DeepEntryParentLite  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha2_teacher_l2_runtime_sweep_20260514"
TEACHER_MODEL = ROOT / "data/ensemble/supervised/alpha1_l2_teacher_deep_parent_20260514/teacher_deep_parent_l2_replay.pt"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha2_teacher_l2_runtime_sweep_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha2_teacher_l2_runtime_sweep_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha2_teacher_l2_runtime_sweep_20260514_grid.csv"


@dataclass(frozen=True)
class Alpha2Runtime:
    name: str
    confidence: float
    parent_notional_scale: float
    max_notional: float


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.30 * c3["pnl"] - 0.35 * abs(c1["mdd"]))


def _runtimes() -> list[Alpha2Runtime]:
    rows: list[Alpha2Runtime] = []
    for conf in (0.56, 0.62, 0.68, 0.74, 0.80):
        for scale in (0.85, 1.00, 1.10):
            rows.append(
                Alpha2Runtime(
                    name=f"noflip_c{conf:.2f}_parent_scale{scale:.2f}",
                    confidence=float(conf),
                    parent_notional_scale=float(scale),
                    max_notional=2.75,
                )
            )
    return rows


def _scale_parent_notional(decisions: pd.DataFrame, rt: Alpha2Runtime) -> pd.DataFrame:
    if abs(float(rt.parent_notional_scale) - 1.0) < 1e-12:
        return decisions
    out = decisions.copy()
    active = (out["action"].astype(int).to_numpy() != 0) & (out["side"].astype(int).to_numpy() != 0)
    notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    leverage = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    scaled = np.minimum(notional * float(rt.parent_notional_scale), float(rt.max_notional))
    out.loc[active, "notional_exposure"] = scaled[active]
    out.loc[active, "position_fraction"] = scaled[active] / np.maximum(leverage[active], 1e-12)
    return out


def _decisions(base_decisions: pd.DataFrame, pred: dict[str, np.ndarray], buckets: tuple[float, ...], rt: Alpha2Runtime) -> pd.DataFrame:
    teacher_rt = teacher.Runtime(
        name=rt.name,
        confidence=float(rt.confidence),
        skip_on_cash=True,
        allow_flip=False,
        use_learned_size=False,
        notional_scale=1.0,
        max_notional=float(rt.max_notional),
    )
    return _scale_parent_notional(teacher._constrained_decisions(base_decisions, pred, buckets, teacher_rt), rt)


def _metrics(
    df: pd.DataFrame,
    parent: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    q: np.ndarray,
    decisions: pd.DataFrame,
    variant: Any,
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    return {
        f"cost{mult}": l2._run_with_l2_proxy(
            df,
            parent,
            jackpot_model,
            add_cfg,
            q,
            decisions,
            variant,
            fee,
            slip,
            cost_mult=float(mult),
        )
        for mult in (1, 2, 3)
    }


def _load_teacher_model(payload: dict[str, Any]) -> DeepEntryParentLite:
    feature_cols = list(payload["feature_cols"])
    buckets = tuple(payload["buckets"])
    model = DeepEntryParentLite(len(feature_cols), notional_classes=len(buckets))
    model.load_state_dict(payload["state_dict"])
    return model.cpu().eval()


def main() -> int:
    print(f"[{MODEL_ID}] loading fixed Alpha2 teacher + L2 stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(TEACHER_MODEL, map_location="cpu", weights_only=False)
    model = _load_teacher_model(payload)
    feature_cols = list(payload["feature_cols"])
    norm = payload["train_meta"]["norm"]
    buckets = tuple(float(x) for x in payload["buckets"])

    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base = dict(parent["config"])
    fee = float(base["fee"])
    slip = float(base["slip"])

    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    contract_features = _feature_cols(train_all, eval_df)
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))
    l2_stats = l2._live_l2_stats()

    print(f"[{MODEL_ID}] predicting base parent, teacher, V27", flush=True)
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    val_features = prepare_features(val, side_hint=0, close=_close(val), feature_cols=contract_features)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=contract_features)
    val_pred = teacher._predict_deep(model, val_features, feature_cols, norm)
    eval_pred = teacher._predict_deep(model, eval_features, feature_cols, norm)
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    replay_variants = [v for v in l2._variants() if v.name != "alpha1_taker_baseline"]
    rows: list[dict[str, Any]] = []
    selected_rt: Alpha2Runtime | None = None
    selected_variant: Any | None = None
    best_score = -1e18
    print(f"[{MODEL_ID}] selection on 2025Q4", flush=True)
    for rt in _runtimes():
        dec = _decisions(val_dec, val_pred, buckets, rt)
        for variant in replay_variants:
            metrics = _metrics(val, parent, jackpot_model, add_cfg, val_q, dec, variant, fee=fee, slip=slip)
            score = _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])
            rows.append(
                {
                    **asdict(rt),
                    "variant": variant.name,
                    "selection_score": score,
                    "val_cost1_pnl": metrics["cost1"]["pnl"],
                    "val_cost1_mdd": metrics["cost1"]["mdd"],
                    "val_cost1_trades": metrics["cost1"]["trades"],
                    "val_cost2_pnl": metrics["cost2"]["pnl"],
                    "val_cost3_pnl": metrics["cost3"]["pnl"],
                }
            )
            if score > best_score:
                best_score = score
                selected_rt = rt
                selected_variant = variant
                print(
                    f"[{MODEL_ID}] new best {rt.name} {variant.name} score={score:.2f} "
                    f"c1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f}",
                    flush=True,
                )
    assert selected_rt is not None and selected_variant is not None
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)

    print(f"[{MODEL_ID}] fixed 2026 OOS", flush=True)
    eval_selected_dec = _decisions(eval_dec, eval_pred, buckets, selected_rt)
    experiments: list[dict[str, Any]] = []
    for name, decisions, variant in (
        ("alpha2_reference", _decisions(eval_dec, eval_pred, buckets, Alpha2Runtime("noflip_c0.56_parent_scale1.00", 0.56, 1.0, 2.75)), selected_variant),
        (f"alpha2_1::{selected_rt.name}::{selected_variant.name}", eval_selected_dec, selected_variant),
    ):
        metrics = _metrics(eval_df, parent, jackpot_model, add_cfg, eval_q, decisions, variant, fee=fee, slip=slip)
        experiments.append(
            {
                "name": name,
                "runtime": asdict(selected_rt) if name.startswith("alpha2_1") else {"confidence": 0.56, "parent_notional_scale": 1.0},
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

    reference = experiments[0]
    candidate = experiments[1]
    warnings = list(parent_audit.get("warnings", []))
    if not l2_stats.get("usable_for_replay", False):
        warnings.append("historical_l2_snapshots_insufficient_conservative_ohlc_replay_only")
    warnings.append("real_live_l2_fill_model_requires_forward_shadow_collection")
    if candidate["score"] <= reference["score"]:
        warnings.append("alpha2_1_runtime_sweep_did_not_beat_alpha2_reference")
    audit = {
        "status": "pass" if not parent_audit.get("blocking") else "fail",
        "verdict": "shadow_collect_l2" if not parent_audit.get("blocking") else "fail",
        "blocking": list(parent_audit.get("blocking", [])),
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "l2_stats": l2_stats,
        "selected_runtime": asdict(selected_rt),
        "selected_variant": asdict(selected_variant),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha2.1 runtime-only improvement sweep over teacher confidence and parent notional scale. The deep teacher checkpoint, HGB parent, V27 scout, V21.2 runner, V31 exit, and L2 replay mechanism are fixed.",
        "experiments": experiments,
        "audit": audit,
        "artifacts": {
            "teacher_model": str(TEACHER_MODEL),
            "report": str(REPORT_OUT),
            "audit": str(AUDIT_OUT),
            "grid": str(GRID_OUT),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "selected": candidate["name"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
