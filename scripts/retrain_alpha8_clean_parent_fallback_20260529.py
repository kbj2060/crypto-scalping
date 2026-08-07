#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    build_training_set,
    predict_policy_frame,
    train_policy,
)
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _combine_primary_fallback,
    _combo_metrics,
    _json_default,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    _scale_decisions,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close  # noqa: E402


MODEL_ID = "alpha8_clean_parent_fallback_retrain_20260529"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
FORBIDDEN_PREFIXES = ("clean_regime_2024_unsup_v4_", "clean_regime4_2024_unsup_v1_")
DERIVABLE_FEATURES = {
    "side_hint",
    "mom_21d",
    "abs_mom_21d",
    "mom_3d",
    "abs_mom_3d",
    "mom_1d",
    "abs_mom_1d",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _assert_clean_frame(df: pd.DataFrame, *, feature_cols: list[str], name: str) -> None:
    forbidden_frame = [c for c in df.columns if str(c).startswith(FORBIDDEN_PREFIXES)]
    if forbidden_frame:
        raise RuntimeError(f"{name} contains forbidden legacy columns: {forbidden_frame[:20]}")
    forbidden_contract = [c for c in feature_cols if str(c).startswith(FORBIDDEN_PREFIXES)]
    if forbidden_contract:
        raise RuntimeError(f"{name} feature contract contains forbidden legacy columns: {forbidden_contract[:20]}")
    missing = [c for c in feature_cols if c not in df.columns and c not in DERIVABLE_FEATURES]
    if missing:
        raise RuntimeError(f"{name} missing feature columns: {missing[:40]}")


def _runtime_from_dict(raw: dict[str, Any] | None) -> alpha2.Alpha2Runtime | None:
    if not raw:
        return None
    return alpha2.Alpha2Runtime(
        name=str(raw["name"]),
        confidence=float(raw["confidence"]),
        parent_notional_scale=float(raw["parent_notional_scale"]),
        max_notional=float(raw["max_notional"]),
    )


def _active(dec: pd.DataFrame) -> int:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int)
    return int(((action != 0) & (side != 0)).sum())


def _train_parent(
    *,
    role: str,
    train_all: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: Any,
    seed: int,
    stride: int,
    out_dir: Path,
) -> tuple[dict[str, Any], alpha2.Alpha2Runtime | None, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    x_train, y_train, train_meta = build_training_set(
        train_df,
        cfg=cfg,
        stride_bars=int(stride),
        batch_size=512,
        feature_cols=feature_cols,
    )
    parent = train_policy(x_train, y_train, cfg=cfg, random_state=int(seed), feature_cols=feature_cols)
    joblib.dump(parent, out_dir / "parent.pkl")

    raw_train = predict_policy_frame(parent, train_df, close=_close(train_df))
    raw_val = predict_policy_frame(parent, val_df, close=_close(val_df))
    raw_eval = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    experiments: list[dict[str, Any]] = []

    def add(name: str, rt: alpha2.Alpha2Runtime | None, train_dec: pd.DataFrame, val_dec: pd.DataFrame, eval_dec: pd.DataFrame) -> None:
        val_c3 = _combo_metrics(val_df, val_dec)["cost3"]
        oos_c3 = _combo_metrics(eval_df, eval_dec)["cost3"]
        score = float(val_c3["pnl"]) / max(abs(float(val_c3["mdd"])), 1e-12)
        score += 0.01 * max(0, int(val_c3["trades"]) - 30)
        experiments.append(
            {
                "name": name,
                "runtime": asdict(rt) if rt is not None else None,
                "selection_score": score,
                "validation_cost3": val_c3,
                "oos_cost3": oos_c3,
                "active": {
                    "train": _active(train_dec),
                    "val": _active(val_dec),
                    "oos": _active(eval_dec),
                },
            }
        )

    add("raw", None, raw_train, raw_val, raw_eval)
    for rt in alpha2._runtimes():
        add(
            f"scaled_{rt.name}",
            rt,
            _scale_decisions(raw_train, rt),
            _scale_decisions(raw_val, rt),
            _scale_decisions(raw_eval, rt),
        )

    best = max(experiments, key=lambda x: float(x["selection_score"]))
    selected_rt = _runtime_from_dict(best.get("runtime"))
    summary = {
        "role": role,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "label_cfg": asdict(cfg),
        "stride": int(stride),
        "seed": int(seed),
        "train_meta": train_meta,
        "best_by_validation": best,
        "experiments": experiments,
    }
    _write_json(out_dir / "summary.json", summary)
    return parent, selected_rt, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--eval-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--stride", type=int, default=6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = Path(args.train_csv)
    eval_csv = Path(args.eval_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"missing train CSV: {train_csv}")
    if not eval_csv.exists():
        raise FileNotFoundError(f"missing eval CSV: {eval_csv}")

    baseline = get_live_baseline()
    primary_ref = joblib.load(baseline.primary_parent)
    fallback_ref = joblib.load(baseline.fallback_parent)
    primary_cols = list(primary_ref["feature_cols"])
    fallback_cols = list(fallback_ref["feature_cols"])
    primary_cfg = primary_ref["config"] if not isinstance(primary_ref["config"], dict) else None
    fallback_cfg = fallback_ref["config"] if not isinstance(fallback_ref["config"], dict) else None
    if primary_cfg is None or fallback_cfg is None:
        from ensemble.fully_learned_governor_policy import FullyLearnedGovernorConfig

        primary_cfg = FullyLearnedGovernorConfig(**dict(primary_ref["config"]))
        fallback_cfg = FullyLearnedGovernorConfig(**dict(fallback_ref["config"]))

    train_all = _read(train_csv)
    eval_df = _read(eval_csv)
    _assert_clean_frame(train_all, feature_cols=primary_cols, name="train_primary")
    _assert_clean_frame(train_all, feature_cols=fallback_cols, name="train_fallback")
    _assert_clean_frame(eval_df, feature_cols=primary_cols, name="eval_primary")
    _assert_clean_frame(eval_df, feature_cols=fallback_cols, name="eval_fallback")
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary, primary_rt, primary_summary = _train_parent(
        role="primary",
        train_all=train_all,
        eval_df=eval_df,
        feature_cols=primary_cols,
        cfg=primary_cfg,
        seed=8052901,
        stride=int(args.stride),
        out_dir=out_dir / "primary",
    )
    fallback, fallback_rt, fallback_summary = _train_parent(
        role="fallback",
        train_all=train_all,
        eval_df=eval_df,
        feature_cols=fallback_cols,
        cfg=fallback_cfg,
        seed=8052902,
        stride=int(args.stride),
        out_dir=out_dir / "fallback",
    )

    primary_val = _predict_scaled(primary, val_df, primary_rt)
    primary_eval = _predict_scaled(primary, eval_df, primary_rt)
    fallback_val = _predict_scaled(fallback, val_df, fallback_rt)
    fallback_eval = _predict_scaled(fallback, eval_df, fallback_rt)
    combo_val = _combine_primary_fallback(primary_val, fallback_val)
    combo_eval = _combine_primary_fallback(primary_eval, fallback_eval)
    baseline_primary_rt = _load_best_scale_runtime(baseline.primary_summary)
    baseline_fallback_rt = _load_best_scale_runtime(baseline.fallback_summary)
    baseline_primary_val = _predict_scaled(primary_ref, val_df, baseline_primary_rt)
    baseline_primary_eval = _predict_scaled(primary_ref, eval_df, baseline_primary_rt)
    baseline_fallback_val = _predict_scaled(fallback_ref, val_df, baseline_fallback_rt)
    baseline_fallback_eval = _predict_scaled(fallback_ref, eval_df, baseline_fallback_rt)
    baseline_combo_val = _combine_primary_fallback(baseline_primary_val, baseline_fallback_val)
    baseline_combo_eval = _combine_primary_fallback(baseline_primary_eval, baseline_fallback_eval)

    rows = []
    for split, frame, variants in [
        ("val", val_df, {"baseline_combo": baseline_combo_val, "clean_retrained_combo": combo_val}),
        ("oos", eval_df, {"baseline_combo": baseline_combo_eval, "clean_retrained_combo": combo_eval}),
    ]:
        for variant, dec in variants.items():
            for cost, metrics in _combo_metrics(frame, dec).items():
                rows.append({"split": split, "variant": variant, "cost": cost, **metrics})
    grid = pd.DataFrame(rows)
    grid_path = out_dir / "grid.csv"
    grid.to_csv(grid_path, index=False)

    report = {
        "model_id": str(args.model_id),
        "live_wired": False,
        "scope": "Retrain Alpha7 primary and fallback parents on Alpha8 clean funding/M7/regime4_pred candidate frames.",
        "train_csv": str(train_csv),
        "eval_csv": str(eval_csv),
        "baseline_model_id": baseline.model_id,
        "stride": int(args.stride),
        "primary": primary_summary,
        "fallback": fallback_summary,
        "cost3": {
            "val_baseline": grid[(grid["split"] == "val") & (grid["variant"] == "baseline_combo") & (grid["cost"] == "cost3")].iloc[0].to_dict(),
            "val_retrained": grid[(grid["split"] == "val") & (grid["variant"] == "clean_retrained_combo") & (grid["cost"] == "cost3")].iloc[0].to_dict(),
            "oos_baseline": grid[(grid["split"] == "oos") & (grid["variant"] == "baseline_combo") & (grid["cost"] == "cost3")].iloc[0].to_dict(),
            "oos_retrained": grid[(grid["split"] == "oos") & (grid["variant"] == "clean_retrained_combo") & (grid["cost"] == "cost3")].iloc[0].to_dict(),
        },
        "artifacts": {
            "grid": str(grid_path),
            "primary_parent": str(out_dir / "primary" / "parent.pkl"),
            "fallback_parent": str(out_dir / "fallback" / "parent.pkl"),
            "primary_summary": str(out_dir / "primary" / "summary.json"),
            "fallback_summary": str(out_dir / "fallback" / "summary.json"),
        },
        "audit": {
            "feature_contract_fail_fast": True,
            "legacy_compat_alias": False,
            "live_overwrite": False,
        },
    }
    report_path = out_dir / "summary.json"
    _write_json(report_path, report)
    print(json.dumps({"summary": str(report_path), "cost3": report["cost3"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
