#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIVE_DIR = ROOT / "data/ensemble/supervised/alpha7_1_01965_live_20260527"
os.environ["ALPHA7_LIVE_BASELINE_DIR"] = str(LIVE_DIR)
os.environ["ALPHA7_LIVE_BASELINE_MODEL_ID"] = "alpha7_1_01965_live_20260527"

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    TP_COL,
    _combine_primary_fallback,
    _combo_metrics,
    _close,
    _json_default,
    _predict_scaled,
    _read,
)
from ensemble.fully_learned_governor_policy import FullyLearnedGovernorConfig, build_training_set, predict_policy_frame, train_policy  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import BASE_PARENT, _compact_costs, _metrics, _score, _scale_decisions  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402

TRAIN_CSV = (
    ROOT
    / "tmp/causal_regen_20260516/alpha7_1_01965_v2only_tp_sl_action_score_20260528/"
    "trade_candidates_2025_alpha6_current_tail111_exact.csv"
)
EVAL_CSV = (
    ROOT
    / "tmp/causal_regen_20260516/alpha7_1_01965_v2only_tp_sl_action_score_20260528/"
    "trade_candidates_2026_alpha6_current_tail111_exact.csv"
)
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_1_01965_tp_sl_decontam_20260528"

PRIMARY_PARENT = LIVE_DIR / "primary_parent.pkl"
FALLBACK_PARENT = LIVE_DIR / "fallback_alpha43_no_legacy_parent.pkl"

FORBIDDEN_PREFIXES = (
    "clean_regime_2024_unsup_v4_",
    "clean_regime4_2024_unsup_v1_",
)
REQUIRED_PREFIX = "clean_regime4_state24_sticky090_v2_"
DERIVABLE_FEATURES = {
    "side_hint",
    "mom_21d",
    "abs_mom_21d",
    "mom_3d",
    "abs_mom_3d",
    "mom_1d",
    "abs_mom_1d",
}


def _forbidden_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if c.startswith(FORBIDDEN_PREFIXES)]


def _contract_report(cols: list[str]) -> dict[str, Any]:
    return {
        "feature_count": len(cols),
        "contains_tp_sl_action_score": TP_COL in cols,
        "forbidden_legacy_count": len(_forbidden_cols(cols)),
        "current_v2_count": sum(c.startswith(REQUIRED_PREFIX) for c in cols),
        "future_regime_pred_count": sum(c.startswith("regime4_pred_") for c in cols),
        "feature_cols": cols,
    }


def _assert_clean_frame(df: pd.DataFrame, *, name: str) -> None:
    forbidden = _forbidden_cols(list(df.columns))
    if forbidden:
        raise ValueError(f"{name} contains forbidden legacy regime columns: {forbidden[:20]}")
    if TP_COL not in df.columns:
        raise ValueError(f"{name} missing required {TP_COL}")
    v2_count = sum(c.startswith(REQUIRED_PREFIX) for c in df.columns)
    if v2_count <= 0:
        raise ValueError(f"{name} has no {REQUIRED_PREFIX} columns")


def _assert_feature_cols(df: pd.DataFrame, cols: list[str], *, name: str) -> None:
    forbidden = _forbidden_cols(cols)
    if forbidden:
        raise ValueError(f"{name} feature contract contains forbidden legacy columns: {forbidden}")
    missing = [c for c in cols if c not in df.columns and c not in DERIVABLE_FEATURES]
    if missing:
        raise ValueError(f"{name} missing feature columns in clean frame: {missing[:30]}")


def _active_count(dec: pd.DataFrame) -> int:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int)
    return int(((action != 0) & (side != 0)).sum())


def _predict_metrics(
    *,
    primary_parent: dict[str, Any],
    primary_rt: Any,
    fallback_parent: dict[str, Any],
    fallback_rt: Any,
    df: pd.DataFrame,
) -> dict[str, Any]:
    primary_dec = _predict_scaled(primary_parent, df, primary_rt)
    fallback_dec = _predict_scaled(fallback_parent, df, fallback_rt)
    combo_dec = _combine_primary_fallback(primary_dec, fallback_dec)
    return {
        "primary_active": _active_count(primary_dec),
        "fallback_active": _active_count(fallback_dec),
        "combo_active": _active_count(combo_dec),
        "combo_costs": _combo_metrics(df, combo_dec),
    }


def _load_or_train(
    *,
    train_all: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    out_dir: Path,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parent_path = out_dir / "parent.pkl"
    summary_path = out_dir / "summary.json"
    if parent_path.exists() and summary_path.exists():
        parent = joblib.load(parent_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected_rt = None
        for exp in summary.get("experiments", []):
            if exp.get("name") != summary.get("best_by_selection"):
                continue
            rt = exp.get("selected_parent_scale_runtime")
            if rt:
                from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2

                selected_rt = alpha2.Alpha2Runtime(
                    name=str(rt["name"]),
                    confidence=float(rt["confidence"]),
                    parent_notional_scale=float(rt["parent_notional_scale"]),
                    max_notional=float(rt["max_notional"]),
                )
        return parent, selected_rt, summary
    return _train_parent_fast(train_all=train_all, eval_df=eval_df, feature_cols=feature_cols, seed=seed, out_dir=out_dir)


def _train_parent_fast(
    *,
    train_all: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    out_dir: Path,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    base_parent_ref = joblib.load(BASE_PARENT)
    label_cfg = FullyLearnedGovernorConfig(**dict(base_parent_ref["config"]))
    fee = float(dict(parent_ref["config"])["fee"])
    slip = float(dict(parent_ref["config"])["slip"])

    x_train, y_train, train_meta = build_training_set(
        train_df,
        cfg=label_cfg,
        stride_bars=6,
        batch_size=512,
        feature_cols=feature_cols,
    )
    parent = train_policy(x_train, y_train, cfg=label_cfg, random_state=int(seed), feature_cols=feature_cols)
    joblib.dump(parent, out_dir / "parent.pkl")

    parent_for_features = copy.deepcopy(parent_ref)
    parent_for_features["feature_cols"] = list(feature_cols)
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")

    base_train_dec = predict_policy_frame(parent, train_df, close=_close(train_df))
    base_val_dec = predict_policy_frame(parent, val_df, close=_close(val_df))
    base_eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    experiments: list[dict[str, Any]] = []

    def add_experiment(name: str, train_dec: pd.DataFrame, val_dec: pd.DataFrame, eval_dec: pd.DataFrame, rt: Any | None) -> None:
        val_metrics = _metrics(
            val_df,
            parent_for_features=parent_for_features,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=val_dec,
            fee=fee,
            slip=slip,
        )
        eval_metrics = _metrics(
            eval_df,
            parent_for_features=parent_for_features,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=eval_dec,
            fee=fee,
            slip=slip,
        )
        item: dict[str, Any] = {
            "name": name,
            "selection_score": float(_score(val_metrics)),
            "validation_metrics": val_metrics,
            "metrics": eval_metrics,
        }
        if rt is not None:
            item["selected_parent_scale_runtime"] = asdict(rt)
        experiments.append(item)

    add_experiment("parent_direct_raw_no_teacher_fast", base_train_dec, base_val_dec, base_eval_dec, None)

    best = max(experiments, key=lambda e: float(e["selection_score"]))
    selected_rt = None
    selected_parent_rt = best.get("selected_parent_scale_runtime")
    if selected_parent_rt:
        selected_rt = alpha2.Alpha2Runtime(
            name=str(selected_parent_rt["name"]),
            confidence=float(selected_parent_rt["confidence"]),
            parent_notional_scale=float(selected_parent_rt["parent_notional_scale"]),
            max_notional=float(selected_parent_rt["max_notional"]),
        )
    summary = {
        "feature_count": len(feature_cols),
        "contains_tp_sl_action_score": TP_COL in feature_cols,
        "best_by_selection": best["name"],
        "selection_score": float(best["selection_score"]),
        "selected_validation_metrics": _compact_costs(best["validation_metrics"]),
        "selected_metrics": _compact_costs(best["metrics"]),
        "selected_parent_scale_runtime": asdict(selected_rt) if selected_rt is not None else None,
        "train_meta": train_meta,
        "experiments": experiments,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return parent, selected_rt, summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all = _read(TRAIN_CSV)
    eval_df = _read(EVAL_CSV)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    _assert_clean_frame(train_all, name="train")
    _assert_clean_frame(eval_df, name="eval")

    live_primary = joblib.load(PRIMARY_PARENT)
    live_fallback = joblib.load(FALLBACK_PARENT)
    primary_v2_cols = list(live_primary["feature_cols"])
    fallback_v2_cols = list(live_fallback["feature_cols"])
    primary_no_tp_cols = [c for c in primary_v2_cols if c != TP_COL]
    fallback_no_tp_cols = [c for c in fallback_v2_cols if c != TP_COL]

    for name, cols in {
        "primary_v2": primary_v2_cols,
        "fallback_v2": fallback_v2_cols,
        "primary_no_tp": primary_no_tp_cols,
        "fallback_no_tp": fallback_no_tp_cols,
    }.items():
        _assert_feature_cols(train_all, cols, name=name)
        _assert_feature_cols(eval_df, cols, name=name)

    model_specs = {
        "primary_v2_tp": (primary_v2_cols, 5281901),
        "primary_no_tp": (primary_no_tp_cols, 5281902),
        "fallback_v2_tp": (fallback_v2_cols, 5284301),
        "fallback_no_tp": (fallback_no_tp_cols, 5284302),
    }
    trained: dict[str, tuple[dict[str, Any], Any, dict[str, Any]]] = {}
    for model_name, (cols, seed) in model_specs.items():
        trained[model_name] = _load_or_train(
            train_all=train_all,
            eval_df=eval_df,
            feature_cols=cols,
            seed=seed,
            out_dir=OUT_DIR / model_name,
        )

    variants = {
        "both_v2_tp": ("primary_v2_tp", "fallback_v2_tp"),
        "primary_v2_fallback_no_tp": ("primary_v2_tp", "fallback_no_tp"),
        "primary_no_tp_fallback_v2": ("primary_no_tp", "fallback_v2_tp"),
        "both_no_tp": ("primary_no_tp", "fallback_no_tp"),
    }

    rows: list[dict[str, Any]] = []
    variant_reports: dict[str, Any] = {}
    for variant, (p_key, f_key) in variants.items():
        p_parent, p_rt, p_summary = trained[p_key]
        f_parent, f_rt, f_summary = trained[f_key]
        val_metrics = _predict_metrics(
            primary_parent=p_parent,
            primary_rt=p_rt,
            fallback_parent=f_parent,
            fallback_rt=f_rt,
            df=val_df,
        )
        oos_metrics = _predict_metrics(
            primary_parent=p_parent,
            primary_rt=p_rt,
            fallback_parent=f_parent,
            fallback_rt=f_rt,
            df=eval_df,
        )
        c3 = oos_metrics["combo_costs"]["cost3"]
        val_c3 = val_metrics["combo_costs"]["cost3"]
        row = {
            "variant": variant,
            "primary_model": p_key,
            "fallback_model": f_key,
            "val_cost3_pnl": float(val_c3["pnl"]),
            "val_cost3_mdd": float(val_c3["mdd"]),
            "val_cost3_trades": int(val_c3["trades"]),
            "val_cost3_wr": float(val_c3["wr"]),
            "oos_cost3_pnl": float(c3["pnl"]),
            "oos_cost3_mdd": float(c3["mdd"]),
            "oos_cost3_trades": int(c3["trades"]),
            "oos_cost3_wr": float(c3["wr"]),
            "oos_combo_active": int(oos_metrics["combo_active"]),
            "oos_primary_active": int(oos_metrics["primary_active"]),
            "oos_fallback_active": int(oos_metrics["fallback_active"]),
        }
        rows.append(row)
        variant_reports[variant] = {
            "row": row,
            "primary_contract": _contract_report(list(p_parent["feature_cols"])),
            "fallback_contract": _contract_report(list(f_parent["feature_cols"])),
            "primary_summary": p_summary,
            "fallback_summary": f_summary,
            "validation": val_metrics,
            "oos": oos_metrics,
        }

    ranking = pd.DataFrame(rows).sort_values(
        ["oos_cost3_pnl", "val_cost3_pnl"],
        ascending=[False, False],
    )
    ranking_path = OUT_DIR / "ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    report = {
        "model_id": "alpha7_1_01965_tp_sl_decontam_20260528",
        "scope": "Retrain Alpha7.1-01965 parent/fallback on v2-only tp_sl_action_score frames and compare no-tp ablations.",
        "inputs": {
            "train_csv": str(TRAIN_CSV),
            "eval_csv": str(EVAL_CSV),
            "live_dir_reference": str(LIVE_DIR),
        },
        "frame_contract": {
            "train_columns": len(train_all.columns),
            "eval_columns": len(eval_df.columns),
            "train": _contract_report(list(train_all.columns)),
            "eval": _contract_report(list(eval_df.columns)),
        },
        "ranking": rows,
        "best_variant_by_oos_cost3_pnl": str(ranking.iloc[0]["variant"]),
        "variants": variant_reports,
    }
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ranking_csv": str(ranking_path),
                "report": str(report_path),
                "best": ranking.iloc[0].to_dict(),
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
