#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/zigzag_second_stage_family_sweep_20260531"
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
BASE_2024 = ROOT / "tmp/causal_regen_20260516/funding_clean_retrain_20260529/rl_training_2024_unified_cleanfunding.csv"
BASE_2025 = ROOT / "tmp/causal_regen_20260516/funding_clean_retrain_20260529/rl_training_2025_unified_cleanfunding.csv"
BASE_2026 = ROOT / "data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv"

YEAR_EXTRA_SOURCES = {
    2024: [
        ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2024_regime3_current_sensitive_hmm_wide24.csv",
        ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2024_regime3_stability_risk_h6.csv",
        ROOT / "data/ensemble/supervised/regime3_transition_hazard_sensitive_h6_withcurrent_20260530/training_features_2024_regime3_transition_hazard_h6_thr046.csv",
    ],
    2025: [
        ROOT / "tmp/causal_regen_20260516/ai_role_specific_eval_20260530/tsfm_role_features_2025_exact.csv",
        ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2025_regime3_current_sensitive_hmm_wide24.csv",
        ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2025_regime3_stability_risk_h6.csv",
        ROOT / "data/ensemble/supervised/regime3_transition_hazard_sensitive_h6_withcurrent_20260530/training_features_2025_regime3_transition_hazard_h6_thr046.csv",
    ],
    2026: [
        ROOT / "tmp/causal_regen_20260516/ai_role_specific_eval_20260530/tsfm_role_features_2026_exact.csv",
        ROOT / "tmp/causal_regen_20260516/ai_timesnet_direction_inputs_bg_20260530/role_features_2026_reworked.csv",
        ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv",
        ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2026_rebuilt_regime3_stability_risk_h6.csv",
        ROOT / "data/ensemble/supervised/regime3_transition_hazard_sensitive_h6_withcurrent_20260530/training_features_2026_rebuilt_regime3_transition_hazard_h6_thr046.csv",
    ],
}

FORBIDDEN_TOKENS = (
    "label",
    "target",
    "future",
    "pnl",
    "cash_after",
    "realized",
    "zigzag_",
    "tp_sl_action_score",
)
P0_M7_TARGET_FAMILY = {
    "m7_target_quality",
    "m7_target_hold",
    "m7_quality_pred",
    "m7_hold_pred",
    "m7_q10",
    "m7_q90",
    "m7_qwidth",
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path} missing timestamp")
    return (
        frame.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
        .replace([np.inf, -np.inf], np.nan)
    )


def _read_year_frame(year: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    base_path = {2024: BASE_2024, 2025: BASE_2025, 2026: BASE_2026}[int(year)]
    frame = _read_csv(base_path)
    years = sorted(frame["timestamp"].dt.year.dropna().astype(int).unique().tolist())
    if years != [int(year)]:
        raise RuntimeError(f"{base_path} year guard failed: expected={[int(year)]} actual={years}")
    joins: list[dict[str, Any]] = [{"source": str(base_path), "rows": int(len(frame)), "role": "base"}]
    for src in YEAR_EXTRA_SOURCES[int(year)]:
        extra = _read_csv(src)
        keep = ["timestamp"] + [c for c in extra.columns if c != "timestamp" and c not in frame.columns]
        if len(keep) == 1:
            joins.append({"source": str(src), "rows": int(len(extra)), "added_cols": 0, "dropped_duplicate_cols": int(len(extra.columns) - 1)})
            continue
        before = len(frame)
        frame = frame.merge(extra[keep], on="timestamp", how="left", validate="one_to_one")
        if len(frame) != before:
            raise RuntimeError(f"{src} join changed rows: {before}->{len(frame)}")
        joins.append(
            {
                "source": str(src),
                "rows": int(len(extra)),
                "added_cols": int(len(keep) - 1),
                "missing_rows_after_join": int(frame[keep[1:]].isna().all(axis=1).sum()),
            }
        )
    return frame, joins


def _join_labels(frame: pd.DataFrame, label_dir: Path, year: int) -> pd.DataFrame:
    labels = _read_csv(label_dir / f"zigzag_action_labels_{int(year)}.csv")
    if "wave3_action" in labels.columns:
        raise ValueError(f"zigzag labels {year} contain removed active contract column: wave3_action")
    required = {"timestamp", "zigzag_action"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"zigzag labels {year} missing {missing}")
    before = len(frame)
    out = frame.merge(labels[["timestamp", "zigzag_action"]], on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError(f"zigzag label join changed rows for {year}: {before}->{len(out)}")
    miss = int(out["zigzag_action"].isna().sum())
    if miss:
        raise RuntimeError(f"zigzag label join missing rows for {year}: {miss}")
    out["zigzag_action"] = out["zigzag_action"].astype(np.int64)
    return out


def _safe_numeric_cols(frame: pd.DataFrame, pred: Callable[[str], bool]) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        low = col.lower()
        if col == "timestamp" or col in P0_M7_TARGET_FAMILY:
            continue
        if any(tok in low for tok in FORBIDDEN_TOKENS):
            continue
        if not pred(col):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return sorted(dict.fromkeys(cols))


def _family_specs() -> dict[str, Callable[[str], bool]]:
    return {
        "ai_all_legacy": lambda c: c.startswith("ai_") or c.startswith("patchtst") or c in {"pred_patchtst", "conf_patchtst"} or c.startswith("tide") or c.startswith("timesnet") or c.startswith("dlinear"),
        "ai_direction_legacy": lambda c: c.startswith("ai_dir_") or c.startswith("patchtst") or c in {"pred_patchtst", "conf_patchtst"},
        "ai_role_risk_context": lambda c: c.startswith("ai_") and not c.startswith("ai_dir_") or c.startswith("tide") or c.startswith("timesnet") or c.startswith("dlinear"),
        "m7_all_nonp0": lambda c: c.startswith("m7_"),
        "m7_direction_legacy": lambda c: c.startswith(("m7_trend_xgb_", "m7_mtl_", "m7_quant_", "m7_prob_")) or c in {"m7_direction", "m7_action", "m7_confidence", "m7_size", "m7_composite_score", "m7_long_edge", "m7_short_edge", "m7_q50", "m7_expected_ret"},
        "m7_unsup_risk_context": lambda c: c.startswith(("m7_gmm_", "m7_iso_", "m7_vae_")) or c in {"m7_gate_block", "m7_tail_risk", "m7_tradeability_score", "m7_long_adverse_prob", "m7_short_adverse_prob", "m7_long_mae_q90", "m7_short_mae_q90"},
        "regime3_current_context": lambda c: c.startswith("regime3_current_") or c in {"cvp_regime", "regime_trending", "regime_persistence"},
        "regime3_risk_context": lambda c: c.startswith(("regime3_stability_", "regime3_transition_", "regime3_churn_")),
        "regime3_all_context": lambda c: (c.startswith("regime3_") and not c.startswith("regime3_pred_")) or c in {"cvp_regime", "regime_trending", "regime_persistence"},
        "all_second_stage_nonp0": lambda c: (
            c.startswith("ai_")
            or c.startswith("patchtst")
            or c in {"pred_patchtst", "conf_patchtst", "cvp_regime", "regime_trending", "regime_persistence"}
            or c.startswith(("tide", "timesnet", "dlinear", "m7_"))
            or (c.startswith("regime3_") and not c.startswith("regime3_pred_"))
        ),
    }


def _class_weights(y: np.ndarray, power: float) -> list[float]:
    counts = np.maximum(np.bincount(y.astype(np.int64), minlength=3).astype(np.float64), 1.0)
    weights = (counts.sum() / (3.0 * counts)) ** float(power)
    return [float(v) for v in weights]


def _fit(x: pd.DataFrame, y: np.ndarray, args: argparse.Namespace) -> tuple[CatBoostClassifier, str]:
    params = {
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1",
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_seed": int(args.seed),
        "class_weights": _class_weights(y, float(args.class_weight_power)),
        "verbose": False,
        "allow_writing_files": False,
    }
    last: Exception | None = None
    for task_type in [str(args.task_type), "CPU"]:
        try:
            model = CatBoostClassifier(**params, task_type=task_type)
            model.fit(Pool(x, y))
            return model, task_type
        except Exception as exc:
            last = exc
            if task_type == "CPU":
                raise
    raise RuntimeError(last)


def _prep(train: pd.DataFrame, score: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], int, int]:
    x_train = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_score = score[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x_train.median(axis=0).fillna(0.0)
    x_train = x_train.fillna(med)
    x_score = x_score.fillna(med)
    return x_train, x_score, {k: float(v) for k, v in med.to_dict().items()}, int(x_train.isna().sum().sum()), int(x_score.isna().sum().sum())


def _metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1).astype(np.int64)
    out: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y.astype(np.int64), minlength=3))},
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=3))},
    }
    try:
        out["ovr_auc"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=[0, 1, 2]))
    except ValueError:
        out["ovr_auc"] = None
    return out


def _scores(frame: pd.DataFrame, proba: np.ndarray, prefix: str) -> pd.DataFrame:
    out = frame[["timestamp"]].copy()
    out[f"{prefix}_p_cash"] = proba[:, 0].astype(np.float32)
    out[f"{prefix}_p_long"] = proba[:, 1].astype(np.float32)
    out[f"{prefix}_p_short"] = proba[:, 2].astype(np.float32)
    out[f"{prefix}_action"] = np.argmax(proba, axis=1).astype(np.int8)
    out[f"{prefix}_confidence"] = np.max(proba, axis=1).astype(np.float32)
    out[f"{prefix}_side_edge"] = (proba[:, 1] - proba[:, 2]).astype(np.float32)
    out[f"{prefix}_trade_prob"] = (1.0 - proba[:, 0]).astype(np.float32)
    return out


def _train_pair(
    family: str,
    train: pd.DataFrame,
    score: pd.DataFrame,
    train_year: int,
    score_year: int,
    pred: Callable[[str], bool],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_cols = set(_safe_numeric_cols(train, pred))
    score_cols = set(_safe_numeric_cols(score, pred))
    cols = sorted(train_cols & score_cols)
    if not cols:
        raise RuntimeError(f"{family} selected no common numeric columns for {train_year}->{score_year}")
    y_train = train["zigzag_action"].to_numpy(dtype=np.int64)
    y_score = score["zigzag_action"].to_numpy(dtype=np.int64)
    x_train, x_score, med, train_nan_after, score_nan_after = _prep(train, score, cols)
    model, task_type = _fit(x_train, y_train, args)
    train_proba = model.predict_proba(x_train)
    score_proba = model.predict_proba(x_score)
    if train_proba.shape[1] != 3 or score_proba.shape[1] != 3:
        raise RuntimeError(f"{family} model did not learn all 3 classes")
    prefix = f"zigzag_{family}"
    score_csv = out_dir / f"{prefix}_scores_train{train_year}_score{score_year}.csv"
    model_path = out_dir / f"{prefix}_train{train_year}_score{score_year}.cbm"
    model.save_model(str(model_path))
    _scores(score, score_proba, prefix).to_csv(score_csv, index=False)
    return {
        "family": family,
        "train_year": int(train_year),
        "score_year": int(score_year),
        "task_type_used": task_type,
        "feature_count": int(len(cols)),
        "features": cols,
        "excluded_p0_m7_target_family": sorted(P0_M7_TARGET_FAMILY),
        "train_nan_after_fill": train_nan_after,
        "score_nan_after_fill": score_nan_after,
        "model_path": str(model_path),
        "score_csv": str(score_csv),
        "train_metrics": _metrics(y_train, train_proba),
        "score_metrics": _metrics(y_score, score_proba),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.035)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--l2-leaf-reg", type=float, default=10.0)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--task-type", choices=("GPU", "CPU"), default="GPU")
    p.add_argument("--families", default="")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[int, pd.DataFrame] = {}
    joins: dict[str, Any] = {}
    for year in (2024, 2025, 2026):
        frame, join_info = _read_year_frame(year)
        frames[year] = _join_labels(frame, args.label_dir, year)
        joins[str(year)] = join_info
    specs = _family_specs()
    selected = list(specs) if not args.families.strip() else [x.strip() for x in args.families.split(",") if x.strip()]
    unknown = sorted(set(selected) - set(specs))
    if unknown:
        raise ValueError(f"unknown families: {unknown}; known={sorted(specs)}")
    results: list[dict[str, Any]] = []
    for family in selected:
        for train_year, score_year in ((2024, 2025), (2025, 2026)):
            print(f"[zigzag-family] {family} {train_year}->{score_year}", flush=True)
            results.append(
                _train_pair(
                    family,
                    frames[train_year],
                    frames[score_year],
                    train_year,
                    score_year,
                    specs[family],
                    args.out_dir,
                    args,
                )
            )
    audit = {
        "type": "zigzag_second_stage_family_sweep",
        "label_contract": str(args.label_dir / "zigzag_action_label_audit.json"),
        "label_column": "zigzag_action",
        "source_joins": joins,
        "families": selected,
        "contract": {
            "exact_timestamp_join_only": True,
            "no_ffill_bfill": True,
            "forbidden_tokens": FORBIDDEN_TOKENS,
            "excluded_p0_m7_target_family": sorted(P0_M7_TARGET_FAMILY),
            "old_2_action_outputs_are_retrained_to_zigzag": True,
            "active_path_overwrite": False,
        },
        "results": results,
    }
    out = args.out_dir / "zigzag_second_stage_family_sweep_audit.json"
    audit["audit"] = str(out)
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"audit": str(out), "families": selected, "results": len(results)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
