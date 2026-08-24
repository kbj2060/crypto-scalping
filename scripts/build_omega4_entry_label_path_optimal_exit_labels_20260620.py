#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402


DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_parent72_loose_20260620"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_entry_label_path_optimal_exit_labels_20260620"
SPLIT_TS = pd.Timestamp("2025-10-01")


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _load_train_frame_only() -> tuple[pd.DataFrame, dict[str, Any]]:
    train = omega._read(omega.TRAIN_CSV)
    train, train_current = omega._overlay_required(train, omega.REGIME3_CURRENT_2025, omega.REGIME3_CURRENT_COLS, tag="train_regime3_current")
    train, train_cmamba = omega._overlay_required(train, omega.REGIME3_CMAMBA_2025, omega.REGIME3_CMAMBA_COLS, tag="train_regime3_cmamba")
    train, train_risk = omega._overlay_required(train, omega.REGIME3_RISK_2025, omega.REGIME3_RISK_COLS, tag="train_regime3_risk")
    return train, {"train_current": train_current, "train_cmamba": train_cmamba, "train_risk": train_risk}


def _numeric_train_feature_cols(train: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in train.columns:
        if col in omega.NON_FEATURE_COLS:
            continue
        if omega._forbidden_feature(str(col)):
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            cols.append(str(col))
    if len(cols) < 80:
        raise RuntimeError(f"unexpectedly small train feature set: {len(cols)}")
    return cols


def _prepare_train_path_frame(label_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    train_all, overlay_report = _load_train_frame_only()
    labels = omega4._read_labels(label_dir, 2025, require_diagnostics=False)
    train_all, train_labels = omega._align(train_all, labels, "omega4 path-exit train labels")
    train_all = train_all.copy()
    train_all["zigzag_action"] = pd.to_numeric(train_labels["zigzag_action"], errors="raise").to_numpy(dtype="int64")
    train_raw = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    tabm_2025 = omega._read(omega.TABM_2025)
    train_df, _train_src = omega._align(train_raw, tabm_2025, "omega4 path-exit tabm train")
    feature_cols = _numeric_train_feature_cols(train_all)
    state = parent._base_input(train_df, feature_cols)
    return train_df, state, feature_cols, overlay_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction-label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--exit-edge-min", type=float, default=0.0020)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--quality-mode", choices=["same_as_direction", "hard_rule", "quality_label_action", "quality_label_hard_rule"], default="same_as_direction")
    args = ap.parse_args()

    del args.quality_mode
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    train_df, state, feature_cols, overlay_report = _prepare_train_path_frame(Path(args.direction_label_dir))
    fee, slip = omega._load_fee_slip()
    x_exit, y_exit, frame_exit, diag = omega4._build_exit_dataset_entry_label_path_optimal(
        train_df,
        state,
        risk_margin=None,
        risk_leverage=None,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        exit_edge_min=float(args.exit_edge_min),
        max_samples=int(args.max_samples),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = frame_exit.copy()
    labels["exit_action"] = y_exit
    labels["exit_action_name"] = ["EXIT" if int(v) == 1 else "HOLD" for v in y_exit]
    label_path = out_dir / "entry_label_path_optimal_exit_labels_2025_train.csv"
    feature_path = out_dir / "entry_label_path_optimal_exit_features_2025_train.csv"
    labels.to_csv(label_path, index=False)
    x_exit.to_csv(feature_path, index=False)
    report = {
        "label_id": "omega4_entry_label_path_optimal_exit_labels_20260620",
        "source_entry_label_dir": str(args.direction_label_dir),
        "split": "2025 train+validation frame aligned to Omega parent training sidecar",
        "label_contract": {
            "mode": "entry_label_path_optimal_stopping_every_in_position_bar",
            "time_axis": "one row for every bar while inside a contiguous non-cash entry-label segment",
            "oracle": "dynamic-programming suffix maximum of realized net exit value inside the same entry-label segment",
            "exit_rule": "EXIT if exit_now_net - best_future_net >= exit_edge_min; final segment bar is forced EXIT",
            "cash_rows": "excluded",
            "model_inputs": "position lifecycle features only; oracle columns are written to label CSV but not feature CSV",
        },
        "exit_edge_min": float(args.exit_edge_min),
        "cost_mult": float(args.cost_mult),
        "fee": float(fee),
        "slip": float(slip),
        "diag": diag,
        "feature_count": int(len(feature_cols)),
        "overlay_report": overlay_report,
        "artifacts": {
            "labels": str(label_path),
            "features": str(feature_path),
            "report": str(out_dir / "report.json"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "labels": str(label_path), "features": str(feature_path), "diag": diag}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
