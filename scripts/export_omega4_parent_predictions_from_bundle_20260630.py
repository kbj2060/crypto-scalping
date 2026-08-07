#!/usr/bin/env python3
"""Export exact-threshold Omega4 parent predictions from an existing bundle.

This is an artifact-generation utility, not a trainer. It fails fast unless it
can reconstruct the parent frame from the saved parent report and, optionally,
the consuming risk-sidecar report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _resolve(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _qtag(value: float) -> str:
    return f"q{int(round(float(value) * 100.0)):03d}"


def _split_timestamps(frame: pd.DataFrame) -> pd.Series:
    if "timestamp" not in frame.columns:
        raise RuntimeError("prepared frame missing timestamp")
    return pd.to_datetime(frame["timestamp"], errors="raise").reset_index(drop=True)


def _assert_written_timestamps(path: Path, frame: pd.DataFrame) -> None:
    got = pd.to_datetime(pd.read_csv(path, usecols=["timestamp"])["timestamp"], errors="raise").reset_index(drop=True)
    expected = _split_timestamps(frame)
    if len(got) != len(expected) or not got.equals(expected):
        raise RuntimeError(f"{path}: timestamp contract mismatch")


@torch.no_grad()
def _prediction_frame(
    frame: pd.DataFrame,
    *,
    models: dict[str, dict[str, Any]],
    base_cols: list[str],
    threshold: float,
    prefix: str,
    device: torch.device,
) -> pd.DataFrame:
    x = parent._base_input(frame, base_cols)
    preds = {expert: parent._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = parent._routed(preds, route, "direction", 3)
    quality = parent._routed(preds, route, "quality", 3)
    return parent._prediction_output(frame, direction, quality, threshold=float(threshold), prefix=prefix)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--risk-report", type=Path, default=None)
    ap.add_argument("--train-csv", type=Path, default=None)
    ap.add_argument("--eval-csv", type=Path, default=None)
    ap.add_argument("--quality-thresholds", required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    parent_dir = _resolve(args.parent_dir)
    if parent_dir is None:
        raise RuntimeError("missing parent-dir")
    parent_report_path = parent_dir / "report.json"
    bundle_path = parent_dir / "true_3head_tabm_bundle.pt"
    parent_report = _read_json(parent_report_path)
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)

    risk_report_path = _resolve(args.risk_report)
    risk_report: dict[str, Any] = {}
    if risk_report_path is not None:
        risk_report = _read_json(risk_report_path)
        risk_model = risk_report.get("risk_model")
        if not isinstance(risk_model, dict):
            raise RuntimeError(f"{risk_report_path}: missing risk_model")
        risk_train_csv = _resolve(risk_model.get("train_csv"))
        risk_eval_csv = _resolve(risk_model.get("eval_csv"))
        if risk_train_csv is None or risk_eval_csv is None:
            raise RuntimeError(f"{risk_report_path}: missing train_csv/eval_csv")
        omega4.omega.TRAIN_CSV = risk_train_csv
        omega4.omega.EVAL_CSV = risk_eval_csv
    train_override = _resolve(args.train_csv)
    eval_override = _resolve(args.eval_csv)
    if train_override is not None:
        omega4.omega.TRAIN_CSV = train_override
    if eval_override is not None:
        omega4.omega.EVAL_CSV = eval_override

    label_contract = parent_report.get("label_contract")
    if not isinstance(label_contract, dict):
        raise RuntimeError(f"{parent_report_path}: missing label_contract")
    direction_label_dir = _resolve(label_contract.get("direction_label_dir"))
    quality_label_dir = _resolve(label_contract.get("quality_label_dir"))
    quality_mode = str(label_contract.get("quality_mode"))
    if direction_label_dir is None:
        raise RuntimeError(f"{parent_report_path}: missing direction_label_dir")

    quality_rule = parent_report.get("quality_target_rule") or {}
    risk_template = parent_report.get("risk_template") or {}
    frames = omega4._prepare_frames(
        disable_tp_sl=bool(risk_template.get("tp_sl_disabled", False)),
        direction_label_dir=direction_label_dir,
        quality_mode=quality_mode,
        quality_label_dir=quality_label_dir,
        quality_min_edge=float(quality_rule.get("net_return_after_cost_min", 0.0010)),
        quality_max_mae=float(quality_rule.get("mae_max", 0.0100)),
        quality_min_mfe_mae=float(quality_rule.get("mfe_mae_min", 1.20)),
        quality_max_hold_bars=int(quality_rule.get("max_hold_bars", 288)),
    )

    device = parent._device(str(args.device))
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    models = dict(bundle["models"])
    base_cols = list(bundle["base_cols"])
    missing_cols = sorted(set(base_cols) - set(frames["train_raw"].columns))
    if missing_cols:
        raise RuntimeError(f"{bundle_path}: prepared frame missing model columns: {missing_cols[:20]}")

    thresholds = [float(x.strip()) for x in str(args.quality_thresholds).split(",") if x.strip()]
    if not thresholds:
        raise RuntimeError("quality-thresholds is empty")

    exports: dict[str, dict[str, str]] = {}
    for q in thresholds:
        tag = _qtag(q)
        split_specs = {
            "train": (frames["train_raw"], "omega1_regime3_expertdq_oof"),
            "validation": (frames["val_raw"], "omega1_regime3_expertdq_oof"),
            "oos": (frames["oos_raw"], "omega1_regime3_expertdq"),
        }
        exports[tag] = {}
        for split, (frame, prefix) in split_specs.items():
            out_path = parent_dir / f"{split}_predictions_{tag}.csv"
            if out_path.exists() and not bool(args.overwrite):
                _assert_written_timestamps(out_path, frame)
                exports[tag][split] = str(out_path)
                continue
            pred = _prediction_frame(frame, models=models, base_cols=base_cols, threshold=q, prefix=prefix, device=device)
            pred.to_csv(out_path, index=False)
            _assert_written_timestamps(out_path, frame)
            exports[tag][split] = str(out_path)

    report = {
        "model_id": "omega4_parent_prediction_export_20260630",
        "source_parent_dir": str(parent_dir),
        "source_parent_report": str(parent_report_path),
        "source_bundle": str(bundle_path),
        "risk_report": str(risk_report_path) if risk_report_path is not None else None,
        "train_csv": str(omega4.omega.TRAIN_CSV),
        "eval_csv": str(omega4.omega.EVAL_CSV),
        "train_eval_override_used": train_override is not None or eval_override is not None,
        "quality_thresholds": thresholds,
        "prediction_artifacts": exports,
        "contract": {
            "tag_format": "qXXX where XXX is round(quality_threshold * 100)",
            "train_and_validation_prefix": "omega1_regime3_expertdq_oof",
            "oos_prefix": "omega1_regime3_expertdq",
            "timestamp_match_checked": True,
            "generated_from_existing_bundle": True,
            "not_a_retrain": True,
        },
    }
    report_path = parent_dir / f"prediction_export_{'_'.join(exports.keys())}_20260630.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "prediction_artifacts": exports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
