#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_regime4_pred_tft_clean_target_20260517 import (  # noqa: E402
    CLASSES4,
    CLEAN4_PREFIX,
    PRED_PREFIX,
    SeqDS,
    TFTLite4,
    _known_cov,
    _output,
    _read,
)
from scripts.build_regime_pred_moe_20260517 import _json_default  # noqa: E402
from scripts.retrain_clean_regime4_hmm_raw_state12_20260517 import (  # noqa: E402
    PREFIX as CLEAN_PREFIX,
    _class_proba4,
    _output_frame,
)
from scripts.retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402


MODEL_ID = "transform_regime4_official_sidecars_20260517"
DEFAULT_SOURCE = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
DEFAULT_HMM_MODEL = ROOT / "data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/clean_regime4_raw_state12_v1_2024.joblib"
DEFAULT_TFT_MODEL = ROOT / "data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/regime4_pred_tft_vsn_selected_2024.pt"
DEFAULT_CLEAN_OUT = ROOT / "data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/training_features_2026_rebuilt_clean_regime4_raw_state12_v1.csv"
DEFAULT_PRED_OUT = ROOT / "data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2026_rebuilt_regime4_pred_tft_vsn_selected.csv"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime4_transform_2026_h12_nomdjd_all74_20260517.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transform a feature frame with frozen official Regime4 HMM and TFT artifacts.")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--hmm-model", type=Path, default=DEFAULT_HMM_MODEL)
    p.add_argument("--tft-model", type=Path, default=DEFAULT_TFT_MODEL)
    p.add_argument("--clean-out", type=Path, default=DEFAULT_CLEAN_OUT)
    p.add_argument("--pred-out", type=Path, default=DEFAULT_PRED_OUT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=72)
    p.add_argument("--batch-size", type=int, default=1536)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.12)
    p.add_argument(
        "--allow-missing-feature-median-fallback",
        action="store_true",
        help="Allow median-filled TFT inputs only when every missing feature is explicitly allowlisted.",
    )
    p.add_argument(
        "--allowed-missing-feature",
        action="append",
        default=[],
        help="Feature name allowed to use artifact median fallback. Repeat for each missing feature.",
    )
    return p.parse_args()


def _clean_transform(frame: pd.DataFrame, model_path: Path, out_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = joblib.load(model_path)
    feature_cols = list(payload["feature_cols"])
    medians = pd.Series(payload["feature_medians"], dtype=float)
    prepared = _with_raw_state12(frame.copy())
    raw = prepared[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    obs = payload["scaler"].transform(raw)
    state = payload["model"].filter_proba(obs)
    proba = _class_proba4(state, np.asarray(payload["state_class_matrix"], dtype=float))
    out = _output_frame(prepared["timestamp"], proba, prepared)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    pred = np.argmax(proba, axis=1)
    return out, {
        "model": str(model_path),
        "output": str(out_path),
        "rows": int(len(out)),
        "prob_sum_min": float(proba.sum(axis=1).min()),
        "prob_sum_max": float(proba.sum(axis=1).max()),
        "pred_counts": {CLASSES4[i]: int((pred == i).sum()) for i in range(len(CLASSES4))},
    }


def _matrix(frame: pd.DataFrame, cols: list[str], medians: pd.Series, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    raw = pd.DataFrame(
        {
            c: pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            for c in cols
        },
        index=frame.index,
    )
    filled = raw.fillna(medians).fillna(0.0).to_numpy(dtype=np.float32)
    x = (filled - mean.astype(np.float32)) / np.clip(scale.astype(np.float32), 1e-12, None)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _predict_tft(model: torch.nn.Module, x: np.ndarray, known: np.ndarray, seq_len: int, batch_size: int, device: torch.device) -> np.ndarray:
    idx = np.arange(len(x), dtype=np.int64)
    loader = DataLoader(SeqDS(x, known, idx, None, seq_len), batch_size=batch_size, shuffle=False)
    model.eval()
    rows = []
    with torch.no_grad():
        for seq, known_batch in loader:
            rows.append(torch.softmax(model(seq.to(device), known_batch.to(device)), dim=1).cpu().numpy())
    out = np.vstack(rows).astype(float)
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out


def _pred_transform(frame: pd.DataFrame, clean: pd.DataFrame, model_path: Path, out_path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    feature_cols = list(payload["feature_cols"])
    merged = frame.merge(clean[["timestamp"] + [c for c in clean.columns if c.startswith(CLEAN4_PREFIX)]], on="timestamp", how="left")
    missing = [c for c in feature_cols if c not in merged.columns]
    medians = pd.Series(payload["feature_medians"], dtype=float)
    fallback_policy = "fail_closed_no_missing"
    if missing:
        allowed_missing = {str(c).strip() for c in getattr(args, "allowed_missing_feature", []) if str(c).strip()}
        not_allowed = sorted(set(missing) - allowed_missing)
        if (not getattr(args, "allow_missing_feature_median_fallback", False)) or not_allowed:
            raise ValueError(
                "missing TFT feature columns for promoted Regime4 transform: "
                f"model={model_path} missing={missing} "
                "Use --allow-missing-feature-median-fallback plus one "
                "--allowed-missing-feature per missing field only for an explicit compatibility run."
            )
        for col in missing:
            merged[col] = float(medians.get(col, 0.0))
        fallback_policy = "artifact_training_median_explicit_allowlist"
    mean = np.asarray(payload["scaler_mean"], dtype=np.float32)
    scale = np.asarray(payload["scaler_scale"], dtype=np.float32)
    x = _matrix(merged, feature_cols, medians, mean, scale)
    known = _known_cov(merged["timestamp"], int(args.horizon))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TFTLite4(len(feature_cols), known.shape[1], int(args.d_model), int(args.heads), int(args.layers), float(args.dropout), int(args.seq_len)).to(device)
    model.load_state_dict(payload["state_dict"])
    proba = _predict_tft(model, x, known, int(args.seq_len), int(args.batch_size), device)
    out = _output(merged["timestamp"], proba)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    pred = np.argmax(proba, axis=1)
    return out, {
        "model": str(model_path),
        "output": str(out_path),
        "rows": int(len(out)),
        "feature_count": int(len(feature_cols)),
        "prob_sum_min": float(proba.sum(axis=1).min()),
        "prob_sum_max": float(proba.sum(axis=1).max()),
        "pred_counts": {CLASSES4[i]: int((pred == i).sum()) for i in range(len(CLASSES4))},
        "missing_feature_fallback": {
            "columns": missing,
            "policy": fallback_policy,
            "count": int(len(missing)),
        },
    }


def _prob_audit(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    cols = [f"{prefix}{name}_prob" for name in CLASSES4]
    sums = frame[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    return {"prob_sum_min": float(sums.min()), "prob_sum_max": float(sums.max()), "nan_count": int(frame[cols].isna().sum().sum())}


def main() -> int:
    args = parse_args()
    source = _read(args.source)
    clean, clean_report = _clean_transform(source, args.hmm_model, args.clean_out)
    pred, pred_report = _pred_transform(source, clean, args.tft_model, args.pred_out, args)
    report = {
        "model_id": MODEL_ID,
        "source": str(args.source),
        "rows": int(len(source)),
        "range": [str(source["timestamp"].iloc[0]), str(source["timestamp"].iloc[-1])],
        "classes": CLASSES4,
        "clean_regime4": {**clean_report, "audit": _prob_audit(clean, CLEAN_PREFIX)},
        "future_regime4": {**pred_report, "audit": _prob_audit(pred, PRED_PREFIX)},
        "contract": {
            "current_regime": "frozen 2024 HMM raw-state12",
            "future_regime": "frozen 2024 TFT VSN-selected h12",
            "horizon_bars": int(args.horizon),
            "normal_class": "disabled",
            "risk_off_transition_classes": "disabled",
            "risk_off_transition_auxiliary_features": "enabled_on_current_regime4_sidecar",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"clean": str(args.clean_out), "pred": str(args.pred_out), "report": str(args.report)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
