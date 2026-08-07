#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_regime_pred_moe_20260517 import (  # noqa: E402
    CLEAN_PREFIX,
    CLASSES,
    DEFAULT_PREDICT_2025,
    DEFAULT_TRAIN_2024,
    _json_default,
)
from scripts.build_regime_pred_moe_tft_20260517 import (  # noqa: E402
    PRED_PREFIX,
    SequenceDataset,
    _eval_report,
    _feature_cols,
    _fit_model,
    _known_future_covariates,
    _merge_clean,
    _output_frame,
    _predict,
    _prepare_arrays,
    _read,
)


MODEL_ID = "regime_pred_moe_tft_clean_target_20260517"
DEFAULT_CLEAN_2024 = ROOT / "data/ensemble/supervised/clean_regime_raw_state12_v9_20260517/training_features_2024_clean_regime_raw_state12_v9.csv"
DEFAULT_CLEAN_2025 = ROOT / "data/ensemble/supervised/clean_regime_raw_state12_v9_20260517/training_features_2025_clean_regime_raw_state12_v9.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime_pred_moe_tft_clean_target_20260517"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime_pred_moe_tft_clean_target_20260517_report.json"


def _clean_prob_cols() -> list[str]:
    return [f"{CLEAN_PREFIX}{name}_prob" for name in CLASSES]


def _clean_future_labels(frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    prob_cols = _clean_prob_cols()
    missing = [c for c in prob_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"missing clean regime probability columns: {missing}")
    probs = frame[prob_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    n = len(frame)
    valid_n = max(0, n - int(horizon))
    target = np.argmax(probs[int(horizon) : int(horizon) + valid_n], axis=1).astype(int)
    target_conf = probs[int(horizon) : int(horizon) + valid_n].max(axis=1)
    out = pd.DataFrame(
        {
            "_label_id": target,
            "_label_name": [CLASSES[i] for i in target],
            "_target_clean_confidence": target_conf,
        },
        index=frame.index[:valid_n],
    )
    counts = out["_label_name"].value_counts().reindex(CLASSES, fill_value=0)
    meta = {
        "horizon": int(horizon),
        "label_source": "argmax_clean_regime_raw_state12_v9_at_t_plus_horizon",
        "label_counts": {k: int(v) for k, v in counts.items()},
        "label_share": {k: float(v / max(len(out), 1)) for k, v in counts.items()},
        "target_clean_confidence_mean": float(out["_target_clean_confidence"].mean()) if len(out) else 0.0,
    }
    return out, meta


def _build_split(
    train_raw: pd.DataFrame,
    cols: list[str],
    *,
    horizon: int,
    seq_len: int,
    val_start: str,
    train_stride: int,
) -> dict[str, Any]:
    label_frame, label_meta = _clean_future_labels(train_raw, int(horizon))
    labeled = train_raw.loc[label_frame.index].copy().join(label_frame)
    y = labeled["_label_id"].astype(int).to_numpy()
    ts = pd.to_datetime(labeled["timestamp"])
    first_val_idx = int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]"), np.datetime64(pd.Timestamp(val_start))))
    embargo = int(horizon)
    train_end = max(int(seq_len) - 1, first_val_idx - embargo)
    train_idx = np.arange(int(seq_len) - 1, train_end, max(1, int(train_stride)), dtype=np.int64)
    val_idx = np.arange(max(int(seq_len) - 1, first_val_idx), len(labeled), dtype=np.int64)
    if len(train_idx) < 1000 or len(val_idx) < 1000:
        split = int(len(labeled) * 0.80)
        train_idx = np.arange(int(seq_len) - 1, split, max(1, int(train_stride)), dtype=np.int64)
        val_idx = np.arange(split, len(labeled), dtype=np.int64)
    x, scaler, medians = _prepare_arrays(labeled, cols, fit_rows=train_idx)
    known = _known_future_covariates(labeled["timestamp"], int(horizon))
    return {
        "labeled": labeled,
        "label_meta": label_meta,
        "y": y,
        "x": x,
        "known": known,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "scaler": scaler,
        "medians": medians,
        "embargo": embargo,
    }


def _fit_eval(split: dict[str, Any], args: argparse.Namespace, device: torch.device, seed: int) -> dict[str, Any]:
    fit = _fit_model(
        split["x"],
        split["known"],
        split["y"],
        split["train_idx"],
        split["val_idx"],
        seq_len=int(args.seq_len),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        d_model=int(args.d_model),
        n_heads=int(args.heads),
        n_layers=int(args.layers),
        dropout=float(args.dropout),
        seed=int(seed),
        device=device,
    )
    val_loader = DataLoader(
        SequenceDataset(split["x"], split["known"], split["val_idx"], split["y"], int(args.seq_len)),
        batch_size=int(args.batch_size) * 2,
        shuffle=False,
        num_workers=0,
    )
    val_proba = _predict(fit.model, val_loader, device)
    return {
        "fit": fit,
        "val_proba": val_proba,
        "validation": _eval_report(split["y"][split["val_idx"]], val_proba),
        "best_epoch": int(min(fit.history, key=lambda row: row.get("val_log_loss", float("inf")))["epoch"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Future regime predictor trained on t+h clean-regime targets.")
    parser.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    parser.add_argument("--predict-2025", type=Path, default=DEFAULT_PREDICT_2025)
    parser.add_argument("--clean-2024", type=Path, default=DEFAULT_CLEAN_2024)
    parser.add_argument("--clean-2025", type=Path, default=DEFAULT_CLEAN_2025)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--seq-len", type=int, default=72)
    parser.add_argument("--val-start", default="2024-10-01")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--train-stride", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--max-features", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2717)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_raw = _merge_clean(_read(args.train_2024), args.clean_2024)
    pred_raw = _merge_clean(_read(args.predict_2025), args.clean_2025)
    cols = _feature_cols(train_raw, pred_raw, int(args.max_features))
    split = _build_split(train_raw, cols, horizon=int(args.horizon), seq_len=int(args.seq_len), val_start=str(args.val_start), train_stride=int(args.train_stride))
    val_result = _fit_eval(split, args, device, int(args.seed))

    full_label_frame, full_label_meta = _clean_future_labels(train_raw, int(args.horizon))
    full_labeled = train_raw.loc[full_label_frame.index].copy().join(full_label_frame)
    y_full = full_labeled["_label_id"].astype(int).to_numpy()
    final_epochs = max(1, int(val_result["best_epoch"]))
    x_full, scaler, medians = _prepare_arrays(full_labeled, cols)
    known_full = _known_future_covariates(full_labeled["timestamp"], int(args.horizon))
    full_idx = np.arange(int(args.seq_len) - 1, len(full_labeled), max(1, int(args.train_stride)), dtype=np.int64)
    final_fit = _fit_model(
        x_full,
        known_full,
        y_full,
        full_idx,
        None,
        seq_len=int(args.seq_len),
        epochs=final_epochs,
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        d_model=int(args.d_model),
        n_heads=int(args.heads),
        n_layers=int(args.layers),
        dropout=float(args.dropout),
        seed=int(args.seed) + 101,
        device=device,
    )

    pred_x, _, _ = _prepare_arrays(pred_raw, cols, scaler=scaler, medians=medians)
    pred_known = _known_future_covariates(pred_raw["timestamp"], int(args.horizon))
    pred_loader = DataLoader(
        SequenceDataset(pred_x, pred_known, np.arange(len(pred_raw), dtype=np.int64), None, int(args.seq_len)),
        batch_size=int(args.batch_size) * 2,
        shuffle=False,
        num_workers=0,
    )
    pred_proba = _predict(final_fit.model, pred_loader, device)
    pred_output = _output_frame(pred_raw["timestamp"], pred_proba)

    full_loader = DataLoader(
        SequenceDataset(x_full, known_full, np.arange(len(full_labeled), dtype=np.int64), None, int(args.seq_len)),
        batch_size=int(args.batch_size) * 2,
        shuffle=False,
        num_workers=0,
    )
    full_proba = _predict(final_fit.model, full_loader, device)
    train_output = _output_frame(full_labeled["timestamp"], full_proba)

    pred_sidecar = args.out_dir / f"{args.predict_2025.stem}_regime_pred_tft_clean_target.csv"
    train_sidecar = args.out_dir / f"{args.train_2024.stem}_regime_pred_tft_clean_target.csv"
    model_path = args.out_dir / "regime_pred_tft_clean_target_2024.pt"
    pred_output.to_csv(pred_sidecar, index=False)
    train_output.to_csv(train_sidecar, index=False)
    torch.save(
        {
            "model_id": MODEL_ID,
            "classes": CLASSES,
            "feature_cols": cols,
            "feature_medians": medians.to_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "state_dict": {k: v.detach().cpu() for k, v in final_fit.model.state_dict().items()},
            "horizon": int(args.horizon),
            "seq_len": int(args.seq_len),
            "d_model": int(args.d_model),
            "heads": int(args.heads),
            "layers": int(args.layers),
            "dropout": float(args.dropout),
        },
        model_path,
    )

    report = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "train_source": str(args.train_2024),
        "predict_source": str(args.predict_2025),
        "clean_2024": str(args.clean_2024),
        "clean_2025": str(args.clean_2025),
        "device": str(device),
        "horizon_bars": int(args.horizon),
        "seq_len": int(args.seq_len),
        "classes": CLASSES,
        "feature_count": int(len(cols)),
        "feature_cols": cols,
        "validation_label_meta": {
            **split["label_meta"],
            "split_policy": "Q4_2024_validation_with_horizon_embargo",
            "embargo_bars": int(split["embargo"]),
        },
        "final_label_meta": full_label_meta,
        "validation_training_history": val_result["fit"].history,
        "validation": val_result["validation"],
        "selected_final_epochs": int(final_epochs),
        "final_training_history": final_fit.history,
        "train_sidecar": str(train_sidecar),
        "predict_sidecar": str(pred_sidecar),
        "predict_probability_sum_min": float(pred_proba.sum(axis=1).min()),
        "predict_probability_sum_max": float(pred_proba.sum(axis=1).max()),
        "predict_counts": {CLASSES[i]: int((np.argmax(pred_proba, axis=1) == i).sum()) for i in range(len(CLASSES))},
        "predict_confidence_mean": float(pred_output[f"{PRED_PREFIX}confidence"].mean()),
        "predict_entropy_mean": float(pred_output[f"{PRED_PREFIX}entropy"].mean()),
        "notes": [
            "Target is the v9 clean-regime argmax at t+horizon, not future MFE/MAE path labels.",
            "This aligns future regime prediction with the current clean HMM taxonomy.",
            "No hard argmax label columns are written to sidecars.",
        ],
    }
    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] model={model_path}", flush=True)
    print(f"[{MODEL_ID}] train_sidecar={train_sidecar}", flush=True)
    print(f"[{MODEL_ID}] predict_sidecar={pred_sidecar}", flush=True)
    print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
