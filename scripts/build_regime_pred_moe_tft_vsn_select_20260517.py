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
from sklearn.metrics import balanced_accuracy_score, log_loss
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_regime_pred_moe_20260517 import (
    CLASSES,
    DEFAULT_PREDICT_2025,
    DEFAULT_TRAIN_2024,
    _future_path_frame,
    _json_default,
    _label_thresholds,
    _labels,
    _predicted_path_diagnostics,
)
from scripts.build_regime_pred_moe_tft_20260517 import (
    MODEL_ID as BASE_MODEL_ID,
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


MODEL_ID = "regime_pred_moe_tft_vsn_select_20260517"
DEFAULT_CLEAN_2024 = ROOT / "data/ensemble/supervised/clean_regime_raw_state12_v9_20260517/training_features_2024_clean_regime_raw_state12_v9.csv"
DEFAULT_CLEAN_2025 = ROOT / "data/ensemble/supervised/clean_regime_raw_state12_v9_20260517/training_features_2025_clean_regime_raw_state12_v9.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime_pred_moe_tft_vsn_select_20260517"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime_pred_moe_tft_vsn_select_20260517_report.json"


def _feature_gate_importance(
    model: torch.nn.Module,
    x: np.ndarray,
    known: np.ndarray,
    indices: np.ndarray,
    feature_cols: list[str],
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    loader = DataLoader(SequenceDataset(x, known, indices, None, seq_len), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    accum = np.zeros(len(feature_cols), dtype=np.float64)
    n = 0
    with torch.no_grad():
        for seq, _known in loader:
            seq = seq.to(device)
            gate = model.feature_gate(seq)
            score = (gate * seq.abs()).mean(dim=(0, 1)).detach().cpu().numpy()
            accum += score
            n += 1
    raw = accum / max(n, 1)
    norm = raw / max(float(raw.sum()), 1e-12)
    out = pd.DataFrame({"feature": feature_cols, "vsn_importance": norm, "raw_gate_abs_score": raw})
    return out.sort_values("vsn_importance", ascending=False).reset_index(drop=True)


def _select_features(importance: pd.DataFrame, threshold: float, min_selected: int, max_selected: int) -> tuple[list[str], str]:
    passing = importance[importance["vsn_importance"] >= float(threshold)]["feature"].tolist()
    policy = f"drop_importance_below_{threshold:g}"
    if len(passing) > int(max_selected):
        passing = importance.head(int(max_selected))["feature"].tolist()
        policy += f"_cap_top_{max_selected}"
    elif len(passing) < int(min_selected):
        passing = importance.head(int(min_selected))["feature"].tolist()
        policy += f"_floor_top_{min_selected}"
    return passing, policy


def _build_split(
    train_raw: pd.DataFrame,
    cols: list[str],
    *,
    horizon: int,
    seq_len: int,
    val_start: str,
    train_stride: int,
) -> dict[str, Any]:
    raw_ts = pd.to_datetime(train_raw["timestamp"])
    raw_train_mask = raw_ts < pd.Timestamp(val_start)
    threshold_path = _future_path_frame(train_raw.loc[raw_train_mask].copy(), int(horizon))
    train_only_thresholds = _label_thresholds(threshold_path)
    label_frame, label_meta = _labels(train_raw, int(horizon), thresholds=train_only_thresholds)
    train_labeled = train_raw.loc[label_frame.index].copy().join(label_frame[["_label_name", "_label_id"]])
    y = train_labeled["_label_id"].astype(int).to_numpy()
    ts = pd.to_datetime(train_labeled["timestamp"])
    first_val_idx = int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]"), np.datetime64(pd.Timestamp(val_start))))
    embargo = int(horizon)
    train_end = max(int(seq_len) - 1, first_val_idx - embargo)
    train_idx = np.arange(int(seq_len) - 1, train_end, max(1, int(train_stride)), dtype=np.int64)
    val_idx = np.arange(max(int(seq_len) - 1, first_val_idx), len(train_labeled), dtype=np.int64)
    if len(train_idx) < 1000 or len(val_idx) < 1000:
        split = int(len(train_labeled) * 0.80)
        train_idx = np.arange(int(seq_len) - 1, split, max(1, int(train_stride)), dtype=np.int64)
        val_idx = np.arange(split, len(train_labeled), dtype=np.int64)
    x, scaler, medians = _prepare_arrays(train_labeled, cols, fit_rows=train_idx)
    known = _known_future_covariates(train_labeled["timestamp"], int(horizon))
    return {
        "train_labeled": train_labeled,
        "label_meta": label_meta,
        "raw_train_rows_for_threshold": int(raw_train_mask.sum()),
        "y": y,
        "x": x,
        "known": known,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "scaler": scaler,
        "medians": medians,
        "embargo": embargo,
    }


def _fit_eval(split: dict[str, Any], cols: list[str], args: argparse.Namespace, device: torch.device, seed: int) -> dict[str, Any]:
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
    parser = argparse.ArgumentParser(description="TFT VSN-style feature selection experiment for future regime prediction.")
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
    parser.add_argument("--importance-threshold", type=float, default=0.01)
    parser.add_argument("--min-selected", type=int, default=30)
    parser.add_argument("--max-selected", type=int, default=35)
    parser.add_argument("--seed", type=int, default=2617)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_raw = _merge_clean(_read(args.train_2024), args.clean_2024)
    pred_raw = _merge_clean(_read(args.predict_2025), args.clean_2025)
    all_cols = _feature_cols(train_raw, pred_raw, int(args.max_features))
    all_split = _build_split(train_raw, all_cols, horizon=int(args.horizon), seq_len=int(args.seq_len), val_start=str(args.val_start), train_stride=int(args.train_stride))
    all_result = _fit_eval(all_split, all_cols, args, device, int(args.seed))

    importance = _feature_gate_importance(
        all_result["fit"].model,
        all_split["x"],
        all_split["known"],
        all_split["val_idx"],
        all_cols,
        seq_len=int(args.seq_len),
        batch_size=int(args.batch_size) * 2,
        device=device,
    )
    importance_path = args.out_dir / "regime_pred_tft_vsn_importance.csv"
    importance.to_csv(importance_path, index=False)
    selected_cols, selection_policy = _select_features(importance, float(args.importance_threshold), int(args.min_selected), int(args.max_selected))

    selected_split = _build_split(train_raw, selected_cols, horizon=int(args.horizon), seq_len=int(args.seq_len), val_start=str(args.val_start), train_stride=int(args.train_stride))
    selected_result = _fit_eval(selected_split, selected_cols, args, device, int(args.seed) + 11)

    full_label_frame, full_label_meta = _labels(train_raw, int(args.horizon))
    full_labeled = train_raw.loc[full_label_frame.index].copy().join(full_label_frame[["_label_name", "_label_id"]])
    y_full = full_labeled["_label_id"].astype(int).to_numpy()
    final_epochs = max(1, int(selected_result["best_epoch"]))
    x_full, scaler, medians = _prepare_arrays(full_labeled, selected_cols)
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

    pred_x, _, _ = _prepare_arrays(pred_raw, selected_cols, scaler=scaler, medians=medians)
    pred_known = _known_future_covariates(pred_raw["timestamp"], int(args.horizon))
    pred_idx = np.arange(len(pred_raw), dtype=np.int64)
    pred_loader = DataLoader(SequenceDataset(pred_x, pred_known, pred_idx, None, int(args.seq_len)), batch_size=int(args.batch_size) * 2, shuffle=False, num_workers=0)
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

    pred_sidecar = args.out_dir / f"{args.predict_2025.stem}_regime_pred_tft_vsn_selected.csv"
    train_sidecar = args.out_dir / f"{args.train_2024.stem}_regime_pred_tft_vsn_selected.csv"
    model_path = args.out_dir / "regime_pred_tft_vsn_selected_2024.pt"
    pred_output.to_csv(pred_sidecar, index=False)
    train_output.to_csv(train_sidecar, index=False)
    torch.save(
        {
            "model_id": MODEL_ID,
            "classes": CLASSES,
            "feature_cols": selected_cols,
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

    y_all_val = all_split["y"][all_split["val_idx"]]
    y_sel_val = selected_split["y"][selected_split["val_idx"]]
    report = {
        "model_id": MODEL_ID,
        "base_model_id": BASE_MODEL_ID,
        "model_path": str(model_path),
        "train_source": str(args.train_2024),
        "predict_source": str(args.predict_2025),
        "clean_2024": str(args.clean_2024),
        "clean_2025": str(args.clean_2025),
        "device": str(device),
        "horizon_bars": int(args.horizon),
        "seq_len": int(args.seq_len),
        "classes": CLASSES,
        "all_feature_count": int(len(all_cols)),
        "all_feature_cols": all_cols,
        "selected_feature_count": int(len(selected_cols)),
        "selected_feature_cols": selected_cols,
        "known_future_covariates": ["future_tod_sin", "future_tod_cos", "future_hour_sin", "future_hour_cos", "future_dow_sin", "future_dow_cos"],
        "importance_path": str(importance_path),
        "importance_threshold": float(args.importance_threshold),
        "selection_policy": selection_policy,
        "top_20_vsn_importance": importance.head(20).to_dict(orient="records"),
        "validation_label_meta": {
            **all_split["label_meta"],
            "threshold_policy": "thresholds_fit_on_pre_validation_2024_rows_only",
            "threshold_fit_rows": int(all_split["raw_train_rows_for_threshold"]),
            "embargo_bars": int(all_split["embargo"]),
        },
        "final_label_meta": {
            **full_label_meta,
            "threshold_policy": "thresholds_fit_on_all_2024_rows_for_final_2024_only_model",
        },
        "all_features": {
            "training_history": all_result["fit"].history,
            "best_epoch": int(all_result["best_epoch"]),
            "validation": all_result["validation"],
            "val_log_loss_check": float(log_loss(y_all_val, all_result["val_proba"], labels=list(range(len(CLASSES))))),
            "val_balanced_accuracy_check": float(balanced_accuracy_score(y_all_val, np.argmax(all_result["val_proba"], axis=1))),
        },
        "selected_features": {
            "training_history": selected_result["fit"].history,
            "best_epoch": int(selected_result["best_epoch"]),
            "validation": selected_result["validation"],
            "val_log_loss_check": float(log_loss(y_sel_val, selected_result["val_proba"], labels=list(range(len(CLASSES))))),
            "val_balanced_accuracy_check": float(balanced_accuracy_score(y_sel_val, np.argmax(selected_result["val_proba"], axis=1))),
        },
        "final_training_history": final_fit.history,
        "selected_final_epochs": int(final_epochs),
        "train_sidecar": str(train_sidecar),
        "predict_sidecar": str(pred_sidecar),
        "predict_probability_sum_min": float(pred_proba.sum(axis=1).min()),
        "predict_probability_sum_max": float(pred_proba.sum(axis=1).max()),
        "predict_counts": {CLASSES[i]: int((np.argmax(pred_proba, axis=1) == i).sum()) for i in range(len(CLASSES))},
        "predict_confidence_mean": float(pred_output[f"{PRED_PREFIX}confidence"].mean()),
        "predict_entropy_mean": float(pred_output[f"{PRED_PREFIX}entropy"].mean()),
        "predict_path_diagnostics": _predicted_path_diagnostics(pred_raw, pred_output, pred_proba, int(args.horizon)),
        "notes": [
            "pytorch_forecasting is not installed, so VSN importance is approximated with the TFT-lite feature gate on validation sequences.",
            "Importance is normalized to sum to 1 over encoder variables; features below threshold are removed, with a 30-35 feature band guard.",
            "Final 2025 sidecar is generated from the selected feature set only.",
            "No hard argmax label columns are written.",
        ],
    }
    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] all_features={len(all_cols)} selected={len(selected_cols)} policy={selection_policy}", flush=True)
    print(f"[{MODEL_ID}] importance={importance_path}", flush=True)
    print(f"[{MODEL_ID}] model={model_path}", flush=True)
    print(f"[{MODEL_ID}] train_sidecar={train_sidecar}", flush=True)
    print(f"[{MODEL_ID}] predict_sidecar={pred_sidecar}", flush=True)
    print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
