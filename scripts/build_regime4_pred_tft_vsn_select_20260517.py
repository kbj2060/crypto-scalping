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

from scripts.build_regime4_pred_tft_clean_target_20260517 import (  # noqa: E402
    CLEAN4_PREFIX,
    CLASSES4,
    DEFAULT_CLEAN_2024,
    DEFAULT_CLEAN_2025,
    DEFAULT_OUT_DIR as _BASE_OUT_DIR,
    DEFAULT_PREDICT_2025,
    DEFAULT_TRAIN_2024,
    MODEL_ID as BASE_MODEL_ID,
    PRED_PREFIX,
    SeqDS,
    _clean_future_labels,
    _eval,
    _feature_cols,
    _fit,
    _known_cov,
    _merge_clean4,
    _output,
    _predict,
    _prepare_arrays,
    _read,
)
from scripts.build_regime_pred_moe_20260517 import _json_default  # noqa: E402


MODEL_ID = "regime4_pred_tft_vsn_select_20260517"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime4_pred_tft_vsn_select_20260517"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime4_pred_tft_vsn_select_20260517_report.json"


def _importance(model: torch.nn.Module, x: np.ndarray, known: np.ndarray, idx: np.ndarray, cols: list[str], args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    loader = DataLoader(SeqDS(x, known, idx, None, args.seq_len), batch_size=args.batch_size * 2, shuffle=False)
    model.eval()
    acc = np.zeros(len(cols), dtype=np.float64)
    n = 0
    with torch.no_grad():
        for seq, _known in loader:
            seq = seq.to(device)
            score = (model.feature_gate(seq) * seq.abs()).mean(dim=(0, 1)).cpu().numpy()
            acc += score
            n += 1
    raw = acc / max(n, 1)
    norm = raw / max(float(raw.sum()), 1e-12)
    return pd.DataFrame({"feature": cols, "vsn_importance": norm, "raw_gate_abs_score": raw}).sort_values("vsn_importance", ascending=False).reset_index(drop=True)


def _select(imp: pd.DataFrame, threshold: float, min_selected: int, max_selected: int) -> tuple[list[str], str]:
    cols = imp[imp["vsn_importance"] >= threshold]["feature"].tolist()
    policy = f"drop_importance_below_{threshold:g}"
    if len(cols) > max_selected:
        cols = imp.head(max_selected)["feature"].tolist()
        policy += f"_cap_top_{max_selected}"
    elif len(cols) < min_selected:
        cols = imp.head(min_selected)["feature"].tolist()
        policy += f"_floor_top_{min_selected}"
    return cols, policy


def _apply_excludes(cols: list[str], excludes: list[str]) -> list[str]:
    blocked = {str(c) for c in excludes}
    return [c for c in cols if c not in blocked]


def _split(train_raw: pd.DataFrame, cols: list[str], args: argparse.Namespace) -> dict[str, Any]:
    labels, meta = _clean_future_labels(train_raw, args.horizon)
    labeled = train_raw.loc[labels.index].copy().join(labels)
    y = labeled["_label_id"].astype(int).to_numpy()
    ts = pd.to_datetime(labeled["timestamp"])
    first_val = int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]"), np.datetime64(pd.Timestamp(args.val_start))))
    train_end = max(args.seq_len - 1, first_val - args.horizon)
    train_idx = np.arange(args.seq_len - 1, train_end, max(1, args.train_stride), dtype=np.int64)
    val_idx = np.arange(max(args.seq_len - 1, first_val), len(labeled), dtype=np.int64)
    x, scaler, medians = _prepare_arrays(labeled, cols, fit_rows=train_idx)
    known = _known_cov(labeled["timestamp"], args.horizon)
    return {"labeled": labeled, "meta": meta, "y": y, "x": x, "known": known, "train_idx": train_idx, "val_idx": val_idx, "scaler": scaler, "medians": medians}


def _fit_eval(split: dict[str, Any], args: argparse.Namespace, seed: int, device: torch.device) -> dict[str, Any]:
    model, hist = _fit(split["x"], split["known"], split["y"], split["train_idx"], split["val_idx"], args, seed, device)
    loader = DataLoader(SeqDS(split["x"], split["known"], split["val_idx"], split["y"], args.seq_len), batch_size=args.batch_size * 2, shuffle=False)
    p = _predict(model, loader, device)
    return {"model": model, "history": hist, "proba": p, "validation": _eval(split["y"][split["val_idx"]], p), "best_epoch": int(min(hist, key=lambda r: r.get("val_log_loss", float("inf")))["epoch"])}


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--exclude-feature", action="append", default=[], help="Feature column to exclude from the TFT input set. Can be repeated.")
    parser.add_argument("--seed", type=int, default=4517)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_raw = _merge_clean4(_read(args.train_2024), args.clean_2024)
    pred_raw = _merge_clean4(_read(args.predict_2025), args.clean_2025)
    excluded_features = list(args.exclude_feature or [])
    all_cols = _apply_excludes(_feature_cols(train_raw, pred_raw, args.max_features), excluded_features)
    all_split = _split(train_raw, all_cols, args)
    all_res = _fit_eval(all_split, args, args.seed, device)
    imp = _importance(all_res["model"], all_split["x"], all_split["known"], all_split["val_idx"], all_cols, args, device)
    imp_path = args.out_dir / "regime4_pred_tft_vsn_importance.csv"
    imp.to_csv(imp_path, index=False)
    selected_cols, policy = _select(imp, args.importance_threshold, args.min_selected, args.max_selected)
    selected_cols = _apply_excludes(selected_cols, excluded_features)
    if excluded_features:
        policy += "_exclude_" + "_".join(excluded_features)

    selected_split = _split(train_raw, selected_cols, args)
    sel_res = _fit_eval(selected_split, args, args.seed + 11, device)

    labels, full_meta = _clean_future_labels(train_raw, args.horizon)
    full = train_raw.loc[labels.index].copy().join(labels)
    y_full = full["_label_id"].astype(int).to_numpy()
    best_epoch = max(1, int(sel_res["best_epoch"]))
    old_epochs = args.epochs
    args.epochs = best_epoch
    x_full, scaler, medians = _prepare_arrays(full, selected_cols)
    known_full = _known_cov(full["timestamp"], args.horizon)
    full_idx = np.arange(args.seq_len - 1, len(full), max(1, args.train_stride), dtype=np.int64)
    final_model, final_hist = _fit(x_full, known_full, y_full, full_idx, None, args, args.seed + 101, device)
    args.epochs = old_epochs

    pred_x, _, _ = _prepare_arrays(pred_raw, selected_cols, scaler=scaler, medians=medians)
    pred_known = _known_cov(pred_raw["timestamp"], args.horizon)
    pred_p = _predict(final_model, DataLoader(SeqDS(pred_x, pred_known, np.arange(len(pred_raw)), None, args.seq_len), batch_size=args.batch_size * 2, shuffle=False), device)
    full_p = _predict(final_model, DataLoader(SeqDS(x_full, known_full, np.arange(len(full)), None, args.seq_len), batch_size=args.batch_size * 2, shuffle=False), device)

    pred_sidecar = args.out_dir / f"{args.predict_2025.stem}_regime4_pred_tft_vsn_selected.csv"
    train_sidecar = args.out_dir / f"{args.train_2024.stem}_regime4_pred_tft_vsn_selected.csv"
    model_path = args.out_dir / "regime4_pred_tft_vsn_selected_2024.pt"
    _output(pred_raw["timestamp"], pred_p).to_csv(pred_sidecar, index=False)
    _output(full["timestamp"], full_p).to_csv(train_sidecar, index=False)
    torch.save({"model_id": MODEL_ID, "classes": CLASSES4, "feature_cols": selected_cols, "feature_medians": medians.to_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()}}, model_path)

    report = {
        "model_id": MODEL_ID,
        "base_model_id": BASE_MODEL_ID,
        "model_path": str(model_path),
        "classes": CLASSES4,
        "clean_2024": str(args.clean_2024),
        "clean_2025": str(args.clean_2025),
        "horizon_bars": args.horizon,
        "seq_len": args.seq_len,
        "all_feature_count": len(all_cols),
        "all_feature_cols": all_cols,
        "selected_feature_count": len(selected_cols),
        "selected_feature_cols": selected_cols,
        "selection_policy": policy,
        "excluded_features": excluded_features,
        "importance_path": str(imp_path),
        "top_20_vsn_importance": imp.head(20).to_dict(orient="records"),
        "validation_label_meta": {**all_split["meta"], "split_policy": "Q4_2024_validation_with_horizon_embargo", "embargo_bars": args.horizon},
        "all_features": {"training_history": all_res["history"], "best_epoch": all_res["best_epoch"], "validation": all_res["validation"]},
        "selected_features": {"training_history": sel_res["history"], "best_epoch": sel_res["best_epoch"], "validation": sel_res["validation"]},
        "final_label_meta": full_meta,
        "selected_final_epochs": best_epoch,
        "final_training_history": final_hist,
        "train_sidecar": str(train_sidecar),
        "predict_sidecar": str(pred_sidecar),
        "predict_probability_sum_min": float(pred_p.sum(axis=1).min()),
        "predict_probability_sum_max": float(pred_p.sum(axis=1).max()),
        "predict_counts": {CLASSES4[i]: int((np.argmax(pred_p, axis=1) == i).sum()) for i in range(len(CLASSES4))},
        "predict_confidence_mean": float(_output(pred_raw["timestamp"], pred_p)[f"{PRED_PREFIX}confidence"].mean()),
        "predict_entropy_mean": float(_output(pred_raw["timestamp"], pred_p)[f"{PRED_PREFIX}entropy"].mean()),
        "notes": ["VSN-style importance uses TFT-lite feature gate because pytorch_forecasting is not installed."],
    }
    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] all_features={len(all_cols)} selected={len(selected_cols)} policy={policy}", flush=True)
    print(f"[{MODEL_ID}] importance={imp_path}", flush=True)
    print(f"[{MODEL_ID}] predict_sidecar={pred_sidecar}", flush=True)
    print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
