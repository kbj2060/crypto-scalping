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

from scripts.retrain_clean_regime_hmm_20260517 import _json_default  # noqa: E402
from scripts.train_regime3_hmm_mamba_20260529 import (  # noqa: E402
    CLASSES3,
    HORIZONS,
    MODEL_ID as BASE_MODEL_ID,
    PRED_PREFIX,
    RISK_COLS,
    RAW_PRIORITY,
    SeqDS,
    SharedMambaRegime,
    _eval_class,
    _fit_mamba,
    _future_labels_and_risk,
    _predict_mamba,
    _prepare,
    _prob_frame,
    _read,
    _risk_eval,
)


MODEL_ID = "regime3_pred_mamba_wide24_current_cleanfunding_20260529"
DEFAULT_TRAIN_2024 = ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv"
DEFAULT_TRANSFORMS = (
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv",
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2025.csv",
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2026_rebuilt.csv",
)
DEFAULT_CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_wide24_experiment_20260529"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime3_pred_mamba_wide24_current_cleanfunding_20260529"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime3_pred_mamba_wide24_current_cleanfunding_20260529_report.json"
CURRENT_PREFIX = "regime3_current_wide24_"
FORBIDDEN_PREFIXES = ("clean_regime_2024_unsup_v4_", "clean_regime4_2024_unsup_v1_", "clean_regime4_state24_sticky090_v2_", "regime4_pred_")
FORBIDDEN_FRAGMENTS = ("future", "target", "label", "realized", "trade_pnl", "cash_after", "legacy", "hdb", "hmm_")
NON_FEATURES = {"timestamp", "open", "high", "low", "close"}


def _current_path(current_dir: Path, source: Path) -> Path:
    return current_dir / f"{source.stem}_regime3_current_hmm_wide24.csv"


def _merge_current(frame: pd.DataFrame, current_path: Path) -> pd.DataFrame:
    current = _read(current_path)
    current_cols = ["timestamp"] + [c for c in current.columns if c.startswith(CURRENT_PREFIX)]
    missing = [
        f"{CURRENT_PREFIX}{name}_prob"
        for name in CLASSES3
        if f"{CURRENT_PREFIX}{name}_prob" not in current.columns
    ]
    if missing:
        raise ValueError(f"{current_path} missing required wide24 current columns: {missing}")
    out = frame.merge(current[current_cols], on="timestamp", how="left", validate="one_to_one")
    null_cols = [c for c in current_cols if c != "timestamp" and out[c].isna().any()]
    if null_cols:
        raise ValueError(f"wide24 current merge produced nulls: {null_cols[:10]}")
    out[f"{CURRENT_PREFIX}directional_bias"] = out[f"{CURRENT_PREFIX}bull_prob"] - out[f"{CURRENT_PREFIX}bear_prob"]
    out[f"{CURRENT_PREFIX}trend_prob"] = out[f"{CURRENT_PREFIX}bull_prob"] + out[f"{CURRENT_PREFIX}bear_prob"]
    out[f"{CURRENT_PREFIX}range_prob"] = out[f"{CURRENT_PREFIX}chop_prob"]
    return out


def _is_raw_feature(col: str) -> bool:
    lower = col.lower()
    if col in NON_FEATURES or lower.startswith("_"):
        return False
    if col.startswith(CURRENT_PREFIX):
        return True
    if any(col.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return False
    if "regime" in lower:
        return False
    if any(x in lower for x in FORBIDDEN_FRAGMENTS):
        return False
    return True


def _feature_cols(frames: list[pd.DataFrame], max_features: int) -> list[str]:
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common &= set(frame.columns)
    current_cols = sorted(c for c in common if c.startswith(CURRENT_PREFIX))
    raw_cols: list[str] = []
    for col in RAW_PRIORITY + sorted(common):
        if col in raw_cols or col in current_cols or col not in common or not _is_raw_feature(col):
            continue
        if pd.to_numeric(frames[0][col], errors="coerce").notna().any():
            raw_cols.append(col)
        if len(current_cols) + len(raw_cols) >= max_features:
            break
    cols = current_cols + raw_cols
    bad = [
        c
        for c in cols
        if any(c.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
        or ("regime" in c.lower() and not c.startswith(CURRENT_PREFIX))
    ]
    if bad:
        raise ValueError(f"forbidden regime features in wide24 PRED input: {bad[:10]}")
    return cols


def _labels_for_frame(frame: pd.DataFrame, max_h: int) -> tuple[dict[int, np.ndarray], np.ndarray, int]:
    n = len(frame) - max_h
    ys: dict[int, np.ndarray] = {}
    risks = []
    for horizon in HORIZONS:
        y, r = _future_labels_and_risk(frame, horizon)
        ys[horizon] = y[:n]
        risks.append(r[:n])
    return ys, np.mean(np.stack(risks, axis=0), axis=0).astype(np.float32), n


def _outputs(frame: pd.DataFrame, x: np.ndarray, model: torch.nn.Module, device: torch.device, args: argparse.Namespace) -> pd.DataFrame:
    idx = np.arange(len(frame), dtype=np.int64)
    probs, risk = _predict_mamba(model, DataLoader(SeqDS(x, idx, None, None, args.seq_len), batch_size=args.batch_size * 2, shuffle=False), device)
    out = pd.DataFrame({"timestamp": frame["timestamp"].reset_index(drop=True)})
    for horizon in HORIZONS:
        pf = _prob_frame(frame["timestamp"], f"{PRED_PREFIX}h{horizon}_", probs[horizon])
        out = out.merge(pf, on="timestamp", how="left")
    for i, name in enumerate(RISK_COLS):
        out[name] = risk[:, i]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train Regime3 PRED Mamba with confirmed wide24 current regime inputs.")
    p.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    p.add_argument("--transform", type=Path, action="append", default=None)
    p.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--val-start", default="2024-10-01")
    p.add_argument("--seq-len", type=int, default=72)
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=640)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--mamba-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-features", type=int, default=112)
    p.add_argument("--seed", type=int, default=8529)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    sources = list(args.transform or DEFAULT_TRANSFORMS)
    raw_frames = [_read(path) for path in sources]
    merged_frames = [_merge_current(frame, _current_path(args.current_dir, path)) for frame, path in zip(raw_frames, sources)]
    train = _merge_current(_read(args.train_2024), _current_path(args.current_dir, args.train_2024))

    cols = _feature_cols([train] + merged_frames, args.max_features)
    y_all, risk_all, n_valid = _labels_for_frame(train, max(HORIZONS))
    labeled = train.iloc[:n_valid].copy()
    ts = pd.to_datetime(labeled["timestamp"])
    first_val = int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]"), np.datetime64(pd.Timestamp(args.val_start))))
    train_end = max(args.seq_len - 1, first_val - max(HORIZONS))
    train_idx = np.arange(args.seq_len - 1, train_end, max(1, args.train_stride), dtype=np.int64)
    val_idx = np.arange(max(args.seq_len - 1, first_val), n_valid, dtype=np.int64)
    x, scaler, medians = _prepare(labeled, cols, fit_idx=train_idx)

    model, val_history, device = _fit_mamba(x, y_all, risk_all, train_idx, val_idx, args, args.seed)
    val_probs, val_risk = _predict_mamba(model, DataLoader(SeqDS(x, val_idx, None, None, args.seq_len), batch_size=args.batch_size * 2, shuffle=False), device)
    validation: dict[str, Any] = {f"h{h}": _eval_class(y_all[h][val_idx], val_probs[h]) for h in HORIZONS}
    validation["risk"] = _risk_eval(risk_all[val_idx], val_risk)

    best_epoch = int(min(val_history, key=lambda row: row.get("val_mean_log_loss", float("inf")))["epoch"])
    full_y, full_risk, full_n = _labels_for_frame(train, max(HORIZONS))
    full = train.iloc[:full_n].copy()
    x_full, scaler_full, medians_full = _prepare(full, cols)
    full_idx = np.arange(args.seq_len - 1, full_n, max(1, args.train_stride), dtype=np.int64)
    saved_epochs = args.epochs
    args.epochs = max(1, best_epoch)
    final_model, final_history, final_device = _fit_mamba(x_full, full_y, full_risk, full_idx, None, args, args.seed + 101)
    args.epochs = saved_epochs

    model_path = args.out_dir / "regime3_pred_mamba_wide24_current_2024.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "base_model_id": BASE_MODEL_ID,
            "classes": CLASSES3,
            "horizons": HORIZONS,
            "risk_cols": RISK_COLS,
            "current_prefix": CURRENT_PREFIX,
            "feature_cols": cols,
            "feature_medians": medians_full.to_dict(),
            "scaler_mean": scaler_full.mean_,
            "scaler_scale": scaler_full.scale_,
            "seq_len": int(args.seq_len),
            "d_model": int(args.d_model),
            "mamba_layers": int(args.mamba_layers),
            "state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
        },
        model_path,
    )

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "training_policy": "2024 fit; 2024Q4 validation for epoch selection; 2025/2026 are tests only",
        "current_input_contract": {
            "status": "wide24_confirmed_current",
            "prefix": CURRENT_PREFIX,
            "source_dir": str(args.current_dir),
        },
        "classes": CLASSES3,
        "horizons": HORIZONS,
        "feature_count": len(cols),
        "feature_cols": cols,
        "current_feature_count": int(sum(c.startswith(CURRENT_PREFIX) for c in cols)),
        "validation_history": val_history,
        "selected_final_epochs": best_epoch,
        "final_training_history": final_history,
        "validation": validation,
        "outputs": {},
        "leakage_audit": {
            "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
            "forbidden_regime_features": [c for c in cols if ("regime" in c.lower() and not c.startswith(CURRENT_PREFIX)) or any(c.startswith(p) for p in FORBIDDEN_PREFIXES)],
            "uses_2026_for_selection": False,
            "future_labels_use_future_path": True,
            "future_labels_are_targets_not_features": True,
        },
    }

    for path, frame in zip(sources, merged_frames):
        x_pred, _, _ = _prepare(frame, cols, scaler=scaler_full, medians=medians_full)
        pred = _outputs(frame, x_pred, final_model, final_device, args)
        out_path = args.out_dir / f"{path.stem}_regime3_pred_mamba_wide24_current.csv"
        pred.to_csv(out_path, index=False)
        y_eval, risk_eval, n_eval = _labels_for_frame(frame, max(HORIZONS))
        pred_eval = pred.iloc[:n_eval].copy()
        row: dict[str, Any] = {
            "source": str(path),
            "sidecar": str(out_path),
            "rows": int(len(frame)),
            "range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
        }
        for horizon in HORIZONS:
            pp = pred_eval[[f"{PRED_PREFIX}h{horizon}_{c}_prob" for c in CLASSES3]].to_numpy(float)
            row[f"future_h{horizon}_accuracy"] = _eval_class(y_eval[horizon], pp)
        row["risk_accuracy"] = _risk_eval(risk_eval, pred_eval[RISK_COLS].to_numpy(float))
        report["outputs"][path.name] = row
        print(f"[{MODEL_ID}] wrote {out_path}", flush=True)

    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] model={model_path}", flush=True)
    print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
