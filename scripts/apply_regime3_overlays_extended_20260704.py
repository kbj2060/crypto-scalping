#!/usr/bin/env python3
"""Apply the EXISTING (frozen) cmamba-h6 prediction model and stability-risk model to the
extended 2026 funding-clean features file (Jan-Jun after the 2026-07-04 data extension),
regenerating their 2026 sidecar CSVs. No retraining -- reuses the saved .pt / .joblib artifacts
and each training script's own preprocessing functions, so the outputs are byte-compatible with
the original build on the overlapping Jan-Feb range (asserted before overwriting; backups kept).

Order matters: both models consume the wide24 current sidecar, which was already extended by
scripts/apply_regime3_wide24_sidecar_extended_20260704.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_regime3_cryptomamba_pred_20260531 as cm  # noqa: E402
import train_regime3_stability_risk_20260530 as sr  # noqa: E402

SOURCE_2026 = ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2026_rebuilt.csv"
CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"

CM_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531"
CM_CKPT = CM_DIR / "regime3_cryptomamba_pred_h6_nocurrent_20260531_2024.pt"
CM_OUT = CM_DIR / "training_features_2026_rebuilt_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv"

SR_DIR = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530"
SR_JOBLIB = SR_DIR / "regime3_stability_risk_h6.joblib"
SR_OUT = SR_DIR / "training_features_2026_rebuilt_regime3_stability_risk_h6.csv"


def _repro_check_and_write(new_df: pd.DataFrame, out_path: Path, tag: str, atol: float = 1e-6) -> None:
    existing = pd.read_csv(out_path, parse_dates=["timestamp"])
    merged = existing.merge(new_df, on="timestamp", suffixes=("_old", "_new"), how="inner")
    if len(merged) != len(existing):
        raise RuntimeError(f"{tag}: overlap join {len(merged)} != existing {len(existing)}")
    max_diff = 0.0
    for c in existing.columns:
        if c == "timestamp":
            continue
        old_v = merged[f"{c}_old"]
        new_v = merged[f"{c}_new"]
        if old_v.dtype == object or new_v.dtype == object:
            mismatch = (old_v.fillna("") != new_v.fillna("")).mean()
            if mismatch > 0.0:
                raise RuntimeError(f"{tag}: string column {c} mismatch rate {mismatch:.4f}")
            continue
        both = np.isfinite(old_v.to_numpy(dtype=np.float64)) & np.isfinite(new_v.to_numpy(dtype=np.float64))
        nan_pattern_same = (old_v.isna() == new_v.isna()).all()
        if not nan_pattern_same:
            raise RuntimeError(f"{tag}: NaN pattern differs in {c}")
        if both.any():
            diff = float(np.max(np.abs(old_v.to_numpy(dtype=np.float64)[both] - new_v.to_numpy(dtype=np.float64)[both])))
            max_diff = max(max_diff, diff)
    print(f"{tag}: reproducibility max abs diff on overlap = {max_diff:.3e}", flush=True)
    if max_diff > atol:
        raise RuntimeError(f"{tag}: reproducibility FAILED (max diff {max_diff} > {atol}) -- not overwriting")
    # write-to-temp-then-atomic-rename: never touch out_path until the new file is fully
    # written -- a crash mid-write used to leave only a stray rename()'d backup with no
    # canonical file at all (2026-08-12 incident, root-caused after the canonical CSV silently
    # went missing on both dev and server).
    tmp_path = out_path.with_name(out_path.name + ".tmp_write")
    new_df.to_csv(tmp_path, index=False)
    backup = out_path.with_suffix(".csv.bak_pre_extend_20260704")
    if not backup.exists():
        out_path.rename(backup)
        print(f"{tag}: backed up to {backup}", flush=True)
    tmp_path.rename(out_path)
    print(f"{tag}: wrote {len(new_df)} rows -> {out_path}", flush=True)


def apply_cmamba() -> None:
    print("=== cmamba h6 apply ===", flush=True)
    ckpt = torch.load(CM_CKPT, map_location="cpu", weights_only=False)
    cols = list(ckpt["feature_cols"])
    seq_len = int(ckpt["seq_len"])
    med = pd.Series(ckpt["feature_medians"])
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
    scale = np.asarray(ckpt["scaler_scale"], dtype=np.float64)

    frame = cm._merge_current(
        cm._add_volume_features(cm._add_rolling_stable_features(cm._read(SOURCE_2026))),
        cm._current_path(CURRENT_DIR, SOURCE_2026),
    )
    print(f"frame: {len(frame)} rows", flush=True)
    raw = frame[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = np.nan_to_num(((raw.fillna(med).fillna(0.0) - mean) / scale).to_numpy(dtype=np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cm.CryptoMambaRegimePred(len(cols), seq_len, int(ckpt["d_model"]), int(ckpt["cblocks"]), int(ckpt["cmblocks"]), int(ckpt["d_state"]), 0.10).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    idx = np.arange(seq_len - 1, len(frame), dtype=np.int64)
    proba = cm._predict(model, x, idx, seq_len, 2048, device)
    out = cm._output(frame, idx, proba)
    _repro_check_and_write(out, CM_OUT, "cmamba_pred_2026")


def apply_stability_risk() -> None:
    print("=== stability risk apply ===", flush=True)
    payload = joblib.load(SR_JOBLIB)
    cols = list(payload["feature_cols"])
    med = pd.Series(payload["feature_medians"])
    frame = sr._merge_current(
        sr._add_stability_features(sr._add_rolling_stable_features(sr._read(SOURCE_2026))),
        sr._current_path(CURRENT_DIR, SOURCE_2026),
    )
    print(f"frame: {len(frame)} rows", flush=True)
    _now, _future, transition, _risk = sr._labels(frame, int(payload["horizon"]), int(payload["min_duration"]))
    eval_frame = frame.iloc[: len(transition)].copy()
    x_eval, _, _ = sr._prepare(eval_frame, cols, scaler=payload["scaler"], medians=med)
    p_eval = sr._proba2(payload["transition_model"], x_eval)
    risk_eval = np.clip(payload["risk_model"].predict(x_eval), 0.0, 1.0)
    out = sr._output(eval_frame["timestamp"], p_eval, risk_eval, float(payload["threshold"]))
    _repro_check_and_write(out, SR_OUT, "stability_risk_2026")


def main() -> int:
    apply_cmamba()
    apply_stability_risk()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
