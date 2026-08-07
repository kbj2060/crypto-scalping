#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ensemble.seven_model_ensemble import SevenModelEnsemble  # noqa: E402


@dataclass
class Summary:
    n_rows: int
    p_up_mean: float
    p_dn_mean: float
    action_long_ratio: float
    action_short_ratio: float
    action_hold_ratio: float
    iso_anom_ratio: float
    vae_anom_ratio: float
    gate_block_ratio: float
    conf_mean: float
    conf_std: float


def _ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask.astype(np.float64)))


def _summarize(pred: pd.DataFrame) -> Summary:
    act = pd.to_numeric(pred.get("m7_action", 0.0), errors="coerce").fillna(0.0).to_numpy()
    p_up = pd.to_numeric(pred.get("m7_prob_up", 0.0), errors="coerce").fillna(0.0).to_numpy()
    p_dn = pd.to_numeric(pred.get("m7_prob_dn", 0.0), errors="coerce").fillna(0.0).to_numpy()
    iso = pd.to_numeric(pred.get("m7_iso_anom", 0.0), errors="coerce").fillna(0.0).to_numpy()
    vae = pd.to_numeric(pred.get("m7_vae_anom", 0.0), errors="coerce").fillna(0.0).to_numpy()
    gate = pd.to_numeric(pred.get("m7_gate_block", 0.0), errors="coerce").fillna(0.0).to_numpy()
    conf = pd.to_numeric(pred.get("m7_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy()
    return Summary(
        n_rows=int(len(pred)),
        p_up_mean=float(np.mean(p_up)),
        p_dn_mean=float(np.mean(p_dn)),
        action_long_ratio=_ratio(act > 0),
        action_short_ratio=_ratio(act < 0),
        action_hold_ratio=_ratio(act == 0),
        iso_anom_ratio=_ratio(iso >= 0.5),
        vae_anom_ratio=_ratio(vae >= 0.5),
        gate_block_ratio=_ratio(gate >= 0.5),
        conf_mean=float(np.mean(conf)),
        conf_std=float(np.std(conf)),
    )


def _regime_bins(df: pd.DataFrame) -> dict[str, np.ndarray]:
    def _col(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.zeros(len(df), dtype=np.float64)
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    rb = _col("regime_bull")
    rr = _col("regime_bear")
    rw = _col("regime_whipsaw")
    rn = _col("regime_normal")
    labels = np.full(len(df), "unknown", dtype=object)
    mat = np.column_stack([rb, rr, rw, rn])
    idx = np.argmax(mat, axis=1)
    names = np.array(["bull", "bear", "whipsaw", "normal"], dtype=object)
    valid = np.max(mat, axis=1) > 0.0
    labels[valid] = names[idx[valid]]
    return {k: (labels == k) for k in ["bull", "bear", "whipsaw", "normal", "unknown"]}


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose M7 feature/action bias")
    p.add_argument("--csv", required=True, help="input feature csv")
    p.add_argument("--rows", type=int, default=30000, help="tail rows used for diagnosis")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--out", default="data/ensemble/diagnostics/m7_bias_report.json")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.rows > 0 and len(df) > args.rows:
        df = df.tail(args.rows).reset_index(drop=True)

    split = int(len(df) * float(args.train_ratio))
    df_train = df.iloc[:split].reset_index(drop=True)
    df_val = df.iloc[split:].reset_index(drop=True)

    m7 = SevenModelEnsemble()
    pred_train = m7.predict_batch(df_train)
    pred_val = m7.predict_batch(df_val)

    report: dict[str, object] = {
        "rows": {"train": int(len(df_train)), "val": int(len(df_val))},
        "train": _summarize(pred_train).__dict__,
        "val": _summarize(pred_val).__dict__,
        "regime": {},
        "low_variance_features": [],
    }

    reg_bins = _regime_bins(df_val)
    regime_stats: dict[str, object] = {}
    for name, mask in reg_bins.items():
        sub = pred_val.loc[mask]
        regime_stats[name] = _summarize(sub).__dict__
    report["regime"] = regime_stats

    check_cols = [
        "m7_prob_dn",
        "m7_prob_up",
        "m7_confidence",
        "m7_gmm_conf",
        "m7_gmm_vol_rank",
        "m7_iso_score",
        "m7_vae_error",
        "m7_composite_score",
    ]
    lows = []
    for c in check_cols:
        if c not in pred_val.columns:
            continue
        arr = pd.to_numeric(pred_val[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        std = float(np.std(arr))
        p01, p99 = np.percentile(arr, [1.0, 99.0])
        if std < 1e-4 or abs(float(p99 - p01)) < 1e-4:
            lows.append({"feature": c, "std": std, "p01": float(p01), "p99": float(p99)})
    report["low_variance_features"] = lows

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

