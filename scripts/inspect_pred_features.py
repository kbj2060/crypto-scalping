#!/usr/bin/env python3
"""Inspect pred_*/conf_* features in rl_training_data_full.csv.

The script is diagnostic-only. It does not modify training code.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect pred_* features for redundancy/noise")
    p.add_argument("--csv", default="data/rl_training_data_full.csv")
    p.add_argument("--target-col", default="log_return")
    p.add_argument("--future-bars", type=int, default=12)
    p.add_argument("--const-std-th", type=float, default=1e-4)
    p.add_argument("--high-corr-th", type=float, default=0.90)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out-json", default="")
    return p.parse_args()


def _safe_corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 3:
        return 0.0
    v = joined.iloc[:, 0].corr(joined.iloc[:, 1], method=method)
    return float(0.0 if pd.isna(v) else v)


def _top_bottom_spread(signal: pd.Series, target: pd.Series, q: float = 0.2) -> float:
    joined = pd.concat([signal, target], axis=1).dropna()
    if len(joined) < 20:
        return 0.0
    sig = joined.iloc[:, 0]
    tgt = joined.iloc[:, 1]
    lo = float(sig.quantile(q))
    hi = float(sig.quantile(1.0 - q))
    top = tgt[sig >= hi]
    bot = tgt[sig <= lo]
    if len(top) == 0 or len(bot) == 0:
        return 0.0
    return float(top.mean() - bot.mean())


def _sign_hit_rate(signal: pd.Series, target: pd.Series, min_abs_signal: float = 0.0) -> float:
    joined = pd.concat([signal, target], axis=1).dropna()
    if len(joined) < 3:
        return 0.0
    sig = joined.iloc[:, 0]
    tgt = joined.iloc[:, 1]
    mask = sig.abs() > float(min_abs_signal)
    if mask.sum() == 0:
        return 0.0
    return float((np.sign(sig[mask]) == np.sign(tgt[mask])).mean())


def _mode_recommendation(rows: list[dict], const_std_th: float, top_k: int) -> dict:
    near_const = [r["feature"] for r in rows if r["std"] <= const_std_th]
    ranked = sorted(
        rows,
        key=lambda r: (
            abs(r["future_corr_12"]),
            abs(r["future_spearman_12"]),
            abs(r["effective_future_corr_12"]),
            r["std"],
        ),
        reverse=True,
    )
    keep = [r["feature"] for r in ranked if r["feature"] not in near_const][:top_k]
    drop = [r["feature"] for r in ranked if r["feature"] not in keep]
    return {
        "keep_candidates": keep,
        "drop_candidates": drop,
        "near_constant": near_const,
    }


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    conf_cols = [c for c in df.columns if c.startswith("conf_")]
    if not pred_cols:
        raise ValueError("no pred_* columns found")
    if args.target_col not in df.columns:
        raise ValueError(f"target column not found: {args.target_col}")

    for col in pred_cols + conf_cols + [args.target_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    future_1 = df[args.target_col].shift(-1)
    future_n = df[args.target_col].shift(-1).rolling(args.future_bars, min_periods=args.future_bars).sum()

    pred_df = df[pred_cols]
    abs_corr = pred_df.corr().abs()
    high_corr_pairs = []
    for i, c1 in enumerate(pred_cols):
        for c2 in pred_cols[i + 1 :]:
            v = float(abs_corr.loc[c1, c2])
            if v >= float(args.high_corr_th):
                high_corr_pairs.append({"a": c1, "b": c2, "abs_corr": v})
    high_corr_pairs.sort(key=lambda x: x["abs_corr"], reverse=True)

    rows: list[dict] = []
    for pred in pred_cols:
        conf = pred.replace("pred_", "conf_")
        conf_s = df[conf] if conf in df.columns else pd.Series(1.0, index=df.index)
        eff = df[pred] * conf_s.fillna(0.0)
        row = {
            "feature": pred,
            "conf_feature": conf if conf in df.columns else "",
            "std": float(df[pred].std(skipna=True) or 0.0),
            "mean": float(df[pred].mean(skipna=True) or 0.0),
            "conf_mean": float(conf_s.mean(skipna=True) or 0.0),
            "current_corr": _safe_corr(df[pred], df[args.target_col]),
            "future_corr_1": _safe_corr(df[pred], future_1),
            "future_corr_12": _safe_corr(df[pred], future_n),
            "future_spearman_12": _safe_corr(df[pred], future_n, method="spearman"),
            "effective_future_corr_12": _safe_corr(eff, future_n),
            "sign_hit_1": _sign_hit_rate(df[pred], future_1),
            "sign_hit_12": _sign_hit_rate(df[pred], future_n),
            "top_bottom_spread_12": _top_bottom_spread(df[pred], future_n),
            "effective_top_bottom_spread_12": _top_bottom_spread(eff, future_n),
        }
        rows.append(row)

    rows.sort(key=lambda r: abs(r["future_corr_12"]), reverse=True)
    recommendation = _mode_recommendation(rows, float(args.const_std_th), int(args.top_k))

    summary = {
        "config": {
            "csv": args.csv,
            "target_col": args.target_col,
            "future_bars": int(args.future_bars),
            "const_std_th": float(args.const_std_th),
            "high_corr_th": float(args.high_corr_th),
            "top_k": int(args.top_k),
            "rows": int(len(df)),
            "pred_cols": pred_cols,
            "conf_cols": conf_cols,
        },
        "dataset": {
            "target_mean": float(df[args.target_col].mean(skipna=True) or 0.0),
            "target_std": float(df[args.target_col].std(skipna=True) or 0.0),
            "future_12_mean": float(future_n.mean(skipna=True) or 0.0),
            "future_12_std": float(future_n.std(skipna=True) or 0.0),
        },
        "feature_rows": rows,
        "high_corr_pairs": high_corr_pairs,
        "recommendation": recommendation,
    }

    out_json = args.out_json.strip()
    if not out_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = f"data/ensemble/metrics/pred_feature_inspection_{ts}.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("== pred_* inspection ==")
    print(f"rows: {len(df)} | preds: {len(pred_cols)} | future_bars: {args.future_bars}")
    print(f"high_corr_pairs(>={args.high_corr_th}): {len(high_corr_pairs)}")
    for row in rows[: max(1, int(args.top_k))]:
        print(
            f"{row['feature']:<16} std={row['std']:.6f} "
            f"f12_corr={row['future_corr_12']:+.5f} "
            f"eff_f12_corr={row['effective_future_corr_12']:+.5f} "
            f"spread12={row['top_bottom_spread_12']:+.6f}"
        )
    print("keep_candidates:", ", ".join(recommendation["keep_candidates"]))
    if recommendation["near_constant"]:
        print("near_constant:", ", ".join(recommendation["near_constant"]))
    print(f"saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
