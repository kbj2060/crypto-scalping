#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ensemble.supervised.train_trend_xgb import XGBTrendBrain


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze trend_xgb lag against shifted returns")
    p.add_argument("--data-path", default="data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--model-path", default="data/ensemble/supervised/trend_xgb.json")
    p.add_argument("--output-path", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.data_path)
    brain = XGBTrendBrain.load(args.model_path)

    x = df.reindex(columns=brain.feature_cols, fill_value=np.nan).astype(np.float32)
    x = x.replace([np.inf, -np.inf], np.nan)
    probs = brain.model.predict_proba(x)
    signal = probs[:, 2] - probs[:, 0]

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
    ret1 = np.zeros_like(close, dtype=np.float64)
    ret1[1:] = np.diff(np.log(np.maximum(close, 1e-8)))

    results = []
    for lag in range(-3, 4):
        shifted = pd.Series(ret1).shift(-lag).to_numpy(dtype=np.float64)
        mask = np.isfinite(signal) & np.isfinite(shifted)
        if int(mask.sum()) < 100:
            continue
        corr = float(np.corrcoef(signal[mask], shifted[mask])[0, 1])
        down_hit = float(np.mean((probs[:, 0] > probs[:, 2])[mask] == (shifted[mask] < 0.0)))
        up_hit = float(np.mean((probs[:, 2] > probs[:, 0])[mask] == (shifted[mask] > 0.0)))
        dir_hit = float(np.mean(np.sign(signal[mask]) == np.sign(shifted[mask])))
        results.append(
            {
                "lag_bars": int(lag),
                "corr_signal_ret1": corr,
                "dir_hit": dir_hit,
                "up_hit": up_hit,
                "down_hit": down_hit,
            }
        )

    best_corr = max(results, key=lambda x: abs(x["corr_signal_ret1"])) if results else {}
    best_dir = max(results, key=lambda x: x["dir_hit"]) if results else {}
    summary = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "data_path": args.data_path,
        "model_path": args.model_path,
        "results": results,
        "best_abs_corr_lag": best_corr,
        "best_dir_hit_lag": best_dir,
    }

    output_path = args.output_path.strip()
    if not output_path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/ensemble/metrics/trend_xgb_lag_{stamp}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
