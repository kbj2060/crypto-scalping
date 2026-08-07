from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_model(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model_path = meta.get("model_path", "")
    threshold = float(meta.get("threshold", 0.5))
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(meta_path), model_path)
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    threshold = float(obj.get("threshold", threshold))
    return obj["model"], obj["feature_cols"], threshold


def main() -> None:
    ap = argparse.ArgumentParser(description="Append unified meta gate probability to csv")
    ap.add_argument("--csv-path", default="/home/llewyn/crypto-scalping/data/rl_training_2025_unified_supdir.csv")
    ap.add_argument("--model-meta-path", default="/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_meta_gate_catboost.json")
    ap.add_argument("--output-path", default="")
    args = ap.parse_args()

    out_path = args.output_path or args.csv_path
    df = pd.read_csv(args.csv_path)
    model, feature_cols, threshold = load_model(args.model_meta_path)
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True)
    x = x.fillna(med)
    prob = model.predict_proba(x)[:, 1]
    df["ud_gate_take_prob"] = prob
    df["ud_gate_pass"] = (prob >= threshold).astype(np.int8)
    df.to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
