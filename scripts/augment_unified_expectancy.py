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
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(meta_path), model_path)
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["feature_cols"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Append unified expectancy prediction to csv")
    ap.add_argument("--csv-path", default="/home/llewyn/crypto-scalping/data/rl_training_2025_unified_supdir_cat.csv")
    ap.add_argument("--model-meta-path", default="/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_expectancy_catboost.json")
    ap.add_argument("--output-path", default="")
    args = ap.parse_args()
    out_path = args.output_path or args.csv_path
    df = pd.read_csv(args.csv_path)
    model, feature_cols = load_model(args.model_meta_path)
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True)
    x = x.fillna(med)
    pred = model.predict(x)
    df["ud_exp_ret_pred"] = pred
    df.to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
