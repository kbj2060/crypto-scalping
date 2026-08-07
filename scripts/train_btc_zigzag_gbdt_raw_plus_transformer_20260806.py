"""Does the transformer's 32-dim deep-feature embedding add anything on top of raw features for
a tree model, or is it redundant/net-negative like the closed 2026-08-04 JEPA line found?

Baseline (train_btc_zigzag_gbdt_baseline_20260806.py): raw single-bar causalfix_final (113 cols)
-> HistGradientBoostingClassifier -> val 65.5% / OOS 63.4% hard top-1 acc, beating the standalone
transformer encoder (val 63.8% / OOS 60.2%).

This script trains the SAME GBDT config on raw (113) + transformer embedding (32) = 145 cols, on
the identical rows/split as both prior runs (window=48, train_stride=4), and reports whether the
combined feature set beats the raw-only baseline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset  # noqa: E402

EMBED_DIR = ROOT / "tmp/btc_deepfeat_encoders_20260806/transformer"
OUT_DIR = ROOT / "tmp/btc_deepfeat_encoders_20260806/gbdt_raw_plus_transformer"


def _load_embeddings(split: str, expected_ts: np.ndarray) -> np.ndarray:
    df = pd.read_parquet(EMBED_DIR / f"deepfeat_embeddings_{split}.parquet")
    if len(df) != len(expected_ts) or not (df["timestamp"].to_numpy() == expected_ts).all():
        raise RuntimeError(f"{split}: embedding rows/timestamps don't match current dataset build")
    return df.drop(columns=["timestamp"]).to_numpy(dtype=np.float32)


def main() -> int:
    ds = build_dataset(window=48, train_stride=4)

    def rows(split: str) -> np.ndarray:
        return ds.end_idx[split]

    results = {}
    Xs, ys = {}, {}
    for split in ("train", "val", "oos"):
        idx = rows(split)
        ts = ds.timestamps_all[idx]
        raw = ds.feat_std[idx]
        emb = _load_embeddings(split, ts)
        Xs[split] = np.concatenate([raw, emb], axis=1)
        ys[split] = ds.y_hard_all[idx]

    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=20260806,
    )
    clf.fit(Xs["train"], ys["train"])

    def _eval(split: str) -> dict:
        proba = clf.predict_proba(Xs[split])
        pred = clf.predict(Xs[split])
        y = ys[split]
        return {
            "n": int(len(y)),
            "hard_top1_acc": float((pred == y).mean()),
            "log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
        }

    for split in ("train", "val", "oos"):
        results[split] = _eval(split)
        print(f"{split}:", results[split])

    # feature importance split between raw (first 113) and embedding (last 32) blocks
    importances = clf.feature_importances_ if hasattr(clf, "feature_importances_") else None
    n_raw = Xs["train"].shape[1] - 32
    importance_summary = None
    if importances is not None:
        importance_summary = {
            "raw_block_importance_sum": float(np.sum(importances[:n_raw])),
            "embed_block_importance_sum": float(np.sum(importances[n_raw:])),
        }
        print("importance:", importance_summary)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "model": "sklearn.HistGradientBoostingClassifier",
                "features": "raw causalfix_final (113) + transformer deep-feature embedding (32) = 145 cols",
                "target": "zigzag_action (hard label)",
                "comparison_baseline": "train_btc_zigzag_gbdt_baseline_20260806.py (raw-only): val 65.5% / oos 63.4% hard top-1 acc",
                **results,
                "importance_summary": importance_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
