"""Raw-feature GBDT baseline for the BTC deep-feature line (docs/btc_deepfeat_cnn_transformer_zigzag_soft_label_20260806.md).

Transformer was chosen as the winning encoder architecture (63.8% val / 60.2% OOS hard top-1 acc
vs 50.6% baseline). Before wiring it into a strategy, check whether a plain gradient-boosted tree
on the SAME raw single-bar features (no windowing, no deep encoder) already gets there -- the
earlier closed JEPA line found deep embeddings ranked *weaker* than raw features when both were
fed to a tree model, so this comparison is not optional.

Uses sklearn's HistGradientBoostingClassifier (LightGBM is not installed in this environment;
HGB is the closest built-in equivalent -- both are histogram-based GBDTs). Reuses
btc_deepfeat_dataset_20260806.build_dataset() for identical rows/splits/standardization/
train_stride as the transformer run, so results are directly comparable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset  # noqa: E402

OUT_DIR = ROOT / "tmp/btc_deepfeat_encoders_20260806/gbdt_raw_baseline"


def main() -> int:
    ds = build_dataset(window=48, train_stride=4)

    def rows(split: str) -> np.ndarray:
        return ds.end_idx[split]

    X_train = ds.feat_std[rows("train")]
    y_train = ds.y_hard_all[rows("train")]
    X_val = ds.feat_std[rows("val")]
    y_val = ds.y_hard_all[rows("val")]
    X_oos = ds.feat_std[rows("oos")]
    y_oos = ds.y_hard_all[rows("oos")]

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
    clf.fit(X_train, y_train)

    def _eval(X: np.ndarray, y: np.ndarray) -> dict:
        proba = clf.predict_proba(X)
        pred = clf.predict(X)
        return {
            "n": int(len(y)),
            "hard_top1_acc": float((pred == y).mean()),
            "log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
        }

    val_metrics = _eval(X_val, y_val)
    oos_metrics = _eval(X_oos, y_oos)
    train_metrics = _eval(X_train, y_train)

    print("train:", train_metrics)
    print("val:", val_metrics)
    print("oos:", oos_metrics)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "model": "sklearn.HistGradientBoostingClassifier (LightGBM not installed; closest available histogram GBDT)",
                "features": "raw single-bar causalfix_final (113 cols), same rows/split/standardization as the transformer run (window=48, train_stride=4)",
                "target": "zigzag_action (hard label)",
                "train": train_metrics,
                "val": val_metrics,
                "oos": oos_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
