"""Does the transformer's deep-feature encoder add anything over raw features for the NEW causal
triple-barrier label (trade_outcome_action), or does a plain tree model do just as well/better --
mirroring the same check already done for the old zigzag label
(train_btc_zigzag_gbdt_baseline_20260806.py: raw GBDT 65.5%/63.4% beat the untuned transformer
there). Uses sklearn HistGradientBoostingClassifier on raw single-bar causalfix_final features
(no windowing, no deep encoder), same rows/split/standardization as the transformer runs
(window=48, train_stride=4).
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

LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_flatsmooth_20260806.parquet"
OUT_DIR = ROOT / "tmp/btc_deepfeat_tripbarrier_20260806/gbdt_raw_baseline"


def main() -> int:
    ds = build_dataset(
        window=48, train_stride=4, label_path=LABEL_PATH, hard_col="trade_outcome_action",
        soft_cols=["trade_outcome_soft_cash", "trade_outcome_soft_long", "trade_outcome_soft_short"],
    )

    def rows(split: str) -> np.ndarray:
        return ds.end_idx[split]

    X_train, y_train = ds.feat_std[rows("train")], ds.y_hard_all[rows("train")]
    X_val, y_val = ds.feat_std[rows("val")], ds.y_hard_all[rows("val")]
    X_oos, y_oos = ds.feat_std[rows("oos")], ds.y_hard_all[rows("oos")]

    clf = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=20260806,
    )
    clf.fit(X_train, y_train)

    def _eval(X, y):
        pred = clf.predict(X)
        proba = clf.predict_proba(X)
        pred_dist = {int(k): float(v) for k, v in zip(*np.unique(pred, return_counts=True))}
        n = len(pred)
        return {
            "n": int(n),
            "hard_top1_acc": float((pred == y).mean()),
            "log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
            "pred_dist": {k: v / n for k, v in pred_dist.items()},
            "true_dist": {int(k): float(v) / n for k, v in zip(*np.unique(y, return_counts=True))},
        }

    train_m, val_m, oos_m = _eval(X_train, y_train), _eval(X_val, y_val), _eval(X_oos, y_oos)
    print("train:", json.dumps(train_m, indent=2))
    print("val:", json.dumps(val_m, indent=2))
    print("oos:", json.dumps(oos_m, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "model": "sklearn.HistGradientBoostingClassifier (raw features, no deep encoder)",
                "target": "trade_outcome_action (causal triple-barrier label)",
                "train": train_m, "val": val_m, "oos": oos_m,
                "comparison_transformer_best": {"note": "flatsmooth cash_weight=0.9: val_acc~0.40 oos_acc~0.344, oos fresh-entry win_rate=35.5%, sum_ret=-9.5%"},
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    import joblib
    joblib.dump(clf, OUT_DIR / "gbdt_model.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
