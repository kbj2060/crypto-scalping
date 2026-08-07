"""Fair-comparison follow-up to train_btc_tripbarrier_gbdt_baseline_20260806.py: that run used one
fixed, untuned GBDT config while the transformer went through an architecture sweep + label_sharpen
sweep + cash_weight sweep. This sweeps GBDT hyperparameters (max_depth, learning_rate,
l2_regularization) AND a cash sample_weight axis (mirroring the transformer's cash_weight=0.9 win,
which fixed its CASH over-prediction problem) to the same degree of effort, selected on VAL only,
confirmed on OOS once for the winner.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset  # noqa: E402

LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_flatsmooth_20260806.parquet"
OUT_DIR = ROOT / "tmp/btc_deepfeat_tripbarrier_20260806/gbdt_sweep"

MAX_DEPTH = [4, 6, 8]
LEARNING_RATE = [0.03, 0.05, 0.1]
L2_REG = [0.5, 2.0]
CASH_WEIGHT = [1.0, 0.9, 0.7]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = build_dataset(
        window=48, train_stride=4, label_path=LABEL_PATH, hard_col="trade_outcome_action",
        soft_cols=["trade_outcome_soft_cash", "trade_outcome_soft_long", "trade_outcome_soft_short"],
    )
    X_train, y_train = ds.feat_std[ds.end_idx["train"]], ds.y_hard_all[ds.end_idx["train"]]
    X_val, y_val = ds.feat_std[ds.end_idx["val"]], ds.y_hard_all[ds.end_idx["val"]]
    X_oos, y_oos = ds.feat_std[ds.end_idx["oos"]], ds.y_hard_all[ds.end_idx["oos"]]

    def _eval(clf, X, y):
        pred = clf.predict(X)
        proba = clf.predict_proba(X)
        return {
            "n": int(len(y)),
            "hard_top1_acc": float((pred == y).mean()),
            "log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
            "pred_cash_frac": float((pred == 0).mean()),
            "true_cash_frac": float((y == 0).mean()),
        }

    results = []
    configs = list(product(MAX_DEPTH, LEARNING_RATE, L2_REG, CASH_WEIGHT))
    print(f"sweeping {len(configs)} configs")
    for max_depth, lr, l2, cash_weight in configs:
        sample_weight = np.where(y_train == 0, cash_weight, 1.0)
        clf = HistGradientBoostingClassifier(
            max_depth=max_depth, learning_rate=lr, max_iter=300, l2_regularization=l2,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=20260806,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        val_m = _eval(clf, X_val, y_val)
        row = {"max_depth": max_depth, "learning_rate": lr, "l2_regularization": l2, "cash_weight": cash_weight, **val_m}
        results.append(row)
        print(json.dumps(row))

    best = min(results, key=lambda r: r["log_loss"])
    print("BEST (by VAL log_loss):", json.dumps(best))

    sample_weight = np.where(y_train == 0, best["cash_weight"], 1.0)
    best_clf = HistGradientBoostingClassifier(
        max_depth=best["max_depth"], learning_rate=best["learning_rate"], max_iter=300,
        l2_regularization=best["l2_regularization"], early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=15, random_state=20260806,
    )
    best_clf.fit(X_train, y_train, sample_weight=sample_weight)
    oos_m = _eval(best_clf, X_oos, y_oos)
    print("OOS CONFIRM:", json.dumps(oos_m))

    import joblib
    joblib.dump(best_clf, OUT_DIR / "best_gbdt_model.joblib")
    (OUT_DIR / "sweep_summary.json").write_text(
        json.dumps({"all_results": results, "best_val_config": best, "oos_confirm": oos_m}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
