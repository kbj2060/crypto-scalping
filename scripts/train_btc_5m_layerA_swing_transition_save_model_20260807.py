"""Retrain and SAVE the BTC 5m Layer A transition detector (swing_transition_prob source model).

The 2026-08-06 session's eval_btc_5m_layerA_layerB_20260806.py trained this LightGBM in-memory and
saved only its predictions (tmp/btc_1h_volregime_20260805/btc5m_layerA_pred.parquet). Those
predictions became the `swing_transition_prob` feature of the promoted h48qual swingtransition
candidate -- but live wiring needs the MODEL, not the prediction file.

This script re-runs the exact same recipe (same panel, same DVOL features, same label, same
params, same train split) and refuses to save unless the regenerated full-panel probA reproduces
the saved parquet (fail-fast: a mismatch means the saved feature is not reproducible and the live
feature would drift from what the candidate was validated on).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_btc_5m_layerA_layerB_20260806 import (  # noqa: E402
    DROP_RAW,
    OOS_END,
    OOS_START,
    PANEL_PATH,
    PIVOT_PATH,
    VAL_START,
    build_dvol_features,
)

SAVED_PRED_PATH = ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerA_pred.parquet"
OUT_DIR = ROOT / "data/ensemble/supervised/btc_swing_transition_layerA_20260807"
REPRO_ATOL = 1e-9


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    panel = pd.read_parquet(PANEL_PATH)
    dvol = build_dvol_features()
    panel = pd.merge_asof(panel.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    feature_cols = [c for c in panel.columns if c not in DROP_RAW]

    piv = pd.read_parquet(PIVOT_PATH, columns=["timestamp", "transition_soon"])
    dfA = panel.merge(piv, on="timestamp", how="inner").dropna(subset=["transition_soon"]).reset_index(drop=True)
    Xa = dfA[feature_cols]
    ya = dfA["transition_soon"].astype(int)
    tr = dfA["timestamp"] < VAL_START
    val = (dfA["timestamp"] >= VAL_START) & (dfA["timestamp"] < OOS_START)
    oos = (dfA["timestamp"] >= OOS_START) & (dfA["timestamp"] < OOS_END)
    print(f"train={tr.sum()} val={val.sum()} oos={oos.sum()} base_rate(train)={ya[tr].mean():.4f}")

    clfA = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=100,
                          class_weight="balanced", verbosity=-1)
    clfA.fit(Xa[tr], ya[tr])
    probA = clfA.predict_proba(Xa)[:, 1]
    dfA["probA"] = probA

    metrics: dict[str, dict[str, float]] = {}
    for name, mask in [("VAL", val), ("OOS", oos)]:
        yt, p = ya[mask], dfA.loc[mask, "probA"]
        metrics[name] = {
            "auc": float(roc_auc_score(yt, p)),
            "ap": float(average_precision_score(yt, p)),
            "base_rate": float(yt.mean()),
            "n": int(mask.sum()),
        }
        print(f"{name}: AUC={metrics[name]['auc']:.4f} AP={metrics[name]['ap']:.4f} base={metrics[name]['base_rate']:.4f}")

    # Reproducibility gate: the regenerated predictions must match the parquet that was used to
    # build the promoted candidate's swing_transition_prob training feature.
    saved = pd.read_parquet(SAVED_PRED_PATH)
    merged = dfA[["timestamp", "probA"]].merge(saved.rename(columns={"probA": "probA_saved"}), on="timestamp", how="inner")
    if len(merged) != len(dfA) or len(merged) != len(saved):
        raise SystemExit(f"REPRO FAIL: row mismatch regenerated={len(dfA)} saved={len(saved)} joined={len(merged)}")
    diff = np.abs(merged["probA"].to_numpy() - merged["probA_saved"].to_numpy())
    corr = float(np.corrcoef(merged["probA"], merged["probA_saved"])[0, 1])
    print(f"repro check: rows={len(merged)} max_abs_diff={diff.max():.3e} corr={corr:.10f}")
    if diff.max() > REPRO_ATOL:
        raise SystemExit(f"REPRO FAIL: max_abs_diff={diff.max():.3e} > {REPRO_ATOL} (corr={corr:.10f}) -- "
                         "saved swing_transition_prob feature is NOT reproducible; do not wire live.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "layerA_lgbm.pkl").open("wb") as f:
        pickle.dump({"model": clfA, "feature_columns": list(feature_cols)}, f)
    report = {
        "model_id": "btc_swing_transition_layerA_20260807",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recipe": "exact re-run of eval_btc_5m_layerA_layerB_20260806.py Layer A",
        "label": "transition_soon (pivot imminent within 24 bars)",
        "train_split_end": VAL_START,
        "params": {"n_estimators": 400, "num_leaves": 31, "learning_rate": 0.05,
                   "min_child_samples": 100, "class_weight": "balanced"},
        "n_features": len(feature_cols),
        "metrics": metrics,
        "repro_check": {"rows": int(len(merged)), "max_abs_diff": float(diff.max()), "corr": corr,
                        "atol": REPRO_ATOL, "reference": str(SAVED_PRED_PATH)},
        "inputs": {
            "panel": {"path": str(PANEL_PATH), "sha256": sha256_file(PANEL_PATH)},
            "pivot_labels": {"path": str(PIVOT_PATH), "sha256": sha256_file(PIVOT_PATH)},
            "dvol_csv": {"path": str(ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"),
                          "sha256": sha256_file(ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv")},
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved model + report to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
