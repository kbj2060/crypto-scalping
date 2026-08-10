"""Direct BTC regime3-current comparison: live 12-state sticky HMM vs the JM(k=3, lambda=4) swap
built in build_btc_regime3_jm_lam4_20260809.py, scored against the SAME ADX/slope/BB
"balancedish_adx16_slope15_bb012" rule label on the SAME 2024/2025/2026 BTC feature files, so the
only difference between the two rows is the classifier family (HMM vs JM) -- feature panel, label
target, and eval protocol are held fixed. Mirrors the comparison the ETH jmlam4 build implicitly
enables (its own report only has the JM side; this script produces both sides for BTC so the
head-to-head is explicit rather than requiring a second lookup).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_regime3_current_hmm_wide24_20260529 import (  # noqa: E402
    _class_proba, _labels, _transform, _with_features,
)
from scripts.train_regime3_hmm_mamba_20260529 import _read  # noqa: E402
from scripts.build_btc_regime3_jm_lam4_20260809 import causal_decode_soft  # noqa: E402

HMM_JOBLIB = ROOT / "data/ensemble/supervised/btc_regime3_current_hmm_sensitive_wide24_20260708/regime3_current_sensitive_hmm_wide24_2024.joblib"
JM_JOBLIB = ROOT / "data/ensemble/supervised/btc_regime3_current_jm_jmlam4_20260809_2024.joblib"
SOURCES = {
    "2024": ROOT / "data/splits/year_oos/btc_features_2024.csv",
    "2025": ROOT / "data/splits/year_oos/btc_features_2025.csv",
    "2026": ROOT / "data/splits/year_oos/btc_features_2026.csv",
}
REPORT_PATH = ROOT / "data/ensemble/reports/btc_regime3_hmm_vs_jm_lam4_20260809_report.json"


def eval_payload(payload: dict, frame: pd.DataFrame) -> dict:
    classes = payload["classes"]
    cols = payload["feature_cols"]
    work = _with_features(frame, cols)
    y = _labels(work, payload["label_mode"])

    if "model" in payload:
        sidecar, _ = _transform(payload, frame)
        prefix = f"{payload['prefix_stem']}_{payload['feature_set']}_"
        proba = sidecar[[f"{prefix}{c}_prob" for c in classes]].to_numpy()
    else:
        med = pd.Series(payload["feature_medians"])
        x_raw = work[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
        x_obs = payload["scaler"].transform(x_raw)
        _, state_prob = causal_decode_soft(x_obs, payload["jm_mu"], payload["jm_lambda"], payload["jm_temperature"])
        proba = _class_proba(state_prob, payload["state_class_matrix"])
    pred = np.argmax(proba, axis=1)

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    cm = confusion_matrix(y, pred, labels=list(range(len(classes))))
    recalls = {classes[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(len(classes))}
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "flip_rate": float(np.mean(pred[1:] != pred[:-1])) if len(pred) > 1 else 0.0,
        "recall_by_class": recalls,
    }


def main() -> None:
    hmm_payload = joblib.load(HMM_JOBLIB)
    jm_payload = joblib.load(JM_JOBLIB)

    report = {"hmm_model_id": hmm_payload["model_id"], "jm_model_id": jm_payload["model_id"], "by_year": {}}
    print(f"{'year':<6}{'model':<6}{'acc':>8}{'bal_acc':>10}{'flip':>8}   bull/bear/chop recall")
    for year, src in SOURCES.items():
        frame = _read(src)
        hmm_ev = eval_payload(hmm_payload, frame)
        jm_ev = eval_payload(jm_payload, frame)
        report["by_year"][year] = {"hmm": hmm_ev, "jm_lam4": jm_ev}
        for tag, ev in (("HMM", hmm_ev), ("JM", jm_ev)):
            r = ev["recall_by_class"]
            print(
                f"{year:<6}{tag:<6}{ev['accuracy']:>8.4f}{ev['balanced_accuracy']:>10.4f}{ev['flip_rate']:>8.4f}"
                f"   {r.get('bull', float('nan')):.3f}/{r.get('bear', float('nan')):.3f}/{r.get('chop', float('nan')):.3f}"
            )
        d_acc = jm_ev["accuracy"] - hmm_ev["accuracy"]
        d_bacc = jm_ev["balanced_accuracy"] - hmm_ev["balanced_accuracy"]
        print(f"{year:<6}{'Δ(JM-HMM)':<6}{d_acc:>+8.4f}{d_bacc:>+10.4f}")

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
