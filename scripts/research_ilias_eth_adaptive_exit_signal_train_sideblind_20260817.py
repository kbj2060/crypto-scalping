#!/usr/bin/env python3
"""RESEARCH ONLY -- Ilias correction session (2026-08-17): side-blind retraining of the
direction-quality-reactive exit signal, per the quasi-separation finding documented in
research_ilias_eth_adaptive_exit_signal_common_sideblind_20260817.py's module docstring.

Documented copy of research_ilias_eth_adaptive_exit_signal_train_20260817.py (that original script is
left UNTOUCHED -- its outputs, `new_exit_signal_bundle.pkl` etc., are preserved as the research record
of the contaminated run). The ONLY change: feature_columns =
research_ilias_eth_adaptive_exit_signal_common_sideblind_20260817.FEATURE_COLUMNS (10 columns --
pos_side/pos_leverage/pos_notional/pos_exposure removed) instead of the original 14-column
common.FEATURE_COLUMNS. Reuses the SAME label CSV
(tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/
train_labels_h48qual_2025q1q3.csv, 65 trades / 60,694 rows) unmodified -- the label (`label_sl`, does
this trade eventually hit stop_loss vs take_profit) is defined independently of side (the counterfactual
barrier simulation resolves SL/TP from the REAL h48qual direction, never overridden), so no
relabeling/regeneration is needed for this correction.

All pre-registered choices from the original script are kept IDENTICAL (decision threshold=0.5,
model-family selection by higher GroupKFold(5, grouped by trade_id) CV ROC-AUC, frozen model refit on
the full TRAIN label set) -- only the feature list changed, and that change was decided BEFORE looking
at any of the 6 evaluation windows (the whole point of this correction is to re-verify criterion 1
without direction/sizing leakage, using the exact same evaluation procedure).

=== New addition vs the original script (asked for explicitly this session) ===
After refitting, this script extracts and logs/reports the LogisticRegression's standardized
coefficients (StandardScaler + LogisticRegression pipeline) by feature name, sorted by |coef| -- the
same inspection that revealed the original 14-feature model's quasi-separation
(pos_side/pos_leverage/pos_exposure/pos_notional all |coef|>21) is re-run here on the 10 remaining
features so a recurrence (e.g. pos_tp/pos_sl happening to correlate with TRAIN-period direction) would
not go unnoticed. Reported honestly regardless of outcome.

Does NOT touch trading_bot.py / trading_bot_modules/* / runtime_config.py / .env. No GPU. conda env
quant_ai.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_ilias_eth_adaptive_exit_signal_common_sideblind_20260817 as common_sb  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817"
LABEL_CSV = OUT_DIR / "train_labels_h48qual_2025q1q3.csv"  # unchanged, reused from the original session
FEATURE_COLUMNS = common_sb.FEATURE_COLUMNS  # the only substantive change vs the original train script
N_CV_FOLDS = 5
RANDOM_STATE = 20260817


def log(msg: str) -> None:
    common_sb.log(msg)


def build_logreg() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
    ])


def build_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.05, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=RANDOM_STATE,
    )


def cv_score(build_fn, x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int) -> dict:
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=splits)
    aucs, aps, lls = [], [], []
    for train_idx, test_idx in gkf.split(x, y, groups):
        model = build_fn()
        model.fit(x[train_idx], y[train_idx])
        proba = model.predict_proba(x[test_idx])[:, 1]
        y_test = y[test_idx]
        if len(np.unique(y_test)) < 2:
            continue
        aucs.append(roc_auc_score(y_test, proba))
        aps.append(average_precision_score(y_test, proba))
        lls.append(log_loss(y_test, proba, labels=[0, 1]))
    return {
        "n_folds_used": len(aucs), "n_folds_requested": splits,
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "auc_per_fold": [float(a) for a in aucs],
        "avg_precision_mean": float(np.mean(aps)) if aps else float("nan"),
        "log_loss_mean": float(np.mean(lls)) if lls else float("nan"),
    }


def logreg_coefficients(pipeline: Pipeline, feature_columns: list[str]) -> list[dict]:
    """Standardized-scale coefficients (StandardScaler + LogisticRegression), sorted by |coef| desc --
    the same inspection that revealed the original model's quasi-separation on
    pos_side/pos_leverage/pos_notional/pos_exposure. Re-run here so a recurrence on the remaining 10
    features would not go unnoticed."""
    clf = pipeline.named_steps["clf"]
    coefs = clf.coef_[0]
    rows = [{"feature": f, "coef_standardized": float(c)} for f, c in zip(feature_columns, coefs)]
    rows.sort(key=lambda r: abs(r["coef_standardized"]), reverse=True)
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"=== stage=load_labels ({LABEL_CSV}) ===")
    df = pd.read_csv(LABEL_CSV)
    log(f"  rows={len(df)} trades={df['trade_id'].nunique()} label_positive_rate={df['label_sl'].mean():.4f}")
    log(f"  FEATURE_COLUMNS (side-blind, n={len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    log(f"  excluded_vs_original: {common_sb.EXCLUDED_DIRECTION_SIZING_COLUMNS}")

    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df["label_sl"].to_numpy(dtype=np.int64)
    groups = df["trade_id"].to_numpy()

    log("=== stage=group_cv (GroupKFold by trade_id, diagnostic-only, decided before any window is touched) ===")
    logreg_cv = cv_score(build_logreg, x, y, groups, N_CV_FOLDS)
    hgb_cv = cv_score(build_hgb, x, y, groups, N_CV_FOLDS)
    log(f"  logreg: {logreg_cv}")
    log(f"  hgb:    {hgb_cv}")

    primary_name = "hgb" if hgb_cv["auc_mean"] >= logreg_cv["auc_mean"] else "logreg"
    log(f"  primary_model_selected_by_cv_auc={primary_name} (logreg_auc={logreg_cv['auc_mean']:.4f} hgb_auc={hgb_cv['auc_mean']:.4f})")

    log("=== stage=refit_on_full_train_label_set (frozen model for arm-eval) ===")
    logreg_full = build_logreg().fit(x, y)
    hgb_full = build_hgb().fit(x, y)
    models = {"logreg": logreg_full, "hgb": hgb_full}
    primary_model = models[primary_name]

    log("=== stage=coefficient_inspection (re-run the quasi-separation check on the 10 remaining features) ===")
    logreg_coefs = logreg_coefficients(logreg_full, FEATURE_COLUMNS)
    for row in logreg_coefs:
        log(f"    {row['feature']:<28s} {row['coef_standardized']:+.4f}")
    max_abs_coef = max(abs(r["coef_standardized"]) for r in logreg_coefs)
    log(f"  max_abs_standardized_coef={max_abs_coef:.4f} (original contaminated model's top-4 leaking "
        f"features were all in the 21-27 range; anything approaching that here would flag a possible "
        f"recurrence -- see report for full sorted list)")

    bundle = {
        "model": primary_model,
        "model_name": primary_name,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": common_sb.NEW_EXIT_THRESHOLD_DEFAULT,
        "label_csv": str(LABEL_CSV),
        "n_train_rows": int(len(df)), "n_train_trades": int(df["trade_id"].nunique()),
        "cv": {"logreg": logreg_cv, "hgb": hgb_cv},
        "excluded_direction_sizing_columns": common_sb.EXCLUDED_DIRECTION_SIZING_COLUMNS,
    }
    bundle_path = OUT_DIR / "new_exit_signal_bundle_sideblind.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    log(f"wrote {bundle_path} primary_model={primary_name} threshold={common_sb.NEW_EXIT_THRESHOLD_DEFAULT}")

    # also persist the secondary (non-primary) model's bundle, for reporting/robustness only -- never
    # used by arm-eval unless explicitly loaded.
    secondary_name = "logreg" if primary_name == "hgb" else "hgb"
    secondary_bundle = dict(bundle)
    secondary_bundle["model"] = models[secondary_name]
    secondary_bundle["model_name"] = secondary_name
    with open(OUT_DIR / "new_exit_signal_bundle_sideblind_secondary.pkl", "wb") as f:
        pickle.dump(secondary_bundle, f)

    report = {
        "design": __doc__,
        "correction_of": "research_ilias_eth_adaptive_exit_signal_train_20260817.py (original, untouched, preserved)",
        "excluded_direction_sizing_columns": common_sb.EXCLUDED_DIRECTION_SIZING_COLUMNS,
        "n_cv_folds_requested": N_CV_FOLDS,
        "label_csv": str(LABEL_CSV),
        "n_train_rows": int(len(df)), "n_train_trades": int(df["trade_id"].nunique()),
        "label_positive_rate": float(df["label_sl"].mean()),
        "feature_columns": FEATURE_COLUMNS,
        "decision_threshold_preregistered": common_sb.NEW_EXIT_THRESHOLD_DEFAULT,
        "cv_logreg": logreg_cv, "cv_hgb": hgb_cv,
        "primary_model_selected_by_cv_auc": primary_name,
        "primary_bundle_path": str(bundle_path),
        "secondary_bundle_path": str(OUT_DIR / "new_exit_signal_bundle_sideblind_secondary.pkl"),
        "logreg_standardized_coefficients_sorted_by_abs": logreg_coefs,
        "max_abs_standardized_coef": max_abs_coef,
        "small_n_caveat": (
            f"Only {df['trade_id'].nunique()} resolved trades feed this label set -- a single-config "
            "baseline result on a small trade population, not a swept/optimized ceiling. GroupKFold "
            "AUC above is a generalization sanity check only; it does not gate the pre-registered "
            "success/kill criteria (those are evaluated on the 6 held-out windows in "
            "research_ilias_eth_adaptive_exit_signal_arm_eval_sideblind_20260817.py)."
        ),
    }
    (OUT_DIR / "train_report_sideblind.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)),
        encoding="utf-8",
    )
    log(f"report={OUT_DIR / 'train_report_sideblind.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
