#!/usr/bin/env python3
"""RESEARCH ONLY -- Ilias #1: baseline-first training of the direction-quality-reactive exit signal.

Per the ml-adoption-playbook "baseline first" principle (also cited in the design doc's own "다음
세션 액션" #4) and .claude/CLAUDE.md's dl_needs_optimization_before_failure_verdict feedback: this is
a SINGLE baseline config per model family (LogisticRegression, HistGradientBoostingClassifier), no
hyperparameter sweep, no GPU/deep-learning. Reads the label dataset produced by
research_ilias_eth_adaptive_exit_signal_labels_20260817.py (h48qual, TRAIN split
2025-01-01..2025-09-30, counterfactual TP/SL-barrier labels, feature columns =
research_ilias_eth_adaptive_exit_signal_common_20260817.FEATURE_COLUMNS).

=== Pre-registered choices (fixed BEFORE any of the 6 evaluation windows in
research_ilias_eth_adaptive_exit_signal_arm_eval_20260817.py are touched -- avoids p-hacking the
primary-model choice or the decision threshold against the actual success/kill criteria) ===
- Decision threshold = 0.5 (common.NEW_EXIT_THRESHOLD_DEFAULT), the natural default for a
  probabilistic binary classifier -- never tuned per-window.
- Primary model = whichever of {logreg, hgb} has the higher GroupKFold(5, grouped by trade_id)
  cross-validated ROC-AUC on the TRAIN label set. GroupKFold (not a plain row-level K-fold) is used
  because rows from the same trade share one label and highly-correlated features (consecutive
  in-trade bars) -- a row-level split would leak trade identity across folds and inflate AUC.
- The FROZEN model shipped to arm-eval is refit on the FULL TRAIN label set (all 65 trades / 60,694
  rows) using the chosen family's same fixed hyperparameters -- the CV score above is a diagnostic
  generalization check only, never a per-window-tuned model.

Only 65 resolved trades feed this label set (see labels_report.json) -- a small-N caveat flagged
explicitly in the training report and repeated in the experiment doc's honest-limitations section, per
[[tabm_hp_low_signal_pattern]] and [[feedback_dl_needs_optimization_before_failure_verdict]] evidence-
strength discipline (single-config baseline result, not a swept/optimized ceiling).

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

import research_ilias_eth_adaptive_exit_signal_common_20260817 as common  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817"
LABEL_CSV = OUT_DIR / "train_labels_h48qual_2025q1q3.csv"
N_CV_FOLDS = 5
RANDOM_STATE = 20260817


def log(msg: str) -> None:
    common.log("ilias_train", msg)


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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"=== stage=load_labels ({LABEL_CSV}) ===")
    df = pd.read_csv(LABEL_CSV)
    log(f"  rows={len(df)} trades={df['trade_id'].nunique()} label_positive_rate={df['label_sl'].mean():.4f}")

    x = df[common.FEATURE_COLUMNS].to_numpy(dtype=np.float64)
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

    bundle = {
        "model": primary_model,
        "model_name": primary_name,
        "feature_columns": common.FEATURE_COLUMNS,
        "threshold": common.NEW_EXIT_THRESHOLD_DEFAULT,
        "label_csv": str(LABEL_CSV),
        "n_train_rows": int(len(df)), "n_train_trades": int(df["trade_id"].nunique()),
        "cv": {"logreg": logreg_cv, "hgb": hgb_cv},
    }
    bundle_path = OUT_DIR / "new_exit_signal_bundle.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    log(f"wrote {bundle_path} primary_model={primary_name} threshold={common.NEW_EXIT_THRESHOLD_DEFAULT}")

    # also persist the secondary (non-primary) model's bundle, for reporting/robustness only -- never
    # used by arm-eval unless explicitly loaded.
    secondary_name = "logreg" if primary_name == "hgb" else "hgb"
    secondary_bundle = dict(bundle)
    secondary_bundle["model"] = models[secondary_name]
    secondary_bundle["model_name"] = secondary_name
    with open(OUT_DIR / "new_exit_signal_bundle_secondary.pkl", "wb") as f:
        pickle.dump(secondary_bundle, f)

    report = {
        "design": __doc__,
        "n_cv_folds_requested": N_CV_FOLDS,
        "label_csv": str(LABEL_CSV),
        "n_train_rows": int(len(df)), "n_train_trades": int(df["trade_id"].nunique()),
        "label_positive_rate": float(df["label_sl"].mean()),
        "feature_columns": common.FEATURE_COLUMNS,
        "decision_threshold_preregistered": common.NEW_EXIT_THRESHOLD_DEFAULT,
        "cv_logreg": logreg_cv, "cv_hgb": hgb_cv,
        "primary_model_selected_by_cv_auc": primary_name,
        "primary_bundle_path": str(bundle_path),
        "secondary_bundle_path": str(OUT_DIR / "new_exit_signal_bundle_secondary.pkl"),
        "small_n_caveat": (
            f"Only {df['trade_id'].nunique()} resolved trades feed this label set -- a single-config "
            "baseline result on a small trade population, not a swept/optimized ceiling. GroupKFold "
            "AUC above is a generalization sanity check only; it does not gate the pre-registered "
            "success/kill criteria (those are evaluated on the 6 held-out windows in "
            "research_ilias_eth_adaptive_exit_signal_arm_eval_20260817.py)."
        ),
    }
    (OUT_DIR / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)),
        encoding="utf-8",
    )
    log(f"report={OUT_DIR / 'train_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
