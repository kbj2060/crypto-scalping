#!/usr/bin/env python3
"""GBM3 (deployed 3-class regime classifier) vs TabPFN, apples-to-apples -- user request 2026-09-02
("이것도 gbm 말고 pfn 모델을 사용해서 다시 평가해줘").

WHAT IS HELD FIXED

  Everything except the learner. Same 136 feature_cols (read from the deployed GBM3 joblib payload,
  which already EXCLUDES the 5 columns confirmed in the 2026-08-26 audit to be literal/monotonic
  proxies of RegimeEngine's own label formula -- verified present in `notes`), same label
  construction (RegimeEngine 3-class: bull / bear / chop, with whipsaw+normal merged into chop --
  the merge GBM3 itself uses), same TRAIN (2024-01-01~2026-06-30) / internal causal VAL
  (2026-04-01~06-30) / OOS (2026-07-01~08-19) split, same data loading path
  (load_data -> _with_raw_state12 -> RegimeEngine) reused verbatim from
  train_eth_regime_gbm2_trend_chop_20260827.py.

THE CONTROL THAT MAKES THIS A FAIR TEST

  TabPFN v2 is designed for <=10k training rows; full TRAIN here is ~262k 5m bars. So TabPFN must be
  fit on a subsample. Comparing subsampled-TabPFN against full-data-GBM would confound "TabPFN vs
  GBM" with "10k rows vs 262k rows". Three arms therefore:

    A. gbm_full      -- GBM3's exact config on the FULL TRAIN. Reproduces the deployed 0.9189.
    B. gbm_matched   -- the SAME GBM config on the SAME subsample TabPFN sees (matched-N control).
    C. tabpfn        -- TabPFN on that subsample.

  B is the arm C must beat for "TabPFN is the better learner" to be true. A is the bar C must beat
  for "TabPFN should replace the deployed model" to be true. They are different claims.

SEEDS

  N_SEEDS genuinely random subsample seeds (drawn from a master RNG, NOT fixed increments) --
  CLAUDE.md's Seed-Diversity gate exists precisely because a +5-increment "5-seed ensemble" once
  agreed with a real one on VAL and flipped sign on OOS. Per-seed results are reported individually,
  plus a soft-vote ensemble for both B and C.

⚠️ OOS PURITY

  2026-07-01~08-19 has already been consumed by ~8+ prior rounds of regime-classifier research in
  this repo (wide24 grid sweeps, the whipsaw 6 rounds, GBM3 itself, GBM2) -- see
  train_eth_regime_gbm2_trend_chop_20260827.py's docstring, which makes the same disclosure. This
  run is a RESEARCH/DEV model-family comparison, not promotion evidence, and it adds one more touch.
  CLAUDE.md's Fresh-Forward rule is not satisfied by this window.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from features.elite import RegimeEngine  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402
from train_eth_regime_gbm2_trend_chop_20260827 import (  # noqa: E402
    OOS_END,
    OOS_START,
    TRAIN_CSVS,
    TRAIN_END,
    TRAIN_START,
    VAL_START,
    _run_lengths,
)

CLASSES3 = ["bull", "bear", "chop"]          # label ints 0/1/2, matching the deployed artifact
GBM3_MODEL_PATH = ROOT / "tmp/eth_regime_gbm3_independent_20260826/model.joblib"
GBM3_HP = dict(max_depth=10, learning_rate=0.04, max_iter=400, l2_regularization=2.0)
MASTER_SEED = 20260902
OUT_DIR = ROOT / "tmp/eth_regime_gbm3_vs_tabpfn_20260902"


def load_frame() -> pd.DataFrame:
    frames = [pd.read_csv(p, parse_dates=["timestamp"]) for p in TRAIN_CSVS]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= OOS_END)].reset_index(drop=True)
    return _with_raw_state12(df)


def build_labels3(df: pd.DataFrame) -> np.ndarray:
    """RegimeEngine 3-class over the FULL continuous series (its rolling windows need an unbroken
    sequence), sliced into splits by the caller -- never recomputed per split."""
    labeled = RegimeEngine().compute(df.copy())
    y = np.full(len(df), 2, dtype=int)                      # default chop (= chop|whipsaw|normal)
    y[labeled["regime_bull"].to_numpy() > 0] = 0
    y[labeled["regime_bear"].to_numpy() > 0] = 1
    return y


def evaluate(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    runs = _run_lengths(pred)
    return {
        "rows": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "recall": {n: (None if cm[i].sum() == 0 else float(cm[i, i] / cm[i].sum()))
                   for i, n in enumerate(CLASSES3)},
        "flip_rate": float(np.mean(pred[1:] != pred[:-1])) if len(pred) > 1 else 0.0,
        "class_share": {n: float((pred == i).mean()) for i, n in enumerate(CLASSES3)},
        "mean_state_duration_bars": float(np.mean(runs)) if runs else 0.0,
        "median_state_duration_bars": float(np.median(runs)) if runs else 0.0,
    }


def stratified_subsample(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Preserve the class base rates of TRAIN in the subsample TabPFN is fit on."""
    idx = []
    for cls in (0, 1, 2):
        pool = np.flatnonzero(y == cls)
        take = min(len(pool), max(1, int(round(n * len(pool) / len(y)))))
        idx.append(rng.choice(pool, size=take, replace=False))
    return np.sort(np.concatenate(idx))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sub", type=int, default=10000, help="TabPFN training subsample size")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--oos-stride", type=int, default=1, help=">1 subsamples OOS rows (smoke tests)")
    ap.add_argument("--skip-full-gbm", action="store_true")
    args = ap.parse_args()

    payload = joblib.load(GBM3_MODEL_PATH)
    feat_cols, medians = payload["feature_cols"], payload["feature_medians"]
    print(f"GBM3 artifact: {len(feat_cols)} features, deployed OOS bal_acc="
          f"{payload['oos_validated_bal_acc']} on {payload['oos_validated_range']}")

    print("Loading data + building 3-class RegimeEngine labels...")
    df = load_frame()
    y_all = build_labels3(df)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing feature columns: {missing}")
    x_all = df[feat_cols].astype(float)
    for c in feat_cols:                                     # same NaN policy as the live scorer
        x_all[c] = x_all[c].fillna(medians.get(c, 0.0))

    ts = df["timestamp"]
    tr_m = ((ts >= TRAIN_START) & (ts <= TRAIN_END)).to_numpy()
    oos_m = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()
    if args.oos_stride > 1:
        keep = np.zeros(len(df), dtype=bool)
        keep[np.flatnonzero(oos_m)[:: args.oos_stride]] = True
        oos_m = keep
    x_tr, y_tr = x_all[tr_m].to_numpy(), y_all[tr_m]
    x_oos, y_oos = x_all[oos_m].to_numpy(), y_all[oos_m]
    shares = {CLASSES3[i]: round(float((y_tr == i).mean()), 4) for i in range(3)}
    print(f"TRAIN {tr_m.sum():,} rows {shares} | OOS {oos_m.sum():,} rows "
          f"(VAL boundary {VAL_START.date()}, stride={args.oos_stride})")

    results: dict[str, Any] = {}

    if not args.skip_full_gbm:
        t0 = time.time()
        gbm_full = HistGradientBoostingClassifier(random_state=7529, **GBM3_HP).fit(x_tr, y_tr)
        results["A_gbm_full"] = evaluate(y_oos, gbm_full.predict(x_oos))
        results["A_gbm_full"]["fit_seconds"] = round(time.time() - t0, 1)
        print(f"  A gbm_full        bal_acc={results['A_gbm_full']['balanced_accuracy']:.4f} "
              f"flip={results['A_gbm_full']['flip_rate']:.4f} ({results['A_gbm_full']['fit_seconds']}s)")

    from tabpfn import TabPFNClassifier  # noqa: E402  (imported late: heavy)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TabPFN device: {device}"
          + ("" if device == "cuda" else "  (CPU -- Prior Labs' guidance is <~1k rows; expect slow)"))

    master = np.random.default_rng(MASTER_SEED)
    seeds = master.integers(0, 2**31 - 1, size=args.n_seeds).tolist()   # genuinely random, not +k
    print(f"Seeds (randomly drawn, per CLAUDE.md seed-diversity gate): {seeds}")

    proba_tab, proba_gbm = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        sub = stratified_subsample(y_tr, args.n_sub, rng)
        xs, ys = x_tr[sub], y_tr[sub]

        t0 = time.time()
        gm = HistGradientBoostingClassifier(random_state=int(s % (2**31)), **GBM3_HP).fit(xs, ys)
        pg = gm.predict_proba(x_oos)
        proba_gbm.append(pg)
        rg = evaluate(y_oos, pg.argmax(1))
        results[f"B_gbm_matched_seed{s}"] = rg | {"fit_seconds": round(time.time() - t0, 1)}

        t0 = time.time()
        tm = TabPFNClassifier(device=device, random_state=int(s % (2**31)),
                              ignore_pretraining_limits=True)
        tm.fit(xs, ys)
        pt = tm.predict_proba(x_oos)
        proba_tab.append(pt)
        rt = evaluate(y_oos, pt.argmax(1))
        results[f"C_tabpfn_seed{s}"] = rt | {"fit_seconds": round(time.time() - t0, 1)}
        print(f"  seed {s}: B gbm_matched bal_acc={rg['balanced_accuracy']:.4f} "
              f"flip={rg['flip_rate']:.4f} | C tabpfn bal_acc={rt['balanced_accuracy']:.4f} "
              f"flip={rt['flip_rate']:.4f} ({results[f'C_tabpfn_seed{s}']['fit_seconds']}s)")

    results["B_gbm_matched_ENSEMBLE"] = evaluate(y_oos, np.mean(proba_gbm, axis=0).argmax(1))
    results["C_tabpfn_ENSEMBLE"] = evaluate(y_oos, np.mean(proba_tab, axis=0).argmax(1))
    print(f"  B gbm_matched ENSEMBLE bal_acc={results['B_gbm_matched_ENSEMBLE']['balanced_accuracy']:.4f}")
    print(f"  C tabpfn      ENSEMBLE bal_acc={results['C_tabpfn_ENSEMBLE']['balanced_accuracy']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {"n_sub": args.n_sub, "n_seeds": args.n_seeds, "seeds": seeds,
                   "oos_stride": args.oos_stride, "gbm_hp": GBM3_HP,
                   "train_range": f"{TRAIN_START}~{TRAIN_END}", "oos_range": f"{OOS_START}~{OOS_END}",
                   "n_features": len(feat_cols), "train_class_shares": shares},
        "oos_purity_disclosure": ("2026-07-01~08-19 already consumed by ~8+ prior regime-classifier "
                                  "rounds; research/dev comparison only, NOT promotion evidence, "
                                  "and this run adds one more touch. Fresh-Forward not satisfied."),
        "deployed_reference": {"bal_acc": payload["oos_validated_bal_acc"],
                               "range": payload["oos_validated_range"]},
        "results": results,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
