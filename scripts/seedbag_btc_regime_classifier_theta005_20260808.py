"""Seed-bagging round for the theta=0.5% regime classifier (2026-08-08) -- final accuracy attempt.

Incumbent: ens_w65_lam0.5 (single-seed LGBM nowcaster + train-calibrated multi-threshold DC vote,
logit-blended at w=0.65, jump-penalized causal decode at lambda=0.5) VAL 67.2 / OOS 65.3.

This round replaces the single nowcaster with a bag of N=8 independently seeded LGBMs, averaging
their probabilities before the blend and the decode.  Complies with the project's Seed-Diversity
Ensemble Promotion Gate (.claude/CLAUDE.md): N>=5 genuinely diverse seeds drawn at RANDOM (not a
fixed-increment ladder), the seed list recorded here and in the report, and per-seed results
published alongside the bagged number so a bagging claim cannot hide seed-variance noise.

PRE-REGISTERED (before the first run):
  PRIMARY TEST is a single comparison, not a sweep: the bagged model at the INCUMBENT's exact
  (w=0.65, lambda=0.5) versus the incumbent.  ADOPT only if VAL > 67.2 AND OOS >= 65.3.
  A small neighbourhood {w 0.5,0.65,0.8} x {lambda 0.5,1.0} is also scored, but purely as context
  -- it cannot produce an adoption, because re-selecting (w, lambda) on VAL after already tuning
  them in the previous round would be double-dipping the same validation window.
  Eligibility (coverage >= 50%, median run >= 8 bars on VAL) still applies.
  Seed-diversity reporting: per-seed VAL/OOS agreement for every one of the 8 seeds, plus the
  spread; a bagged win whose per-seed spread swamps the gain is reported as noise, not a result.
  If this fails, the theta=0.5% classifier is FROZEN at ens_w65_lam0.5 (user's instruction).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from chart_btc_jm_regime_verification_20260808 import causal_zigzag  # noqa: E402
from ensemble_btc_regime_classifier_theta005_20260808 import logit, sigmoid  # noqa: E402
from refine_btc_regime_classifier_theta005_20260808 import (  # noqa: E402
    PANEL_PATH, PURGE, SCORE_SCALES, VOTE_THETAS, MIN_COVERAGE, MIN_MEDIAN_RUN,
    jump_decode_proba, summarize, to_named,
)
from test_statistical_jump_model_regimes_20260808 import zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, SEED, TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

OUT_PARQUET = ROOT / "data/research/btc_regime_theta005_seedbag_20260808.parquet"
OUT_DIR = ROOT / "tmp/regime_theta005_20260808"
N_SEEDS = 8
W_INCUMBENT, LAM_INCUMBENT = 0.65, 0.5
INCUMBENT_VAL, INCUMBENT_OOS = 67.2, 65.3
CONTEXT_W = [0.5, 0.65, 0.8]
CONTEXT_LAM = [0.5, 1.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seeds = sorted(int(s) for s in np.random.default_rng(SEED).choice(1_000_000, size=N_SEEDS, replace=False))
    print(json.dumps({"seeds": seeds}), flush=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    train_mask = (ts <= TRAIN_END).to_numpy()
    oracles = {t: zigzag_oracle(close, threshold=t)[0] for t in SCORE_SCALES}
    y_dir = oracles[0.005]

    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    czz_mat = np.column_stack([causal_zigzag(close, threshold=t) for t in VOTE_THETAS]).astype(np.float32)
    x_aug = np.column_stack([x, czz_mat])

    tr_all = np.flatnonzero(train_mask)
    tr_idx = tr_all[:-PURGE]
    tr_idx = tr_idx[(y_dir[tr_idx] != 0) & np.isfinite(x_aug[tr_idx]).any(axis=1)]
    y = (y_dir[tr_idx] == 1).astype(int)

    vote_sum = czz_mat.sum(axis=1).astype(int)
    prior = float(y.mean())
    tab = {}
    for v in range(-len(VOTE_THETAS), len(VOTE_THETAS) + 1):
        sel = vote_sum[tr_idx] == v
        tab[v] = float(y[sel].mean()) if sel.sum() >= 200 else prior
    p_vote = np.clip(np.vectorize(tab.get)(vote_sum).astype(np.float64), 0.02, 0.98)

    p_seeds = []
    for s in seeds:
        clf = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05,
                                 num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                                 bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                                 random_state=s, n_jobs=-1, verbosity=-1)
        clf.fit(x_aug[tr_idx], y)
        p_seeds.append(clf.predict_proba(x_aug)[:, 1])
        print(json.dumps({"trained_seed": s}), flush=True)
    p_bag = np.mean(p_seeds, axis=0)

    # the incumbent's own nowcaster (project SEED) for an exact regression check
    clf0 = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05,
                              num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                              bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                              random_state=SEED, n_jobs=-1, verbosity=-1)
    clf0.fit(x_aug[tr_idx], y)
    p_single = clf0.predict_proba(x_aug)[:, 1]

    windows = {
        "val_2025Q4": np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()),
        "oos_2026Q1": np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()),
        "full": np.arange(len(close)),
    }

    def blended_state(p, w, lam):
        return to_named(jump_decode_proba(sigmoid(w * logit(p) + (1.0 - w) * logit(p_vote)), lam))

    states = {"incumbent_single": blended_state(p_single, W_INCUMBENT, LAM_INCUMBENT),
              "seedbag_primary": blended_state(p_bag, W_INCUMBENT, LAM_INCUMBENT)}
    per_seed = {}
    for s, p in zip(seeds, p_seeds):
        st = blended_state(p, W_INCUMBENT, LAM_INCUMBENT)
        per_seed[str(s)] = {"val": summarize(st, oracles, windows["val_2025Q4"])["agree"]["0.005"],
                            "oos": summarize(st, oracles, windows["oos_2026Q1"])["agree"]["0.005"]}
        states[f"seed_{s}"] = st
    for w in CONTEXT_W:
        for lam in CONTEXT_LAM:
            states[f"ctx_bag_w{int(w * 100)}_lam{lam:g}"] = blended_state(p_bag, w, lam)

    report = {wt: {k: summarize(v, oracles, idx) for k, v in states.items()} for wt, idx in windows.items()}
    inc = report["val_2025Q4"]["incumbent_single"]["agree"]["0.005"]
    regression_ok = bool(abs(inc - INCUMBENT_VAL) <= 0.2)

    prim_val = report["val_2025Q4"]["seedbag_primary"]["agree"]["0.005"]
    prim_oos = report["oos_2026Q1"]["seedbag_primary"]["agree"]["0.005"]
    prim_cov = report["val_2025Q4"]["seedbag_primary"]["coverage_pct"]
    prim_run = report["val_2025Q4"]["seedbag_primary"]["median_run_bars"]
    eligible = bool(prim_cov >= MIN_COVERAGE and prim_run >= MIN_MEDIAN_RUN)
    adopt = bool(eligible and prim_val > INCUMBENT_VAL and prim_oos >= INCUMBENT_OOS)

    vals = [v["val"] for v in per_seed.values()]
    ooss = [v["oos"] for v in per_seed.values()]
    spread = {"val_min": min(vals), "val_max": max(vals), "val_std": round(float(np.std(vals)), 2),
              "oos_min": min(ooss), "oos_max": max(ooss), "oos_std": round(float(np.std(ooss)), 2),
              "n_seeds_oos_above_incumbent": int(sum(o >= INCUMBENT_OOS for o in ooss))}

    out = {"seeds": seeds, "n_seeds": N_SEEDS,
           "primary_test": {"w": W_INCUMBENT, "lambda": LAM_INCUMBENT,
                            "incumbent": {"val": INCUMBENT_VAL, "oos": INCUMBENT_OOS,
                                          "reproduced_val": inc, "regression_ok": regression_ok},
                            "seedbag": {"val": prim_val, "oos": prim_oos, "coverage_pct": prim_cov,
                                        "median_run_bars": prim_run, "eligible": eligible},
                            "adopt_rule": f"val > {INCUMBENT_VAL} AND oos >= {INCUMBENT_OOS}",
                            "adopt": adopt},
           "per_seed": per_seed, "seed_spread": spread,
           "context_only_no_adoption": {k: {"val": report["val_2025Q4"][k]["agree"]["0.005"],
                                            "oos": report["oos_2026Q1"][k]["agree"]["0.005"],
                                            "median_run_bars": report["val_2025Q4"][k]["median_run_bars"]}
                                        for k in states if k.startswith("ctx_")},
           "report": report}
    (OUT_DIR / "seedbag.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(json.dumps({"regression_ok": regression_ok, "incumbent_reproduced_val": inc}), flush=True)
    print("=== per-seed (w=0.65, lambda=0.5)", flush=True)
    for s, v in per_seed.items():
        print(f"  seed {s:>7}  VAL {v['val']:5}  OOS {v['oos']:5}", flush=True)
    print(json.dumps({"seed_spread": spread}, indent=2), flush=True)
    print("=== context grid (no adoption possible)", flush=True)
    for k, v in out["context_only_no_adoption"].items():
        print(f"  {k:22} VAL {v['val']:5}  OOS {v['oos']:5}  run {v['median_run_bars']}", flush=True)
    print(json.dumps({"PRIMARY": out["primary_test"]["seedbag"], "ADOPT": adopt}, indent=2), flush=True)

    pd.DataFrame({"timestamp": ts, "close": close, "oracle005": y_dir,
                  "p_bag": p_bag, "p_single": p_single, "p_vote": p_vote,
                  **{k: v for k, v in states.items() if not k.startswith("seed_")}}
                 ).to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
