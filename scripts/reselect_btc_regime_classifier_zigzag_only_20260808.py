"""Re-selection of the theta=0.5% regime classifier using ZIGZAG GEOMETRY ONLY (2026-08-08).

Why this round exists: the lag audit's ablation found that a nowcaster trained on nothing but the
5 multi-scale causal-zigzag states beats the frozen 130-panel-feature config on BOTH VAL (68.0 vs
67.5) and OOS (66.5 vs 65.3).  The earlier selection rounds never ran a zigzag-only candidate, so
the freeze was revoked as an omission rather than swapped on an OOS number.  This script redoes
the selection properly: VAL-only, staged, with the zigzag-only family as the candidate space, and
exactly one OOS read at the end.

FEATURE FAMILY (no panel features at all):
  threshold sets   S1 current5  {0.2, 0.35, 0.5, 0.8, 1.2}%
                   S2 fine5     {0.1, 0.2, 0.35, 0.5, 0.8}%
                   S3 wide7     {0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0}%
                   S4 coarse5   {0.35, 0.5, 0.8, 1.2, 2.0}%
                   S5 fine3     {0.1, 0.2, 0.35}%
  feature modes    "state" = the causal zigzag direction per threshold
                   "geo"   = direction + log1p(bars since its last flip) + current overshoot from
                             that threshold's running extreme in theta units (3x features).  Both
                             are pure geometry of the causal zigzags; neither reads the panel.

PRE-REGISTERED PROCEDURE (fixed before the first run):
  seeds       N=5 drawn at random via default_rng(903174+1), recorded below; probability bagging.
  Stage 1     score all 10 (threshold set x mode) cells on VAL at the INHERITED blend/decode
              (w=0.65, lambda=0.5).  Pick the best VAL agreement subject to eligibility
              (coverage >= 50% AND median run >= 8 bars).
  Stage 2     for the Stage-1 winner ONLY, sweep lambda in {0.25,0.5,1,2} x w in {0.5,0.65,0.8,1.0}
              (w=1.0 means no vote blend).  No retraining is needed -- these are post-processing
              of the same probabilities.  Same eligibility, VAL agreement decides.
  OOS         ONE read of the Stage-2 winner.  ADOPT as the new frozen classifier only if
              OOS >= 65.6 (the revoked config's OOS) AND it is eligible on VAL.  If it misses,
              nothing is frozen and the line stops -- no third round on this validation window.
  26 VAL cells are scored in total; the single OOS read is the referee.
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
from ensemble_btc_regime_classifier_theta005_20260808 import logit, sigmoid  # noqa: E402
from refine_btc_regime_classifier_theta005_20260808 import (  # noqa: E402
    PANEL_PATH, PURGE, SCORE_SCALES, MIN_COVERAGE, MIN_MEDIAN_RUN,
    jump_decode_proba, summarize, to_named,
)
from test_statistical_jump_model_regimes_20260808 import zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    SEED, TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

OUT_DIR = ROOT / "tmp/regime_theta005_20260808"
OUT_PARQUET = ROOT / "data/research/btc_regime_theta005_zigzagonly_20260808.parquet"
THETA = 0.005
N_SEEDS = 5
THRESHOLD_SETS = {
    "S1_current5": [0.002, 0.0035, 0.005, 0.008, 0.012],
    "S2_fine5": [0.001, 0.002, 0.0035, 0.005, 0.008],
    "S3_wide7": [0.001, 0.002, 0.0035, 0.005, 0.008, 0.012, 0.020],
    "S4_coarse5": [0.0035, 0.005, 0.008, 0.012, 0.020],
    "S5_fine3": [0.001, 0.002, 0.0035],
}
MODES = ["state", "geo"]
STAGE1_W, STAGE1_LAM = 0.65, 0.5
LAMBDAS = [0.25, 0.5, 1.0, 2.0]
WEIGHTS = [0.5, 0.65, 0.8, 1.0]
ADOPT_OOS_BAR = 65.6


def zigzag_geometry(close: np.ndarray, theta: float):
    """Causal per-bar (direction, bars_since_flip, overshoot_in_theta_units) for one threshold.
    Same online state machine as causal_zigzag; nothing here reads a future bar."""
    n = len(close)
    d = np.zeros(n, dtype=np.float32)
    since = np.zeros(n, dtype=np.float32)
    over = np.zeros(n, dtype=np.float32)
    hi_i = lo_i = ext_i = 0
    up: bool | None = None
    last_flip = 0
    for t in range(1, n):
        if close[t] > close[hi_i]:
            hi_i = t
        if close[t] < close[lo_i]:
            lo_i = t
        flipped = False
        if up is None:
            if close[t] >= close[lo_i] * (1 + theta):
                up, ext_i, flipped = True, t, True
            elif close[t] <= close[hi_i] * (1 - theta):
                up, ext_i, flipped = False, t, True
        elif up:
            if close[t] > close[ext_i]:
                ext_i = t
            elif close[t] <= close[ext_i] * (1 - theta):
                up, ext_i, flipped = False, t, True
        else:
            if close[t] < close[ext_i]:
                ext_i = t
            elif close[t] >= close[ext_i] * (1 + theta):
                up, ext_i, flipped = True, t, True
        if flipped:
            last_flip = t
        d[t] = 0.0 if up is None else (1.0 if up else -1.0)
        since[t] = t - last_flip
        over[t] = (close[t] - close[ext_i]) / max(close[ext_i], 1e-12) / theta
    return d, np.log1p(since), over


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = sorted(int(s) for s in np.random.default_rng(SEED + 1).choice(1_000_000, size=N_SEEDS, replace=False))
    print(json.dumps({"seeds": seeds}), flush=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    train_mask = (ts <= TRAIN_END).to_numpy()
    oracles = {t: zigzag_oracle(close, threshold=t)[0] for t in SCORE_SCALES}
    y_dir = oracles[0.005]

    all_thetas = sorted({t for s in THRESHOLD_SETS.values() for t in s})
    geo = {}
    for t in all_thetas:
        geo[t] = zigzag_geometry(close, t)
        print(f"geometry built theta={t}", flush=True)

    tr_all = np.flatnonzero(train_mask)
    tr_idx = tr_all[:-PURGE]
    tr_idx = tr_idx[y_dir[tr_idx] != 0]
    y = (y_dir[tr_idx] == 1).astype(int)
    v_idx = np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy())
    o_idx = np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy())
    windows = {"val_2025Q4": v_idx, "oos_2026Q1": o_idx,
               "week": np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=7)).to_numpy())}

    def build(set_name, mode):
        thetas = THRESHOLD_SETS[set_name]
        cols = [geo[t][0] for t in thetas]
        if mode == "geo":
            cols += [geo[t][1] for t in thetas] + [geo[t][2] for t in thetas]
        xm = np.column_stack(cols).astype(np.float32)
        vote_sum = np.column_stack([geo[t][0] for t in thetas]).sum(axis=1).astype(int)
        prior = float(y.mean())
        tab = {}
        for v in range(-len(thetas), len(thetas) + 1):
            sel = vote_sum[tr_idx] == v
            tab[v] = float(y[sel].mean()) if sel.sum() >= 200 else prior
        p_vote = np.clip(np.vectorize(tab.get)(vote_sum).astype(np.float64), 0.02, 0.98)
        ps = []
        for s in seeds:
            clf = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05,
                                     num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                                     bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                                     random_state=s, n_jobs=-1, verbosity=-1)
            clf.fit(xm[tr_idx], y)
            ps.append(clf.predict_proba(xm)[:, 1])
        return np.mean(ps, axis=0), p_vote, xm.shape[1]

    def decode(p, p_vote, w, lam):
        z = logit(p) if w >= 1.0 else w * logit(p) + (1.0 - w) * logit(p_vote)
        return to_named(jump_decode_proba(sigmoid(z), lam))

    # ---- Stage 1
    probs, stage1 = {}, {}
    for sn in THRESHOLD_SETS:
        for md in MODES:
            p, pv, nfeat = build(sn, md)
            probs[(sn, md)] = (p, pv)
            st = decode(p, pv, STAGE1_W, STAGE1_LAM)
            s = summarize(st, oracles, v_idx)
            stage1[f"{sn}|{md}"] = {"n_features": nfeat, "val_agree": s["agree"]["0.005"],
                                    "coverage_pct": s["coverage_pct"], "median_run_bars": s["median_run_bars"]}
            print(json.dumps({f"{sn}|{md}": stage1[f"{sn}|{md}"]}), flush=True)
    elig1 = {k: v for k, v in stage1.items()
             if v["coverage_pct"] >= MIN_COVERAGE and v["median_run_bars"] >= MIN_MEDIAN_RUN}
    win1 = max(elig1, key=lambda k: (elig1[k]["val_agree"], elig1[k]["coverage_pct"])) if elig1 else None
    print(json.dumps({"STAGE1_WINNER": win1, **({} if win1 is None else stage1[win1])}, indent=2), flush=True)
    if win1 is None:
        (OUT_DIR / "reselect.json").write_text(json.dumps({"stage1": stage1, "winner": None}, indent=2))
        return 1

    # ---- Stage 2 (no retraining: post-processing the winner's probabilities)
    sn, md = win1.split("|")
    p, pv = probs[(sn, md)]
    stage2 = {}
    for w in WEIGHTS:
        for lam in LAMBDAS:
            st = decode(p, pv, w, lam)
            s = summarize(st, oracles, v_idx)
            stage2[f"w{int(w * 100)}_lam{lam:g}"] = {"val_agree": s["agree"]["0.005"],
                                                     "coverage_pct": s["coverage_pct"],
                                                     "median_run_bars": s["median_run_bars"]}
    elig2 = {k: v for k, v in stage2.items()
             if v["coverage_pct"] >= MIN_COVERAGE and v["median_run_bars"] >= MIN_MEDIAN_RUN}
    win2 = max(elig2, key=lambda k: (elig2[k]["val_agree"], elig2[k]["median_run_bars"])) if elig2 else None
    for k, v in stage2.items():
        print(f"  {k:14} VAL {v['val_agree']:6}  cov {v['coverage_pct']:6}  run {v['median_run_bars']}", flush=True)
    print(json.dumps({"STAGE2_WINNER": win2, **({} if win2 is None else stage2[win2])}, indent=2), flush=True)
    if win2 is None:
        (OUT_DIR / "reselect.json").write_text(json.dumps({"stage1": stage1, "stage2": stage2, "winner": None}, indent=2))
        return 1

    # ---- single OOS read
    w_sel = int(win2.split("_")[0][1:]) / 100.0
    lam_sel = float(win2.split("lam")[1])
    final = decode(p, pv, w_sel, lam_sel)
    rep = {wt: summarize(final, oracles, idx) for wt, idx in windows.items()}
    oos_agree = rep["oos_2026Q1"]["agree"]["0.005"]
    adopt = bool(oos_agree is not None and oos_agree >= ADOPT_OOS_BAR)
    out = {"seeds": seeds, "procedure": "stage1 featset/mode -> stage2 w/lambda -> single OOS read",
           "n_val_cells_scored": len(stage1) + len(stage2),
           "stage1": stage1, "stage1_winner": win1,
           "stage2": stage2, "stage2_winner": win2,
           "final": {"threshold_set": sn, "thresholds": THRESHOLD_SETS[sn], "mode": md,
                     "w": w_sel, "lambda": lam_sel},
           "measured": rep, "adopt_bar_oos": ADOPT_OOS_BAR, "oos_agreement_pct": oos_agree,
           "revoked_config_for_reference": {"val": 67.6, "oos": 65.6}, "adopt": adopt}
    (OUT_DIR / "reselect.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"FINAL": out["final"],
                      "val": rep["val_2025Q4"]["agree"]["0.005"], "oos": oos_agree,
                      "coverage": rep["val_2025Q4"]["coverage_pct"],
                      "median_run": rep["val_2025Q4"]["median_run_bars"],
                      "ADOPT": adopt}, indent=2), flush=True)

    pd.DataFrame({"timestamp": ts, "close": close, "oracle005": y_dir,
                  "p_zigzag": p, "p_vote": pv, "zigzagonly_final": final}).to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
