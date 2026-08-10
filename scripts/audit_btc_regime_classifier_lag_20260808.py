"""Is the frozen theta=0.5% classifier just a LAGGED COPY of the oracle? (2026-08-08 audit)

User's challenge: the ensemble's 65.6% OOS agreement might come from nothing but following the
oracle's wave state with a delay -- i.e. it "wins" by switching after the turn has already
happened, which any mechanical rule does for free.

Framing: a causal detector of a retrospective wave label MUST lag; a pivot is unknowable before
its confirmation move.  So the question is not "does it lag" but "does it beat the mechanical
lag" -- i.e. does it add information beyond the causal zigzag czz05, which is exactly the
free lagged detector at this threshold.  Five tests, none of which can be passed by a lagged copy:

  A  LAG PROFILE      agreement(state_t, oracle_{t-k}) for k in [-288, 288].  A pure lagged copy
                      peaks at k = its delay and is markedly worse at k=0.  A genuine nowcaster
                      peaks at or near k=0.  czz05 is measured identically as the lag reference.
  B  DETECTION LAG    bars from each oracle pivot until the detector shows the new direction.
                      Beating czz05 here means turning earlier than the confirmation rule.
  C  WAVE-POSITION    agreement by quintile of position within the oracle wave.  A lagged copy is
                      near-0% in quintile 1 (it has not caught up yet) and near-100% in Q5.
  D  VALUE ADDED      agreement restricted to bars where czz05 DISAGREES with the oracle.  A copy
                      of czz05 scores ~0% there by construction; anything above ~50% on that
                      subset is information czz05 does not have.
  E  ABLATION         retrain the nowcaster (i) without the causal-zigzag features and (ii) with
                      ONLY the zigzag features, then rebuild the full pipeline for each.  If the
                      panel-only variant keeps most of the edge, the model is not a zigzag echo.

All measured on VAL and OOS separately.  Diagnostic only -- no selection, nothing is adopted here.
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
    PANEL_PATH, PURGE, VOTE_THETAS, jump_decode_proba, to_named,
)
from test_statistical_jump_model_regimes_20260808 import zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

SEEDBAG_PATH = ROOT / "data/research/btc_regime_theta005_seedbag_20260808.parquet"
FROZEN_PATH = ROOT / "data/research/btc_regime_theta005_frozen_config_20260808.json"
OUT_DIR = ROOT / "tmp/regime_theta005_20260808"
THETA = 0.005
W, LAM = 0.65, 0.5
LAGS = list(range(-288, 289, 6))


def dir_of(named: np.ndarray) -> np.ndarray:
    return np.where(named == 2, 1, np.where(named == 0, -1, 0)).astype(np.int8)


def agree(d: np.ndarray, o: np.ndarray, idx: np.ndarray) -> float | None:
    m = idx[(d[idx] != 0) & (o[idx] != 0)]
    return None if len(m) < 50 else round(float(np.mean(d[m] == o[m])) * 100, 1)


def lag_profile(d, o, idx):
    prof = {}
    n = len(o)
    for k in LAGS:
        src = idx - k
        keep = (src >= 0) & (src < n)
        m = idx[keep]
        s = src[keep]
        sel = (d[m] != 0) & (o[s] != 0)
        prof[k] = round(float(np.mean(d[m][sel] == o[s][sel])) * 100, 1) if sel.sum() > 50 else None
    best = max((v for v in prof.values() if v is not None), default=None)
    peak = next((k for k, v in prof.items() if v == best), None)
    return {"peak_lag_bars": peak, "peak_agreement_pct": best, "at_lag0_pct": prof.get(0), "profile": prof}


def detection_lag(d, o, pivots, lo, hi, horizon=576):
    lags = []
    for p in pivots:
        if not (lo <= p <= hi - horizon):
            continue
        target = o[min(p + 1, len(o) - 1)]
        if target == 0:
            continue
        w = d[p: p + horizon]
        hit = np.flatnonzero(w == target)
        if len(hit):
            lags.append(int(hit[0]))
    return {"median_bars": float(np.median(lags)) if lags else None,
            "mean_bars": round(float(np.mean(lags)), 1) if lags else None, "n_pivots": len(lags)}


def wave_position(o, pivots, n):
    pos = np.full(n, np.nan)
    bounds = list(pivots) + [n - 1]
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b > a:
            pos[a:b] = np.linspace(0.0, 1.0, b - a, endpoint=False)
    return pos


def by_quintile(d, o, pos, idx):
    out = {}
    for q in range(5):
        m = idx[(pos[idx] >= q / 5) & (pos[idx] < (q + 1) / 5)]
        out[f"Q{q + 1}"] = agree(d, o, m)
    return out


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(FROZEN_PATH.read_text())
    seeds = frozen["pipeline"]["2_nowcaster"]["seed_bag"]

    bag = pd.read_parquet(SEEDBAG_PATH)
    ts = pd.to_datetime(bag["timestamp"])
    close = bag["close"].to_numpy(dtype=np.float64)
    o_dir, pivots = zigzag_oracle(close, threshold=THETA)
    final = bag["seedbag_primary"].to_numpy().astype(np.int8)
    d_final = dir_of(final)
    czz = causal_zigzag(close, threshold=THETA)
    d_czz = czz.astype(np.int8)

    v_idx = np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy())
    o_idx = np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy())
    pos = wave_position(o_dir, pivots, len(close))
    res: dict = {"theta": THETA, "detectors": ["frozen_ensemble", "czz05"]}

    for wname, idx in (("val_2025Q4", v_idx), ("oos_2026Q1", o_idx)):
        blk = {}
        blk["A_lag_profile"] = {"frozen": lag_profile(d_final, o_dir, idx),
                                "czz05": lag_profile(d_czz, o_dir, idx)}
        blk["B_detection_lag"] = {"frozen": detection_lag(d_final, o_dir, pivots, idx[0], idx[-1]),
                                  "czz05": detection_lag(d_czz, o_dir, pivots, idx[0], idx[-1])}
        blk["C_by_wave_quintile"] = {"frozen": by_quintile(d_final, o_dir, pos, idx),
                                     "czz05": by_quintile(d_czz, o_dir, pos, idx)}
        wrong = idx[(d_czz[idx] != 0) & (o_dir[idx] != 0) & (d_czz[idx] != o_dir[idx])]
        right = idx[(d_czz[idx] != 0) & (o_dir[idx] != 0) & (d_czz[idx] == o_dir[idx])]
        blk["D_value_added"] = {
            "frozen_where_czz_wrong_pct": agree(d_final, o_dir, wrong), "n_bars_czz_wrong": int(len(wrong)),
            "frozen_where_czz_right_pct": agree(d_final, o_dir, right), "n_bars_czz_right": int(len(right)),
            "frozen_equals_czz_pct": round(float(np.mean(d_final[idx] == d_czz[idx])) * 100, 1)}
        res[wname] = blk
        print(f"=== {wname}", flush=True)
        print(json.dumps({k: v for k, v in blk.items() if k != "A_lag_profile"}, indent=2), flush=True)
        print(json.dumps({"A_peaks": {m: {kk: blk["A_lag_profile"][m][kk] for kk in
                                          ("peak_lag_bars", "peak_agreement_pct", "at_lag0_pct")}
                                      for m in ("frozen", "czz05")}}, indent=2), flush=True)

    # ---- E: ablation
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    czz_mat = np.column_stack([causal_zigzag(close, threshold=t) for t in VOTE_THETAS]).astype(np.float32)
    train_mask = (ts <= TRAIN_END).to_numpy()
    tr_all = np.flatnonzero(train_mask)
    tr_idx = tr_all[:-PURGE]
    tr_idx = tr_idx[o_dir[tr_idx] != 0]
    y = (o_dir[tr_idx] == 1).astype(int)
    vote_sum = czz_mat.sum(axis=1).astype(int)
    prior = float(y.mean())
    tab = {}
    for v in range(-len(VOTE_THETAS), len(VOTE_THETAS) + 1):
        sel = vote_sum[tr_idx] == v
        tab[v] = float(y[sel].mean()) if sel.sum() >= 200 else prior
    p_vote = np.clip(np.vectorize(tab.get)(vote_sum).astype(np.float64), 0.02, 0.98)

    variants = {"full (panel+czz)": np.column_stack([x, czz_mat]),
                "panel_only (no czz)": x,
                "czz_only": czz_mat}
    abl = {}
    for name, xm in variants.items():
        ps = []
        for s in seeds[:4]:  # 4 of the 8 frozen seeds -- ablation is a diagnostic, not a selection
            clf = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05,
                                     num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                                     bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                                     random_state=s, n_jobs=-1, verbosity=-1)
            clf.fit(xm[tr_idx], y)
            ps.append(clf.predict_proba(xm)[:, 1])
        p = np.mean(ps, axis=0)
        st = to_named(jump_decode_proba(sigmoid(W * logit(p) + (1 - W) * logit(p_vote)), LAM))
        d = dir_of(st)
        wrong_v = v_idx[(d_czz[v_idx] != 0) & (o_dir[v_idx] != 0) & (d_czz[v_idx] != o_dir[v_idx])]
        wrong_o = o_idx[(d_czz[o_idx] != 0) & (o_dir[o_idx] != 0) & (d_czz[o_idx] != o_dir[o_idx])]
        abl[name] = {"val": agree(d, o_dir, v_idx), "oos": agree(d, o_dir, o_idx),
                     "val_where_czz_wrong": agree(d, o_dir, wrong_v),
                     "oos_where_czz_wrong": agree(d, o_dir, wrong_o),
                     "val_detection_lag_median": detection_lag(d, o_dir, pivots, v_idx[0], v_idx[-1])["median_bars"]}
        print(json.dumps({name: abl[name]}), flush=True)
    res["E_ablation_4seed"] = abl
    res["note"] = ("blend/decode held at the frozen w=0.65, lambda=0.5; ablation uses 4 of the 8 "
                   "frozen seeds and is diagnostic only -- nothing here selects or adopts.")
    (OUT_DIR / "lag_audit.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_DIR / 'lag_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
