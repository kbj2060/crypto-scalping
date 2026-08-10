"""Oracle-wave regime classifier (2026-08-08, user-proposed): supervised prediction of the
ZIGZAG ORACLE's current wave direction (up-wave=bull / down-wave=bear) from causal features.

Baseline to beat: the causal 4% zigzag itself -- oracle agreement 67.2%, pivot lag 190 bars.
A classifier that beats it causally "knows the turn before the 4% confirmation".

Setup:
  - target: oracle wave direction at bar t (binary, from zigzag_oracle 4%) -- retrospective label,
    fine as a TRAINING target, never as an input;
  - features: the BTC regimeline panel (130 causal features) + the causal zigzag state and bars/
    move since its last confirmed flip (the lagging baseline handed to the model as inputs);
  - train <= 2025-08-31 minus a 2880-bar purge (oracle labels near the boundary depend on future
    waves; median wave 692 bars, purge is ~4x that), VAL 2025-09..12-31;
  - scoring on VAL: oracle agreement, pivot lag, median run length -- the detector scorecard.
    Threshold grid on P(up) {0.5 fixed, hysteresis 0.6/0.4} for run-length control.
NO PnL claim; this is a detector experiment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from test_statistical_jump_model_regimes_20260808 import zigzag_oracle, causal_zigzag_regime, contiguous_runs, runs_of  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import RAW_LEVEL_COLS, TRAIN_END, VAL_START, VAL_END, SEED  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026_regimeline.csv"
OUT_DIR = ROOT / "tmp/jump_model_regimes_20260808"
THRESH = 0.04
PURGE = 2880
C_BULL, C_BEAR, C_CHOP = "#2563EB", "#D9542B", "#9AA0A6"
INK = "#1F2430"


def czz_state_features(close: np.ndarray, cdir: np.ndarray):
    """bars since the causal state's last flip + signed move since flip (causal)."""
    n = len(close)
    bars_since = np.zeros(n)
    move_since = np.zeros(n)
    last_flip = 0
    for t in range(1, n):
        if cdir[t] != cdir[t - 1]:
            last_flip = t
        bars_since[t] = t - last_flip
        move_since[t] = close[t] / close[last_flip] - 1.0
    return bars_since, move_since


def lag_and_agreement(det_dir: np.ndarray, odir: np.ndarray, pivots: list[int], idx: np.ndarray):
    active = det_dir[idx] != 0
    agree = float(np.mean(det_dir[idx][active] == odir[idx][active])) if active.any() else np.nan
    lags = []
    for p in pivots[1:]:
        if not (idx[0] <= p <= idx[-1] - 288):
            continue
        d = odir[min(p + 1, len(odir) - 1)]
        window = det_dir[p: min(p + 864, idx[-1])]
        hits = np.flatnonzero(window == d)
        if len(hits):
            lags.append(int(hits[0]))
    return round(agree * 100, 1), (int(np.median(lags)) if lags else None), len(lags)


def main() -> int:
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x_panel = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)

    odir, pivots = zigzag_oracle(close, THRESH)
    cdir = causal_zigzag_regime(close, THRESH)
    bars_since, move_since = czz_state_features(close, cdir)
    x = np.column_stack([x_panel, cdir.astype(np.float32), bars_since.astype(np.float32), move_since.astype(np.float32)])
    all_cols = feat_cols + ["czz_dir", "czz_bars_since_flip", "czz_move_since_flip"]

    train_mask = (ts <= TRAIN_END).to_numpy()
    tr_all = np.flatnonzero(train_mask)
    train_mask[tr_all[-PURGE:]] = False
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    tr_idx = np.flatnonzero(train_mask & (odir != 0))
    v_idx = np.flatnonzero(val_mask)

    y = (odir > 0).astype(int)
    clf = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.05, num_leaves=63,
                             min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
                             bagging_freq=1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1)
    clf.fit(x[tr_idx], y[tr_idx])
    p_up = np.zeros(len(close))
    p_up[v_idx] = clf.booster_.predict(x[v_idx])

    variants = {}
    det_fixed = np.zeros(len(close), dtype=np.int8)
    det_fixed[v_idx] = np.where(p_up[v_idx] >= 0.5, 1, -1)
    variants["model_p50"] = det_fixed
    det_h = np.zeros(len(close), dtype=np.int8)
    cur = 0
    for t in v_idx:
        if p_up[t] >= 0.6:
            cur = 1
        elif p_up[t] <= 0.4:
            cur = -1
        det_h[t] = cur
    variants["model_hyst_60_40"] = det_h

    report = {}
    for name, det in variants.items():
        agree, lag, n_p = lag_and_agreement(det, odir, pivots, v_idx)
        active = det[v_idx] != 0
        runs = runs_of(det[v_idx][active]) if active.any() else [0]
        report[name] = {"val_oracle_agreement_pct": agree, "median_pivot_lag_bars": lag,
                        "median_run_bars": float(np.median(runs)), "coverage_pct": round(float(active.mean()) * 100, 1)}
    agree_b, lag_b, _ = lag_and_agreement(cdir, odir, pivots, v_idx)
    report["baseline_causal_zigzag"] = {"val_oracle_agreement_pct": agree_b, "median_pivot_lag_bars": lag_b,
                                        "median_run_bars": float(np.median(runs_of(cdir[v_idx]))), "coverage_pct": 100.0}
    imp = sorted(zip(all_cols, clf.booster_.feature_importance(importance_type="gain")), key=lambda kv: -kv[1])
    report["top10_features_by_gain"] = [k for k, _ in imp[:10]]
    (OUT_DIR / "oracle_wave_classifier_eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # 6mo VAL-era chart: model vs baseline vs oracle
    idx = v_idx[::3]
    h_ts = ts.to_numpy()[idx]
    fig, axes = plt.subplots(4, 1, figsize=(16, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [10, 0.7, 0.7, 0.7], "hspace": 0.07})
    ax = axes[0]
    det = variants["model_hyst_60_40"]
    s3 = np.where(det > 0, 2, np.where(det < 0, 0, 1))
    for s, e, stt in contiguous_runs(s3[idx]):
        ax.axvspan(h_ts[s], h_ts[e], color={0: C_BEAR, 1: C_CHOP, 2: C_BULL}[stt], alpha=0.16, linewidth=0)
    ax.plot(h_ts, close[idx], color=INK, linewidth=1.0)
    ax.set_title("BTC VAL (2025-09..12) — LEARNED oracle-wave regime (hysteresis 0.6/0.4) vs causal zigzag vs oracle",
                 loc="left", fontsize=12, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=C_BULL, alpha=0.6, label="bull"), Patch(facecolor=C_BEAR, alpha=0.6, label="bear")],
              loc="upper left", frameon=False, fontsize=9, ncol=2)
    for strip_ax, dd, label in ((axes[1], det, "learned model  "),
                                (axes[2], cdir, "causal zigzag  "),
                                (axes[3], odir, "ORACLE  ")):
        s3 = np.where(dd > 0, 2, np.where(dd < 0, 0, 1))
        for s, e, stt in contiguous_runs(s3[idx]):
            strip_ax.axvspan(h_ts[s], h_ts[e], color={0: C_BEAR, 1: C_CHOP, 2: C_BULL}[stt], alpha=0.9, linewidth=0)
        strip_ax.set_yticks([])
        strip_ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=INK)
        for side in ("top", "right", "left", "bottom"):
            strip_ax.spines[side].set_visible(False)
    fig.savefig(OUT_DIR / "oracle_wave_classifier_val.png", dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_DIR / 'oracle_wave_classifier_val.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
