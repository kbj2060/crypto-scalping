"""Accuracy refinement round for the theta=0.5% regime classifier (2026-08-08).

Baseline to beat: czz05 (causal 0.5% directional change, no fitting) VAL 61.3 / OOS 60.8 at 100%
coverage and 13-18-bar runs.

Reopening argument for supervised learning here (it failed 4 times before): every prior attempt
predicted the 4% wave direction -- a multi-day swing whose driver is macro trend.  At theta=0.5%
the median wave is 16 bars (80 min) and amplitude 0.97%, so "which wave am I in" is a NOWCASTING
question about whether a reversal has already happened, which is what the panel's order-flow /
positioning / CVD features describe.  Different question, different scale, so it gets one attempt
under the same pre-registered bar.

The central idea of this round: supervised models were accurate but produced 1-bar confetti
(CNN 72% full-period agreement, unusable).  Instead of thresholding probabilities per bar, decode
them with the Jump Model's own machinery -- a CAUSAL online DP with an explicit per-transition
penalty lambda over the 2-state probability sequence:
    V_t(s) = -log p_t(s) + min( V_{t-1}(s),  min_s' V_{t-1}(s') + lambda ),   state_t = argmin V_t
That buys supervised accuracy AND jump-model persistence in one object, and lambda is a single
knob swept on VAL.

CANDIDATES (all scored vs the theta=0.005 oracle):
  czz05                     the incumbent (carried through for comparison)
  vote_multi                multi-threshold DC vote: causal zigzag at {0.2,0.35,0.5,0.8,1.2}%,
                            sign-sum; |sum|>=vote_min -> directional else chop (Tsang's
                            multi-threshold DC idea, no fitting)
  lgbm_raw                  per-bar argmax of the nowcaster probability (the confetti control --
                            included precisely to show the decoder is doing the work)
  lgbm_jm_lam{L}            nowcaster probability + jump-penalized causal decode, L in the sweep
  lgbm_jm_lam{L}_m{M}       same, plus an abstention margin M on the decoded state's probability

PRE-REGISTERED (unchanged from the selection round, so results are comparable):
  primary   = VAL agreement vs the theta=0.005 oracle
  eligible  = coverage >= 50% AND median run >= 8 bars
  tiebreak  = coverage, then median run
  OOS       = single confirmation read on the selected candidate only
  ADOPT the refinement only if it beats czz05 on VAL **and** its OOS does not fall below czz05's
  OOS (60.8) -- a VAL-only win is exactly the failure mode this project has hit 10 times.
Label purge: 576 bars (36x the 16-bar median wave) dropped from the end of TRAIN.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_btc_regime_classifier_zoo_20260808 import REGIME_COLORS, C_BULL, C_BEAR, C_CHOP, INK  # noqa: E402
from chart_btc_jm_regime_verification_20260808 import causal_zigzag  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs, zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, SEED, TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026_regimeline.csv"
OUT_PARQUET = ROOT / "data/research/btc_regime_theta005_refined_20260808.parquet"
OUT_DIR = ROOT / "tmp/regime_theta005_20260808"
THETA = 0.005
VOTE_THETAS = [0.002, 0.0035, 0.005, 0.008, 0.012]
VOTE_MIN = 3
LAMBDAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
MARGINS = [0.55, 0.60]
PURGE = 576
SCORE_SCALES = [0.005, 0.010, 0.015]
MIN_COVERAGE, MIN_MEDIAN_RUN = 50.0, 8.0
CZZ05_VAL, CZZ05_OOS = 61.3, 60.8

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def jump_decode_proba(p_bull: np.ndarray, lam: float) -> np.ndarray:
    """Causal online DP over the 2-state negative-log-likelihood sequence with a per-transition
    penalty lambda.  Same recursion as the Jump Model's causal decode, but the emission cost comes
    from a supervised probability instead of a distance to a cluster centre.  Uses only p_<=t."""
    eps = 1e-6
    cost = np.column_stack([-np.log(np.clip(1.0 - p_bull, eps, 1.0)),
                            -np.log(np.clip(p_bull, eps, 1.0))])
    n = len(p_bull)
    out = np.zeros(n, dtype=np.int8)
    V = cost[0].copy()
    out[0] = int(V.argmin())
    for t in range(1, n):
        switch = V.min() + lam
        V = cost[t] + np.minimum(V, switch)
        V -= V.min()
        out[t] = int(V.argmin())
    return out


def to_named(bull_flag: np.ndarray) -> np.ndarray:
    return np.where(bull_flag == 1, 2, 0).astype(np.int8)


def agreement(named, oracle_dir, idx):
    det = np.where(named == 2, 1, np.where(named == 0, -1, 0))
    act = det[idx] != 0
    if not act.any():
        return None
    return round(float(np.mean(det[idx][act] == oracle_dir[idx][act])) * 100, 1)


def summarize(named, oracles, idx):
    runs = [e - s + 1 for s, e, _ in contiguous_runs(named[idx])]
    det = np.where(named == 2, 1, np.where(named == 0, -1, 0))
    return {"agree": {f"{t:.3f}": agreement(named, oracles[t], idx) for t in SCORE_SCALES},
            "coverage_pct": round(float((det[idx] != 0).mean()) * 100, 1),
            "median_run_bars": float(np.median(runs)), "n_flips": len(runs) - 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    # multi-threshold causal DC states, used both as a standalone vote and as model inputs
    czz = {t: causal_zigzag(close, threshold=t) for t in VOTE_THETAS}
    czz_mat = np.column_stack([czz[t] for t in VOTE_THETAS]).astype(np.float32)
    x_aug = np.column_stack([x, czz_mat])
    aug_cols = feat_cols + [f"czz_{int(t * 1000)}" for t in VOTE_THETAS]
    print(json.dumps({"n_features": len(aug_cols)}), flush=True)

    states: dict[str, np.ndarray] = {}
    cz5 = czz[0.005]
    states["czz05"] = np.where(cz5 == 1, 2, np.where(cz5 == -1, 0, 1)).astype(np.int8)

    vote = czz_mat.sum(axis=1)
    states["vote_multi"] = np.where(vote >= VOTE_MIN, 2, np.where(vote <= -VOTE_MIN, 0, 1)).astype(np.int8)

    tr_all = np.flatnonzero(train_mask)
    tr_idx = tr_all[:-PURGE]
    tr_idx = tr_idx[(y_dir[tr_idx] != 0) & np.isfinite(x_aug[tr_idx]).any(axis=1)]
    y = (y_dir[tr_idx] == 1).astype(int)
    print(json.dumps({"train_rows": int(len(tr_idx)), "bull_frac": round(float(y.mean()), 3)}), flush=True)

    clf = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05,
                             num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                             bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                             random_state=SEED, n_jobs=-1, verbosity=-1)
    clf.fit(x_aug[tr_idx], y)
    p_bull = clf.predict_proba(x_aug)[:, 1]
    gains = sorted(zip(aug_cols, clf.booster_.feature_importance("gain")), key=lambda kv: -kv[1])[:10]
    print(json.dumps({"top_gain": [g[0] for g in gains]}), flush=True)

    states["lgbm_raw"] = to_named((p_bull > 0.5).astype(int))
    for lam in LAMBDAS:
        dec = jump_decode_proba(p_bull, lam)
        states[f"lgbm_jm_lam{lam:g}"] = to_named(dec)
        for m in MARGINS:
            conf = np.where(dec == 1, p_bull, 1.0 - p_bull)
            nm = to_named(dec).copy()
            nm[conf < m] = 1
            states[f"lgbm_jm_lam{lam:g}_m{int(m * 100)}"] = nm
        print(f"decoded lam={lam:g}", flush=True)

    windows = {
        "val_2025Q4": np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()),
        "oos_2026Q1": np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()),
        "full": np.arange(len(close)),
        "week": np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=args.days)).to_numpy()),
    }
    report = {w: {k: summarize(v, oracles, idx) for k, v in states.items()} for w, idx in windows.items()}
    val = report["val_2025Q4"]
    eligible = {k: v for k, v in val.items()
                if v["coverage_pct"] >= MIN_COVERAGE and v["median_run_bars"] >= MIN_MEDIAN_RUN
                and v["agree"]["0.005"] is not None}
    selected = max(eligible, key=lambda k: (eligible[k]["agree"]["0.005"], eligible[k]["coverage_pct"],
                                            eligible[k]["median_run_bars"])) if eligible else None
    sel_oos = None if selected is None else report["oos_2026Q1"][selected]["agree"]["0.005"]
    adopt = bool(selected is not None and selected != "czz05"
                 and val[selected]["agree"]["0.005"] > CZZ05_VAL and (sel_oos or 0) >= CZZ05_OOS)
    out = {"theta": THETA, "baseline": {"czz05_val": CZZ05_VAL, "czz05_oos": CZZ05_OOS},
           "eligibility": {"min_coverage_pct": MIN_COVERAGE, "min_median_run_bars": MIN_MEDIAN_RUN},
           "adopt_rule": "selected != czz05 AND val > 61.3 AND oos >= 60.8",
           "top_gain_features": [g[0] for g in gains],
           "report": report, "eligible": list(eligible), "selected": selected,
           "selected_oos_agree_005": sel_oos, "adopt": adopt}
    (OUT_DIR / "refinement.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for w in ("val_2025Q4", "oos_2026Q1"):
        print(f"=== {w}", flush=True)
        for k, v in report[w].items():
            print(f"  {k:22} agree {v['agree']['0.005']:6}  cov {v['coverage_pct']:6}  "
                  f"run {v['median_run_bars']:7}  flips {v['n_flips']}", flush=True)
    print(json.dumps({"SELECTED": selected, "val": None if selected is None else val[selected]["agree"]["0.005"],
                      "oos": sel_oos, "ADOPT": adopt}, indent=2), flush=True)

    pd.DataFrame({"timestamp": ts, "close": close, "oracle005": y_dir,
                  "p_bull": p_bull, **states}).to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}", flush=True)

    widx = windows["week"]
    h_ts = ts.to_numpy()[widx]
    show = [selected or "czz05", "czz05", "vote_multi", "lgbm_raw"]
    show += [k for k in ("lgbm_jm_lam2", "lgbm_jm_lam8") if k not in show]
    onamed = np.where(y_dir == 1, 2, np.where(y_dir == -1, 0, 1)).astype(np.int8)
    fig, axes = plt.subplots(2 + len(show), 1, figsize=(15, 9), sharex=True,
                             gridspec_kw={"height_ratios": [10] + [0.72] * (len(show) + 1), "hspace": 0.08})
    ax = axes[0]
    main_named = states[show[0]]
    for s, e, stt in contiguous_runs(main_named[widx]):
        seg = slice(s, min(e + 2, len(widx)))
        ax.plot(h_ts[seg], close[widx][seg], color=REGIME_COLORS[stt], linewidth=1.4)
    v0 = val[show[0]]
    ax.set_title(f"정확도 개선 라운드 — 라인 = {show[0]}  VAL {v0['agree']['0.005']}% / "
                 f"OOS {report['oos_2026Q1'][show[0]]['agree']['0.005']}%  "
                 f"커버리지 {v0['coverage_pct']}%  중앙런 {int(v0['median_run_bars'])}bar   "
                 f"(기준 czz05 {CZZ05_VAL}/{CZZ05_OOS}) — 최근 {args.days}일",
                 loc="left", fontsize=12, color=INK)
    ax.grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in
                       (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
              loc="upper left", frameon=False, fontsize=9, ncol=3)
    strips = [(onamed, "오라클 0.5% (사후)  ")] + [(states[k], k + "  ") for k in show]
    for sax, (arr, lb) in zip(axes[1:], strips):
        for s, e, stt in contiguous_runs(arr[widx]):
            sax.axvspan(h_ts[s], h_ts[min(e + 1, len(widx) - 1)], color=REGIME_COLORS[stt], linewidth=0)
        sax.set_yticks([])
        sax.set_ylabel(lb, rotation=0, ha="right", va="center", fontsize=9, color=INK)
        for side in ("top", "right", "left", "bottom"):
            sax.spines[side].set_visible(False)
    outp = OUT_DIR / "week_refined_theta005.png"
    fig.savefig(outp, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
