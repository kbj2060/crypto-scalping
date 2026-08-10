"""Ensemble round for the theta=0.5% regime classifier (2026-08-08).

Motivation from the refinement round: the multi-threshold DC vote scored the highest agreement of
anything tried (VAL 69.5 / OOS 68.4) but was disqualified for 5-6-bar runs, while the adopted
lgbm_jm_lam1 sits at VAL 66.0 / OOS 63.5 with 9-bar runs.  The vote's information may be additive
to the LGBM's, so this round turns the vote into a calibrated PROBABILITY and combines it with the
nowcaster in logit space before the same jump-penalized causal decode supplies persistence.

Members
  p_lgbm   the refinement round's nowcaster, rebuilt with identical config/seed so the
           lgbm_jm_lam1 numbers reproduce exactly (regression-checked below).
  p_vote   calibrated multi-threshold DC vote: vote_sum = sum of causal zigzag directions at
           {0.2, 0.35, 0.5, 0.8, 1.2}% in {-5..5}; P(bull | vote_sum) estimated EMPIRICALLY ON
           TRAIN ONLY (purged), then looked up per bar.  No future information: both the zigzag
           states and the lookup table are causal/train-fit.
Combination
  logit(p_ens) = w * logit(p_lgbm) + (1 - w) * logit(p_vote),  w in {0.2, 0.35, 0.5, 0.65, 0.8}
  state        = jump-penalized causal decode of p_ens, lambda in {0.5, 1, 2, 4}

PRE-REGISTERED (before the first run; same bar as the previous two rounds so results compose):
  primary   = VAL agreement vs the theta=0.005 oracle
  eligible  = coverage >= 50% AND median run >= 8 bars
  tiebreak  = coverage, then median run
  OOS       = single confirmation read on the selected candidate only
  ADOPT only if VAL > 66.0 AND OOS >= 63.5 (i.e. it must beat the incumbent lgbm_jm_lam1 on VAL
  and not give back OOS).  20 (w, lambda) cells are scored, so VAL selection is
  multiple-comparison inflated by construction; the OOS floor is the protection, and a VAL-only
  win is recorded as a failure, not a result.
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
from refine_btc_regime_classifier_theta005_20260808 import (  # noqa: E402
    PANEL_PATH, PURGE, SCORE_SCALES, VOTE_THETAS, MIN_COVERAGE, MIN_MEDIAN_RUN,
    jump_decode_proba, summarize, to_named,
)
from test_statistical_jump_model_regimes_20260808 import contiguous_runs, zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, SEED, TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

OUT_PARQUET = ROOT / "data/research/btc_regime_theta005_ensemble_20260808.parquet"
OUT_DIR = ROOT / "tmp/regime_theta005_20260808"
WEIGHTS = [0.2, 0.35, 0.5, 0.65, 0.8]
LAMBDAS = [0.5, 1.0, 2.0, 4.0]
INCUMBENT_VAL, INCUMBENT_OOS = 66.0, 63.5
EPS = 1e-4

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


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
    czz = {t: causal_zigzag(close, threshold=t) for t in VOTE_THETAS}
    czz_mat = np.column_stack([czz[t] for t in VOTE_THETAS]).astype(np.float32)
    x_aug = np.column_stack([x, czz_mat])

    tr_all = np.flatnonzero(train_mask)
    tr_idx = tr_all[:-PURGE]
    tr_idx = tr_idx[(y_dir[tr_idx] != 0) & np.isfinite(x_aug[tr_idx]).any(axis=1)]
    y = (y_dir[tr_idx] == 1).astype(int)

    clf = lgb.LGBMClassifier(objective="binary", n_estimators=700, learning_rate=0.05,
                             num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                             bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                             random_state=SEED, n_jobs=-1, verbosity=-1)
    clf.fit(x_aug[tr_idx], y)
    p_lgbm = clf.predict_proba(x_aug)[:, 1]
    print("nowcaster rebuilt", flush=True)

    vote_sum = czz_mat.sum(axis=1).astype(int)
    tab, prior = {}, float(y.mean())
    for v in range(-len(VOTE_THETAS), len(VOTE_THETAS) + 1):
        m = tr_idx[vote_sum[tr_idx] == v]
        tab[v] = float(y[vote_sum[tr_idx] == v].mean()) if len(m) >= 200 else prior
    p_vote = np.clip(np.vectorize(tab.get)(vote_sum).astype(np.float64), 0.02, 0.98)
    print(json.dumps({"vote_calibration_train": {str(k): round(v, 3) for k, v in tab.items()}}), flush=True)

    states: dict[str, np.ndarray] = {}
    windows = {
        "val_2025Q4": np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()),
        "oos_2026Q1": np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()),
        "full": np.arange(len(close)),
        "week": np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=args.days)).to_numpy()),
    }
    # incumbent, rebuilt for an exact regression check
    states["lgbm_jm_lam1"] = to_named(jump_decode_proba(p_lgbm, 1.0))
    states["vote_only_jm_lam1"] = to_named(jump_decode_proba(p_vote, 1.0))
    for w in WEIGHTS:
        p_ens = sigmoid(w * logit(p_lgbm) + (1.0 - w) * logit(p_vote))
        for lam in LAMBDAS:
            states[f"ens_w{int(w * 100)}_lam{lam:g}"] = to_named(jump_decode_proba(p_ens, lam))
        print(f"decoded w={w}", flush=True)

    report = {wt: {k: summarize(v, oracles, idx) for k, v in states.items()} for wt, idx in windows.items()}
    inc = report["val_2025Q4"]["lgbm_jm_lam1"]["agree"]["0.005"]
    regression_ok = bool(abs(inc - INCUMBENT_VAL) <= 0.2)
    print(json.dumps({"incumbent_reproduced_val": inc, "regression_ok": regression_ok}), flush=True)

    val = report["val_2025Q4"]
    cand = {k: v for k, v in val.items() if k.startswith("ens_")}
    eligible = {k: v for k, v in cand.items()
                if v["coverage_pct"] >= MIN_COVERAGE and v["median_run_bars"] >= MIN_MEDIAN_RUN
                and v["agree"]["0.005"] is not None}
    selected = max(eligible, key=lambda k: (eligible[k]["agree"]["0.005"], eligible[k]["coverage_pct"],
                                            eligible[k]["median_run_bars"])) if eligible else None
    sel_val = None if selected is None else val[selected]["agree"]["0.005"]
    sel_oos = None if selected is None else report["oos_2026Q1"][selected]["agree"]["0.005"]
    adopt = bool(selected is not None and sel_val > INCUMBENT_VAL and (sel_oos or 0) >= INCUMBENT_OOS)
    out = {"incumbent": {"name": "lgbm_jm_lam1", "val": INCUMBENT_VAL, "oos": INCUMBENT_OOS,
                         "reproduced_val": inc, "regression_ok": regression_ok},
           "adopt_rule": f"val > {INCUMBENT_VAL} AND oos >= {INCUMBENT_OOS}",
           "eligibility": {"min_coverage_pct": MIN_COVERAGE, "min_median_run_bars": MIN_MEDIAN_RUN},
           "n_cells_scored": len(cand), "report": report, "eligible": list(eligible),
           "selected": selected, "selected_val": sel_val, "selected_oos": sel_oos, "adopt": adopt}
    (OUT_DIR / "ensemble.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for wt in ("val_2025Q4", "oos_2026Q1"):
        print(f"=== {wt}", flush=True)
        for k, v in report[wt].items():
            print(f"  {k:22} agree {v['agree']['0.005']:6}  cov {v['coverage_pct']:6}  "
                  f"run {v['median_run_bars']:7}  flips {v['n_flips']}", flush=True)
    print(json.dumps({"SELECTED": selected, "val": sel_val, "oos": sel_oos, "ADOPT": adopt}, indent=2), flush=True)

    pd.DataFrame({"timestamp": ts, "close": close, "oracle005": y_dir,
                  "p_lgbm": p_lgbm, "p_vote": p_vote, **states}).to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}", flush=True)

    if selected is None:
        print("no ensemble cell cleared eligibility")
        return 0
    widx = windows["week"]
    h_ts = ts.to_numpy()[widx]
    show = [selected, "lgbm_jm_lam1", "vote_only_jm_lam1"]
    onamed = np.where(y_dir == 1, 2, np.where(y_dir == -1, 0, 1)).astype(np.int8)
    fig, axes = plt.subplots(2 + len(show), 1, figsize=(15, 8.4), sharex=True,
                             gridspec_kw={"height_ratios": [10] + [0.72] * (len(show) + 1), "hspace": 0.08})
    ax = axes[0]
    for s, e, stt in contiguous_runs(states[selected][widx]):
        seg = slice(s, min(e + 2, len(widx)))
        ax.plot(h_ts[seg], close[widx][seg], color=REGIME_COLORS[stt], linewidth=1.4)
    v0 = val[selected]
    verdict = "채택" if adopt else "미채택"
    ax.set_title(f"앙상블 라운드 [{verdict}] — 라인 = {selected}  VAL {sel_val}% / OOS {sel_oos}%  "
                 f"커버리지 {v0['coverage_pct']}%  중앙런 {int(v0['median_run_bars'])}bar   "
                 f"(기존 lgbm_jm_lam1 {INCUMBENT_VAL}/{INCUMBENT_OOS}) — 최근 {args.days}일",
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
    outp = OUT_DIR / "week_ensemble_theta005.png"
    fig.savefig(outp, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
