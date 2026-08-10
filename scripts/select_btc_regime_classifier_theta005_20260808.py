"""Select the project's regime classifier at the theta=0.5% wave scale (2026-08-08).

Decision by the user after the oracle-scale diagnosis: the 0.5% zigzag oracle is the reference
definition of bull/bear to classify against.  This script builds the scale-matched candidates,
selects one under a rule fixed BEFORE any result was read, and freezes it.

PRE-REGISTERED SELECTION RULE (written before the first run):
  primary metric  = agreement with the theta=0.005 oracle on VAL (2025-09-01..2025-12-31)
  eligibility     = coverage >= 50%  AND  median run >= 8 bars
                    (the 0.005 oracle's own median wave is 16 bars, so 8 bars = half a wave;
                     without this floor the metric selects bar-to-bar confetti, which is how the
                     CNN scored 72% full-period agreement while being unusable)
  tie-break       = higher coverage, then longer median run
  OOS (2026-01-01..2026-03-31) is a single confirmation read on the selected candidate only;
  it does not re-open selection.  Secondary scales 0.01/0.015 are reported for every candidate
  per the multi-scale scorecard rule adopted in the same session.

CANDIDATES (all train-only fit, causal decode, 3 states named by train-mean trailing 24h return):
  czz05        causal 0.5% directional-change direction (no fitting; the definitional baseline)
  dc05         DC-indicator HMM on 0.5% DC events (TMV/logT/R/OSV/dir -> 3-state sticky HMM)
  jm_u_lam{1,2,4}  Jump Model k3 with ultra-short halflives {3,12,36}
  hmm_s        the retired 3-state sticky Gaussian HMM, SCALE-MATCHED: trailing 3h log return +
               3h realized vol instead of the 24h pair.  Included because the scale diagnosis
               showed the old HMM matched or beat JM at every threshold -- its documented failure
               was flicker, not accuracy, so it deserves a fair short-scale entry.
EXCLUDED with reason: the 1D-CNN (VAL agreement 49.7% = coin flip, 1-bar runs) and the QCML
observables (46-52% at every scale and every window length tried) both failed as directional
classifiers earlier in this session; re-running them at 0.5% would only add multiple-comparison
surface.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_btc_regime_classifier_zoo_20260808 import (  # noqa: E402
    SEED, dc_indicator_features, name_states, REGIME_COLORS, C_BULL, C_BEAR, C_CHOP, INK,
)
from build_btc_regime_classifier_intraday_scale_20260808 import (  # noqa: E402
    build_dc_scaled, jm_features_scaled,
)
from chart_btc_jm_regime_verification_20260808 import causal_zigzag  # noqa: E402
from retrain_clean_regime_hmm_20260517 import GaussianStateModel  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import (  # noqa: E402
    causal_decode, contiguous_runs, fit_jm, zigzag_oracle,
)
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

ZOO_PATH = ROOT / "data/research/btc_regime_classifier_zoo_20260808.parquet"
OUT_PARQUET = ROOT / "data/research/btc_regime_theta005_20260808.parquet"
OUT_DIR = ROOT / "tmp/regime_theta005_20260808"
THETA = 0.005
JM_HALFLIVES = [3, 12, 36]
JM_LAMBDAS = [1.0, 2.0, 4.0]
HMM_SHORT_BARS = 36
SCORE_SCALES = [0.005, 0.010, 0.015]
MIN_COVERAGE, MIN_MEDIAN_RUN = 50.0, 8.0

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def build_jm(close, r288, train_mask, lam):
    xj = jm_features_scaled(close, JM_HALFLIVES)
    valid = np.isfinite(xj).all(axis=1)
    sc = RobustScaler().fit(xj[train_mask & valid])
    z = np.zeros_like(xj)
    z[valid] = sc.transform(xj[valid])
    mu = fit_jm(z[train_mask & valid], 3, lam, SEED)
    st = np.full(len(close), 1, dtype=int)
    st[valid] = causal_decode(z[valid], mu, lam)
    return name_states(st, r288, train_mask)


def build_hmm_short(close, r288, train_mask, bars=HMM_SHORT_BARS):
    logc = np.log(close)
    r = np.full(len(close), np.nan)
    r[bars:] = logc[bars:] - logc[:-bars]
    lr = np.diff(logc, prepend=logc[0])
    vol = pd.Series(lr).rolling(bars, min_periods=bars).std().to_numpy()
    f = np.column_stack([r, vol])
    valid = np.isfinite(f).all(axis=1)
    sc = RobustScaler().fit(f[train_mask & valid])
    z = np.zeros_like(f)
    z[valid] = sc.transform(f[valid])
    hmm = GaussianStateModel(n_states=3, n_iter=50, seed=SEED)
    hmm.fit(z[train_mask & valid])
    st = np.full(len(close), -1, dtype=int)
    st[valid] = np.nanargmax(hmm.filter_proba(z[valid]), axis=1)
    st[~valid] = st[valid][0] if valid.any() else 0
    return name_states(st, r288, train_mask)


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

    zoo = pd.read_parquet(ZOO_PATH)
    ts = pd.to_datetime(zoo["timestamp"])
    close = zoo["close"].to_numpy(dtype=np.float64)
    train_mask = (ts <= TRAIN_END).to_numpy()
    r288 = np.full(len(close), np.nan)
    r288[288:] = np.log(close[288:] / close[:-288])
    oracles = {t: zigzag_oracle(close, threshold=t)[0] for t in SCORE_SCALES}

    states: dict[str, np.ndarray] = {}
    cz = causal_zigzag(close, threshold=THETA)
    states["czz05"] = np.where(cz == 1, 2, np.where(cz == -1, 0, 1)).astype(np.int8)
    print("built czz05", flush=True)
    states["dc05"] = build_dc_scaled(close, r288, train_mask, THETA)
    print("built dc05", flush=True)
    for lam in JM_LAMBDAS:
        states[f"jm_u_lam{int(lam)}"] = build_jm(close, r288, train_mask, lam)
        print(f"built jm_u_lam{int(lam)}", flush=True)
    states["hmm_s"] = build_hmm_short(close, r288, train_mask)
    print("built hmm_s", flush=True)

    windows = {
        "val_2025Q4": np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()),
        "oos_2026Q1": np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()),
        "full": np.arange(len(close)),
        "week": np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=args.days)).to_numpy()),
    }
    report = {w: {k: summarize(v, oracles, idx) for k, v in states.items()} for w, idx in windows.items()}

    val = report["val_2025Q4"]
    eligible = {k: v for k, v in val.items()
                if (v["coverage_pct"] >= MIN_COVERAGE and v["median_run_bars"] >= MIN_MEDIAN_RUN
                    and v["agree"]["0.005"] is not None)}
    print(json.dumps({"eligible": list(eligible)}, indent=2), flush=True)
    if not eligible:
        selected = None
    else:
        selected = max(eligible,
                       key=lambda k: (eligible[k]["agree"]["0.005"], eligible[k]["coverage_pct"],
                                      eligible[k]["median_run_bars"]))
    out = {"theta": THETA,
           "preregistered_rule": {"primary": "VAL agreement vs theta=0.005 oracle",
                                  "eligibility": f"coverage>={MIN_COVERAGE}% and median_run>={MIN_MEDIAN_RUN} bars",
                                  "tiebreak": "coverage, then median run",
                                  "oos": "single confirmation read on the selected candidate only"},
           "report": report, "eligible": list(eligible), "selected": selected,
           "selected_val": None if selected is None else val[selected],
           "selected_oos": None if selected is None else report["oos_2026Q1"][selected]}
    (OUT_DIR / "selection.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for w in ("val_2025Q4", "oos_2026Q1", "full"):
        print(f"=== {w}", flush=True)
        for k, v in report[w].items():
            print(f"  {k:12} agree {v['agree']}  cov {v['coverage_pct']:5}  "
                  f"run {v['median_run_bars']:7}  flips {v['n_flips']}", flush=True)
    print(json.dumps({"SELECTED": selected, "val": out["selected_val"], "oos": out["selected_oos"]},
                     indent=2), flush=True)

    pd.DataFrame({"timestamp": ts, "close": close,
                  "oracle005": oracles[0.005], **states}).to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}", flush=True)

    if selected is None:
        print("no candidate cleared the pre-registered eligibility bar -- nothing frozen")
        return 0

    widx = windows["week"]
    h_ts = ts.to_numpy()[widx]
    onamed = np.where(oracles[0.005] == 1, 2, np.where(oracles[0.005] == -1, 0, 1)).astype(np.int8)
    others = [k for k in states if k != selected]
    fig, axes = plt.subplots(2 + len(others), 1, figsize=(15, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [10] + [0.72] * (len(others) + 1), "hspace": 0.08})
    ax = axes[0]
    for s, e, stt in contiguous_runs(states[selected][widx]):
        seg = slice(s, min(e + 2, len(widx)))
        ax.plot(h_ts[seg], close[widx][seg], color=REGIME_COLORS[stt], linewidth=1.4)
    sv, so = out["selected_val"], out["selected_oos"]
    ax.set_title(f"선정 분류기 = {selected}  (θ=0.5% 오라클 기준)   "
                 f"VAL {sv['agree']['0.005']}% / OOS {so['agree']['0.005']}%   "
                 f"커버리지 {sv['coverage_pct']}%  중앙런 {int(sv['median_run_bars'])}bar — 최근 {args.days}일",
                 loc="left", fontsize=12.5, color=INK)
    ax.grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in
                       (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
              loc="upper left", frameon=False, fontsize=9, ncol=3)
    strips = [(onamed, "오라클 0.5% (사후)  ")] + [(states[k], k + "  ") for k in others]
    for sax, (arr, lb) in zip(axes[1:], strips):
        for s, e, stt in contiguous_runs(arr[widx]):
            sax.axvspan(h_ts[s], h_ts[min(e + 1, len(widx) - 1)], color=REGIME_COLORS[stt], linewidth=0)
        sax.set_yticks([])
        sax.set_ylabel(lb, rotation=0, ha="right", va="center", fontsize=9, color=INK)
        for side in ("top", "right", "left", "bottom"):
            sax.spines[side].set_visible(False)
    outp = OUT_DIR / "week_selected_theta005.png"
    fig.savefig(outp, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
