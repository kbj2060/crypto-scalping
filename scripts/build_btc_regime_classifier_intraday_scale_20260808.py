"""Intraday-SCALE rebuild of the regime classifiers (2026-08-08, after the oracle-scale diagnosis).

diagnose_zigzag_oracle_scale_20260808.py showed the 4% oracle has ZERO turning points inside a
7-day window (median wave 692 bars / 7.4% amplitude), so every week chart was being graded
against a single label, and every classifier's headline agreement was inflated by the coarse
definition (JM 69.9% at theta=4% collapses to 48.5% at theta=0.5%).  The eye-matching scale for
a 7-day 5m chart is theta ~= 1.5-2% (7 turning points in the week).

This script rebuilds the three families that survived, retuned to that scale, and grades them
against a MULTI-SCALE oracle panel instead of a single threshold:
  czz15   causal 1.5% directional-change direction (scale-matched zigzag)
  dc15    DC-indicator HMM built on 1.5% DC events (Chen & Tsang features, same HMM class)
  jm_it   Jump Model k3 with intraday halflives {12, 48, 144} and lambda swept {4, 8, 16}
          (the 4%-scale version used {72, 288, 864} and lambda 32 -- a multi-day configuration)
  qcml_it QCML observables with a 72-bar window instead of 288
Old 4%-scale states are carried through for side-by-side comparison.
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
import build_btc_regime_classifier_zoo_20260808 as zoo_mod  # noqa: E402
from build_btc_regime_classifier_zoo_20260808 import (  # noqa: E402
    SEED, dc_indicator_features, name_states, build_qcml, REGIME_COLORS, C_BULL, C_BEAR, C_CHOP, INK,
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
OUT_PARQUET = ROOT / "data/research/btc_regime_classifier_intraday_20260808.parquet"
OUT_DIR = ROOT / "tmp/regime_classifier_zoo_20260808"
THETA_IT = 0.015
JM_HALFLIVES = [12, 48, 144]
JM_LAMBDAS = [4.0, 8.0, 16.0]
SCORE_SCALES = [0.01, 0.015, 0.02, 0.03]

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def jm_features_scaled(close: np.ndarray, halflives) -> np.ndarray:
    logc = np.log(close)
    lr = pd.Series(np.diff(logc, prepend=logc[0]))
    feats = []
    for hl in halflives:
        feats.append(lr.ewm(halflife=hl).mean().to_numpy())
        feats.append(np.sqrt((lr.clip(upper=0.0) ** 2).ewm(halflife=hl).mean().to_numpy()))
    return np.column_stack(feats)


def build_jm_intraday(close, r288, train_mask, lam):
    xj = jm_features_scaled(close, JM_HALFLIVES)
    valid = np.isfinite(xj).all(axis=1)
    sc = RobustScaler().fit(xj[train_mask & valid])
    z = np.zeros_like(xj)
    z[valid] = sc.transform(xj[valid])
    mu = fit_jm(z[train_mask & valid], 3, lam, SEED)
    st = np.full(len(close), 1, dtype=int)
    st[valid] = causal_decode(z[valid], mu, lam)
    return name_states(st, r288, train_mask)


def build_dc_scaled(close, r288, train_mask, theta):
    feats = dc_indicator_features(close, theta=theta)
    valid = np.isfinite(feats).all(axis=1) & (np.arange(len(close)) > 0)
    sc = RobustScaler().fit(feats[train_mask & valid])
    z = np.zeros_like(feats)
    z[valid] = sc.transform(feats[valid])
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
    return {"agree_by_scale": {f"{t:.3f}": agreement(named, oracles[t], idx) for t in SCORE_SCALES},
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

    states = {}
    cz = causal_zigzag(close, threshold=THETA_IT)
    states["czz15"] = np.where(cz == 1, 2, np.where(cz == -1, 0, 1)).astype(np.int8)
    print("built czz15", flush=True)
    states["dc15"] = build_dc_scaled(close, r288, train_mask, THETA_IT)
    print("built dc15", flush=True)
    for lam in JM_LAMBDAS:
        states[f"jm_it_lam{int(lam)}"] = build_jm_intraday(close, r288, train_mask, lam)
        print(f"built jm_it_lam{int(lam)}", flush=True)
    zoo_mod.QCML_WINDOW = 72
    states["qcml_it"], _obs = build_qcml(close, r288, train_mask)
    print("built qcml_it (window=72)", flush=True)
    # carry the 4%-scale versions for comparison
    for old in ("jm", "dc", "qcml", "czz4"):
        states[f"{old}_4pct"] = zoo[old].to_numpy().astype(np.int8)

    windows = {
        "week": np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=args.days)).to_numpy()),
        "full": np.arange(len(close)),
        "val_2025Q4": np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()),
        "oos_2026Q1": np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()),
    }
    report = {w: {k: summarize(v, oracles, idx) for k, v in states.items()} for w, idx in windows.items()}
    (OUT_DIR / "intraday_scale_scorecard.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for w in ("full", "val_2025Q4", "oos_2026Q1"):
        print(f"=== {w}", flush=True)
        for k, v in report[w].items():
            print(f"  {k:16} agree {v['agree_by_scale']}  cov {v['coverage_pct']:5}  "
                  f"run {v['median_run_bars']:7}  flips {v['n_flips']}", flush=True)

    widx = windows["week"]
    h_ts = ts.to_numpy()[widx]
    show = ["czz15", "dc15", "jm_it_lam8", "qcml_it", "jm_4pct", "dc_4pct"]
    label = {"czz15": "causal zigzag 1.5%", "dc15": "DC 지표 HMM (1.5%)",
             "jm_it_lam8": "Jump Model 인트라데이 (λ8)", "qcml_it": "QCML (72바 창)",
             "jm_4pct": "(이전) JM 4% 스케일", "dc_4pct": "(이전) DC 4% 스케일"}
    ora = np.where(oracles[0.015] == 1, 2, np.where(oracles[0.015] == -1, 0, 1))
    fig, axes = plt.subplots(2 + len(show), 1, figsize=(15, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [10] + [0.75] * (len(show) + 1), "hspace": 0.08})
    ax = axes[0]
    for s, e, stt in contiguous_runs(states["czz15"][widx]):
        seg = slice(s, min(e + 2, len(widx)))
        ax.plot(h_ts[seg], close[widx][seg], color=REGIME_COLORS[stt], linewidth=1.3)
    ax.set_title("스케일 정합 재구성 — 라인 색 = causal zigzag 1.5% (최근 7일)", loc="left", fontsize=13, color=INK)
    ax.grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in
                       (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
              loc="upper left", frameon=False, fontsize=9, ncol=3)
    strips = [(ora, "오라클 1.5% (사후)  ")] + [(states[k], label[k] + "  ") for k in show]
    for sax, (arr, lb) in zip(axes[1:], strips):
        for s, e, stt in contiguous_runs(arr[widx]):
            sax.axvspan(h_ts[s], h_ts[min(e + 1, len(widx) - 1)], color=REGIME_COLORS[stt], linewidth=0)
        sax.set_yticks([])
        sax.set_ylabel(lb, rotation=0, ha="right", va="center", fontsize=9, color=INK)
        for side in ("top", "right", "left", "bottom"):
            sax.spines[side].set_visible(False)
    out = OUT_DIR / "week_intraday_scale.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")

    pd.DataFrame({"timestamp": ts, "close": close, **states}).to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
