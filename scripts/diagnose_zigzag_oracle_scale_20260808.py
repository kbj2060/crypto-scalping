"""Is the 4% zigzag oracle the wrong SCALE for a 7-day 5m chart? (2026-08-08 diagnostic)

Trigger: on the last-7-day charts every classifier looked visually wrong, yet czz4 scored 100%
week agreement with 0 flips and a 2017-bar median run -- i.e. the whole week is ONE oracle wave,
so week-level "agreement" compares a single label against a single label and is vacuous.

Two things are tested here, both of which the earlier detector work missed:
  1. SCALE.  The oracle threshold sweep in sweep_btc_causal_zigzag_regime_20260808.py only went
     UP (4/6/8%).  Here we go DOWN (0.5/1/1.5/2/3/4%) and measure, per threshold, the wave count,
     median wave length and amplitude over the full panel AND the pivot count inside the last
     7 days -- the number that decides whether a week-long chart can be segmented at all.
  2. CIRCULARITY.  Every detector so far was scored against a 4% oracle while czz4 IS a 4% DC
     rule, so czz4 was graded against itself.  Here all classifiers are re-scored against every
     oracle scale, which separates "agrees with the 4% definition" from "tracks direction".

Output: a week chart with one oracle strip per threshold (so the eye can pick the scale that
matches what a human calls bull/bear on this window), plus a scale x classifier agreement table.
"""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from chart_btc_jm_regime_verification_20260808 import causal_zigzag  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs, zigzag_oracle  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    VAL_START, VAL_END, OOS_START, OOS_END,
)

ZOO_PATH = ROOT / "data/research/btc_regime_classifier_zoo_20260808.parquet"
OUT_DIR = ROOT / "tmp/regime_classifier_zoo_20260808"
THRESHOLDS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04]
CLASSIFIERS = ["jm", "dc", "cnn", "qcml"]
C_BULL, C_BEAR, C_CHOP = "#2563EB", "#D9542B", "#9AA0A6"
REGIME_COLORS = {0: C_BEAR, 1: C_CHOP, 2: C_BULL}
INK = "#1F2430"

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def agreement(det_named, oracle_dir, idx):
    det = np.where(det_named == 2, 1, np.where(det_named == 0, -1, 0))
    act = det[idx] != 0
    if not act.any():
        return None
    return round(float(np.mean(det[idx][act] == oracle_dir[idx][act])) * 100, 1)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zoo = pd.read_parquet(ZOO_PATH)
    ts = pd.to_datetime(zoo["timestamp"])
    close = zoo["close"].to_numpy(dtype=np.float64)
    week_idx = np.flatnonzero((ts >= ts.iloc[-1] - pd.Timedelta(days=7)).to_numpy())
    windows = {
        "full": np.arange(len(close)),
        "val_2025Q4": np.flatnonzero(((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()),
        "oos_2026Q1": np.flatnonzero(((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()),
        "week": week_idx,
    }

    scale_report = {}
    oracles, causals = {}, {}
    for thr in THRESHOLDS:
        odir, pivots = zigzag_oracle(close, threshold=thr)
        oracles[thr] = odir
        causals[thr] = causal_zigzag(close, threshold=thr)
        wave_lens = np.diff(pivots) if len(pivots) > 1 else np.array([])
        amps = np.array([abs(close[pivots[i + 1]] - close[pivots[i]]) / close[pivots[i]]
                         for i in range(len(pivots) - 1)]) if len(pivots) > 1 else np.array([])
        in_week = [p for p in pivots if week_idx[0] <= p <= week_idx[-1]]
        scale_report[f"{thr:.3f}"] = {
            "n_waves": int(max(len(pivots) - 1, 0)),
            "median_wave_bars": int(np.median(wave_lens)) if len(wave_lens) else None,
            "median_wave_amplitude_pct": round(float(np.median(amps)) * 100, 2) if len(amps) else None,
            "pivots_inside_last_7d": len(in_week),
            "causal_flips_week": int(len(contiguous_runs(causals[thr][week_idx])) - 1),
        }
        print(json.dumps({f"theta={thr}": scale_report[f"{thr:.3f}"]}), flush=True)

    # scale x classifier agreement (does any classifier track a FINER wave scale?)
    table = {}
    for wtag, idx in windows.items():
        table[wtag] = {}
        for thr in THRESHOLDS:
            row = {c: agreement(zoo[c].to_numpy().astype(np.int8), oracles[thr], idx) for c in CLASSIFIERS}
            row["czz4"] = agreement(zoo["czz4"].to_numpy().astype(np.int8), oracles[thr], idx)
            row["hmm_old"] = agreement(zoo["hmm_old"].to_numpy().astype(np.int8), oracles[thr], idx)
            cz = causals[thr]
            row[f"causal_zz@{thr}"] = agreement(np.where(cz == 1, 2, np.where(cz == -1, 0, 1)).astype(np.int8),
                                                oracles[thr], idx)
            table[wtag][f"oracle_{thr:.3f}"] = row
    (OUT_DIR / "oracle_scale_diagnosis.json").write_text(
        json.dumps({"scale_report": scale_report, "agreement_by_scale": table}, indent=2, ensure_ascii=False))
    print(json.dumps(table["full"], indent=2), flush=True)

    # week chart: one oracle strip per threshold
    h_ts = ts.to_numpy()[week_idx]
    fig, axes = plt.subplots(1 + len(THRESHOLDS), 1, figsize=(15, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [10] + [0.75] * len(THRESHOLDS), "hspace": 0.08})
    axes[0].plot(h_ts, close[week_idx], color=INK, linewidth=1.1)
    axes[0].set_title("이번 주 오라클은 어느 스케일에서 사람 눈과 맞는가 — 지그재그 임계값별 파동 분할 (최근 7일)",
                      loc="left", fontsize=13, color=INK)
    axes[0].grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)
    for side in ("top", "right"):
        axes[0].spines[side].set_visible(False)
    axes[0].legend(handles=[Patch(facecolor=c, label=l) for l, c in
                            (("상승 파동", C_BULL), ("하락 파동", C_BEAR))],
                   loc="upper left", frameon=False, fontsize=9, ncol=2)
    for ax, thr in zip(axes[1:], THRESHOLDS):
        named = np.where(oracles[thr] == 1, 2, np.where(oracles[thr] == -1, 0, 1))
        for s, e, stt in contiguous_runs(named[week_idx]):
            ax.axvspan(h_ts[s], h_ts[min(e + 1, len(week_idx) - 1)], color=REGIME_COLORS[stt], linewidth=0)
        ax.set_yticks([])
        n_piv = scale_report[f"{thr:.3f}"]["pivots_inside_last_7d"]
        ax.set_ylabel(f"오라클 {thr * 100:g}%  (주내 전환점 {n_piv}개)  ",
                      rotation=0, ha="right", va="center", fontsize=9, color=INK)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    out = OUT_DIR / "oracle_scale_week.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
