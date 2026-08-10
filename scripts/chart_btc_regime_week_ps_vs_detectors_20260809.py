"""One-week chart: the Pagan-Sossounov regime LABEL against the causal detectors (2026-08-09).

Line colour = the PS label (P_min=48 bars, window 96, min_cycle 192, censor 2%), which is the
label the 2026-08-08 arc settled on: description-defined (window extrema + duration constraints),
not profit-defined.

Strips, top to bottom:
  PS label              RETROSPECTIVE — the target. Uses future bars by design.
  czz0.5 / czz1.2       causal zigzag, NO learning — the definitional baselines any detector must beat
  LGBM+jump lam2/lam16  the G3b nowcaster decoded at two persistence settings, from cached probabilities
  frozen theta=0.5%     the project's incumbent stability-first detector, for scale reference

Read the strips against each other, not against the price: the point of the figure is how much of
the label's block structure each CAUSAL detector reproduces, and at what flicker cost.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from chart_btc_jm_regime_verification_20260808 import (  # noqa: E402
    C_BEAR, C_BULL, C_CHOP, INK, REGIME_COLORS, causal_zigzag,
)
from refine_btc_regime_classifier_theta005_20260808 import (  # noqa: E402
    PANEL_PATH, jump_decode_proba, to_named,
)
from stage0_btc_regime_label_design_20260808 import BEAR, BULL, CHOP  # noqa: E402
from stage0e_btc_regime_label_pagan_sossounov_20260808 import ps_label, ps_pivots  # noqa: E402
from test_statistical_jump_model_regimes_20260808 import contiguous_runs  # noqa: E402

PROB_DIR = ROOT / "tmp/btc_regime_ps_g3b_20260808"
FROZEN = ROOT / "data/research/btc_regime_theta005_zigzagonly_20260808.parquet"
PS_P, PS_A = 48, 0.02

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def named(st3: np.ndarray) -> np.ndarray:
    return np.where(st3 == BULL, 2, np.where(st3 == BEAR, 0, 1)).astype(np.int8)


def stats(st3: np.ndarray, idx: np.ndarray, ref: np.ndarray) -> dict:
    runs = [e - s + 1 for s, e, _ in contiguous_runs(named(st3)[idx])]
    d = np.where(st3 == BULL, 1, np.where(st3 == BEAR, -1, 0))
    r = np.where(ref == BULL, 1, np.where(ref == BEAR, -1, 0))
    m = idx[(d[idx] != 0) & (r[idx] != 0)]
    return {"run_min": round(float(np.median(runs)) * 5, 1) if runs else None,
            "flips": max(len(runs) - 1, 0),
            "agree": round(float(np.mean(d[m] == r[m])) * 100, 1) if len(m) >= 20 else None,
            "cov": round(float((d[idx] != 0).mean()) * 100, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", default="tmp/btc_regime_label_design_20260808/week_ps_vs_detectors.png")
    args = ap.parse_args()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, low_memory=False, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    ts, close = panel["timestamp"], panel["close"].to_numpy(dtype=np.float64)

    ps = ps_label(close, ps_pivots(close, 2 * PS_P, PS_P, 4 * PS_P, PS_A), net_gate=True)
    strips: list[tuple[np.ndarray, str]] = [(ps, "PS 라벨 (사후, 타깃)")]
    for th in (0.005, 0.012):
        cz = causal_zigzag(close, threshold=th)
        strips.append((np.where(cz == 1, BULL, np.where(cz == -1, BEAR, CHOP)).astype(np.int8),
                       f"czz{th*100:g}%  (무학습 인과)"))
    for fs in ("S_wide7", "S2_fine5"):
        p = PROB_DIR / f"probs_{fs}.npy"
        if not p.exists():
            continue
        pr = np.load(p)
        for lam in (2.0, 16.0):
            nm = to_named(jump_decode_proba(pr, lam))
            st = np.where(nm == 2, BULL, np.where(nm == 0, BEAR, CHOP)).astype(np.int8)
            strips.append((st, f"LGBM {fs} λ={lam:g}"))
        break
    if FROZEN.exists():
        fz = pd.read_parquet(FROZEN)[["timestamp", "zigzagonly_final"]]
        fz["timestamp"] = pd.to_datetime(fz["timestamp"])
        mg = panel[["timestamp"]].merge(fz, on="timestamp", how="left")
        v = mg["zigzagonly_final"].to_numpy()
        strips.append((np.where(v == 2, BULL, np.where(v == 0, BEAR, CHOP)).astype(np.int8),
                       "frozen θ=0.5% (참조)"))

    start = ts.iloc[-1] - pd.Timedelta(days=args.days)
    idx = np.flatnonzero((ts >= start).to_numpy())
    h_ts = ts.to_numpy()[idx]
    rep = {lb: stats(st, idx, ps) for st, lb in strips}
    print(json.dumps({"window": [str(start), str(ts.iloc[-1])], "bars": int(len(idx)),
                      "per_strip": rep}, indent=2, ensure_ascii=False), flush=True)

    fig, axes = plt.subplots(1 + len(strips), 1, figsize=(16, 9.2), sharex=True,
                             gridspec_kw={"height_ratios": [10] + [0.72] * len(strips), "hspace": 0.07})
    ax = axes[0]
    for s, e, stt in contiguous_runs(named(ps)[idx]):
        seg = slice(s, min(e + 2, len(idx)))
        ax.plot(h_ts[seg], close[idx][seg], color=REGIME_COLORS[stt], linewidth=1.3)
    ax.set_title(
        f"BTC 5m — 라인 색 = Pagan-Sossounov 라벨 (최소국면 {PS_P}봉 / 윈도우 {2*PS_P} / 최소사이클 {4*PS_P} / censor {PS_A*100:g}%)"
        f" — 최근 {args.days}일\n"
        "스트립은 서로 비교해서 읽으십시오: 각 인과 탐지기가 라벨의 블록 구조를 얼마나 재현하는지, 그 대가로 얼마나 깜빡이는지",
        loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in
                       (("bull", C_BULL), ("chop", C_CHOP), ("bear", C_BEAR))],
              loc="upper left", frameon=False, fontsize=9, ncol=3)

    for sax, (st, lb) in zip(axes[1:], strips):
        for s, e, stt in contiguous_runs(named(st)[idx]):
            sax.axvspan(h_ts[s], h_ts[min(e + 1, len(idx) - 1)], color=REGIME_COLORS[stt], linewidth=0)
        r = rep[lb]
        sax.set_yticks([])
        sax.set_ylabel(f"{lb}\n{r['flips']}전환 · 중앙런 {r['run_min']}분  ", rotation=0,
                       ha="right", va="center", fontsize=8.5, color=INK)
        for side in ("top", "right", "left", "bottom"):
            sax.spines[side].set_visible(False)

    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
