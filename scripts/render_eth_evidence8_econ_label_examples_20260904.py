#!/usr/bin/env python3
"""증거신호 **경제성 라벨** 시각 검증 -- 라벨 1(수익) 10개 · 라벨 0(손실) 10개 (2026-09-04).

오늘 8종 재라벨링에 쓴 라벨을 눈으로 확인한다:

    라벨 1  진입(다음 봉 시가) -> 트레일링 청산 -> 비용 10bp 차감 후 **순이익 > 0**
    라벨 0  같은 조건에서 **순손실**

각 패널이 보여주는 것:
  · 발동 봉(트리거)과 방향(bottom=롱, top=숏)
  · 진입가(o[i+1])와 초기 손절선(4.0xATR)
  · 무장선(1.0xATR) -- 여기 닿으면 트레일링 시작
  · 실제 청산 봉과 청산가, 그리고 net bp

⚠️차트는 크게 만든다(figsize 32x40, dpi 145, 폰트 18pt+) --
`feedback_large_chart_images` 참고. 작게 뽑으면 사용자가 판단 자체를 못 한다.
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

for cand in (Path("/mnt/c/Windows/Fonts/malgun.ttf"),
             Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")):
    if cand.exists():
        fm.fontManager.addfont(str(cand))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(cand)).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_pf = _load("pf_rend", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0, sim_exit = _pf.TIER0, _pf.sim_exit

SIGNALS = {"liquidity_sweep": 30, "taker_delta_z_climax": 24, "short_term_return_z": 12,
           "orthogonal_combo": 24, "smt_divergence": 72, "fib_extension_exhaustion": 20,
           "demarker_extreme": 8, "kalman_deviation_meanrev": 12}
SL, ARM, TRAIL = 4.0, 1.0, 0.1
GAP, COST_BP, SEED = 12, 10.0, 20260904
N_EACH, PAD = 10, 14
OUT_PATH = ROOT / "tmp/eth_evidence8_econ_label_examples_20260904.png"


def log(m): print(f"[render] {m}", flush=True)


def causal_first_fire(fire, gap):
    keep = np.zeros(len(fire), bool); last = -10**9
    for i in np.flatnonzero(fire):
        if i - last > gap:
            keep[i] = True
        last = i
    return keep


def main() -> int:
    log("프레임 빌드...")
    sig, feat, eth = _s1.build_sig()
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    if kl["timestamp"].dt.tz is not None:
        kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    o, h, l, c = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    ts = kl["timestamp"].to_numpy()
    n = len(kl)
    pos_of = {t: i for i, t in enumerate(ts)}
    atr_all = pd.to_numeric(sig["atr"], errors="coerce").to_numpy(float)
    sig_ts = sig["timestamp"]
    if sig_ts.dt.tz is not None:
        sig_ts = sig_ts.dt.tz_localize(None)
    sp = np.array([pos_of.get(np.datetime64(t), -1) for t in sig_ts.to_numpy()])
    atr_bar = np.full(n, np.nan)
    ok = sp >= 0
    atr_bar[sp[ok]] = atr_all[ok]

    rows = []
    for SIGNAL, HZ in SIGNALS.items():
        for side, is_down in ((f"bottom_{SIGNAL}", True), (f"top_{SIGNAL}", False)):
            if side not in sig.columns:
                continue
            fv = np.zeros(n, bool)
            fv[sp[ok]] = sig[side].fillna(False).to_numpy(bool)[ok]
            for i in np.flatnonzero(causal_first_fire(fv, GAP)):
                if i < PAD or i + 1 + HZ >= n or not np.isfinite(atr_bar[i]) or atr_bar[i] <= 0:
                    continue
                rows.append((SIGNAL, HZ, int(i), bool(is_down), float(atr_bar[i])))
    D = pd.DataFrame(rows, columns=["signal", "hz", "pos", "is_down", "atr"])
    log(f"  후보 {len(D):,}건")

    # 라벨 = 트레일링 청산 후 net_bp > 0 (청산 봉 오프셋도 함께 받는다)
    net = np.empty(len(D)); exi = np.empty(len(D), int)
    for HZ in sorted(D["hz"].unique()):
        m = (D["hz"] == HZ).to_numpy()
        ii = D.loc[m, "pos"].to_numpy()
        sg = np.where(D.loc[m, "is_down"].to_numpy(), 1.0, -1.0)
        entry = o[ii + 1]
        H = np.stack([h[i + 1:i + 1 + HZ] for i in ii])
        L = np.stack([l[i + 1:i + 1 + HZ] for i in ii])
        C = np.stack([c[i + 1:i + 1 + HZ] for i in ii])
        pn, ex = sim_exit(entry, D.loc[m, "atr"].to_numpy(float), sg, H, L, C, SL, ARM, TRAIL)
        net[m] = pn * 1e4 - COST_BP
        exi[m] = ex
    D["net_bp"] = net
    D["exit_off"] = exi
    D["label"] = (D["net_bp"] > 0).astype(int)
    log(f"  라벨 1 {int((D['label'] == 1).sum()):,} / 라벨 0 {int((D['label'] == 0).sum()):,} "
        f"(양성률 {D['label'].mean():.3f})")

    rng = np.random.default_rng(SEED)
    picks = []
    for lab in (1, 0):
        sub = D.loc[D["label"] == lab]
        # 신호를 골고루 섞어 뽑는다(한 신호로 쏠리면 라벨 특성이 아니라 신호 특성을 보게 된다)
        take = sub.sample(n=min(N_EACH, len(sub)), random_state=SEED + lab)
        picks.append(take)
    P = pd.concat(picks).reset_index(drop=True)

    plt.rcParams.update({"font.size": 19, "axes.titlesize": 21, "axes.labelsize": 18,
                         "xtick.labelsize": 15, "ytick.labelsize": 15})
    fig, axes = plt.subplots(5, 4, figsize=(34, 40), dpi=145)
    axes = axes.ravel()
    for ax, (_, r) in zip(axes, P.iterrows()):
        i, hz, sgn = int(r["pos"]), int(r["hz"]), (1.0 if r["is_down"] else -1.0)
        a_, off = float(r["atr"]), int(r["exit_off"])
        s0, s1 = max(0, i - PAD), min(n - 1, i + 1 + hz + 2)
        x = np.arange(s0, s1 + 1)
        for q in x:                                     # 캔들
            col = "#d64545" if c[q] >= o[q] else "#3a7bd5"
            ax.plot([q, q], [l[q], h[q]], color=col, lw=1.6, zorder=2)
            ax.plot([q, q], [o[q], c[q]], color=col, lw=6.5, solid_capstyle="butt", zorder=3)
        entry = o[i + 1]
        ex_pos = i + 1 + off
        ax.axvline(i, color="#111", ls="--", lw=2.2, zorder=4)          # 발동 봉
        ax.axhline(entry, color="#0a8f3c", lw=2.4, zorder=4)            # 진입가
        ax.axhline(entry - sgn * SL * a_, color="#c62828", ls=":", lw=2.2, zorder=4)
        ax.axhline(entry + sgn * ARM * a_, color="#f39c12", ls=":", lw=2.2, zorder=4)
        ax.scatter([ex_pos], [c[min(ex_pos, n - 1)]], s=190, marker="X",
                   color="#111", zorder=6)
        lab, side = int(r["label"]), ("롱" if r["is_down"] else "숏")
        ax.set_title(f"[라벨 {lab}] {r['signal'][:20]} · {side} · {r['net_bp']:+.1f}bp\n"
                     f"{pd.Timestamp(ts[i]).strftime('%Y-%m-%d %H:%M')} · "
                     f"보유 {off + 1}봉 / H={hz}",
                     color=("#0a6b2c" if lab == 1 else "#a01c1c"), fontweight="bold")
        ax.set_xlim(s0 - 0.7, s1 + 0.7)
        ax.grid(alpha=0.25)
        ax.set_xticks([])
    for ax in axes[len(P):]:
        ax.axis("off")
    fig.suptitle("증거신호 경제성 라벨 — 위 10개 = 라벨 1(비용 후 순이익), 아래 10개 = 라벨 0(순손실)\n"
                 f"진입=다음 봉 시가 · 청산=트레일링(SL {SL}·무장 {ARM}·트레일 {TRAIL} ×ATR) · 비용 {COST_BP:.0f}bp"
                 "   |   검은 파선=발동봉, 초록 실선=진입가, 빨강 점선=손절, 주황 점선=무장선, X=청산",
                 fontsize=25, fontweight="bold", y=0.996)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    log(f"⭐저장: {OUT_PATH}")
    log(f"  라벨1 평균 {P.loc[P['label']==1,'net_bp'].mean():+.1f}bp · "
        f"라벨0 평균 {P.loc[P['label']==0,'net_bp'].mean():+.1f}bp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
