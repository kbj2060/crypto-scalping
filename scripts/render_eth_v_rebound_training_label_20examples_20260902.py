#!/usr/bin/env python3
"""배포판이 **실제로 학습에 쓴 라벨**(매 봉 giveback, wick 앵커) 20예시 시각검증.

사용자 요청(2026-09-02): "학습할 때 쓴 라벨 차트를 한 이미지로 보여줘".

동결 컨텍스트(`tabpfn_train_context_frozen_every_bar_20260901.csv`, 18,000행)에서 실제로 뽑힌
행만 대상으로 한다 -- "학습에 쓴 라벨"이 정확히 그것이다(TRAIN 182,969행 전체가 아니라
그 중 무작위 추출된 18,000행이 모델이 본 전부).

구성: 1~2행 V자반등(label=1, fast_mult 백분위 5~95 층화) / 3행 횡보(label=0) /
      4행 **앵커 갭이 큰 V자반등**(진입 시점에 목표가 이미 소진된 사례 -- 2026-09-02 발견)

각 패널에 라벨 산술을 전부 그린다:
  · 회색 파선 = 앵커(`low[i]`/`high[i]`, 그 봉의 꼬리) -- **라벨의 기준점**
  · 초록 파선 = 목표(앵커 ± 1.5×ATR) -- 이걸 넘어야 fast leg 성공
  · ⭐주황 실선 = **진입가 `open[i+1]`** -- 백테스트/실거래가 실제로 잡는 가격
  · 주황 음영 = fast 창(6봉=30분), 회색 음영 = full 창(12봉=60분)
  · 파란 점 = full 창 정점(peak), 보라 점 = 종료가(close[i+12])

제목에 fast_mult / giveback / **소진비율**(진입가가 목표까지 얼마나 먹었는지)을 같이 찍는다.
소진비율 ≥ 1.0이면 **진입 시점에 이미 목표 도달** = 거래로는 먹을 게 없다는 뜻.

Run on the server via handoff, then pull the PNG.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_s1 = _load("s1_render", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_vs = _s1._vs

CTX_CSV = ROOT / "data/labels/eth_5m_v_rebound_every_bar_20260901/tabpfn_train_context_frozen_every_bar_20260901.csv"
DEPLOYED = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12}
FAST_BARS, FULL_BARS, ATR_MULT = _s1.FAST_BARS_FIXED, 12, 1.50
CTX_BARS, SEED = 70, 20260902
OUT_PATH = ROOT / "data/research/eth_v_rebound_training_label_20examples_20260902/label_20examples.png"

for cand in ("/mnt/c/Windows/Fonts/malgun.ttf",
             "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if Path(cand).exists():
        fm.fontManager.addfont(cand)
        plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
        print(f"[render] font: {cand}", flush=True)
        break
else:
    print("[render] ⚠️한글 폰트 없음 -- 제목이 깨진다", flush=True)
plt.rcParams["axes.unicode_minus"] = False


def draw(ax, sub, ev, anchor, target, entry, peak_v, end_v, label):
    lows, highs = [], []
    for i, (_, b) in enumerate(sub.iterrows()):
        col = "#2E86AB" if b["close"] >= b["open"] else "#C73E1D"
        ax.plot([i, i], [b["low"], b["high"]], color=col, linewidth=1.1, zorder=3)
        blo, bhi = sorted([b["open"], b["close"]])
        ax.add_patch(Rectangle((i - 0.32, blo), 0.64,
                               max(bhi - blo, (b["high"] - b["low"]) * 0.03),
                               facecolor=col, edgecolor=col, zorder=4))
        lows.append(b["low"]); highs.append(b["high"])
    ax.axvspan(ev + 0.55, ev + FAST_BARS + 0.45, color="#f4c95d", alpha=0.30, zorder=0)
    ax.axvspan(ev + FAST_BARS + 0.45, ev + FULL_BARS + 0.45, color="#bbbbbb", alpha=0.22, zorder=0)
    ax.axvline(ev, color="dimgray", linestyle=":", linewidth=1.5, zorder=1)
    ax.axhline(anchor, color="dimgray", linestyle="--", linewidth=1.2, zorder=2)
    ax.axhline(target, color="#2a9d3f", linestyle="--", linewidth=1.6, zorder=2)
    ax.axhline(entry, color="#e8730c", linestyle="-", linewidth=1.8, zorder=2)   # ⭐진입가
    if np.isfinite(peak_v):
        ax.plot([ev + FULL_BARS], [peak_v], "o", color="#1f77b4", ms=7, zorder=6)
    if np.isfinite(end_v):
        ax.plot([ev + FULL_BARS], [end_v], "o", color="#7b2cbf", ms=7, zorder=6)
    lo, hi = min(lows + [target, entry]), max(highs + [target, entry])
    pad = (hi - lo) * 0.08 or 1.0
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)
    ax.tick_params(labelsize=10)
    # 클래스가 한눈에 보이도록 패널 배경을 옅게 물들인다
    ax.set_facecolor("#eaf7ec" if label == 1 else "#fdeeee")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a9d3f" if label == 1 else "#c0392b")
        sp.set_linewidth(2.2)


def main() -> int:
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", **DEPLOYED)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick", **DEPLOYED)
    long = _s1.long_frame_for(sig, feat, sb, st)

    ctx = pd.read_csv(CTX_CSV)
    ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], utc=True)
    ctx = ctx[["timestamp", "is_downside", "label"]]
    print(f"[render] 동결 컨텍스트 {len(ctx):,}행 (라벨률 {ctx['label'].mean():.4f})", flush=True)

    long = long.merge(ctx.assign(in_ctx=True), on=["timestamp", "is_downside"],
                      how="inner", suffixes=("", "_c"))
    print(f"[render] 컨텍스트 매칭 {len(long):,}행", flush=True)

    close, high, low = (sig[c].to_numpy() for c in ("close", "high", "low"))
    op, atr = sig["open"].to_numpy(), sig["atr"].to_numpy()
    pre_atr = _vs.shifted_at(atr, -1)
    # ⚠️tz-aware Series의 .to_numpy()는 datetime64가 아니라 Timestamp 객체 배열을 준다 --
    # naive np.datetime64로 조회하면 하나도 안 맞아 조용히 0건이 된다(이 저장소 상습 함정,
    # v_rebound_open_issues 16절 말미에 기록됨). tz를 벗겨서 키를 만든다.
    ts_pos = {t: i for i, t in enumerate(sig["timestamp"].dt.tz_localize(None).to_numpy())}

    rows = []
    for _, r in long.iterrows():
        i = ts_pos.get(np.datetime64(r["timestamp"].tz_localize(None)))
        if i is None or i < CTX_BARS or i + FULL_BARS + 2 >= len(sig):
            continue
        dn = r["is_downside"] == 1
        anc = low[i] if dn else high[i]
        tgt = anc + (1 if dn else -1) * ATR_MULT * pre_atr[i]
        ent = op[i + 1]
        fastw = close[i + 1:i + 1 + FAST_BARS]
        fastm = ((fastw.max() - anc) if dn else (anc - fastw.min())) / pre_atr[i]
        pk = high[i + 1:i + 1 + FULL_BARS].max() if dn else low[i + 1:i + 1 + FULL_BARS].min()
        endv = close[i + FULL_BARS]
        den = (pk - anc) if dn else (anc - pk)
        gb = ((pk - endv) if dn else (endv - pk)) / den if abs(den) > 1e-12 else np.nan
        consumed = ((ent - anc) if dn else (anc - ent)) / (ATR_MULT * pre_atr[i])
        rows.append({"i": i, "dn": bool(dn), "label": float(r["label"]), "anchor": anc,
                     "target": tgt, "entry": ent, "fast_mult": fastm, "giveback": gb,
                     "consumed": consumed, "peak": pk, "end": endv,
                     "ts": r["timestamp"]})
    if not rows:
        raise RuntimeError("타임스탬프 매칭 0건 -- ts_pos 키 타입을 확인할 것")
    ex = pd.DataFrame(rows).dropna(subset=["fast_mult"])
    print(f"[render] 후보 {len(ex):,} (양성 {int((ex['label']==1).sum()):,} / "
          f"음성 {int((ex['label']==0).sum()):,})", flush=True)

    pos = ex.loc[ex["label"] == 1].sort_values("fast_mult").reset_index(drop=True)
    neg = ex.loc[ex["label"] == 0].sort_values("fast_mult").reset_index(drop=True)

    def strat(d, k=10):
        """fast_mult 백분위 5~95로 층화 -- 양쪽 클래스에 같은 규칙을 쓴다."""
        return d.iloc[[int(round(q / 100 * (len(d) - 1))) for q in np.linspace(5, 95, k)]]

    # 1~2행 = 라벨 1(V자반등) 10건 / 3~4행 = 라벨 0(횡보) 10건
    sel = pd.concat([strat(pos), strat(neg)], ignore_index=True)
    tag = ["V자반등"] * 10 + ["횡보"] * 10

    fig, axes = plt.subplots(4, 5, figsize=(32, 24))
    fig.suptitle(
        "배포판 학습 라벨 20예시 — 매 봉 giveback (wick 앵커, 1.5×ATR / giveback≤0.20 / 60분)\n"
        "■1~2행 = 라벨 1 (V자반등, 초록 배경)   ■3~4행 = 라벨 0 (횡보, 붉은 배경)   "
        "— 양쪽 다 fast_mult 백분위 5~95 층화\n"
        "라벨 1 조건: fast_mult≥1.5 (초록선 도달) AND giveback≤0.20    "
        "라벨 0 조건: fast_mult<1.0 (반등 시도 자체가 없음)    그 사이는 ambiguous로 학습 제외\n"
        "회색파선=앵커(그 봉 꼬리, 라벨 기준점)   초록파선=목표(앵커±1.5×ATR, 대시보드 익절선)   "
        "주황실선=진입가 open[i+1]   주황음영=fast 30분   회색음영=full 60분   "
        "파랑●=정점  보라●=종료가",
        fontsize=19, y=0.99)
    for k, ax in enumerate(axes.flat):
        r = sel.iloc[k]
        i = int(r["i"])
        sub = sig.iloc[i - CTX_BARS: i + FULL_BARS + 2][["open", "high", "low", "close"]]
        lb = int(r["label"])
        draw(ax, sub, CTX_BARS, r["anchor"], r["target"], r["entry"], r["peak"], r["end"], lb)
        side = "바닥" if r["dn"] else "천장"
        ax.set_title(f"[라벨 {lb}] {tag[k]} · {side} · {r['ts']:%Y-%m-%d %H:%M}\n"
                     f"fast_mult {r['fast_mult']:.2f}  giveback {r['giveback']:.2f}  "
                     f"소진 {r['consumed']*100:.0f}%",
                     fontsize=14, color="#1d6f2b" if lb == 1 else "#8e2420", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=145)
    print(f"[render] saved -> {OUT_PATH}", flush=True)
    print(f"[render] 소진비율 중앙값(양성 전체): {pos['consumed'].median()*100:.0f}%  "
          f"100%↑ 비율 {float((pos['consumed']>=1).mean())*100:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
