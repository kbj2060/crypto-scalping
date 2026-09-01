#!/usr/bin/env python3
"""8트리거 일치 구성의 **라벨 시각검증 20예시** -- 사용자 승인 관문.

## 왜

`feedback_visual_verification_chart_gate_explain_before_proceed_20260830`: 라벨/후보 정의가
바뀌면 사용자가 20예시를 직접 보고 승인해야 다음 단계로 간다. 이 저장소에서 사용자가 이 방식으로
실제 라벨 결함을 최소 4번 잡아냈다.

2026-09-01에 보여드린 20예시는 **"미포착 사건이 진짜인가"** 질문용이었고, **8트리거 풀 라벨
자체**는 검증된 적이 없다. 후보풀 정의(9트리거 -> 8트리거, local_extreme 제외)가 바뀌었으므로
새로 봐야 한다.

## 이 구성이 무엇인지

`local_extreme`을 **학습셋 구성에서도** 빼고 나머지 8개 증거신호로 학습·서빙을 모두 게이트한다.
실측 확인(research_eth_v_rebound_8trigger_matched_population_20260901.py):
  - held_up 인플레 1.89x(9트리거) -> **1.09x**(8트리거) = 얽힘 제거됨
  - 풀 크기 전체 봉의 8.40%, TRAIN 라벨행 15,244 (서브샘플 없이 전부 컨텍스트로)
  - VAL AUC 0.7551 / OOS 0.7654 (정직한 수치, OOS가 더 높음)
  - ⚠️경제성은 여전히 실패 (OOS 격자 정방향 12 / 뒤집기 27)

## 구성 (20예시, 4행 x 5열)

  1행: v_rebound **bottom** (양성) -- giveback 낮은쪽~높은쪽 고르게
  2행: v_rebound **top** (양성)
  3행: **chop** (음성) -- 음성이 정말 "반등 시도 자체가 없음"인지 확인용
  4행: **ambiguous** (학습 제외) -- ⚠️게이트가 트리거 기반이라 이 봉들도 **라이브에선 채점된다**.
       전체의 약 49%를 차지하고, 2026-09-01에 "모델이 여기서 방향성을 못 낸다"고 확인된 구간이다.
       무엇이 버려지는지 눈으로 봐야 한다.

각 패널: 점선세로=대표봉(라이브 신호 시점), 회색가로=그 봉의 극값, 노란음영=라벨 판정창
(FAST 6봉/FULL 12봉). 제목에 발동 트리거 이름을 적는다.

TRAIN+VAL(< 2026-01-01)에서만 샘플링 -- **OOS/HOLDOUT 미노출**.

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/render_eth_v_rebound_8trigger_label_20examples_20260901.py
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

S1 = ROOT / "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py"
_spec = importlib.util.spec_from_file_location("vreb_s1_render8", S1)
_s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s1)
_feas, _vs = _s1._feas, _s1._vs

ALL9 = _feas.ALL9
EIGHT = [t for t in ALL9 if t != "local_extreme"]
FAST_BARS, FULL_BARS = 6, 12
DEPLOYED = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12}
CTX_BARS = 70
SEED = 20260901
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")

OUT_PATH = ROOT / "data/research/eth_v_rebound_8trigger_matched_20260901/label_20examples.png"

# 이 dev 머신(WSL2)에는 리눅스 한글 폰트가 없다 -- 다른 렌더 스크립트들이 쓰는 윈도우 폰트를
# 그대로 재사용한다(render_eth_v_rebound_every_bar_uncovered_events_20examples_20260901.py:49).
for cand in ("/mnt/c/Windows/Fonts/malgun.ttf",
             "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if Path(cand).exists():
        fm.fontManager.addfont(cand)
        plt.rcParams["font.family"] = fm.FontProperties(fname=cand).get_name()
        print(f"[render8] font: {cand}", flush=True)
        break
else:
    print("[render8] ⚠️한글 폰트 없음 -- 제목이 깨진다", flush=True)
plt.rcParams["axes.unicode_minus"] = False


def draw_candles(ax, sub, event_pos, event_level, span):
    """render_eth_v_rebound_every_bar_uncovered_events_20examples_20260901.py에서 verbatim 재사용."""
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.1, zorder=3)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height,
                               facecolor=color, edgecolor=color, zorder=4))
        lows.append(bar["low"]); highs.append(bar["high"])
    ax.axhline(event_level, color="dimgray", linestyle="--", linewidth=1.1, zorder=1)
    ax.axvline(event_pos, color="dimgray", linestyle=":", linewidth=1.4, zorder=1)
    lo, hi = span
    if hi >= lo:
        ax.axvspan(lo - 0.45, hi + 0.45, color="#f4c95d", alpha=0.28, zorder=0)
    pad = (max(highs) - min(lows)) * 0.08 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def outcome_fields(sig, is_down):
    close, high, low = (sig[c].to_numpy() for c in ("close", "high", "low"))
    atr = sig["atr"].to_numpy()
    pre_atr = _vs.shifted_at(atr, -1)
    extreme = low if is_down else high
    if is_down:
        fast_move = _vs.fwd_window(close, 1, FAST_BARS, "max") - extreme
        peak = _vs.fwd_window(high, 1, FULL_BARS, "max")
    else:
        fast_move = extreme - _vs.fwd_window(close, 1, FAST_BARS, "min")
        peak = _vs.fwd_window(low, 1, FULL_BARS, "min")
    end_price = _vs.shifted_at(close, FULL_BARS)
    with np.errstate(invalid="ignore", divide="ignore"):
        fm_ = fast_move / pre_atr
        denom = (peak - extreme) if is_down else (extreme - peak)
        gb = np.where(np.abs(denom) >= 1e-12,
                      (peak - end_price) / denom if is_down else (end_price - peak) / denom, np.nan)
    return fm_, gb


def main() -> int:
    print("[render8] building frame + labels...", flush=True)
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", **DEPLOYED)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick", **DEPLOYED)

    ts = sig["timestamp"]
    in_range = (ts >= pd.Timestamp("2024-01-01", tz="UTC")) & (ts < VAL_END)
    n = len(sig)

    pools = {}
    for side, is_down, status in (("bottom", True, sb), ("top", False, st)):
        g8 = np.any([sig[f"{side}_{t}"].fillna(False).to_numpy() for t in EIGHT], axis=0)
        fm_, gb = outcome_fields(sig, is_down)
        names = {t: sig[f"{side}_{t}"].fillna(False).to_numpy() for t in EIGHT}
        for st_name in ("v_rebound", "chop", "ambiguous"):
            idx = np.flatnonzero(g8 & (status == st_name) & in_range.to_numpy()
                                 & (np.arange(n) > CTX_BARS) & (np.arange(n) < n - CTX_BARS))
            pools[(side, st_name)] = (idx, fm_, gb, names)

    rng = np.random.default_rng(SEED)

    def pick(side, st_name, k, spread_by_gb):
        idx, fm_, gb, names = pools[(side, st_name)]
        if len(idx) == 0:
            return []
        if spread_by_gb:
            g = gb[idx]
            ok = idx[np.isfinite(g)]
            if len(ok) >= k:
                qs = np.nanpercentile(gb[ok], np.linspace(5, 95, k))
                chosen = [ok[np.nanargmin(np.abs(gb[ok] - q))] for q in qs]
                return [(i, fm_, gb, names) for i in dict.fromkeys(chosen)][:k]
        take = rng.choice(idx, size=min(k, len(idx)), replace=False)
        return [(int(i), fm_, gb, names) for i in np.sort(take)]

    rows = [("v_rebound bottom (양성)", pick("bottom", "v_rebound", 5, True), "#fdf1f0"),
            ("v_rebound top (양성)", pick("top", "v_rebound", 5, True), "#fdf1f0"),
            ("chop (음성 -- 반등 시도 없음)", pick("bottom", "chop", 5, False), "#eef6ef"),
            ("ambiguous (학습 제외 ⚠️라이브에선 채점됨)", pick("bottom", "ambiguous", 5, False), "#f2f0fa")]

    fig, axes = plt.subplots(4, 5, figsize=(30, 22))
    fig.suptitle(
        "8트리거 일치 구성 라벨 검증 -- local_extreme을 학습셋 구성에서도 제외 (held_up 인플레 1.89x -> 1.09x)\n"
        "라벨식 현행 그대로(30분내 종가 1.5xATR AND 60분 giveback<=0.20). "
        "점선세로=대표봉(라이브 신호시점), 회색가로=그 봉 극값, 노란음영=라벨 판정창(12봉)",
        fontsize=19, y=0.985)

    for r, (title, items, bg) in enumerate(rows):
        for cslot in range(5):
            ax = axes[r][cslot]
            ax.set_facecolor(bg)
            if cslot >= len(items):
                ax.axis("off"); continue
            i, fm_, gb, names = items[cslot]
            lo_i, hi_i = max(0, i - CTX_BARS), min(n, i + CTX_BARS + 1)
            sub = sig.iloc[lo_i:hi_i]
            is_down = "bottom" in title
            level = sig["low"].iloc[i] if is_down else sig["high"].iloc[i]
            span = (i - lo_i + 1, min(i + FULL_BARS, hi_i - 1) - lo_i)
            draw_candles(ax, sub, i - lo_i, level, span)
            fired = [t for t in EIGHT if names[t][i]]
            short = ",".join(t.replace("_", "")[:9] for t in fired[:3]) or "-"
            head = title if cslot == 0 else ("v_rebound" if r < 2 else ("chop" if r == 2 else "ambiguous"))
            ax.set_title(f"{head}\nfast={fm_[i]:.2f}x | gb={gb[i]:.3f} | {short}\n"
                         f"{sig['timestamp'].iloc[i]:%Y-%m-%d %H:%M}", fontsize=13)
            ax.tick_params(labelsize=10)
            ax.set_xticks([0, CTX_BARS, min(2 * CTX_BARS, hi_i - lo_i - 1)])
            ax.set_xticklabels([f"-{CTX_BARS}", "0", f"+{CTX_BARS}"])

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=145)
    print(f"[render8] saved: {OUT_PATH}", flush=True)
    print("[render8] 풀 크기(TRAIN+VAL, 8트리거 게이트):", flush=True)
    for (side, stn), (idx, *_) in sorted(pools.items()):
        print(f"    {side:6s} {stn:10s} {len(idx):>6,}건", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
