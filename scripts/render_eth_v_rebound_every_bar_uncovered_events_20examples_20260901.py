#!/usr/bin/env python3
"""매 봉 스코어링 재설계 -- 트리거 미포착 사건의 육안검증 (프로젝트 관례상 필수 관문).

feedback_visual_verification_chart_gate_explain_before_proceed_20260830 원칙: 라벨/후보 정의가
바뀌면 사용자가 20예시를 직접 보고 승인해야 다음 단계(피쳐빌드/학습)로 갈 수 있다. 이 저장소에서
사용자가 이 방식으로 실제 라벨 결함을 최소 4번 잡아냈다.

## 이 차트가 검증하려는 것

9-6 사건감사: 현행 9트리거 게이트는 별개 사건의 55.9~70.9%만 포착하고, 나머지 **29~44%는 어떤
트리거도 발동하지 않은 채 라벨만 V자반등으로 붙은 사건**이다. 매 봉 스코어링 설계의 recall 이득은
전적으로 이 미포착 사건들이 "진짜"라는 전제 위에 서 있다 -- 그런데 이 population은 **한 번도 육안
검증된 적이 없다**(2026-08-31 21예시/보충 10예시는 전부 트리거 발동봉이었다). fast_mult 중앙값이
포착(2.42~2.51)과 미포착(2.28~2.31)이 거의 같다는 숫자상 근거는 있으나, 숫자가 같아도 모양이
쓰레기일 수 있다(예: 추세 중간 노이즈가 우연히 산술을 충족).

## 구성 (20예시, 4행 x 5열)

  1행: 미포착 bottom, giveback 하위(깨끗한 V 쪽)
  2행: 미포착 bottom, giveback 상위(경계선 쪽)
  3행: 미포착 top
  4행: **포착됨(대조군)** -- 이미 2026-08-31에 승인된 population, 직접 비교용

대표봉은 사건의 **첫 봉**(라이브에서 실제로 신호가 뜰 시점). 사건 전체 구간은 음영으로 표시해
라벨이 몇 봉 지속됐는지 보이게 한다.

TRAIN+VAL(< 2026-01-01)에서만 샘플링 -- OOS/HOLDOUT 미노출.

draw_candles()는 render_eth_5m_v_rebound_multitrigger_20examples_20260831.py에서 verbatim 재사용.

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/render_eth_v_rebound_every_bar_uncovered_events_20examples_20260901.py
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

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("/home/kbj20/crypto-scalping")
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

VARIANT_SCRIPT = ROOT / "scripts/research_eth_v_rebound_label_redesign_variant_screen_20260901.py"
_vspec = importlib.util.spec_from_file_location("label_variants_render_20260901", VARIANT_SCRIPT)
_vs = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(_vs)

AUDIT_SCRIPT = ROOT / "scripts/research_eth_v_rebound_every_bar_label_event_audit_20260901.py"
_aspec = importlib.util.spec_from_file_location("event_audit_render_20260901", AUDIT_SCRIPT)
_audit = importlib.util.module_from_spec(_aspec)
_aspec.loader.exec_module(_audit)

OUT_PATH = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/every_bar_uncovered_events_20examples.png"
WINDOW_BARS = 14
GAP = 12
ALL9 = _vs.ALL9
FAST_BARS = _vs.FAST_BARS
FULL_BARS = _vs.FULL_BARS


def draw_candles(ax, sub: pd.DataFrame, event_pos: int, event_level: float, span: tuple[int, int]) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.1, zorder=3)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=4))
        lows.append(bar["low"]); highs.append(bar["high"])
    ax.axhline(event_level, color="dimgray", linestyle="--", linewidth=1.1, zorder=1)
    ax.axvline(event_pos, color="dimgray", linestyle=":", linewidth=1.4, zorder=1)
    lo, hi = span
    if hi >= lo:
        ax.axvspan(lo - 0.45, hi + 0.45, color="#f4c95d", alpha=0.28, zorder=0)
    pad = (max(highs) - min(lows)) * 0.08 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def outcome_fields(sig: pd.DataFrame, is_down: bool) -> tuple[np.ndarray, np.ndarray]:
    """fast_mult / giveback per bar, mirroring label_side() exactly."""
    close = sig["close"].to_numpy()
    high = sig["high"].to_numpy()
    low = sig["low"].to_numpy()
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
        fast_mult = fast_move / pre_atr
        denom = (peak - extreme) if is_down else (extreme - peak)
        giveback = np.where(np.abs(denom) >= 1e-12,
                            (peak - end_price) / denom if is_down else (end_price - peak) / denom,
                            np.nan)
    return fast_mult, giveback


def collect_events(sig: pd.DataFrame):
    """Returns list of dicts: one per V_REBOUND event, with covered/uncovered flag."""
    out = []
    for side, is_down in (("bottom", True), ("top", False)):
        st = _vs.label_variant(sig, is_down=is_down, anchor_mode="wick", shift=0)
        fm_arr, gb_arr = outcome_fields(sig, is_down)
        trig = np.any([sig[f"{side}_{nm}"].fillna(False).to_numpy() for nm in ALL9], axis=0)
        v_idx = np.flatnonzero(st == "v_rebound")
        for ev in _audit.cluster_events(v_idx, GAP):
            first = int(ev[0])
            out.append({
                "side": side, "is_down": is_down,
                "first_idx": first, "last_idx": int(ev[-1]), "n_bars": len(ev),
                "covered": bool(trig[ev].any()),
                "fast_mult": float(fm_arr[first]), "giveback": float(gb_arr[first]),
                "n_trig_bars": int(trig[ev].sum()),
            })
    return out


def pick(pool: list[dict], n: int, pcts: list[float]) -> list[dict]:
    pool = sorted([e for e in pool if np.isfinite(e["giveback"])], key=lambda e: e["giveback"])
    if len(pool) <= n:
        return pool
    idxs = sorted({min(int(p * (len(pool) - 1)), len(pool) - 1) for p in pcts})
    return [pool[i] for i in idxs][:n]


def main() -> int:
    sig = _vs.build_base()
    frame = sig[["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)

    print("[render] collecting V_REBOUND events (GAP=12, TRAIN+VAL)...")
    events = collect_events(sig)
    unc_b = [e for e in events if not e["covered"] and e["side"] == "bottom"]
    unc_t = [e for e in events if not e["covered"] and e["side"] == "top"]
    cov = [e for e in events if e["covered"]]
    print(f"[render] 미포착 bottom {len(unc_b)} / 미포착 top {len(unc_t)} / 포착 {len(cov)}")

    rows = [
        ("미포착 bottom — giveback 하위(깨끗한 V쪽)", pick(unc_b, 5, [0.02, 0.10, 0.20, 0.30, 0.40])),
        ("미포착 bottom — giveback 상위(경계선쪽)", pick(unc_b, 5, [0.55, 0.68, 0.80, 0.90, 0.98])),
        ("미포착 top — 전 구간", pick(unc_t, 5, [0.05, 0.28, 0.50, 0.75, 0.95])),
        ("⭐대조군: 트리거 포착 사건 (이미 승인된 population)", pick(cov, 5, [0.05, 0.28, 0.50, 0.75, 0.95])),
    ]

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(4, 5, figsize=(32, 6.4 * 4), dpi=145)
    fig.suptitle(
        "매 봉 스코어링 재설계 육안검증 — 트리거가 한 번도 발동 안 한 'V자반등' 사건이 진짜인가?\n"
        "라벨식은 현행 v7b 그대로(30분내 종가 1.5xATR 도달 AND 60분 giveback<=0.20), 바뀐 건 "
        "'트리거 게이트 없이 전 봉을 채점한다'는 것뿐. 노란음영=라벨이 V자반등으로 붙은 구간, "
        "점선세로=사건 첫봉(라이브 신호시점), 회색가로=그 봉의 극값",
        fontsize=18, y=0.997,
    )

    for r, (row_title, picks) in enumerate(rows):
        for c in range(5):
            ax = axes[r][c]
            if c >= len(picks):
                ax.axis("off")
                continue
            e = picks[c]
            idx = e["first_idx"]
            lo_i = max(0, idx - WINDOW_BARS)
            hi_i = min(len(frame) - 1, idx + WINDOW_BARS)
            sub = frame.iloc[lo_i:hi_i + 1].reset_index(drop=True)
            evpos = idx - lo_i
            level = float(frame["low"].iloc[idx]) if e["is_down"] else float(frame["high"].iloc[idx])
            span = (e["first_idx"] - lo_i, min(e["last_idx"], hi_i) - lo_i)
            draw_candles(ax, sub, evpos, level, span)
            ax.set_facecolor("#eef8f0" if e["covered"] else "#fdf3ef")
            ticks = list(range(0, len(sub), 4))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{(t - evpos) * 5:+d}" for t in ticks], fontsize=10)
            ax.tick_params(axis="y", labelsize=10)
            dir_kr = "상승반등(bottom)" if e["is_down"] else "하락반전(top)"
            head = row_title if c == 0 else ("포착" if e["covered"] else "미포착")
            ax.set_title(
                f"{head}\n{dir_kr} | fast={e['fast_mult']:.2f}x | giveback={e['giveback']:.3f}\n"
                f"사건길이 {e['n_bars']}봉 | {frame['timestamp'].iloc[idx]:%Y-%m-%d %H:%M}",
                fontsize=11,
            )
            ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.972))
    fig.savefig(OUT_PATH)
    print(f"[render] saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
