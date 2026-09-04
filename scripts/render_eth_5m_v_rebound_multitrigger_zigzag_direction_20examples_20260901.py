#!/usr/bin/env python3
"""ZDC(지그재그 방향확인) 라벨 육안검증 -- hit(label=1) 10건 + miss(label=0) 10건, 한 이미지.

Step A(research_eth_v_rebound_multitrigger_zigzag_direction_raw_lift_check_20260901.py)가
raw-lift REJECTED(VAL/OOS 4칸 전부 lift<1.0)로 판정한 뒤, 그 라벨 정의 자체가 눈으로 봐도
합리적인지 별도로 확인하기 위한 진단용 차트. Step A의 zdc_first_pivot()을 그대로 재사용
(재구현 없음) -- 파라미터/앵커(종가) 전부 동일.

캔들 드로우 함수는 render_eth_5m_v_rebound_multitrigger_20examples_20260831.py::draw_candles를
그대로 재사용. 다른 점: 이벤트 레벨을 wick 극값이 아니라 종가(ZDC의 실제 앵커)로 표시하고,
창(window)을 고정폭이 아니라 이벤트별 실제 해상 시점(첫 피벗 확정 봉)까지 가변으로 잡는다
(ZDC는 30/60분 고정창인 giveback과 달리 해상 시점 자체가 매 이벤트 다름).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import research_eth_v_rebound_multitrigger_zigzag_direction_raw_lift_check_20260901 as zdc  # noqa: E402

TRIGGER_LABELS = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/eth_5m_v_rebound_multitrigger_labels.csv"
OUT_PATH = ROOT / "tmp/eth_v_rebound_multitrigger_zigzag_direction_raw_lift_check_20260901/zdc_hit_miss_20examples.png"
LOOKBACK_BARS = 12
LOOKFORWARD_PAD_BARS = 8
UNRESOLVED_DISPLAY_CAP = 96  # 미해결 이벤트를 그려야 할 때 화면에 보여줄 상한(전체 288은 너무 넓음)
N_PER_CLASS = 10
SEED = 20260901


def draw_candles(ax, sub: pd.DataFrame, event_pos: int, event_level: float) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.1, zorder=3)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=4))
        lows.append(bar["low"]); highs.append(bar["high"])
    ax.axhline(event_level, color="dimgray", linestyle="--", linewidth=1.1, zorder=1)
    ax.axvline(event_pos, color="dimgray", linestyle=":", linewidth=1.1, zorder=1)
    pad = (max(highs) - min(lows)) * 0.08 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def main() -> int:
    print("[1/3] klines+atr 로딩...", flush=True)
    df = zdc.load_klines()
    close = df["close"].to_numpy(dtype=np.float64)
    atr_pct = zdc._atr_pct(df, zdc.ATR_WINDOW) if hasattr(zdc, "_atr_pct") else None
    if atr_pct is None:
        import build_wave3_action_labels_20260531 as zz
        atr_pct = zz._atr_pct(df, zdc.ATR_WINDOW)

    print("[2/3] 트리거 population에 ZDC 재계산 + hit/miss 샘플링...", flush=True)
    trig = pd.read_csv(TRIGGER_LABELS, usecols=["idx", "timestamp", "direction", "triggers", "n_triggers"])
    trig["timestamp"] = pd.to_datetime(trig["timestamp"], utc=True)
    window_mask = ((trig["timestamp"] >= zdc.VAL_START) & (trig["timestamp"] <= zdc.VAL_END)) | \
                  ((trig["timestamp"] >= zdc.OOS_START) & (trig["timestamp"] <= zdc.OOS_END))
    trig = trig[window_mask].reset_index(drop=True)

    records = []
    for _, row in trig.iterrows():
        idx = int(row["idx"])
        is_bottom = row["direction"] == "upside"
        pivot_type, extreme_idx, confirm_idx = zdc.zdc_first_pivot(close, atr_pct, idx, max_lookforward=zdc.MAX_LOOKFORWARD_BARS)
        if pivot_type is None:
            continue  # 미해결 이벤트는 hit/miss 육안검증 표본에서 제외 (Step A와 동일하게 학습제외 취급)
        matched = (pivot_type == "L") if is_bottom else (pivot_type == "H")
        records.append({
            "idx": idx, "direction": row["direction"], "triggers": row["triggers"],
            "n_triggers": row["n_triggers"], "pivot_type": pivot_type, "extreme_idx": extreme_idx,
            "confirm_idx": confirm_idx, "hit": bool(matched),
        })
    pool = pd.DataFrame(records)
    print(f"  population n={len(trig)}, resolved n={len(pool)}, hit={pool['hit'].sum()}, miss={(~pool['hit']).sum()}", flush=True)

    rng = np.random.default_rng(SEED)
    hit_pool = pool[pool["hit"]].reset_index(drop=True)
    miss_pool = pool[~pool["hit"]].reset_index(drop=True)
    hit_sample = hit_pool.iloc[rng.choice(len(hit_pool), size=min(N_PER_CLASS, len(hit_pool)), replace=False)]
    miss_sample = miss_pool.iloc[rng.choice(len(miss_pool), size=min(N_PER_CLASS, len(miss_pool)), replace=False)]
    samples = [("HIT(라벨=1, 함의방향 첫확정)", r) for _, r in hit_sample.iterrows()] + \
              [("MISS(라벨=0, 반대방향 첫확정)", r) for _, r in miss_sample.iterrows()]

    print("[3/3] 차트 렌더링...", flush=True)
    n_cols = 5
    n_rows = 4
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 6.4 * n_rows), dpi=145)
    fig.suptitle(
        "ZDC(지그재그 방향확인) 라벨 육안검증 -- HIT 10건(위 2줄) / MISS 10건(아래 2줄), 무작위 샘플(seed=20260901)\n"
        "라벨 원리: 트리거 idx의 종가(회색 점선=이벤트시점, 회색 가로선=종가앵커)에서 지그재그 상태머신을 새로 시작해 "
        "가격이 먼저 위/아래 임계치(max(1%, ATR%×1.0))를 넘는 쪽이 트리거 함의방향과 일치하면 HIT. "
        "굵은 주황 세로선=실제 확정봉(임계치를 처음 넘긴 시점, hit/miss 판정은 이 봉의 결과). "
        "가는 보라 세로선=피벗 기록봉(반대편 극값이 idx 이후 안 움직였으면 idx와 겹쳐 안 보일 수 있음, 표시용) — "
        "Step A(raw-lift)에서 VAL/OOS 4칸 전부 lift<1.0으로 이미 REJECTED, 이 차트는 그 판정의 육안 근거 확인용.",
        fontsize=15, y=0.999,
    )

    for i, (group_label, event) in enumerate(samples):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        idx = int(event["idx"])
        extreme_idx = int(event["extreme_idx"])
        confirm_idx = int(event["confirm_idx"])
        is_bottom = event["direction"] == "upside"
        start = max(0, idx - LOOKBACK_BARS)
        end = min(len(df), max(confirm_idx + LOOKFORWARD_PAD_BARS + 1, idx + 1), idx + UNRESOLVED_DISPLAY_CAP)
        sub = df.iloc[start:end].reset_index(drop=True)
        event_pos = idx - start
        confirm_pos = confirm_idx - start
        extreme_pos = extreme_idx - start
        level = float(close[idx])
        draw_candles(ax, sub, event_pos, level)
        ax.axvline(confirm_pos, color="#E8871E", linestyle="-", linewidth=2.0, zorder=2)
        if extreme_idx != idx:
            ax.axvline(extreme_pos, color="#7B2FBE", linestyle="--", linewidth=1.2, zorder=2)
        ax.set_facecolor("#eef8f0" if event["hit"] else "#fdeeee")
        ticks = list(range(0, len(sub), max(1, len(sub) // 8)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{(t - event_pos) * 5:+d}" for t in ticks], fontsize=8)
        ax.tick_params(axis="y", labelsize=9)
        dir_kr = "bottom(상승기대)" if is_bottom else "top(하락기대)"
        bars_to_confirm = confirm_idx - idx
        bars_to_extreme = extreme_idx - idx
        title = (f"{group_label if c == 0 else ''}\n"
                 f"{dir_kr} | 첫피벗={event['pivot_type']} | 확정={bars_to_confirm}봉({bars_to_confirm*5}분) | "
                 f"기록봉={bars_to_extreme}봉\n"
                 f"triggers={event['triggers']}")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH} ({len(samples)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
