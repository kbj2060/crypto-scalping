#!/usr/bin/env python3
"""ZDC(지그재그 방향확인) 라벨의 "완전 wick" 변형 raw-lift 사전점검.

wick-앵커판(research_eth_v_rebound_multitrigger_zigzag_direction_wick_anchor_raw_lift_check_
20260901.py, 4칸 전부 lift>1.0으로 개선됐으나 1.01~1.05x로 여전히 보류권 최하단)에서 한 걸음
더: 시작 앵커뿐 아니라 **subsequent 봉의 추적 자체**도 종가(close) 대신 그 봉의 고가/저가로
바꾼다 -- "가격이 먼저 위/아래 임계치를 넘는다"를 종가 확정이 아니라 봉중(intrabar) 터치
기준으로 재정의하는 셈.

주의: 이건 wick-앵커판보다 한 단계 더 나아간 변형이다. giveback(V자반등) 라벨 자체도 "peak"
추적에는 고가(`full_slice["high"].max()`)를 쓰지만 "fast move"(진행 측정) 자체는 여전히
종가(`fast_slice["close"].max()`) 기준이라 -- 이번 변형은 giveback보다도 더 적극적으로
wick을 쓴다. 봉중 터치는 실제 체결 가능성(스프레드/슬리피지)을 반영 안 하므로, lift가 개선되더라도
"진짜 거래 가능한 엣지"인지는 별도 판단이 필요하다는 점을 결과에서 함께 봐야 한다.

트리거 population은 재사용(재계산 없음), 비-동어반복 베이스라인 원칙도 동일.
research_diagnostic_not_live_wired.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import research_eth_v_rebound_multitrigger_zigzag_direction_raw_lift_check_20260901 as close_anchor  # noqa: E402

wilson_ci = close_anchor.wilson_ci
load_klines = close_anchor.load_klines
TRIGGER_LABELS = close_anchor.TRIGGER_LABELS
VAL_START, VAL_END = close_anchor.VAL_START, close_anchor.VAL_END
OOS_START, OOS_END = close_anchor.OOS_START, close_anchor.OOS_END
MIN_REVERSAL_PCT = close_anchor.MIN_REVERSAL_PCT
ATR_MULTIPLIER = close_anchor.ATR_MULTIPLIER
MAX_LOOKFORWARD_BARS = close_anchor.MAX_LOOKFORWARD_BARS

OUT_DIR = ROOT / "tmp/eth_v_rebound_multitrigger_zigzag_direction_full_wick_raw_lift_check_20260901"


def zdc_first_pivot_full_wick(
    low: np.ndarray, high: np.ndarray, atr_pct: np.ndarray, idx: int, is_bottom: bool,
    *, max_lookforward: int,
) -> tuple[str | None, int | None, int | None]:
    """앵커=wick(bottom->low[idx], top->high[idx]) + subsequent 봉도 그 봉의 저가/고가로 추적
    (종가 아님). 나머지 구조(early-exit, threshold, 반환형)는 close_anchor.zdc_first_pivot()과
    동일하게 유지 -- 바뀐 건 "어떤 가격을 볼 것인가"뿐.
    """
    n = len(low)
    anchor = float(low[idx]) if is_bottom else float(high[idx])
    low_idx = high_idx = idx
    low_price = high_price = anchor
    end = min(idx + 1 + max_lookforward, n)
    for i in range(idx + 1, end):
        bar_low, bar_high = low[i], high[i]
        if not (np.isfinite(bar_low) and np.isfinite(bar_high)):
            continue
        if bar_low < low_price:
            low_idx, low_price = i, bar_low
        if bar_high > high_price:
            high_idx, high_price = i, bar_high
        thr = max(MIN_REVERSAL_PCT, float(atr_pct[i]) * ATR_MULTIPLIER)
        if high_price / max(low_price, 1e-12) - 1.0 >= thr:
            if low_idx < high_idx:
                return "L", low_idx, i
            return "H", high_idx, i
    return None, None, None


def score_population(low, high, atr_pct, indices: np.ndarray, is_bottom: bool) -> dict[str, Any]:
    hits = 0
    n_resolved = 0
    n_unresolved = 0
    for idx in indices:
        pivot_type, _, _ = zdc_first_pivot_full_wick(low, high, atr_pct, int(idx), is_bottom, max_lookforward=MAX_LOOKFORWARD_BARS)
        if pivot_type is None:
            n_unresolved += 1
            continue
        n_resolved += 1
        matched = (pivot_type == "L") if is_bottom else (pivot_type == "H")
        hits += int(matched)
    rate = hits / n_resolved if n_resolved else float("nan")
    ci_lo, ci_hi = wilson_ci(hits, n_resolved)
    return {
        "n_events": int(len(indices)), "n_resolved": n_resolved, "n_unresolved": n_unresolved,
        "unresolved_pct": (n_unresolved / len(indices) * 100.0) if len(indices) else float("nan"),
        "hits": hits, "hit_rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi,
    }


def manual_trace_check(low, high, atr_pct, idx: int, is_bottom: bool) -> None:
    anchor = low[idx] if is_bottom else high[idx]
    thr0 = max(MIN_REVERSAL_PCT, atr_pct[idx + 1] * ATR_MULTIPLIER)
    print(f"  [manual trace] idx={idx} is_bottom={is_bottom} anchor(wick)={anchor:.4f} "
          f"low[idx+1]={low[idx+1]:.4f} high[idx+1]={high[idx+1]:.4f} thr@idx+1={thr0:.5f}")
    result = zdc_first_pivot_full_wick(low, high, atr_pct, idx, is_bottom, max_lookforward=MAX_LOOKFORWARD_BARS)
    print(f"  [manual trace] zdc_first_pivot_full_wick -> {result}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/3] klines+atr 로딩...", flush=True)
    df = load_klines()
    low = df["low"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    atr_pct = close_anchor.zz._atr_pct(df, close_anchor.ATR_WINDOW)
    print(f"  bars={len(df)}", flush=True)

    print("[2/3] 수동추적 스팟체크(2건)...", flush=True)
    trig_all = pd.read_csv(TRIGGER_LABELS, usecols=["idx", "timestamp", "direction", "triggers", "n_triggers"])
    manual_trace_check(low, high, atr_pct, int(trig_all.iloc[100]["idx"]), trig_all.iloc[100]["direction"] == "upside")
    manual_trace_check(low, high, atr_pct, int(trig_all.iloc[200]["idx"]), trig_all.iloc[200]["direction"] == "upside")

    print("[3/3] 트리거 population + baseline 스코어링...", flush=True)
    trig_all["timestamp"] = pd.to_datetime(trig_all["timestamp"], utc=True)
    windows = {"VAL": (VAL_START, VAL_END), "OOS": (OOS_START, OOS_END)}
    ts_all = df["timestamp"]

    rows = []
    for window_name, (w_start, w_end) in windows.items():
        window_mask_klines = (ts_all >= w_start) & (ts_all <= w_end)
        all_idx = np.flatnonzero(window_mask_klines.to_numpy())
        print(f"  window={window_name} baseline n_bars={len(all_idx)}...", flush=True)

        window_mask_trig = (trig_all["timestamp"] >= w_start) & (trig_all["timestamp"] <= w_end)
        trig_w = trig_all[window_mask_trig]

        for side_label, is_bottom, direction_value in (("bottom(upside)", True, "upside"), ("top(downside)", False, "downside")):
            triggered_idx = trig_w.loc[trig_w["direction"] == direction_value, "idx"].to_numpy()
            triggered_stats = score_population(low, high, atr_pct, triggered_idx, is_bottom)
            baseline_stats = score_population(low, high, atr_pct, all_idx, is_bottom)
            lift = (triggered_stats["hit_rate"] / baseline_stats["hit_rate"]) if baseline_stats["hit_rate"] else float("nan")
            rows.append({
                "window": window_name, "side": side_label,
                "n_triggered": triggered_stats["n_events"], "n_resolved": triggered_stats["n_resolved"],
                "unresolved_pct": triggered_stats["unresolved_pct"],
                "hit_rate": triggered_stats["hit_rate"], "ci_lo": triggered_stats["ci_lo"], "ci_hi": triggered_stats["ci_hi"],
                "baseline_n": baseline_stats["n_events"], "baseline_unresolved_pct": baseline_stats["unresolved_pct"],
                "baseline_rate": baseline_stats["hit_rate"], "lift": lift,
                "low_n": triggered_stats["n_resolved"] < 30,
            })
            print(f"    {side_label}: n={triggered_stats['n_events']} hit_rate={triggered_stats['hit_rate']:.4f} "
                  f"baseline={baseline_stats['hit_rate']:.4f} lift={lift:.3f} unresolved={triggered_stats['unresolved_pct']:.1f}%", flush=True)

    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(OUT_DIR / "scorecard.csv", index=False)
    report = {
        "method": "zigzag_direction_confirmation_full_wick_raw_lift_check",
        "anchor": "wick (low[idx] for bottom, high[idx] for top)",
        "subsequent_bar_tracking": "wick (bar's own low/high, NOT close) -- more aggressive than giveback's own convention",
        "min_reversal_pct": MIN_REVERSAL_PCT, "atr_multiplier": ATR_MULTIPLIER,
        "max_lookforward_bars": MAX_LOOKFORWARD_BARS,
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scorecard": rows,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nsaved: {OUT_DIR / 'scorecard.csv'}")
    with pd.option_context("display.width", 160):
        print(scorecard.to_string(index=False))


if __name__ == "__main__":
    main()
