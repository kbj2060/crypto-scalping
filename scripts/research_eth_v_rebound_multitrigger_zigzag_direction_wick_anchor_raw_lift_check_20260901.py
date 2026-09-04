#!/usr/bin/env python3
"""ZDC(지그재그 방향확인) 라벨의 wick-앵커 변형 raw-lift 사전점검.

종가 앵커판(research_eth_v_rebound_multitrigger_zigzag_direction_raw_lift_check_20260901.py,
VAL/OOS 4칸 전부 lift<1.0으로 REJECTED)의 유일한 변경점: 지그재그 상태머신의 시작 앵커를
close[idx] 대신 트리거 봉 자신의 wick 극값(bottom→low[idx], top→high[idx])으로 바꾼다 —
giveback(V자반등) 라벨이 쓰는 것과 동일한 앵커(`extreme = frame["low"/"high"].iloc[idx]`,
종가 아님). 가설: 종가 앵커는 발동봉 자체의 봉내 되돌림(wick→종가)을 이미 "써버린" 시점부터
시계를 재는 셈이라 진짜 엣지가 일부 소진돼 있었을 수 있다 — wick 앵커는 그 봉내 되돌림도
그대로 반영한다(giveback이 정확히 이렇게 함).

subsequent 봉 추적은 종가 앵커판과 동일하게 종가(close) 기준 그대로 유지 -- 이번에 바꾸는
변수는 오직 "시작점"뿐, 여러 개를 한꺼번에 바꾸면 어느 쪽이 원인인지 알 수 없어진다.

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

OUT_DIR = ROOT / "tmp/eth_v_rebound_multitrigger_zigzag_direction_wick_anchor_raw_lift_check_20260901"


def zdc_first_pivot_wick_anchor(
    close: np.ndarray, low: np.ndarray, high: np.ndarray, atr_pct: np.ndarray, idx: int, is_bottom: bool,
    *, max_lookforward: int,
) -> tuple[str | None, int | None, int | None]:
    """close_anchor.zdc_first_pivot()과 완전히 동일한 로직, 유일한 차이: 시작 앵커가
    close[idx]가 아니라 트리거 방향의 wick 극값(bottom->low[idx], top->high[idx]).
    subsequent 봉은 여전히 close로 추적(바뀐 변수를 하나로 한정하기 위함).
    """
    n = len(close)
    anchor = float(low[idx]) if is_bottom else float(high[idx])
    assert low[idx] <= close[idx] <= high[idx], f"sanity: bar {idx} OHLC 모순"
    low_idx = high_idx = idx
    low_price = high_price = anchor
    end = min(idx + 1 + max_lookforward, n)
    for i in range(idx + 1, end):
        price = close[i]
        if not np.isfinite(price):
            continue
        if price < low_price:
            low_idx, low_price = i, price
        if price > high_price:
            high_idx, high_price = i, price
        thr = max(MIN_REVERSAL_PCT, float(atr_pct[i]) * ATR_MULTIPLIER)
        if high_price / max(low_price, 1e-12) - 1.0 >= thr:
            if low_idx < high_idx:
                return "L", low_idx, i
            return "H", high_idx, i
    return None, None, None


def score_population(close, low, high, atr_pct, indices: np.ndarray, is_bottom: bool) -> dict[str, Any]:
    hits = 0
    n_resolved = 0
    n_unresolved = 0
    for idx in indices:
        pivot_type, _, _ = zdc_first_pivot_wick_anchor(close, low, high, atr_pct, int(idx), is_bottom, max_lookforward=MAX_LOOKFORWARD_BARS)
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


def manual_trace_check(close, low, high, atr_pct, idx: int, is_bottom: bool) -> None:
    """수동추적 스팟체크 -- 첫 몇 봉을 직접 손으로 계산해 함수 출력과 대조."""
    anchor = low[idx] if is_bottom else high[idx]
    thr0 = max(MIN_REVERSAL_PCT, atr_pct[idx + 1] * ATR_MULTIPLIER)
    print(f"  [manual trace] idx={idx} is_bottom={is_bottom} anchor(wick)={anchor:.4f} close[idx]={close[idx]:.4f} "
          f"close[idx+1]={close[idx+1]:.4f} thr@idx+1={thr0:.5f} "
          f"(close[idx+1]/anchor-1={close[idx+1]/anchor-1:.5f} if is_bottom else anchor/close[idx+1]-1)")
    result = zdc_first_pivot_wick_anchor(close, low, high, atr_pct, idx, is_bottom, max_lookforward=MAX_LOOKFORWARD_BARS)
    print(f"  [manual trace] zdc_first_pivot_wick_anchor -> {result}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/3] klines+atr 로딩...", flush=True)
    df = load_klines()
    close = df["close"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    atr_pct = close_anchor.zz._atr_pct(df, close_anchor.ATR_WINDOW)
    print(f"  bars={len(df)}", flush=True)

    print("[2/3] 수동추적 스팟체크(2건)...", flush=True)
    trig_all = pd.read_csv(TRIGGER_LABELS, usecols=["idx", "timestamp", "direction", "triggers", "n_triggers"])
    manual_trace_check(close, low, high, atr_pct, int(trig_all.iloc[100]["idx"]), trig_all.iloc[100]["direction"] == "upside")
    manual_trace_check(close, low, high, atr_pct, int(trig_all.iloc[200]["idx"]), trig_all.iloc[200]["direction"] == "upside")

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
            triggered_stats = score_population(close, low, high, atr_pct, triggered_idx, is_bottom)
            baseline_stats = score_population(close, low, high, atr_pct, all_idx, is_bottom)
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
        "method": "zigzag_direction_confirmation_wick_anchor_raw_lift_check",
        "anchor": "wick (low[idx] for bottom, high[idx] for top) -- subsequent bars still close-tracked",
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
