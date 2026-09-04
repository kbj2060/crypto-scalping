#!/usr/bin/env python3
"""9트리거 통합 population에 "지그재그 방향확인"(ZDC) 라벨을 적용한 raw-lift 사전점검.

TabPFN 없이, 트리거 이벤트에서 지그재그 상태머신을 새로 시작했을 때 확정되는 첫 피벗이
트리거의 함의 방향과 일치하는지만 본다. `build_wave3_action_labels_20260531.py::_zigzag_pivots()`의
trend==0 분기 로직을 그대로 이벤트별로 재현하되(early-exit), 매 봉 분류(bar-classification, 이미
momentum-chasing 편향으로 실패 판정난 zigzag_action의 원인)는 전혀 쓰지 않는다 — 트리거당 정확히
1개 라벨만 부여한다(V자반등 giveback과 동일한 이벤트-앵커링 원칙).

트리거 population은 재계산하지 않고 기존 9트리거 라벨 CSV(idx/timestamp/direction)를 그대로
재사용한다 — 트리거(어느 봉이 후보인가)와 라벨(정답 공식)은 완전히 분리된 축이라는 이 라인업의
기존 설계원칙 그대로.

비-동어반복 베이스라인: "트리거 게이트만 뺀 동일 ZDC 공식"을 VAL/OOS 창의 모든 적격봉(bottom/top
방향 각각)에 무조건 적용 — breakout_continuation v1이 겪은 동어반복 버그와 같은 함정을 피한다.

research_diagnostic_not_live_wired. trading_bot.py 변경 없음.
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

import build_wave3_action_labels_20260531 as zz  # noqa: E402

ETH_KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
TRIGGER_LABELS = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/eth_5m_v_rebound_multitrigger_labels.csv"
OUT_DIR = ROOT / "tmp/eth_v_rebound_multitrigger_zigzag_direction_raw_lift_check_20260901"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")

# 기존 zigzag_action 파이프라인(BTC/SOL/ETH 전부)이 이미 쓰는 값 그대로 재사용 -- 신규 자유파라미터 아님.
MIN_REVERSAL_PCT = 0.01
ATR_WINDOW = 14
ATR_MULTIPLIER = 1.0
MAX_LOOKFORWARD_BARS = 288  # 24h 안전 상한 (자유선택, Step A 결과로 재검토 대상)

Z_95 = 1.959963984540054


def wilson_ci(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(ETH_KLINES, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def zdc_first_pivot(close: np.ndarray, atr_pct: np.ndarray, idx: int, *, max_lookforward: int) -> tuple[str | None, int | None, int | None]:
    """_zigzag_pivots()의 trend==0 분기를 idx에서 새로 시작해 재현(early-exit).

    Returns (pivot_type, extreme_bar_idx, confirm_bar_idx):
    - extreme_bar_idx: _zigzag_pivots()와 동일하게 실제 극값이 발생한 봉(low_idx/high_idx) --
      hit/miss 판정에 쓰이는 값(트리거 무관 동일 공식).
    - confirm_bar_idx(신규, 2026-09-01 시각검증 중 추가): 임계치가 실제로 확정된 봉(i) --
      extreme_bar_idx==idx인 경우(극값이 idx 자신에서 한 번도 안 움직인 채 반대편이 임계치를 넘긴
      경우) 둘이 크게 벌어질 수 있어(예: idx는 그대로인데 6봉 뒤에야 확정), 차트에서 "언제 실제로
      해상됐는지" 보여주려면 이 값이 필요함 -- hit/miss 판정 자체와는 무관, 순수 표시용.
    (None, None, None)이면 max_lookforward 안에 미해결.
    """
    n = len(close)
    low_idx = high_idx = idx
    low_price = high_price = close[idx]
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


def self_check(df: pd.DataFrame, close: np.ndarray, atr_pct: np.ndarray) -> None:
    """idx=0에서 재현한 첫 피벗이 _zigzag_pivots()의 진짜 첫 글로벌 피벗과 일치하는지 확인."""
    real_pivots = zz._zigzag_pivots(df, min_reversal_pct=MIN_REVERSAL_PCT, atr_window=ATR_WINDOW, atr_multiplier=ATR_MULTIPLIER)
    assert real_pivots, "self-check: _zigzag_pivots() returned no pivots on full series"
    real_type, real_bar_idx = real_pivots[0][2], real_pivots[0][0]
    my_type, my_bar_idx, _ = zdc_first_pivot(close, atr_pct, 0, max_lookforward=len(close))
    assert my_type == real_type and my_bar_idx == real_bar_idx, (
        f"self-check FAILED: real first pivot=({real_type},{real_bar_idx}) vs zdc_first_pivot(idx=0)=({my_type},{my_bar_idx})"
    )
    print(f"[self-check OK] first global pivot ({real_type}@{real_bar_idx}) matches zdc_first_pivot(idx=0) exactly.")


def score_population(close: np.ndarray, atr_pct: np.ndarray, indices: np.ndarray, is_bottom: bool) -> dict[str, Any]:
    hits = 0
    n_resolved = 0
    n_unresolved = 0
    for idx in indices:
        pivot_type, _, _ = zdc_first_pivot(close, atr_pct, int(idx), max_lookforward=MAX_LOOKFORWARD_BARS)
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] klines 로딩...", flush=True)
    df = load_klines()
    close = df["close"].to_numpy(dtype=np.float64)
    atr_pct = zz._atr_pct(df, ATR_WINDOW)
    print(f"  bars={len(df)}, {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}", flush=True)

    print("[2/4] 셀프체크(idx=0 재현이 진짜 첫 글로벌 피벗과 일치하는지)...", flush=True)
    self_check(df, close, atr_pct)

    print("[3/4] 트리거 population 로딩...", flush=True)
    trig = pd.read_csv(TRIGGER_LABELS, usecols=["idx", "timestamp", "direction", "triggers", "n_triggers"])
    trig["timestamp"] = pd.to_datetime(trig["timestamp"], utc=True)

    windows = {"VAL": (VAL_START, VAL_END), "OOS": (OOS_START, OOS_END)}
    ts_all = df["timestamp"]

    rows = []
    for window_name, (w_start, w_end) in windows.items():
        window_mask_klines = (ts_all >= w_start) & (ts_all <= w_end)
        all_idx = np.flatnonzero(window_mask_klines.to_numpy())
        # 마지막 MAX_LOOKFORWARD_BARS는 해당 구간 자체를 벗어나 라벨을 볼 수 있어(라벨은 미래를 보는 게
        # 정상이지만) 그래도 원본 시계열 범위 밖으로 나가진 않게 함(load_klines가 이미 원본 끝까지 포함).
        print(f"[4/4] window={window_name} baseline n_bars={len(all_idx)}...", flush=True)

        window_mask_trig = (trig["timestamp"] >= w_start) & (trig["timestamp"] <= w_end)
        trig_w = trig[window_mask_trig]

        for side_label, is_bottom, direction_value in (("bottom(upside)", True, "upside"), ("top(downside)", False, "downside")):
            triggered_idx = trig_w.loc[trig_w["direction"] == direction_value, "idx"].to_numpy()
            triggered_stats = score_population(close, atr_pct, triggered_idx, is_bottom)
            baseline_stats = score_population(close, atr_pct, all_idx, is_bottom)
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
        "method": "zigzag_direction_confirmation_raw_lift_check",
        "min_reversal_pct": MIN_REVERSAL_PCT, "atr_window": ATR_WINDOW, "atr_multiplier": ATR_MULTIPLIER,
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
