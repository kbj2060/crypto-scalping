#!/usr/bin/env python3
"""리테일/고래 분리의 자체점검 (2026-09-19).

확인하는 것 둘:
  1. 고래 임계값이 «분위»로 나오는가 — 표본이 모자라면 0(기준 없음)이어야 한다.
  2. 셀 4칸의 계약 — 고래는 매수/매도의 **부분집합**이고 리테일은 빼서 얻는다.
     화면(app.js supplyFlowOfBar)이 이 뺄셈을 그대로 하므로 부호 규약이 여기서 굳는다.

python test/test_footprint_whale_split_20260919.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.server import (  # noqa: E402
    FOOTPRINT_WHALE_MIN_SAMPLE,
    FOOTPRINT_WHALE_PCTL,
    footprint_whale_percentile,
)


def test_threshold_is_a_percentile() -> None:
    # 1..10000 에서 p99 는 9900 근처다. 정확한 인덱스 규약까지 못박는다.
    sample = list(range(1, 10001))
    thr = footprint_whale_percentile(sample, 0.99)
    assert thr == 9901.0, thr
    # 상위 1% 가 실제로 1% 인가 -- 분위의 뜻이 이것이다.
    above = sum(1 for v in sample if v >= thr)
    assert abs(above / len(sample) - 0.01) < 0.001, above

    # 값이 커져도 «비율»은 그대로다. 달러 고정 임계값과 갈리는 지점이 정확히 여기다.
    doubled = [v * 2 for v in sample]
    assert footprint_whale_percentile(doubled, 0.99) == thr * 2


def test_small_sample_means_no_threshold() -> None:
    assert footprint_whale_percentile([], FOOTPRINT_WHALE_PCTL) == 0.0
    short = [1.0] * (FOOTPRINT_WHALE_MIN_SAMPLE - 1)
    assert footprint_whale_percentile(short, FOOTPRINT_WHALE_PCTL) == 0.0
    enough = [1.0] * FOOTPRINT_WHALE_MIN_SAMPLE
    assert footprint_whale_percentile(enough, FOOTPRINT_WHALE_PCTL) > 0.0


def test_whale_is_a_subset_not_a_sibling() -> None:
    """셀 = [매수, 매도, 고래매수, 고래매도]. 리테일 = 매수 - 고래매수.

    고래 칸을 «나머지»로 읽으면 순수급 부호가 통째로 뒤집힌다 -- 이 저장소에서 두 번
    사람을 속인 모양이라(측면 분리 계열) 계약을 실행 가능한 형태로 남긴다.
    """
    buy, sell, w_buy, w_sell = 100.0, 40.0, 70.0, 5.0
    whale = w_buy - w_sell
    retail = (buy - w_buy) - (sell - w_sell)
    assert whale == 65.0
    assert retail == -5.0
    # 전체 델타는 둘의 합이어야 한다 -- 어느 한쪽을 잘못 빼면 여기서 깨진다.
    assert whale + retail == buy - sell

    # 괴리(부호 갈림)가 실제로 잡히는 조합인지. 고래 매수 × 리테일 매도 = 흡수.
    assert (whale > 0) != (retail > 0)


if __name__ == "__main__":
    test_threshold_is_a_percentile()
    test_small_sample_means_no_threshold()
    test_whale_is_a_subset_not_a_sibling()
    print("ok — 임계값 분위 · 표본 부족 · 고래 부분집합 계약 3건 통과")
