"""2026-09-30 호가 불균형 중심 = 지금 가격. 창 중앙값을 중심으로 쓰면 가격이 창 안에서 0.5% 넘게 움직였을 때 ±1 로 붙었다."""
import numpy as np
from dashboard import server


def test_obi_centered_on_current_price():
    price = 2600 + 0.5 * np.arange(400)          # 2600 ~ 2799.5
    now = 2690.0
    q = np.where(price < now, 10.0, -8.0)         # 지금가 아래 매수 10 · 위 매도 8 (칸마다)
    assert server.heatmap_obi(q, price, now) > 0, "지금가 중심이면 매수 우위(+)"
    assert server.heatmap_obi(q, price, 2706.0) == -1.0, "옛 방식(창 중앙값 = 0.6% 위) 재현 → −1"


def test_obi_bounds_and_empty_band():
    price = np.array([100.0, 100.5]); q = np.array([5.0, -5.0])
    assert server.heatmap_obi(q, price, 100.25) == 0.0
    assert server.heatmap_obi(q, price, 1000.0) == 0.0   # 범위 안 호가 없음 → 0(나눗셈 보호)
