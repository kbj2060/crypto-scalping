"""tail_risk z 기준선이 방금 넣은 분(롤링 1분 합과 겹치는 표본)을 빼는지 -- 넣은 채면 z 가 sqrt(N-1) 에 묶인다."""
import math
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tail_risk_interceptor import TailRiskInterceptor


def test_burst_not_in_own_baseline():
    t = object.__new__(TailRiskInterceptor)
    quiet = [1000.0 + 10 * (i % 3) for i in range(29)]
    t._history_long = deque(quiet + [5e6], maxlen=30)
    t._history_short = deque(quiet + [0.0], maxlen=30)
    t._recalculate_stats()
    z = (5e6 - t.mu_long) / max(t.sigma_long, 1.0)
    assert abs(t.mu_long - sum(quiet) / 29) < 1e-9
    assert z > math.sqrt(29) * 10, z          # 옛 판: 제 표본이 섞여 z ≤ 5.39


def test_single_sample_keeps_defaults():
    t = object.__new__(TailRiskInterceptor)
    t.mu_long, t.sigma_long, t.mu_short, t.sigma_short = 0.0, 1.0, 0.0, 1.0
    t._history_long, t._history_short = deque([7.0]), deque([3.0])
    t._recalculate_stats()
    assert (t.mu_long, t.sigma_long) == (0.0, 1.0)


if __name__ == "__main__":
    test_burst_not_in_own_baseline()
    test_single_sample_keeps_defaults()
    print("ok")
