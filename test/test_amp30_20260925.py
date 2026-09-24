"""30분 고저폭 예보(2026-09-25).   python -m pytest -q test/test_amp30_20260925.py

🔴가장 중요한 검사는 마지막 것이다 -- 서버의 `amp30_item` 이 상수를 적합한 연구 스크립트
(scripts/research_amp30_fit_20260925.py)와 **같은 숫자**를 내는지. 두 곳이 어긋나면 화면의
확률은 아무 근거가 없다(상수는 연구 쪽 정의로 적합됐다).
"""
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.server import (AMP30_LOGIT_A, AMP30_LOGIT_B, AMP30_LOOK,  # noqa: E402
                              AMP30_RATIO, AMP30_WIN, amp30_item)


def frame(ranges, close=1000.0):
    """봉마다 고저폭이 `ranges[i]`(가격 단위)인 프레임. 종가는 고정이라 bp 환산이 자명하다."""
    return pd.DataFrame([{"high": close + r / 2, "low": close - r / 2, "close": close} for r in ranges])


def test_too_few_bars_is_empty():
    assert amp30_item(frame([10.0] * (AMP30_LOOK + AMP30_WIN - 1))) == {}
    assert amp30_item(None) == {}


def test_flat_history_gives_mult_one_and_base_probability():
    # 모든 봉이 같은 고저폭이면 직전 30분 = 최근 24시간 중앙 ⇒ 배수 1.0, 확률 = sigmoid(A).
    out = amp30_item(frame([10.0] * (AMP30_LOOK + AMP30_WIN)))
    assert out["mult"] == pytest.approx(1.0)
    assert out["p_big"] == pytest.approx(1 / (1 + math.exp(-AMP30_LOGIT_A)), abs=1e-12)
    assert out["range_bp"] == pytest.approx(100.0)          # 10/1000 = 100bp
    assert out["pred_bp"] == pytest.approx(round(AMP30_RATIO * 100.0, 1))


def test_recent_spike_raises_mult_and_probability():
    r = [10.0] * (AMP30_LOOK + AMP30_WIN)
    r[-AMP30_WIN:] = [40.0] * AMP30_WIN                     # 마지막 30분만 4배로 넓어진다
    out = amp30_item(frame(r))
    assert out["mult"] > 3.9 and out["p_big"] > 0.6
    assert out["p_big"] > 1 / (1 + math.exp(-AMP30_LOGIT_A))


def test_window_is_exactly_24h_deep():
    """기준창 밖(288봉보다 오래된) 봉을 아무리 흔들어도 값이 안 바뀌어야 한다."""
    n = AMP30_LOOK + AMP30_WIN + 50
    a = [10.0] * n
    b = list(a)
    b[:50] = [999.0] * 50                                   # 창 밖만 교체
    assert amp30_item(frame(a)) == amp30_item(frame(b))


def test_matches_research_definition_on_real_panel():
    """서버 구현 == 상수를 적합한 연구 정의. 패널이 없는 기기에서는 건너뛴다."""
    panel = Path(__file__).resolve().parents[1] / "data/binance_vision/panel/ETHUSDT.parquet"
    if not panel.exists():
        pytest.skip("패널 없음 (dev 전용 검사)")
    df = pd.read_parquet(panel, columns=["high", "low", "close"]).tail(AMP30_LOOK + AMP30_WIN + 20)
    out = amp30_item(df)
    hi, lo, cl = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    rng = (hi.rolling(AMP30_WIN).max() - lo.rolling(AMP30_WIN).min()) / cl * 1e4
    win = rng.iloc[-AMP30_LOOK:]
    assert out["range_bp"] == pytest.approx(round(float(rng.iloc[-1]), 1))
    assert out["base_bp"] == pytest.approx(round(float(win.median()), 1))
    assert out["mult"] == pytest.approx(float(rng.iloc[-1]) / float(win.median()))
    assert out["p_big"] == pytest.approx(
        1 / (1 + math.exp(-(AMP30_LOGIT_A + AMP30_LOGIT_B * math.log(out["mult"])))))
