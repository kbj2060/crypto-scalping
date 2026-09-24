"""OI 5분 레인의 OKX 커버리지 가드(2026-09-25).  python -m pytest -q test/test_oi5m_okx_coverage_guard_20260925.py

계약: `/api/oi-5m` 은 풋프린트(`api_footprint`)와 **같은 경계**로 자른다 — `b > okx_fp["first_bar"]`.
안 그러면 한 창에 1거래소 봉과 2거래소 봉이 섞여 봉 끝 OI 가 튄다(09-25 실측 +24%).
🔴클라(quadLane/cumLane)는 풋프린트 봉이 없는 캔들을 이미 건너뛴다. 두 경계가 같아야
  «그려지는 봉 = OI 있는 봉»이 구성상 성립한다.

서버 전체를 띄우지 않고 그 식만 떼어 돌린다 — 로직이 바뀌면 여기가 깨진다.
"""
import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "dashboard/server.py").read_text(encoding="utf-8")


def apply_guard(buckets, okx_d, okx_from):
    """server.py 의 bars_out 식을 그대로 실행한다(아래 test_source_still_matches 가 동일성을 지킨다)."""
    return ([[b[0], round(b[1] + okx_d[b[0]][0], 3), round(b[2] + okx_d[b[0]][1], 3)] + b[3:]
             for b in buckets if b[0] > okx_from and b[0] in okx_d]
            if okx_d else buckets)


BUCKETS = [[300, 10.0, 1000.0, 5, 0], [600, 20.0, 1020.0, 5, 0],
           [900, -5.0, 1015.0, 5, 0], [1200, 7.0, 1022.0, 5, 0]]


def test_uncovered_bars_are_dropped_not_left_binance_only():
    okx_d = {900: (3.0, 500.0), 1200: (-2.0, 498.0)}          # OKX 는 900 부터
    out = apply_guard(BUCKETS, okx_d, okx_from=600)           # 600 = OKX 가 중간에 합류한 봉
    assert [b[0] for b in out] == [900, 1200], "앞 봉을 바이낸스만으로 남기면 단차가 생긴다"
    assert out[0][2] == 1515.0 and out[1][2] == 1520.0        # 봉 끝 OI = 두 거래소 합
    assert out[0][1] == -2.0 and out[1][1] == 5.0             # Δ 도 합


def test_first_bar_itself_is_excluded():
    """`first_bar` 는 OKX 가 **봉 중간에** 합류한 봉이라 반쪽이다 -- 풋프린트와 같이 `>` 로 뺀다."""
    okx_d = {600: (1.0, 400.0), 900: (3.0, 403.0)}
    assert [b[0] for b in apply_guard(BUCKETS, okx_d, okx_from=600)] == [900]


def test_no_okx_at_all_keeps_every_binance_bar():
    """수집기가 죽었다고 레인을 통째로 지우지는 않는다 -- venues 가 바이낸스 단독이라고 말한다."""
    out = apply_guard(BUCKETS, {}, okx_from=600)
    assert out == BUCKETS


def test_bar_missing_from_okx_is_dropped_rather_than_half_summed():
    okx_d = {600: (1.0, 400.0), 1200: (-2.0, 398.0)}          # 900 이 빠졌다(OI WS 딸꾹질)
    assert [b[0] for b in apply_guard(BUCKETS, okx_d, okx_from=300)] == [600, 1200]


def test_source_still_matches_this_harness():
    """🔴이 테스트의 값은 위 사본을 돌린 것이다 -- 서버 식이 바뀌면 사본이 낡는다.
    주석에 속지 않도록 **주석을 지운 소스**에서 앵커를 찾는다(09-23 교훈)."""
    code = re.sub(r"(?m)^\s*#.*$", "", SRC)
    assert "okx_from = okx_fp[\"first_bar\"]" in code
    assert "for b in buckets if b[0] > okx_from and b[0] in okx_d" in code
    assert "if okx_d else buckets" in code
    # 음성 대조군: 고쳐지기 전 식이 남아 있으면 안 된다
    assert "if b[0] in okx_d else b for b in buckets" not in code
