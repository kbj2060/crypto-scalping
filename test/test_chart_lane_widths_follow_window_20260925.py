"""차트 레인 폭이 창 토글을 따라오나 (2026-09-25).
   python -m pytest -q test/test_chart_lane_widths_follow_window_20260925.py

왜 이 테스트가 있나 — **같은 실수가 세 번 나왔다.** 창 토글이 72봉이던 시절 상수들이 그대로 남아
창이 144봉(12h)으로 늘어난 뒤에도 안 따라왔다:
  · `/api/oi-5m?bars=96`        -> 앞 48봉이 OI 없이 그려지고 사분면이 `|| 0` 으로 받아
                                   **전부 «신규 롱/숏»** 이라는 없는 사실을 찍었다.
  · 청산 이력 `..., 96)`         -> 앞 48봉에 청산 원이 없어 «청산이 없었다»로 읽혔다.
  · 레짐 `HISTORY_BARS_RETURNED` -> 앞 24봉이 비고, 리본 규약이 «빈 칸 = 횡보»라 **모름이 횡보**로 읽혔다.

⭐고침의 공통 원리는 «쓰는 쪽»이 아니라 **«먹이는 쪽»을 창에 묶는 것**이다. 이 테스트가 그걸 지킨다.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "dashboard/live/app.js").read_text(encoding="utf-8")
SERVER = (ROOT / "dashboard/server.py").read_text(encoding="utf-8")


def strip_js_comments(s: str) -> str:
    """🔴주석에 속지 않는다 -- 고침을 설명하는 내 주석이 앵커와 같은 문자열을 담는다(09-23 교훈)."""
    return re.sub(r"(?m)^\s*//.*$", "", s)


def window_max() -> int:
    m = re.search(r"const CHART_WINDOW_BARS = \[([0-9,\s]+)\]", APP)
    assert m, "CHART_WINDOW_BARS 를 못 찾았다 -- 이름이 바뀌었나?"
    return max(int(x) for x in m.group(1).split(","))


def test_window_max_is_what_we_think():
    assert window_max() == 144


@pytest.mark.parametrize("name,pattern", [
    ("풋프린트", r"API_FOOTPRINT_URL\}\?bars=\$\{chartWindowBars\}"),
    ("수급 프로파일", r"API_SUPPLY_PROFILE_URL\}\?bars=\$\{chartWindowBars\}"),
    ("청산 이력", r"API_LIQUIDATION_5M_HIST_URL\}\?asset=\$\{asset\}&bars=\$\{chartWindowBars\}"),
])
def test_lane_fetch_width_is_derived_from_the_window(name, pattern):
    assert re.search(pattern, strip_js_comments(APP)), f"{name} 폭이 창에서 파생되지 않는다"


def test_oi_fetch_width_is_derived_from_the_window_max():
    """OI 는 Δ 의 기준봉이 필요해 창 최대치 + 여유로 **한 번만** 받는다(창마다 다시 안 받는다)."""
    code = strip_js_comments(APP)
    assert "Math.max(...CHART_WINDOW_BARS) + 4" in code
    assert re.search(r"API_OI_5M_URL\}\?bars=\$\{oiBarsWanted\}", code)


def test_no_hardcoded_bar_width_survives_in_lane_fetches():
    """음성 대조군 -- 고쳐지기 전 형태가 하나라도 남아 있으면 실패한다."""
    code = strip_js_comments(APP)
    for dead in ("?bars=96", "?bars=72", "API_LIQUIDATION_5M_HIST_URL}?asset=${asset}`"):
        assert dead not in code, f"옛 고정 폭이 남아 있다: {dead}"


def test_window_toggle_refetches_every_width_bearing_lane():
    """폭을 들고 가는 레인은 토글이 **즉시 다시 받아야** 한다 -- 안 그러면 다음 폴링까지 옛 폭이다."""
    m = re.search(r"chartWindowBars = bars;(.{0,900}?)\}\);", strip_js_comments(APP), re.S)
    assert m, "창 토글 핸들러를 못 찾았다"
    body = m.group(1)
    for fn in ("refreshFootprint()", "refreshSupplyProfile()", "refreshLiquidation5mSignal()"):
        assert fn in body, f"토글이 {fn} 를 다시 안 부른다"


def test_server_liq_history_uses_the_shared_window_parser():
    code = re.sub(r"(?m)^\s*#.*$", "", SERVER)
    assert "bars = footprint_window_bars(request)" in code
    assert "compute_liquidation_5m_history, asset, bars" in code
    assert "compute_liquidation_5m_history, asset, 96" not in code, "옛 고정 96 이 남아 있다"
    assert 'f"liq5m_hist_{asset}_{bars}"' in code, "캐시 키에 bars 가 없으면 창을 바꿔도 옛 폭이 나온다"


def test_regime_workers_cover_the_window():
    """레짐 이력은 워커가 자른다 -- 창보다 짧으면 «모름»이 «횡보»로 읽힌다(리본 규약)."""
    files = sorted((ROOT / "scripts").glob("live_regime_*.py"))
    assert files, "레짐 워커를 못 찾았다"
    for f in files:
        m = re.search(r"^HISTORY_BARS_RETURNED = (\d+)", f.read_text(encoding="utf-8"), re.M)
        assert m, f"{f.name}: HISTORY_BARS_RETURNED 가 없다"
        assert int(m.group(1)) >= window_max(), f"{f.name}: {m.group(1)}봉 < 창 {window_max()}봉"


def test_ribbon_tells_unknown_apart_from_chop():
    """워커를 늘려도 콜드스타트·결손은 남는다 -- 화면이 둘을 갈라 말해야 한다."""
    code = strip_js_comments(APP)
    assert "regimeFromTs" in code
    assert "레짐 모름" in APP, "리본/툴팁이 «모름»을 말하지 않는다"
