#!/usr/bin/env python3
"""렌더 경로에 «타이머가 와야만 풀리는 불리언 래치»가 다시 생기지 않았는지 본다 (2026-09-21).

배경: 배경 탭·가려진 창에서는 setTimeout/requestAnimationFrame 콜백이 안 올 수 있다.
그때 `isScrolling = true` 나 pending rAF id 를 **잠금**으로 쓰면 영구 조기반환이 되어
차트가 그 시점에 굳는데, updateLivePriceFast() 에는 그 가드가 없어 현재가 라벨만 움직인다.
사용자 신고(04:40 에 멈춘 풋프린트 vs 04:52 까지 도는 가격)가 정확히 그 모양이었다.

실행: python3 test/test_dashboard_render_latches_20260921.py
"""
import re, sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "dashboard" / "live" / "app.js"
raw = APP.read_text(encoding="utf-8")


def strip_comments(t: str) -> str:
    """주석은 코드가 아니다 -- 설명문에 적힌 «isScrolling = true» 를 위반으로 세면 안 된다.
    줄 번호를 보존하려고 지우는 대신 같은 길이의 공백으로 바꾼다."""
    t = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), t, flags=re.S)
    return "\n".join(re.sub(r"//.*$", lambda m: " " * len(m.group(0)), ln) for ln in t.split("\n"))


src = strip_comments(raw)


def test_scroll_gate_is_time_based() -> None:
    assert re.search(r"const isScrolling = \(\) =>", src), "isScrolling 이 시각 기반 함수가 아니다"
    bad = re.findall(r"\bisScrolling\s*=\s*(?:true|false)\b", src)
    assert not bad, f"불리언 래치가 되살아났다: {bad}"
    # 읽는 쪽은 전부 호출이어야 한다 -- `if (isScrolling)` 는 함수 객체라 **항상 참**이다
    for m in re.finditer(r"\bisScrolling\b(?!\s*\()", src):
        line = src[:m.start()].count("\n") + 1
        ctx = raw.splitlines()[line - 1].strip()
        assert "const isScrolling" in ctx, \
            f"{line}행: isScrolling 을 호출 없이 읽는다 -- 함수 객체라 **항상 참**이다: {ctx}"


def test_raf_reservation_has_stall_bound() -> None:
    m = re.search(r"function scheduleSnapshotChartRender\(\) \{(.*?)\n\}", src, re.S)
    assert m, "scheduleSnapshotChartRender 를 못 찾았다"
    body = m.group(1)
    assert "RAF_STALL_MS" in body and "cancelAnimationFrame" in body, \
        "rAF 예약에 시한이 없다 -- 유실된 예약이 영구 잠금이 된다"
    assert not re.search(r"if \(snapshotChartRafId\) return;", body), \
        "무조건 조기반환이 되살아났다"


def test_fast_price_path_stays_unguarded() -> None:
    """현재가 라벨은 이 게이트들과 **독립**이어야 한다(가려진 창에서도 움직여야 한다)."""
    m = re.search(r"function updateLivePriceFast\(price\) \{(.*?)\n\}", src, re.S)
    assert m, "updateLivePriceFast 를 못 찾았다"
    assert "isScrolling" not in m.group(1), "빠른 가격 경로에 스크롤 게이트가 붙었다"


def test_density_opacity_is_mode_independent() -> None:
    """같은 밀도값은 두 모드에서 **같은 색**이어야 한다.

    알파를 모드마다 다르게 주면 색이 흐려지는 게 아니라 척도가 배경 쪽으로 눌린다 --
    실측(다크): 풋프린트 0.25 의 t=1.0 이 청산맵 0.85 의 t=0.0 과 거의 같은 밝기였다.
    겹치기를 없애고(게이트) 알파는 하나로 되돌린 것이 2026-09-21 의 수정이다.
    """
    # 🔴파일 전체에서 첫 fill-opacity 를 집으면 안 된다(다른 rect 가 여럿이다) --
    #   drawDensitySeg 본문으로 범위를 좁힌다. 2026-09-21 이 검사기 자신의 첫 판이
    #   그 실수로 주입한 회귀를 놓쳤다.
    body = re.search(r"const drawDensitySeg = \(.*?\n  \};", src, re.S)
    assert body, "drawDensitySeg 를 못 찾았다"
    m = re.search(r'setAttribute\("fill-opacity", ([^)]+)\)', body.group(0))
    assert m, "밀도 rect 의 fill-opacity 를 못 찾았다"
    assert "footprint" not in m.group(1), \
        f"밀도 알파가 모드에 의존한다 -- 같은 값이 두 색으로 보인다: {m.group(1)}"


def test_density_drawing_is_one_path_for_both_modes() -> None:
    """두 모드가 **같은 경로**로 그려야 한다.

    2026-09-21 에 풋프린트만 왼쪽 게이트로 빼 본 적이 있는데(알파를 낮춰 겹치던 것을
    피하려던 우회), 사용자 요청으로 전체폭으로 되돌렸다. 모드별 분기가 다시 생기면
    «같은 값 다른 색» 이 재발하기 쉬우므로 분기 자체를 막는다.
    """
    block = re.search(r'cachedLayer\("density".*?\n  \}\);', src, re.S)
    assert block, "밀도 층 블록을 못 찾았다"
    assert "if (footprint)" not in block.group(0), \
        "밀도 그리기에 모드 분기가 생겼다 -- 두 화면 색이 갈릴 수 있다"


def test_density_colormap_is_theme_aware() -> None:
    """밀도가 높을수록 배경 대비가 강해야 한다 -- 배경이 둘이므로 색표도 둘이다.

    2026-09-21 실측: 색표가 하나였을 때 라이트에서 t=0 대비 6.79 / t=1 대비 2.19 로
    척도가 뒤집혀 있었다(사용자 스크린샷의 진한 블록 = 밀도 최저 구간).
    """
    def stops(name):
        m = re.search(r"const " + name + r" = \[(.*?)\n\];", src, re.S)
        assert m, f"{name} 를 못 찾았다"
        rows = re.findall(r"\[([\d.]+), \[(\d+), (\d+), (\d+)\]\]", m.group(1))
        assert len(rows) >= 2, f"{name} 정지점이 부족하다"
        lum = lambda r: 0.2126 * int(r[1]) + 0.7152 * int(r[2]) + 0.0722 * int(r[3])
        return [lum(r) for r in rows]

    # 🔴이름이 아니라 **값**을 본다. 이름만 검사하면 선언을 지워도 참조가 남아 통과한다
    #   (2026-09-21 이 검사기 자신의 첫 판이 그 주입을 놓쳤다).
    dark, light = stops("DENSITY_STOPS_DARK"), stops("DENSITY_STOPS_LIGHT")
    assert all(dark[i] < dark[i + 1] for i in range(len(dark) - 1)), \
        "다크: 밀도가 높을수록 밝아져야 한다(어두운 배경 위)"
    assert all(light[i] > light[i + 1] for i in range(len(light) - 1)), \
        "라이트: 밀도가 높을수록 **어두워져야** 한다 -- 뒤집히면 척도가 거꾸로 읽힌다"
    assert re.search(r"const densityStops = \(\) =>", src), "densityStops 접근자가 없다"
    # 2026-09-21 범례가 헤더 HTML -> 캔들 SVG 안(프로파일 바닥글 아래)으로 옮겨가며
    #   densityLegendGradient() 가 사라졌다. **검사 의도는 그대로다** -- 범례 색이
    #   densityStops() 에서 나오는가. 지워진 이름 대신 범례를 만드는 자리를 찾아 본다.
    i = src.find("liqDensLegendGrad")
    assert i > 0, "범례 그라디언트를 못 찾았다"
    assert "densityStops()" in src[max(0, i - 500):i + 500], \
        "범례가 테마를 안 따라간다 -- 색 사본이 갈린다"
    assert re.search(r"const baseGeomSig = \[themeSig,", src), "캐시 서명에 테마가 없다"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  ok  {name}")
    print("모두 통과")
