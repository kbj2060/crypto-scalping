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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  ok  {name}")
    print("모두 통과")
