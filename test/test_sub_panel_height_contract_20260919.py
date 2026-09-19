"""수급 서브패널 높이는 app.js 와 styles.css **두 파일**에 갈라져 있다. 한쪽만 고치면
가격 플롯이 조용히 눌린다 -- 이 검사가 그 드리프트를 잡는다.

계약: styles.css 의 상자 높이 == 400 + SUB_TOTAL  (컨테이너는 +12)
  데스크톱 SUB_TOTAL = GAP + PROFILE_H + GAP + 1S_H
  모바일   SUB_TOTAL = GAP + HEAT_H + GAP + PROFILE_H + GAP + 1S_H   (히트맵이 제 줄)
그리고 모바일 경계(720px)가 app.js 와 styles.css 에서 같아야 한다.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "dashboard/live/app.js").read_text(encoding="utf-8")
CSS = (ROOT / "dashboard/live/styles.css").read_text(encoding="utf-8")


def _num(pat, text, what):
    m = re.search(pat, text)
    assert m, f"못 찾음: {what}"
    return int(m.group(1))


def test_heights_match_between_js_and_css():
    gap = _num(r"const SUB_GAP = (\d+)", JS, "SUB_GAP")
    prof = _num(r"SUB_PROFILE_H = subOn \? (\d+)", JS, "SUB_PROFILE_H")
    one_s = _num(r"SUB_1S_H = subOn \? (\d+)", JS, "SUB_1S_H")
    heat = _num(r"SUB_HEAT_H = subOn \? (\d+)", JS, "SUB_HEAT_H")

    # 2026-09-19 히트맵이 프로파일 아래 제 줄 -- 데스크톱·모바일 한 가지 배치다.
    total = gap + prof + gap + heat + gap + one_s

    css_svg = _num(r"#candleSvgSnapshot \{ height: (\d+)px; \}", CSS, "SVG 높이")
    css_box = _num(r"\.candle-container \{ height: (\d+)px; \}", CSS, "컨테이너 높이")
    assert css_svg == 400 + total, f"SVG {css_svg} != {400 + total}"
    assert css_box == 412 + total, f"상자 {css_box} != {412 + total}"


def test_layout_is_single_not_split():
    """좌우 분할(SUB_HEAT_W/SUB_PROFILE_W)과 모바일 분기(subStack)는 걷어냈다 --
    주석 말고 **코드**에 남아 있으면 배치가 두 가지라는 뜻이다."""
    for dead in ("SUB_HEAT_W", "SUB_PROFILE_W"):
        assert dead not in JS, f"죽은 상수가 남아 있다: {dead}"
    code = "\n".join(l for l in JS.splitlines() if not l.lstrip().startswith("//"))
    assert "subStack" not in code, "subStack 분기가 코드에 남아 있다"


def test_other_sessions_panel_repaint_survives():
    """🔴2026-09-19 실사고: app.js 를 통째로 복사해 다른 세션의 «패널만 따로 다시 그린다»
    수정(2bb2b2f1)을 지운 채 배포했다. 그 부품이 남아 있는지 본다."""
    for name in ("supplyProfileSubBox", "supply1sSubBox", "flowHeatmapSubBox",
                 "repaintSupplyProfilePanel", "repaintSupply1sPanel", "repaintFlowHeatmapPanel"):
        assert name in JS, f"사라짐: {name}"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print(f"ok  {n}")
    print("all ok")
