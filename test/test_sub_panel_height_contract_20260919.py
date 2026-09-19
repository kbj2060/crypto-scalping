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
    heat = _num(r"SUB_HEAT_H = subOn \? \(subStack \? (\d+)", JS, "SUB_HEAT_H")

    desktop = gap + prof + gap + one_s
    mobile = gap + heat + gap + prof + gap + one_s

    css_desktop = _num(r"#candleSvgSnapshot \{ height: (\d+)px; \}", CSS, "데스크톱 SVG 높이")
    css_desk_box = _num(r"\.candle-container \{ height: (\d+)px; \}", CSS, "데스크톱 컨테이너")
    mq = re.search(r"@media \(max-width: 720px\) \{\s*"
                   r"#candleSvgSnapshot \{ height: (\d+)px; \}[^}]*\s*"
                   r"\.candle-container \{ height: (\d+)px; \}", CSS)
    assert mq, "모바일 @media 블록을 못 찾음 -- 상하 배치인데 높이를 안 올렸다"
    css_mob, css_mob_box = int(mq.group(1)), int(mq.group(2))

    assert css_desktop == 400 + desktop, f"데스크톱 SVG {css_desktop} != {400 + desktop}"
    assert css_desk_box == 412 + desktop, f"데스크톱 상자 {css_desk_box} != {412 + desktop}"
    assert css_mob == 400 + mobile, f"모바일 SVG {css_mob} != {400 + mobile}"
    assert css_mob_box == 412 + mobile, f"모바일 상자 {css_mob_box} != {412 + mobile}"


def test_mobile_breakpoint_is_the_same_in_both_files():
    js_bp = _num(r"matchMedia\(\"\(max-width: (\d+)px\)\"\)", JS, "isMobileChartMode 경계")
    assert f"@media (max-width: {js_bp}px)" in CSS, f"CSS 에 {js_bp}px 미디어쿼리가 없다"


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
