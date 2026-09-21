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
    # 2026-09-19 히트맵 그림 제거 -- 프로파일과 1초 수급 둘뿐이다.
    # 🔴2026-09-21 이 공식이 낡아 검사가 **계속 실패 중**이었다(배포본에서도). 상자는 두 덩이를
    #   더 갖는다: 청산밀도 범례(SUB_LEGEND_H)와 가격 플롯 아래 레인 5종.
    legend = _num(r"const SUB_LEGEND_H = subOn \? (\d+)", JS, "SUB_LEGEND_H")
    total = gap + prof + legend + gap + one_s          # = app.js 의 SUB_TOTAL

    # 레인 5종(거래대금·델타/CVD·OI·수급·청산)과 그 간격. 데스크톱 값으로 센다.
    lanes = (_num(r"LIQ_PANEL_H = mobileChart \? \d+ : (\d+)", JS, "LIQ_PANEL_H")
             + _num(r"LIQ_PANEL_GAP = (\d+)", JS, "LIQ_PANEL_GAP")
             + _num(r"OI_PANEL_H = oiBars\.length \? \(mobileChart \? \d+ : (\d+)\)", JS, "OI_PANEL_H")
             + _num(r"OI_PANEL_GAP = oiBars\.length \? (\d+)", JS, "OI_PANEL_GAP")
             + _num(r"SUP_PANEL_H = fpBars\.length \? (\d+)", JS, "SUP_PANEL_H")
             + _num(r"SUP_PANEL_GAP = fpBars\.length \? (\d+)", JS, "SUP_PANEL_GAP")
             + _num(r"TURN_H = fpBars\.length \? (\d+)", JS, "TURN_H")
             + _num(r"DCVD_H = fpBars\.length \? (\d+)", JS, "DCVD_H")
             + 2 * _num(r"FLOW_GAP = fpBars\.length \? (\d+)", JS, "FLOW_GAP"))

    # 🔴가격 플롯(ch)은 «나머지»다. 이 계약은 그 나머지가 얼마로 남는지를 고정한다 --
    #   상자만 줄이거나 레인만 키우면 캔들이 **조용히** 눌린다(그게 이 검사의 이유다).
    mt_top, mb, price_plot = 12, 70, 400

    css_svg = _num(r"#candleSvgSnapshot \{ height: (\d+)px; \}", CSS, "SVG 높이")
    css_box = _num(r"\.candle-container \{ height: (\d+)px; \}", CSS, "컨테이너 높이")
    want = mt_top + total + price_plot + lanes + mb
    assert css_svg == want, (
        f"SVG {css_svg} != {want} (여백 {mt_top}+{mb} · SUB {total} · 레인 {lanes} · "
        f"가격 플롯 {price_plot}) -- 가격 플롯이 {css_svg - want + price_plot}px 로 눌린다")
    assert css_box == css_svg + 12, f"상자 {css_box} != {css_svg + 12} (SVG + margin-top 12)"


def test_layout_is_single_not_split():
    """좌우 분할(SUB_HEAT_W/SUB_PROFILE_W)과 모바일 분기(subStack)는 걷어냈다 --
    주석 말고 **코드**에 남아 있으면 배치가 두 가지라는 뜻이다."""
    for dead in ("SUB_HEAT_W", "SUB_PROFILE_W", "SUB_HEAT_H", "renderFlowHeatmapSvg"):
        assert dead not in JS, f"죽은 상수가 남아 있다: {dead}"
    code = "\n".join(l for l in JS.splitlines() if not l.lstrip().startswith("//"))
    assert "subStack" not in code, "subStack 분기가 코드에 남아 있다"


def test_other_sessions_panel_repaint_survives():
    """🔴2026-09-19 실사고: app.js 를 통째로 복사해 다른 세션의 «패널만 따로 다시 그린다»
    수정(2bb2b2f1)을 지운 채 배포했다. 그 부품이 남아 있는지 본다."""
    # 🔴히트맵 계열(flowHeatmapSubBox/repaintFlowHeatmapPanel)은 **내가 의도적으로** 지웠다
    #   (2026-09-19 사용자 지시). 여기서 지키는 건 2bb2b2f1 의 네 부품뿐이다.
    for name in ("supplyProfileSubBox", "supply1sSubBox",
                 "repaintSupplyProfilePanel", "repaintSupply1sPanel"):
        assert name in JS, f"사라짐: {name}"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print(f"ok  {n}")
    print("all ok")
