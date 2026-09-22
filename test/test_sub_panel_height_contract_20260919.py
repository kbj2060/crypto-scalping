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
    # 2026-09-23 OKX 레인이 1초 수급 **바로 아래**에 같은 높이로 붙었다(눈으로 대조하려고).
    okx = _num(r"const SUB_OKX_H = subOn \? (\d+)", JS, "SUB_OKX_H")
    total = gap + prof + legend + gap + one_s + gap + okx     # = app.js 의 SUB_TOTAL

    # 2026-09-22 레인 5종 -> 두 행(사분면 + 누적 CVD). 데스크톱 값으로 센다.
    # 2026-09-23 RVOL 은 **전용 행을 안 쓴다** -- 누적 CVD 레인 안에 자기 축으로 겹친다.
    #   (한 번 세 행으로 갔다가 사용자 지시로 되돌렸다. 되돌릴 때 여기도 같이 와야 한다.)
    lanes = (_num(r"QUAD_H = fpBars\.length \? \(mobileChart \? \d+ : (\d+)\)", JS, "QUAD_H")
             + _num(r"QUAD_TXT = \(fpBars\.length && QUAD_TEXT_OK\) \? \(mobileChart \? \d+ : (\d+)\)", JS, "QUAD_TXT")
             + _num(r"CUM_H = fpBars\.length \? \(mobileChart \? \d+ : (\d+)\)", JS, "CUM_H")
             + 2 * _num(r"LANE_GAP = fpBars\.length \? (\d+)", JS, "LANE_GAP"))

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


def test_lanes_tile_without_overlap():
    """레인 두 행(사분면 -> 누적 CVD)이 **겹치지 않아야** 한다.

    🔴SVG 는 안 잘라준다 -- 겹치면 그냥 포개 그린다. 에러도 경고도 없다.
    데스크톱 기준(ROW_H=0 · PRICE_ROW_H=0)으로 y 좌표 식을 그대로 평가한다.
    """
    env = {"QUAD_H": _num(r"QUAD_H = fpBars\.length \? \(mobileChart \? \d+ : (\d+)\)", JS, "QUAD_H"),
           "QUAD_TXT": _num(r"QUAD_TXT = \(fpBars\.length && QUAD_TEXT_OK\) \? \(mobileChart \? \d+ : (\d+)\)", JS, "QUAD_TXT"),
           "CUM_H": _num(r"CUM_H = fpBars\.length \? \(mobileChart \? \d+ : (\d+)\)", JS, "CUM_H"),
           "LANE_GAP": _num(r"LANE_GAP = fpBars\.length \? (\d+)", JS, "LANE_GAP"),
           "ROW_H": 0, "PRICE_ROW_H": 0, "plotBottom": 0}
    for name in ("quadY", "cumY", "cumBottom"):
        m = re.search(rf"  const {name} = ([^;]+?);", JS)
        assert m, f"못 찾음: {name}"
        env[name] = eval(m.group(1).split("//")[0].strip(), {"__builtins__": {}}, env)  # noqa: S307

    lanes = [("사분면", env["quadY"], env["QUAD_H"] + env["QUAD_TXT"]),
             ("누적 CVD", env["cumY"], env["CUM_H"])]
    for (n1, y1, h1), (n2, y2, _) in zip(lanes, lanes[1:]):
        assert y1 + h1 <= y2, f"{n1}({y1}+{h1})가 {n2}({y2}) 위로 올라탄다"
    # «두 행 + 간격»의 합이 높이 계약의 lanes 와 같아야 한다(둘이 갈라지면 조용히 눌린다)
    total = lanes[-1][1] + lanes[-1][2] - lanes[0][1] + env["LANE_GAP"]
    want = env["QUAD_H"] + env["QUAD_TXT"] + env["CUM_H"] + 2 * env["LANE_GAP"]
    assert total == want, f"레인 스택 {total} != 높이 예산 {want}"
    # 🔴RVOL 이 전용 행으로 되돌아가면 여기도 같이 와야 한다 -- 한쪽만 고치면 캔들이 눌린다.
    assert "RVOL_H" not in JS, "RVOL_H 가 생겼다 -- 레인 예산과 styles.css 를 같이 고쳤는지 확인"


def test_sub_panels_tile_without_overlap():
    """네 패널(프로파일·1초 수급·OKX 수급·밀도 범례)이 **겹치지 않고** SUB_TOTAL 을 채워야 한다.

    🔴겹쳐도 SVG 는 안 잘라준다 -- 그냥 포개져 그려진다. 에러도 경고도 없다.
      모자라면 가격 플롯 위에 빈 띠가 생기고, 넘치면 캔들 위로 올라탄다.
    2026-09-22 사용자 지시로 순서를 뒤집었다(프로파일이 위, 1초 수급이 아래). 밀도 범례는
    풋프린트 배경을 설명하는 것이라 **따라 올라가지 않고** 가격 플롯 바로 위에 남는다.
    """
    env = {"mtTop": 12,
           "SUB_GAP": _num(r"const SUB_GAP = (\d+)", JS, "SUB_GAP"),
           "SUB_PROFILE_H": _num(r"SUB_PROFILE_H = subOn \? (\d+)", JS, "SUB_PROFILE_H"),
           "SUB_1S_H": _num(r"SUB_1S_H = subOn \? (\d+)", JS, "SUB_1S_H"),
           "SUB_LEGEND_H": _num(r"const SUB_LEGEND_H = subOn \? (\d+)", JS, "SUB_LEGEND_H"),
           "SUB_OKX_H": _num(r"const SUB_OKX_H = subOn \? (\d+)", JS, "SUB_OKX_H")}
    for name in ("subProfileY", "sub1sY", "subOkxY", "subLegendY"):
        m = re.search(rf"  const {name} = ([^;]+);", JS)
        assert m, f"못 찾음: {name}"
        env[name] = eval(m.group(1).strip(), {"__builtins__": {}}, env)  # noqa: S307 -- 저장소 제 코드

    panels = sorted([("프로파일", env["subProfileY"], env["SUB_PROFILE_H"]),
                     ("1초 수급", env["sub1sY"], env["SUB_1S_H"]),
                     ("OKX 수급", env["subOkxY"], env["SUB_OKX_H"]),
                     ("밀도 범례", env["subLegendY"], env["SUB_LEGEND_H"])], key=lambda r: r[1])
    for (n1, y1, h1), (n2, y2, _) in zip(panels, panels[1:]):
        assert y1 + h1 <= y2, f"{n1}({y1}+{h1})가 {n2}({y2}) 위로 올라탄다 -- SVG 는 안 잘라준다"

    assert panels[0][0] == "프로파일", f"맨 위가 프로파일이 아니다: {panels[0][0]}"
    assert panels[1][0] == "1초 수급", f"둘째가 1초 수급이 아니다: {panels[1][0]}"
    assert panels[2][0] == "OKX 수급", (
        f"OKX 레인은 바이낸스 **바로 아래**여야 한다(사용자 지시): {panels[2][0]}")
    assert panels[3][0] == "밀도 범례", "밀도 범례는 가격 플롯 바로 위에 남아야 한다"
    assert env["SUB_OKX_H"] == env["SUB_1S_H"], (
        "🔴두 레인은 **같은 높이**여야 한다 -- 눈금을 공유해도 높이가 다르면 비교가 왜곡된다")

    total = (env["SUB_GAP"] + env["SUB_PROFILE_H"] + env["SUB_LEGEND_H"]
             + env["SUB_GAP"] + env["SUB_1S_H"]
             + env["SUB_GAP"] + env["SUB_OKX_H"])         # = app.js 의 SUB_TOTAL
    used = panels[-1][1] + panels[-1][2] + env["SUB_GAP"] - env["mtTop"]
    assert used == total, (
        f"패널이 쓰는 높이 {used} != SUB_TOTAL {total} -- "
        f"{'가격 플롯 위에 빈 띠가 생긴다' if used < total else '캔들 위로 올라탄다'}")


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
