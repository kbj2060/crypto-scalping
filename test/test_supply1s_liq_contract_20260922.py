"""1초 수급 차트의 **조용한 실패 모드**들. 전부 «에러 없이 틀린 그림»이 되는 것들이다.

① 부호 규약: 서버가 [초, 롱청산, 숏청산] 로 보내는데 클라가 [숏, 롱] 으로 읽으면 그대로
   뒤집힌다 -- 에러도 빈 화면도 없이 **반대 방향이 맞는 것처럼** 그려진다.
     롱 청산 = 시장에 강제 SELL -> 아래쪽(음수) · 숏 청산 = 강제 BUY -> 위쪽(양수)
     따라서 값은 «숏 − 롱» 이다.
② 두 판 배치(2026-09-22): 누적 스택의 층 경계가 어긋나거나, 청산이 다시 공유 축으로
   돌아가거나, 크기 스케일이 선형이 되면 -- 셋 다 **그림만 조용히 틀린다**.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRV = (ROOT / "dashboard/server.py").read_text(encoding="utf-8")
JS = (ROOT / "dashboard/live/app.js").read_text(encoding="utf-8")


def test_server_emits_long_then_short():
    """서버 칸은 [롱, 숏] 순서다. side == "long" 이 0번 칸."""
    assert re.search(r'cell\[0 if e\.get\("side"\) == "long" else 1\]', SRV), \
        "청산 칸의 롱/숏 배치가 바뀌었다 -- 클라의 c[1] - c[0] 가 뒤집힌다"
    assert re.search(r'liq_rows = \[\[s, round\(v\[0\], 3\), round\(v\[1\], 3\)\]', SRV), \
        "liq_rows 가 [초, v[0]=롱, v[1]=숏] 순서가 아니다"
    assert '"liq": liq_rows' in SRV, "payload 에 liq 가 없다"


def test_client_reads_same_order_and_sign():
    """클라는 [롱, 숏] 로 받아 «숏 − 롱» 을 그린다."""
    assert re.search(r"liq1s\.set\(r\[0\], \[r\[1\], r\[2\]\]\)", JS), \
        "클라가 liq 행을 [r[1]=롱, r[2]=숏] 로 안 읽는다"
    assert re.search(r"liqEv\.push\(\{ s: s2, v: c\[1\] - c\[0\] \}\)", JS), \
        "청산 부호가 «숏 − 롱» 이 아니다 -- 점의 색이 통째로 뒤집힌다"


def test_in_progress_second_is_excluded():
    """진행 중인 초를 보내면 클라가 «받은 초»로 알고 반쪽으로 굳힌다(seconds 와 같은 규칙)."""
    assert "if not (liq_floor < s < newest):" in SRV, \
        "청산 집계가 진행 중인 초(newest)를 제외하지 않는다"


def test_cursor_is_separate_from_trade_cursor():
    """청산은 이벤트가 없는 초가 많다 -- 체결 초 커서를 공유하면 통째로 건너뛰어진다."""
    assert 'request.query.get("sinceLiq"' in SRV, "서버에 sinceLiq 커서가 없다"
    assert "sinceLiq=${liq1sSince}" in JS, "클라가 sinceLiq 를 안 보낸다"
    assert "let liq1sSince = 0;" in JS, "클라에 청산 전용 커서가 없다"


def test_restored_on_restart():
    """청산은 11분에 몇 건이라 재시작 후 빈 deque 로 두면 새로고침해도 선이 안 그려진다.

    다른 수급 계열(체결·OI)은 초당 들어와 1초면 창이 다시 차므로 이 문제가 없다 --
    청산만 디스크의 jsonl 을 되읽어야 하고, 안 읽어도 **에러가 없다**. 그래서 검사한다.
    """
    assert "def liq_events_load()" in SRV, "청산 복원 함수가 없다"
    assert re.search(r"async def collect_force_orders\(app: web\.Application\) -> None:\n"
                     r'\s+""".*?"""\n\s+liq_events_load\(\)', SRV, re.S), \
        "liq_events_load() 가 수집기 시작에서 안 불린다 -- 재시작 후 청산선이 빈다"
    assert "deque(fh, maxlen=liq_events.maxlen)" in SRV, \
        "복원이 deque 꼬리로 제한되지 않는다 -- 파일이 커지면 통째로 메모리에 올린다"


def test_quantity_not_usd():
    """이 차트의 다른 선(고래·리테일·OI)은 전부 ETH 단위다. USD 를 섞으면 축이 깨진다."""
    assert re.search(r'\+= float\(e\.get\("qty"\) or 0\.0\)', SRV), \
        "청산을 수량이 아니라 다른 값(USD 등)으로 집계한다 -- 같은 축에 못 얹는다"


# ── 두 판 배치 (2026-09-22) ──────────────────────────────────────────────────

def test_stack_is_an_identity_not_three_lines():
    """CVD = 고래 + 중형 + 리테일. 층 경계가 0 → 고래 → (CVD−리테일) → CVD 순이어야 한다.

    경계를 바꾸면 층이 서로를 파고들거나 뒤집히는데, path 는 그래도 그려진다.
    특히 stackMid 를 «cvd − whale» 로 잘못 쓰면 가운데 층이 리테일이 되어 **이름만 틀린**
    그림이 나온다 -- 눈으로는 구별이 안 된다.
    """
    assert re.search(r"const stackMid = cvd\.map\(\(r, i\) => \(\{ s: r\.s, v: r\.v - retail\[i\]\.v \}\)\)", JS), \
        "스택 가운데 경계가 «CVD − 리테일» 이 아니다 -- 중형 층이 리테일 층이 된다"
    order = re.findall(r'band\((\w+), (\w+), [\d.]+, "(\w+)"\)', JS)
    assert order == [("zeroRows", "whale", "고래"), ("whale", "stackMid", "중형"),
                     ("stackMid", "cvd", "리테일")], f"스택 순서가 바뀌었다: {order}"


def test_liquidation_is_not_on_the_shared_axis():
    """청산을 yF 로 그리면 0선에 붙어 사라진다 -- 2026-09-22 첫 판의 실제 버그다.

    봉당 중앙 6.8 ETH 는 CVD 진폭(7,548)의 0.09% 다. 아무 에러도 안 나고 선만 안 보인다.
    """
    assert not re.search(r"yF\(.{0,20}liq", JS), "청산이 다시 공유 ETH 축(yF)에 올라갔다"
    assert "Math.log10(1 + Math.abs(e.v))" in JS, \
        "청산 점 크기가 로그가 아니다 -- 건당 0.86~2,556 ETH(2,970배)라 선형이면 큰 것만 남는다"


def test_liquidation_dots_sit_on_the_zero_line():
    """청산 원은 0선 위다 -- 아래 판이 사라졌으므로(2026-09-22) 다른 자리에 두면 떠돈다.

    0선인 이유: 청산은 강제 유출입이라 이 축(순수급)의 원점에 앉는 게 맞고, 시각축이
    스택과 같아 «언제»가 바로 맞춰진다.
    """
    assert re.search(r"const cy = mid;", JS), "청산 원이 0선(mid)에 안 앉는다"
    assert "const LOW_H" not in JS and "const PANE_GAP" not in JS, \
        "아래 판 껍데기 상수가 남아 있다 -- 지웠으면 같이 지운다"


def test_one_pane_uses_the_whole_drawing_area():
    """그리기 영역(flowH) 전부를 누적 스택이 쓴다. SUB_1S_H(400) − mt(16) − mb(14) = 370.

    🔴이 산수가 어긋나면 SVG 는 잘라주지 않는다 -- 넘치면 옆 패널을 침범하고, 모자라면
      빈 띠가 생긴다. 캔들 상자 높이 계약(styles.css)과 이 파일이 갈라져 있어 한쪽만
      고치면 조용히 깨지므로 여기서 다시 센다.
    """
    mt = int(re.search(r"const mt = (\d+), mb = (\d+);", JS).group(1))
    mb = int(re.search(r"const mt = (\d+), mb = (\d+);", JS).group(2))
    sub_h = int(re.search(r"SUB_1S_H = subOn \? (\d+)", JS).group(1))
    flow_h = sub_h - mt - mb
    assert flow_h == 370, f"그리기 영역이 {flow_h}px 다(400-16-14=370 이어야)"
    assert re.search(r"const mid = flowTop \+ flowH / 2;", JS), "0선이 그리기 영역 한가운데가 아니다"
    assert re.search(r"const half = flowH / 2 - 4;", JS), "반폭이 flowH 기준이 아니다"
    body = JS[JS.index("function renderSupply1s"):JS.index("function renderSupplyProfileSvg")]
    assert "tMax" not in body and "barMax" not in body, \
        "거래대금 막대가 되살아났다 -- 2026-09-22 사용자 지시로 이 차트에서 뺐다"


def test_no_new_colour_was_invented():
    """3색 계약(초록·빨강·주황 + 중립). 층 구분은 색이 아니라 농담이어야 한다."""
    body = JS[JS.index("function renderSupply1s"):JS.index("function renderSupplyProfileSvg")]
    # var(--토큰, #폴백) 은 토큰 사용이다 -- 폴백만 빼고 센다.
    bare = re.sub(r"var\(--[\w-]+,\s*#[0-9a-fA-F]{3,8}\)", "", body)
    hexes = set(re.findall(r"#[0-9a-fA-F]{3,8}", bare))
    assert not hexes, f"1초 차트에 리터럴 색이 생겼다(토큰만 써야 한다): {hexes}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("all ok")
