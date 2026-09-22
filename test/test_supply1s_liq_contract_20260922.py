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
    assert re.search(r"cell\[0 if long_ else 1\] \+= float\(e\.get\(\"qty\"\) or 0\.0\)", SRV), \
        "청산 수량 칸의 롱/숏 배치가 바뀌었다 -- 클라의 c[1] - c[0] 가 뒤집힌다"
    assert re.search(r"cell\[2 if long_ else 3\] \+= float\(e\.get\(\"usd\"\) or 0\.0\)", SRV), \
        "청산 금액 칸의 롱/숏 배치가 수량 칸과 다르다 -- 꼬리표의 롱/숏이 뒤집힌다"
    assert re.search(r"liq_rows = \[\[s, round\(v\[0\], 3\), round\(v\[1\], 3\), round\(v\[2\]\), round\(v\[3\]\)\]", SRV), \
        "liq_rows 가 [초, 롱수량, 숏수량, 롱USD, 숏USD] 순서가 아니다"
    assert '"liq": liq_rows' in SRV, "payload 에 liq 가 없다"


def test_client_reads_same_order_and_sign():
    """클라는 [롱, 숏] 로 받아 «숏 − 롱» 을 그린다."""
    assert re.search(r"liq1s\.set\(r\[0\], \[r\[1\], r\[2\], r\[3\], r\[4\]\]\)", JS), \
        "클라가 liq 행을 [롱수량, 숏수량, 롱USD, 숏USD] 로 안 읽는다"
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


def test_size_is_quantity_label_is_money():
    """점 **크기**는 수량(ETH), 꼬리표는 **금액**(USD)이다 -- 둘을 바꾸면 조용히 틀린다.

    크기가 ETH 인 이유: 이 차트의 축이 ETH 이고 다른 선(고래·리테일·OI)도 전부 ETH 다.
    꼬리표가 USD 인 이유: 같은 카드 아래 5분봉 청산이 USD 라 단위가 섞이면 안 된다
    (2026-09-22 사용자 «단위나 금액으로 맞춰줘»).
    """
    assert re.search(r"Math\.log10\(1 \+ Math\.abs\(e\.v\)\)", JS), \
        "점 크기가 수량 기반 로그가 아니다"
    assert "fmtUsdCompact(liqSum[2])" in JS and "fmtUsdCompact(liqSum[3])" in JS, \
        "청산 꼬리표가 금액(USD)이 아니다 -- 아래 5분봉 청산과 단위가 갈린다"


def test_liq_labels_match_the_cumulative_chart():
    """1초 차트와 5분봉 누적 CVD 행의 범례는 **같은 크기**여야 한다(2026-09-22 사용자 지시).

    같은 카드 안에서 역할이 같은데 19/15 와 12/11 로 갈려 있었다. 5분봉 쪽을 19 로 올릴
    수는 없다 -- 그 행의 꼬리표 자리가 mr 86px 인데 «CVD +12k» 가 19px 로 약 105px 다.
    """
    one = re.search(r"const LBL = narrow \? (\d+) : (\d+);", JS)
    one_head = re.search(r"const LBL_HEAD = narrow \? (\d+) : (\d+);", JS)
    five = re.search(r'font-size", mobileChart \? \(k === 0 \? (\d+) : (\d+)\) : \(k === 0 \? (\d+) : (\d+)\)', JS)
    assert one and one_head and five, "라벨 크기 선언을 못 찾았다"
    assert (int(one_head.group(2)), int(one.group(2))) == (int(five.group(3)), int(five.group(4))), \
        (f"데스크톱이 안 맞는다: 1초 {one_head.group(2)}/{one.group(2)} vs "
         f"5분봉 {five.group(3)}/{five.group(4)}")
    assert (int(one_head.group(1)), int(one.group(1))) == (int(five.group(1)), int(five.group(2))), \
        (f"모바일이 안 맞는다: 1초 {one_head.group(1)}/{one.group(1)} vs "
         f"5분봉 {five.group(1)}/{five.group(2)}")


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


def test_liquidation_dots_sit_on_the_oi_line():
    """청산 원은 **신규계약(OI) 선 위**다(2026-09-22 사용자 지시).

    0선에 일렬로 늘어놓던 때는 시각만 맞고 뜻이 안 붙었다. 청산은 포지션을 강제로 닫는
    사건이라 미결제약정을 줄인다 -- 그 선 위에 앉혀야 «이 청산이 OI 를 어디서 꺾었나»가
    같은 자리에서 읽힌다. OI 는 3~7초 갱신이라 그 초 이전의 마지막 관측을 쓴다.
    """
    assert re.search(r'c\.setAttribute\("cy", \(oiRows\.length \? yF\(oiAt\(e\.s\)\) : mid\)', JS), \
        "청산 원이 신규계약(OI) 선 위에 안 앉는다"
    assert "const LOW_H" not in JS and "const PANE_GAP" not in JS, \
        "아래 판 껍데기 상수가 남아 있다 -- 지웠으면 같이 지운다"


def test_one_pane_uses_the_whole_drawing_area():
    """그리기 영역(flowH) 전부를 누적 스택이 쓴다.

    데스크톱 SUB_1S_H(400) − mt(16) − mb(14) = 370.
    2026-09-22 모바일은 범례가 플롯 밖 **바닥 한 줄**로 내려가 mb 가 28 이다 → 356.
    그 14px 이 범례 줄 값이고, 그만큼만 선이 짧아져야 한다.

    🔴이 산수가 어긋나면 SVG 는 잘라주지 않는다 -- 넘치면 옆 패널을 침범하고, 모자라면
      빈 띠가 생긴다. 캔들 상자 높이 계약(styles.css)과 이 파일이 갈라져 있어 한쪽만
      고치면 조용히 깨지므로 여기서 다시 센다.
    """
    m = re.search(r"const mt = (\d+), mb = narrow \? (\d+) : (\d+);", JS)
    assert m, "mt/mb 선언 모양이 바뀌었다 -- 계약을 다시 세운다"
    mt, mb_narrow, mb_wide = (int(g) for g in m.groups())
    sub_h = int(re.search(r"SUB_1S_H = subOn \? (\d+)", JS).group(1))
    assert sub_h - mt - mb_wide == 370, \
        f"데스크톱 그리기 영역이 {sub_h - mt - mb_wide}px 다(400-16-14=370 이어야)"
    assert sub_h - mt - mb_narrow == 356, \
        f"모바일 그리기 영역이 {sub_h - mt - mb_narrow}px 다(400-16-28=356 이어야)"
    # 🔴narrow 는 mb 보다 **먼저** 선언돼야 한다. 순서가 뒤집히면 TDZ ReferenceError 로
    #   app.js 전체가 죽는데 node --check 는 문법이 옳아 못 잡는다(2026-09-22 실제 발생).
    assert JS.index("const narrow = w < 760;") < m.start(), \
        "narrow 선언이 mb 보다 뒤다 -- TDZ 로 app.js 가 통째로 죽는다"
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
