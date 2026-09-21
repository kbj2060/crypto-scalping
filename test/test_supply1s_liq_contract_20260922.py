"""/api/supply-1s 의 청산 계열은 **서버와 클라가 부호 규약을 공유**해야 한다.

서버가 [초, 롱청산, 숏청산] 로 보내는데 클라가 [숏, 롱] 으로 읽으면 선이 그대로 뒤집힌다 --
에러도, 빈 화면도 없이 **반대 방향이 맞는 것처럼** 그려진다. 그게 이 검사의 이유다.

규약:
  롱 청산 = 시장에 강제 SELL -> 이 차트의 아래쪽(음수)
  숏 청산 = 시장에 강제 BUY  -> 위쪽(양수)
  따라서 누적선의 값은 «숏 − 롱» 이다.
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
    assert re.search(r"liq1s\.get\(s\); return c \? c\[1\] - c\[0\] : 0", JS), \
        "누적 부호가 «숏 − 롱» 이 아니다 -- 청산선이 뒤집힌다"


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("all ok")
