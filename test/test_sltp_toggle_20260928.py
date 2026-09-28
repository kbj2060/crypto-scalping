"""SL/TP 체크 해제(2026-09-28 사용자 지시) -- 진입에 SL/TP 를 걸지도, 기존 것을 갱신하지도 않는다.

실주문 경로라 소스 계약으로 묶는다(주석에 속지 않게 주석을 지운 코드로 본다).
  python3 test/test_sltp_toggle_20260928.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
code = lambda p, c: "\n".join(l for l in (ROOT / p).read_text(encoding="utf-8").splitlines() if not l.strip().startswith(c))
SRV, JS = code("dashboard/server.py", "#"), code("dashboard/live/app.js", "//")


def fn(src, name):
    i = src.index(f"async def {name}(")
    j = re.search(r"\n    (async )?def ", src[i + 10:])
    return src[i:i + 10 + j.start()]


def test_disabled_bracket_touches_neither_orders_nor_watch_file():
    body = fn(SRV, "entry_then_bracket")
    d = body.index('if b.get("disabled"):')
    assert d < body.index("place_bracket(") and d < body.index("bracket_save("), "끔인데 주문/감시 파일을 먼저 건드린다"
    assert "return" in body[d:body.index("place_bracket(")], "끔 분기가 돌아가지 않는다"


def test_preview_and_submit_honour_sltp_off():
    for name in ("api_manual_entry_preview", "api_manual_entry_submit"):
        assert 'if sltp_off(request):\n            plan["bracket"] = dict(SLTP_OFF_BRACKET)' in fn(SRV, name), name
    assert "sltp_off(" not in fn(SRV, "api_manual_exit_submit"), "청산에는 SL/TP 토글이 없다"
    assert '"disabled": True' in SRV


def test_client_sends_sltp_zero_only_when_unchecked():
    assert '(manualSltpOn() ? "" : "&sltp=0")' in JS
    assert '(pending.sltp === false ? "&sltp=0" : "")' in JS
    assert "sltp: manualSltpOn()" in JS


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print("ok ", n)
