"""텔레그램 알림 정답지(2026-11-21): 원본 `bracket_tick_alert`(dashboard/orders.py 안 닫힌 함수)와 `send_telegram`(scripts/ops_watchdog.py)을
**모듈을 import 하지 않고**(ops_watchdog 는 import 때 점-env 를 읽는다) 소스에서 함수만 꺼내 그대로 돌린다. urlopen 은 가짜(요청을 적기만) -- 네트워크 없음.
토큰·채팅 번호는 합성 값. 실행: TZ=UTC unshare -rn python3 -I rtorder/tools/gen_telegram_fixture.py
"""
import ast
import json
import os
import sys
import types
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rtorder" / "tests" / "fixtures" / "telegram.json"


def func(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.Module(body=[n], type_ignores=[])
    raise SystemExit(f"{name} 없음: {path}")


sent: list[dict] = []
mode = {"fail": False}


class Resp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b'{"ok":true}'


def fake_urlopen(req, timeout=None):
    sent.append({"url": req.full_url, "method": req.get_method(), "ctype": req.get_header("Content-type"), "body": req.data.decode("ascii"), "timeout": timeout})
    if mode["fail"]:
        raise OSError("가짜 실패")
    return Resp()


urllib.request.urlopen = fake_urlopen
ns_tg: dict = {"os": os, "json": json, "urllib": urllib}
exec(compile(func(ROOT / "scripts" / "ops_watchdog.py", "send_telegram"), "ops_watchdog.py", "exec"), ns_tg)
send_telegram = ns_tg["send_telegram"]

# bracket_tick_alert 안의 `from scripts.ops_watchdog import send_telegram` 이 진짜 모듈(점-env 읽기)을 부르지 않게 가짜 모듈을 넣는다
fake_mod = types.ModuleType("scripts.ops_watchdog")
fake_mod.send_telegram = send_telegram
sys.modules["scripts"] = types.ModuleType("scripts")
sys.modules["scripts.ops_watchdog"] = fake_mod


class Task:
    def add_done_callback(self, cb):
        pass


returns: list = []


class FakeAsyncio:
    @staticmethod
    def to_thread(f, *a):
        return (f, a)

    @staticmethod
    def create_task(job):
        f, a = job
        returns.append(f(*a))  # 원본은 뒤에서 -- 여기서는 곧바로(보낸 바이트만 본다)
        return Task()


printed: list[str] = []
ns_al: dict = {"bracket_tick_alerted": set(), "alert_tasks": set(), "asyncio": FakeAsyncio, "Exception": Exception,
               "print": lambda *a, **k: printed.append(" ".join(map(str, a)))}
exec(compile(func(ROOT / "dashboard" / "orders.py", "bracket_tick_alert"), "orders.py", "exec"), ns_al)
alert = ns_al["bracket_tick_alert"]

os.environ.pop("TELEGRAM_BOT_TOKEN", None)
os.environ.pop("TELEGRAM_CHAT_ID", None)
cases = []
# 설정 없음 = None(보내지 않음)
assert send_telegram("x") is None and not sent
os.environ["TELEGRAM_BOT_TOKEN"] = "123456:test_token"
os.environ["TELEGRAM_CHAT_ID"] = "-100987"
calls = [("ETHUSDC:LONG", KeyError("qty")), ("ETHUSDC:LONG", KeyError("other")), ("ETHUSDC:LONG", ValueError("bad <x> & 'y' \"z\"")),
         ("SOXLUSDT:SHORT", KeyError("qty")), ("XRPUSDC:LONG", RuntimeError("줄\n바꿈\t탭 \x7f 끝 🔴")), ("SOXLUSDT:SHORT", TypeError("unsupported operand"))]
for key, exc in calls:
    n0, p0 = len(sent), len(printed)
    alert(key, exc)
    cases.append({"key": key, "why": f"{type(exc).__name__}: {exc}", "printed": printed[p0:], "sent": sent[n0:]})
mode["fail"] = True
assert send_telegram("실패") is False
fail = sent[-1]
OUT.write_text(json.dumps({"cases": cases, "returns": returns, "fail_request": fail}, ensure_ascii=False, indent=1))
print(f"호출 {len(calls)} · 보냄 {sum(len(c['sent']) for c in cases)} · {OUT}")
