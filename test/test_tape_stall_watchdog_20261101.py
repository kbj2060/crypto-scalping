"""체결 테이프 정지 감시 (2026-11-01).   python -m pytest -q test/test_tape_stall_watchdog_20261101.py
이벤트 루프가 동기 호출에 막히면 스택을 남기고 종료(코드 70)해야 supervisor 가 재기동한다. 정상이면 건드리지 않는다."""
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(body: str, timeout: float):
    head = (f"import asyncio, sys, time\nsys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
            "import live_trade_tape_collector_20260916 as m\nm.WD_LOOP_STALL_S, m.WD_WRITE_STALL_S = 2.0, 600.0\n")
    code = head + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)


def test_blocked_loop_exits_with_stack():
    r = _run("""
async def stuck():
    await asyncio.sleep(0.1)
    time.sleep(60)            # 동기 호출에 루프가 막힌 상황
asyncio.run(m.run_all([stuck()]))""", timeout=50)
    assert r.returncode == 70, (r.returncode, r.stderr[-400:])
    assert "정지 감지" in r.stdout + r.stderr and "stuck" in r.stderr, "스택 덤프에 멈춘 지점이 없다"


def test_healthy_loop_is_left_alone():
    r = _run("""
m.WD_LOOP_STALL_S = 8.0       # 박동 주기(5초)보다 길게 -- 운영값은 120초
async def fine():
    await asyncio.sleep(14)    # 감시 주기(10초)를 한 번 넘겨 산다
asyncio.run(m.run_all([fine()]))""", timeout=50)
    assert r.returncode == 0, (r.returncode, r.stderr[-400:])
