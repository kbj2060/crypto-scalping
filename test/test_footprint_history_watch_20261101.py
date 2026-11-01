"""풋프린트 과거 봉 감시 (2026-11-01).   python -m pytest -q test/test_footprint_history_watch_20261101.py
봉이 거의 안 오면 WARN/CRITICAL, 대시보드 재기동 직후 복원 유예 중이면 OK, 엔드포인트를 못 읽으면 BLOCKED."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.ops_watchdog as w  # noqa: E402


def _st(bars, started=None):
    return {c.component: c.status for c in w.check_footprint_history(lambda a: bars[a], started_at=started if started else 0.0)}


def test_status_by_bar_count():
    got = _st({"eth": 12, "btc": 8, "sol": 5, "xrp": 3})
    assert got == {"footprint_history_eth": "OK", "footprint_history_btc": "OK",
                   "footprint_history_sol": "WARN", "footprint_history_xrp": "CRITICAL"}


def test_restore_grace_after_dashboard_restart():
    assert set(_st({a: 3 for a in w.FOOTPRINT_ASSETS}, started=time.time() - 600).values()) == {"OK"}


def test_unreadable_endpoint_is_blocked():
    def boom(a):
        raise OSError("down")
    assert {c.status for c in w.check_footprint_history(boom, started_at=0.0)} == {"BLOCKED"}


def test_dashboard_started_at_reads_last_start(tmp_path):
    log = tmp_path / "d.log"
    log.write_text("[2026-11-01T20:55:10+09:00] dashboard server starting host=x port=1\nnoise\n"
                   "[2026-11-01T22:24:06+09:00] dashboard server starting host=x port=1\n")
    assert w.dashboard_started_at(log) == w.datetime.fromisoformat("2026-11-01T22:24:06+09:00").timestamp()
    assert w.dashboard_started_at(tmp_path / "없음.log") is None
