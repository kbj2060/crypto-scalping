"""ops_watchdog.check_data_sources (2026-10-12 복구) -- 미시구조 연결 칸(microstructure_1m 최신 행) +
청산 스트림(liq_burst_state.json valid_liq_stream) 판정이 옛 봇 state 규칙과 같은가."""
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ops_watchdog as w  # noqa: E402


def _case(depth=True, trade=True, poll=True, stale=False, trade_age=1.0, ws=True, valid=True, liq_age_min=0.1):
    d = Path(tempfile.mkdtemp())
    w.LIVE = d
    w._DUCKDB_CHECK_CACHE.clear()
    now = dt.datetime.now(dt.timezone.utc)
    (d / "liq_burst_state.json").write_text(json.dumps({
        "updated_at": (now - dt.timedelta(minutes=liq_age_min)).isoformat().replace("+00:00", "Z"),
        "ws_connected": ws, "valid_liq_stream": valid}))
    db = d / "microstructure.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create table microstructure_1m(ts timestamp, depth_connected boolean, trade_connected boolean, "
                "poll_connected boolean, data_stale boolean, trade_age_sec double)")
    con.execute("insert into microstructure_1m values (now(), ?, ?, ?, ?, ?)", [depth, trade, poll, stale, trade_age])
    con.close()
    return w.check_data_sources(db).status


def test_data_sources_rules():
    assert _case() == "OK"
    assert _case(trade=False) == "WARN"                       # 체결 스트림만 끊김
    assert _case(valid=False) == "WARN"                       # 청산 스트림 실메시지 없음(핸드셰이크만)
    assert _case(liq_age_min=5) == "WARN"                     # 요격기가 상태 파일을 안 씀
    assert _case(depth=False) == "CRITICAL"                   # 핵심(호가) 끊김
    assert _case(trade=False, valid=False) == "CRITICAL"      # 두 개 이상
    assert _case(stale=True, trade_age=30) == "WARN"
    assert _case(stale=True, trade_age=900) == "CRITICAL"


def test_missing_liq_state_blocks():
    d = Path(tempfile.mkdtemp())
    w.LIVE = d
    w._DUCKDB_CHECK_CACHE.clear()
    assert w.check_data_sources(d / "none.duckdb").status == "BLOCKED"
