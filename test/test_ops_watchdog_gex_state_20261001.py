"""ops_watchdog -- ① check_gex_state: 옵션 수집기 요약 JSON 신선도(옛 deribit_gex.duckdb 거짓 CRITICAL 대체)
② 락 충돌 시 파일 mtime 으로 신선도(writer 가 쥐고 있으면 살아 있다는 뜻) ③ hot SQLite 신선도(심볼별, KST)."""
import datetime as dt
import json
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ops_watchdog as w  # noqa: E402


def test_gex_state_freshness():
    d = Path(tempfile.mkdtemp())
    w.LIVE = d
    assert w.check_gex_state().status == "BLOCKED"
    now = dt.datetime.now(dt.timezone.utc)
    for minutes, expected in [(5, "OK"), (45, "WARN"), (3281, "CRITICAL")]:
        (d / "deribit_gex_state.json").write_text(json.dumps({"generated_at": (now - dt.timedelta(minutes=minutes)).isoformat()}))
        assert w.check_gex_state().status == expected, minutes


def test_lock_conflict_uses_mtime():
    db = Path(tempfile.mkdtemp()) / "t.duckdb"
    holder = subprocess.Popen([sys.executable, "-c",
                               "import duckdb,time,sys; c=duckdb.connect(sys.argv[1]); "
                               "c.execute('create table t(ts timestamp)'); c.execute('insert into t values (now())'); "
                               "print('ready', flush=True); time.sleep(20)", str(db)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        c = w._check_duckdb_table_freshness_uncached("x", db, "t", "ts", 5, 10)
        assert c.details.get("lock_conflict") is True and c.status == "OK", c
        wal = db.with_name(db.name + ".wal")
        assert wal.exists(), "holder 의 insert 는 체크포인트 전이라 .wal 에 있다"
        os.utime(db, (time.time() - 3600,) * 2)               # 본체만 낡음 = 쓰는 중(10-01 리뷰: .wal 을 안 보면 거짓 CRITICAL)
        c = w._check_duckdb_table_freshness_uncached("x", db, "t", "ts", 5, 10)
        assert c.status == "OK", c
        os.utime(wal, (time.time() - 3600,) * 2)              # 둘 다 낡음 = 진짜 정지
        c = w._check_duckdb_table_freshness_uncached("x", db, "t", "ts", 5, 10)
        assert c.status == "CRITICAL", c
    finally:
        holder.kill()


def test_hot_sqlite_freshness():
    import sqlite3
    db = Path(tempfile.mkdtemp()) / "binance_tape.sqlite"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE trade_tape_1s(symbol TEXT, ts_sec INTEGER)")
    c.executemany("INSERT INTO trade_tape_1s VALUES (?, ?)", [("ethusdt", int(time.time()) - 30),
                                                             ("btcusdt", int(time.time()) - 3600)])
    c.commit()
    c.close()
    ts = "datetime(max(ts_sec), 'unixepoch', 'localtime')"
    ok = w._check_duckdb_table_freshness_uncached("x", db, "trade_tape_1s WHERE symbol = 'ethusdt'", ts, 5, 10)
    assert ok.status == "OK" and ok.details["age_minutes"] < 2, ok                      # localtime = KST 로 읽힌다
    assert w._check_duckdb_table_freshness_uncached("x", db, "trade_tape_1s WHERE symbol = 'btcusdt'", ts, 5, 10).status == "CRITICAL"
    assert w._check_duckdb_table_freshness_uncached("x", db, "trade_tape_1s WHERE symbol = 'solusdt'", ts, 5, 10).status == "BLOCKED"


if __name__ == "__main__":
    test_hot_sqlite_freshness()
    test_gex_state_freshness()
    test_lock_conflict_uses_mtime()
    print("ok")
