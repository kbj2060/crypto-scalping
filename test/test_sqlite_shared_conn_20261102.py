import sqlite3, sys, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_trade_tape_collector_20260916 as m


def test_one_connection_serialised_and_dropped_on_error(tmp_path):
    db = tmp_path / "t.sqlite"
    with m.sqlite_shared(db) as c1:
        c1.execute("CREATE TABLE x(v INTEGER)")
    seen = []

    def w(i):
        with m.sqlite_shared(db) as c:
            seen.append(c)
            c.execute("BEGIN"); c.execute("INSERT INTO x VALUES (?)", [i]); c.execute("COMMIT")
    ts = [threading.Thread(target=w, args=(i,)) for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert all(c is c1 for c in seen), "스레드가 달라도 같은 연결을 써야 한다"
    with m.sqlite_shared(db) as c:                       # BEGIN 만 하고 빠져도 다음 사용자에게 새지 않는다
        c.execute("BEGIN"); c.execute("INSERT INTO x VALUES (99)")
    with m.sqlite_shared(db) as c:
        assert c.execute("SELECT count(*) FROM x").fetchone()[0] == 8
    try:
        with m.sqlite_shared(db) as c:
            raise sqlite3.OperationalError("disk I/O error")
    except sqlite3.OperationalError:
        pass
    with m.sqlite_shared(db) as c2:
        assert c2 is not c1 and c2.execute("SELECT count(*) FROM x").fetchone()[0] == 8
    r = sqlite3.connect(db)                              # 읽는 쪽(대시보드)은 붙들린 연결과 상관없이 읽는다
    assert r.execute("SELECT count(*) FROM x").fetchone()[0] == 8
