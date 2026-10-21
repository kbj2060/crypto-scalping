"""read_rows(.sqlite) 연결 묶음 (2026-10-22 재부팅 뒤 대시보드: 무거운 쿼리 하나가 파일의 유일한 연결을 5~10분 쥐어 가벼운 읽기가 전부 멈춤).

(a) 한 스레드가 느린 쿼리(수 초)를 도는 동안 다른 스레드의 가벼운 쿼리는 바로 끝난다(옛 단일 연결이면 느린 쿼리 뒤에 섰다)
(b) 연결은 호출마다 새로 열지 않는다(10-01 사고: connect/close 에서 수 분 멈춤)
(c) 깨진 쿼리 뒤에도 다음 읽기가 되고 칸이 새지 않는다
실행: python -m pytest -q test/test_data_store_ro_pool_20261022.py
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import data_store  # noqa: E402

SLOW = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 30000000) SELECT count(*) FROM c"


class RoPoolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.sqlite"
        con = sqlite3.connect(self.db)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE t (a INTEGER)")
        con.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(100)])
        con.commit(); con.close()
        data_store._RO.clear()

    def tearDown(self) -> None:
        data_store._RO.clear()
        self.tmp.cleanup()

    def test_light_query_not_blocked_by_heavy(self) -> None:
        heavy_done = threading.Event()
        threading.Thread(target=lambda: (data_store.read_rows(self.db, SLOW), heavy_done.set()), daemon=True).start()
        time.sleep(0.3)                                      # 무거운 쿼리가 연결을 잡았다
        self.assertFalse(heavy_done.is_set(), "느린 쿼리가 너무 빨리 끝났다 -- 시험이 아무것도 재지 않는다")
        t0 = time.monotonic()
        self.assertEqual(data_store.read_rows(self.db, "SELECT count(*) FROM t"), [(100,)])
        self.assertLess(time.monotonic() - t0, 0.5)
        self.assertFalse(heavy_done.is_set(), "가벼운 쿼리가 무거운 쿼리 뒤에 섰다")
        heavy_done.wait(60)

    def test_connections_reused_not_reopened(self) -> None:
        real = sqlite3.connect
        with mock.patch.object(sqlite3, "connect", side_effect=real) as spy:
            for _ in range(50):
                data_store.read_rows(self.db, "SELECT 1")
        self.assertLessEqual(spy.call_count, data_store._RO_POOL)

    def test_error_then_recovers(self) -> None:
        with self.assertRaises(sqlite3.Error):
            data_store.read_rows(self.db, "SELECT * FROM no_such_table")
        self.assertEqual(data_store.read_rows(self.db, "SELECT count(*) FROM t"), [(100,)])
        self.assertEqual(data_store._RO[str(self.db)].qsize(), data_store._RO_POOL)   # 칸이 새지 않았다


if __name__ == "__main__":
    unittest.main()
