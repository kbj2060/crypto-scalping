"""data_store.read_rows(.sqlite) 를 여러 스레드가 동시에 불러도 안 멈춘다 (2026-10-01).

사고: 대시보드가 같은 hot WAL 파일을 호출마다 열고 닫자(4d 의 OI 꼬리 읽기 3개 × 0.5초) 스레드들이
`sqlite3.connect` / `close` 안에서 서로 기다렸다 -- faulthandler 스택이 전부 data_store.py 의 connect·close
줄이었다. 시장 맥락 TimeoutError · OI 링 · 상황 계산이 함께 멈췄다.
🔴이 합성 시험은 그 멈춤을 **재현하지 못했다**(옛 코드도 서버·dev 에서 통과). 지키는 것은 고친 방식(파일당 연결 하나를
  스레드끼리 공유)이 다른 프로세스가 쓰는 중에도 굶김·멈춤 없이 읽는가다 -- 파이썬 잠금을 따로 건 판은 3.5초 굶어 여기서 실패했다.
실행: python test/test_read_rows_sqlite_threads_20261001.py
"""
import multiprocessing as mp
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import data_store as ds  # noqa: E402


def _writer(path: str, stop) -> None:
    i = 0
    while not stop.is_set():                      # 수집기처럼: 쓸 때마다 열고 닫는다(마지막 close 가 WAL 을 정리)
        con = sqlite3.connect(path, timeout=30)
        con.executemany("INSERT INTO t VALUES (?, ?)", [(i * 100 + k, 1.0) for k in range(100)])
        con.commit()
        con.close()
        i += 1
        time.sleep(0.01)


def test_many_threads_do_not_stall(seconds: float = 8.0, threads: int = 12) -> None:
    assert sqlite3.threadsafety == 3, "read_rows 는 연결 하나를 스레드끼리 같이 쓴다 -- 직렬화(serialized) 빌드여야 안전"
    path = Path(tempfile.mkdtemp()) / "hot.sqlite"
    ds.sqlite_init(path)
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE t(ts INTEGER PRIMARY KEY, v REAL)")
    stop = mp.Event()
    w = mp.Process(target=_writer, args=(str(path), stop))
    w.start()
    worst = [0.0] * threads
    n = [0] * threads
    end = time.time() + seconds

    def reader(k: int) -> None:
        while time.time() < end:
            t0 = time.time()
            ds.read_rows(path, "SELECT count(*), max(ts) FROM t WHERE ts > ?", [0])
            worst[k] = max(worst[k], time.time() - t0)
            n[k] += 1

    ts = [threading.Thread(target=reader, args=(k,), daemon=True) for k in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(seconds + 30)
    stop.set()
    w.join(10)
    stuck = sum(t.is_alive() for t in ts)
    assert not stuck and max(worst) < 2.0, f"멈춤 {stuck}/{threads} · 가장 긴 읽기 {max(worst):.1f}초 (읽기 {sum(n)}회)"
    print(f"ok -- {threads} 스레드 {sum(n)}회 읽기, 가장 긴 읽기 {max(worst):.3f}초")


if __name__ == "__main__":
    test_many_threads_do_not_stall()
