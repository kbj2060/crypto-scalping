"""기준선 읽기의 잠금 재시도 (2026-09-26).   python -m pytest -q test/test_micro_ref_lock_retry_20260926.py

수집기가 trade_tape.duckdb 잠금을 ~0.5초(최대 1.1초) 쥐는 동안 read_only 연결도 거부된다.
다른 프로세스가 실제로 잠금을 쥔 상태를 만들어 재시도가 기다렸다 여는지, 재시도 0회면 실패하는지(대조군) 본다.
"""
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.micro_ref import _connect_read_only_retry  # noqa: E402


def _hold(db: Path, seconds: float) -> subprocess.Popen:
    """쓰기 연결로 잠금을 쥐는 별도 프로세스(수집기 흉내)."""
    p = subprocess.Popen([sys.executable, "-c",
                          "import duckdb,sys,time; c=duckdb.connect(sys.argv[1]);"
                          "print('held', flush=True); time.sleep(float(sys.argv[2])); c.close()",
                          str(db), str(seconds)], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    return p


def test_waits_out_a_writer_holding_the_lock(tmp_path):
    db = tmp_path / "t.duckdb"
    duckdb.connect(str(db)).close()
    p = _hold(db, 1.0)
    try:
        with pytest.raises(duckdb.IOException):          # 대조군: 재시도 없으면 잠금에 걸린다
            _connect_read_only_retry(db, retries=0)
        t0 = time.monotonic()
        _connect_read_only_retry(db, retries=4, wait_s=0.5).close()
        assert time.monotonic() - t0 >= 0.4, "잠금이 풀릴 때까지 기다려야 한다"
    finally:
        p.wait()


def test_non_lock_error_is_not_retried(tmp_path):
    t0 = time.monotonic()
    with pytest.raises(duckdb.Error):
        _connect_read_only_retry(tmp_path / "missing.duckdb", retries=4, wait_s=1.0)
    assert time.monotonic() - t0 < 0.5, "파일 없음 같은 오류는 곧바로 올린다"
