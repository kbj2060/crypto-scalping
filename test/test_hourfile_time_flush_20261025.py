"""HourFile 시간 flush(2026-10-25): 2,000행이 안 차도 30초 지나면 디스크에 닿는다. 실행: python -m pytest -q test/test_hourfile_time_flush_20261025.py"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.live_book_ticker_collector_20260914 as bt  # noqa: E402

TS = 1_761_300_000_000   # 아무 시각 -- 같은 시각 파일 안


def test_quiet_stream_reaches_disk_after_30s(tmp_path: Path) -> None:
    clock = [1000.0]
    real_open = open
    with mock.patch.object(bt.time, "monotonic", lambda: clock[0]), \
         mock.patch("builtins.open", lambda p, m="r", *a, **k: real_open(p, m, buffering=131072)):   # Pi 3.14 의 128KiB 버퍼
        hf = bt.HourFile(tmp_path, "HYPE")
        for i in range(10):
            hf.write(TS + i, 40.0, 1.0, 40.01, 1.0, 1)
        path = next((tmp_path / "HYPE").glob("*.bt"))
        assert path.stat().st_size == 0, "30초 전 -- 아직 버퍼 안"
        clock[0] += 31
        hf.write(TS + 10, 40.0, 1.0, 40.01, 1.0, 1)
        assert path.stat().st_size == bt.HDR.size + 11 * bt.ROW.size
        hf.close()
