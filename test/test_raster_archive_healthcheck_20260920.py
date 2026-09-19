"""ops_watchdog 의 raster_archive 점검 -- 네 상태를 다 재현해 판정을 확인한다.

🔴가장 중요한 건 «첫 보관 전에는 조용해야» 한다는 것이다. 파일이 없다고 경보를 울리면
09-28 까지 거짓 경보가 이어지고, 그러면 아무도 감시표를 안 보게 된다.
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def _mod(tmp: Path):
    for m in list(sys.modules):
        if m.startswith("ops_watchdog"):
            del sys.modules[m]
    import ops_watchdog as W
    W.LIVE = tmp
    return W


def _mk(tmp: Path, ages_days, parquet=0, gz=0):
    live = tmp / "orderflow" / "raster" / "ETHUSDT"
    live.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    for a in ages_days:
        (live / ((now - timedelta(days=a)).strftime("%Y-%m-%dT%H") + ".f32")).write_bytes(b"x")
    arc = tmp / "orderflow" / "raster_archive" / "ETHUSDT" / "2026-09-14"
    if parquet or gz:
        arc.mkdir(parents=True, exist_ok=True)
    for i in range(parquet):
        (arc / f"f{i}.f32.parquet").write_bytes(b"x")
    for i in range(gz):
        (arc / f"f{i}.f32.gz").write_bytes(b"x")


def _run(tmp):
    return _mod(tmp).check_raster_archive()


def test_pending_before_first_archive_is_quiet():
    """🔴핵심. 6일치만 쌓였고 보관본 0 -- 아직 만료 전이니 **OK** 여야 한다."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); _mk(tmp, [6, 5, 4, 3, 2, 1, 0])
        c = _run(tmp)
        assert c.status == "OK", (c.status, c.summary)
        assert "pending" in c.summary, c.summary
        assert c.details["parquet_files"] == 0
        assert c.details["first_archive_due"], "예정일을 안 적으면 언제 볼지 모른다"


def test_healthy_after_archiving():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); _mk(tmp, [13, 7, 1], parquet=3)
        c = _run(tmp)
        assert c.status == "OK" and "healthy" in c.summary, (c.status, c.summary)


def test_warns_when_expired_files_pile_up():
    """보존 14일인데 20일짜리가 살아 있다 -- 아카이버가 안 돈다."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); _mk(tmp, [20, 10, 1], parquet=1)
        c = _run(tmp)
        assert c.status == "WARN", c
        assert c.details["oldest_live_age_days"] >= 20


def test_warns_on_gzip_fallback():
    """gz 가 섞였다 = parquet 변환이 실패해 폴백했다. 파일은 있지만 duckdb 로 못 읽는다."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); _mk(tmp, [13, 1], parquet=2, gz=1)
        c = _run(tmp)
        assert c.status == "WARN" and "gzip" in c.summary, c
        assert c.details["gzip_fallback_files"] == 1


def test_no_live_files_is_not_an_alarm():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); (tmp / "orderflow" / "raster").mkdir(parents=True)
        c = _run(tmp)
        assert c.status == "OK", c


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print(f"ok  {n}")
    print("all ok")
