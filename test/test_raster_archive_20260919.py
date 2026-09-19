"""래스터 만료 보관의 «원본을 언제 지우는가»만 잡는다. 데이터 유실 경로라 게으르면 안 된다.

① 정상: parquet 이 생기고 원본이 사라진다 · **값이 왕복한다**
② 멱등: 온전한 gz 가 이미 있으면 재압축 없이 원본만 지운다
③ 🔴잘린 보관본(앞선 실행이 쓰다 죽음)을 «보관됨»으로 오인해 원본을 지우면 안 된다
④ 압축 실패면 원본을 **유지**한다
"""
import gzip
import os
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ.setdefault("OF_ARCHIVE", "1")


def _mod(tmp: Path):
    os.environ["OF_ARCHIVE_ROOT"] = str(tmp / "archive")
    for m in list(sys.modules):
        if "live_orderflow_raster_collector" in m:
            del sys.modules[m]
    import live_orderflow_raster_collector_20260914 as M
    return M


def _src(tmp: Path, name="2026-09-01T03.f32", nrows=5, nbins=4) -> Path:
    """진짜 .f32 를 만든다 -- 아무 바이트나 주면 parquet 변환이 헤더에서 튕겨 gzip 폴백으로
    새고, 그러면 이 검사가 **정작 볼 것을 안 본다**."""
    hdr = struct.Struct("<4sHHfIqII").pack(b"FLWR", 1, nbins, 0.5, 0, 1_700_000_000_000, 1000, 0)
    rows = b""
    for i in range(nrows):
        q = [0.0] * nbins
        q[i % nbins] = float(i + 1)                       # 행마다 한 칸만 0 아님
        rows += struct.Struct("<qif").pack(1_700_000_000_000 + i * 1000, 100, 2000.0 + i)
        rows += struct.pack("<%df" % nbins, *q)
    p = tmp / name
    p.write_bytes(hdr + rows)
    return p


def _pq(tmp: Path) -> Path:
    return tmp / "archive" / "ETHUSDT" / "2026-09-01" / "2026-09-01T03.f32.parquet"


def test_archives_then_deletes_and_roundtrips():
    import pyarrow.parquet as pq
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); M = _mod(tmp)
        p = _src(tmp, nrows=5, nbins=4)
        M._archive_then_unlink(p, "ethusdt")
        dst = _pq(tmp)
        assert dst.exists(), "parquet 이 없다"
        assert not p.exists(), "원본이 안 지워졌다"
        t = pq.read_table(dst).to_pydict()
        assert len(t["qty"]) == 5, f"0 아닌 칸 5개여야 하는데 {len(t['qty'])}"
        assert sorted(t["qty"]) == [1.0, 2.0, 3.0, 4.0, 5.0], t["qty"]
        assert sorted(t["bin"]) == [100, 100, 101, 102, 103], "bin = bin_lo + 열"
        mid = pq.read_table(dst.with_name(dst.stem + "_mid.parquet")).to_pydict()
        assert mid["mid"] == [2000.0, 2001.0, 2002.0, 2003.0, 2004.0], mid["mid"]
        assert not list(dst.parent.glob("*.part")), ".part 가 남았다"


def test_duckdb_view_queries_the_parquet():
    """«duckdb 에 저장» 의 실체 -- 복사가 아니라 뷰다. SELECT 가 실제로 되는지 본다."""
    import duckdb
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); M = _mod(tmp)
        M._archive_then_unlink(_src(tmp), "ethusdt")
        con = duckdb.connect(str(M.OF_ARCHIVE_DB))
        try:
            n, = con.execute("SELECT count(*) FROM book").fetchone()
            s, = con.execute("SELECT sum(qty) FROM book").fetchone()
            m, = con.execute("SELECT count(*) FROM book_mid").fetchone()
        finally:
            con.close()
        assert n == 5 and s == 15.0, (n, s)
        assert m == 5, m


def test_idempotent_when_archive_already_complete():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); M = _mod(tmp)
        p = _src(tmp)
        M._archive_then_unlink(p, "ethusdt")
        dst = _pq(tmp)
        before = dst.stat().st_mtime_ns
        p2 = _src(tmp)                       # 같은 이름이 다시 생긴 상황
        M._archive_then_unlink(p2, "ethusdt")
        assert not p2.exists(), "원본이 안 지워졌다"
        assert dst.stat().st_mtime_ns == before, "다시 썼다(멱등 아님)"


def test_truncated_archive_is_not_mistaken_for_complete():
    """🔴핵심. .part -> replace 를 안 쓰면 여기서 원본이 영원히 사라진다."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); M = _mod(tmp)
        dst = _pq(tmp)
        dst.parent.mkdir(parents=True)
        dst.write_bytes(b"")                 # 크기 0 = 쓰다 죽은 흔적
        p = _src(tmp)
        M._archive_then_unlink(p, "ethusdt")
        assert dst.stat().st_size > 0, "빈 보관본을 그대로 뒀다"
        assert not p.exists()
        import pyarrow.parquet as pq
        assert len(pq.read_table(dst).to_pydict()["qty"]) == 5


def test_keeps_original_when_compression_fails():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); M = _mod(tmp)
        p = _src(tmp)
        M.OF_ARCHIVE_ROOT = Path("/proc/nonexistent-cannot-mkdir")   # mkdir 이 실패한다
        M._archive_then_unlink(p, "ethusdt")
        assert p.exists(), "압축 실패인데 원본을 지웠다"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("all ok")
