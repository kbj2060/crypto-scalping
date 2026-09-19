"""래스터 만료 보관의 «원본을 언제 지우는가»만 잡는다. 데이터 유실 경로라 게으르면 안 된다.

① 정상: gz 가 생기고 원본이 사라진다 · 내용이 왕복한다
② 멱등: 온전한 gz 가 이미 있으면 재압축 없이 원본만 지운다
③ 🔴잘린 gz(앞선 실행이 쓰다 죽음)를 «보관됨»으로 오인해 원본을 지우면 안 된다
④ 압축 실패면 원본을 **유지**한다
"""
import gzip
import os
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


def _src(tmp: Path, name="2026-09-01T03.f32", payload=b"RASTER" * 4096) -> Path:
    p = tmp / name
    p.write_bytes(payload)
    return p


def test_archives_then_deletes_and_roundtrips():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); M = _mod(tmp)
        payload = b"FLWR" + bytes(9999)
        p = _src(tmp, payload=payload)
        M._archive_then_unlink(p, "ethusdt")
        gz = tmp / "archive" / "ETHUSDT" / "2026-09-01" / "2026-09-01T03.f32.gz"
        assert gz.exists(), "gz 가 없다"
        assert not p.exists(), "원본이 안 지워졌다"
        assert gzip.decompress(gz.read_bytes()) == payload, "내용이 깨졌다"
        assert not list(gz.parent.glob("*.part")), ".part 가 남았다"


def test_idempotent_when_archive_already_complete():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); M = _mod(tmp)
        p = _src(tmp)
        M._archive_then_unlink(p, "ethusdt")
        gz = tmp / "archive" / "ETHUSDT" / "2026-09-01" / "2026-09-01T03.f32.gz"
        before = gz.stat().st_mtime_ns
        p2 = _src(tmp)                       # 같은 이름이 다시 생긴 상황
        M._archive_then_unlink(p2, "ethusdt")
        assert not p2.exists(), "원본이 안 지워졌다"
        assert gz.stat().st_mtime_ns == before, "재압축했다(멱등 아님)"


def test_truncated_archive_is_not_mistaken_for_complete():
    """🔴핵심. .part -> replace 를 안 쓰면 여기서 원본이 영원히 사라진다."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); M = _mod(tmp)
        gz = tmp / "archive" / "ETHUSDT" / "2026-09-01" / "2026-09-01T03.f32.gz"
        gz.parent.mkdir(parents=True)
        gz.write_bytes(b"")                  # 크기 0 = 쓰다 죽은 흔적
        p = _src(tmp)
        M._archive_then_unlink(p, "ethusdt")
        assert gz.stat().st_size > 0, "빈 gz 를 그대로 뒀다"
        assert not p.exists() and gzip.decompress(gz.read_bytes()).startswith(b"RASTER")


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
