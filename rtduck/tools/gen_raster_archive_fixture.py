"""래스터 14일 보관 정답지: 원본 live_orderflow_raster_collector_20260914._purge_old 를 합성 .f32 파일들 + 가짜 시계로 돌린다.

경우: 만료(보관 → 원본 지움) · 아직 안 만료(그대로) · 이름이 시각 아님(그대로) · 행 0(«행이 없다» → gzip 폴백) · magic 틀림(gzip 폴백) ·
  끝 행 반쪽(온전한 행만) · qty 0·-0.0·NaN·음수 · mid NaN · 이미 보관본 있음(원본만 지움) · 심볼 둘.
출력 = tests/fixtures/raster_archive_{input,golden}.json. 바깥 접속 없음.
  python3 rtduck/tools/gen_raster_archive_fixture.py            # 정답지
  python3 rtduck/tools/gen_raster_archive_fixture.py --check D  # Rust 가 쓴 보관 폴더 D 를 이 파이썬(pyarrow·duckdb)으로 되읽어 정답지와 대조
"""
from __future__ import annotations

import gzip
import json
import random
import struct
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_orderflow_raster_collector_20260914 as C  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

FIX = ROOT / "rtduck/tests/fixtures"
NOW = datetime(2026, 11, 21, 3, 30, tzinfo=timezone.utc)
rnd = random.Random(1121)


def f32(nb: int, binsz: float, hour_ms: int, rows: list, magic: bytes = C.MAGIC, half: bool = False) -> bytes:
    b = C.HEADER.pack(magic, 1, nb, binsz, 0, hour_ms, 1000, 0)
    for ts, lo, mid, q in rows:
        b += C.ROW_HEAD.pack(ts, lo, mid) + struct.pack(f"<{nb}f", *q)
    if half:
        b += b"\x01" * (C.ROW_HEAD.size + 4 * nb // 2)
    return b


def rows(hour_ms: int, n: int, nb: int, lo0: int) -> list:
    out = []
    for i in range(n):
        q = [0.0] * nb
        for _ in range(rnd.randrange(0, nb // 2)):
            q[rnd.randrange(nb)] = round(rnd.uniform(-50, 50), 3)
        if i % 7 == 3:
            q[1] = -0.0
            q[2] = float("nan")
        mid = float("nan") if i % 11 == 5 else 2500 + rnd.uniform(-3, 3)
        out.append((hour_ms + i * 1000, lo0 + rnd.randrange(-3, 4), mid, q))
    return out


def hms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc).timestamp() * 1000)


def build_inputs() -> dict:
    files = {}
    for sym, nb, bs in (("ETHUSDT", 24, 0.5), ("BTCUSDT", 16, 5.0)):
        for stamp, n, kind in (("2026-11-01T00", 40, "ok"), ("2026-11-06T23", 25, "half"), ("2026-11-07T02", 30, "ok"), ("2026-11-07T03", 9, "ok"),
                               ("2026-11-07T04", 5, "fresh"), ("2026-11-20T10", 12, "fresh")):
            h = hms(stamp)
            files[f"{sym}/{stamp}.f32"] = f32(nb, bs, h, rows(h, n, nb, 5000), half=kind == "half")
    h = hms("2026-11-02T05")
    files["ETHUSDT/2026-11-02T05.f32"] = f32(24, 0.5, h, [])                                   # 행 0 → gzip
    files["ETHUSDT/2026-11-02T06.f32"] = f32(24, 0.5, h, rows(h, 3, 24, 5000), magic=b"XXXX")  # magic 틀림 → gzip
    files["ETHUSDT/2026-11-02T07.f32"] = b"short"                                             # 헤더보다 짧다 → gzip
    files["ETHUSDT/notes.f32"] = b"not a stamp"                                               # 이름이 시각 아님 → 그대로
    files["ETHUSDT/2026-11-03T01.f32"] = f32(24, 0.5, hms("2026-11-03T01"), rows(hms("2026-11-03T01"), 4, 24, 5000))  # 보관본이 이미 있다 → 원본만 지움
    return files


def run_original(td: Path, files: dict, pre: dict) -> None:
    C.OF_ROOT, C.OF_ARCHIVE_ROOT = td / "raster", td / "archive"
    C.OF_ARCHIVE_DB = C.OF_ARCHIVE_ROOT / "orderbook.duckdb"
    C._ARCHIVE_VIEW_DONE = False
    for rel, b in files.items():
        (C.OF_ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        (C.OF_ROOT / rel).write_bytes(b)
    for rel, b in pre.items():
        (C.OF_ARCHIVE_ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        (C.OF_ARCHIVE_ROOT / rel).write_bytes(b)

    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW
    C.datetime = FakeDT
    for sym in ("ethusdt", "btcusdt"):
        o = C.OrderflowRasterCollector.__new__(C.OrderflowRasterCollector)
        o.symbol = sym
        o._purge_old()


def describe(root: Path, archive: Path) -> dict:
    """남은 원본 · 보관 파일마다 (열 이름·형, 행) · gz 는 푼 바이트 · 뷰 집계."""
    import duckdb
    out = {"raster_left": sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()), "archive": {}}
    for p in sorted(archive.rglob("*")):
        if not p.is_file() or p.name.endswith(".duckdb") or p.name.endswith(".wal"):
            continue
        rel = str(p.relative_to(archive))
        if p.name.endswith(".gz"):
            out["archive"][rel] = {"gz": gzip.decompress(p.read_bytes()).hex()}
        else:
            t = pq.read_table(p)
            out["archive"][rel] = {"schema": [[f.name, str(f.type)] for f in t.schema],
                                   "rows": [[None if isinstance(v, float) and v != v else v for v in r.values()] for r in t.to_pylist()]}
    db = archive / "orderbook.duckdb"
    if db.exists():
        con = duckdb.connect(str(db), read_only=True)
        out["view"] = {"book": [list(r) for r in con.execute("select symbol, count(*), count(bin_size), sum(bin) from book group by 1 order by 1").fetchall()],
                       "book_mid": [list(r) for r in con.execute("select symbol, count(*), count(mid) from book_mid group by 1 order by 1").fetchall()]}
        con.close()
    return out


def main() -> None:
    files = build_inputs()
    with tempfile.TemporaryDirectory() as t0:
        # 이미 있던 보관본 = 원본 함수로 미리 떠 둔 것(같은 파일을 다른 내용으로 -- 원본은 다시 안 쓰고 지우기만)
        src = Path(t0) / "x.f32"
        src.write_bytes(f32(24, 0.5, hms("2026-11-03T01"), rows(hms("2026-11-03T01"), 2, 24, 5000)))
        C._f32_to_parquet(src, Path(t0) / "pre.parquet")
        pre = {"ETHUSDT/2026-11-03/2026-11-03T01.f32.parquet": (Path(t0) / "pre.parquet").read_bytes()}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run_original(td, files, pre)
        gold = describe(td / "raster", td / "archive")
        # Rust 가 «원본이 쓴 parquet» 을 읽을 수 있는지 보려고 원본 보관본 바이트도 넘긴다
        written = {str(p.relative_to(td / "archive")): p.read_bytes().hex() for p in sorted((td / "archive").rglob("*.parquet"))}
    (FIX / "raster_archive_input.json").write_text(json.dumps({"now_s": NOW.timestamp(), "files": {k: v.hex() for k, v in files.items()},
                                                                "pre": {k: v.hex() for k, v in pre.items()}}, separators=(",", ":")))
    (FIX / "raster_archive_golden.json").write_text(json.dumps({**gold, "python_parquet": written}, separators=(",", ":")))
    print(f"정답지: 남은 원본 {len(gold['raster_left'])} · 보관 파일 {len(gold['archive'])} · 행 {sum(len(v.get('rows', [])) for v in gold['archive'].values())} · 뷰 {gold.get('view')}")


def check(d: Path) -> None:
    gold = json.loads((FIX / "raster_archive_golden.json").read_text())
    got = describe(d / "raster", d / "archive")
    bad = [k for k in ("raster_left", "archive", "view") if got.get(k) != gold.get(k)]
    for k in bad:
        if k == "archive":
            for f in sorted(set(got[k]) | set(gold[k])):
                if got[k].get(f) != gold[k].get(f):
                    print("다름", f)
        else:
            print("다름", k, got.get(k), gold.get(k))
    assert not bad, bad
    print(f"파이썬(pyarrow {pq.__name__} · duckdb)이 Rust 보관본을 되읽음: 파일 {len(got['archive'])} · 행·형·값·뷰 정답지와 같음")


if __name__ == "__main__":
    check(Path(sys.argv[2])) if "--check" in sys.argv else main()
