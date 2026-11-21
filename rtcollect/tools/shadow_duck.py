#!/usr/bin/env python3
"""그림자 대조(DuckDB): 파이썬이 쓴 DuckDB 와 Rust(rtduck)가 쓴 DuckDB 를 같은 시간 창에서 표별로, 또는 두 lake 를 파일별로 비교한다(2026-11-21).
읽기 전용 -- 쓰는 프로세스가 잡고 있어 바로 못 여니 두 파일을 사본으로 떠서 읽는다(seal_to_lake 방식 · 사본은 --tmp 아래, 끝나면 지운다).

표(둘 다 있는 것만 · 열쇠 비교 = 값 칸 전부 비트까지):
  hl_trades            열쇠 (coin, tid)            · gaps 구간 ±margin 초의 분은 뺀다
  option_trades        열쇠 trade_id               · block_rfq_trades 열쇠 rfq_id · block_future_legs 열쇠 trade_id
  oi_lsratio_5m*       열쇠 (ts, symbol) -- collected_at 은 뺀다(받은 시각)
  나머지 표(미시구조·청산 요격기처럼 각자 벽시계로 찍는 줄 · REST 전용 표)는 창 안 행 수만 적는다.
lake(--lake): <venue>/<stream>/coin=*/date=*/part.parquet 마다 행 수·hash(row) XOR 지문(seal_to_lake 가 «건너뜀»을 가리는 것과 같은 지문).

    nice -n 10 python3 rtcollect/tools/shadow_duck.py <파이썬.duckdb> <rust.duckdb> --from S --to S [--tmp 폴더] [--margin 5] [--show 3]
    nice -n 10 python3 rtcollect/tools/shadow_duck.py --lake data/lake data/rtduck/lake [--only binance/tape] [--days 2026-11-21,2026-11-27]
    python3 rtcollect/tools/shadow_duck.py --selftest
🔴--tmp 는 6TB 쪽(기본 = Rust 파일 옆 · 그림자 폴더)이어야 한다 -- 사본 크기 = 원본 크기.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import duckdb

KEYED = {  # 표 → (시각 열(ms 정수 또는 TIMESTAMP), 열쇠, 빼는 열)
    "hl_trades": ("ts_ms", ("coin", "tid"), ()),
    "option_trades": ("ts_ms", ("trade_id",), ()),
    "block_rfq_trades": ("ts_ms", ("rfq_id",), ()),
    "block_future_legs": ("ts_ms", ("trade_id",), ()),
}


def spec(t: str):
    return ("ts", ("ts", "symbol"), ("collected_at",)) if t.startswith("oi_lsratio_5m") else KEYED.get(t)


def snapshot(src: Path, tmp: Path) -> duckdb.DuckDBPyConnection:
    """사본을 떠서 표마다 count 가 되는지(찢긴 사본이면 2초 뒤 다시 · 3번)."""
    for i in range(3):
        dst = tmp / f"{i}_{src.name}"
        shutil.copyfile(src, dst)
        wal = src.with_name(src.name + ".wal")
        if wal.exists():
            shutil.copyfile(wal, dst.with_name(dst.name + ".wal"))
        try:
            con = duckdb.connect(str(dst))
            for (t,) in con.execute("SELECT table_name FROM duckdb_tables()").fetchall():
                con.execute(f'SELECT count(*) FROM "{t}"').fetchone()
            return con
        except duckdb.Error as e:
            print(f"찢긴 사본({e}) -- 다시", file=sys.stderr)
            time.sleep(2)
    raise SystemExit(f"{src}: 사본 3번 실패")


def cols(con, t: str) -> list[tuple[str, str]]:
    return con.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position", [t]).fetchall()


def ms_expr(c: str, ty: str) -> str:
    return f'epoch_ms("{c}")' if ty.startswith("TIMESTAMP") else f'"{c}"'


def gap_minutes(cons, margin: int) -> set[int]:
    # ponytail: gaps 는 코인 구분 없이 뺀다(코인 하나가 끊겨도 그 분은 전 코인 제외) -- 많이 빠지면 코인별로
    out: set[int] = set()
    for con in cons:
        if con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'gaps'").fetchone()[0]:
            for a, b in con.execute("SELECT from_ms, to_ms FROM gaps").fetchall():
                out.update(range((a // 1000 - margin) // 60, (b // 1000 + margin) // 60 + 1))
    return out


def compare(py, rs, lo: int, hi: int, margin: int, show: int) -> list[str]:
    tp = {r[0] for r in py.execute("SELECT table_name FROM duckdb_tables()").fetchall()}
    tr = {r[0] for r in rs.execute("SELECT table_name FROM duckdb_tables()").fetchall()}
    excl = gap_minutes((py, rs), margin)
    out = [f"한쪽에만 있는 표: 파이썬 {sorted(tp - tr)} · Rust {sorted(tr - tp)}"] if tp ^ tr else []
    for t in sorted(tp & tr):
        if t == "gaps":
            continue
        ca, cb = cols(py, t), cols(rs, t)
        if [c for c, _ in ca] != [c for c, _ in cb]:
            out.append(f"{t}: 열이 다르다 {ca} · {cb}")
            continue
        s = spec(t)
        tcol = s[0] if s else next((c for c, _ in ca if c in ("ts", "ts_ms", "recorded_at_utc")), None)
        if tcol is None:
            n = [c.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for c in (py, rs)]
            out.append(f"{t}: 시각 열 없음 -- 전체 행 파이썬 {n[0]:,} · Rust {n[1]:,}")
            continue
        te = ms_expr(tcol, dict(ca)[tcol])
        where = f"WHERE {te} >= {lo * 1000} AND {te} < {hi * 1000}"
        if not s:
            n = [c.execute(f'SELECT count(*) FROM "{t}" {where}').fetchone()[0] for c in (py, rs)]
            out.append(f"{t}: (행 수만) 파이썬 {n[0]:,} · Rust {n[1]:,}")
            continue
        names = [c for c, _ in ca if c not in s[2]]
        sel = ", ".join([f"{te} AS __t"] + [ms_expr(c, dict(ca)[c]).replace("epoch_ms", "epoch_us") for c in names])
        rows = []
        for c in (py, rs):
            d = {}
            for r in c.execute(f'SELECT {sel} FROM "{t}" {where}').fetchall():
                if r[0] // 60000 in excl:
                    continue
                v = r[1:]
                d[tuple(v[names.index(k)] for k in s[1])] = v
            rows.append(d)
        a, b = rows
        both = a.keys() & b.keys()
        bad = [k for k in both if a[k] != b[k]]
        line = (f"{t}: 비교 {len(both):,}행 · 불일치 {len(bad):,} · 파이썬에만 {len(a.keys() - b.keys()):,} · Rust에만 {len(b.keys() - a.keys()):,}"
                f" · 범위 [{lo}, {hi}) 초 · 뺀 분 {len(excl):,}")
        first = [f"다름 {k}: 파이썬 {a[k]} · Rust {b[k]}" for k in sorted(bad, key=repr)[:show]]
        first += [f"파이썬에만 {k}" for k in sorted(a.keys() - b.keys(), key=repr)[:show]]
        first += [f"Rust에만 {k}" for k in sorted(b.keys() - a.keys(), key=repr)[:show]]
        out.append(line + "".join(f"\n    {x}" for x in first))
    return out or ["비교할 표가 없다"]


def run(py_path: Path, rs_path: Path, lo: int, hi: int, tmp: Path | None = None, margin: int = 5, show: int = 3) -> list[str]:
    base = tmp or rs_path.parent
    d = Path(tempfile.mkdtemp(prefix=".shadow_duck_", dir=base))
    try:
        (d / "py").mkdir()
        (d / "rs").mkdir()
        py, rs = snapshot(py_path, d / "py"), snapshot(rs_path, d / "rs")
        try:
            return compare(py, rs, lo, hi, margin, show)
        finally:
            py.close()
            rs.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def lake(py_root: Path, rs_root: Path, only: str = "", d0: str = "", d1: str = "~", show: int = 3) -> list[str]:
    def files(root: Path) -> dict:
        out = {}
        for p in root.glob("*/*/coin=*/date=*/part.parquet"):
            rel = p.relative_to(root)
            day = rel.parts[3][5:]
            if (not only or str(rel).startswith(only + "/")) and d0 <= day <= d1:
                out[rel] = p
        return out
    con = duckdb.connect()

    def fp(p: Path):
        return con.execute("SELECT count(*), bit_xor(hash(t)) FROM read_parquet(?) t", [str(p)]).fetchone()
    a, b = files(py_root), files(rs_root)
    per: dict = {}
    for rel in sorted(a.keys() | b.keys()):
        k = "/".join(rel.parts[:2])
        c = per.setdefault(k, Counter())
        if rel not in b:
            c["파이썬에만"] += 1
            continue
        if rel not in a:
            c["Rust에만"] += 1
            continue
        x, y = fp(a[rel]), fp(b[rel])
        c["같음" if x == y else "다름"] += 1
        c["행"] += x[0]
        if x != y and c["다름"] <= show:
            per.setdefault(k + "#", []).append(f"다름 {rel}: 파이썬 {x} · Rust {y}")
    out = []
    for k in sorted(x for x in per if not x.endswith("#")):
        c = per[k]
        out.append(f"{k}: 파일 같음 {c['같음']:,} · 다름 {c['다름']:,} · 파이썬에만 {c['파이썬에만']:,} · Rust에만 {c['Rust에만']:,} · 파이썬 행 {c['행']:,}"
                   + "".join(f"\n    {x}" for x in per.get(k + "#", [])))
    return out or ["lake: 파일이 없다"]


def selftest() -> None:
    T = 1_763_683_200
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, bump in (("py.duckdb", 0), ("rs.duckdb", 1)):
            con = duckdb.connect(str(d / name))
            con.execute("CREATE TABLE hl_trades (coin VARCHAR, ts_ms BIGINT, tid BIGINT, hash VARCHAR, side VARCHAR, px DOUBLE, sz DOUBLE, buyer VARCHAR, seller VARCHAR)")
            con.execute("CREATE TABLE gaps (coin VARCHAR, from_ms BIGINT, to_ms BIGINT, reason VARCHAR)")
            con.executemany("INSERT INTO hl_trades VALUES (?,?,?,?,?,?,?,?,?)",
                            [("ETH", 1000 * (T + s), s, "h", "B", 3000.0 + s, 0.1, "a", "b") for s in range(0, 1200, 10) if not (bump and s == 500)])
            con.execute("CREATE TABLE oi_lsratio_5m (ts TIMESTAMPTZ, symbol VARCHAR, sum_open_interest DOUBLE, collected_at TIMESTAMPTZ)")
            con.executemany("INSERT INTO oi_lsratio_5m VALUES (to_timestamp(?), 'ETHUSDT', ?, to_timestamp(?))",
                            [(T + 300 * k, 1.5 + (bump and k == 1), T + 300 * k + 7 + bump) for k in range(4)])
            con.execute("CREATE TABLE microstructure (ts TIMESTAMPTZ, obi DOUBLE)")
            con.executemany("INSERT INTO microstructure VALUES (to_timestamp(?), 0.1)", [(T + s + bump,) for s in range(0, 600, 60)])
            if bump:
                con.execute("UPDATE hl_trades SET px = 1 WHERE tid = 100")
                con.execute("INSERT INTO gaps VALUES ('ETH', ?, ?, 'ws')", [1000 * (T + 900), 1000 * (T + 905)])
                con.execute("UPDATE hl_trades SET sz = 9 WHERE tid = 900")        # 끊긴 분 → 뺀다
            con.close()
        out = run(d / "py.duckdb", d / "rs.duckdb", T, T + 1200, tmp=d)
        print("\n".join(out))
        hl = next(x for x in out if x.startswith("hl_trades"))
        # 120행 · 끊긴 분(±5초 → 분 14·15) 12행 뺌 · tid 500 은 Rust 에 없다 · tid 100 값 다름
        assert "비교 107행 · 불일치 1 · 파이썬에만 1 · Rust에만 0" in hl and "뺀 분 2" in hl, hl
        oi = next(x for x in out if x.startswith("oi_lsratio_5m"))
        assert "비교 3행 · 불일치 1 · 파이썬에만 0 · Rust에만 0" in oi, oi     # collected_at 은 안 본다 · T+900 봉은 끊긴 분(전 표 공통)
        assert any(x == "microstructure: (행 수만) 파이썬 10 · Rust 10" for x in out), out
        assert list(d.glob(".shadow_duck_*")) == [], "사본이 남았다"
        # lake: 같은 파일 · 행 하나 다른 파일 · 한쪽에만
        con = duckdb.connect()
        for root, extra in ((d / "lp", 0), (d / "lr", 1)):
            for day in ("2026-11-20", "2026-11-21"):
                p = root / "binance/tape/coin=ETH" / f"date={day}"
                p.mkdir(parents=True)
                v = 2 if (extra and day == "2026-11-21") else 1
                con.execute(f"COPY (SELECT 1 AS a UNION ALL SELECT {v}) TO '{p / 'part.parquet'}' (FORMAT parquet)")
        p = d / "lp/okx/oi/coin=ETH/date=2026-11-20"
        p.mkdir(parents=True)
        con.execute(f"COPY (SELECT 1 AS a) TO '{p / 'part.parquet'}' (FORMAT parquet)")
        lo = lake(d / "lp", d / "lr")
        print("\n".join(lo))
        assert lo[0].startswith("binance/tape: 파일 같음 1 · 다름 1 · 파이썬에만 0 · Rust에만 0 · 파이썬 행 4"), lo
        assert lo[1].startswith("okx/oi: 파일 같음 0 · 다름 0 · 파이썬에만 1"), lo
        assert lake(d / "lp", d / "lr", only="binance/tape", d0="2026-11-20", d1="2026-11-20")[0].startswith("binance/tape: 파일 같음 1 · 다름 0"), "거르기"
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("py", nargs="?", type=Path)
    ap.add_argument("rs", nargs="?", type=Path)
    ap.add_argument("--from", dest="lo", type=int)
    ap.add_argument("--to", dest="hi", type=int)
    ap.add_argument("--tmp", type=Path)
    ap.add_argument("--margin", type=int, default=5)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--lake", action="store_true", help="두 경로 = lake 뿌리 · --only venue/stream · --days 시작,끝(YYYY-MM-DD)")
    ap.add_argument("--only", default="")
    ap.add_argument("--days", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.py and a.rs):
        ap.error("파이썬 쪽과 Rust 쪽 경로가 필요하다")
    if a.lake:
        d0, _, d1 = a.days.partition(",")
        return print("\n".join(lake(a.py, a.rs, a.only, d0, d1 or "~", a.show)))
    if a.lo is None or a.hi is None:
        ap.error("--from/--to(초)가 필요하다 -- 창 하나씩")
    print("\n".join(run(a.py, a.rs, a.lo, a.hi, a.tmp, a.margin, a.show)))


if __name__ == "__main__":
    sys.exit(main())
