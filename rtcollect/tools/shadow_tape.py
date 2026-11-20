#!/usr/bin/env python3
"""그림자 대조: 파이썬 수집기가 쓴 SQLite 와 Rust(rtcollect)가 쓴 SQLite(같은 스키마, 다른 경로)를 같은 시간 범위에서 표별로 비교한다
(2026-11-20). 표준 라이브러리 sqlite3 만 · 읽기 전용(mode=ro) · 시간 창 단위로(큰 표를 통째로 메모리에 안 올린다).

비교하는 표(둘 다 있는 것만):
  trade_tape_1s                 열쇠 (symbol, ts_sec, price_bin) → 나머지 칸 전부(REAL 은 비트까지 -- 같은 규칙이면 같은 값)
  bybit_liquidations            행 묶음(중복 포함) -- recv_ms 는 뺀다(두 프로세스가 받은 시각은 원래 다르다)
  bybit_oi · bybit_mark · bybit_funding   행 묶음
빼는 분(뺀 분 수를 같이 찍는다):
  · 각 DB 의 gaps 구간 ±margin 초(재연결·재시작 주변 -- 두 프로세스가 끊긴 때가 다르다). symbol 열이 있으면 그 종목만.
  · 두 쪽 backfill_1m 의 ok=1 분(REST 로 통째로 갈아 끼운 분 -- Rust 는 REST 가 기본 꺼짐이라 보통 파이썬 쪽뿐)
  · 두 쪽 verify_1m 의 |rel_err| > 1e-4 분(1분봉과 안 맞은 분 = 곧 복구 대상이거나 유실)
  · 범위 끝 10초(진행 중인 초 · 5초 플러시 지연)
범위 = 두 DB 에 그 표 자료가 함께 있는 구간(--from/--to 로 좁힌다, 초).

    python3 rtcollect/tools/shadow_tape.py <파이썬.sqlite> <rust.sqlite> [--from S] [--to S] [--margin 5] [--window 3600] [--show 3]
    python3 rtcollect/tools/shadow_tape.py --selftest
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

TOL = 1e-4
TABLES = {  # 표 → (시각 열, 시각 단위 ms 인가, 열쇠 열 또는 None(행 묶음), 빼는 열)
    "trade_tape_1s": ("ts_sec", False, ("symbol", "ts_sec", "price_bin"), ()),
    "bybit_liquidations": ("ts_ms", True, None, ("recv_ms",)),
    "bybit_oi": ("ts_ms", True, None, ()),
    "bybit_mark": ("ts_ms", True, None, ()),
    "bybit_funding": ("ts_ms", True, None, ()),
}
SYM_COL = {"trade_tape_1s": "symbol", "bybit_liquidations": "symbol", "bybit_oi": "inst", "bybit_mark": "inst", "bybit_funding": "inst"}


def ro(p: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=10)


def tables(con) -> set[str]:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}


def cols(con, t: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({t})")]


def excluded_minutes(py, rs, margin: int) -> dict:
    """{종목 또는 None(전 종목): {분}}"""
    out: dict = {}
    for con in (py, rs):
        if "gaps" not in tables(con):
            continue
        has_sym = "symbol" in cols(con, "gaps")
        q = "SELECT symbol, from_ms, to_ms FROM gaps" if has_sym else "SELECT NULL, from_ms, to_ms FROM gaps"
        for sym, a, b in con.execute(q):
            lo, hi = (a // 1000 - margin) // 60, (b // 1000 + margin) // 60
            out.setdefault(sym, set()).update(range(lo, hi + 1))
    for con in (py, rs):      # Rust 도 REST 를 켜면(RTCOLLECT_REST=1) 자기 표에 적는다 -- 꺼져 있으면 비어 있다
        t = tables(con)
        if "backfill_1m" in t:
            for sym, m in con.execute("SELECT symbol, ts_min FROM backfill_1m WHERE ok"):
                out.setdefault(sym, set()).add(m // 60)
        if "verify_1m" in t:
            for sym, m in con.execute("SELECT symbol, ts_min FROM verify_1m WHERE abs(rel_err) > ?", [TOL]):
                out.setdefault(sym, set()).add(m // 60)
    return out


def span(con, t: str, tcol: str, ms: bool) -> tuple[int, int] | None:
    lo, hi = con.execute(f"SELECT min({tcol}), max({tcol}) FROM {t}").fetchone()
    if lo is None:
        return None
    return (lo // 1000, hi // 1000) if ms else (lo, hi)


def compare_table(py, rs, t: str, lo: int | None, hi: int | None, margin: int, window: int, show: int, excl: dict) -> str:
    tcol, ms, key, drop = TABLES[t]
    a, b = span(py, t, tcol, ms), span(rs, t, tcol, ms)
    if not a or not b:
        return f"{t}: 한쪽이 비었다 (파이썬 {a} · Rust {b})"
    s0, s1 = max(a[0], b[0], lo if lo is not None else a[0]), min(a[1], b[1], hi if hi is not None else a[1]) - 10
    if s1 <= s0:
        return f"{t}: 겹치는 범위 없음 (파이썬 {a} · Rust {b})"
    names = [c for c in cols(py, t) if c not in drop]
    assert names == [c for c in cols(rs, t) if c not in drop], f"{t} 열이 다르다"
    sel = ", ".join(names)
    si, ti = names.index(SYM_COL[t]), names.index(tcol)
    kidx = [names.index(k) for k in key] if key else None
    allsym = excl.get(None, set())
    n_cmp = n_bad = only_py = only_rs = 0
    skipped: set = set()
    first: list = []

    def keep(r) -> bool:
        sec = r[ti] // 1000 if ms else r[ti]
        m = sec // 60
        if m in allsym or m in excl.get(r[si], ()):
            skipped.add((r[si], m))
            return False
        return True

    for w0 in range(s0, s1, window):
        w1 = min(w0 + window, s1)
        args = [w0 * 1000, w1 * 1000] if ms else [w0, w1]
        q = f"SELECT {sel} FROM {t} WHERE {tcol} >= ? AND {tcol} < ?"
        rp = [r for r in py.execute(q, args) if keep(r)]
        rr = [r for r in rs.execute(q, args) if keep(r)]
        if kidx:
            dp = {tuple(r[i] for i in kidx): r for r in rp}
            dr = {tuple(r[i] for i in kidx): r for r in rr}
            for k in dp.keys() & dr.keys():
                n_cmp += 1
                if dp[k] != dr[k]:
                    n_bad += 1
                    if len(first) < show:
                        first.append(f"다름 {k}: 파이썬 {dp[k]} · Rust {dr[k]}")
            po, ro_ = dp.keys() - dr.keys(), dr.keys() - dp.keys()
        else:
            cp, cr = Counter(rp), Counter(rr)
            n_cmp += sum((cp & cr).values())
            po, ro_ = cp - cr, cr - cp
        only_py += sum(po.values()) if isinstance(po, Counter) else len(po)
        only_rs += sum(ro_.values()) if isinstance(ro_, Counter) else len(ro_)
        for tag, extra in (("파이썬에만", po), ("Rust에만", ro_)):
            for k in sorted(extra)[: max(0, show - len(first))]:
                first.append(f"{tag} {k}")
    line = (f"{t}: 비교 {n_cmp:,}행 · 불일치 {n_bad:,} · 파이썬에만 {only_py:,} · Rust에만 {only_rs:,} · 범위 [{s0}, {s1}) 초 · "
            f"뺀 (종목,분) {len(skipped):,}")
    return line + "".join(f"\n    {x}" for x in first)


def run(py_path: Path, rs_path: Path, lo=None, hi=None, margin=5, window=3600, show=3) -> list[str]:
    py, rs = ro(py_path), ro(rs_path)
    try:
        excl = excluded_minutes(py, rs, margin)
        common = sorted(tables(py) & tables(rs) & TABLES.keys())
        return [compare_table(py, rs, t, lo, hi, margin, window, show, excl) for t in common] or ["비교할 표가 없다"]
    finally:
        py.close()
        rs.close()


def selftest() -> None:
    tape = ("CREATE TABLE trade_tape_1s(symbol TEXT, ts_sec INTEGER, price_bin INTEGER, buy_qty REAL, sell_qty REAL, "
            "PRIMARY KEY (symbol, ts_sec, price_bin)) WITHOUT ROWID")
    ddl = [tape, "CREATE TABLE gaps(symbol TEXT, from_ms INTEGER, to_ms INTEGER, reason TEXT)",
           "CREATE TABLE verify_1m(symbol TEXT, ts_min INTEGER, tape_qty REAL, kline_qty REAL, rel_err REAL, checked_at TEXT)",
           "CREATE TABLE backfill_1m(symbol TEXT, ts_min INTEGER, source TEXT, n_rows INTEGER, tape_qty REAL, kline_qty REAL, "
           "rel_err REAL, ok INTEGER, attempts INTEGER, note TEXT, done_at TEXT)",
           "CREATE TABLE bybit_liquidations(ts_ms INTEGER, symbol TEXT, side TEXT, qty REAL, price REAL, recv_ms INTEGER)"]
    T = 1_763_683_200                          # 분 경계
    with tempfile.TemporaryDirectory() as d:
        p, r = Path(d) / "py.sqlite", Path(d) / "rs.sqlite"
        for f in (p, r):
            con = sqlite3.connect(f)
            for x in ddl:
                con.execute(x)
            # 같은 행: 20분 × 초마다 2칸 = 2,400행
            con.executemany("INSERT INTO trade_tape_1s VALUES (?,?,?,?,?)",
                            [("ethusdt", T + s, b, 1.5 + s, 0.25) for s in range(1200) for b in (25000, 25001)])
            con.executemany("INSERT INTO bybit_liquidations VALUES (?,?,?,?,?,?)",
                            [(1000 * (T + s), "ETHUSDT", "Buy", 0.1 * s, 2500.0, 1000 * (T + s) + (7 if f == p else 9)) for s in range(0, 1200, 30)])
            con.commit()
            con.close()
        con = sqlite3.connect(p)
        con.execute("UPDATE trade_tape_1s SET buy_qty = 99 WHERE ts_sec = ? AND price_bin = 25000", [T + 100])       # 분 1: 불일치
        con.execute("DELETE FROM trade_tape_1s WHERE ts_sec = ? AND price_bin = 25001", [T + 130])                   # 분 2: Rust에만
        con.execute("UPDATE trade_tape_1s SET buy_qty = 99 WHERE ts_sec = ?", [T + 305])                             # 분 5: REST 교체 → 뺀다
        con.execute("INSERT INTO backfill_1m VALUES ('ethusdt', ?, 'rest', 1, 0, 0, 0, 1, 1, '', '')", [T + 300])
        con.execute("UPDATE trade_tape_1s SET sell_qty = 7 WHERE ts_sec = ?", [T + 365])                             # 분 6: 1분봉 불일치 → 뺀다
        con.execute("INSERT INTO verify_1m VALUES ('ethusdt', ?, 1, 2, -0.5, '')", [T + 360])
        con.execute("INSERT INTO backfill_1m VALUES ('ethusdt', ?, 'rest', 1, 0, 0, 0, 0, 3, '', '')", [T + 420])   # ok=0 → 안 뺀다
        con.execute("DELETE FROM bybit_liquidations WHERE ts_ms = ?", [1000 * (T + 600)])                            # 청산 하나 파이썬에 없음
        con.commit()
        con.close()
        con = sqlite3.connect(r)
        con.execute("DELETE FROM trade_tape_1s WHERE ts_sec BETWEEN ? AND ?", [T + 610, T + 615])                    # 분 10: 재연결로 빔
        con.execute("INSERT INTO gaps VALUES ('ethusdt', ?, ?, 'ws_reconnect')", [1000 * (T + 610), 1000 * (T + 616)])
        con.execute("UPDATE trade_tape_1s SET buy_qty = 5 WHERE ts_sec = ?", [T + 1195])                             # 끝 10초 → 범위 밖
        con.commit()
        con.close()
        out = run(p, r, window=250, show=5)
        print("\n".join(out))
        tape_line = next(x for x in out if x.startswith("trade_tape_1s"))
        # 범위 [T, T+1199-10) · 뺀 분 5·6·10(gaps ±5초 = 분 10 하나) → 비교 = (1189 - 180) 초 × 2 - Rust에만 1
        assert "범위 [1763683200, 1763684389)" in tape_line, tape_line
        assert "뺀 (종목,분) 3" in tape_line, tape_line
        assert "불일치 1 · 파이썬에만 0 · Rust에만 1" in tape_line, tape_line
        assert f"비교 {(1189 - 180) * 2 - 1:,}행" in tape_line, tape_line
        assert "다름 ('ethusdt', 1763683300, 25000)" in tape_line, tape_line
        liq_line = next(x for x in out if x.startswith("bybit_liquidations"))
        assert "불일치 0 · 파이썬에만 0 · Rust에만 1" in liq_line and "비교 38행" in liq_line, liq_line   # recv_ms 는 달라도 같다
        # 읽기 전용: 두 파일은 그대로
        try:
            ro(p).execute("DELETE FROM gaps")
            raise AssertionError("mode=ro 인데 쓰기가 됐다")
        except sqlite3.OperationalError:
            pass
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("py", nargs="?", type=Path)
    ap.add_argument("rs", nargs="?", type=Path)
    ap.add_argument("--from", dest="lo", type=int)
    ap.add_argument("--to", dest="hi", type=int)
    ap.add_argument("--margin", type=int, default=5)
    ap.add_argument("--window", type=int, default=3600)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.py and a.rs):
        ap.error("파이썬 DB 와 Rust DB 경로가 필요하다")
    print("\n".join(run(a.py, a.rs, a.lo, a.hi, a.margin, a.window, a.show)))


if __name__ == "__main__":
    sys.exit(main())
