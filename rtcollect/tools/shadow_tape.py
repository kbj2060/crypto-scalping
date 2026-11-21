#!/usr/bin/env python3
"""그림자 대조: 파이썬 수집기가 쓴 SQLite 와 Rust(rtcollect)가 쓴 SQLite(같은 스키마, 다른 경로)를 같은 시간 범위에서 표별로 비교한다
(2026-11-20). 표준 라이브러리 sqlite3 만 · 읽기 전용(mode=ro) · 시간 창 단위로(큰 표를 통째로 메모리에 안 올린다).

비교하는 표(둘 다 있는 것만):
  trade_tape_1s                 열쇠 (symbol, ts_sec, price_bin) → 나머지 칸 전부(REAL 은 비트까지 -- 같은 규칙이면 같은 값)
  bybit_liquidations            행 묶음(중복 포함) -- recv_ms 는 뺀다(두 프로세스가 받은 시각은 원래 다르다)
  bybit_oi · bybit_mark · bybit_funding   행 묶음
  okx_oi · okx_mark · okx_funding · okx_liquidations(OKX 맥락)   행 묶음
  mark_price_1s · oi_1s(바이낸스 맥락)   열쇠 (ts_ms, symbol) · liquidations   행 묶음
  hl_asset_ctx(HL 자산 맥락)   행 묶음 -- recv_ms 는 뺀다(받은 시각)
  hl_xyz_1m(xyz100)   열쇠 (coin, ts_sec) · hl_xyz_ctx_1s   행 묶음 -- recv_sec 는 뺀다(받은 초 = 벽시계, 두 프로세스가 원래 다르다)
  bars(ob5m 5분 봉 · 호가 깊이 10열)   열쇠 t(초) -- written_ms 는 뺀다 · marks(ob5m 표식)   열쇠 (t, side, px)
     두 쪽 다 파이썬 depthdiff·hot 테이프를 읽으니(그림자 기본 DD_ROOT) «불일치 0» 이어야 한다 · 파일이 종목마다라 종목 열이 없다
  --events: 아이스버그 사건 JSONL 두 파일 -- 줄 글자 집합(t_end 가 --from/--to 안인 줄만 · 같은 원천을 읽으니 같아야 한다)
  🔴바이낸스 맥락에는 gaps 표가 없다(원본에 없다) -- 재연결 언저리의 «한쪽에만»은 로그(`ws:`)의 끊긴 시각과 맞춰 본다.
     xyz100 의 ctx 는 받은 초가 1칸 밀리면 같은 초로 합쳐지는 행 수가 갈린다 -- 불일치 0 이 아니라 «비율»로 본다(SHADOW.md §5).
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
TALLY = [0, 0]   # --summary: [견준 수, 설명 못 하는 차이 수] -- 모든 비교 함수가 더한다
TABLES = {  # 표 → (시각 열, 시각 단위 ms 인가, 열쇠 열 또는 None(행 묶음), 빼는 열)
    "trade_tape_1s": ("ts_sec", False, ("symbol", "ts_sec", "price_bin"), ()),
    "bybit_liquidations": ("ts_ms", True, None, ("recv_ms",)),
    "bybit_oi": ("ts_ms", True, None, ()),
    "bybit_mark": ("ts_ms", True, None, ()),
    "bybit_funding": ("ts_ms", True, None, ()),
    "okx_oi": ("ts_ms", True, None, ()),
    "okx_mark": ("ts_ms", True, None, ()),
    "okx_funding": ("ts_ms", True, None, ()),
    "okx_liquidations": ("ts_ms", True, None, ()),
    "mark_price_1s": ("ts_ms", True, ("ts_ms", "symbol"), ()),
    "oi_1s": ("ts_ms", True, ("ts_ms", "symbol"), ()),
    "liquidations": ("ts_ms", True, None, ()),
    "hl_xyz_1m": ("ts_sec", False, ("coin", "ts_sec"), ()),
    "hl_xyz_ctx_1s": ("recv_sec", False, None, ("recv_sec",)),
    "hl_asset_ctx": ("recv_ms", True, None, ("recv_ms",)),
    "bars": ("t", True, ("t",), ("written_ms", "chain_age_s")),   # chain_age_s = 사슬 시작부터 초(각자 띄운 때에 달렸다 -- 진단 칸)
    "marks": ("t", True, ("t", "side", "px"), ()),
}
# 🔧의도한 차이(rtcollect/SHADOW.md §11 -- 결함은 Rust 에서만 고쳤다): 이 표들의 «Rust에만» 행은 원본이 잃는 행일 수 있다 -- 불일치로 세지 않고 따로 적는다.
#   «파이썬에만»·값 불일치는 그대로 실패다(고친 Rust 가 원본보다 적게 쓸 까닭은 없다).
INTENDED_EXTRA = {
    "bybit_liquidations": "원본은 바인딩 못 하는 행 하나로 재시작까지 쓰기가 멈춘다",
    "bybit_oi": "원본은 못 읽는 tickers 칸 뒤 OI 행을 잃는다",
    "bybit_mark": "원본은 못 읽는 tickers 칸이 그 종목 맥락을 재연결 반복으로 막는다",
    "bybit_funding": "같은 이유",
    "oi_1s": "원본은 OI 를 못 읽은 시각을 영구히 버린다",
    "mark_price_1s": "원본은 깨진 JSON 한 프레임에 그 연결을 끊는다",
    "liquidations": "같은 이유",
    "hl_xyz_1m": "원본은 끊길 때 마지막 플러시 뒤 행을 버린다",
    "hl_xyz_ctx_1s": "같은 이유",
}
SYM_COL = {"trade_tape_1s": "symbol", "bybit_liquidations": "symbol", "bybit_oi": "inst", "bybit_mark": "inst", "bybit_funding": "inst",
           "okx_oi": "inst", "okx_mark": "inst", "okx_funding": "inst", "okx_liquidations": "inst_id",
           "mark_price_1s": "symbol", "oi_1s": "symbol", "liquidations": "symbol", "hl_xyz_1m": "coin", "hl_xyz_ctx_1s": "coin", "hl_asset_ctx": "coin"}


def ro(p: Path) -> sqlite3.Connection:
    """읽기 전용. 경로에 «+» 가 있으면 여러 Rust 파일을 이어 붙인 하나로(hl_bt 그림자의 hl_ctx.sqlite + hl_ctx 그림자의 hl_ctx_only.sqlite)."""
    parts = str(p).split("+")
    if len(parts) == 1:
        return sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=10)
    con = sqlite3.connect("file::memory:", uri=True, timeout=10)
    have: dict = {}
    for i, f in enumerate(parts):
        con.execute(f"ATTACH DATABASE ? AS r{i}", [f"file:{f}?mode=ro"])
        for (t,) in con.execute(f"SELECT name FROM r{i}.sqlite_master WHERE type = 'table'"):
            have.setdefault(t, []).append(i)
    for t, ix in have.items():
        con.execute(f"CREATE TEMP VIEW {t} AS " + " UNION ALL ".join(f"SELECT * FROM r{i}.{t}" for i in ix))
    return con


def tables(con) -> set[str]:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
                                      "UNION SELECT name FROM sqlite_temp_master WHERE type = 'view'")}


def down_intervals(times, starts: list[int], margin: int) -> list[tuple[float, float]]:
    """그림자 Rust 가 꺼져 있던 구간(초): 감독 로그의 «SUPERVISOR starting» 시각 T 마다 [T 전 Rust 마지막 자료, T 뒤 Rust 첫 자료] ± margin.
    times(T) = (T 전 마지막 초 또는 None, T 뒤 첫 초 또는 None). 첫 기동 전은 -inf 부터(그림자가 뜨기 전 자료는 견주지 않는다)."""
    out = []
    for t in sorted(starts):
        a, b = times(t)
        out.append((a - margin if a is not None else float("-inf"), b + margin if b is not None else float("inf")))
    merged: list = []
    for a, b in out:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def down_note(d: list, n: int) -> str:
    if not d:
        return ""
    fin = sum(b - a for a, b in d if a > float("-inf") and b < float("inf"))
    return f" · 그림자 꺼진 구간 {len(d)}개(첫 기동 전 포함 · 유한 합 {fin:.0f}초)에서 뺀 행 {n:,}"


def sql_times(con, starts_tables):
    """sqlite 꼴: 표마다 시각 열에서 T 앞·뒤 Rust 자료(초)."""
    def times(T):
        lo = hi = None
        for t, tcol, ms in starts_tables:
            f = 1000 if ms else 1
            a = con.execute(f"SELECT max({tcol}) FROM {t} WHERE {tcol} < ?", [T * f]).fetchone()[0]
            b = con.execute(f"SELECT min({tcol}) FROM {t} WHERE {tcol} >= ?", [T * f]).fetchone()[0]
            lo = a // f if a is not None and (lo is None or a // f > lo) else lo
            hi = b // f if b is not None and (hi is None or b // f < hi) else hi
        return lo, hi
    return times


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


def compare_table(py, rs, t: str, lo: int | None, hi: int | None, margin: int, window: int, show: int, excl: dict, down: list = ()) -> str:
    tcol, ms, key, drop = TABLES[t]
    a, b = span(py, t, tcol, ms), span(rs, t, tcol, ms)
    if not a or not b:
        return f"{t}: 한쪽이 비었다 (파이썬 {a} · Rust {b})"
    s0, s1 = max(a[0], b[0], lo if lo is not None else a[0]), min(a[1], b[1], hi if hi is not None else a[1]) - 10
    if s1 <= s0:
        return f"{t}: 겹치는 범위 없음 (파이썬 {a} · Rust {b})"
    names = cols(py, t)
    assert names == cols(rs, t), f"{t} 열이 다르다"
    sel = ", ".join(names)
    si, ti = names.index(SYM_COL[t]) if t in SYM_COL else None, names.index(tcol)
    kidx = [names.index(k) for k in key] if key else None
    keepcols = [i for i, c in enumerate(names) if c not in drop]   # 비교에서 빼는 열(받은 시각) -- 창·종목 거르기에는 쓴다
    allsym = excl.get(None, set())
    n_cmp = n_bad = only_py = only_rs = n_down = n_warm = 0
    skipped: set = set()
    wi = names.index("warm") if "warm" in names else None   # 준비 구간 봉(warm=1)은 한쪽이라도 그러면 견주지 않는다(사슬 시작에 달렸다)
    first: list = []
    carry: tuple = (Counter(), Counter())

    def keep(r) -> bool:
        nonlocal n_down
        sec = r[ti] // 1000 if ms else r[ti]
        if any(a <= sec < b for a, b in down):
            n_down += 1
            return False
        m = sec // 60
        sym = r[si] if si is not None else None
        if m in allsym or m in excl.get(sym, ()):
            skipped.add((sym, m))
            return False
        return True

    for w0 in range(s0, s1, window):
        w1 = min(w0 + window, s1)
        args = [w0 * 1000, w1 * 1000] if ms else [w0, w1]
        q = f"SELECT {sel} FROM {t} WHERE {tcol} >= ? AND {tcol} < ?"
        rp = [r for r in py.execute(q, args) if keep(r)]
        rr = [r for r in rs.execute(q, args) if keep(r)]
        if not kidx:
            rp, rr = [tuple(r[i] for i in keepcols) for r in rp], [tuple(r[i] for i in keepcols) for r in rr]
        if kidx:
            if wi is not None:
                wp, wr = {tuple(r[i] for i in kidx) for r in rp if r[wi] == 1}, {tuple(r[i] for i in kidx) for r in rr if r[wi] == 1}
                n_warm += len(wp | wr)
                rp, rr = [r for r in rp if tuple(r[i] for i in kidx) not in wp | wr], [r for r in rr if tuple(r[i] for i in kidx) not in wp | wr]
            dp = {tuple(r[i] for i in kidx): tuple(r[i] for i in keepcols) for r in rp}
            dr = {tuple(r[i] for i in kidx): tuple(r[i] for i in keepcols) for r in rr}
            for k in dp.keys() & dr.keys():
                n_cmp += 1
                if dp[k] != dr[k]:
                    n_bad += 1
                    if len(first) < show:
                        first.append(f"다름 {k}: 파이썬 {dp[k]} · Rust {dr[k]}")
            po, ro_ = dp.keys() - dr.keys(), dr.keys() - dp.keys()
        else:
            # 행 묶음: 못 맞춘 행은 다음 창으로 넘긴다(창 경계에 걸린 행 · 받은 시각이 한쪽만 늦은 행)
            cp, cr = Counter(rp) + carry[0], Counter(rr) + carry[1]
            n_cmp += sum((cp & cr).values())
            carry = (cp - cr, cr - cp)
            continue
        only_py += len(po)
        only_rs += len(ro_)
        for tag, extra in (("파이썬에만", po), ("Rust에만", ro_)):
            for k in sorted(extra)[: max(0, show - len(first))]:
                first.append(f"{tag} {k}")
    if not kidx:
        only_py, only_rs = sum(carry[0].values()), sum(carry[1].values())
        for tag, extra in (("파이썬에만", carry[0]), ("Rust에만", carry[1])):
            for k in sorted(extra, key=repr)[: max(0, show - len(first))]:
                first.append(f"{tag} {k}")
    rs_note = f"(의도한 차이 후보 -- {INTENDED_EXTRA[t]})" if t in INTENDED_EXTRA and only_rs else ""
    bad = n_bad + only_py + (0 if t in INTENDED_EXTRA else only_rs)
    if t == "hl_xyz_ctx_1s" and only_py + only_rs <= 0.005 * n_cmp:   # 받은 초가 1칸 밀린 합침 -- §5 의 0.5% 기준
        bad = n_bad
    TALLY[0] += n_cmp
    TALLY[1] += bad
    line = (f"{t}: 비교 {n_cmp:,}행 · 불일치 {n_bad:,} · 파이썬에만 {only_py:,} · Rust에만 {only_rs:,}{rs_note} · 범위 [{s0}, {s1}) 초 · "
            f"뺀 (종목,분) {len(skipped):,}{down_note(down, n_down)}{f' · 준비 구간(warm) 봉 {n_warm:,} 뺌' if n_warm else ''}")
    return line + "".join(f"\n    {x}" for x in first)


def run(py_path: Path, rs_path: Path, lo=None, hi=None, margin=5, window=3600, show=3, starts=()) -> list[str]:
    py, rs = ro(py_path), ro(rs_path)
    try:
        excl = excluded_minutes(py, rs, margin)
        common = sorted(tables(py) & tables(rs) & TABLES.keys())
        down = down_intervals(sql_times(rs, [(t, TABLES[t][0], TABLES[t][1]) for t in common]), list(starts), margin) if starts else []
        return [compare_table(py, rs, t, lo, hi, margin, window, show, excl, down) for t in common] or ["비교할 표가 없다"]
    finally:
        py.close()
        rs.close()


def bisect_times(ts: list):
    """정렬된 Rust 시각(초) 목록 → down_intervals 의 times(T)."""
    import bisect

    def times(T):
        i = bisect.bisect_left(ts, T)
        return (ts[i - 1] if i else None), (ts[i] if i < len(ts) else None)
    return times


def compare_depth(py_dir: Path, rs_dir: Path, lo: str = "", hi: str = "~", starts=(), margin: int = 5) -> list[str]:
    """depth_diff 시 파일(<폴더>/<SYM>/<UTC 시>.jsonl[.gz]) 대조: 시마다 거래소 프레임 줄(첫 줄 스냅숏·_gap 줄 빼고)을 글자 그대로 집합 비교.
    스냅숏은 두 쪽이 다른 순간에 REST 를 불러 원래 다르다(Rust 는 기본 꺼짐 = null) · _gap 은 각자 연결 시각에 달렸다 -- 둘 다 센 수만 적는다."""
    import gzip

    def read(d: Path) -> dict:
        out: dict = {}
        for p in sorted(d.glob("*.jsonl*")):
            h = p.name.split(".")[0]
            if not (lo <= h <= hi):
                continue
            raw = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
            lines = [ln for ln in raw.decode("utf-8", "replace").split("\n") if ln]
            out.setdefault(h, []).extend(lines)
        return out
    import re
    ev = re.compile(r'"E":(\d+)')

    def sec(x: str):
        m = ev.search(x)
        return int(m.group(1)) // 1000 if m else None
    a, b = read(py_dir), read(rs_dir)
    down = down_intervals(bisect_times(sorted(t for h in b for x in b[h] if (t := sec(x)) is not None)), list(starts), margin) if starts else []
    n_down = 0

    def frames(lines):
        nonlocal n_down
        out = Counter()
        for x in lines:
            if x.startswith('{"_'):
                continue
            t = sec(x)
            if t is not None and any(lo_ <= t < hi_ for lo_, hi_ in down):
                n_down += 1
                continue
            out[x] += 1
        return out
    res = []
    for h in sorted(a.keys() & b.keys()):
        fa, fb = frames(a[h]), frames(b[h])
        ga, gb = sum(x.startswith('{"_gap"') for x in a[h]), sum(x.startswith('{"_gap"') for x in b[h])
        sa, sb = sum(x.startswith('{"_snapshot":{') for x in a[h]), sum(x.startswith('{"_snapshot":{') for x in b[h])
        TALLY[0] += sum((fa & fb).values())
        TALLY[1] += sum((fa - fb).values()) + sum((fb - fa).values())
        res.append(f"depth {h}: 같은 프레임 {sum((fa & fb).values()):,} · 파이썬에만 {sum((fa - fb).values()):,} · Rust에만 {sum((fb - fa).values()):,} · "
                   f"_gap 파이썬 {ga}·Rust {gb} · 스냅숏 있음 파이썬 {sa}·Rust {sb}")
    return (res + [f"depth:{down_note(down, n_down)}"] if down else res) or ["depth: 두 쪽에 같은 시 파일이 없다"]


def compare_raster(py_dir: Path, rs_dir: Path, lo: str = "", hi: str = "~", starts=(), margin: int = 5) -> list[str]:
    """호가 래스터 시 파일(<폴더>/<SYM>/<UTC 시>.f32) 대조: 같은 초에 두 쪽 다 유효한 행의 bin_lo·mid 같음 · 잔량 칸 같음 비율.
    🔴바이트 일치는 기대하지 않는다 -- 두 프로세스가 스냅숏을 다른 순간에 받고(병합 범위가 다르다) 틱이 각자 벽시계라 같은 초의 북이 몇 이벤트 갈린다.
    통과 기준(제안): 두 쪽 유효 초의 99% 이상에서 bin_lo·mid 같음 · 칸 같음 비율 중앙 0.95 이상. 무효 초 수는 따로 적는다."""
    import struct

    def rows(p: Path) -> dict:
        b = p.read_bytes()
        n = struct.unpack_from("<H", b, 6)[0]   # 머리 "FLWR" u16 판 · u16 칸 수
        row = 16 + 4 * n
        out = {}
        for k in range((len(b) - 32) // row):
            r = b[32 + k * row:32 + (k + 1) * row]
            ts, bl, mid = struct.unpack_from("<qif", r, 0)
            if ts and mid == mid:
                out[ts] = (bl, mid, r[16:])
        return out
    hours = [(pp.stem, pp, rs_dir / pp.name) for pp in sorted(py_dir.glob("*.f32")) if lo <= pp.stem <= hi and (rs_dir / pp.name).exists()]
    data = [(h, rows(pp), rows(rp)) for h, pp, rp in hours]
    down = down_intervals(bisect_times(sorted(t // 1000 for _, _, b in data for t in b)), list(starts), margin) if starts else []
    res, n_down = [], 0
    for h, a, b in data:
        both = [t for t in sorted(a.keys() & b.keys()) if not any(x <= t // 1000 < y for x, y in down)]
        n_down += len(a.keys() & b.keys()) - len(both)
        same_hdr = sum(a[t][:2] == b[t][:2] for t in both)
        fr, near, l1 = [], [], []
        for t in both:
            if a[t][:2] != b[t][:2]:
                continue
            qa, qb = [x[0] for x in struct.iter_unpack("<f", a[t][2])], [x[0] for x in struct.iter_unpack("<f", b[t][2])]
            n, c = len(qa), len(qa) // 2
            fr.append(sum(x == y for x, y in zip(qa, qb)) / n)
            near.append(sum(qa[j] == qb[j] for j in range(c - 20, c + 20)) / 40)   # mid 칸 = n/2(raster.rs _row) · ±20칸 = diff 가 늘 갱신하는 곳
            l1.append(sum(abs(x - y) for x, y in zip(qa, qb)) / (sum(abs(x) + abs(y) for x, y in zip(qa, qb)) or 1))
        med = lambda v: sorted(v)[len(v) // 2] if v else None   # noqa: E731
        TALLY[0] += len(both)
        TALLY[1] += int(bool(both) and (same_hdr < 0.99 * len(both) or med(near) is None or med(near) < 0.95))   # 기준(제안): bin_lo·mid 99% · mid ±20칸 같음 중앙 0.95
        res.append(f"raster {h}: 두 쪽 유효 초 {len(both):,} · bin_lo·mid 같음 {same_hdr:,} · 칸 같음 비율 중앙 전체 {med(fr)} · mid ±20칸 {med(near)} · "
                   f"잔량 상대차(L1) 중앙 {med(l1)} · 파이썬만 유효 {len(a.keys() - b.keys()):,} · Rust만 유효 {len(b.keys() - a.keys()):,}")
    return (res + [f"raster:{down_note(down, n_down)}"] if down else res) or ["raster: 두 쪽에 같은 시 파일이 없다"]


def compare_events(py: Path, rs: Path, lo=None, hi=None, show: int = 3) -> list[str]:
    import json

    if lo is None:   # 기본 = Rust 첫 사건부터(그림자가 뜨기 전 파이썬 사건은 견주지 않는다)
        lo = min((json.loads(ln)["t_end"] for ln in rs.read_text(encoding="utf-8").splitlines() if ln.strip()), default=None)

    def read(p: Path) -> Counter:
        c: Counter = Counter()
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip() and (lo is None or json.loads(ln)["t_end"] >= lo) and (hi is None or json.loads(ln)["t_end"] < hi):
                c[ln] += 1
        return c
    a, b = read(py), read(rs)
    TALLY[0] += sum((a & b).values())
    TALLY[1] += sum((a - b).values()) + sum((b - a).values())
    first = [f"파이썬에만 {x}" for x in sorted(a - b)[:show]] + [f"Rust에만 {x}" for x in sorted(b - a)[:show]]
    return [f"events: 같은 줄 {sum((a & b).values()):,} · 파이썬에만 {sum((a - b).values()):,} · Rust에만 {sum((b - a).values()):,}"
            + "".join(f"\n    {x}" for x in first)]


def compare_json(py_dir: Path, rs_dir: Path, glob: str, drop: str) -> list[str]:
    """두 폴더의 같은 이름 JSON(glob) -- drop 이름의 열쇠를 어느 깊이에서든 빼고 같은지(청산 탭 = generated_ms)."""
    import json

    def strip(v):
        if isinstance(v, dict):
            return {k: strip(x) for k, x in v.items() if k != drop}
        return [strip(x) for x in v] if isinstance(v, list) else v
    res = []
    for rp in sorted(rs_dir.glob(glob)):
        pp = py_dir / rp.name
        if not pp.exists():
            TALLY[1] += 1
            res.append(f"{rp.name}: 파이썬 쪽 없음")
            continue
        same = strip(json.loads(pp.read_text())) == strip(json.loads(rp.read_text()))
        TALLY[0] += 1
        TALLY[1] += 0 if same else 1
        res.append(f"{rp.name}: {'같음' if same else '다름'}")
    return res or [f"json: Rust 쪽에 {glob} 가 없다"]


def selftest() -> None:
    tape = ("CREATE TABLE trade_tape_1s(symbol TEXT, ts_sec INTEGER, price_bin INTEGER, buy_qty REAL, sell_qty REAL, "
            "PRIMARY KEY (symbol, ts_sec, price_bin)) WITHOUT ROWID")
    ddl = [tape, "CREATE TABLE gaps(symbol TEXT, from_ms INTEGER, to_ms INTEGER, reason TEXT)",
           "CREATE TABLE verify_1m(symbol TEXT, ts_min INTEGER, tape_qty REAL, kline_qty REAL, rel_err REAL, checked_at TEXT)",
           "CREATE TABLE backfill_1m(symbol TEXT, ts_min INTEGER, source TEXT, n_rows INTEGER, tape_qty REAL, kline_qty REAL, "
           "rel_err REAL, ok INTEGER, attempts INTEGER, note TEXT, done_at TEXT)",
           "CREATE TABLE bybit_liquidations(ts_ms INTEGER, symbol TEXT, side TEXT, qty REAL, price REAL, recv_ms INTEGER)",
           "CREATE TABLE hl_xyz_ctx_1s(coin TEXT NOT NULL, recv_sec INTEGER NOT NULL, mid REAL, mark REAL, oracle REAL, PRIMARY KEY(coin, recv_sec))"]
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
            # 받은 초가 한쪽만 1초 늦다(벽시계) -- 값이 같으면 같은 행
            con.executemany("INSERT INTO hl_xyz_ctx_1s VALUES (?,?,?,?,?)",
                            [("xyz:XYZ100", T + s + (f == r), 30000.0 + s, 30001.0, None) for s in range(0, 1100, 10)])
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
        assert "불일치 0 · 파이썬에만 0 · Rust에만 1(의도한 차이 후보" in liq_line and "비교 38행" in liq_line, liq_line   # recv_ms 는 달라도 같다
        xyz_line = next(x for x in out if x.startswith("hl_xyz_ctx_1s"))
        # recv_sec 는 빼고 비교 · 창 경계를 넘어 맞춘다. 남는 1행 = 범위 첫 초(파이썬 T 는 범위 밖, Rust 의 같은 행은 T+1 이라 안) -- 벽시계 열의 한계
        assert "불일치 0 · 파이썬에만 0 · Rust에만 1(의도한 차이 후보" in xyz_line and "비교 107행" in xyz_line, xyz_line
        for f, w in ((p, 1), (r, 2)):
            con = sqlite3.connect(f)
            con.execute("CREATE TABLE bars(t INTEGER PRIMARY KEY, dimb1 REAL, chain_age_s REAL, warm REAL, written_ms INTEGER)")
            # 봉 0 = Rust 만 준비 구간(warm=1, 값 다름) → 뺀다 · 봉 2 값 다름 · chain_age_s 는 두 쪽이 늘 다르다(안 본다)
            con.executemany("INSERT INTO bars VALUES (?,?,?,?,?)", [(1000 * (T + 300 * k), 0.5 if (f == r and k in (0, 2)) else 0.25,
                                                                   300.0 * k + (99 if f == r else 7), 1.0 if (f == r and k == 0) else 0.0, w) for k in range(5)])
            con.commit()
            con.close()
        bars_line = next(x for x in run(p, r) if x.startswith("bars"))
        # 봉 1·2·3 견줌(봉 4 = 끝 10초 안) · 봉 0 = warm
        assert "비교 3행 · 불일치 1 · 파이썬에만 0 · Rust에만 0" in bars_line and "준비 구간(warm) 봉 1 뺌" in bars_line, bars_line
        # 그림자 꺼진 구간: Rust 가 T+400~T+560 에 비었고(지움) 감독 로그 기동 T+500 → 그 사이 파이썬에만 행은 뺀다 · T+700 의 파이썬에만 행은 그대로 실패
        con = sqlite3.connect(r)
        con.execute("DELETE FROM trade_tape_1s WHERE ts_sec BETWEEN ? AND ?", [T + 400, T + 560])   # 꺼진 프로세스는 어느 표에도 안 쓴다
        con.execute("DELETE FROM bybit_liquidations WHERE ts_ms BETWEEN ? AND ?", [1000 * (T + 400), 1000 * (T + 560)])
        con.execute("DELETE FROM hl_xyz_ctx_1s WHERE recv_sec BETWEEN ? AND ?", [T + 400, T + 560])
        con.commit()
        con.close()
        con = sqlite3.connect(p)
        con.execute("DELETE FROM trade_tape_1s WHERE ts_sec = ? AND price_bin = 25000", [T + 700])
        con.commit()
        con.close()
        TALLY[:] = [0, 0]
        dl = next(x for x in run(p, r, window=250, show=5, starts=[T + 500]) if x.startswith("trade_tape_1s"))
        assert "그림자 꺼진 구간 1개" in dl and "Rust에만 2" in dl and "파이썬에만 0" in dl, dl   # T+700 은 Rust 에만(파이썬에서 지움) + 분 2 의 그것
        dl0 = next(x for x in run(p, r, window=250, show=5) if x.startswith("trade_tape_1s"))
        assert "파이썬에만 " in dl0 and "파이썬에만 0" not in dl0, dl0   # 기동 시각을 안 주면 빈 구간이 «파이썬에만»
        # depth: Rust 가 T+100~T+130 초에 프레임이 없고 기동 T+120 → 그 사이 파이썬 프레임은 뺀다 · 바깥의 파이썬에만 프레임은 실패
        import gzip
        import struct
        for side in ("dp", "dr"):
            (Path(d) / side).mkdir()
            fr = [f'{{"e":"depthUpdate","E":{1000 * (T + i)},"u":{i}}}' for i in range(200) if not (side == "dr" and (100 <= i <= 130 or i == 170))]
            (Path(d) / side / "2026-11-21T07.jsonl.gz").write_bytes(gzip.compress(("\n".join(['{"_snapshot":null}'] + fr) + "\n").encode()))
        dd = compare_depth(Path(d) / "dp", Path(d) / "dr", starts=[T + 120])
        assert dd[0].startswith("depth 2026-11-21T07: 같은 프레임 157 · 파이썬에만 1 · Rust에만 0") and "뺀 행 53" in dd[1], dd   # 뺀 구간 = [99-5, 131+5) 초 · 파이썬 42 + Rust 11
        # raster: 머리 n 을 읽는다 · mid ±20칸만 같으면 통과 쪽
        def f32(path, vals):
            n = 60
            h = b"FLWR" + struct.pack("<HHfIqII", 1, n, 0.5, 0, 0, 1000, 0)
            rows_ = b"".join(struct.pack("<qif", 1000 * (T + k), 100, 50.0) + struct.pack(f"<{n}f", *v) for k, v in enumerate(vals))
            path.write_bytes(h + rows_)
        (Path(d) / "rp").mkdir()
        (Path(d) / "rr").mkdir()
        base = [float(j) for j in range(60)]
        far = [9.0 if (j < 5) else float(j) for j in range(60)]   # 먼 칸 5개만 다름
        f32(Path(d) / "rp/2026-11-21T07.f32", [base] * 10)
        f32(Path(d) / "rr/2026-11-21T07.f32", [far] * 10)
        TALLY[:] = [0, 0]
        rz = compare_raster(Path(d) / "rp", Path(d) / "rr")
        assert "mid ±20칸 1.0" in rz[0] and TALLY == [10, 0], (rz, TALLY)
        # 첫 기동 전(-inf)은 뺀다: 기동 T+600 하나면 그 앞 Rust 자료가 있어도 첫 기동 = 로그의 가장 이른 것
        assert down_intervals(lambda t: (None, t + 3), [t0 := T + 600], 5) == [(float("-inf"), t0 + 8)]
        (Path(d) / "a.jsonl").write_text('{"t_end":60,"side":1}\n{"t_end":90,"side":-1}\n{"t_end":120,"side":1}\n')
        (Path(d) / "b.jsonl").write_text('{"t_end":60,"side":1}\n{"t_end":120,"side":1}\n')
        ev = compare_events(Path(d) / "a.jsonl", Path(d) / "b.jsonl", lo=60, hi=120)[0]
        assert ev.startswith("events: 같은 줄 1 · 파이썬에만 1 · Rust에만 0"), ev   # t_end 120 = 창 밖
        (Path(d) / "jp").mkdir()
        (Path(d) / "jr").mkdir()
        (Path(d) / "jp/liq_tab_eth.json").write_text('{"generated_ms":1,"x":[{"generated_ms":5,"y":2}]}')
        (Path(d) / "jr/liq_tab_eth.json").write_text('{"generated_ms":2,"x":[{"generated_ms":6,"y":2}]}')
        (Path(d) / "jr/liq_tab_btc.json").write_text('{"y":1}')
        TALLY[:] = [0, 0]
        js = compare_json(Path(d) / "jp", Path(d) / "jr", "liq_tab_*.json", "generated_ms")
        assert js == ["liq_tab_btc.json: 파이썬 쪽 없음", "liq_tab_eth.json: 같음"] and TALLY == [1, 1], (js, TALLY)
        TALLY[:] = [0, 0]
        run(p, r)
        assert TALLY[1] > 0, TALLY   # 위 표들의 불일치가 요약에 잡힌다
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
    ap.add_argument("--depth", action="store_true", help="두 경로 = depth_diff 종목 폴더(<root>/<SYM>) · --from/--to 대신 --hours 시작,끝(YYYY-MM-DDTHH)")
    ap.add_argument("--hours", default="")
    ap.add_argument("--raster", action="store_true", help="두 경로 = 호가 래스터 종목 폴더(<root>/<SYM>) · --hours 시작,끝")
    ap.add_argument("--json", default="", help="두 경로 = 폴더 · 값 = glob(예: 'liq_tab_*.json') · --drop 열쇠를 빼고 비교")
    ap.add_argument("--drop", default="generated_ms")
    ap.add_argument("--starts", default="", help="그림자 Rust 감독 «SUPERVISOR starting» 시각(초, 쉼표) -- 그 앞뒤 Rust 가 비어 있던 구간을 뺀다(takeover.sh 가 로그에서 읽어 넘긴다)")
    ap.add_argument("--summary", action="store_true", help="끝에 «요약: 견준 N · 설명 못 함 M» · M>0 이거나 N=0 이면 종료코드 1(takeover.sh check)")
    ap.add_argument("--events", action="store_true", help="두 경로 = 아이스버그 <sym>_events.jsonl · --from/--to = t_end 초")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.py and a.rs):
        ap.error("파이썬 DB 와 Rust DB 경로가 필요하다")
    lo, _, hi = a.hours.partition(",")
    st = [int(x) for x in a.starts.split(",") if x.strip()]
    if a.json:
        out = compare_json(a.py, a.rs, a.json, a.drop)
    elif a.events:
        out = compare_events(a.py, a.rs, a.lo, a.hi, a.show)
    elif a.raster:
        out = compare_raster(a.py, a.rs, lo, hi or "~", st, a.margin)
    elif a.depth:
        out = compare_depth(a.py, a.rs, lo, hi or "~", st, a.margin)
    else:
        out = run(a.py, a.rs, a.lo, a.hi, a.margin, a.window, a.show, st)
    print("\n".join(out))
    if a.summary:
        print(f"요약: 견준 {TALLY[0]:,} · 설명 못 함 {TALLY[1]:,}")
        return 1 if TALLY[1] or not TALLY[0] else 0


if __name__ == "__main__":
    sys.exit(main())
