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
import bisect
import sqlite3
import time
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
    "treasury_fut_1m": ("ts_sec", False, ("symbol", "ts_sec"), ()),   # 국채선물 1분봉(Yahoo) -- 받은 시각 칸 없음
    "yahoo_5m": ("ts_sec", False, ("symbol", "ts_sec"), ()),          # macro 5분봉(Yahoo)
    "items": ("pub_ms", True, ("source", "link"), ("first_seen_ms", "backfill")),   # 뉴스 -- 처음 본 시각·메움 표시는 각자
    "bars": ("t", True, ("t",), ("written_ms", "chain_age_s")),   # chain_age_s = 사슬 시작부터 초(각자 띄운 때에 달렸다 -- 진단 칸)
    "marks": ("t", True, ("t", "side", "px"), ()),
}
# 🔧의도한 차이(rtcollect/SHADOW.md §11 -- 결함은 Rust 에서만 고쳤다): 이 표들의 «Rust에만» 행은 원본이 잃는 행일 수 있다 -- 불일치로 세지 않고 따로 적는다.
#   «파이썬에만»·값 불일치는 그대로 실패다(고친 Rust 가 원본보다 적게 쓸 까닭은 없다).
INTENDED_EXTRA = {
    "bars": "Rust 는 기동 때 depthdiff 시 파일(최대 26시간)을 다시 읽어 파이썬이 재시작 틈으로 못 쓴 봉을 채운다",
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
# 받은 시각(초·ms)을 빼면 같은 값이 여러 번 찍힌 줄 -- 몇 번 찍혔는지는 각자 받은 초에 달렸다 → 값 묶음의 «종류»(집합)로 견준다
#   (xyz ctx: 파이썬에만 2,219 · Rust에만 1,024 가 대부분 같은 값의 찍힌 횟수 차였다)
BAR_W = {"hl_xyz_1m": 60}
# 1분봉 «같은 분의 앞선 스냅숏»: 거래소가 연결마다 candle 갱신을 묶어 보내 한 연결은 그 분 마지막 갱신(체결 1~수백 건)을 못 받는다
#   (11-28 서버 7.5일: 값 다른 79봉 전부 이 꼴 · Rust 가 앞선 50·파이썬이 앞선 29 -- 양쪽 다 · REST 덮은 파이썬 봉과 Rust 는 644 중 636 같음)
SNAP_TABLES = {"hl_xyz_1m"}
PY_REST: list = []   # --py-rest-log: 원본 로그의 «접속 · 종목 REST 메움 N봉»(초, 종목, N) -- 그 N봉은 웹소켓이 아니라 REST 값(그림자 Rust 는 REST 꺼짐)


def earlier(a: tuple, b: tuple) -> bool:
    """a 가 b 와 같은 분의 앞선 스냅숏인가(열 = coin, ts_sec, open, high, low, close, volume, n): 시가 같음 · 체결 수 적음 · 거래량·고가 안 큼 · 저가 안 작음."""
    return a[2] == b[2] and a[7] < b[7] and a[6] <= b[6] and a[3] <= b[3] and a[4] >= b[4]


def rest_minutes(py) -> dict:
    """{종목: {분}} -- 원본이 REST 로 쓴 봉 = 그 로그 줄 시각까지 그 종목 파이썬 봉 마지막 N개."""
    out: dict = {}
    for t, coin, n in PY_REST:
        for (s,) in py.execute("SELECT ts_sec FROM hl_xyz_1m WHERE coin = ? AND ts_sec <= ? ORDER BY ts_sec DESC LIMIT ?", [coin, int(t), n]):
            out.setdefault(coin, set()).add(s // 60)
    return out


def read_rest_log(pattern: str) -> list:
    import glob
    import re
    out = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f, errors="replace"):
            m = re.match(r"\[xyz100\] (\S+ \S+) 접속 · (\S+) REST 메움 (\d+)봉", line)
            if m and int(m[3]):
                out.append((time.mktime(time.strptime(m[1], "%Y-%m-%d %H:%M:%S")), m[2], int(m[3])))   # 로그는 서버 지역 시각
    return out
SET_TABLES = {"hl_xyz_ctx_1s", "hl_asset_ctx"}
SYMS: set = set()   # --syms: 이 종목만 견준다(hl_bt 코인 하나 -- 다른 코인 맥락은 다른 그림자·다른 재기동이라 hl_ctx check 몫)
SUBSET = False   # --subset: Rust 가 받은 것이 파이썬에 있나만(파이썬이 오래 모은 «파이썬에만»은 세지 않음 -- 뉴스 한 회차)
SYM_COL = {"treasury_fut_1m": "symbol", "yahoo_5m": "symbol", "items": "source","trade_tape_1s": "symbol", "bybit_liquidations": "symbol", "bybit_oi": "inst", "bybit_mark": "inst", "bybit_funding": "inst",
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


def holes(ts: list, starts: list, margin: int) -> list[tuple[float, float]]:
    """그림자 Rust 가 꺼져 있던 구간(초). ts = 그 프로세스가 쓴 자료의 시각(정렬 · 가장 촘촘한 표 하나면 충분 -- 프로세스가 꺼지면 모든 표가 빈다).
    starts = 감독 로그의 «SUPERVISOR starting» 시각.
      첫 기동: (-inf, 그 뒤 첫 자료] -- 그림자가 뜨기 전은 전부(구독 때 받는 «지난 사건»의 옛 ts 에 속지 않게 «앞 자료»는 안 본다)
      재기동 T: 앞 기동 뒤 ~ T+CAP 안에서 끝이 T−CAP 뒤인 **가장 큰 빈틈**(자료 사이) -- 죽인 때(K)는 로그에 없고,
        다시 붙으면 거래소가 «최근 체결 몇 개»를 다시 보내 T 바로 앞까지 메우므로 «T 앞 마지막 자료»로는 K 를 못 찾는다(HL 체결 23:09 BTC 436행)."""
    import bisect
    st, out = sorted(starts), []
    for i, T in enumerate(st):
        j = bisect.bisect_left(ts, T)
        nxt = ts[j] if j < len(ts) else float("inf")   # T 뒤 첫 자료(없으면 끝까지)
        if i == 0:   # 첫 기동: 처음 쓴 때까지 전부 -- 감독이 곧바로 죽고 다시 뜨기를 몇 시간 되풀이했어도(11-21 OKX 맥락 00:56~10:53)
            out.append((float("-inf"), nxt + margin))
            continue
        seg = ts[bisect.bisect_left(ts, st[i - 1]):bisect.bisect_right(ts, T + CAP)]
        best = (st[i - 1], nxt) if not seg else None
        cands = list(zip(seg, seg[1:])) + ([(seg[-1], nxt)] if seg and seg[-1] < T else [])
        for a, b in cands:
            if b >= T - CAP and (best is None or b - a > best[1] - best[0]):
                best = (a, b)
        out.append((best[0] - margin, best[1] + margin) if best else (T - margin, T + margin))
    return merge(out)


def silences(a_ts: list, b_ts: list, gap: float, skip: list) -> list:
    """a 쪽 자료 사이 빈틈 중 gap 초보다 길고 그 안에 b 쪽 자료가 있는 것(= a 만 비었다) -- skip(재기동 구간)과 겹치면 뺀다. 긴 것부터."""
    import bisect
    out = []
    for x, y in zip(a_ts, a_ts[1:]):
        if y - x > gap and not any(p < y and x < q for p, q in skip):
            k = bisect.bisect_right(b_ts, x)
            if k < len(b_ts) and b_ts[k] < y:
                out.append((x, y))
    return sorted(out, key=lambda g: g[0] - g[1])


EPISODE_S = 10   # 한 끊김 = 한 일화: 사이가 이 초보다 짧은 빈틈들(다시 붙은 직후 드문 줄 1~2개가 끼어 쪼개진 것)은 하나로 -- 11-23 14:00 xyz 1분 정지가 3·7·…초 빈틈 여럿으로 쪼개져 «끊김» 한 줄이 한 조각만 설명


def episodes(gs: list) -> list:
    """빈틈들 → 일화(사이 < EPISODE_S 면 잇는다) · 긴 것부터."""
    out: list = []
    for a, b in sorted(gs):
        if out and a - out[-1][1] < EPISODE_S:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return sorted(out, key=lambda g: g[0] - g[1])


def merge(out: list) -> list:
    merged: list = []
    for a, b in sorted(out):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


RASTER_L1 = 0.02
RASTER_KEEP = 0.035   # 래스터 기준 칸 = mid ±3.5%(raster.rs PRUNE_PCT 4% − 가지치기 사이 mid 이동 0.5%p) -- compare_raster
RECONNECT_MS = 3000   # .bt 연결 끊김 빈틈(재접속 전 3초 쉼 · 두 구현 같음)
SKIP_RUN = 3   # .bt 연결마다 다른 표본: 공통 행 사이 한쪽만의 행 수 상한(다른 쪽도 그 칸에 제 행이 없을 때 · compare_bt)
EXT = 30    # 빈틈 뒤 종목마다 다시 붙는 늦음은 30초까지만(실측 bybit BTC mark ~10초) -- 드문 표가 빈틈 뒤를 통째로 빼지 않게
CAP = 300   # ponytail: 재기동 빈틈은 T 앞뒤 300초 안에서 찾는다(드문 종목 하나 때문에 몇 시간을 빼지 않게) -- 재기동이 더 길면 늘린다


def down_note(d: list, n: int) -> str:
    if not d:
        return ""
    fin = sum(b - a for a, b in d if a > float("-inf") and b < float("inf"))
    return f" · 그림자 꺼진 구간 {len(d)}개(첫 기동 전 포함 · 유한 합 {fin:.0f}초)에서 뺀 행 {n:,}"


def sec_expr(tcol: str, ms: bool) -> str:
    return f"({tcol} / 1000)" if ms else tcol


def proc_holes(con, tabs: list, starts: list, margin: int, py=None, reconnects: int = 0) -> tuple[list, str]:
    """sqlite: 가장 행이 많은 표의 초 목록으로 (뺄 구간, 설명 글).
      재기동 구간 = holes() · 연결 끊김(재기동 아님):
        Rust 만 빈 곳 = 그림자 Rust 의 끊김 -- 감독 로그의 «끊김» 줄 수(reconnects)만큼만 긴 것부터 설명으로 친다(로그보다 많으면 나머지는 설명 못 함)
        파이썬만 빈 곳 = 원본의 끊김 -- 그 안의 «Rust 에만» 은 Rust 가 더 받은 것(둘 다 받은 대로 적음)."""
    if not tabs:
        return [], ""
    t, tcol, ms = max(tabs, key=lambda x: con.execute(f"SELECT count(*) FROM {x[0]}").fetchone()[0])
    w = f" WHERE {SYM_COL[t]} IN ({','.join('?' * len(SYMS))})" if SYMS and t in SYM_COL else ""
    q = f"SELECT DISTINCT {sec_expr(tcol, ms)} FROM {t}{w} ORDER BY 1"
    ts = [r[0] for r in con.execute(q, sorted(SYMS) if w else [])]
    hs = holes(ts, starts, margin) if starts else []
    if py is None:
        return hs, ""
    pts = [r[0] for r in py.execute(q, sorted(SYMS) if w else [])]
    d = sorted(b - a for a, b in zip(ts, ts[1:]))
    gap = max(5.0, 10.0 * (d[len(d) // 2] if d else 1.0))
    if t.startswith("hl_"):   # HL 맥락(activeAssetCtx·xyz ctx)은 매 블록 ~1/초로 끊이지 않는다 -- 재접속 전 3초 쉼(두 구현 같음)이 만든 빈틈(초 차 ≥3)을 잡는다(11-22: 10초 문턱이라 «끊김» 134줄이 0번 설명)
        gap = 2.0
    rsil, psil = episodes(silences(ts, pts, gap, hs)), silences(pts, ts, gap, hs)
    took = rsil[:reconnects]
    note = (f" · 연결 끊김(촘촘한 표 {t} 에서 {gap:.0f}초 넘게 한쪽만 빔): Rust {len(rsil)}번(그림자 로그 «끊김» {reconnects}줄 → {len(took)}번 설명 · 합 {sum(b - a for a, b in took):.0f}초)"
            f" · 파이썬 {len(psil)}번(합 {sum(b - a for a, b in psil):.0f}초)") if rsil or psil else ""
    return merge(hs + [(a, b) for a, b in took] + [(a, b) for a, b in psil]), note


def sym_down(con, t: str, tcol: str, ms: bool, symcol, proc: list, margin: int) -> dict:
    """표 하나의 뺄 구간을 종목마다({종목: 구간}, 종목 열이 없으면 {None: …}):
    프로세스 빈틈(proc · 첫 기동 전 포함) + 종목마다 «그 빈틈 뒤 그 종목 첫 자료»까지(종목마다 다시 붙는 때가 다르다 · EXT 까지).
    기동 시각(proc)이 없을 때만 «그 종목 Rust 첫 자료 전» 전부 -- 있으면 그렇게 빼지 않는다(11-28: 드문 표 OKX 청산 536종목은
    «그 종목 Rust 첫 청산 전» 파이썬 행 2,577 이 통째로 빠져 Rust 가 못 받은 것을 숨겼다)."""
    x = sec_expr(tcol, ms)
    g, sel = (f" GROUP BY {symcol}", f"{symcol}, ") if symcol else ("", "NULL, ")
    first = dict(con.execute(f"SELECT {sel}min({x}) FROM {t}{g}").fetchall())
    out = {s: list(proc) if proc else [(float("-inf"), f0)] for s, f0 in first.items() if f0 is not None}
    for a, b in proc:
        if b == float("inf"):
            continue
        late = dict(con.execute(f"SELECT {sel}min({x}) FROM {t} WHERE {x} >= ?{g}", [int(b - margin)]).fetchall())
        for s in out:
            f1 = late.get(s)
            out[s].append((a, min(f1 if f1 is not None else b + EXT, b + EXT) + margin))
    return {s: merge(v) for s, v in out.items()}


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


def compare_table(py, rs, t: str, lo: int | None, hi: int | None, margin: int, window: int, show: int, excl: dict, proc=()) -> str:
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
    n_cmp = n_bad = only_py = only_rs = n_down = n_warm = n_snap = 0
    skipped: set = set()
    down = sym_down(rs, t, tcol, ms, SYM_COL.get(t), list(proc), margin)
    wi = names.index("warm") if "warm" in names else None
    knames = [names[i] for i in keepcols]
    # 호가 깊이 10열 원값 = 원본이 BLAS 내적(서버 OpenBLAS) -- 끝자리가 기계·합 차례마다 다르다(ob5m_parity 930칸 1e-12 안) → 상대 1e-9 안이면 같음
    rel = 1e-9 if "chain_age_s" in names else 0.0
    worst = 0.0
    po_min: Counter = Counter()

    def same(x, y) -> bool:
        if rel and isinstance(x, float) and isinstance(y, float):
            return x == y or abs(x - y) <= rel * max(abs(x), abs(y))
        return x == y   # 준비 구간 봉(warm=1)은 한쪽이라도 그러면 견주지 않는다(사슬 시작에 달렸다)
    first: list = []
    carry: tuple = (Counter(), Counter())
    sp_: set = set()
    sr_: set = set()
    matched: set = set()
    sec_p: dict = {}
    sec_r: dict = {}
    cs_p: set = set()
    cs_r: set = set()   # 묶음 표: 두 쪽에 다 있던 줄 -- 남은 «한쪽에만»이 그 줄의 또 한 벌이면 «같은 줄 두 번»(받기 차이 · 값 정보는 같다)

    def keep(r) -> bool:
        nonlocal n_down
        sec = r[ti] // 1000 if ms else r[ti]
        sym = r[si] if si is not None else None
        if SYMS and si is not None and sym not in SYMS:
            return False
        w = BAR_W.get(t, 1)   # 봉 표: 그 봉 [시작, 시작+폭) 이 뺄 구간에 걸치면 뺀다(끊김이 걸친 1분봉은 받은 체결 수가 갈린다 -- SHADOW.md §5)
        if any(a < sec + w and sec < b for a, b in down.get(sym, proc)):   # Rust 표에 한 번도 없는 종목(OKX 청산 ADBE 등)도 프로세스 빈틈은 뺀다
            n_down += 1
            return False
        m = sec // 60
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
        if t in SET_TABLES:   # 받은 초(와 종목)를 따로 둔다 -- 아래 «같은 초에 둘 다 받았다» 판정용
            for side, rows_ in ((sec_p, rp), (sec_r, rr)):
                for r in rows_:
                    sc = r[ti] // 1000 if ms else r[ti]
                    side.setdefault(tuple(r[i] for i in keepcols), []).append(sc)
                    (cs_p if side is sec_p else cs_r).add((r[si], sc))
        if not kidx:
            rp, rr = [tuple(r[i] for i in keepcols) for r in rp], [tuple(r[i] for i in keepcols) for r in rr]
        if kidx:
            if wi is not None:
                wp, wr = {tuple(r[i] for i in kidx) for r in rp if r[wi] == 1}, {tuple(r[i] for i in kidx) for r in rr if r[wi] == 1}
                n_warm += len(wp | wr)
                rp, rr = [r for r in rp if tuple(r[i] for i in kidx) not in wp | wr], [r for r in rr if tuple(r[i] for i in kidx) not in wp | wr]
            tp_ = {tuple(r[i] for i in kidx): (r[ti] // 1000 if ms else r[ti]) for r in rp}
            dp = {tuple(r[i] for i in kidx): tuple(r[i] for i in keepcols) for r in rp}
            dr = {tuple(r[i] for i in kidx): tuple(r[i] for i in keepcols) for r in rr}
            for k in dp.keys() & dr.keys():
                n_cmp += 1
                diff = [(kn, x, y) for kn, x, y in zip(knames, dp[k], dr[k]) if not same(x, y)]
                if diff and t in SNAP_TABLES and (earlier(dp[k], dr[k]) or earlier(dr[k], dp[k])):
                    n_snap += 1
                elif diff:
                    n_bad += 1
                    if len(first) < show:
                        first.append(f"다름 {k}: " + " · ".join(f"{kn} 파이썬 {x} / Rust {y}" for kn, x, y in diff[:4]) + (f" … 외 {len(diff) - 4}칸" if len(diff) > 4 else ""))
                elif rel and dp[k] != dr[k]:
                    worst = max(worst, max(abs(x - y) / max(abs(x), abs(y)) for x, y in zip(dp[k], dr[k]) if isinstance(x, float) and isinstance(y, float) and x != y))
            po, ro_ = dp.keys() - dr.keys(), dr.keys() - dp.keys()
            po_min.update((tp_[k] // 60) * 60 for k in po)
        elif t in SET_TABLES:
            sp_.update(rp)
            sr_.update(rr)
            continue
        else:
            # 행 묶음: 못 맞춘 행은 다음 창으로 넘긴다(창 경계에 걸린 행 · 받은 시각이 한쪽만 늦은 행)
            cp, cr = Counter(rp) + carry[0], Counter(rr) + carry[1]
            n_cmp += sum((cp & cr).values())
            matched.update(cp & cr)
            carry = (cp - cr, cr - cp)
            continue
        only_py += len(po)
        only_rs += len(ro_)
        for tag, extra in (("파이썬에만", po), ("Rust에만", ro_)):
            for k in sorted(extra)[: max(0, show - len(first))]:
                first.append(f"{tag} {k}")
    same_sec = [0, 0]
    if t in SET_TABLES:
        n_cmp, carry = len(sp_ & sr_), (Counter(sp_ - sr_), Counter(sr_ - sp_))
        # 같은 초에 둘 다 받았는데 값만 다름 = 거래소가 연결마다 맥락 값을 묶어 보낸다(xyz ctx 는 (coin, recv_sec) INSERT OR REPLACE 라 그 초 마지막 값만 남는다)
        #   -- 그 줄의 종목·초(±1)에 다른 쪽도 줄이 있으면 설명(받기는 했다) · 다른 쪽이 그 초에 아무것도 없으면 설명 못 함(못 받음)
        kpos = keepcols.index(si) if si in keepcols else None
        for k in carry[0]:
            if kpos is not None and any((k[kpos], x + d) in cs_r for x in sec_p.get(k, ()) for d in (-1, 0, 1)):
                same_sec[0] += 1
        for k in carry[1]:
            if kpos is not None and any((k[kpos], x + d) in cs_p for x in sec_r.get(k, ()) for d in (-1, 0, 1)):
                same_sec[1] += 1
    if not kidx:
        only_py, only_rs = sum(carry[0].values()), sum(carry[1].values())
        for tag, extra in (("파이썬에만", carry[0]), ("Rust에만", carry[1])):
            for k in sorted(extra, key=repr)[: max(0, show - len(first))]:
                first.append(f"{tag} {k}")
    dup_py = sum(c for k, c in carry[0].items() if k in matched) if not kidx and t not in SET_TABLES else 0
    dup_rs = sum(c for k, c in carry[1].items() if k in matched) if not kidx and t not in SET_TABLES else 0
    # 재구독 현재값: 다시 붙은 쪽은 거래소가 «지금 값»을 한 번 더 보낸다(OKX funding-rate 등) -- 값이 그 종목 직전 파이썬 줄과 같으면 새 정보가 아니다
    resub = 0
    if not kidx and si is not None and carry[1]:
        tpos = keepcols.index(ti) if ti in keepcols else None
        spos = keepcols.index(si)
        for k, c in list(carry[1].items())[:10000]:
            if k in matched or tpos is None:
                continue
            prev = py.execute(f"SELECT {', '.join(knames)} FROM {t} WHERE {SYM_COL[t]} = ? AND {tcol} <= ? ORDER BY {tcol} DESC LIMIT 1", [k[spos], k[tpos]]).fetchone()
            if prev and all(x == y for j, (x, y) in enumerate(zip(prev, k)) if j != tpos):
                resub += c
    rs_note = f"(의도한 차이 후보 -- {INTENDED_EXTRA[t]})" if t in INTENDED_EXTRA and only_rs else ""
    hot = (" · 파이썬에만 많은 분(서버 시각) " + ", ".join(f"{time.strftime('%F %R', time.localtime(m))}({c})" for m, c in po_min.most_common(3))) if po_min else ""
    bad = n_bad + (0 if SUBSET else only_py - dup_py - same_sec[0]) + (0 if t in INTENDED_EXTRA else only_rs - dup_rs - resub - same_sec[1])
    if t == "hl_xyz_ctx_1s" and only_py + only_rs <= 0.005 * n_cmp:   # 받은 초가 1칸 밀린 합침 -- §5 의 0.5% 기준
        bad = n_bad
    if n_snap > 0.005 * n_cmp:   # 앞선 스냅숏이 0.5% 를 넘으면 연결 잡음이 아니다 -- 전부 설명 못 함
        bad += n_snap
    TALLY[0] += n_cmp
    TALLY[1] += bad
    line = (f"{t}: 비교 {n_cmp:,}행 · 불일치 {n_bad:,} · 파이썬에만 {only_py:,} · Rust에만 {only_rs:,}{rs_note} · 범위 [{s0}, {s1}) 초 · "
            f"뺀 (종목,분) {len(skipped):,}{down_note(list(proc), n_down)}"
            f"{f' · 끝자리 차(상대 1e-9 안) 최대 {worst:.1e}' if worst else ''}"
            f"{hot}{f' · 준비 구간(warm) 봉 {n_warm:,} 뺌' if n_warm else ''}"
            f"{f' · 같은 줄 두 번(설명) 파이썬 {dup_py:,}·Rust {dup_rs:,}' if dup_py or dup_rs else ''}"
            f"{f' · Rust 재구독 현재값(직전 파이썬 줄과 같은 값 · 설명) {resub:,}' if resub else ''}"
            f"{f' · 같은 분의 앞선 스냅숏(한 연결이 마감 갱신을 못 받음 · 0.5% 안이면 설명) {n_snap:,}' if n_snap else ''}"
            f"{f' · 같은 초(±1)에 다른 쪽도 받음(값만 다름 · 설명) 파이썬에만 {same_sec[0]:,}·Rust에만 {same_sec[1]:,}' if any(same_sec) else ''}")
    return line + "".join(f"\n    {x}" for x in first)


def run(py_path: Path, rs_path: Path, lo=None, hi=None, margin=5, window=3600, show=3, starts=(), reconnects: int = 0) -> list[str]:
    py, rs = ro(py_path), ro(rs_path)
    try:
        excl = excluded_minutes(py, rs, margin)
        common = sorted(tables(py) & tables(rs) & TABLES.keys())
        proc, note = proc_holes(rs, [(t, TABLES[t][0], TABLES[t][1]) for t in common], list(starts), margin, py, reconnects)
        rm = rest_minutes(py) if PY_REST and "hl_xyz_1m" in common else {}
        ex1 = {s: excl.get(s, set()) | rm.get(s, set()) for s in excl.keys() | rm.keys()}   # REST 로 쓴 분은 1분봉만 뺀다(맥락 줄은 웹소켓 그대로)
        return ([note.lstrip(" ·")] if note else []) + [compare_table(py, rs, t, lo, hi, margin, window, show, ex1 if t == "hl_xyz_1m" else excl, proc) for t in common] or ["비교할 표가 없다"]
    finally:
        py.close()
        rs.close()


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
    down = holes(sorted({t for h in b for x in b[h] if (t := sec(x)) is not None}), list(starts), margin) if starts else []
    n_down = 0
    ua, pa = re.compile(r'"u":(\d+)'), re.compile(r'"pu":(\d+)')

    def reconnects(d: dict) -> list:
        """그 쪽이 다시 붙은 자리(ms): 이웃 프레임 둘 사이에 _gap 줄이 없는데 뒤 프레임의 pu ≠ 앞 프레임의 u.
        원본 수집기는 연결 안에서 pu 가 끊기면 _gap 줄을 쓰고, 다시 붙을 때는 prev_u 를 비워 _gap 을 안 쓴다(depth.rs on_connect = 원본 :136)
        → 이 모양 = 그 쪽 연결이 끊겼다 다시 붙음. 그 사이 다른 쪽만 받은 프레임은 «받는 쪽 연결 차이»(둘 다 받은 대로 적음 · 읽는 쪽은 스냅숏으로 다시 맞춘다)."""
        out, prev, gap = [], None, False
        for h in sorted(d):
            for x in d[h]:
                if x.startswith('{"_gap"'):
                    gap = True
                    continue
                if x.startswith('{"_'):
                    continue
                e, u, q = ev.search(x), ua.search(x), pa.search(x)
                if not (e and u and q):
                    continue
                cur = (int(e.group(1)), int(u.group(1)), int(q.group(1)))
                if prev and not gap and cur[2] != prev[1]:
                    out.append((prev[0], cur[0]))
                prev, gap = cur, False
        return out
    ra, rb = reconnects(a), reconnects(b)   # 파이썬 쪽 · Rust 쪽 다시 붙은 자리
    n_rc = [0, 0]

    def in_win(x: str, wins) -> bool:
        e = ev.search(x)
        return bool(e) and any(p < int(e.group(1)) < q for p, q in wins)

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
        ex_py = sum(c for x, c in (fa - fb).items() if in_win(x, rb))   # Rust 가 다시 붙는 사이 파이썬만 받은 프레임
        ex_rs = sum(c for x, c in (fb - fa).items() if in_win(x, ra))   # 파이썬이 다시 붙는 사이 Rust 만 받은 프레임
        n_rc[0] += ex_py
        n_rc[1] += ex_rs
        TALLY[1] += sum((fa - fb).values()) + sum((fb - fa).values()) - ex_py - ex_rs
        pt = sorted(t for x in (fa - fb) if (t := sec(x)) is not None)
        when = f" · 파이썬에만 프레임 시각(서버) {time.strftime('%T', time.localtime(pt[0]))}~{time.strftime('%T', time.localtime(pt[-1]))}" if pt else ""
        res.append(f"depth {h}: 같은 프레임 {sum((fa & fb).values()):,} · 파이썬에만 {sum((fa - fb).values()):,} · Rust에만 {sum((fb - fa).values()):,} · "
                   f"_gap 파이썬 {ga}·Rust {gb} · 스냅숏 있음 파이썬 {sa}·Rust {sb}{when}")
    rc = (f"depth: 다시 붙은 자리 Rust {len(rb)}·파이썬 {len(ra)}(_gap 없이 pu 끊김) -- 그 사이 한쪽만 받은 프레임은 설명: 파이썬에만 {n_rc[0]:,}·Rust에만 {n_rc[1]:,}"
          if ra or rb else "")
    res += [x for x in (rc, f"depth:{down_note(down, n_down)}" if down else "") if x]
    return res or ["depth: 두 쪽에 같은 시 파일이 없다"]


def compare_raster(py_dir: Path, rs_dir: Path, lo: str = "", hi: str = "~", starts=(), margin: int = 5) -> list[str]:
    """호가 래스터 시 파일(<폴더>/<SYM>/<UTC 시>.f32) 대조: 같은 초에 두 쪽 다 유효한 행의 bin_lo·mid 같음 · 잔량 칸 같음 비율.
    🔴바이트 일치는 기대하지 않는다 -- 두 프로세스가 스냅숏을 다른 순간에 받고(병합 범위가 다르다) 틱이 각자 벽시계라 같은 초의 북이 몇 이벤트 갈린다.
    통과 기준(제안): 두 쪽 유효 초의 99% 이상에서 bin_lo·mid 같음 · 칸 같음 비율 중앙 0.95 이상. 무효 초 수는 따로 적는다."""
    import struct

    def rows(p: Path) -> dict:
        b = p.read_bytes()
        n, bs = struct.unpack_from("<Hf", b, 6)   # 머리 "FLWR" u16 판 · u16 칸 수 · f32 칸 크기
        BS[0] = bs
        row = 16 + 4 * n
        out = {}
        for k in range((len(b) - 32) // row):
            r = b[32 + k * row:32 + (k + 1) * row]
            ts, bl, mid = struct.unpack_from("<qif", r, 0)
            if ts and mid == mid:
                out[ts] = (bl, mid, r[16:])
        return out
    BS = [0.0]
    hours = [(pp.stem, pp, rs_dir / pp.name) for pp in sorted(py_dir.glob("*.f32")) if lo <= pp.stem <= hi and (rs_dir / pp.name).exists()]
    data = [(h, rows(pp), rows(rp)) for h, pp, rp in hours]
    down = holes(sorted({t // 1000 for _, _, b in data for t in b}), list(starts), margin) if starts else []
    res, n_down = [], 0
    for h, a, b in data:
        both = [t for t in sorted(a.keys() & b.keys()) if not any(x <= t // 1000 < y for x, y in down)]
        n_down += len(a.keys() & b.keys()) - len(both)
        same_hdr = sum(a[t][:2] == b[t][:2] for t in both)
        fr, near, l1, l1n, l1all = [], [], [], [], []
        L = lambda u, v: sum(abs(x - y) for x, y in zip(u, v)) / (sum(abs(x) + abs(y) for x, y in zip(u, v)) or 1)   # noqa: E731
        for t in both:
            if a[t][:2] != b[t][:2]:
                continue
            qa, qb = [x[0] for x in struct.iter_unpack("<f", a[t][2])], [x[0] for x in struct.iter_unpack("<f", b[t][2])]
            n, c = len(qa), len(qa) // 2
            fr.append(sum(x == y for x, y in zip(qa, qb)) / n)
            near.append(sum(qa[j] == qb[j] for j in range(c - 20, c + 20)) / 40)   # mid 칸 = n/2(raster.rs _row) · ±20칸 = diff 가 늘 갱신하는 곳
            # 기준 칸 = 가지치기(±PRUNE_PCT, 60틱마다 그때 mid 로 -- raster.rs prune) 안쪽만(11-28): 그 밖 칸은 «마지막 가지치기 뒤 diff 가 다시 넣은 레벨»이라
            #   각 프로세스의 틱 위상(기동 시각)에 달렸다 -- 둘 다 맞다. 얇은 주식 선물은 래스터 폭이 ±4% 를 넘는다(SOXL 0.02×240 ≈ ±9% · SKHYNIX 0.25×240).
            #   ETH·BTC 등은 래스터 전체가 ±4% 안이라 그대로. 가지치기 뒤 mid 가 움직인 몫으로 0.5%p 안쪽만(서버 SOXL 11-25T16: ±4% 안 L1 중앙 1.3e-4 · ±3.5% 0).
            bl, mid, bs = a[t][0], a[t][1], BS[0]
            inr = [j for j in range(n) if (bl + j) * bs >= mid * (1 - RASTER_KEEP) and (bl + j + 1) * bs <= mid * (1 + RASTER_KEEP)]
            l1.append(L([qa[j] for j in inr], [qb[j] for j in inr]))
            l1all.append(L(qa, qb))
            na, nb = qa[c - 20:c + 20], qb[c - 20:c + 20]   # 진단(11-28): 먼 칸은 스냅숏 받은 순간 값이 diff 가 닿을 때까지 남는다 -- 얇은 주식 선물(SOXL·SKHYNIX)은 L1 이 먼 칸 몫인지
            l1n.append(sum(abs(x - y) for x, y in zip(na, nb)) / (sum(abs(x) + abs(y) for x, y in zip(na, nb)) or 1))
        med = lambda v: sorted(v)[len(v) // 2] if v else None   # noqa: E731
        TALLY[0] += len(both)
        # 기준(11-22 확정 제안): 그 시 «잔량 상대차(L1) 중앙» ≤ RASTER_L1 -- 칸 같음 비율은 진단만.
        #   까닭: 두 프로세스가 각자 벽시계 초에 북을 찍어 붐비는 종목(ETH·BTC)은 mid 근처 칸도 몇 이벤트씩 갈린다(조용한 SOL 은 1.0) -- 같음 비율은 «시각 차»를 잰다.
        #   실측(11-21 12시간): L1 중앙 ETH 2e-5~6e-3 · BTC 1e-3~1.7e-2(재기동 시) · SOL 0~2.5e-10. 계산이 틀리면(부호·칸 밀림) L1 이 수십 % 로 뛴다.
        TALLY[1] += int(bool(both) and (med(l1) is None or med(l1) > RASTER_L1))
        res.append(f"raster {h}: 두 쪽 유효 초 {len(both):,} · bin_lo·mid 같음 {same_hdr:,} · 칸 같음 비율 중앙 전체 {med(fr)} · mid ±20칸 {med(near)} · "
                   f"잔량 상대차(L1 · 가지치기 안 ±{RASTER_KEEP:.1%}) 중앙 {med(l1)}(전체 칸 {med(l1all)} · mid ±20칸 {med(l1n)}) · 파이썬만 유효 {len(a.keys() - b.keys()):,} · Rust만 유효 {len(b.keys() - a.keys()):,}")
    return (res + [f"raster:{down_note(down, n_down)}"] if down else res) or ["raster: 두 쪽에 같은 시 파일이 없다"]


def compare_bt(py_dir: Path, rs_dir: Path, lo: str = "", hi: str = "~", starts=(), margin: int = 5, reconnects: int = 0) -> list[str]:
    """HL 최우선 호가 .bt 시 파일(<폴더>/<COIN>/<UTC 시>.bt[.gz] · 헤더 32B + 행 32B `<qdfdf` = 거래소 ts_ms·매수가·량·매도가·량) 대조:
    시마다 행 바이트 묶음(중복 포함). 행에 받은 시각이 없다(거래소 ts) → 두 쪽이 같은 메시지를 받았으면 바이트가 같다.
    빼는 곳: 그림자 재기동 구간(holes) · Rust 만 빈 연결 끊김(로그 «끊김» 줄 수만큼) · 파이썬만 빈 끊김."""
    import gzip
    import struct

    def read(d: Path) -> dict:
        out: dict = {}
        for f in sorted(d.glob("*.bt*")):
            h = f.name.split(".")[0]
            if lo <= h <= hi:
                raw = gzip.decompress(f.read_bytes()) if f.suffix == ".gz" else f.read_bytes()
                out.setdefault(h, []).extend(raw[32 + 32 * k:64 + 32 * k] for k in range((len(raw) - 32) // 32))
        return out
    a, b = read(py_dir), read(rs_dir)
    ts = lambda rows: sorted({struct.unpack_from("<q", r)[0] for h in rows for r in rows[h]})   # noqa: E731 -- 거래소 ms
    ta, tb = ts(a), ts(b)
    hs = holes(sorted({t // 1000 for t in tb}), list(starts), margin) if starts else []
    # 연결 끊김 = 다른 쪽은 받는데 이쪽 bbo 가 RECONNECT_MS 넘게 빔(원본 asyncio.sleep(3) · Rust 도 3초 + 접속 -- 11-22 서버: 초 단위 «5초 넘게»로는
    # 이 3~4초 빈틈을 못 잡아 Rust 로그 «끊김» 33줄이 0번 설명). 같은 연결 하나가 받은 bbo 는 둘 다 거래소 값 그대로 -- 빈 쪽은 그동안 연결이 없었을 뿐이다.
    hs_ms = [(x * 1000, y * 1000) for x, y in hs]
    rsil, psil = silences(tb, ta, RECONNECT_MS, hs_ms), silences(ta, tb, RECONNECT_MS, hs_ms)
    down = merge(hs_ms + rsil[:reconnects] + psil)
    d0 = [x for x, _ in down]
    sa_, sb_ = set(ta), set(tb)
    res, n_down = [], 0
    gone: tuple = ({}, {})   # 뺀 구간에서 빠진 행 {값 24B: [ms]} -- 쪽마다(③)

    def keep(rows, side):
        nonlocal n_down
        out = Counter()
        for r in rows:
            t = struct.unpack_from("<q", r)[0]
            i = bisect.bisect_right(d0, t) - 1   # down = 겹치지 않게 합친 구간
            if i >= 0 and t < down[i][1]:
                n_down += 1
                gone[side].setdefault(r[8:], []).append(t)
            else:
                out[r] += 1
        return out
    kept = {h: (keep(a[h], 0), keep(b[h], 1)) for h in sorted(a.keys() & b.keys())}
    # 연결마다 다른 표본(11-28): HL bbo 는 연결마다 갱신 일부를 묶어 보내고 같은 상태에 다른 ms 를 찍는다 -- 앞뒤 공통 행은 같아 다음 상태는 둘 다 같다.
    #   공통 행(sh) 사이 칸마다 «다른 쪽에 그 ms 가 없는» 행을 값(ms 뺀 24B)으로 모아:
    #   ① 같은 칸에 다른 쪽도 같은 값이 있으면 설명(같은 호가 상태 · 찍힌 ms 만 다름 -- 11-28 서버 XRP 남은 2,918행 중 2,676)
    #   ② 나머지는 칸 폭 RECONNECT_MS 안이고 (SKIP_RUN 개 이하 또는 다른 쪽도 그 칸에 제 행이 있음 = 둘 다 다른 부분집합) 이면 설명
    #      -- 그 시에 두 쪽 «한쪽에만» 수가 비슷할 때만(작은 쪽 ≥ 큰 쪽 절반 -- 한쪽만 빠지면 구현 결함이라 설명 안 함). 칸 폭 상한 = 끊김 문턱(그보다 긴 한쪽 빈틈은 끊김 몫).
    #   ③ 다른 쪽의 같은 값 행이 뺀 구간(끊김·재기동)에 들어 빠졌고 RECONNECT_MS 안이면 설명(조용한 코인은 같은 상태를 3초+ 띄워 찍어 서로의 빈틈이
    #      끊김으로 잡힌다 -- 한 짝만 빠지고 남은 짝이 고아가 됐다 · 11-28 XRP 남은 81 중 50).
    #   공통 행은 시를 넘어 이어 본다(시 첫·끝 칸이 빠지지 않게). 같은 ms 다른 값은 ①②③ 어디에도 안 든다(설명 못 함).
    sh = sorted(struct.unpack_from("<q", r)[0] for fa, fb in kept.values() for r in (fa & fb).elements())

    def segs(d, other, g):
        out: dict = {}
        n_g = 0
        for r in d.elements():
            t = struct.unpack_from("<q", r)[0]
            if t in other:
                continue
            if any(abs(t - u) <= RECONNECT_MS for u in g.get(r[8:], ())):
                n_g += 1
                continue
            i = bisect.bisect(sh, t)
            if 0 < i < len(sh):
                out.setdefault(i, Counter())[r[8:]] += 1
        return out, n_g

    def explained(mine, theirs):
        twin = samp = 0
        for i, v in mine.items():
            o = theirs.get(i)
            tw = sum((v & o).values()) if o else 0
            rest = sum(v.values()) - tw
            twin += tw
            if rest and sh[i] - sh[i - 1] <= RECONNECT_MS and (rest <= SKIP_RUN or o):
                samp += rest
        return twin, samp
    for h, (fa, fb) in kept.items():
        same, op, orr = sum((fa & fb).values()), sum((fa - fb).values()), sum((fb - fa).values())
        (gp, gop), (gr, gor) = segs(fa - fb, sb_, gone[1]), segs(fb - fa, sa_, gone[0])
        (twp, sp_), (twr, sr_) = explained(gp, gr), explained(gr, gp)
        twp, twr = twp + gop, twr + gor
        if min(op, orr) < 0.5 * max(op, orr):   # 대칭은 그 시 한쪽에만 행 전체로(①을 뺀 ② 는 수십 행이라 그것끼리 견주면 우연히 깨진다 -- 11-28 XRP 24T15 ② 23 대 51)
            sp_ = sr_ = 0
        TALLY[0] += same
        TALLY[1] += op + orr - twp - twr - sp_ - sr_
        res.append(f"bt {h}: 같은 행 {same:,} · 파이썬에만 {op:,} · Rust에만 {orr:,}" + (f" · 같은 값 다른 ms(설명) {twp:,}" if twp else "")
                   + (f" · 연결마다 다른 표본(설명) 파이썬 {sp_:,}·Rust {sr_:,}" if sp_ else ""))
        for who, d, other in (("파이썬에만", fa - fb, sb_), ("Rust에만", fb - fa, sa_)):   # 진단: 남은 행 앞 2개 · 다른 쪽에 같은 ms 행이 있나(있으면 같은 시각 다른 값)
            for r in sorted(d)[:2]:
                v = struct.unpack("<qdfdf", r)
                res.append(f"    {who} {v} · 다른 쪽 같은 ms 행 {'있음' if v[0] in other else '없음'}")
    res.append(f"bt: 연결 끊김(다른 쪽만 {RECONNECT_MS / 1000:.0f}초 넘게 받음) Rust {len(rsil)}번(로그 «끊김» {reconnects}줄 → {min(len(rsil), reconnects)} 설명) · 파이썬 {len(psil)}번"
               f"{down_note([(x / 1000, y / 1000) for x, y in merge(hs_ms + rsil[:reconnects] + psil)], n_down)}")
    return res if len(res) > 1 else ["bt: 두 쪽에 같은 시 파일이 없다"]


VOLATILE = ("recorded", "generated", "written", "updated", "_at", "at_ms", "elapsed")


def strip_vol(v):
    if isinstance(v, dict):
        return {k: strip_vol(x) for k, x in v.items() if not any(w in k for w in VOLATILE)}
    return [strip_vol(x) for x in v] if isinstance(v, list) else v


def compare_jsonl(py: Path, rs: Path, key: str) -> list[str]:
    """전진 장부 JSONL(청산지도 점수 origin · SOXL 주말 weekend): 같은 열쇠 줄끼리 받은·쓴 시각 칸을 빼고 글자 비교.
    Rust 에만 있는 열쇠가 파이썬 마지막 열쇠보다 뒤면 «파이썬이 아직 안 쓴 회차»(세지 않음) · 앞이면 설명 못 함."""
    import json
    rd = lambda f: {o[key]: o for o in (json.loads(x) for x in f.read_text().splitlines() if x.strip()) if key in o}   # noqa: E731
    a, b = rd(py), rd(rs)
    both = sorted(a.keys() & b.keys())
    bad = [k for k in both if strip_vol(a[k]) != strip_vol(b[k])]
    later = [k for k in b.keys() - a.keys() if a and k > max(a)]
    lost = sorted(b.keys() - a.keys() - set(later))
    TALLY[0] += len(both)
    TALLY[1] += len(bad) + len(lost)
    out = [f"jsonl {key}: 같은 열쇠 {len(both):,} · 다름 {len(bad):,} · Rust 에만(파이썬 마지막 뒤 = 아직 안 쓴 회차) {len(later):,} · Rust 에만(앞) {len(lost):,} · 파이썬에만 {len(a.keys() - b.keys()):,}"]
    out += [f"    다름 {k}" for k in bad[:3]] + [f"    Rust 에만(앞) {k}" for k in lost[:3]]
    return out


def compare_cboe(py: Path, rs: Path) -> list[str]:
    """CBOE 요약 상태 JSON: 값은 체인을 받은 순간의 호가라 두 번 받으면 원래 다르다 -- 열쇠 모양(2단까지)이 같은지만 센다(값 차는 적기만).
    계산 일치 근거는 정답지(cboe_summary_parity)뿐."""
    import json
    a, b = json.loads(py.read_text()), json.loads(rs.read_text())

    def keys(v, pre=""):
        if isinstance(v, dict):
            for k, x in v.items():
                yield pre + str(k)
                if pre.count("/") < 1:
                    yield from keys(x, pre + str(k) + "/")
    ka, kb = set(keys(strip_vol(a))), set(keys(strip_vol(b)))
    TALLY[0] += len(ka & kb)
    TALLY[1] += len(ka - kb)
    return [f"cboe 상태: 같은 열쇠 {len(ka & kb):,} · 파이썬에만 {len(ka - kb):,} {sorted(ka - kb)[:5]} · Rust 에만 {len(kb - ka):,} · 값은 받은 순간 호가라 견주지 않음"]


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
            con.executemany("INSERT INTO bars VALUES (?,?,?,?,?)", [(1000 * (T + 300 * k), 0.5 if (f == r and k in (0, 2)) else (0.25 + 1e-14 if (f == r and k == 3) else 0.25),
                                                                   300.0 * k + (99 if f == r else 7), 1.0 if (f == r and k == 0) else 0.0, w) for k in range(5)])
            con.commit()
            con.close()
        bars_line = next(x for x in run(p, r) if x.startswith("bars"))
        # 봉 1·2·3 견줌(봉 4 = 끝 10초 안) · 봉 0 = warm
        assert "비교 3행 · 불일치 1 · 파이썬에만 0 · Rust에만 0" in bars_line and "준비 구간(warm) 봉 1 뺌" in bars_line and "끝자리 차" in bars_line, bars_line   # 봉 3 = 끝자리만 다름
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
        dl = next(x for x in run(p, r, window=250, show=5, starts=[T - 100, T + 500]) if x.startswith("trade_tape_1s"))   # 첫 기동(자료 전) · 재기동
        assert "그림자 꺼진 구간 2개" in dl and "유한 합 172초" in dl and "Rust에만 2" in dl and "파이썬에만 0" in dl, dl   # T+700 은 Rust 에만(파이썬에서 지움) + 분 2 의 그것
        dl0 = next(x for x in run(p, r, window=250, show=5) if x.startswith("trade_tape_1s"))
        assert "파이썬에만 " in dl0 and "파이썬에만 0" not in dl0, dl0   # 기동 시각을 안 주면 빈 구간이 «파이썬에만»
        # depth: Rust 가 T+100~T+130 초에 프레임이 없고 기동 T+120 → 그 사이 파이썬 프레임은 뺀다 · 바깥의 파이썬에만 프레임은 실패
        import gzip
        import struct
        for side in ("dp", "dr"):
            (Path(d) / side).mkdir()
            fr = [f'{{"e":"depthUpdate","E":{1000 * (T + i)},"u":{i}}}' for i in range(200) if not (side == "dr" and (100 <= i <= 130 or i == 170))]
            (Path(d) / side / "2026-11-21T07.jsonl.gz").write_bytes(gzip.compress(("\n".join(['{"_snapshot":null}'] + fr) + "\n").encode()))
        dd = compare_depth(Path(d) / "dp", Path(d) / "dr", starts=[T - 50, T + 120])
        assert dd[0].startswith("depth 2026-11-21T07: 같은 프레임 152 · 파이썬에만 1 · Rust에만 0") and "뺀 행 63" in dd[1], dd   # 뺀 구간 = 첫 기동 전(~T+5) + [99-5, 131+5) 초
        # 다시 붙은 자리: Rust 가 E 20~29 를 못 받고(재접속 -- _gap 없이 pu 끊김) 파이썬만 받음 → 설명 · Rust 가 E 50 을 못 받았는데 pu 가 이어지면(연결 안) → 설명 못 함
        for side in ("cp", "cr"):
            (Path(d) / side).mkdir()
            fr = [f'{{"e":"depthUpdate","E":{1000 * (T + i)},"U":{10 * i + 1},"u":{10 * i + 10},"pu":{10 * i}}}' for i in range(1, 80)
                  if not (side == "cr" and (20 <= i < 30 or i == 50))]
            if side == "cr":   # 연결 안에서 끊김은 원본이 _gap 줄을 쓴다
                fr.insert([k for k, x in enumerate(fr) if f'"E":{1000 * (T + 51)}' in x][0], '{"_gap":true,"expected_pu":500,"got_pu":510}')
            (Path(d) / side / "2026-11-21T08.jsonl.gz").write_bytes(gzip.compress(("\n".join(['{"_snapshot":null}'] + fr) + "\n").encode()))
        TALLY[:] = [0, 0]
        cd = compare_depth(Path(d) / "cp", Path(d) / "cr")
        assert TALLY[1] == 1 and "다시 붙은 자리 Rust 1·파이썬 0" in cd[-1] and "파이썬에만 10" in cd[-1], (TALLY, cd)
        # .bt: 같은 행 · Rust 끊김(로그 1줄) 설명 · 값 하나 다른 행은 실패
        for side in ("bp", "br"):
            (Path(d) / side).mkdir()
            rows = [struct.pack("<qdfdf", 1000 * (T + i), 100.0 + i, 1.0, 101.0 + i, 2.0) for i in range(100)
                    if not (side == "br" and 40 <= i < 60)]
            if side == "br":
                rows[5] = struct.pack("<qdfdf", 1000 * (T + 5), 999.0, 1.0, 101.0, 2.0)
            (Path(d) / side / "2026-11-21T08.bt").write_bytes(b"BTKR" + b"\0" * 28 + b"".join(rows))
        TALLY[:] = [0, 0]
        bt0 = compare_bt(Path(d) / "bp", Path(d) / "br", reconnects=1)
        assert TALLY == [78, 2] and "Rust 1번(로그 «끊김» 1줄 → 1 설명)" in bt0[-1], (TALLY, bt0)   # 끊김 [39,60) 안 행(파이썬 21·Rust 1) 뺌 · 다른 값 1 행 = 양쪽 1씩
        TALLY[:] = [0, 0]
        compare_bt(Path(d) / "bp", Path(d) / "br", reconnects=0)
        assert TALLY[1] == 22, TALLY   # 로그에 끊김이 없으면 그 20행도 설명 못 함
        # 3~4초 재접속 빈틈(11-22 서버 hl_bt): 200ms 간격 · Rust 가 [5.0, 8.6)초 빔 · 파이썬이 [12.0, 15.6)초 빔 · 2.4초 빈틈(Rust)은 끊김 아님
        for side in ("sp", "sr"):
            (Path(d) / side).mkdir()
            ms = [200 * i for i in range(100)]
            ms = [x for x in ms if not ((side == "sr" and 5000 <= x < 8600) or (side == "sp" and 12000 <= x < 15600) or (side == "sr" and 17000 <= x < 19400))]
            (Path(d) / side / "2026-11-21T08.bt").write_bytes(b"BTKR" + b"\0" * 28 + b"".join(struct.pack("<qdfdf", 1000 * T + x, 100.0, 1.0, 101.0, 2.0 + x) for x in ms))
        TALLY[:] = [0, 0]
        b1 = compare_bt(Path(d) / "sp", Path(d) / "sr", reconnects=1)
        assert TALLY[1] == 12 and "Rust 1번" in b1[-1] and "파이썬 1번" in b1[-1], (TALLY, b1)   # 남는 것 = 2.4초 빈틈의 파이썬 12행
        TALLY[:] = [0, 0]
        compare_bt(Path(d) / "sp", Path(d) / "sr", reconnects=0)
        assert TALLY[1] == 12 + 18, TALLY   # 로그에 끊김이 없으면 Rust 3.6초 빈틈의 파이썬 18행도 설명 못 함 · 파이썬 쪽 빈틈은 그대로 설명
        # 연결마다 다른 표본(11-28 · 양성 대조): 110ms 간격 · 두 쪽이 서로 다른 외톨이 행 5개씩 못 받음 = 설명 · 한쪽만 5개 = 설명 못 함 · 한 자리에서 4개 연달아 = 설명 못 함
        def kbt(name, drop, bad=-1):
            (Path(d) / name).mkdir()
            (Path(d) / name / "2026-11-21T09.bt").write_bytes(b"BTKR" + b"\0" * 28 + b"".join(
                struct.pack("<qdfdf", 1000 * T + 110 * i, 100.0, 1.0, 101.0, 2.0 + i + (i == bad)) for i in range(500) if i not in drop))
        for name, drop in (("k1", {10, 50, 90, 130, 170}), ("k2", {30, 70, 110, 150, 190}), ("k3", set()), ("k5", {300, 301, 302, 303, 350, 400, 450})):
            kbt(name, drop)
        kbt("k6", set(), bad=200)   # 같은 ms 값만 다름 = 설명 못 함(양쪽 1씩)
        for p_, r_, want in (("k1", "k2", 0), ("k3", "k1", 5), ("k1", "k3", 5), ("k5", "k2", 4), ("k6", "k3", 2)):
            TALLY[:] = [0, 0]
            kb = compare_bt(Path(d) / p_, Path(d) / r_)
            assert TALLY[1] == want, (p_, r_, TALLY, kb)   # k3·k1 = 한쪽만 빠짐(구현 결함 모양) · k5 = 외톨이 3은 설명, 4개 연달아는 설명 못 함
        # 조용한 코인(11-28 서버 XRP 2,875): 공통 행 간격이 1초를 넘어도(1.5초 · 외톨이 칸 3초) 두 쪽이 서로 다른 외톨이 = 설명 ·
        #   같은 값이 연결마다 다른 ms(2초 간격 · 칸 4초 > 끊김 문턱이라 ②는 안 됨) = 설명 · 양성 대조: ms 만이 아니라 값도 다르면 설명 못 함 2
        def qbt(name, step, drop=(), shift=None, bad=False):
            (Path(d) / name).mkdir()
            (Path(d) / name / "2026-11-21T10.bt").write_bytes(b"BTKR" + b"\0" * 28 + b"".join(
                struct.pack("<qdfdf", 1000 * T + step * i + (700 if i == shift else 0), 100.0, 1.0, 101.0, 2.0 + i + (bad and i == shift)) for i in range(100) if i not in drop))
        qbt("q1", 1500, {10, 30, 50}); qbt("q2", 1500, {20, 40, 60}); qbt("t0", 2000); qbt("t1", 2000, shift=50); qbt("t2", 2000, shift=50, bad=True)
        for p_, r_, want in (("q1", "q2", 0), ("t1", "t0", 0), ("t2", "t0", 2)):
            TALLY[:] = [0, 0]
            kb = compare_bt(Path(d) / p_, Path(d) / r_)
            assert TALLY[1] == want, (p_, r_, TALLY, kb)
        # ③ Rust 끊김(4.8~9.0초 빔 · 로그 1줄)으로 빠진 파이썬 8.8초 행과 같은 값을 Rust 가 9.1초에 찍음 = 설명 · 값이 다르면 설명 못 함 1
        for name, rows in (("g0", [(200 * i, i) for i in range(100)]), ("g1", [(200 * i, i) for i in range(100) if not 25 <= i <= 44] + [(9100, 44)]),
                           ("g2", [(200 * i, i) for i in range(100) if not 25 <= i <= 44] + [(9100, 45.5)])):
            (Path(d) / name).mkdir()
            (Path(d) / name / "2026-11-21T10.bt").write_bytes(b"BTKR" + b"\0" * 28 + b"".join(
                struct.pack("<qdfdf", 1000 * T + x, 100.0, 1.0, 101.0, 2.0 + i) for x, i in sorted(rows)))
        for p_, r_, want in (("g0", "g1", 0), ("g0", "g2", 1)):
            TALLY[:] = [0, 0]
            kb = compare_bt(Path(d) / p_, Path(d) / r_, reconnects=1)
            assert TALLY[1] == want, (p_, r_, TALLY, kb)
        # raster: 머리 n 을 읽는다 · mid ±20칸만 같으면 통과 쪽
        def f32(path, vals):
            n = 60
            h = b"FLWR" + struct.pack("<HHfIqII", 1, n, 0.5, 0, 0, 1000, 0)
            rows_ = b"".join(struct.pack("<qif", 1000 * (T + k), 70, 50.0) + struct.pack(f"<{n}f", *v) for k, v in enumerate(vals))   # bin_lo = mid//bin − n/2
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
        # 가지치기 밖(±4% 넘는 칸 = 칸 0~25·34~59 · mid 50 · 칸 0.5)만 크게 달라도 통과(SOXL·SKHYNIX 꼴) · 양성 대조: 안쪽 3칸 밀림(오르막 잔량이라 1칸은 L1 1.7%) · 안쪽 ×1.1 = 불통과
        inner = range(27, 34)
        for name, v, bad in (("far", [float(j) * (1 if j in inner else 3) for j in range(60)], 0),
                             ("shift", [float(j + (3 if j in inner else 0)) for j in range(60)], 1),
                             ("scale", [float(j) * (1.1 if j in inner else 1) for j in range(60)], 1)):
            (Path(d) / f"rr_{name}").mkdir()
            f32(Path(d) / f"rr_{name}/2026-11-21T07.f32", [v] * 10)
            TALLY[:] = [0, 0]
            rz = compare_raster(Path(d) / "rp", Path(d) / f"rr_{name}")
            assert TALLY == [10, bad], (name, rz, TALLY)
        # 설명 규칙 양성 대조(11-22): 같은 줄 두 번 · 재구독 현재값 · 연결 끊김(로그 줄 수만큼)
        q, w = Path(d) / "q.sqlite", Path(d) / "w.sqlite"
        for f in (q, w):
            con = sqlite3.connect(f)
            con.execute("CREATE TABLE okx_mark(inst TEXT, ts_ms INTEGER, mark REAL)")
            con.execute("CREATE TABLE okx_funding(inst TEXT, ts_ms INTEGER, rate REAL)")
            rows = [("ETH", 1000 * (T + s), 1.0 + s) for s in range(600)]
            if f == w:   # Rust: T+100~T+130 끊김(연결) · T+300~T+330 끊김(연결) -- 둘
                rows = [r for r in rows if not (100 < r[1] // 1000 - T < 131 or 300 < r[1] // 1000 - T < 331)]
            con.executemany("INSERT INTO okx_mark VALUES (?,?,?)", rows)
            fr = [("ETH", 1000 * (T + 60 * k), 0.1 * k) for k in range(10)]
            con.executemany("INSERT INTO okx_funding VALUES (?,?,?)", fr)
            if f == q:
                con.execute("INSERT INTO okx_funding VALUES ('ETH', ?, 0.4)", [1000 * (T + 240)])   # 파이썬: 같은 줄 두 번 → 설명
                con.execute("INSERT INTO okx_funding VALUES ('ETH', ?, 0.77)", [1000 * (T + 450)])  # 파이썬에만 (진짜) → 설명 못 함
            else:
                con.execute("INSERT INTO okx_funding VALUES ('ETH', ?, ?)", [1000 * (T + 200), 0.1 * 3])   # Rust: 직전 파이썬 줄(T+180, 0.3)과 같은 값 → 재구독 현재값
                con.execute("INSERT INTO okx_funding VALUES ('ETH', ?, 0.99)", [1000 * (T + 500)])  # Rust 에만 다른 값 → 설명 못 함
            con.commit()
            con.close()
        TALLY[:] = [0, 0]
        o0 = run(q, w, reconnects=0)
        TALLY0 = TALLY[1]
        TALLY[:] = [0, 0]
        o1 = run(q, w, reconnects=1)
        TALLY1 = TALLY[1]
        TALLY[:] = [0, 0]
        o2 = run(q, w, reconnects=2)
        fl = next(x for x in o2 if x.startswith("okx_funding"))
        assert "같은 줄 두 번(설명) 파이썬 1·Rust 0" in fl and "재구독 현재값(직전 파이썬 줄과 같은 값 · 설명) 1" in fl, fl
        assert o2[0].startswith("연결 끊김") and "Rust 2번(그림자 로그 «끊김» 2줄 → 2번 설명" in o2[0], o2[0]
        # 끊김 줄 0 → 두 끊김(30초씩 = 60행) 다 설명 못 함 · 1 → 하나만 · 2 → 0 · 진짜 파이썬에만 1·Rust 에만 1 은 늘 남는다
        assert TALLY0 - TALLY1 == 30 and TALLY1 - TALLY[1] == 30 and TALLY[1] == 2, (TALLY0, TALLY1, TALLY[1], o0)
        # Rust 표에 한 번도 없는 종목: 첫 기동 전 행은 빼고, 뒤 행은 설명 못 함(11-22 OKX 청산 ADBE 165)
        l1, l2 = Path(d) / "l1.sqlite", Path(d) / "l2.sqlite"
        for f in (l1, l2):
            con = sqlite3.connect(f)
            con.execute("CREATE TABLE okx_mark(inst TEXT, ts_ms INTEGER, mark REAL)")
            con.execute("CREATE TABLE okx_liquidations(inst_id TEXT, inst_family TEXT, ts_ms INTEGER, side TEXT, pos_side TEXT, bk_px REAL, sz_contracts REAL, sz_base REAL, bk_loss REAL)")
            con.executemany("INSERT INTO okx_mark VALUES ('ETH', ?, 1.0)", [(1000 * (T + k),) for k in range(0, 600) if f == l1 or k >= 200])   # Rust 는 T+200 부터
            con.executemany("INSERT INTO okx_liquidations VALUES ('ETH', 'ETH', ?, 'b', 'l', 1, 1, 1, 0)", [(1000 * (T + 10),), (1000 * (T + 500),)])   # T+10 = Rust 가 구독 때 받은 옛 사건(ts 가 앞)
            if f == l1:
                con.executemany("INSERT INTO okx_liquidations VALUES ('ADBE', 'ADBE', ?, 'b', 'l', 1, 1, NULL, 0)", [(1000 * (T + 50),), (1000 * (T + 400),)])
            con.commit()
            con.close()
        TALLY[:] = [0, 0]
        ll = next(x for x in run(l1, l2, starts=[T + 190]) if x.startswith("okx_liquidations"))
        assert "파이썬에만 1 " in ll and TALLY[1] == 1, (TALLY, ll)   # ADBE T+50(그림자 뜨기 전)은 빼고 T+400 만 남는다
        # 기동 뒤 «그 종목 Rust 첫 행 전» 파이썬 행은 빼지 않는다(11-28): BTC 파이썬 T+300·T+550 · Rust T+550 만 → T+300 은 설명 못 함
        for f in (l1, l2):
            con = sqlite3.connect(f)
            con.executemany("INSERT INTO okx_liquidations VALUES ('BTC', 'BTC', ?, 'b', 'l', 1, 1, 1, 0)", [(1000 * (T + k),) for k in ((300, 550) if f == l1 else (550,))])
            con.commit()
            con.close()
        TALLY[:] = [0, 0]
        ll = next(x for x in run(l1, l2, starts=[T + 190]) if x.startswith("okx_liquidations"))
        assert "파이썬에만 2 " in ll and TALLY[1] == 2, (TALLY, ll)
        # 전진 장부 JSONL: 같은 열쇠 줄(기록 시각만 다름) · 다른 값 · 파이썬 마지막 뒤 Rust 에만(안 셈) · 앞 Rust 에만(셈)
        (Path(d) / "j1.jsonl").write_text('{"origin":"a","cp":1,"recorded_at":1}\n{"origin":"b","cp":2}\n{"origin":"d","cp":4}\n')
        (Path(d) / "j2.jsonl").write_text('{"origin":"a","cp":1,"recorded_at":2}\n{"origin":"b","cp":3}\n{"origin":"c","cp":9}\n{"origin":"d","cp":4}\n{"origin":"e","cp":5}\n')
        TALLY[:] = [0, 0]
        jl = compare_jsonl(Path(d) / "j1.jsonl", Path(d) / "j2.jsonl", "origin")
        assert TALLY == [3, 2] and "아직 안 쓴 회차) 1" in jl[0], (TALLY, jl)
        # 같은 초 다른 값(xyz ctx): 파이썬 T+50 에 1.5(Rust 는 같은 초에 1.6) → 설명 · 파이썬 T+300 에 9.9(Rust 는 T+298~302 에 줄 없음) → 설명 못 함
        e1, e2 = Path(d) / "e1.sqlite", Path(d) / "e2.sqlite"
        for f in (e1, e2):
            con = sqlite3.connect(f)
            con.execute("CREATE TABLE hl_xyz_ctx_1s(coin TEXT NOT NULL, recv_sec INTEGER NOT NULL, mid REAL, mark REAL, oracle REAL, PRIMARY KEY(coin, recv_sec))")
            con.executemany("INSERT INTO hl_xyz_ctx_1s VALUES ('xyz:JPY', ?, ?, 1.0, 1.0)",
                            [(T + k, 1.0 + k) for k in range(0, 600) if not (f == e2 and 295 <= k <= 305)])
            if f == e1:
                con.execute("UPDATE hl_xyz_ctx_1s SET mid = 1.5 WHERE recv_sec = ?", [T + 50])
                con.execute("UPDATE hl_xyz_ctx_1s SET mid = 9.9 WHERE recv_sec = ?", [T + 300])
            else:
                con.execute("UPDATE hl_xyz_ctx_1s SET mid = 1.6 WHERE recv_sec = ?", [T + 50])
            con.commit()
            con.close()
        TALLY[:] = [0, 0]
        xl = next(x for x in run(e1, e2) if x.startswith("hl_xyz_ctx_1s"))
        assert "같은 초(±1)에 다른 쪽도 받음(값만 다름 · 설명) 파이썬에만 3·Rust에만 1" in xl, xl
        # 파이썬에만 12 = 1.5(설명) + T+295~305 의 11(가장자리 295·305 는 Rust 가 ±1초 안에 받음 → 설명 · 296~304 의 9 는 못 받음 → 설명 못 함)
        assert "파이썬에만 12" in xl and TALLY[1] == 9, (TALLY, xl)
        # --syms(11-28 hl_bt): 다른 그림자가 쓴 코인(파이썬에만 xyz:EUR 5줄)은 빼고 견준다 -- 양성 대조: 안 거르면 5줄이 설명 못 함
        con = sqlite3.connect(e1)
        con.executemany("INSERT INTO hl_xyz_ctx_1s VALUES ('xyz:EUR', ?, ?, 1.0, 1.0)", [(T + k, 2.0 + k) for k in range(5)])
        con.commit()
        con.close()
        TALLY[:] = [0, 0]
        run(e1, e2)
        assert TALLY[1] == 9 + 5, TALLY
        SYMS.add("xyz:JPY")
        TALLY[:] = [0, 0]
        run(e1, e2)
        SYMS.clear()
        assert TALLY[1] == 9, TALLY
        # xyz 1분봉(11-28): 같은 분의 앞선 스냅숏 = 설명(0.5% 안) · REST 로 쓴 분 = 뺀다 -- 양성 대조: 값 0.01% · 봉 0.1% 빼기 · 앞선 스냅숏 1%
        def bars(f, edit=()):
            con = sqlite3.connect(f)
            con.execute("CREATE TABLE IF NOT EXISTS hl_xyz_1m(coin TEXT, ts_sec INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL, n INTEGER, PRIMARY KEY(coin, ts_sec))")
            con.execute("DELETE FROM hl_xyz_1m")
            rows = {k: ["xyz:SP500", T + 60 * k, 100.0, 101.0, 99.0, 100.5, 10.0, 50] for k in range(1000)}
            for k, v in edit:
                if v is None:
                    rows.pop(k)
                else:
                    rows[k][2:] = v
            con.executemany("INSERT INTO hl_xyz_1m VALUES (?,?,?,?,?,?,?,?)", list(rows.values()))
            con.commit()
            con.close()
        h1, h2 = Path(d) / "h1.sqlite", Path(d) / "h2.sqlite"
        snap = [100.0, 100.8, 99.0, 100.2, 9.0, 48]   # 앞선 스냅숏: 체결 2건 덜 받음
        bars(h1)

        def bad(rs_edit, py_edit=()):
            bars(h1, py_edit)
            bars(h2, rs_edit)
            TALLY[:] = [0, 0]
            out = next(x for x in run(h1, h2) if x.startswith("hl_xyz_1m"))
            return TALLY[1], out
        b0, o = bad([(10, snap)], [(30, snap)])   # Rust 가 앞선 1 · 파이썬이 앞선 1
        assert b0 == 0 and "앞선 스냅숏(한 연결이 마감 갱신을 못 받음 · 0.5% 안이면 설명) 2" in o, o
        assert bad([(20, [100.0, 101.0, 99.0, 100.5 * 1.0001, 10.0, 50])])[0] == 1   # 값 0.01%(체결 수 같음) → 설명 못 함
        assert bad([(20, [100.0, 101.0, 99.0, 100.5, 10.0 * 1.0001, 50])])[0] == 1
        assert bad([(k, snap) for k in range(0, 1000, 100)])[0] == 10   # 1% 가 앞선 스냅숏 → 연결 잡음 아님 → 전부
        assert bad([(500, None)])[0] == 1   # Rust 봉 0.1% 빠짐
        assert bad([(k, None) for k in range(50, 1000, 100)])[0] == 10   # 1%
        (Path(d) / "xyz100_x.log").write_text(f"[xyz100] {time.strftime('%F %T', time.localtime(T + 60 * 501 + 5))} 접속 · xyz:SP500 REST 메움 2봉\n")
        PY_REST[:] = read_rest_log(str(Path(d) / "xyz100_*.log"))
        b1, o = bad([(500, None), (501, snap[:3] + [100.4, 9.5, 49])])   # 원본이 REST 로 쓴 두 분(500 = Rust 에 없음 · 501 = 다름)은 뺀다
        assert b1 == 0 and "뺀 (종목,분) 2" in o, o
        assert bad([(499, None)])[0] == 1   # REST 분 밖은 그대로
        PY_REST.clear()
        assert episodes([(0, 3), (5, 8), (30, 33), (100, 101)]) == [(0, 8), (30, 33), (100, 101)]
        # 첫 기동 전(-inf)은 뺀다: 기동 T+600 하나면 그 앞 Rust 자료가 있어도 첫 기동 = 로그의 가장 이른 것
        assert holes([T + 603], [t0 := T + 600], 5) == [(float("-inf"), t0 + 8)]
        # 재기동: 죽인 때(T+100) 뒤 빈틈을 다시 붙을 때 받은 «최근 체결»(T+148~149)이 T 바로 앞까지 메워도, 가장 큰 빈틈 [T+100, T+148] 을 찾는다
        tl = list(range(T, T + 101)) + [T + 148, T + 149] + list(range(T + 152, T + 400))
        assert holes(tl, [T - 10, T + 150], 5) == [(float("-inf"), T + 5), (T + 95, T + 153)], holes(tl, [T - 10, T + 150], 5)
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
    ap.add_argument("--reconnects", type=int, default=0, help="그림자 Rust 감독 로그의 «끊김» 줄 수(takeover.sh 가 센다) -- 그만큼의 Rust 만 빈 곳을 설명으로")
    ap.add_argument("--summary", action="store_true", help="끝에 «요약: 견준 N · 설명 못 함 M» · M>0 이거나 N=0 이면 종료코드 1(takeover.sh check)")
    ap.add_argument("--subset", action="store_true", help="Rust 에만·다름만 센다(파이썬이 오래 모은 것은 안 셈)")
    ap.add_argument("--syms", default="", help="쉼표 목록 -- 이 종목만 견준다(hl_bt 코인)")
    ap.add_argument("--py-rest-log", default="", help="원본 xyz100 로그 glob -- «REST 메움 N봉» 줄의 봉(원본이 REST 로 덮은 분)은 hl_xyz_1m 에서 뺀다(takeover.sh xyz)")
    ap.add_argument("--jsonl", default="", help="두 경로 = JSONL · 값 = 열쇠 칸(origin · weekend)")
    ap.add_argument("--cboe", action="store_true", help="두 경로 = cboe_option_state.json")
    ap.add_argument("--bt", action="store_true", help="두 경로 = HL .bt 코인 폴더 · --hours 시작,끝")
    ap.add_argument("--events", action="store_true", help="두 경로 = 아이스버그 <sym>_events.jsonl · --from/--to = t_end 초")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.py and a.rs):
        ap.error("파이썬 DB 와 Rust DB 경로가 필요하다")
    lo, _, hi = a.hours.partition(",")
    st = [int(x) for x in a.starts.split(",") if x.strip()]
    global SUBSET
    SUBSET = a.subset
    SYMS.update(x for x in a.syms.split(",") if x)
    PY_REST[:] = read_rest_log(a.py_rest_log) if a.py_rest_log else []
    if a.jsonl:
        out = compare_jsonl(a.py, a.rs, a.jsonl)
    elif a.cboe:
        out = compare_cboe(a.py, a.rs)
    elif a.json:
        out = compare_json(a.py, a.rs, a.json, a.drop)
    elif a.bt:
        out = compare_bt(a.py, a.rs, lo, hi or "~", st, a.margin, a.reconnects)
    elif a.events:
        out = compare_events(a.py, a.rs, a.lo, a.hi, a.show)
    elif a.raster:
        out = compare_raster(a.py, a.rs, lo, hi or "~", st, a.margin)
    elif a.depth:
        out = compare_depth(a.py, a.rs, lo, hi or "~", st, a.margin)
    else:
        out = run(a.py, a.rs, a.lo, a.hi, a.margin, a.window, a.show, st, a.reconnects)
    print("\n".join(out))
    if a.summary:
        print(f"요약: 견준 {TALLY[0]:,} · 설명 못 함 {TALLY[1]:,}")
        return 1 if TALLY[1] or not TALLY[0] else 0


if __name__ == "__main__":
    sys.exit(main())
