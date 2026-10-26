#!/usr/bin/env python3
"""ETH 5분 호가 요약 실시간 워커 (2026-10-26, 사용자 «ETH 만 먼저 · 호가 세션 검증 받고 코드»).

정의 = 호가 5분 지표 연구(워크트리 orderbook-5min-indicator-67e28f, 미커밋)의
  research_orderbook_5m_indicator_20261026.py(day_job·bars_from·wall_age·classify·tape_bad)
  + research_eth_wall_pull_jump_20261004.py(replay·fill_share) 를 그대로 옮겼다 -- 연구 세션 권고(정의가 SOL·XRP 재보정으로 바뀔 수 있어 커밋 import 대신 이식).
  🔴정의 확정 전 · 정보 표시용(예측 아님). 문턱은 아래 상수 한 곳. 이식이 맞는지는 --parity(연구 bars v2 와 1e-9 대조)가 지킨다.
입력(읽기만): data/live/orderflow/depthdiff/ETHUSDT/<UTC시>.jsonl(.gz) (첫 줄 REST 스냅숏 + @depth@100ms) · data/hot/binance_tape.sqlite trade_tape_1s·gaps.
출력: data/live/ob5m/ethusdt.sqlite -- bars(봉 1행, 연구 bars_from 열) · marks(돌파 eat · 가짜벽 fake 표식). 대시보드 /api/ob5m 이 읽는다.
연구 세션 검증 회신(10-26)의 «꼭 반영» 다섯:
  ① 완성된 줄('\\n' 끝)만 넘긴다 · 새 시 파일이 생기면 옛 파일을 EOF 까지 마저 읽고 넘어간다(gzip 뒤 지워져도 열린 핸들은 읽힌다).
  ② 끊김(pu 불연속·스냅숏 없음)이면 다음 시 파일 스냅숏에서 다시 -- 재시작 뒤 30분 = warm(북 값 없음).
  ③ 봉 마감 = 거래소 T 기준 · 사건 판정(테이프 빈 구간 ±60초)까지 봉 끝 + 70초(SETTLE_MS) 뒤에 쓴다.
     북 값(기울기·깊이·벽)은 테이프와 무관해 봉 끝에서 바로 먼저 쓴다(10-26 사용자 «기울기 막대가 늦다» -- 봉 끝 뒤 ~3분 걸렸다).
  ④ 벽 나이·«가장 오래 선 벽»은 워커가 계속 든 상태(칸별 시작·마지막 본 시각)로 -- 봉마다 새로 세면 틀린다 · 같은 초 도장(t−1) 중복은 첫 것만.
  ⑤ replay 는 사건을 콜백(on_event)으로 넘긴다(실시간엔 끝나지 않는다) · min_share 1% 밑은 안 쌓는다.
ponytail: 체결 몫은 봉 끝 +70초의 hot 테이프로 판정한다 -- 연구의 lake 는 수집기가 몇 분 뒤 REST 로 고친 판이라 드물게 eat↔fake 가 갈릴 수 있다
  (고치려면 봉을 ~15분 뒤 한 번 더 닫는다). 수집기 gaps 표 ±60초도 판정 불가로 친다(연구 세션 권고 -- 연구보다 NULL 이 조금 많다).
주문 없음 · 거래소 호출 없음.
사용: python -u scripts/live_ob5m_20261026.py | --parity 2026-10-02 (로컬 백업 미러) | --seed <bars.parquet> <events.parquet> | --selftest
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SYM, TICK, BUCKET = "ETHUSDT", 100, 0.1      # 거래소 틱 1/$ · 테이프 칸(live_trade_tape_collector BUCKETS)
DD = ROOT / f"data/live/orderflow/depthdiff/{SYM}"
TAPE = ROOT / "data/hot/binance_tape.sqlite"
DB = ROOT / "data/live/ob5m/ethusdt.sqlite"
IDX_MAX = 1_000_000   # ponytail: 연구는 10M($100k) -- 16MB 로 줄임. ETH 가 $10k 를 넘으면 올린다(넘는 가격 변경은 버려진다)
BAR = 300_000
# ── 정의 문턱(연구 그대로 -- 바뀌면 여기만) ──
WALL_SHARE = 0.05      # 벽 = 같은 쪽 ±50bp 깊이의 5% 이상 한 칸
BIG_AGE_MS = 30_000    # 사건 벽은 30초 이상 서 있던 것만
NEAR_BP = 5.0          # 현재가 5bp 안에서 체결 없이 사라짐 = 가짜벽 의심
EAT_FILL, PULL_FILL = 0.8, 0.2
JUMP_WIN = 5_000
WARM_MS = 1_800_000    # 사슬 (재)시작 뒤 30분 = 깊은 칸이 덜 찼다 -> 북 값 없음
TAPE_GAP_S, TAPE_PAD_S = 15, 60
WALL_BACK_MS = 10_000  # 다른 칸이 10초 이하 잠깐 최대였다 돌아오면 같은 벽으로 이어 센다
KS = (20, 50, 100)     # 사건 추적 하한 = 같은 쪽 ±50bp 비영 레벨 중앙의 KS[0]배
DROP_FRAC, DROP_WIN = 0.8, 1000
MIN_SHARE = 0.01
SETTLE_MS = 70_000     # 봉 끝 + 60초(테이프 빈 구간 ±60초) + 테이프 지연 10초
BOOK_COLS = ["imb10", "imb50", "imb50_last", "dep10", "dep50", "dep50_min",
             *[f"{k}_{s}" for s in "ba" for k in ("wall_frac", "wall_age", "sh_max", "wall_px", "wall_dist", "wall_usd")]]
EV_COLS = [f"{k}_{m}_{s}" for k in ("eat", "fake", "pull") for s in "ba" for m in ("n", "usd", "jump")]


# ── 북 재구성 + 사건 추적 (research_eth_wall_pull_jump_20261004.replay -- 대조 표본 줄은 뺐다) ──
def _best(arr, cur, low_side):
    w = max(int(cur * 0.02), 100)
    lo = max(cur - w, 0)
    nz = np.flatnonzero(arr[lo:cur + w + 1])
    if len(nz):
        return lo + (nz[-1] if low_side else nz[0])
    nz = np.flatnonzero(arr)
    return (nz[-1] if low_side else nz[0]) if len(nz) else -1


def replay(lines, on_sec, on_event, idx_max: int = IDX_MAX) -> str:
    """lines: 완성된 줄 반복자(첫 줄 = 스냅숏). 끊기면 상태 문자열을 돌려준다 -- 그 뒤 북은 복구 불가(다음 시 파일 스냅숏에서 다시)."""
    bid = np.zeros(idx_max); ask = np.zeros(idx_max)
    try:
        first = json.loads(next(lines))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):   # 수집기가 스냅숏 줄을 쓰다 죽고 같은 시에 이어 쓴 파일
        return "no_snapshot"
    snap = first.get("_snapshot") if isinstance(first, dict) else None
    if not snap:
        return "no_snapshot"
    for arr, key in ((bid, "bids"), (ask, "asks")):
        for p, q in snap[key]:
            i = int(round(float(p) * TICK))
            if i < idx_max:
                arr[i] = float(q)
    last_u, synced = snap["lastUpdateId"], False
    bb, ba = int(np.flatnonzero(bid)[-1]), int(np.flatnonzero(ask)[0])
    med, dep = [np.nan, np.nan], [np.nan, np.nan]
    tracked: dict = {}
    cur_sec = None
    for line in lines:
        try:
            m = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return "truncated"
        if m.get("_gap"):
            return "gap"
        if m.get("e") != "depthUpdate":
            continue
        if not synced:
            if m["u"] < last_u:
                continue
            if not (m["U"] <= last_u <= m["u"]):
                return "sync_fail"
            synced = True
        elif m["pu"] != last_u:
            return "gap"
        last_u, t = m["u"], int(m["T"])
        sec = t // 1000
        if cur_sec is None or sec > cur_sec:
            cur_sec = sec
            mid_i = (bb + ba) / 2
            w = int(mid_i * 50e-4)
            for side, base, sl in ((0, max(bb - w, 0), bid[max(bb - w, 0):bb + 1]), (1, ba, ask[ba:ba + w + 1])):
                nz = sl[sl > 0]
                med[side] = md = float(np.median(nz)) if len(nz) else np.nan
                dep[side] = float(sl.sum())
                if side == 1:              # 매 초 첫 diff 적용 전 북
                    on_sec(t, bid, ask, bb, ba)
                if np.isfinite(md):
                    for j in np.flatnonzero(sl >= KS[0] * md):
                        tracked.setdefault((side, base + int(j)), [(t, float(sl[j])) + tuple(t if sl[j] >= k * md else None for k in KS)])
            far = int(mid_i * 60e-4)
            for key in [k for k, h in tracked.items() if abs(k[1] - mid_i) > far or (h[-1][0] < t - 1500 and h[-1][2] is None)]:
                del tracked[key]
        pre_bb, pre_ba = bb, ba
        touched = []
        for side, key, arr in ((0, "b", bid), (1, "a", ask)):
            for p, q in m[key]:
                i = int(round(float(p) * TICK))
                if i < idx_max:
                    arr[i] = float(q)
                    touched.append((side, i, float(q)))
        bb, ba = _best(bid, bb, True), _best(ask, ba, False)
        if bb < 0 or ba < 0:
            return "empty_book"
        pre_mid = (pre_bb + pre_ba) / 2
        for side, i, q in touched:
            md = med[side]
            if not np.isfinite(md):
                continue
            h = tracked.get((side, i))
            if h is None:
                if q >= KS[0] * md and abs(i - pre_mid) <= pre_mid * 50e-4:
                    tracked[(side, i)] = [(t, q) + tuple(t if q >= k * md else None for k in KS)]
                continue
            s_new = tuple((ps if ps is not None else t) if q >= k * md else None for k, ps in zip(KS, h[-1][2:]))
            ref = next((e for e in reversed(h) if e[0] <= t - DROP_WIN), None)
            if ref is not None and ref[2] is not None and q <= (1 - DROP_FRAC) * ref[1] \
                    and (ref[0] - ref[2]) >= 0 and (t - DROP_WIN) - ref[2] >= 1000:
                if ref[1] >= MIN_SHARE * dep[side]:
                    dist = ((pre_mid - i) if side == 0 else (i - pre_mid)) / pre_mid * 1e4
                    on_event(dict(t=t, side=side, px=i / TICK, qref=ref[1], q=q, share=ref[1] / dep[side], dist_bp=dist,
                                  age20=(t - DROP_WIN) - ref[2]))
                del tracked[(side, i)]
                continue
            h.append((t, q) + s_new)
            while len(h) > 2 and h[1][0] <= t - DROP_WIN:
                h.pop(0)
    return "eof"


def sec_row(t, bid, ask, bb, ba):
    """research_orderbook_5m_indicator_20261026.day_job.on_sec 한 줄 -- 도장 t−1."""
    m = (bb + ba) / 2
    w50, w10 = int(m * 50e-4), int(m * 10e-4)
    b50, a50 = bid[bb - w50:bb + 1], ask[ba:ba + w50 + 1]
    jb, ja = int(b50.argmax()), int(a50.argmax())
    return (t - 1, bb, ba, bid[bb - w10:bb + 1].sum(), ask[ba:ba + w10 + 1].sum(), b50.sum(), a50.sum(),
            bb - w50 + jb, b50[jb], ba + ja, a50[ja])


def tape_bad(sec, x):
    """x(초)가 테이프 빈 구간(연속 두 테이프 초 간격 > TAPE_GAP_S) ±TAPE_PAD_S 안이거나 테이프 범위 밖이면 True."""
    s = np.unique(sec)
    if not len(s):
        return np.ones(len(x), bool)
    out = (x < s[0]) | (x > s[-1])
    g = np.flatnonzero(np.diff(s) > TAPE_GAP_S)
    if not len(g):
        return out
    lo, hi = s[g] - TAPE_PAD_S, s[g + 1] + TAPE_PAD_S
    j = np.searchsorted(lo, x, "right") - 1
    return out | ((j >= 0) & (x <= hi[np.clip(j, 0, None)]))


class Bars:
    """1초 줄·사건 -> 5분봉 (research bars_from 을 봉 하나씩, 벽 나이는 계속 든 상태로).
    tape(lo_s, hi_s) -> (rows[(ts_sec, price_bin, buy, sell)], gaps[(from_s, to_s)]) · out(봉 dict, 표식 list)."""

    def __init__(self, tape, out):
        self.tape, self.out = tape, out
        self.age = [({}, {}), ({}, {})]          # 쪽별 (칸 -> 시작, 칸 -> 마지막 본 시각)
        self.rows: dict[int, list] = collections.defaultdict(list)
        self.evs: dict[int, list] = collections.defaultdict(list)
        self.book: dict[int, dict] = {}           # 북 값만 쓴 봉 -> 사건 판정(SETTLE_MS) 대기
        self.mids: collections.deque = collections.deque()
        self.last_t = self.c0 = None

    def chain(self, c0=None):
        """새 사슬(재생 재시작) -- 그 첫 거래소 T 부터 30분 warm. 연구 c0 = 첫 on_sec 의 T(줄 도장 t−1 이 아니라)."""
        self.c0 = c0

    def sec(self, row):
        t = row[0]
        if t == self.last_t:                      # 같은 초 도장 중복 = 첫 것만(연구 drop_duplicates('t'))
            return
        self.last_t = t
        if self.c0 is None:
            self.c0 = t + 1
        _, bb, ba, b10, a10, b50, a50, wb_i, wb_q, wa_i, wa_q = row
        mid = (bb + ba) / 2 / TICK
        r = dict(t=t, mid=mid, warm=(t - self.c0) < WARM_MS,
                 imb10=(b10 - a10) / (b10 + a10), imb50=(b50 - a50) / (b50 + a50), dep10=(b10 + a10) * mid, dep50=(b50 + a50) * mid)
        for s, (i, q, d) in enumerate(((wb_i, wb_q, b50), (wa_i, wa_q, a50))):
            sh = q / d
            on = sh >= WALL_SHARE
            age = 0.0
            if on:                                # research wall_age(인과) -- 이 칸이 10초 넘게 최대 벽이 아니었으면 새로
                start, seen = self.age[s]
                if t - seen.get(i, -10**18) > WALL_BACK_MS:
                    start[i] = t
                seen[i] = t
                age = t - start[i]
            r[s] = (sh, on, age, i, abs(i / TICK - mid) / mid * 1e4, q * mid)
        self.rows[t // BAR].append(r)
        self.mids.append((t, mid))
        while self.mids and self.mids[0][0] < t - 1_200_000:
            self.mids.popleft()
        for b in [b for b in self.rows if (b + 1) * BAR <= t]:
            self.close_book(b)
        for b in [b for b in self.book if (b + 1) * BAR + SETTLE_MS <= t]:
            self.close(b)
        if t // 1000 % 600 == 0:                  # 10분마다 정리: 10초 넘게 안 본 벽 칸은 다음에 보면 새로 센다(지워도 같다) · 줄 없는 봉의 사건
            for start, seen in self.age:
                for i in [i for i, v in seen.items() if t - v > WALL_BACK_MS]:
                    del start[i], seen[i]
            for k in [k for k in self.evs if k not in self.rows and k not in self.book and (k + 1) * BAR + SETTLE_MS <= t]:
                del self.evs[k]

    def event(self, e):
        self.evs[e["t"] // BAR].append(e)

    def _mid_at(self, x):
        """x 이하 마지막 1초 mid (3초 넘게 비면 NaN) -- research _mid_at."""
        ts = [m[0] for m in self.mids]
        i = int(np.searchsorted(ts, x, "right")) - 1
        return self.mids[i][1] if i >= 0 and x - ts[i] <= 3000 else np.nan

    def close_book(self, b):
        """봉 끝: 북 값만 계산해 먼저 내보낸다(out(B, None)) -- 사건 열은 close 가 SETTLE_MS 뒤에 채운다."""
        R = self.rows.pop(b)
        mids = [r["mid"] for r in R]
        B = dict(t=b * BAR, n_sec=len(R), warm=int(any(r["warm"] for r in R)), o=mids[0], h=max(mids), l=min(mids), c=mids[-1],
                 imb10=float(np.mean([r["imb10"] for r in R])), imb50=float(np.mean([r["imb50"] for r in R])), imb50_last=R[-1]["imb50"],
                 dep10=float(np.mean([r["dep10"] for r in R])), dep50=float(np.mean([r["dep50"] for r in R])),
                 dep50_min=min(r["dep50"] for r in R))
        for s, n in ((0, "b"), (1, "a")):
            ages = [r[s][2] for r in R]
            B[f"wall_frac_{n}"] = float(np.mean([r[s][1] for r in R]))
            B[f"wall_age_{n}"] = max(ages) / 1000
            B[f"sh_max_{n}"] = max(r[s][0] for r in R)
            k = int(np.argmax([a if r[s][1] else -1.0 for a, r in zip(ages, R)]))   # 봉 안 가장 오래 선 벽(벽 아닌 초는 안 뽑힘 · 같으면 앞)
            on = R[k][s][1]
            B[f"wall_px_{n}"] = R[k][s][3] / TICK if on else None
            B[f"wall_dist_{n}"] = R[k][s][4] if on else None
            B[f"wall_usd_{n}"] = R[k][s][5] if on else None
        if B["warm"]:
            B.update({k: None for k in BOOK_COLS})
        B["ret"] = float(np.log(B["c"] / B["o"]) * 1e4)
        B["rng"] = float(np.log(B["h"] / B["l"]) * 1e4)
        self.book[b] = B
        self.out(B, None)

    def close(self, b):
        if b in self.rows:
            self.close_book(b)
        B = self.book.pop(b)
        first: dict = {}
        for e in self.evs.pop(b, []):
            first.setdefault((e["t"], e["side"], e["px"]), e)    # 중복은 첫 것(연구 drop_duplicates)
        E = list(first.values())
        lo_s, hi_s = b * BAR // 1000, (b + 1) * BAR // 1000
        if E:   # 체결 몫: [사건−1초, 사건] 초들의 그 칸 반대쪽 테이커 · 테이프 빈 구간(±60초)·수집기 gaps 근처는 판정 불가(research fill_share·tape_bad)
            rows, gaps = self.tape(lo_s - 180, hi_s + 80)
            # 창 양 끝에 테이프가 있다고 본다(연구는 하루 전체 테이프) -- 창 밖에서 이어진 빈 구간의 ±60초도 잡힌다(리뷰 10-26)
            bad = tape_bad(np.r_[lo_s - 180, [r[0] for r in rows], hi_s + 75], np.array([e["t"] // 1000 for e in E]))
            cell = collections.defaultdict(float)
            for ts, pb, bq, sq in rows:
                cell[(ts, pb, 0)] += sq; cell[(ts, pb, 1)] += bq    # 매수벽(0)을 먹는 건 테이커 매도
            for e, bd in zip(E, bad):
                if bd or any(g0 - TAPE_PAD_S <= e["t"] // 1000 <= g1 + TAPE_PAD_S for g0, g1 in gaps):
                    e["fill"] = np.nan; continue
                pb = int(round(e["px"] / BUCKET))
                tot = sum(cell.get((s, pb, e["side"]), 0.0) for s in range((e["t"] - 1000) // 1000, e["t"] // 1000 + 1))
                e["fill"] = min(1.0, tot / max(e["qref"] - e["q"], 1e-12))
        n_usd = {k: 0.0 for k in EV_COLS if "_jump_" not in k}
        jumps = collections.defaultdict(list)
        marks, known = [], not B["warm"]
        for e in E:
            if not (e["share"] >= WALL_SHARE and e["age20"] >= BIG_AGE_MS and e["dist_bp"] <= 50):
                continue
            f = e["fill"]
            kind = ("unk" if np.isnan(f) else "eat" if f >= EAT_FILL else "fake" if f < PULL_FILL and e["dist_bp"] <= NEAR_BP
                    else "pull" if f < PULL_FILL else "mixed")
            known &= kind != "unk"
            if kind in ("unk", "mixed"):
                continue
            n = "ba"[e["side"]]
            n_usd[f"{kind}_n_{n}"] += 1
            n_usd[f"{kind}_usd_{n}"] += (e["qref"] - e["q"]) * e["px"]
            post = self._mid_at(min(e["t"] + JUMP_WIN, (b + 1) * BAR - 1))      # 봉 끝에서 자른다
            jumps[f"{kind}_jump_{n}"].append(np.log(post / self._mid_at(e["t"] - JUMP_WIN)) * 1e4 * (1 if e["side"] == 1 else -1))
            if kind in ("eat", "fake"):
                marks.append(dict(t=e["t"], bar_t=b * BAR, side="bid" if e["side"] == 0 else "ask", kind=kind, px=e["px"],
                                  usd=(e["qref"] - e["q"]) * e["px"], dist_bp=e["dist_bp"]))
        if known:   # 판정 못 한 사건이 있는 봉 · 덜 찬 봉 = 0 이 아니라 모름(None)
            B.update(n_usd)
            for k in EV_COLS:
                if "_jump_" in k:
                    j = [x for x in jumps.get(k, []) if not np.isnan(x)]
                    B[k] = float(max(j)) if j else None
        else:
            B.update({k: None for k in EV_COLS})
        self.out(B, marks if known else [])


# ── 저장 ──
def db_open(path=DB):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    cols = ["n_sec", "warm", "o", "h", "l", "c", *BOOK_COLS, "ret", "rng", *EV_COLS]
    con.execute(f"CREATE TABLE IF NOT EXISTS bars(t INTEGER PRIMARY KEY, {', '.join(c + ' REAL' for c in cols)})")
    con.execute("CREATE TABLE IF NOT EXISTS marks(t INTEGER, bar_t INTEGER, side TEXT, kind TEXT, px REAL, usd REAL, dist_bp REAL, "
                "PRIMARY KEY (t, side, px))")
    return con


def db_write(con, B, marks, mine=None) -> bool:
    """이미 있는 온전한(warm 아님) 봉은 덮지 않는다 -- 재시작 뒤 재생은 짧은 사슬이라 벽 나이가 잘린 더 나쁜 값이다(리뷰 10-26).
    marks None = 봉 끝에 북 값만 먼저(사건 열 NULL) -- 그 봉은 mine 에 들어 SETTLE_MS 뒤 같은 프로세스의 사건 쓰기만 덮을 수 있다.
    ponytail: 두 쓰기 사이(~70초)에 워커가 죽으면 그 봉 사건 열은 NULL(모름)로 남는다 -- 화면은 표식 없음으로 보일 뿐."""
    old = con.execute("SELECT warm FROM bars WHERE t = ?", (B["t"],)).fetchone()
    if old is not None and not old[0] and B["t"] not in (mine or ()):
        return False
    con.execute(f"INSERT OR REPLACE INTO bars({', '.join(B)}) VALUES ({', '.join('?' * len(B))})", list(B.values()))
    if marks is not None:
        con.execute("DELETE FROM marks WHERE bar_t = ?", (B["t"],))
        con.executemany("INSERT OR REPLACE INTO marks VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [(m["t"], m["bar_t"], m["side"], m["kind"], m["px"], m["usd"], m["dist_bp"]) for m in marks])
    con.commit()
    if mine is not None:
        (mine.add if marks is None else mine.discard)(B["t"])
    return True


# ── 실시간 ──
def hour_path(h: int) -> Path | None:
    name = time.strftime("%Y-%m-%dT%H", time.gmtime(h * 3600))
    for p in (DD / f"{name}.jsonl", DD / f"{name}.jsonl.gz"):   # .jsonl 먼저 -- 수집기는 .gz 를 다 쓴 뒤에야 .jsonl 을 지운다(압축 중 .gz 는 반쪽)
        if p.exists():
            return p
    return None


class Source:
    """시 파일을 이어 완성된 줄만 내준다(①). 다음 시 파일이 생기면 지금 파일을 EOF 까지 읽고 넘어간다."""

    def __init__(self, hour: int):
        self.hour, self.n, self.fh, self.start = hour, 0, None, hour

    def lines(self):
        buf, self.start = b"", self.hour        # 사슬 첫 파일(restart 규칙)
        while True:
            if self.fh is None:
                p = hour_path(self.hour)
                if p is None:
                    if time.time() > (self.hour + 1) * 3600 + 120:   # 수집기가 그 시 통째로 죽었다 -- 건너뛴다(다음 파일 둘째 줄 pu 끊김 -> 그 스냅숏에서)
                        self.hour += 1
                    time.sleep(2)
                    continue
                self.fh, self.n, buf = (gzip.open(p, "rb") if p.suffix == ".gz" else open(p, "rb")), 0, b""
            chunk = self.fh.read(1 << 20)
            if chunk:
                buf += chunk
                *done, buf = buf.split(b"\n")
                for ln in done:
                    if ln:
                        self.n += 1
                        yield ln
                continue
            if hour_path(self.hour + 1) is None and time.time() <= (self.hour + 1) * 3600 + 120:   # 지금 시가 아직 자란다(시 끝 2분 넘게 다음 파일이 없으면 수집기가 죽었다 -- 넘어간다)
                time.sleep(1)
                continue
            for ln in (buf + self.fh.read()).split(b"\n"):   # 새 시 파일이 생겼다 = 수집기가 이 파일을 닫았다 -- 남은 것까지
                if ln:
                    self.n += 1
                    yield ln
            self.fh.close(); self.fh = None; self.hour += 1

    def restart(self, st: str):
        """끊긴 뒤: 사슬 첫 파일이 아닌 파일 첫머리(줄 ≤2)의 끊김이면 그 파일 스냅숏에서, 아니면 다음 시 파일에서(연구 day_job 의 `k > i`) --
        사슬 첫 파일에서 다시 끊기면 같은 파일을 영원히 돌았다(리뷰 10-26)."""
        if self.fh:
            self.fh.close(); self.fh = None
        if not (self.n <= 2 and self.hour > self.start):
            self.hour += 1


def hot_tape(lo: int, hi: int):
    from scripts.data_store import read_rows
    try:
        return _hot_tape(read_rows, lo, hi)
    except (sqlite3.Error, OSError) as exc:     # 테이프를 못 읽으면 그 봉 사건은 판정 불가(NULL) -- 죽지 않는다
        print(f"테이프 읽기 실패 {exc!r}", flush=True)
        return [], []


def _hot_tape(read_rows, lo, hi):
    rows = read_rows(TAPE, "SELECT ts_sec, price_bin, buy_qty, sell_qty FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ?",
                     (SYM.lower(), lo, hi))
    gaps = read_rows(TAPE, "SELECT from_ms / 1000, to_ms / 1000 FROM gaps WHERE symbol = ? AND to_ms >= ? AND from_ms < ?",
                     (SYM.lower(), lo * 1000, hi * 1000))
    return [(int(a), int(b), float(c or 0), float(d or 0)) for a, b, c, d in rows], [(int(a), int(b)) for a, b in gaps]


def main():
    con = db_open()
    last = con.execute("SELECT max(t) FROM bars WHERE warm = 0").fetchone()[0]
    now_h = int(time.time()) // 3600
    # 지난 온전한 봉 1시간 앞 시 파일부터(30분 warm 이 그 앞에서 끝나게) -- 오래 멈췄으면 최근 26시간만
    h0 = max(now_h - 26, (int(last) // 3_600_000 - 1) if last else now_h - 1)
    print(f"시작 · 마지막 온전한 봉 {last} · {time.strftime('%m-%d %H시', time.gmtime(h0 * 3600))} 파일부터", flush=True)
    n, mine = [0], set()

    def out(B, marks):
        if db_write(con, B, marks, mine) and marks is not None:
            n[0] += 1
            if B["t"] // BAR % 12 == 0 or marks:
                print(f"봉 {time.strftime('%m-%d %H:%M', time.gmtime(B['t'] / 1000))} warm={B['warm']} 기울기={B['imb50']} 표식={len(marks)} (누적 {n[0]})", flush=True)

    bars, src = Bars(hot_tape, out), Source(h0)
    while True:
        bars.chain()
        st = replay(src.lines(), lambda t, bid, ask, bb, ba: bars.sec(sec_row(t, bid, ask, bb, ba)), bars.event)
        print(f"사슬 끝 {st} · {time.strftime('%m-%d %H시', time.gmtime(src.hour * 3600))} {src.n}줄 -- 스냅숏에서 다시", flush=True)
        src.restart(st)


# ── 대조·백필·자체점검 ──
MIRROR = Path.home() / "backups/crypto-scalping-server/data"
RESEARCH = Path.home() / "crypto-scalping/.claude/worktrees/orderbook-5min-indicator-67e28f/tmp/ob5m_20261026"


def lake_tape(lo: int, hi: int):
    import pandas as pd
    days = sorted({time.strftime("%Y-%m-%d", time.gmtime(s)) for s in (lo, hi)})
    T = pd.concat([pd.read_parquet(p, columns=["ts_sec", "price_bin", "buy_qty", "sell_qty"])
                   for d in days for p in (MIRROR / f"lake/binance/tape/coin=ETH/date={d}").glob("*.parquet")])
    T = T[(T.ts_sec >= lo) & (T.ts_sec < hi)]
    return list(T.itertuples(index=False, name=None)), []


def parity(day: str) -> bool:
    """연구 bars v2 와 대조. 연구처럼 날마다 전날 22시 파일부터 사슬(끊기면 그 파일 첫머리 또는 다음 파일 스냅숏에서 다시) · keep 00~24시 ·
    전날도 같이 돌려 벽 나이를 잇는다. 통과 = warm 아닌 봉에서 imb10·imb50·dep10·dep50·wall_px·wall_age·eat/fake/pull_n 이 1e-9 안."""
    import pandas as pd
    d1 = pd.Timestamp(day + "T00:00Z")
    got = {}
    bars = Bars(lake_tape, lambda B, m: got.__setitem__(B["t"], B))
    all_files = sorted(MIRROR.glob(f"live/orderflow/depthdiff/{SYM}/*.jsonl.gz"))
    for d in (d1 - pd.Timedelta(days=1), d1):
        lo = d.value // 10**6
        hi = lo + 86_400_000
        files = [f for f in all_files if lo - 7_200_000 <= pd.Timestamp(f.name[:13] + ":00Z").value // 10**6 < hi]
        chains, evs, i = [], [], 0
        while i < len(files):
            cur, rows = [0, 0], []

            def gen(fs=files[i:]):
                for k, p in enumerate(fs):
                    cur[0], cur[1] = k, 0
                    with gzip.open(p, "rb") as f:
                        for ln in f:
                            cur[1] += 1
                            yield ln
            g = gen()
            st = replay(g, lambda t, bid, ask, bb, ba: rows.append(sec_row(t, bid, ask, bb, ba)), evs.append)
            g.close()
            if rows:
                chains.append(rows)
            if st == "eof":
                break
            k = i + cur[0]
            i = k if (cur[1] <= 2 and k > i) else k + 1
        # 실시간처럼 시간 순서로 섞어 넣는다(사건이 봉 마감 전에 들어가야 한다 -- 1초 줄을 다 넣고 사건을 넣으면 봉이 이미 닫혀 사건이 버려졌다)
        feed = [(r[0], 1, k, r) for k, rows in enumerate(chains) for r in rows if lo <= r[0] < hi] + \
               [(e["t"], 0, -1, e) for e in evs if lo <= e["t"] < hi]
        cur_chain = None
        for _, kind, k, x in sorted(feed, key=lambda f: (f[0], f[1])):
            if kind == 0:
                bars.event(x); continue
            if k != cur_chain:
                bars.chain(chains[k][0][0] + 1); cur_chain = k
            bars.sec(x)
    for b in sorted({*bars.rows, *bars.book}):
        bars.close(b)
    R = pd.read_parquet(RESEARCH / "bars_ETH.parquet")
    R = R[(R.index >= d1) & (R.index < d1 + pd.Timedelta(days=1))]
    cols = ["imb10", "imb50", "dep10", "dep50", "wall_px_b", "wall_px_a", "wall_age_b", "wall_age_a",
            *[f"{k}_n_{s}" for k in ("eat", "fake", "pull") for s in "ba"]]
    bad, n = collections.Counter(), 0
    for ts, r in R.iterrows():
        if r.warm:
            continue
        g = got.get(ts.value // 10**6)
        if g is None:
            bad["봉 없음"] += 1; continue
        n += 1
        for c in cols:
            a, v = r[c], g[c]
            if (pd.isna(a) and v is None) or (v is not None and not pd.isna(a) and abs(a - v) <= 1e-9 * max(1.0, abs(a))):
                continue
            bad[c] += 1
            if bad[c] <= 3:
                print(f"  {ts} {c}: 연구 {a} · 이식 {v}")
    print(f"parity {day}: 대조 봉 {n} · 불일치 {dict(bad) or '없음'}")
    return n > 0 and not bad


def seed(bars_pq: str, events_pq: str):
    """연구 v2 산출물(로컬 백업 미러로 만든 봉·사건)의 마지막 8일을 DB 에 한 번 싣는다 -- 지난 1주일 화면용. 실시간 워커가 그 뒤를 잇는다."""
    import pandas as pd
    R = pd.read_parquet(bars_pq)
    R = R[R.index >= R.index.max() - pd.Timedelta(days=8)]
    E = pd.read_parquet(events_pq).drop_duplicates(["t", "side", "px"])
    E = E[(E.share >= WALL_SHARE) & (E.age20 >= BIG_AGE_MS) & (E.dist_bp <= 50) & E.fill.notna()]
    E = E[(E.fill >= EAT_FILL) | ((E.fill < PULL_FILL) & (E.dist_bp <= NEAR_BP))]
    con, n = db_open(), 0
    for ts, r in R.iterrows():
        t = ts.value // 10**6
        B = {"t": t, "warm": int(bool(r.warm)),
             **{k: (None if pd.isna(r[k]) else float(r[k])) for k in ["n_sec", "o", "h", "l", "c", *BOOK_COLS, "ret", "rng", *EV_COLS]}}
        M = E[E.t // BAR * BAR == t] if (not B["warm"] and B["eat_n_a"] is not None) else E.iloc[:0]
        n += db_write(con, B, [dict(t=int(e.t), bar_t=t, side="bid" if e.side == 0 else "ask", kind="eat" if e.fill >= EAT_FILL else "fake",
                                    px=float(e.px), usd=float((e.qref - e.q) * e.px), dist_bp=float(e.dist_bp)) for e in M.itertuples()])
    print(f"seed {n}봉 · 마지막 {R.index.max()}")


def selftest():
    """봉 경계·벽 나이 이어 세기·가짜벽 판정·판정 불가·warm·덮어쓰기 규칙 (연구 selftest 와 같은 답)."""
    row = lambda t: (t, 9999, 10021 if t >= 300_000 else 10001, 3.0, 1.0, 100.0, 100.0, 9990, 1.0,   # noqa: E731
                     10012 if 100_000 <= t < 200_000 else 10010, 5.0)   # 매도 5% 벽 칸 10010 · 100~200초 다른 칸 · 300초부터 mid +10틱

    def run(evs, tape, n_sec=700, c0=-10**9, every=False):
        got = []
        bars = Bars(tape, lambda B, m: (every or m is not None) and got.append((dict(B), m)))
        bars.chain(c0)
        for e in evs:
            bars.event(e)
        for k in range(n_sec):
            bars.sec(row(k * 1000 + 1))
        return got
    full = lambda lo, hi: ([(s, 1001, 0.0, 0.0) for s in range(lo, hi)], [])   # noqa: E731  빈 구간 없는 테이프, 그 칸 체결 0
    fake = dict(t=299_999, side=1, px=100.10, qref=5.0, q=0.0, share=0.05, dist_bp=3.0, age20=40_000)
    (B0, M0), *_ = run([fake], full)
    assert B0["n_sec"] == 300 and abs(B0["imb10"] - 0.5) < 1e-12 and abs(B0["imb50"]) < 1e-12
    assert B0["wall_frac_a"] == 1 and B0["wall_frac_b"] == 0 and B0["wall_px_b"] is None
    assert abs(B0["wall_age_a"] - 99) < 1e-9, B0["wall_age_a"]           # 200초에 칸 복귀 -> 200~299초 = 99초
    assert B0["fake_n_a"] == 1 and [m["kind"] for m in M0] == ["fake"] and M0[0]["side"] == "ask"
    assert B0["eat_jump_a"] is None and B0["pull_n_a"] == 0 and abs(B0["fake_jump_a"]) < 1e-9   # +5초 = 봉 1 이지만 봉 끝에서 잘림
    sec = np.r_[np.arange(0, 101), np.arange(200, 301)]
    assert tape_bad(sec, np.array([10, 39, 41, 150, 259, 261, 500])).tolist() == [False, False, True, True, True, False, True]
    (B1, M1), *_ = run([fake], lambda lo, hi: ([], []))                  # 테이프 없음 = 판정 불가 -> 사건 열 None · 표식 없음
    assert B1["fake_n_a"] is None and B1["eat_n_b"] is None and M1 == []
    (B2, M2), *_ = run([fake], full, c0=1)                              # 사슬 시작 30분 = warm -> 북·사건 값 None
    assert B2["warm"] == 1 and B2["imb50"] is None and B2["fake_n_a"] is None
    P = run([fake], full, n_sec=320, every=True)                       # 봉 0 북 값은 봉 끝(300초) 바로 · 사건 열은 370초 전엔 없다
    assert [m for _, m in P] == [None] and abs(P[0][0]["imb10"] - 0.5) < 1e-12 and "fake_n_a" not in P[0][0]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        con = db_open(Path(d) / "x.sqlite")
        assert db_write(con, B0, M0) and not db_write(con, B2, [])      # 온전한 봉을 warm 봉이 못 덮는다
        assert con.execute("SELECT count(*) FROM marks").fetchone()[0] == 1
        mine, t1 = set(), B0["t"] + BAR                                 # 북 먼저 -> 같은 프로세스 사건 쓰기만 덮는다 · 다른 프로세스(빈 mine)는 못 덮는다
        Bp = {k: v for k, v in B0.items() if k not in EV_COLS} | {"t": t1}
        assert db_write(con, Bp, None, mine) and mine == {t1} and not db_write(con, Bp, None, set())
        assert con.execute("SELECT fake_n_a FROM bars WHERE t = ?", (t1,)).fetchone()[0] is None
        assert db_write(con, {**B0, "t": t1}, [], mine) and mine == set() and not db_write(con, Bp, None, mine)
        assert con.execute("SELECT fake_n_a FROM bars WHERE t = ?", (t1,)).fetchone()[0] == 1
        # 따라 읽기(①): 줄을 반쯤 쓰다 멈춰도 완성된 줄만 · 다음 시 파일이 생기면 옛 파일 끝까지 · 옛 파일이 gz 로 바뀌고 지워져도 그대로
        import threading
        global DD
        DD, h = Path(d), int(time.time()) // 3600     # 지금 시(지난 시는 «수집기 죽음»으로 보고 바로 넘어간다)
        p0, p1 = DD / time.strftime("%Y-%m-%dT%H.jsonl", time.gmtime(h * 3600)), DD / time.strftime("%Y-%m-%dT%H.jsonl", time.gmtime((h + 1) * 3600))

        def writer():
            with open(p0, "wb") as f:
                for k in range(5):
                    f.write(f'{{"n": {k}}}\n'.encode()[:5]); f.flush(); time.sleep(0.3)   # 반쪽 줄
                    f.write(f'{{"n": {k}}}\n'.encode()[5:]); f.flush()
                f.write(b'{"n": 5}\n{"n": 6')                                             # 반쪽 줄로 잠깐 멈춘다(수집기는 옛 파일을 닫은 뒤 새 시 파일을 연다)
                f.flush(); time.sleep(0.3)
                f.write(b'}\n')
            p1.write_bytes(b'{"n": 100}\n')
            with open(p0, "rb") as fi, gzip.open(str(p0) + ".gz", "wb") as fo:
                fo.write(fi.read())
            p0.unlink()
        th = threading.Thread(target=writer); th.start()
        src, got = Source(h), []
        for ln in src.lines():
            got.append(json.loads(ln)["n"])
            if got[-1] == 100:
                break
        th.join()
        assert got == [0, 1, 2, 3, 4, 5, 6, 100] and src.hour == h + 1, got
        # 끊김 뒤 다시: 사슬 첫 파일 첫머리 끊김 = 다음 시(같은 파일 무한 반복 금지) · 이어 읽던 다음 파일 첫머리 끊김 = 그 파일 스냅숏
        s1 = Source(h); s1.n = 2; s1.restart("sync_fail")
        s2 = Source(h); s2.hour, s2.n = h + 1, 2; s2.restart("gap")
        assert (s1.hour, s2.hour) == (h + 1, h + 1)
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--parity")
    ap.add_argument("--seed", nargs=2)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.parity:
        sys.exit(0 if parity(a.parity) else 1)
    elif a.seed:
        seed(*a.seed)
    else:
        main()
