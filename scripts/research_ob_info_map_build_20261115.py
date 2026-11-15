#!/usr/bin/env python3
"""호가 정보 지도 1초 텐서 빌더 (사전등록 docs/prereg/ob_info_map_dl_20261115.md, 커밋 87178d89).

코인·UTC 날짜 하나 = 작업 하나 → tmp/ob_info_map_20261115/<COIN>/<YYYY-MM-DD>.npz (86400초):
  X     (86400, 2, 64, 4) float16  log1p(명목 USD) · 쪽(0 매수·1 매도) × bp 칸 × (rest, add, rem, fill)
  S     (86400, 9) float32         spread_bp, ret_bp, taker_buy_usd, taker_sell_usd, n_upd, n_buy_ord, n_sell_ord, best_bid_usd, best_ask_usd
  mid, bb, ba (86400,) float64     초 끝 중간가·최우선(가격) — 대상 계산 전용(입력 수준 열 아님)
  ok    (86400,) uint8             북 유효(사슬 안 · warm 30분 지남)
  tape_ok (86400,) uint8           15초+ 테이프 빈 구간 ±60초 밖
  sell_edge, buy_edge (86400,)     그 초 매도 테이커 체결 칸 윗변 최소 · 매수 테이커 칸 아랫변 최대(관통 체결 판정용, 없으면 ±inf)
  lob10 (86400, 40) float32        상위 10호가 [매도가·매도량·매수가·매수량]×10 (DeepLOB 재현 팔)
결정 시각 τ = 초 s 끝: 상태 = T < (s+1)·1000 인 diff 전부 적용한 북 · add/rem = 그 초 [s·1000, (s+1)·1000) diff 의 Δq · 칸 위치는 초 끝 중간가 기준.
재생 규칙 = scripts/live_ob5m_20261026.replay 와 같음(첫 줄 스냅숏 · U/u/pu · 끊기면 다음 시 파일 스냅숏) — 그 함수는 diff 를 밖에 안 내줘서 같은 동기화 규칙으로 다시 썼다.
날짜 D 작업 = D−1 22:00 파일부터 재생(사슬 시작 뒤 30분 = warm) · 주문 없음 · 거래소 호출 없음 · 로컬 미러만 읽음.
사용: python scripts/research_ob_info_map_build_20261115.py --day ETH 2026-11-10 | --selftest
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import os
MIRROR = Path(os.environ.get("OBIM_MIRROR", Path.home() / "backups/crypto-scalping-server/data"))   # 서버에선 저장소 data/
OUT = Path(os.environ.get("OBIM_OUT", ROOT / "tmp/ob_info_map_20261115"))
# 틱(1/$)·idx 상한 = live_ob5m COINS · 테이프 칸($) = 수집기 BUCKETS
COIN = {"ETH": (100, 1_000_000, 0.1), "BTC": (10, 3_000_000, 1.0), "SOL": (100, 1_000_000, 0.01), "XRP": (10_000, 1_000_000, 0.0001)}
EDGES = np.r_[np.arange(0, 8, 0.5), np.arange(8, 24, 1.0), np.arange(24, 56, 2.0), np.arange(56, 121, 4.0)]   # 65 경계 → 64칸, 끝 120bp
NB, MAXBP, WARM_S, DAY = 64, 120.0, 1800, 86400
assert len(EDGES) == NB + 1 and EDGES[-1] == MAXBP


def bucket(off_bp):
    """bp 거리 → 칸(0..63), 120bp 이상은 −1."""
    b = np.searchsorted(EDGES, off_bp, side="right") - 1
    return np.where((off_bp >= 0) & (off_bp < MAXBP), b, -1)


class Day:
    """초 끝마다 한 줄씩 채운다. day0 = 그 날 00:00 UTC 초."""

    def __init__(self, day0, tick):
        self.day0, self.tick = day0, tick
        self.X = np.zeros((DAY, 2, NB, 4), np.float32)
        self.S = np.zeros((DAY, 9), np.float32)
        self.mid, self.bb, self.ba = (np.full(DAY, np.nan) for _ in range(3))
        self.ok = np.zeros(DAY, np.uint8)
        self.lob10 = np.full((DAY, 40), np.nan, np.float32)

    def put(self, s, bid, ask, bbi, bai, flow, n_upd, chain0, prev_mid):
        k = s - self.day0
        if not 0 <= k < DAY:
            return
        t = self.tick
        mid = (bbi + bai) / 2 / t
        self.mid[k], self.bb[k], self.ba[k] = mid, bbi / t, bai / t
        w = int(mid * t * MAXBP * 1e-4) + 2
        for side, lo, hi in ((0, max(bbi - w, 0), bbi + 1), (1, bai, min(bai + w, len(ask)))):
            arr = bid if side == 0 else ask
            seg = arr[lo:hi]
            nz = np.flatnonzero(seg)
            if len(nz):
                px = (lo + nz) / t
                off = (mid - px) / mid * 1e4 if side == 0 else (px - mid) / mid * 1e4
                b = bucket(off)
                m = b >= 0
                self.X[k, side, :, 0] = np.bincount(b[m], weights=seg[nz][m] * px[m], minlength=NB)
            top = nz[::-1][:10] if side == 0 else nz[:10]
            c = 2 if side == 0 else 0
            self.lob10[k, c:40:4][:len(top)] = (lo + top) / t
            self.lob10[k, c + 1:40:4][:len(top)] = seg[top]
        for (side, i), (a, r) in flow.items():
            px = i / t
            off = (mid - px) / mid * 1e4 if side == 0 else (px - mid) / mid * 1e4
            b = int(bucket(np.array([off]))[0])
            if b >= 0:
                self.X[k, side, b, 1] += a * px
                self.X[k, side, b, 2] += r * px
        self.S[k, 0] = (bai - bbi) / t / mid * 1e4
        self.S[k, 1] = (np.log(mid / prev_mid) * 1e4) if prev_mid else 0.0
        self.S[k, 4] = n_upd
        self.S[k, 7], self.S[k, 8] = bid[bbi] * bbi / t, ask[bai] * bai / t
        self.ok[k] = int(s - chain0 >= WARM_S)


def replay(lines, day, idx_max, tick):
    """한 사슬. 끊김·끝이면 상태 문자열. 동기화 규칙 = live_ob5m.replay."""
    bid = np.zeros(idx_max); ask = np.zeros(idx_max)
    try:
        snap = json.loads(next(lines)).get("_snapshot")
    except (StopIteration, json.JSONDecodeError, UnicodeDecodeError, ValueError, AttributeError):
        return "no_snapshot"
    if not snap:
        return "no_snapshot"
    for arr, key in ((bid, "bids"), (ask, "asks")):
        for p, q in snap[key]:
            i = int(round(float(p) * tick))
            if i < idx_max:
                arr[i] = float(q)
    last_u, synced = snap["lastUpdateId"], False
    bb, ba = int(np.flatnonzero(bid)[-1]), int(np.flatnonzero(ask)[0])
    cur = chain0 = None; flow, n_upd, prev_mid = {}, 0, None
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
        last_u, sec = m["u"], int(m["T"]) // 1000
        if cur is None:
            cur = chain0 = sec
        while sec > cur:                       # 초 cur 끝 = 이 diff 적용 전 북
            day.put(cur, bid, ask, bb, ba, flow, n_upd, chain0, prev_mid)
            prev_mid, flow, n_upd, cur = (bb + ba) / 2 / tick, {}, 0, cur + 1
        n_upd += 1
        for side, key, arr in ((0, "b", bid), (1, "a", ask)):
            for p, q in m[key]:
                i = int(round(float(p) * tick))
                if i < idx_max:
                    q = float(q); d = q - arr[i]; arr[i] = q
                    if d:
                        a, r = flow.get((side, i), (0.0, 0.0))
                        flow[(side, i)] = (a + d, r) if d > 0 else (a, r - d)
        nzb = np.flatnonzero(bid[max(bb - 2000, 0):bb + 2001])
        bb = (max(bb - 2000, 0) + nzb[-1]) if len(nzb) else (int(np.flatnonzero(bid)[-1]) if bid.any() else -1)
        nza = np.flatnonzero(ask[max(ba - 2000, 0):ba + 2001])
        ba = (max(ba - 2000, 0) + nza[0]) if len(nza) else (int(np.flatnonzero(ask)[0]) if ask.any() else -1)
        if bb < 0 or ba < 0 or bb >= ba:
            return "bad_book"
    return "eof"                               # 마지막 초는 다음 diff 가 없어 닫지 않는다(초 끝 상태 미확정)


def hour_files(coin, day0):
    d = MIRROR / f"live/orderflow/depthdiff/{coin}USDT"
    out = []
    for h in range(day0 // 3600 - 2, day0 // 3600 + 24):
        name = np.datetime64(h * 3600, "s").astype(str)[:13]
        p = next((q for q in (d / f"{name}.jsonl.gz", d / f"{name}.jsonl") if q.exists()), None)
        out.append(p)
    return out


def chain_lines(files, pos):
    """시 파일을 이어 준다 · pos[0] = 지금 읽는 파일 순번(끊긴 파일 찾기)."""
    for n, p in enumerate(files):
        pos[0] = n
        with (gzip.open(p, "rb") if p.suffix == ".gz" else open(p, "rb")) as fh:
            for ln in fh:
                if ln.strip():
                    yield ln


def tape_fill(day, coin, B):
    """테이프 → fill 채널·테이커 명목·주문 수·관통 체결 가장자리·tape_ok."""
    import pandas as pd
    D = np.datetime64(day.day0, "s").astype(str)[:10]
    fs = sorted((MIRROR / f"lake/binance/tape/coin={coin}/date={D}").glob("*.parquet"))
    sell_edge, buy_edge = np.full(DAY, np.inf), np.full(DAY, -np.inf)
    tape_ok = np.zeros(DAY, np.uint8)
    if not fs:
        return sell_edge, buy_edge, tape_ok
    T = pd.concat([pd.read_parquet(f, columns=["ts_sec", "price_bin", "buy_qty", "sell_qty", "order_buy_n", "order_sell_n"]) for f in fs])
    T = T[(T.ts_sec >= day.day0) & (T.ts_sec < day.day0 + DAY)]
    k = (T.ts_sec.values - day.day0).astype(int)
    # price_bin = round(가격 / B) (수집기 live_trade_tape_collector:339) → 칸 = [B·bin − B/2, B·bin + B/2] (경계 가격은 반올림 규칙상 양쪽 칸 어디든)
    ctr = T.price_bin.values * B; lo = ctr - B / 2; hi = ctr + B / 2; mid = day.mid[k]
    for side, qty in ((0, T.sell_qty.values), (1, T.buy_qty.values)):   # 매도 테이커 = 매수칸을 먹음 · 매수 테이커 = 매도칸 (칸 가운데로 위치)
        off = (mid - ctr) / mid * 1e4 if side == 0 else (ctr - mid) / mid * 1e4
        b = bucket(np.where(np.isfinite(off), np.maximum(off, 0), -1))
        m = (b >= 0) & (qty > 0)
        np.add.at(day.X[:, side, :, 3], (k[m], b[m]), qty[m] * ctr[m])
    np.add.at(day.S[:, 2], k, T.buy_qty.values * ctr)
    np.add.at(day.S[:, 3], k, T.sell_qty.values * ctr)
    np.add.at(day.S[:, 5], k, T.order_buy_n.values)
    np.add.at(day.S[:, 6], k, T.order_sell_n.values)
    # 관통 체결 가장자리: 매수 지정가 L 은 «칸 윗변 < L»(체결가 ≤ 윗변 < L), 매도 지정가 U 는 «칸 아랫변 > U»(체결가 ≥ 아랫변 > U) — frame 에서 엄격 부등호
    s = T.sell_qty.values > 0; np.minimum.at(sell_edge, k[s], hi[s])
    b_ = T.buy_qty.values > 0; np.maximum.at(buy_edge, k[b_], lo[b_])
    have = np.zeros(DAY, bool); have[k] = True
    tape_ok[:] = 1
    run = 0
    for i in range(DAY):                       # 15초+ 빈 구간 ±60초 = 판정 불가(ob5m 규약)
        run = 0 if have[i] else run + 1
        if run == 15:
            tape_ok[max(i - 14 - 60, 0):i + 1] = 0
        if run >= 15:
            tape_ok[i:min(i + 61, DAY)] = 0
    return sell_edge, buy_edge, tape_ok


def retape(coin, D, out_dir=OUT):
    """이미 만든 날짜 파일의 테이프 몫(fill 채널·S[2:7]·가장자리·tape_ok)만 다시 계산 — 재생 결과(rest·add·rem·mid)는 그대로."""
    f = Path(out_dir) / coin / f"{D}.npz"
    z = dict(np.load(f))
    day = Day(int(np.datetime64(D, "s").astype(np.int64)), COIN[coin][0])
    X = np.expm1(z["X"].astype(np.float32)); X[..., 3] = 0
    day.X, day.S, day.mid = X, z["S"].copy(), z["mid"]
    day.S[:, [2, 3, 5, 6]] = 0
    z["sell_edge"], z["buy_edge"], z["tape_ok"] = tape_fill(day, coin, COIN[coin][2])
    z["X"], z["S"] = np.log1p(day.X).astype(np.float16), day.S
    tmp = f.with_suffix(".tmp.npz"); np.savez(tmp, **z); tmp.replace(f)


def build_day(coin, D, out_dir=OUT):
    tick, idx_max, B = COIN[coin]
    day0 = int(np.datetime64(D, "s").astype(np.int64))
    day = Day(day0, tick)
    files = hour_files(coin, day0)
    log, i = [], 0
    while i < len(files):
        if files[i] is None:
            i += 1; continue
        j = i
        while j < len(files) and files[j] is not None:
            j += 1
        pos = [0]
        st = replay(chain_lines(files[i:j], pos), day, idx_max, tick)
        log.append((i, i + pos[0], st))
        i = j if st == "eof" else i + pos[0] + 1   # 끊기면 끊긴 파일 다음 시 파일 스냅숏부터(같은 파일 첫머리 반복 방지)
    sell_edge, buy_edge, tape_ok = tape_fill(day, coin, B)
    out_dir = Path(out_dir) / coin; out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{D}.npz", X=np.log1p(day.X).astype(np.float16), S=day.S, mid=day.mid, bb=day.bb, ba=day.ba,
             ok=day.ok, tape_ok=tape_ok, sell_edge=sell_edge, buy_edge=buy_edge, lob10=day.lob10)
    return log, int(day.ok.sum())


def selftest():
    """합성 북: 칸 합·add/rem·초 경계 인과·동기화·warm·관통 가장자리."""
    tick = 100
    assert bucket(np.array([0.0, 0.49, 0.5, 7.99, 8.0, 119.9, 120.0]))[::].tolist() == [0, 0, 1, 15, 16, 63, -1]
    snap = {"_snapshot": {"lastUpdateId": 10, "bids": [["100.00", "2"], ["99.90", "5"]], "asks": [["100.02", "3"], ["101.50", "7"]]}}
    msgs = [{"e": "depthUpdate", "T": 1_000_000_000_100, "U": 9, "u": 11, "pu": 0, "b": [["100.00", "4"]], "a": []},
            {"e": "depthUpdate", "T": 1_000_000_000_900, "U": 12, "u": 12, "pu": 11, "b": [], "a": [["100.02", "1"]]},
            {"e": "depthUpdate", "T": 1_000_000_001_200, "U": 13, "u": 13, "pu": 12, "b": [["99.90", "0"]], "a": []},
            {"e": "depthUpdate", "T": 1_000_000_002_000, "U": 14, "u": 14, "pu": 13, "b": [], "a": []}]
    day0 = 1_000_000_000 - 10
    day = Day(day0, tick)
    st = replay(iter([json.dumps(snap)] + [json.dumps(m) for m in msgs]), day, 20_000, tick)
    assert st == "eof", st
    k = 10                                     # 초 1_000_000_000 끝 = 처음 두 diff 만
    assert abs(day.mid[k] - 100.01) < 1e-9 and abs(day.S[k, 0] - 0.02 / 100.01 * 1e4) < 1e-4
    X = day.X
    assert abs(X[k, 0, 1, 0] - 400.0) < 1e-6 and abs(X[k, 0, 18, 0] - 5 * 99.9) < 1e-6      # 100.00 = 0.9999bp 칸 1 · 99.90 = 10.999bp 칸 18
    assert abs(X[k, 1, 1, 0] - 100.02) < 1e-6 and X[k, 1, :, 0].sum() - 100.02 < 1e-6       # 101.50 = 149bp 밖
    assert abs(X[k, 0, 1, 1] - 200.0) < 1e-6 and abs(X[k, 1, 1, 2] - 2 * 100.02) < 1e-6     # add 2 · rem 2
    assert abs(X[k, 0, 18, 0] - 499.5) < 1e-6 and X[k, 0, 18, 2] == 0                        # 인과: 다음 초 diff(99.90 → 0)는 아직 없다
    assert X[11, 0, 18, 0] == 0 and abs(X[11, 0, 18, 2] - 499.5) < 1e-6 and X[11, 0, 1, 1] == 0
    assert np.isnan(day.mid[12]) and day.ok[10] == 0 and day.S[10, 4] == 2 and day.S[11, 4] == 1
    assert day.lob10[k, 2] == 100.00 and day.lob10[k, 3] == 4 and day.lob10[k, 0] == 100.02 and day.lob10[k, 6] == 99.90
    bad = [dict(msgs[0]), dict(msgs[1], pu=999)]
    assert replay(iter([json.dumps(snap)] + [json.dumps(m) for m in bad]), Day(day0, tick), 20_000, tick) == "gap"
    assert replay(iter([json.dumps(snap), json.dumps(dict(msgs[0], U=11, u=12))]), Day(day0, tick), 20_000, tick) == "sync_fail"
    d2 = Day(day0 - 2000, tick)                  # warm: 사슬 시작 30분 뒤부터 ok
    d2.put(1_000_000_000 + 1800, np.r_[np.zeros(10000), 1, np.zeros(9999)], np.r_[np.zeros(10002), 1, np.zeros(9997)], 10000, 10002, {}, 0, 1_000_000_000, None)
    assert d2.ok[1_000_000_000 + 1800 - (day0 - 2000)] == 1
    print("selftest OK")


# ── 대상·손 피쳐 (사전등록 §대상 × 지평 · §팔 ⓪①) ──
BANDS = [(0, 4), (4, 10), (10, 18), (18, 32), (32, 48), (48, 64)]      # 0–2 · 2–5 · 5–10 · 10–24 · 24–56 · 56–120bp (보정 1)
H_DIR, H_RV, H_FILL, DS, H_DEP = (1, 10, 60, 300, 900, 3600), (60, 300, 900, 3600), (10, 60, 300), (0.0, 2e-4), (60, 300)
CUT = {"train": "2026-10-25", "val": "2026-11-04", "test": "2026-11-14T18:00:00"}   # 각 구간 끝(미포함)
COIN_ID = {"ETH": 0, "BTC": 1, "SOL": 2, "XRP": 3}
STRIDE = 5


def reduce_coin(coin, out_dir=OUT):
    """날짜 npz → 연속 초 배열(칸 → 띠 합) 하나. 빠진 날은 ok=0."""
    fs = sorted((Path(out_dir) / coin).glob("20*.npz"))
    d0 = int(np.datetime64(fs[0].stem, "s").astype(np.int64)); d1 = int(np.datetime64(fs[-1].stem, "s").astype(np.int64)) + DAY
    n = d1 - d0
    R = dict(t=np.arange(d0, d1), mid=np.full(n, np.nan), bb=np.full(n, np.nan), ba=np.full(n, np.nan), ok=np.zeros(n, np.uint8),
             tape_ok=np.zeros(n, np.uint8), sell_edge=np.full(n, np.inf), buy_edge=np.full(n, -np.inf),
             S=np.zeros((n, 9), np.float32), B=np.zeros((n, 2, len(BANDS), 4), np.float32))
    for f in fs:
        z = np.load(f); k = int(np.datetime64(f.stem, "s").astype(np.int64)) - d0
        for c in ("mid", "bb", "ba", "ok", "tape_ok", "sell_edge", "buy_edge", "S"):
            R[c][k:k + DAY] = z[c]
        X = np.expm1(z["X"].astype(np.float32))
        for j, (a, b) in enumerate(BANDS):
            R["B"][k:k + DAY, :, j, :] = X[:, :, a:b, :].sum(2)
    np.savez(Path(out_dir) / coin / "_reduced.npz", **R)
    return R


def _cs(x):
    return np.r_[0.0, np.cumsum(np.nan_to_num(np.asarray(x, np.float64)))]


def _past_sum(x, k):
    c = _cs(x); i = np.arange(len(x)) + 1
    return c[i] - c[np.maximum(i - k, 0)]


def _rel24(v):
    """log 값 − 과거 24h(자기 포함) 평균 — 수준 대신 상대(dl_feature_reference §5)."""
    return v - _past_sum(v, DAY) / np.minimum(np.arange(len(v)) + 1, DAY)


def _fwd_ok(bad, h):
    c = _cs(bad); n = len(bad); i = np.arange(n)
    j = np.minimum(i + h, n - 1)
    return ((c[j + 1] - c[i + 1]) == 0) & (i + h < n)


def _fwd_roll(x, h, fn):
    import pandas as pd
    r = getattr(pd.Series(x).rolling(h, min_periods=1), fn)().values    # [i−h+1, i]
    out = np.full(len(x), np.nan); out[:-h] = r[h:]                      # (s, s+h]
    return out


def frame(coin, R=None, out_dir=OUT):
    import pandas as pd
    R = R or dict(np.load(Path(out_dir) / coin / "_reduced.npz"))
    t, ok, tok = R["t"], R["ok"].astype(bool), R["tape_ok"].astype(bool)
    lm = np.where(ok, np.log(R["mid"]), np.nan)
    r1 = np.r_[np.nan, np.diff(lm)]
    S, B = R["S"].astype(np.float64), R["B"].astype(np.float64)
    split = np.select([t < np.datetime64(CUT["train"], "s").astype(np.int64), t < np.datetime64(CUT["val"], "s").astype(np.int64),
                       t < np.datetime64(CUT["test"], "s").astype(np.int64)], ["train", "val", "test"], "fwd")
    keep = ok & (t % STRIDE == 0)

    class _K(dict):
        def __setitem__(self, k, v):
            super().__setitem__(k, np.asarray(v)[keep].astype(np.float32 if k.startswith(("p_", "b_")) else np.float64))
    F = _K()
    # ⓪ 가격·체결
    for k in (1, 5, 10, 30, 60, 300, 900, 3600):
        F[f"p_ret{k}"] = (lm - np.r_[np.full(k, np.nan), lm[:-k]]) * 1e4
    for k in (60, 300, 900, 3600):
        F[f"p_lrv{k}"] = np.log(_past_sum(r1 ** 2, k) + 1e-12)
    tb, ts = S[:, 2], S[:, 3]
    for k in (1, 10, 60, 300):
        b_, s_ = _past_sum(tb, k), _past_sum(ts, k)
        F[f"p_timb{k}"] = np.where(b_ + s_ > 0, (b_ - s_) / np.maximum(b_ + s_, 1e-9), 0.0)
    for k in (10, 60, 300):
        F[f"p_tvol{k}"] = _rel24(np.log1p(_past_sum(tb + ts, k)))
    F["p_nord60"] = _rel24(np.log1p(_past_sum(S[:, 5] + S[:, 6], 60)))
    tod = (t % DAY) / DAY * 2 * np.pi
    F["p_tod_s"], F["p_tod_c"] = np.sin(tod), np.cos(tod)
    F["p_coin"] = np.full(len(t), COIN_ID[coin], float)
    # ① 손 호가
    bq, aq = S[:, 7], S[:, 8]
    F["b_qi1"] = (bq - aq) / np.maximum(bq + aq, 1e-9)
    micro = (R["ba"] * bq + R["bb"] * aq) / np.maximum(bq + aq, 1e-9)
    F["b_micro_bp"] = (micro / R["mid"] - 1) * 1e4
    F["b_spread"] = S[:, 0]
    for j, (a, b) in enumerate(BANDS):
        rb, ra = B[:, 0, j, 0], B[:, 1, j, 0]
        tot = rb + ra
        F[f"b_dimb{j}"] = (rb - ra) / np.maximum(tot, 1e-9)
        F[f"b_ldep{j}"] = _rel24(np.log1p(tot))
        flow = (B[:, 0, j, 1] - B[:, 0, j, 2]) - (B[:, 1, j, 1] - B[:, 1, j, 2])
        for k in (1, 10, 60):
            F[f"b_ofi{j}_{k}"] = _past_sum(flow, k) / np.maximum(_past_sum(tot, k) / k, 1e-9)
        den = np.maximum(_past_sum(tot, 60) / 60, 1e-9)
        add, rem, fil = (_past_sum(B[:, 0, j, c] + B[:, 1, j, c], 60) for c in (1, 2, 3))
        F[f"b_add{j}"], F[f"b_canc{j}"], F[f"b_fill{j}"] = add / den, np.maximum(rem - fil, 0) / den, fil / den
    for k in (10, 60):
        F[f"b_nupd{k}"] = _rel24(np.log1p(_past_sum(S[:, 4], k)))
    # 대상
    Y = _K()
    bad = (~ok).astype(float)
    for h in H_DIR:
        Y[f"y_ret{h}"] = np.where(_fwd_ok(bad, h), np.r_[lm[h:], np.full(h, np.nan)] - lm, np.nan)
    for h in H_RV:
        Y[f"y_rv{h}"] = np.where(_fwd_ok(bad, h), _fwd_roll(np.nan_to_num(r1 ** 2), h, "sum"), np.nan)
        mx, mn = _fwd_roll(lm, h, "max"), _fwd_roll(lm, h, "min")
        Y[f"y_jmp{h}"] = np.where(_fwd_ok(bad, h), np.maximum(mx - lm, lm - mn), np.nan)   # 문턱은 make_targets 에서(학습 99분위)
    badt = (~(ok & tok)).astype(float)
    for h in H_FILL:
        okh = _fwd_ok(badt, h)
        se, be = _fwd_roll(R["sell_edge"], h, "min"), _fwd_roll(R["buy_edge"], h, "max")
        for d in DS:
            tag = f"{h}_{int(d * 1e4)}"
            Y[f"y_fillb{tag}"] = np.where(okh, (se < R["bb"] * (1 - d)).astype(float), np.nan)
            Y[f"y_fills{tag}"] = np.where(okh, (be > R["ba"] * (1 + d)).astype(float), np.nan)
    ld10 = np.log(np.maximum(B[:, :, :3, 0].sum((1, 2)), 1.0))
    for h in H_DEP:
        Y[f"y_dep{h}"] = np.where(_fwd_ok(bad, h), np.r_[ld10[h:], np.full(h, np.nan)] - ld10, np.nan)
    df = pd.DataFrame({"t": t[keep], "coin": coin, "split": split[keep], **F, **Y})
    # 퍼지: 각 구간 끝에서 τ + 최대 창이 경계를 넘는 행의 대상은 그 구간에서 결측(사전등록 §구간)
    for name in ("train", "val", "test"):
        end = np.datetime64(CUT[name], "s").astype(np.int64)
        for c in [c for c in df if c.startswith("y_")]:
            h = int("".join(ch for ch in c.split("_", 1)[1].split("_")[0] if ch.isdigit()))
            df.loc[(df.split == name) & (df.t + h >= end), c] = np.nan
    return df


def make_targets(df):
    """학습 구간 문턱으로 분류 대상 만들기(코인별): 방향 3분류(33·67분위) · log RV(ε = 1%분위) · 급변(99분위). 문턱 dict 도 돌려준다."""
    th = {}
    out = df                                                            # 제자리(메모리 — 4코인 357만 행 복사가 10GB 를 넘었다)
    for coin, g in df.groupby("coin"):
        tr = g[g.split == "train"]; m = df.coin == coin
        for h in H_DIR:
            q1, q2 = np.nanquantile(tr[f"y_ret{h}"], [1 / 3, 2 / 3]); th[(coin, "ret", h)] = (q1, q2)
            r = df.loc[m, f"y_ret{h}"]
            out.loc[m, f"c_dir{h}"] = np.where(r.isna(), np.nan, np.where(r < q1, 0, np.where(r > q2, 2, 1)))
        for h in H_RV:
            eps = np.nanquantile(tr[f"y_rv{h}"], 0.01); th[(coin, "rv", h)] = eps
            out.loc[m, f"c_lrv{h}"] = np.log(df.loc[m, f"y_rv{h}"] + max(eps, 1e-12))
            q = np.nanquantile(tr[f"y_jmp{h}"], 0.99); th[(coin, "jmp", h)] = q
            j = df.loc[m, f"y_jmp{h}"]
            out.loc[m, f"c_jmp{h}"] = np.where(j.isna(), np.nan, (j > q).astype(float))
    return out, th


def targets_selftest():
    """대상 창 인과·퍼지: (s, s+h] 만 · 창 안 결측이면 NaN · 미래 변조가 피쳐를 안 바꿈."""
    n = 4000
    rng = np.random.default_rng(0)
    lmid = np.cumsum(rng.normal(0, 1e-4, n)) + np.log(2000)
    R = dict(t=np.arange(1_790_000_000, 1_790_000_000 + n), mid=np.exp(lmid), bb=np.exp(lmid) * (1 - 1e-5), ba=np.exp(lmid) * (1 + 1e-5),
             ok=np.ones(n, np.uint8), tape_ok=np.ones(n, np.uint8), sell_edge=np.full(n, np.inf), buy_edge=np.full(n, -np.inf),
             S=np.abs(rng.normal(1, .1, (n, 9))).astype(np.float32), B=np.abs(rng.normal(1, .1, (n, 2, 6, 4))).astype(np.float32))
    R["ok"][2000] = 0
    df = frame("ETH", R)
    r = df.set_index("t")
    s = 1_790_000_000 + 1000
    assert abs(r.loc[s, "y_ret10"] - (lmid[1010] - lmid[1000])) < 1e-12
    assert abs(r.loc[s, "y_rv60"] - np.sum(np.diff(lmid[1000:1061]) ** 2)) < 1e-15
    assert np.isnan(r.loc[1_790_000_000 + 1995, "y_ret10"]) and not np.isnan(r.loc[1_790_000_000 + 1985, "y_ret10"])   # 2000 결측이 창 (s, s+10] 에 들면 NaN
    assert abs(r.loc[s, "p_ret10"] - (lmid[1000] - lmid[990]) * 1e4) < 1e-3
    R2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in R.items()}
    R2["mid"][1001:] *= 1.05; R2["S"][1001:] *= 3; R2["B"][1001:] *= 3; R2["bb"][1001:] *= 1.05; R2["ba"][1001:] *= 1.05
    d2 = frame("ETH", R2).set_index("t")
    fc = [c for c in df if c.startswith(("p_", "b_"))]
    assert np.allclose(r.loc[s, fc].astype(float).values, d2.loc[s, fc].astype(float).values, equal_nan=True), "피쳐가 미래를 봄"
    se = np.full(n, np.inf); se[1007] = R["bb"][1000] * (1 - 2e-4) * 0.9999
    R3 = dict(R, sell_edge=se); d3 = frame("ETH", R3).set_index("t")
    assert d3.loc[s, "y_fillb10_2"] == 1 and d3.loc[s, "y_fillb10_0"] == 1 and d3.loc[s - 5, "y_fillb10_2"] == 0   # s−5 창은 (995,1005] — 1007 체결 밖
    print("targets selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", nargs=2, metavar=("COIN", "YYYY-MM-DD"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--frame", metavar="COIN")
    ap.add_argument("--retape", nargs=2, metavar=("COIN", "YYYY-MM-DD"))
    a = ap.parse_args()
    if a.selftest:
        selftest(); targets_selftest()
    elif a.retape:
        retape(a.retape[0].upper(), a.retape[1])
    elif a.frame:
        c = a.frame.upper(); red = OUT / c / "_reduced.npz"
        fresh = red.exists() and red.stat().st_mtime > max(f.stat().st_mtime for f in (OUT / c).glob("20*.npz"))
        R = dict(np.load(red)) if fresh else reduce_coin(c)                 # 날짜 파일이 그 뒤로 안 바뀌었으면 다시 안 묶는다
        df = frame(c, R)
        df.to_parquet(OUT / f"frame_{a.frame.upper()}.parquet")
        print(a.frame, len(df), df.split.value_counts().to_dict())
    elif a.day:
        print(build_day(a.day[0].upper(), a.day[1]))
