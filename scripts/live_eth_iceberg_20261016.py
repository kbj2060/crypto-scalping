#!/usr/bin/env python3
"""ETH 칸별 아이스버그 실시간 탐지기 (2026-10-16, 사용자 «같이 해서 시안 만들어줘» → «배포해줘», 시안 ①).

정의 = 전진 사전등록 docs/prereg/eth_dom_iceberg_forward_frozen_20261016.json · research_eth_dom_levels_absorption_20261016.events() 그대로:
  depthdiff 시간 파일을 **이어 읽어** 30초 창마다 창 시작 mid 칸 ±30칸($0.1) 잔량(시작·최대·끝)·최우선 극값, 체결 = hot 테이프 price_bin.
  매수 흡수 칸 P: 최우선 매수 2칸 안 · 그 칸 테이커 매도 ≥ 직전 2880창(24h) «창별 최대 칸 매도» 95분위 · 버팀(최우선 매수 ≥ P−0.05)
  · 다시 채움(체결 ≥ 시작 잔량 & 끝 ≥ 시작 × 0.5) → 아이스버그 = 체결 ≥ 창 안 최대 표시 잔량 × 2. 매도 쪽은 거울.
판정은 연구 스크립트 --judge(오프라인 재생)가 한다 -- 여기 건수는 화면 «검증 중 n/150» 용이라 재시작 공백만큼 적을 수 있다. 신호·알림 아님.
출력: data/live/iceberg/eth_events.jsonl(아이스버그만, 한 줄 한 건) · eth_state.json(2초마다 -- ops_watchdog 가 신선도를 본다).
주문 없음 · 거래소 호출 없음(파일·hot SQLite 만 읽는다).
사용: python -u scripts/live_eth_iceberg_20261016.py | --parity <시간 파일>(연구 replay_levels 와 창 대조) | --selftest
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.data_store import read_rows  # noqa: E402

DD = ROOT / "data/live/orderflow/depthdiff/ETHUSDT"
TAPE = ROOT / "data/hot/binance_tape.sqlite"
OUT = ROOT / "data/live/iceberg"
TICK, K, WIN, NQ, NMIN = 100, 30, 30, 2880, 1000
IDX_MAX = 1_000_000   # ponytail: 연구는 10M($100k) -- 메모리 16MB 로 줄임. ETH 가 $10k 를 넘으면 올린다(넘는 가격 변경은 버려진다)
FWD0 = 1792108800     # 2026-10-16 00:00 UTC = 사전등록 전진 시작
SETTLE = 5            # 창 끝 뒤 테이프가 다 들어올 때까지 기다리는 초


def buckets(arr, base):
    """반올림 칸 b = round(가격×10) = 눈금 [10b−5, 10b+4] → base−K..base+K 칸 합 (연구 buckets 와 같다)."""
    lo = (base - K) * 10 - 5
    return arr[lo:lo + (2 * K + 1) * 10].reshape(2 * K + 1, 10).sum(1)


class Book:
    """연구 replay_levels 의 루프를 줄 단위로(같은 U/u/pu 동기화·끊김 규칙·best()·30초 창). 끝난 창은 self.done 에 쌓인다."""

    def __init__(self, snap: dict):
        self.bid, self.ask = np.zeros(IDX_MAX), np.zeros(IDX_MAX)
        for p, q in snap["bids"]:
            self.bid[int(round(float(p) * TICK))] = float(q)
        for p, q in snap["asks"]:
            self.ask[int(round(float(p) * TICK))] = float(q)
        self.last_u, self.synced, self.valid, self.cur, self.W = snap["lastUpdateId"], False, 1, None, None
        self.bb, self.ba = int(np.flatnonzero(self.bid)[-1]), int(np.flatnonzero(self.ask)[0])
        self.done: list[dict] = []

    def _best(self):
        w = max(int(self.bb * 0.02), 100)
        nz = np.flatnonzero(self.bid[max(self.bb - w, 0):self.bb + w + 1]); self.bb = max(self.bb - w, 0) + nz[-1] if len(nz) else self.bb
        nz = np.flatnonzero(self.ask[max(self.ba - w, 0):self.ba + w + 1]); self.ba = max(self.ba - w, 0) + nz[0] if len(nz) else self.ba

    def _tick(self, sec: int):
        W, w = self.W, sec // WIN
        if W is not None and W["w"] != w:
            self.done.append(W); W = None
        if W is None:
            base = int(round((self.bb + self.ba) / 2 / 10))
            b0, a0 = buckets(self.bid, base), buckets(self.ask, base)
            W = {"w": w, "base": base, "valid": self.valid, "bS": b0, "bMin": b0.copy(), "bMax": b0.copy(), "bE": b0,
                 "aS": a0, "aMin": a0.copy(), "aMax": a0.copy(), "aE": a0, "bbMin": self.bb, "baMax": self.ba, "bb0": self.bb, "ba0": self.ba}
        b1, a1 = buckets(self.bid, W["base"]), buckets(self.ask, W["base"])
        np.minimum(W["bMin"], b1, out=W["bMin"]); np.maximum(W["bMax"], b1, out=W["bMax"])
        np.minimum(W["aMin"], a1, out=W["aMin"]); np.maximum(W["aMax"], a1, out=W["aMax"])
        W["bE"], W["aE"] = b1, a1
        W["bbMin"] = min(W["bbMin"], self.bb); W["baMax"] = max(W["baMax"], self.ba); W["valid"] = min(W["valid"], self.valid)
        self.W = W

    def feed(self, line: bytes):
        try:
            m = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            self.valid = 0; return
        if m.get("e") != "depthUpdate":
            return
        if not self.synced:
            if m["u"] < self.last_u:
                return
            if not (m["U"] <= self.last_u <= m["u"]):
                self.valid = 0
            self.synced = True
        elif m["pu"] != self.last_u:
            self.valid = 0
        self.last_u = m["u"]; sec = m["E"] // 1000
        if self.cur is None:
            self.cur = sec
        while sec > self.cur:
            self._tick(self.cur); self.cur += 1
        for p, q in m["b"]:
            i = int(round(float(p) * TICK))
            if i < IDX_MAX:
                self.bid[i] = float(q)
        for p, q in m["a"]:
            i = int(round(float(p) * TICK))
            if i < IDX_MAX:
                self.ask[i] = float(q)
        self._best()

    def close(self):
        """파일 끝(연구: 마지막 초 tick + 남은 창)."""
        if self.cur is not None:
            self._tick(self.cur)
        if self.W is not None:
            self.done.append(self.W); self.W = None


def window_events(W: dict, SB: np.ndarray, SS: np.ndarray, qs: float, qb: float) -> list[dict]:
    """연구 events() 의 한 창 몫 -- 아이스버그만. 산술 순서·float32 는 연구와 같게(경계 동률이 갈리지 않게)."""
    if W["valid"] != 1:
        return []
    f = lambda k: np.asarray(W[k], np.float32)
    bbk = int(np.round(W["bb0"] / TICK * 10)) - W["base"] + K; bak = int(np.round(W["ba0"] / TICK * 10)) - W["base"] + K
    out = []
    for side in (1, -1):
        k0 = bbk if side == 1 else bak
        ks = [k for k in (range(k0 - 2, k0 + 1) if side == 1 else range(k0, k0 + 3)) if 0 <= k <= 2 * K]
        if not ks:
            continue
        ex = SS if side == 1 else SB
        k = max(ks, key=lambda c: ex[c]); vol = float(ex[k]); thr = qs if side == 1 else qb
        if not (np.isfinite(thr) and thr > 0 and vol >= thr):
            continue
        px = (W["base"] + k - K) * 0.1
        start = float(f("bS" if side == 1 else "aS")[k]); end = float(f("bE" if side == 1 else "aE")[k]); mx = float(f("bMax" if side == 1 else "aMax")[k])
        held = W["bbMin"] / TICK >= px - 0.05 if side == 1 else W["baMax"] / TICK <= px + 0.05
        if held and start > 0 and vol >= start and end >= 0.5 * start and vol >= 2 * mx:
            out.append({"t_end": int((W["w"] + 1) * WIN), "side": side, "px": round(px, 2), "vol": round(vol, 3),
                        "max": round(mx, 3), "start": round(start, 3), "end": round(end, 3)})
    return out


def tape_cells(lo: int, hi: int) -> dict[int, list[tuple[int, float, float]]]:
    """[lo, hi) 초의 창별 (price_bin, 매수, 매도). SQLite 정수 나눗셈 = 연구 ts_sec // WIN."""
    got: dict[int, list] = {}
    for w, b, bq, sq in read_rows(TAPE, "SELECT ts_sec / ?, price_bin, sum(buy_qty), sum(sell_qty) FROM trade_tape_1s "
                                        "WHERE symbol = 'ethusdt' AND ts_sec >= ? AND ts_sec < ? GROUP BY 1, 2", (WIN, lo, hi)):
        got.setdefault(int(w), []).append((int(b), float(bq or 0), float(sq or 0)))
    return got


def warm(hs: collections.deque, hb: collections.deque, w0: int):
    """기준선(직전 2880창 창별 최대 칸 체결)을 hot 테이프로 채운다. ponytail: 연구는 창 시작 mid ±30칸 안 최대, 여기는 그 창 전체 칸 최대
    (30초에 $3 넘게 움직인 창만 다르다) -- 2880창(24h) 돌고 나면 전부 정확한 값으로 바뀐다."""
    cells = tape_cells((w0 - NQ) * WIN, w0 * WIN)
    for w in range(w0 - NQ, w0):
        c = cells.get(w, ())
        hs.append(max((s for _, _, s in c), default=0.0)); hb.append(max((b for _, b, _ in c), default=0.0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev_path, st_path = OUT / "eth_events.jsonl", OUT / "eth_state.json"
    seen, n = set(), 0
    if ev_path.exists():
        for ln in ev_path.read_text().splitlines():
            e = json.loads(ln); seen.add((e["t_end"], e["side"])); n += e["t_end"] >= FWD0
    hs, hb = collections.deque(maxlen=NQ), collections.deque(maxlen=NQ)
    book = fh = path = None; buf, pend, last_w = b"", [], None
    print(f"시작 · 기존 아이스버그 {n}건(전진)", flush=True)
    while True:
        cur = DD / f"{time.strftime('%Y-%m-%dT%H', time.gmtime())}.jsonl"
        if cur != path and cur.exists():
            if fh:                                # 새 시 파일이 생겼다 = 수집기가 옛 파일을 닫았다 -- 끝까지 읽고(지워져도 열린 핸들은 읽힌다) 넘어간다
                for ln in (buf + fh.read()).split(b"\n"):
                    if ln and book:
                        book.feed(ln)
                fh.close()
            if book:
                book.close(); pend += book.done
            fh, path, buf, book = open(cur, "rb"), cur, b"", None
            print(f"파일 {cur.name}", flush=True)
        if fh:
            buf += fh.read()
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                if book is None:                  # 첫 줄 = 시각 스냅샷(없으면 그 시는 연구도 버린다 -- 다음 시까지 기다린다)
                    first = json.loads(ln); snap = first.get("_snapshot") if isinstance(first, dict) else None
                    if not snap:
                        print(f"{path.name} 스냅샷 없음 -- 이 시 건너뜀", flush=True); fh.close(); fh = None; break
                    book = Book(snap)
                else:
                    book.feed(ln)
            if book:
                pend += book.done; book.done = []
        pend = [W for W in pend if last_w is None or W["w"] > last_w]
        ready = [W for W in pend if (W["w"] + 1) * WIN + SETTLE <= time.time()]
        if ready:
            if last_w is None:
                warm(hs, hb, ready[0]["w"])
            cells = tape_cells(ready[0]["w"] * WIN, (ready[-1]["w"] + 1) * WIN)
            for W in ready:
                SB = np.zeros(2 * K + 1, np.float32); SS = np.zeros_like(SB)
                for b, bq, sq in cells.get(W["w"], ()):
                    c = b - W["base"] + K
                    if 0 <= c <= 2 * K:
                        SB[c] += bq; SS[c] += sq
                qs = float(np.quantile(hs, .95)) if len(hs) >= NMIN else float("nan")
                qb = float(np.quantile(hb, .95)) if len(hb) >= NMIN else float("nan")
                for e in window_events(W, SB, SS, qs, qb):
                    if (e["t_end"], e["side"]) not in seen:
                        with open(ev_path, "a") as fo:
                            fo.write(json.dumps(e) + "\n")
                        seen.add((e["t_end"], e["side"])); n += e["t_end"] >= FWD0
                        print("아이스버그", e, flush=True)
                hs.append(float(SS.max())); hb.append(float(SB.max())); last_w = W["w"]
            pend = pend[len(ready):]
        tmp = st_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"updated_at": time.time(), "file": path.name if path else None, "valid": int(book.valid) if book else 0,
                                   "last_window_end": (last_w + 1) * WIN if last_w is not None else None, "n_forward": n, "target": 150, "fwd0": FWD0}))
        tmp.replace(st_path)
        time.sleep(2)


def parity(path: str):
    """연구 replay_levels 와 같은 파일에서 창이 같은지(dev 백업 미러로 돌린다)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import research_eth_dom_levels_absorption_20261016 as R
    ref = R.replay_levels(path)
    with (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")) as f:
        bk = Book(json.loads(f.readline())["_snapshot"])
        for ln in f:
            bk.feed(ln)
    bk.close()
    assert len(bk.done) == len(ref["w"]), (len(bk.done), len(ref["w"]))
    for k in ("w", "base", "valid"):
        assert np.array_equal(np.array([W[k] for W in bk.done]), ref[k]), k
    for k in ("bb0", "ba0", "bbMin", "baMax"):
        assert np.array_equal(np.array([W[k] for W in bk.done]) / TICK, ref[k]), k
    for k in ("bS", "bMin", "bMax", "bE", "aS", "aMin", "aMax", "aE"):
        assert np.array_equal(np.stack([W[k] for W in bk.done]).astype(np.float32), ref[k]), k
    print(f"parity ok · 창 {len(ref['w'])} · 유효 {int(ref['valid'].sum())}")


def selftest():
    assert time.strftime("%Y-%m-%dT%H", time.gmtime(FWD0)) == "2026-10-16T00"
    a = np.zeros(10_000); a[995:1005] = 1.0
    assert buckets(a, 100)[K] == 10
    z = np.zeros(2 * K + 1, np.float32)
    W = {"w": 0, "base": 25000, "valid": 1, "bb0": 250000, "ba0": 250001, "bbMin": 250000, "baMax": 250001,
         "bS": z + 5, "bE": z + 4, "bMax": z + 6, "aS": z + 5, "aE": z + 5, "aMax": z + 6}
    SS = z.copy(); SS[K] = 13                   # 매수 최우선 칸(2500.0)에 매도 13 ≥ 최대 표시 6 × 2 · 버팀 · 다시 채움(끝 4 ≥ 2.5)
    e = window_events(W, z, SS, 10.0, 10.0)
    assert len(e) == 1 and e[0]["side"] == 1 and e[0]["px"] == 2500.0 and e[0]["t_end"] == 30, e
    assert window_events(W, z, SS, 14.0, 10.0) == []                       # 기준선 못 넘음
    assert window_events({**W, "bbMin": 249990}, z, SS, 10.0, 10.0) == []  # 뚫림(최우선 매수 2499.90 < 2499.95)
    SS[K] = 11; assert window_events(W, z, SS, 10.0, 10.0) == []           # 11 < 6 × 2 = 흡수지만 아이스버그 아님
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--parity" in sys.argv:
        parity(sys.argv[sys.argv.index("--parity") + 1])
    else:
        main()
