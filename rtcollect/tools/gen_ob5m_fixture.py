#!/usr/bin/env python3
"""ob5m(5분 호가 요약) · bookdepth(호가 깊이 10열) 일치 시험의 입력·정답지 (2026-11-21).

입력 = 합성 depthdiff 시 파일 14개(`tests/fixtures/ob5m/ETHUSDT/<UTC 시>.jsonl[.gz]` -- 첫 줄 스냅숏 + 15초 간격 depthUpdate, 벽이 섰다 빠지는 사건 ·
끊김 한 번) + hot 테이프 행(사건 체결 몫). 원본 `live_ob5m_20261026.main()`·`live_bookdepth_feat_20261028.main()` 을 **그대로** 돌린다 --
바뀌는 것은 바깥뿐: 경로(DD·TAPE·DB) · 시계(time.time = 처음 한 번은 «마지막 시 + 26시간»(시작 시 계산) 뒤로는 «방금 본 거래소 시각 + 10초»,
time.sleep = 파일이 끝나면 멈춤). 🔴numpy 는 서버와 같은 쪽으로: NPY_DISABLE_CPU_FEATURES=X86_V4(서버 CPU 는 AVX-512 없음 -- 개발기 numpy 2.4 의
AVX-512 np.log 는 libm 과 1ulp 다르다. numpy 2.4 는 묶음 이름 X86_V4 만 받는다 -- AVX512F 같은 옛 이름은 조용히 무시된다).
🔴bookdepth `cum` 의 `arr @ 가격` 은 BLAS 내적(개발기 MKL · 서버 OpenBLAS)이라 끝자리가 기계마다 다르다 -- 그 열(dm·dp)과 거기서 나온 열은 상대 1e-12 로 본다.

    NPY_DISABLE_CPU_FEATURES=X86_V4 python3 rtcollect/tools/gen_ob5m_fixture.py
    NPY_DISABLE_CPU_FEATURES=X86_V4 python3 rtcollect/tools/gen_ob5m_fixture.py --check-readers
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_venue_fixture as gv  # noqa: E402 -- 소켓 막기

assert "X86_V4" in os.environ.get("NPY_DISABLE_CPU_FEATURES", ""), "NPY_DISABLE_CPU_FEATURES=X86_V4 로 돌린다(서버와 같은 np.log)"
FIX, ROOT = gv.FIX, gv.ROOT
SEED = 20261127
H0 = 491_000            # 시(시각 번호) -- 2026-01-05T08 언저리(합성)
NH = 14
DDIR = FIX / "ob5m" / "ETHUSDT"


class Gen:
    def __init__(self, rng):
        self.rng, self.mid, self.u = rng, 2500.0, 7_000_000
        self.bid: dict[str, float] = {}
        self.ask: dict[str, float] = {}
        self.wall = None            # (side, px 글자, 사라질 ms, 체결: eat|none)
        self.tape: list = []        # (ts_sec, price_bin, buy, sell)
        for k in range(1, 261):     # 0.5달러 간격 ±130달러(±5% 를 덮는다) + 가까운 0.01 칸
            self.bid[f"{2500 - 0.5 * k:.2f}"] = round(rng.uniform(1, 30), 3)
            self.ask[f"{2500 + 0.5 * (k - 1) + 0.01:.2f}"] = round(rng.uniform(1, 30), 3)

    def snap(self, t_ms):
        bids = sorted(((p, q) for p, q in self.bid.items() if q > 0), key=lambda x: -float(x[0]))
        asks = sorted(((p, q) for p, q in self.ask.items() if q > 0), key=lambda x: float(x[0]))
        return {"lastUpdateId": self.u + 1, "E": t_ms + 3, "T": t_ms,
                "bids": [[p, f"{q:.3f}"] for p, q in bids], "asks": [[p, f"{q:.3f}"] for p, q in asks]}

    def frame(self, t_ms):
        r = self.rng
        self.mid = round(self.mid + r.gauss(0, 0.15), 2)
        b, a = [], []
        for _ in range(r.randint(1, 4)):
            off = r.choice((r.randint(1, 60) * 0.01, r.randint(1, 260) * 0.5))
            q = 0.0 if r.random() < 0.15 else round(r.lognormvariate(1, 1), 3)
            if r.random() < 0.5:
                b.append([f"{self.mid - off:.2f}", q])
            else:
                a.append([f"{self.mid + off:.2f}", q])
        # 벽: 10분마다 가까운 매수/매도 칸에 큰 물량 → 60초 서 있다 → 한 번에 0
        if self.wall is None and t_ms % 600_000 < 10_000:
            side = r.choice((0, 1))
            px = f"{self.mid - 0.03:.2f}" if side == 0 else f"{self.mid + 0.03:.2f}"
            kind = r.choice(("eat", "none", "none_far"))
            if kind == "none_far":
                px = f"{self.mid * (1 - 0.0015):.2f}" if side == 0 else f"{self.mid * (1 + 0.0015):.2f}"
            self.wall = (side, px, t_ms + 61_000, kind)
        if self.wall:
            side, px, end, kind = self.wall
            if t_ms >= end:
                (b if side == 0 else a).append([px, 0.0])
                if kind == "eat":    # 매수벽을 먹는 것 = 테이커 매도 · 칸 = round(가격 / 0.1)
                    q = 900.0
                    self.tape.append((t_ms // 1000, int(round(float(px) / 0.1)), 0.0 if side == 0 else q, q if side == 0 else 0.0))
                self.wall = None
            else:
                (b if side == 0 else a).append([px, 900.0])
        b += [[p, 0.0] for p in list(self.bid) if float(p) >= self.mid]
        a += [[p, 0.0] for p in list(self.ask) if float(p) <= self.mid]
        for book, rows in ((self.bid, b), (self.ask, a)):
            for p, q in rows:
                if q == 0.0:
                    book.pop(p, None)
                else:
                    book[p] = q
        u0 = self.u + 1
        self.u += r.randint(1, 30)
        return {"e": "depthUpdate", "E": t_ms + 2, "T": t_ms, "s": "ETHUSDT", "U": u0, "u": self.u, "pu": u0 - 1,
                "b": [[p, f"{q:.3f}"] for p, q in b], "a": [[p, f"{q:.3f}"] for p, q in a]}


def gen(rng) -> None:
    g = Gen(rng)
    DDIR.mkdir(parents=True, exist_ok=True)
    for p in DDIR.iterdir():
        p.unlink()
    for h in range(H0, H0 + NH):
        name = time.strftime("%Y-%m-%dT%H", time.gmtime(h * 3600))
        lines = [json.dumps({"_snapshot": g.snap(h * 3_600_000), "at_ms": h * 3_600_000 + 50}, separators=(",", ":"))]
        t = h * 3_600_000 + 1_000 + rng.randint(0, 900)
        while t < (h + 1) * 3_600_000:
            f = g.frame(t)
            if h == H0 and len(lines) == 100:           # 끊김 하나(사슬 재시작 → 다음 시 파일 스냅숏 -- bookdepth 의 12시간 warm 이 끝에 풀리게 첫 시에)
                lines.append(json.dumps({"_gap": True, "expected_pu": f["pu"], "got_pu": f["pu"] + 5}, separators=(",", ":")))
            lines.append(json.dumps(f, separators=(",", ":")))
            t += 15_000 + rng.randint(-300, 300)
        body = ("\n".join(lines) + "\n").encode()
        if h == H0 + NH - 1:
            (DDIR / f"{name}.jsonl").write_bytes(body)                         # 마지막 시 = 압축 전(.jsonl)
        else:
            (DDIR / f"{name}.jsonl.gz").write_bytes(gzip.compress(body, compresslevel=9, mtime=0))
    (FIX / "ob5m_tape.json").write_text(json.dumps({"seed": SEED, "h0": H0, "nh": NH, "tape": g.tape}, separators=(",", ":")))


def make_tape(path: Path, rows) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE trade_tape_1s(symbol TEXT, ts_sec INTEGER, price_bin INTEGER, buy_qty REAL, sell_qty REAL, buy_n INTEGER, sell_n INTEGER, "
                "buy_max REAL, sell_max REAL, retail_buy_qty REAL, retail_sell_qty REAL, whale_buy_qty REAL, whale_sell_qty REAL, whale_buy_n INTEGER, "
                "whale_sell_n INTEGER, retail_buy_n INTEGER, retail_sell_n INTEGER, order_buy_n INTEGER, order_sell_n INTEGER, PRIMARY KEY (symbol, ts_sec, price_bin)) WITHOUT ROWID")
    con.execute("CREATE TABLE gaps(symbol TEXT, from_ms INTEGER, to_ms INTEGER, reason TEXT)")
    # 테이프는 1초마다 한 줄 이상 있어야 «빈 구간» 이 아니다 -- 사건 없는 초는 작은 체결 하나
    have = {(r[0], r[1]) for r in rows}
    allrows = list(rows) + [(s, 25000, 0.01, 0.0) for s in range(H0 * 3600, (H0 + NH) * 3600) if (s, 25000) not in have]
    con.executemany("INSERT INTO trade_tape_1s(symbol, ts_sec, price_bin, buy_qty, sell_qty) VALUES ('ethusdt', ?, ?, ?, ?)", allrows)
    con.execute("INSERT INTO gaps VALUES ('ethusdt', ?, ?, 'ws_reconnect')", ((H0 + 9) * 3_600_000 + 100_000, (H0 + 9) * 3_600_000 + 130_000))
    con.commit()
    con.close()


class _End(BaseException):
    pass


def python_reference(d: Path, which: str) -> None:
    """원본 main() 을 그대로(무한 루프) -- 마지막 시 파일 끝에서 Source 가 잠들려 하면 멈춘다."""
    meta = json.loads((FIX / "ob5m_tape.json").read_text())
    ob = __import__("scripts.live_ob5m_20261026", fromlist=["x"])
    bf = __import__("scripts.live_bookdepth_feat_20261028", fromlist=["x"])
    ob.DD, ob.TAPE = DDIR, d / "binance_tape.sqlite"
    ob.DB, bf.DB = d / "ob5m.sqlite", d / "bookdepth.sqlite"
    # 🔴db_open(path=DB) 의 기본값은 import 때 굳는다 -- 바꾸지 않으면 저장소 data/live 에 쓴다(2026-11-21 처음 그랬다 · 지움)
    ob.db_open.__defaults__, bf.db_open.__defaults__ = (ob.DB,), (bf.DB,)
    real = ROOT / "data" / "live"
    before = real.exists()
    d.mkdir(parents=True, exist_ok=True)
    if not ob.TAPE.exists():
        make_tape(ob.TAPE, [tuple(r) for r in meta["tape"]])
    st = {"first": True, "t": 0}

    def now():
        if st["first"]:
            st["first"] = False
            return (meta["h0"] + 26) * 3600 + 5.0      # 시작 시 h0 = 이 시 − 26 = 첫 파일
        return st["t"] / 1000 + 10.0

    def sleep(_s):
        raise _End()

    fake = gv._ns(time, time=now, sleep=sleep)
    ob.time = bf.time = fake
    if which == "ob5m":
        real_sec = ob.Bars.sec

        def sec(self, row):
            st["t"] = row[0] + 1
            return real_sec(self, row)
        ob.Bars.sec = sec
        try:
            with gv._quiet():
                ob.main()
        except _End:
            pass
        finally:
            ob.Bars.sec = real_sec
        assert real.exists() == before, "저장소 data/live 에 썼다"
    else:
        real_on = bf.Bars.on_sec

        def on_sec(self, t, *a):
            st["t"] = t
            return real_on(self, t, *a)
        bf.Bars.on_sec = on_sec
        try:
            with gv._quiet():
                bf.main()
        except _End:
            pass
        finally:
            bf.Bars.on_sec = real_on
        assert real.exists() == before, "저장소 data/live 에 썼다"


def dump(d: Path) -> dict:
    out = {}
    c = sqlite3.connect(d / "ob5m.sqlite")
    cols = [r[1] for r in c.execute("PRAGMA table_info(bars)")]
    out["ob5m_cols"] = cols
    out["ob5m_bars"] = [list(r) for r in c.execute("SELECT * FROM bars ORDER BY t")]
    out["ob5m_marks"] = [list(r) for r in c.execute("SELECT * FROM marks ORDER BY t, side, px")]
    out["ob5m_live"] = [list(r) for r in c.execute("SELECT k, bar, marks FROM live")]     # upd = 벽시계
    out["ob5m_schema"] = [r[0] for r in c.execute("SELECT sql FROM sqlite_master ORDER BY name")]
    c.close()
    c = sqlite3.connect(d / "bookdepth.sqlite")
    out["bd_cols"] = [r[1] for r in c.execute("PRAGMA table_info(bars)")]
    out["bd_bars"] = [list(r[:-1]) for r in c.execute("SELECT * FROM bars ORDER BY t")]       # written_ms = 벽시계
    out["bd_schema"] = [r[0] for r in c.execute("SELECT sql FROM sqlite_master ORDER BY name")]
    c.close()
    return out


def cargo_replay(out: Path) -> None:
    env = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:{os.environ.get('PATH', '')}")
    r = subprocess.run(["bash", "-c", 'ulimit -v 4000000; exec timeout 300 "$@"', "_", "cargo", "run", "-q", "-j", "2", "--release",
                        "--example", "replay_ob5m", "--", str(FIX), str(out)], cwd=ROOT / "rtcollect", env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"cargo replay_ob5m 실패:\n{r.stderr}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-readers", action="store_true")
    a = ap.parse_args()
    if a.check_readers:
        return check_readers()
    gen(random.Random(SEED))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        python_reference(td, "ob5m")
        python_reference(td, "bookdepth")
        golden = dump(td)
    (FIX / "ob5m_golden.json").write_text(json.dumps(golden, separators=(",", ":")))
    sz = sum(p.stat().st_size for p in DDIR.iterdir()) + (FIX / "ob5m_tape.json").stat().st_size + (FIX / "ob5m_golden.json").stat().st_size
    nonwarm = sum(1 for r in golden["ob5m_bars"] if r[golden["ob5m_cols"].index("warm")] == 0)
    bdw = sum(1 for r in golden["bd_bars"] if r[golden["bd_cols"].index("warm")] == 0)
    print(f"ob5m 봉 {len(golden['ob5m_bars'])}(온전 {nonwarm}) · 표식 {len(golden['ob5m_marks'])} · bookdepth 봉 {len(golden['bd_bars'])}(온전 {bdw}) · 합 {sz // 1024}KB")


def api_ob5m(db: Path, now_s: float) -> dict:
    """dashboard/server.py:2992-3028 `api_ob5m.produce` 와 같은 읽기(서버 모듈은 import 하지 않는다 -- 대시보드를 띄우지 않는다). now_s = 그 시각의 벽시계."""
    from scripts.data_store import read_rows
    cut = int((now_s - 7 * 86400) * 1000)
    rows = read_rows(db, "SELECT t, warm, imb50, eat_n_a, wall_px_b, wall_usd_b, wall_px_a, wall_usd_a FROM bars WHERE t >= ? ORDER BY t", (cut,))
    lv = read_rows(db, "SELECT upd, bar, marks FROM live WHERE k = 0", ())          # 30초 신선도 거르기는 벽시계라 뺀다
    lb, lmarks = (json.loads(lv[0][1]), json.loads(lv[0][2])) if lv else (None, [])
    live_t = None
    if lb and (not rows or lb["t"] > rows[-1][0]):
        rows.append((lb["t"], lb["warm"], lb["imb50"], None, lb["wall_px_b"], lb["wall_usd_b"], lb["wall_px_a"], lb["wall_usd_a"]))
        live_t = lb["t"]
    bars, walls, run = [], [], {}
    for t, warm, imb, ev, pb, ub, pa, ua in rows:
        bars.append({"t": t, "tilt_pct": None if imb is None else round(imb * 100, 2), "book_ok": not warm and imb is not None,
                     "events_known": ev is not None, **({"live": True} if t == live_t else {})})
        for side, px, usd in (("bid", pb, ub), ("ask", pa, ua)):
            r = run.get(side)
            if r and px is not None and r["px"] == px and r["t_end"] == t:
                r["t_end"], r["usd_max"] = t + 300_000, max(r["usd_max"], usd or 0.0)
                continue
            if r:
                walls.append(run.pop(side))
            if px is not None:
                run[side] = {"side": side, "px": px, "t_start": t, "t_end": t + 300_000, "usd_max": usd or 0.0}
    walls += run.values()
    marks = [{"t": t, "bar_t": bt, "side": s, "kind": k, "px": px, "usd": round(u or 0), "dist_bp": round(d, 2)}
             for t, bt, s, k, px, u, d in read_rows(db, "SELECT t, bar_t, side, kind, px, usd, dist_bp FROM marks WHERE bar_t >= ? ORDER BY t", (cut,))]
    known = {b["t"] for b in bars if b["events_known"]} | {m["bar_t"] for m in marks}
    marks += [{"t": m["t"], "bar_t": m["bar_t"], "side": m["side"], "kind": m["kind"], "px": m["px"], "usd": round(m["usd"]),
               "dist_bp": round(m["dist_bp"], 2), "live": True} for m in lmarks if m["bar_t"] not in known]
    return {"bars": bars, "walls": walls, "marks": marks}


def check_readers() -> None:
    """읽는 쪽: /api/ob5m 의 읽기(위 api_ob5m) · 방향 예측 dir_short 의 book_row(호가 깊이 10열 → float32, fwd_dirshort 입력)."""
    meta = json.loads((FIX / "ob5m_tape.json").read_text())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        python_reference(td / "py", "ob5m")
        python_reference(td / "py", "bookdepth")
        cargo_replay(td / "rs")
        now_s = (meta["h0"] + meta["nh"]) * 3600
        a, b = api_ob5m(td / "py" / "ob5m.sqlite", now_s), api_ob5m(td / "rs" / "ob5m.sqlite", now_s)
        for k in a:
            print(f"  {'같음' if a[k] == b[k] else '다름'}  api_ob5m.{k}: {len(a[k])}")
        assert a == b
        ds = __import__("scripts.live_eth_dir_short_20261028", fromlist=["x"])
        import numpy as np
        ca, cb = sqlite3.connect(td / "py" / "bookdepth.sqlite"), sqlite3.connect(td / "rs" / "bookdepth.sqlite")
        ts = [r[0] for r in ca.execute("SELECT t FROM bars ORDER BY t")]
        same = st = 0
        for t in ts + [ts[-1] + 300_000]:
            (x, sx), (y, sy) = ds.book_row(ca, t), ds.book_row(cb, t)
            same += sx == sy and (x is None and y is None or (x is not None and y is not None and np.array_equal(x.view(np.uint32), y.view(np.uint32))))
            st += sx == "ok"
        print(f"  {'같음' if same == len(ts) + 1 else '다름'}  dir_short.book_row: {same}/{len(ts) + 1}봉 (상태 ok {st} · float32 비트)")
        assert same == len(ts) + 1
        d1, d2 = dump(td / "py"), dump(td / "rs")
        for k in ("ob5m_bars", "ob5m_marks", "ob5m_live", "ob5m_schema"):
            assert d1[k] == d2[k], k
    print("check-readers OK")


if __name__ == "__main__":
    main()
