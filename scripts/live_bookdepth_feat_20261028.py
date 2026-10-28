#!/usr/bin/env python3
"""ETH 호가 깊이 10열 실시간 워커 (2026-10-28, 사용자 «호가 깊이 10열을 서버에서 실시간으로 계산하는 워커부터 만들어줘»).

정의 = 방향 예측 «book» 블록(scripts/diag_rl_dir_liqfeat_20261028.py feats) — 학습 원천 binance.vision bookDepth 와 같은 값을 서버 diff 전체북에서 만든다.
  패리티(docs/prereg/eth_direction_accuracy_push_20261028.md §7-7): live_ob5m.replay 재구성 ±1·2·5% 누적 수량 / bookDepth 비 중앙 1.000 ·
  실시간 대체 예측 상관 .998(부호 99%) — 다른 시드로 다시 학습한 차이(.97)보다 작다.
  스냅숏 = 매 분 :01·:31 초(UTC) — bookDepth 는 3년 내내 :00~03·:30~33 초. 그 초에 diff 가 없으면 그 반분의 첫 초.
  봉 τ 값 = 봉 끝 1초 전(τ+5분−1초)까지 마지막 스냅숏 · d_age = 그 나이 + 1초(liqlab 규약) → 평소 29초.
  10열: dimb1·2·5 = (매수−매도)/(매수+매도) 수량 · dimbk_d1h = 12봉 전 dimbk 와 차 · depth1_z = log(±1% 매수+매도) − 직전 288봉(자기 포함, 최소 144) 평균 ·
        bid_near = 매수 ±1%/±5% · ask_near = 매도 ±1%/±5% · d_age.
입력(읽기만): data/live/orderflow/depthdiff/ETHUSDT/<UTC시>.jsonl(.gz) — live_ob5m 의 Source·replay 를 그대로 쓴다(끊김·재시작 규칙 같음).
출력: data/live/bookdepth/ethusdt.sqlite · bars(t = 봉 시작 ms, dm1..dm5·dp1..dp5 수량, 10열, chain_age_s, warm).
  warm = 사슬 (재)시작 뒤 WARM_S 안 — ±0.43% 밖 먼 칸은 diff 가 그 가격을 건드려야 차므로 ±2·5% 가 덜 찼다. 피쳐로 쓰지 말 것.
  온전한(warm 아님) 봉은 warm 봉으로 덮지 않는다(재시작 재생이 더 나쁜 값으로 덮는 것을 막음 — live_ob5m 리뷰 10-26 과 같은 규칙).
ponytail: 재시작하면 마지막 온전한 봉 13시간 앞 파일부터 다시 재생한다(WARM_S 12h + 여유) — 재생 중엔 CPU 를 더 쓴다. 오래 멈췄으면 최근 26시간만.
과거 재생 대조(10-03~04, 차가운 시작): warm 아닌·24h 찬 287봉에서 학습 원천 대비 10열 상관 .964~.988 · d_age 전부 29초.
주문 없음 · 거래소 호출 없음.
사용: python -u scripts/live_bookdepth_feat_20261028.py | --replay 2026-10-03T00 2026-10-05T00 OUT.sqlite | --selftest
"""
from __future__ import annotations

import argparse
import gzip
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.live_ob5m_20261026 import DD, TICK, Source, replay  # noqa: E402

DB = ROOT / "data/live/bookdepth/ethusdt.sqlite"
KS, BAR_S, WARM_S, HIST = (1, 2, 5), 300, 12 * 3600, 300   # WARM_S: 10-03 차가운 재생 — ±5% 매도 비 1h .89 · 6~12h .976 · 12h+ .995(±1·2% 는 바로 1.00)
RAW = [f"dm{k}" for k in KS] + [f"dp{k}" for k in KS]                     # dm = 매수(−k%) · dp = 매도(+k%) 누적 명목 $
FEAT = ["dimb1", "dimb2", "dimb5", "dimb1_d1h", "dimb2_d1h", "dimb5_d1h", "depth1_z", "bid_near", "ask_near", "d_age"]
COLS = RAW + FEAT + ["chain_age_s", "warm"]


def cum(bid, ask, bb, ba):
    """중간가 기준 ±k% 안 누적 명목 $(틱 = 1/100 $ 인덱스 · 가격×수량) — 학습 원천 liqlab d±k = bookDepth «notional».
    (패리티 장치는 수량 «depth» 로 쟀다 — 비 중앙 1.000 · 명목은 같은 칸에 가격을 곱할 뿐.)"""
    mid = (bb + ba) / 2; out = []
    for arr, lo, hi in [(bid, int(np.ceil(mid * (1 - k / 100))), bb + 1) for k in KS] + [(ask, ba, int(mid * (1 + k / 100)) + 1) for k in KS]:
        hi = min(hi, len(arr))                     # IDX_MAX($10k) 넘는 칸은 replay 가 버린다 — 길이 맞춤(리뷰 10-28)
        out.append(float(arr[lo:hi] @ (np.arange(lo, hi) / TICK)))
    return out


class Bars:
    """초 콜백 → 반분 스냅숏 → 봉 마감 → 10열. hist = 최근 봉 원값(봉 시작 초 → RAW) — 사슬 재시작에도 유지."""

    def __init__(self, out):
        self.out, self.hist = out, {}
        self.chain()

    def chain(self):
        """사슬 (재)시작. 끊긴 순간 진행 중 봉이 마지막 반분 스냅숏(:04:31 이후)까지 받았으면 완성 봉이므로 먼저 닫는다(리뷰 10-28 — 전엔 버려졌다)."""
        if getattr(self, "bar", None) is not None and self.snap and self.snap[0] >= self.bar + BAR_S - 29:
            self.close(self.bar)
        self.chain0 = self.snap = self.slot = self.bar = None

    def on_sec(self, t, bid, ask, bb, ba):
        s = t // 1000
        if self.chain0 is None:
            self.chain0 = s
        bar = s - s % BAR_S
        if self.bar is not None and bar > self.bar:
            self.close(self.bar)                       # 다음 봉 첫 초 — 스냅숏보다 먼저(이 초의 스냅숏은 다음 봉 것)
        self.bar = bar
        if s % 30 >= 1 and s // 30 != self.slot:
            self.slot, self.snap = s // 30, (s, cum(bid, ask, bb, ba))

    def close(self, bar):
        if self.snap is None:
            return
        s, v = self.snap; cut = bar + BAR_S - 1
        if cut - self.chain0 >= WARM_S or bar not in self.hist:   # warm 사슬 값은 DB 에서 불러온 온전한 원값을 덮지 않는다
            self.hist[bar] = v
        for k in [k for k in self.hist if k <= bar - HIST * BAR_S]:
            del self.hist[k]
        self.out(row(bar, v, cut - s + 1, self.hist, cut - self.chain0))


def row(bar, v, age, hist, chain_age):
    """한 봉의 10열(학습 식과 같음 — selftest 가 pandas rolling 식과 대조)."""
    R = dict(zip(RAW, v)); imb = lambda w: [(w[i] - w[i + 3]) / (w[i] + w[i + 3]) if w[i] + w[i + 3] > 0 else np.nan for i in range(3)]
    now, ago = imb(v), (imb(hist[bar - 12 * BAR_S]) if bar - 12 * BAR_S in hist else [np.nan] * 3)
    lg = [np.log(w[0] + w[3]) for k, w in hist.items() if bar - 287 * BAR_S <= k <= bar and w[0] + w[3] > 0]
    R.update(dimb1=now[0], dimb2=now[1], dimb5=now[2], dimb1_d1h=now[0] - ago[0], dimb2_d1h=now[1] - ago[1], dimb5_d1h=now[2] - ago[2],
             depth1_z=(np.log(v[0] + v[3]) - np.mean(lg)) if len(lg) >= 144 and v[0] + v[3] > 0 else np.nan,
             bid_near=v[0] / v[2] if v[2] > 0 else np.nan, ask_near=v[3] / v[5] if v[5] > 0 else np.nan, d_age=float(age),
             chain_age_s=float(chain_age), warm=int(chain_age < WARM_S))
    if R["warm"]:                                  # warm = 먼 칸 덜 참 → 피쳐 비움(원값·d_age 는 진단용으로 남김) — 표시만 하는 규칙 금지(리뷰 10-28)
        R.update({k: np.nan for k in FEAT if k != "d_age"})
    return dict(t=bar * 1000, **{k: (None if isinstance(x, float) and not np.isfinite(x) else x) for k, x in R.items()})


def db_open(path=DB):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(f"CREATE TABLE IF NOT EXISTS bars(t INTEGER PRIMARY KEY, {', '.join(c + ' REAL' for c in COLS)}, written_ms INTEGER)")
    return con


def db_write(con, B) -> bool:
    old = con.execute("SELECT warm FROM bars WHERE t = ?", (B["t"],)).fetchone()
    if old is not None and not old[0]:             # 온전한 봉은 다시 쓰지 않는다 — 라이브에 쓰인 값이 사후에 바뀌면 재현·감사가 깨진다(리뷰 10-28)
        return False
    B = {**B, "written_ms": int(time.time() * 1000)}
    con.execute(f"INSERT OR REPLACE INTO bars({', '.join(B)}) VALUES ({', '.join('?' * len(B))})", list(B.values()))
    con.commit()
    return True


def db_hist(con, bars):
    """재시작 때 최근 봉 원값을 hist 로(depth1_z 24h 창 · d1h 12봉 전)."""
    for t, *v in con.execute(f"SELECT t, {', '.join(RAW)} FROM bars WHERE warm = 0 ORDER BY t DESC LIMIT {HIST}"):
        if None not in v:
            bars.hist[t // 1000] = list(v)


def main():
    con = db_open()
    last = con.execute("SELECT max(t) FROM bars WHERE warm = 0").fetchone()[0]
    now_h = int(time.time()) // 3600
    h0 = max(now_h - 26, (int(last) // 3_600_000 - 13) if last else now_h - 26)   # 먼 칸이 찬 사슬로 마지막 온전한 봉을 잇게 WARM_S + 1시간 앞부터
    n = [0]

    def out(B):
        if db_write(con, B):
            n[0] += 1
            if B["t"] // 1000 // BAR_S % 12 == 0:
                print(f"봉 {time.strftime('%m-%d %H:%M', time.gmtime(B['t'] / 1000))} warm={B['warm']} dimb1={B['dimb1']} d_age={B['d_age']} (누적 {n[0]})", flush=True)

    bars, src = Bars(out), Source(h0)
    db_hist(con, bars)
    print(f"시작 · 마지막 온전한 봉 {last} · {time.strftime('%m-%d %H시', time.gmtime(h0 * 3600))} 파일부터 · hist {len(bars.hist)}봉", flush=True)
    while True:
        bars.chain()
        st = replay(src.lines(), bars.on_sec, lambda e: None)
        print(f"사슬 끝 {st} · {time.strftime('%m-%d %H시', time.gmtime(src.hour * 3600))} {src.n}줄 -- 스냅숏에서 다시", flush=True)
        src.restart(st)


def replay_range(lo_h: str, hi_h: str, out_db: str):
    """과거 시 파일 [lo, hi) 를 같은 Bars 로 재생해 out_db 에 쓴다(대조용 — 실시간 DB 안 건드림). 끊기면 live_ob5m.parity 와 같은 규칙."""
    import pandas as pd
    lo, hi = pd.Timestamp(lo_h + ":00"), pd.Timestamp(hi_h + ":00")
    hours = sorted({p.name[:13] for p in DD.glob("*.jsonl*") if lo <= pd.Timestamp(p.name[:13] + ":00") < hi})
    files = [DD / f"{h}.jsonl" if (DD / f"{h}.jsonl").exists() else DD / f"{h}.jsonl.gz" for h in hours]   # 압축 중 시는 .jsonl 하나만(hour_path 와 같은 순서)
    con = db_open(Path(out_db)); bars = Bars(lambda B: db_write(con, B)); i = 0
    while i < len(files):
        cur = [0, 0]

        def gen(fs=files[i:]):
            for k, p in enumerate(fs):
                cur[0], cur[1] = k, 0
                with (gzip.open(p, "rb") if p.suffix == ".gz" else open(p, "rb")) as f:
                    for ln in f:
                        cur[1] += 1; yield ln
        bars.chain(); g = gen(); st = replay(g, bars.on_sec, lambda e: None); g.close()
        print(f"사슬 끝 {st} · 파일 {files[i + cur[0]].name}", flush=True)
        if st == "eof":
            break
        k = i + cur[0]; i = k if (cur[1] <= 2 and k > i) else k + 1
    print(con.execute("SELECT count(*), sum(warm) FROM bars").fetchone(), flush=True)


def selftest():
    import pandas as pd
    bid, ask = np.zeros(20000), np.zeros(20000); bid[9990], bid[9900], bid[9899] = 2, 1, 5; ask[10010], ask[10100], ask[10110] = 3, 4, 9
    c = cum(bid, ask, 9990, 10010)                                            # 중간 100.00 · ±1% = 99.00~101.00(경계 포함)
    assert np.isclose(c[0], 2 * 99.90 + 1 * 99.00) and np.isclose(c[3], 3 * 100.10 + 4 * 101.00)
    # 반분 스냅숏·봉 마감·나이: 봉 0 의 마지막 스냅숏 = 271초(:04:31) → 나이 299−271+1 = 29
    got = []; B = Bars(got.append)
    for s in range(0, 3 * BAR_S + 5):
        bid[9990] = 2 + s; B.on_sec(s * 1000 + 7, bid, ask, 9990, 10010)
    assert [g["t"] for g in got] == [0, 300_000, 600_000] and got[0]["d_age"] == 29 and np.isclose(got[0]["dm1"], 273 * 99.90 + 99.00) and not B.snap[0] < 900
    assert got[0]["warm"] == 1 and got[0]["chain_age_s"] == 299 and got[0]["dimb1"] is None and got[0]["d_age"] == 29   # warm = 피쳐 비움
    # 사슬이 끊길 때: 마지막 반분(:04:31) 스냅숏까지 받은 봉은 닫고, 못 받은 봉은 버린다
    for upto, want in ((BAR_S * 3 + 272, 900_000), (BAR_S * 3 + 200, None)):
        got = []; B = Bars(got.append)
        for s in range(0, upto):
            B.on_sec(s * 1000, bid, ask, 9990, 10010)
        n0 = len(got); B.chain(); assert (got[-1]["t"] if len(got) > n0 else None) == want, (upto, want)
    # 가격 상한(IDX_MAX) 근처: 매도 +5% 가 배열 끝을 넘어도 죽지 않는다
    big = np.zeros(1000); big[990] = 1; assert np.isfinite(cum(big, big, 980, 990)).all()
    # 온전한 봉은 다시 쓰지 않는다 · warm 봉은 덮을 수 있다
    con = sqlite3.connect(":memory:")
    con.execute(f"CREATE TABLE bars(t INTEGER PRIMARY KEY, {', '.join(c + ' REAL' for c in COLS)}, written_ms INTEGER)")
    base = dict(t=1, **{c: 1.0 for c in COLS})
    assert db_write(con, {**base, "warm": 1}) and db_write(con, {**base, "warm": 0}) and not db_write(con, {**base, "warm": 0, "dm1": 2.0})
    assert con.execute("SELECT dm1 FROM bars").fetchone()[0] == 1.0
    # 10열 식 = 학습 식(pandas rolling — diag_rl_dir_liqfeat_20261028.feats 의 book 줄과 같은 꼴)
    rng = np.random.default_rng(0); n = 400; V = rng.uniform(1, 10, (n, 6)); hist = {}; rows = []
    for i in range(n):
        hist[i * BAR_S] = list(V[i]); rows.append(row(i * BAR_S, list(V[i]), 29, hist, 10**6))
        for k in [k for k in hist if k <= i * BAR_S - HIST * BAR_S]:
            del hist[k]
    E = pd.DataFrame(V, columns=["d-1", "d-2", "d-5", "d1", "d2", "d5"]); b = {k: E[f"d-{k}"] for k in KS}; a = {k: E[f"d{k}"] for k in KS}
    imb = {k: (b[k] - a[k]) / (b[k] + a[k]) for k in KS}; ld1 = np.log(b[1] + a[1])
    ref = pd.DataFrame(dict(**{f"dimb{k}": imb[k] for k in KS}, **{f"dimb{k}_d1h": imb[k] - imb[k].shift(12) for k in KS},
                            depth1_z=ld1 - ld1.rolling(288, min_periods=144).mean(), bid_near=b[1] / b[5], ask_near=a[1] / a[5]))
    got = pd.DataFrame(rows)[ref.columns].astype(float)
    assert np.allclose(got.to_numpy(), ref.to_numpy(), equal_nan=True, atol=1e-12), "10열 식이 학습 식과 다름"
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", nargs=3, metavar=("FROM_UTC_HOUR", "TO_UTC_HOUR", "OUT_SQLITE"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else replay_range(*a.replay) if a.replay else main()
