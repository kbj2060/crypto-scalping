"""liqmap_scorer 정답지: 원본 live_liqmap_scorer_20261008.main() 을 합성 klines(가짜 urlopen) · 합성 hot SQLite 셋 · 가짜 시계로 세 번 돌린다.

  1회: 이미 적힌 origin 몇 개(건너뜀) · 진행 중 봉(버림) · 청산 = 세 거래소·다른 종목·모르는 쪽·마크 없음·5초 경계·같은 ms
  2회: 한 시간 뒤(새 봉 하나 → 한 줄) · 3회: 같은 시각(새 기준 없음 → status 만)
출력 = tests/fixtures/liqmap_{input,golden}.json. 바깥 접속 없음.
  NPY_DISABLE_CPU_FEATURES=X86_V4 python3 rtcollect/tools/gen_liqmap_fixture.py
"""
from __future__ import annotations

import io
import json
import random
import sqlite3
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_liqmap_scorer_20261008 as S  # noqa: E402

FIX = ROOT / "rtcollect/tests/fixtures"
rnd = random.Random(20261121)
T0 = 1_791_590_400_000          # 2026-10-10T00:00:00Z (정시)
HR = 3_600_000
N = S.W + S.H + S.BACK_H + 2     # 198 = 원본 limit


def kl(n_hours: int) -> list[list]:
    rows, px = [], 2500.0
    for j in range(n_hours):
        o = px
        c = round(o * (1 + rnd.gauss(0, 0.006)), 2)
        hi = round(max(o, c) * (1 + abs(rnd.gauss(0, 0.003))), 2)
        lo = round(min(o, c) * (1 - abs(rnd.gauss(0, 0.003))), 2)
        v = round(rnd.uniform(2e4, 9e4), 3) if j % 29 else round(rnd.uniform(0.5, 20), 3)   # 아주 얇은 봉 -- 지도 칸이 1e-6~1e-5 에 들게(버림 경계)
        qv = round(v * (hi + lo) / 2, 4)
        tb = round(qv * rnd.uniform(0.3, 0.7), 4)
        t = T0 + j * HR
        rows.append([t, f"{o:.2f}", f"{hi:.2f}", f"{lo:.2f}", f"{c:.2f}", f"{v:.3f}", t + HR - 1, f"{qv:.4f}", 1234, f"{v/2:.3f}", f"{tb:.4f}", "0"])
        px = c
    return rows


ALL = kl(N + 1)                 # 2회차에 새 봉 하나 더
PX = {r[0]: (float(r[3]), float(r[2])) for r in ALL}


def make_hot(d: Path) -> dict:
    """합성 청산·마크. 반환 = 표별 행(입력 기록)."""
    t_hi = ALL[-1][0] + HR
    marks, bl, ok, bb = [], [], [], []
    for _ in range(2600):
        t = rnd.randrange(ALL[-40][0], t_hi)
        v = rnd.random()
        if v < 0.5:
            bl.append([t, rnd.choice(["long", "short"]), round(rnd.uniform(0.01, 40), 3), "ETHUSDT"])
        elif v < 0.75:
            ok.append(["ETH-USDT-SWAP", t, rnd.choice(["long", "short"]), round(rnd.uniform(0.01, 30), 2)])
        else:
            bb.append([t, "ETHUSDT", rnd.choice(["Buy", "Sell"]), round(rnd.uniform(0.01, 25), 3)])
    # 경계·잡음: 마크 구멍 안(5초 넘음 → 버림) · 정확히 5000ms 뒤(남음) · 다른 종목 · 모르는 쪽 · 같은 ms 다른 거래소 · NULL 마크 시각
    g = ALL[-20][0] + 100_000
    bl += [[g + 5_000, "long", 1.5, "ETHUSDT"], [g + 5_001, "short", 2.5, "ETHUSDT"], [g - 1000 + 5000, "long", 3.25, "ETHUSDT"],
           [ALL[-12][0] + 3000, "long", 9.0, "BTCUSDT"], [ALL[-15][0] + 50_000, "long", 4.0, "ETHUSDT"]]
    ok += [["BTC-USDT-SWAP", ALL[-12][0] + 3000, "long", 1.0], ["ETH-USDT-SWAP", ALL[-30][0] + 1, "net", 2.0]]
    bb += [[ALL[-30][0] + 2, "ETHUSDT", "Unknown", 1.0]]
    bl += [[ALL[-k][0], rnd.choice(["long", "short"]), 5.0 + k, "ETHUSDT"] for k in range(4, 30, 3)]   # 정시 = 앞 기준의 창 끝(안 셈) · 뒤 기준의 창 시작
    for j in range(8):          # 같은 ms 를 세 거래소가 같이
        t = ALL[-25 + j][0] + 777_000
        bl.append([t, "long", 0.1 + j, "ETHUSDT"]); ok.append(["ETH-USDT-SWAP", t, "long", 0.2 + j]); bb.append([t, "ETHUSDT", "Buy", 0.3 + j])
    # 마크 1초 -- 청산마다 앞 6초만(파일 크기) · 구멍 둘(6초 · 30초) · NULL 하나 · 다른 종목
    sec = sorted({t // 1000 * 1000 - j * 1000 for t in [r[0] for r in bl] + [r[1] for r in ok] + [r[0] for r in bb] for j in range(7)})
    for t in sec:
        if ALL[-20][0] + 100_000 <= t < ALL[-20][0] + 106_000 or ALL[-10][0] + 7_000 <= t < ALL[-10][0] + 37_000:
            continue
        lo, hi = PX.get(T0 + (t - T0) // HR * HR, PX[ALL[-1][0]])
        marks.append([t, "ethusdt", None if t == ALL[-15][0] + 50_000 else round(rnd.uniform(lo, hi), 2)])
    marks.append([ALL[-12][0] + 3000, "btcusdt", 60000.0])
    rnd.shuffle(bl); rnd.shuffle(ok); rnd.shuffle(bb)
    c = sqlite3.connect(d / "binance_ctx.sqlite")
    c.execute("CREATE TABLE mark_price_1s(ts_ms INTEGER, symbol TEXT, mark REAL, index_px REAL, funding_rate REAL, next_funding_ms INTEGER, PRIMARY KEY (ts_ms, symbol)) WITHOUT ROWID")
    c.execute("CREATE TABLE liquidations(ts_ms INTEGER, side TEXT, qty REAL, price REAL, usd REAL, symbol TEXT)")
    c.execute("CREATE INDEX mark_sym ON mark_price_1s(symbol, ts_ms)")
    c.executemany("INSERT INTO mark_price_1s(ts_ms, symbol, mark) VALUES (?,?,?)", marks)
    c.executemany("INSERT INTO liquidations(ts_ms, side, qty, symbol) VALUES (?,?,?,?)", bl)
    c.commit(); c.close()
    c = sqlite3.connect(d / "okx_ctx.sqlite")
    c.execute("CREATE TABLE okx_liquidations(inst_id VARCHAR, inst_family VARCHAR, ts_ms BIGINT, side VARCHAR, pos_side VARCHAR, bk_px DOUBLE, sz_contracts DOUBLE, sz_base DOUBLE, bk_loss DOUBLE)")
    c.executemany("INSERT INTO okx_liquidations(inst_id, ts_ms, pos_side, sz_base) VALUES (?,?,?,?)", ok)
    c.commit(); c.close()
    c = sqlite3.connect(d / "bybit_liq.sqlite")
    c.execute("CREATE TABLE bybit_liquidations(ts_ms INTEGER, symbol TEXT, side TEXT, qty REAL, price REAL, recv_ms INTEGER)")
    c.execute("CREATE INDEX bybit_liq_sym_ts ON bybit_liquidations(symbol, ts_ms)")
    c.executemany("INSERT INTO bybit_liquidations(ts_ms, symbol, side, qty) VALUES (?,?,?,?)", bb)
    c.commit(); c.close()
    return {"marks": marks, "binance": bl, "okx": ok, "bybit": bb}


def main() -> None:
    reqs: list = []
    resp: dict = {}

    def urlopen(req, timeout=None):
        reqs.append([req.full_url, req.get_header("User-agent"), timeout])
        return io.BytesIO(resp["body"].encode())

    S.urllib.request.urlopen = urlopen
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        hot = make_hot(d)
        S.HOT, S.OUT, S.STATUS = d, d / "out/liqmap_score.jsonl", d / "out/status.json"
        S.OUT.parent.mkdir()
        # 이미 적힌 기준 시각 셋(건너뜀) -- 원본은 origin 만 읽는다
        pre = "".join(json.dumps({"origin": S.pd.Timestamp(ALL[i][0], unit="ms", tz="UTC").isoformat(), "x": 1}) + "\n"
                      for i in range(S.W, 150) if i not in (40, 41, 120))   # 오래된 기준 셋만 비워 둔다(채움) -- 크기 때문에 나머지는 적힌 것으로
        pre += "\n"             # 빈 줄은 건너뛴다
        S.OUT.write_text(pre)
        runs, gold = [], []
        for n_rows, now in ((N, ALL[N - 1][0] / 1000 + 600), (N + 1, ALL[N][0] / 1000 + 300), (N + 1, ALL[N][0] / 1000 + 900)):
            rows = ALL[n_rows - N:n_rows]       # 원본은 마지막 198 봉을 받는다 · 마지막 봉은 진행 중(ct > now)
            resp["body"] = json.dumps(rows)
            S.time = types.SimpleNamespace(time=lambda now=now: now, gmtime=lambda *_a, now=now: time.gmtime(now), strftime=time.strftime)
            reqs.clear()
            S.main()
            runs.append({"now": now, "body": resp["body"]})
            txt = S.OUT.read_text()
            gold.append({"out_added": txt[sum(len(g["out_added"]) for g in gold) + len(pre):], "status": S.STATUS.read_text(), "reqs": list(reqs)})
        # 서버 파이썬이 3.12+ 이면 sum() 이 Neumaier 보정합이다 -- 그때 쪽 합(tot)이 갈리는 칸 수(정보용)
        S.time = types.SimpleNamespace(time=lambda: runs[0]["now"], gmtime=time.gmtime, strftime=time.strftime)
        diff = neumaier_cells(json.loads(runs[0]["body"]), runs[0]["now"], d)
    (FIX / "liqmap_input.json").write_text(json.dumps({"runs": runs, "pre": pre, "hot": hot, "t0": T0}, separators=(",", ":")))
    (FIX / "liqmap_golden.json").write_text(json.dumps({"runs": gold, "neumaier_cells_differ": diff}, separators=(",", ":"), ensure_ascii=False))
    print(f"정답지: 새 줄 {[g['out_added'].count(chr(10)) for g in gold]} · 요청 {[len(g['reqs']) for g in gold]} · 마크 {len(hot['marks'])} · Neumaier 로 갈리는 지도 칸 {diff}")


def neumaier_cells(rows: list, now: float, d: Path) -> int:
    """sum() 을 Neumaier 로 바꿨을 때 round(v/tot, 6) 이 갈리는 칸 수."""
    def neu(xs):
        s = c = 0.0
        for x in xs:
            t = s + x
            c += (s - t) + x if abs(s) >= abs(x) else (x - t) + s
            s = t
        return s + c
    k = S.pd.DataFrame({"t": [int(r[0]) for r in rows], "high": [float(r[2]) for r in rows], "low": [float(r[3]) for r in rows],
                        "close": [float(r[4]) for r in rows], "volume": [float(r[5]) for r in rows], "qv": [float(r[7]) for r in rows],
                        "tbq": [float(r[10]) for r in rows]})
    n = 0
    for i in range(S.W, len(k) - S.H + 1):
        for name, sp in S.variants(k).items():
            _, mp = S.V.liq_map(k, i, S.W, sp["tiers"], "splice", sp["vol"], halflife=sp.get("halflife", S.V.LM.RECENCY_HALFLIFE_HOURS))
            for s in ("long", "short"):
                vs = [v for (a, o), v in mp.items() if a == s and abs(o) <= S.SPAN]
                t1, t2 = (sum(vs) or 1.0), (neu(vs) or 1.0)
                n += sum(round(v / t1, 6) != round(v / t2, 6) for v in vs)
    return n


if __name__ == "__main__":
    main()
