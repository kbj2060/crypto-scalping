"""합성 풋프린트·캔들(실응답과 같은 모양·크기: 144봉 · 버킷 0.5 · 칸 7열 · 봉당 5~60여 칸). 실응답은 쓰지 않는다.
python -I make_fixture.py  →  fixture/footprint.json · fixture/market-history.json (두 쪽 비교 화면이 똑같이 읽는다)"""
import json, math, pathlib, random

R = random.Random(20261120)
OUT = pathlib.Path(__file__).resolve().parent / "fixture"
T_END = 1795156500          # 마지막(형성 중) 봉 시각 -- 하네스 시계(1795156790)가 이 봉 안이다
N, BUCKET = 144, 0.5
q = lambda p: round(p / BUCKET) * BUCKET

bars, candles, px = [], [], 1742.0
for i in range(N):
    t = T_END - (N - 1 - i) * 300
    rng = max(1.0, R.lognormvariate(math.log(5.0), 0.45))          # 실응답 36봉 고저폭 2.2~9.8
    o = px
    c = o + R.gauss(0, rng * 0.45)
    hi = max(o, c) + (0.15 + 0.35 * R.random()) * rng
    lo = min(o, c) - (0.15 + 0.35 * R.random()) * rng
    o, c, hi, lo = (round(v, 2) for v in (o, c, hi, lo))
    levels, k, top, mid = [], q(lo), q(hi), (hi + lo) / 2
    while k <= top + 1e-9:
        w = math.exp(-((k - mid) / max(rng * 0.35, 0.5)) ** 2)       # 가운데가 두꺼운 체결 분포
        buy = R.lognormvariate(math.log(80 + 900 * w), 0.8)
        sell = R.lognormvariate(math.log(80 + 900 * w), 0.8)
        wb = buy * R.random() * 0.5 if R.random() < 0.6 else 0.0
        ws = sell * R.random() * 0.5 if R.random() < 0.6 else 0.0
        levels.append([k, round(buy, 3), round(sell, 3), round(wb, 3), round(ws, 3),
                       round(buy * R.random() * 0.3, 3), round(sell * R.random() * 0.3, 3)])
        k = round(k + BUCKET, 1)
    d = sum(l[1] - l[2] for l in levels)
    bars.append({"time": t, "agg": False, "levels": levels, "dhl": [round(d, 3), 0.0, round(d, 3)]})
    cd = {"time": t, "open": o, "high": hi, "low": lo, "close": c, "volume": round(sum(l[1] + l[2] for l in levels), 3)}
    if i == N - 1:
        cd["partial"] = True
    candles.append(cd)
    px = c

fp = {"symbol": "ETHUSDT", "bucket": BUCKET, "barSeconds": 300, "barsExpected": N, "ready": True, "full": True,
      "retailMaxUsd": 10000.0, "whaleMinUsd": 100000.0, "venues": ["binance-perp"], "bars": bars}
OUT.mkdir(exist_ok=True)
(OUT / "footprint.json").write_text(json.dumps(fp, separators=(",", ":")))
(OUT / "market-history.json").write_text(json.dumps({"asset": "eth", "candles": candles}, separators=(",", ":")))
print("bars", N, "levels/bar", min(len(b["levels"]) for b in bars), "~", max(len(b["levels"]) for b in bars),
      "bytes", (OUT / "footprint.json").stat().st_size)
