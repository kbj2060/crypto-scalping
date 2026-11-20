"""합성 풋프린트·캔들(실응답과 같은 모양·크기: 144봉 · 버킷 0.5 · 칸 7열 · 봉당 5~60여 칸). 실응답은 쓰지 않는다.
python -I make_fixture.py  →  fixture/footprint.json · fixture/market-history.json (두 쪽 비교 화면이 똑같이 읽는다)"""
import json, math, pathlib, random

R = random.Random(20261120)
R2 = random.Random(20261121)   # 봉 안 델타 범위(dhl)만
R3 = random.Random(20261122)   # 스택 불균형 줄만
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
    # 스택 불균형(같은 쪽 3배 넘는 줄이 3줄 이상 이어짐)이 몇 봉에 생기게 -- 실응답엔 흔하다. 따로 뽑아 R 순서는 그대로
    if len(levels) >= 10 and R3.random() < 0.4:
        n = R3.choice((6, 7, 8)); j0, s = R3.randrange(max(1, len(levels) - n)), R3.choice((1, 2))
        for l in levels[j0:j0 + n]:
            l[s] = round(l[s] * 4.5, 3); l[3 - s] = round(l[3 - s] * 0.6, 3)
    d = sum(l[1] - l[2] for l in levels)
    # 봉 안 누적 델타 [최저, 최고, 마감](서버 1초 체결 순서) -- 마감을 품는 범위. 따로 뽑아 위 자료(R 순서)는 그대로
    lo_x, hi_x = R2.random() * (abs(d) * 0.6 + 300), R2.random() * (abs(d) * 0.6 + 300)
    bars.append({"time": t, "agg": False, "levels": levels, "dhl": [round(min(0, d) - lo_x, 3), round(max(0, d) + hi_x, 3), round(d, 3)]})
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

# 1주일(/api/footprint-week -- 휠로 넓히기·30m/1h·과거로 옮기기가 부른다): 라이브 첫 봉보다 앞 1,972봉 + 라이브와 겹치는 44봉(값이 달라
# «같은 봉이면 라이브가 이긴다»가 보인다). 따로 뽑아 위 두 파일은 바이트 그대로.
RW = random.Random(20261123)
NW, W_END = 2016, T_END - 100 * 300
wc, wb, px = [], [], 1742.0 - 60.0
for i in range(NW):
    t = W_END - (NW - 1 - i) * 300
    rng = max(1.0, RW.lognormvariate(math.log(4.0), 0.4))
    o = px; c = o + RW.gauss(0, rng * 0.45) + 0.03
    hi = max(o, c) + 0.3 * rng; lo = min(o, c) - 0.3 * rng
    o, c, hi, lo = (round(v, 2) for v in (o, c, hi, lo))
    levels, k = [], q(lo)
    while k <= q(hi) + 1e-9:
        b, s = RW.lognormvariate(5.0, 0.9), RW.lognormvariate(5.0, 0.9)
        levels.append([k, round(b, 3), round(s, 3), round(b * 0.2, 3), round(s * 0.1, 3), round(b * 0.1, 3), round(s * 0.2, 3)])
        k = round(k + BUCKET, 1)
    d = sum(l[1] - l[2] for l in levels)
    bar = {"time": t, "levels": levels}
    if i % 3: bar["dhl"] = [round(min(0, d) - 50, 3), round(max(0, d) + 50, 3), round(d, 3)]
    wb.append(bar)
    wc.append({"time": t, "open": o, "high": hi, "low": lo, "close": c, "volume": round(sum(l[1] + l[2] for l in levels), 3)})
    px = c
(OUT / "footprint-week.json").write_text(json.dumps({"asset": "eth", "candles": wc, "bars": wb, "oi": [], "liq": [], "volPre": []}, separators=(",", ":")))
print("week bars", NW, "bytes", (OUT / "footprint-week.json").stat().st_size)
