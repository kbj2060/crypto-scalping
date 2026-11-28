#!/usr/bin/env python3
"""상황 카드 통합 정답지(2026-11-22 · 일곱째 조각): «같은 입력 → /api/situation 본문 · SSE 한 줄».

원본 몫은 이미 있는 것을 불러 쓴다(ctx 쪽 도구 파일은 고치지 않는다):
  · _flow_read_ctx · _card30_bars · cctx/flows 대역 = tools/gen_ctx_x_fixture.py 의 정의부(모듈이 맨 위에서 출력 폴더를 지우고 시험을 다 돌리므로
    소스를 읽어 «def cell():» 앞까지만 실행한다 -- 지우기·폴더 만들기 두 줄은 빼고)
  · situation.classify · flow_read.read · flow_read.vol_size_mult = 원본 모듈 그대로
  · situation_payload = server.py 2734~2738 · SSE 한 줄 = 3175~3179(옮겨 적음 -- Read 로 본 줄)
rtserve 에 아직 없는 재료는 원본 쪽 입력도 «없음»으로 맞춘다: candles_day·candles = [] · bybit_bars = {} · flow_scales·flow_scales_all = {} ·
micro payload = {"available": False}(코어 호가가 없을 때와 같게) · imb40 링 = [] · _flow_read_ctx 의 inp.bars = [](코어 Situation 에 안 실림) ·
evidence 의 veto 0 · obi·persist None(래스터 요약 없음) · vol_mult = vol_size_mult([]).
코어 Situation = classify(inp) 결과 그대로(rtcore 가 같은 칸으로 민다) · 풋프린트 봉 칸 = gen_ctx_x_fixture 의 bar_cells 로 만들어 같이 싣는다(Rust 시험이 rtcore flow_nets_tick → Micro.flow 로 흘린다).
    source <(bash rtmodel/tools/server_env.sh <venv> --env) && $PY rtserve/tools/gen_sit_fixture.py     # 저장소 루트에서 · 출력 rtserve/tests/fixtures/gen/sit/sit.json(커밋 안 함)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = (ROOT / "rtserve" / "tools" / "gen_ctx_x_fixture.py").read_text()
head = SRC[:SRC.index("\ndef cell():")]
head = head.replace("shutil.rmtree(OUT, ignore_errors=True)\n", "").replace("OUT.mkdir(parents=True)\n", "")
gx: dict = {"__name__": "gen_ctx_x_defs", "__file__": str(ROOT / "rtserve" / "tools" / "gen_ctx_x_fixture.py")}
exec(compile(head, "gen_ctx_x_fixture.py(정의부)", "exec"), gx)   # noqa: S102
rest = SRC[SRC.index("\ndef cell():"):SRC.index("\ndef run(")]
exec(compile(rest, "gen_ctx_x_fixture.py(cell·bar_cells)", "exec"), gx)   # noqa: S102

import dashboard.situation as sit  # noqa: E402
import dashboard.flow_read as fr  # noqa: E402
from types import SimpleNamespace  # noqa: E402

OUT = ROOT / "rtserve" / "tests" / "fixtures" / "gen" / "sit"
R = random.Random(7)


def situation_payload(st, fo_state, mp_state, situation_by):   # server.py 2734~2738
    return {"now": st["now"], "computed_at": st["computed_at"], "streams": {"fo": dict(fo_state), "mp": dict(mp_state)},
            "read": st.get("read"), "vol_mult": st.get("vol_mult"), "vol_mult_by": {a: s.get("vol_mult") for a, s in situation_by.items()}}


def sse(p):   # server.py 3175~3179
    return f"event: situation\ndata: {json.dumps(p, ensure_ascii=False, separators=(',', ':'))}\n\n"


def stream_stat(events):
    return {"connected": R.random() < 0.9, "since": R.choice([None, 1_763_000_000.5]), "last_ms": R.choice([None, 1_763_000_123_456]),
            "events": events, "errors": R.randint(0, 3), "last_error": R.choice([None, "ConnectionResetError"]), "next_funding_ms": R.choice([None, 1_763_028_800_000])}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gx["cctx"].clear(); gx["flows"].clear(); gx["okx_bar_net_by"].clear(); gx["flow_all_hold"].clear()
    now = 1_763_700_000.0 + 3600 * 5 + 17.25
    st = {"bybit_bars": {}, "candles_day": [], "candles": []}
    ms = {"payload": {"available": False}, "flow_scales": {}, "flow_scales_all": {}}
    gx["cctx"]["eth"] = SimpleNamespace(situation_state=st, micro_state=ms, fr_bar_net={}, imb40_ring=[])
    b0 = int(now) // 300 * 300 - 60 * 300
    fb = {b0 + 300 * k: gx["bar_cells"]() for k in range(60)}
    okx = {"first_bar": 0}
    ob: dict = {}
    gx["flows"]["eth"] = SimpleNamespace(footprint_state={"bars": fb}, okx_fp=okx, okx_bars=ob)
    by = {a: {"vol_mult": fr.vol_size_mult([], a)} for a in ("eth", "sol", "xrp")}
    steps, prev_dir, px = [], 0, 3000.0
    for s in range(40):
        now += R.choice([1.0, 1.0, 1.0, 17.0, 300.0])
        cur = int(now) // 300 * 300
        if cur not in fb:
            fb[cur] = gx["bar_cells"]()
        for b in sorted(fb)[-2:]:
            if R.random() < 0.5:
                fb[b][R.randrange(9990, 10050)] = gx["cell"]()
        if s == 10:
            okx["first_bar"] = sorted(fb)[5]
        if okx["first_bar"]:
            for b in fb:
                if b > okx["first_bar"] - 600 and b not in ob:
                    ob[b] = gx["bar_cells"]()
        bars = []
        for k in range(R.choice([3, 8, 13])):   # 3 = 완결 봉 부족(ok False)
            px *= 1 + R.gauss(0, 0.003)
            bars.append({"time": cur - 300 * (13 - k), "high": px * 1.001, "low": px * 0.999, "close": px, "delta": R.gauss(0, 200),
                         "liq_long": abs(R.gauss(0, 1e4)), "liq_short": abs(R.gauss(0, 1e4)), "veto": 0, "oi_delta": R.gauss(0, 100)})
        res = sit.classify({"bars": bars, "mid": px, "prev_dir": prev_dir})
        prev_dir = res.get("dir") or 0
        core = json.loads(json.dumps(res))
        res["computed_at"] = now
        x = gx["_flow_read_ctx"](now, {"bars": []}, "eth")
        st["now"], st["computed_at"] = res, now
        st["read"] = fr.read(res.get("evidence") or {}, x)
        st["vol_mult"] = by["eth"]["vol_mult"]
        fo, mp = stream_stat(R.randint(0, 50)), stream_stat(R.choice([0, 0, 7]))
        fo_state = {"connected": fo["connected"], "since": fo["since"], "last_event_ms": fo["last_ms"], "events": fo["events"], "errors": fo["errors"], "last_error": fo["last_error"]}
        mp_state = {"connected": mp["connected"], "since": mp["since"], "last_ms": mp["last_ms"], "events": mp["events"], "errors": mp["errors"], "last_error": mp["last_error"]}
        if mp["events"] > 0:
            mp_state["next_funding_ms"] = mp["next_funding_ms"]
        p = situation_payload(st, fo_state, mp_state, by)
        steps.append({"now": now, "core": core, "streams": {"fo": fo, "mp": mp}, "fbars": [[b, list(c.values())] for b, c in fb.items()],
                      "okx_first_bar": okx["first_bar"], "okx_bars": [[b, list(c.values())] for b, c in ob.items()],
                      "body": json.dumps(p), "sse": sse(p)})
    (OUT / "sit.json").write_text(json.dumps(steps))
    print(f"상황 카드 걸음 {len(steps)} · ok {sum(1 for x in steps if x['core'].get('ok'))}")


# ── 둘째 판(2026-11-28): inp.bars(move60·oi60) + candles_day(move60_p75·range30_pct·vol_mult) 가 있는 사례 → sit_bars.json ──
# 원본 몫(Read 로 본 줄): load_5m_day 2347~2367(swr 60/600 · 식힘) · compute_situation 2375~2381·2390 · _situation_inputs 2107~2147(봉 접기 -- close·oi_delta 만) ·
# oi_5m_buckets 984~1020. ETH 의 candles 는 원본이 증거 프레임(1500)이지만 rtserve 는 같은 klines 인 candles_day 를 쓴다(설계) -- 여기도 candles = candles_day.
# 초 링(footprint_state["sec"])은 비워 둔다(종가 대체는 VWAP 갈래만 -- 칸은 round(x, 3) · 가격 차례로 넣어 Rust 칸과 같은 값·같은 합 차례).
OI_5M_BAR_SECONDS = 300


def oi_5m_buckets(db, bars, symbol, now):   # server.py 984~1020 옮겨 적음(time.time() 자리 = now)
    floor = (int(now) // OI_5M_BAR_SECONDS - (bars - 1)) * OI_5M_BAR_SECONDS
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    raw = con.execute("SELECT ts_ms, open_interest FROM oi_1s WHERE symbol = ? AND ts_ms >= ? ORDER BY ts_ms", [symbol, (floor - OI_5M_BAR_SECONDS) * 1000]).fetchall()
    con.close()
    agg = {}
    for ts_ms, v in raw:
        bar = int(ts_ms) // 1000 // OI_5M_BAR_SECONDS * OI_5M_BAR_SECONDS
        a = agg.get(bar)
        if a is None:
            agg[bar] = [float(v), float(v), 1]
        else:
            a[1] = float(v); a[2] += 1
    rows = [(bar, o, c, n) for bar, (o, c, n) in sorted(agg.items())]
    out = []
    prev_bar, prev_close = None, 0.0
    for bar, oi_open, oi_close, n in rows:
        bar = int(bar)
        gap = prev_bar is None or bar - prev_bar > OI_5M_BAR_SECONDS
        delta = (oi_close - oi_open) if gap else (oi_close - prev_close)
        prev_bar, prev_close = bar, oi_close
        if bar >= floor:
            out.append([bar, round(delta, 3), round(oi_close, 3), int(n), 1 if gap else 0])
    return out


def main_bars() -> None:
    import sqlite3
    import tempfile
    gx["cctx"].clear(); gx["flows"].clear(); gx["okx_bar_net_by"].clear(); gx["flow_all_hold"].clear()
    RB = random.Random(11)
    t0 = 1_763_900_000 // 300 * 300 + 3600 * 3
    # 5분봉 가격 길(봉 시각 → o,h,l,c,v,tb) -- 언제 받아도 마감봉 값은 같다
    path, px = {}, 3000.0
    for t in range(t0 - 400 * 300, t0 + 200 * 300, 300):
        o = px; px *= 1 + RB.gauss(0, 0.002); c = px
        h, lo = max(o, c) * (1 + abs(RB.gauss(0, 0.001))), min(o, c) * (1 - abs(RB.gauss(0, 0.001)))
        v = RB.uniform(500, 5000)
        path[t] = (o, h, lo, c, v, v * RB.uniform(0.3, 0.7))

    def raw_at(tf, n_rows):   # 바이낸스 klines limit 300: 형성 봉은 경계 ~5초 뒤에 붙는다(원본 주석 2361~2362)
        bs = int(tf) // 300 * 300
        last = bs - 300 if tf - bs < 5 else bs
        ts = [t for t in range(last - 299 * 300, last + 300, 300)][-n_rows:]
        return [[t * 1000, f"{path[t][0]:.2f}", f"{path[t][1]:.2f}", f"{path[t][2]:.2f}", f"{path[t][3]:.2f}", f"{path[t][4]:.3f}", t * 1000 + 299_999,
                 "0", 100, f"{path[t][5]:.3f}", "0", "0"] for t in ts]

    def load(raw, now_s):   # load_5m_day produce 2363~2366
        return [{"time": int(r[0]) // 1000, "high": float(r[2]), "low": float(r[3]), "close": float(r[4]),
                 "volume": float(r[5]), "taker": float(r[9])} for r in raw if int(r[0]) // 1000 + 300 <= now_s]

    # hot oi_1s(10초마다 · 봉 몇 개는 통째로 빔 → gap·None)
    tmp = Path(tempfile.mkdtemp())
    db = tmp / "binance_ctx.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE oi_1s (ts_ms INTEGER, symbol TEXT, open_interest REAL)")
    oi, rows = 1_500_000.0, []
    holes = set(RB.sample(range(t0 - 30 * 300, t0 + 60 * 300, 300), 12))
    for t in range(t0 - 60 * 300, t0 + 80 * 300, 10):
        oi += RB.gauss(0, 50)
        if t // 300 * 300 not in holes:
            rows.append((t * 1000 + RB.randrange(1000), "ethusdt", round(oi, RB.choice([1, 3, 4]))))
    con.executemany("INSERT INTO oi_1s VALUES (?,?,?)", rows)
    con.commit(); con.close()

    def cells_sorted(n=None):   # 가격 차례 · round(x, 3)(Rust 풋프린트 칸 = round(x, 3) · 가격 차례)
        ks = sorted(RB.sample(range(5990, 6030), n or RB.randint(1, 5)))
        return {k: [round(RB.uniform(0, 30), 3) for _ in range(6)] for k in ks}

    now = float(t0) - 2 * 300 + 280.5
    st = {"bybit_bars": {}, "candles_day": [], "candles": []}
    ms = {"payload": {"available": False}, "flow_scales": {}, "flow_scales_all": {}}
    gx["cctx"]["eth"] = SimpleNamespace(situation_state=st, micro_state=ms, fr_bar_net={}, imb40_ring=[])
    fb = {}
    for k in range(40):
        b = (int(now) // 300 - 40 + k) * 300
        if RB.random() > 0.08:
            fb[b] = cells_sorted()
    gx["flows"]["eth"] = SimpleNamespace(footprint_state={"bars": fb, "sec": {}}, okx_fp={"first_bar": 0}, okx_bars={})
    cache = {"ts": None, "payload": None}
    # 걸음: 처음 받기 · 신선(1·17초) · 경계 직후 2초 안(방금 닫힌 봉 없음 → VWAP) · 경계 뒤 식힘 · 700초(새로) · 짧은 응답(봉 개수 70·120·50)
    # ("at", 초) = 661초 넘게 건너뛰어 봉 안 그 초에 내림(새로 받기) -- 298.8 에서 받고 +1.5 = 경계 뒤 0.3초 · 나이 1.5 ≤ 2 라 안 식힘 → 방금 닫힌 봉 없음(VWAP)
    incs = [0, 1, 17, 1, 2, 1, 1, 1, 17, 700, 1, 17, 17, ("at", 298.8), 1.5, 1, 17, 17, 17, 700, 1, 1, 1, 700, 17, 1, ("at", 299.5), 1.0, 1.0, 1, 17, 700, 2, 1, 1, 1, 17]
    shorts = {9: 120, 13: 70, 19: 50, 23: 300}
    steps, prev_dir, n_rows = [], 0, 300
    for s, inc in enumerate(incs):
        if isinstance(inc, tuple):
            nxt = now + 661
            now = (int(nxt) // 300 + 1) * 300 + inc[1]
        else:
            now += inc
        n_rows = shorts.get(s, n_rows)
        cur = int(now) // 300 * 300
        for b in range(max(fb) + 300, cur + 300, 300):
            if RB.random() > 0.1 or b == cur:
                fb[b] = cells_sorted()
        for b in sorted(fb)[:-60]:
            del fb[b]
        for b in sorted(fb)[-2:]:
            if RB.random() < 0.4:
                fb[b] = cells_sorted()
        # load_5m_day swr(60/600) · 식힘(2354~2355) -- 묵은 값 구간(60 ≤ 나이 < 660)은 뒤에서 갱신이라 걸음을 그 밖으로만 짰다
        pl = cache["payload"]
        if pl and int(pl[-1]["time"]) < cur - 300 and now - cache["ts"] > 2.0:
            cache["ts"] = 0.0
        age = None if cache["payload"] is None else now - cache["ts"]
        fetched = age is None or age >= 660.0
        assert age is None or age < 60.0 or fetched, (s, age)
        raw = raw_at(now, n_rows)
        if fetched:
            cache.update(ts=now, payload=load(raw, now))
        day5 = cache["payload"]
        st["candles_day"] = day5
        # _situation_inputs 의 봉(close·oi_delta) -- 2110 · 2118~2147
        oi_map = {int(b): (None if gap else float(dl)) for b, dl, _c, _n, gap in oi_5m_buckets(db, 2 * sit.WINDOW + 3, "ethusdt", now)}
        candles = {int(c["time"]): c for c in day5 if not c.get("partial")}
        want = [b for b in sorted(fb) if b < cur][-(2 * sit.WINDOW + 1):]
        bars = []
        for b in want:
            c = candles.get(b)
            close = float(c["close"]) if c else None
            if close is None:   # 초 링은 비었다 → VWAP(2146~2147)
                cells = fb[b]; tot = sum(v[0] + v[1] for v in cells.values())
                close = (sum(k * 0.5 * (v[0] + v[1]) for k, v in cells.items()) / tot) if tot > 0 else None
            bars.append({"time": b, "close": close, "oi_delta": oi_map.get(b)})
        cbars = []
        for k in range(R.choice([3, 8, 13])):
            px *= 1 + R.gauss(0, 0.003)
            cbars.append({"time": cur - 300 * (13 - k), "high": px * 1.001, "low": px * 0.999, "close": px, "delta": R.gauss(0, 200),
                          "liq_long": abs(R.gauss(0, 1e4)), "liq_short": abs(R.gauss(0, 1e4)), "veto": 0, "oi_delta": R.gauss(0, 100)})
        res = sit.classify({"bars": cbars, "mid": px, "prev_dir": prev_dir})
        prev_dir = res.get("dir") or 0
        core = json.loads(json.dumps(res))
        res["computed_at"] = now
        x = gx["_flow_read_ctx"](now, {"bars": bars}, "eth")
        st["now"], st["computed_at"] = res, now
        st["read"] = fr.read(res.get("evidence") or {}, x)
        vm = fr.vol_size_mult([c for c in day5 if int(c["time"]) < cur], "eth")   # 2380~2381
        st["vol_mult"] = vm
        by = {"eth": {"vol_mult": vm}, "sol": {"vol_mult": fr.vol_size_mult([], "sol")}, "xrp": {"vol_mult": fr.vol_size_mult([], "xrp")}}
        fo, mp = stream_stat(R.randint(0, 50)), stream_stat(R.choice([0, 0, 7]))
        fo_state = {"connected": fo["connected"], "since": fo["since"], "last_event_ms": fo["last_ms"], "events": fo["events"], "errors": fo["errors"], "last_error": fo["last_error"]}
        mp_state = {"connected": mp["connected"], "since": mp["since"], "last_ms": mp["last_ms"], "events": mp["events"], "errors": mp["errors"], "last_error": mp["last_error"]}
        if mp["events"] > 0:
            mp_state["next_funding_ms"] = mp["next_funding_ms"]
        p = situation_payload(st, fo_state, mp_state, by)
        steps.append({"now": now, "core": core, "streams": {"fo": fo, "mp": mp}, "fbars": [[b, list(c.values())] for b, c in fb.items()],
                      "fp_prices": [[b, [k * 0.5 for k in c]] for b, c in fb.items()],
                      "okx_first_bar": 0, "okx_bars": [], "raw": json.dumps(raw), "fetched": fetched,
                      "x": json.dumps(x), "body": json.dumps(p), "sse": sse(p),
                      "has": {"move60": "move60" in x, "oi60": "oi60" in x, "p75": "move60_p75" in x, "range30": "range30_pct" in x, "vm": vm is not None,
                              "vwap": any(b["time"] not in candles for b in bars)}})
    (OUT / "sit_bars.json").write_text(json.dumps({"oi_db": base64.b64encode(db.read_bytes()).decode(), "steps": steps}))
    cnt = {k: sum(1 for s in steps if s["has"][k]) for k in steps[0]["has"]}
    print(f"bars·day5 걸음 {len(steps)} · 받기 {sum(s['fetched'] for s in steps)} · {cnt}")


if __name__ == "__main__":
    import base64
    main()
    main_bars()
