#!/usr/bin/env python3
"""rtserve 관계 읽기 입력 x(src/ctx_x.rs) · 장부 한 줄 · 수신 상태 일치 시험의 입력·정답지 (2026-11-22).

정답 = 원본 함수. import 되는 것은 그대로 부른다: dashboard/micro_ref.py is_weekend · flow_hour_scales_from_bars · sum_venue_bars ·
flow_hour_scales_combos · flow_hour_scales(시계는 time.time 을 고정해서).
server.py 의 것은 import 할 수 없어(make_app 안 닫힌 함수 · 모듈은 서버 전체를 띄운다) **옮겨 적었다**(Read 로 본 실제 줄 · 들여쓰기만 바꿈):
  · _flow_read_ctx 2288~2406 (아래 _flow_read_ctx -- 본문 글자 그대로. C·F·okx_bar_net_by·flow_all_hold 는 원본 이름 그대로의 대역)
  · _card30_bars 2282~2286 (그대로) · _card30_pred 는 대역(정해 둔 확률을 돌려준다 -- 모델 대조는 rtmodel 몫)
  · bybit_bars 759~774 (_read_only_rows = 읽기 전용 sqlite 로)
  · VENUE_SETS 756
  · 장부 한 줄 2494~2513 · _last_logged_bar 2408~2416
합성 입력을 걸음(1초~1분)마다 고치며(새 봉 · 최근 봉 늦은 칸 · 굳은 봉 고침 · OKX 시작 · Bybit 늦음 · 기준선 빠짐) 같은 상태로 이어 부른다.
🔴서버 판 파이썬으로(numpy 2.2.5 -- convolve·std·percentile·median 합 차례가 서버와 같아야 한다):
    source <(bash rtmodel/tools/server_env.sh <venv> --env) && $PY rtserve/tools/gen_ctx_x_fixture.py   # 저장소 루트에서
출력: rtserve/tests/fixtures/gen/ctx_x/ (커밋 안 함)
"""
from __future__ import annotations

import json
import random
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rtserve" / "tests" / "fixtures" / "gen" / "ctx_x"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import dashboard.micro_ref as mref  # noqa: E402
import dashboard.flow_read as fr  # noqa: E402

R = random.Random(20261123)
FOOTPRINT_BAR_SECONDS = 300
VENUE_SETS = (("binance", "okx", "bybit"), ("binance", "okx"))       # server.py 756

# ── 원본 make_app 안 이름들의 대역 ──
cctx: dict[str, SimpleNamespace] = {}
flows: dict[str, SimpleNamespace] = {}
okx_bar_net_by: dict[str, dict[int, tuple[float, ...]]] = {}
flow_all_hold: dict[str, dict[str, Any]] = {}
CARD: dict[str, float] = {}


def _card30_pred(bars, asset="eth"):   # 대역
    return dict(CARD) if bars else {}


def _card30_bars(now: float, situation_state: dict[str, Any]) -> list[dict[str, Any]]:   # server.py 2282~2286 옮겨 적음
    """_card30_pred 입력 = 완결 5분봉(테이커 있는 것). _flow_read_ctx 와 compute_situation 미리 계산이 같은 입력을 쓰게 한 곳에서."""
    bar_start = int(now) // FOOTPRINT_BAR_SECONDS * FOOTPRINT_BAR_SECONDS
    day5 = situation_state.get("candles_day") or situation_state.get("candles") or []
    return [c for c in day5 if int(c["time"]) < bar_start and "taker" in c]


def _flow_read_ctx(now: float, inp: dict[str, Any], asset: str = "eth") -> dict[str, Any]:   # server.py 2288~2406 옮겨 적음
    C, F = cctx[asset], flows[asset]
    situation_state, micro_state, fr_bar_net, imb40_ring = C.situation_state, C.micro_state, C.fr_bar_net, C.imb40_ring
    footprint_state, okx_fp, okx_bars = F.footprint_state, F.okx_fp, F.okx_bars
    bar = FOOTPRINT_BAR_SECONDS
    bar_start = int(now) // bar * bar
    fbars = footprint_state["bars"]
    done = [b for b in sorted(fbars) if b < bar_start]
    for b in done:
        # 최근 두 봉은 늦게 오는 백필이 있을 수 있어 매번 다시 센다. 그 앞은 굳었다.
        if b not in fr_bar_net or b >= bar_start - 2 * bar:
            cells = fbars[b].values()
            fr_bar_net[b] = (sum(v[0] - v[1] for v in cells), sum(v[2] - v[3] for v in cells),
                             sum(v[4] - v[5] for v in cells))
    for b in [b for b in fr_bar_net if b not in fbars]:
        del fr_bar_net[b]
    x: dict[str, Any] = {}
    if len(done) >= 12:
        grid = range(done[0], done[-1] + bar, bar)       # 빈 봉은 NaN -- 창 합이 구멍을 건너뛰지 않게
        a = np.array([fr_bar_net.get(b, (np.nan, np.nan, np.nan)) for b in grid], dtype=float)
        ser = {"whale": a[:, 1], "mid": a[:, 0] - a[:, 1] - a[:, 2], "retail": a[:, 2]}
        roll = lambda v, k: np.convolve(v, np.ones(k), "valid")   # noqa: E731
        r60 = {g: roll(v, 12) for g, v in ser.items()}
        if all(np.isfinite(r[-1]) for r in r60.values()):
            x["net60"] = {g: float(r[-1]) for g, r in r60.items()}
            past = {g: r[:-1][np.isfinite(r[:-1])] for g, r in r60.items()}
            if all(len(p) >= 72 and p.std() > 0 for p in past.values()):
                x["z60"] = {g: float(r60[g][-1] / past[g].std()) for g in r60}
            # 2026-10-01 화면용: 같은 UTC 시 14일 MAD 로 나눈 값(docs/experiments/flow_z60_baseline_20261001 -- 방향성 동률,
            #   새벽·미국장 공정). z60 은 융합·관계 문장이 검정된 정의라 그대로 둔다.
            sc = (micro_state.get("flow_scales") or {}).get((mref.is_weekend(done[-1]), time.gmtime(done[-1]).tm_hour))
            if sc:
                x["z60h"] = {g: float(r60[g][-1] / sc[g]) for g in r60}
                # 2026-10-26 사용자 «② 다섯 줄 모두 30분»: 같은 기준선 칸의 30분 MAD(mid30·whale30·retail30, micro_ref)로 30분 합 σ
                r30g = {g: roll(v, 6) for g, v in ser.items()}
                if all(g + "30" in sc for g in r30g) and all(np.isfinite(r[-1]) for r in r30g.values()):
                    x["z30h"] = {g: float(r30g[g][-1] / sc[g + "30"]) for g in r30g}
        r30 = roll(a[:, 0], 6)
        p30 = r30[:-1][np.isfinite(r30[:-1])]
        if np.isfinite(r30[-1]) and len(p30) >= 72 and p30.std() > 0:
            x["cvd30_z"] = float(r30[-1] / p30.std())
        # 2026-10-06 거래소 합산(사용자 «바로 합산»): 같은 격자에 OKX·Bybit 봉을 더해 60분 크기별 합·σ · 30분 CVD σ · 테이커 비율.
        #   σ 분모 = 같은 조합의 14일 기준선(flow_scales_all). 조합은 VENUE_SETS 앞부터 -- 60분 창과 기준선 칸이 다 있는 첫 조합.
        #   한 거래소라도 빈 봉은 그 합을 «모름»(NaN)으로 둔다. 판정(z60·맞대결·융합)은 바이낸스 그대로다(바이낸스로 검정됐다).
        ofb, onet, bb = okx_fp["first_bar"], okx_bar_net_by.setdefault(asset, {}), situation_state.get("bybit_bars") or {}
        for b in grid:
            if ofb and b > ofb and b in okx_bars and (b not in onet or b >= bar_start - 2 * bar):
                cells = okx_bars[b].values()
                onet[b] = (sum(v[0] - v[1] for v in cells), sum(v[2] - v[3] for v in cells), sum(v[4] - v[5] for v in cells),
                           sum(v[0] for v in cells), sum(v[1] for v in cells))
        for b in [b for b in onet if b not in okx_bars]:
            del onet[b]
        nan3 = [np.nan] * 3
        ven = {"okx": np.array([list(onet[b][:3]) if b in onet else nan3 for b in grid], dtype=float),
               "bybit": np.array([[bb[b][0] - bb[b][1], bb[b][2], bb[b][3]] if b in bb else nan3 for b in grid], dtype=float)}
        hkey = (mref.is_weekend(done[-1]), time.gmtime(done[-1]).tm_hour)
        # 봉 마감 뒤 Bybit 마지막 봉은 ~10초+캐시 20초 늦게 온다 -- 그동안 조합을 바이낸스+OKX 로 바꾸면 5분마다 값·호버가 튄다.
        #   직전(한 봉 전) 합산값을 그대로 둔다. 다음 봉 안에도 안 오면(수집기 정지) 아래 루프가 두 거래소로 물러선다.
        hold = flow_all_hold.get(asset)
        late = bool(hold) and "bybit" in hold["venues"] and done[-1] not in bb and now < done[-1] + 2 * bar
        if late:
            x.update(hold)
        for vs in (() if late else VENUE_SETS):
            sca = ((micro_state.get("flow_scales_all") or {}).get(vs) or {}).get(hkey)
            A = a + sum(ven[v] for v in vs[1:])
            s60 = {g: roll(v, 12) for g, v in (("whale", A[:, 1]), ("mid", A[:, 0] - A[:, 1] - A[:, 2]), ("retail", A[:, 2]))}
            if not sca or not all(np.isfinite(q[-1]) for q in s60.values()):
                continue
            x["venues"] = list(vs)
            x["net60_all"] = {g: float(q[-1]) for g, q in s60.items()}
            x["z60h_all"] = {g: float(s60[g][-1] / sca[g]) for g in s60}
            s30 = {g: roll(v, 6) for g, v in (("whale", A[:, 1]), ("mid", A[:, 0] - A[:, 1] - A[:, 2]), ("retail", A[:, 2]))}
            if all(g + "30" in sca for g in s30) and all(np.isfinite(q[-1]) for q in s30.values()):   # 2026-10-26 30분판(같은 조합·같은 칸)
                x["net30_all"] = {g: float(q[-1]) for g, q in s30.items()}
                x["z30h_all"] = {g: float(s30[g][-1] / sca[g + "30"]) for g in s30}
            r30a = roll(A[:, 0], 6)
            p30a = r30a[:-1][np.isfinite(r30a[:-1])]
            if np.isfinite(r30a[-1]) and len(p30a) >= 72 and p30a.std() > 0:
                x["cvd30_all_z"] = float(r30a[-1] / p30a.std())
            lb = done[-1]   # 테이커 매수÷매도 = 마지막 닫힌 봉, 같은 조합(바이낸스 API 비율 대신 체결 테이프로 직접)
            buy = sum(v[0] for v in fbars[lb].values()) + onet[lb][3] + (bb[lb][0] if "bybit" in vs else 0.0)
            sell = sum(v[1] for v in fbars[lb].values()) + onet[lb][4] + (bb[lb][1] if "bybit" in vs else 0.0)
            if sell > 0:
                x["taker_all"] = buy / sell
            flow_all_hold[asset] = {k: x[k] for k in ("venues", "net60_all", "z60h_all", "net30_all", "z30h_all", "cvd30_all_z", "taker_all") if k in x}
            break
    last6 = done[-6:]
    if okx_fp["first_bar"] and len(last6) == 6 and all(b > okx_fp["first_bar"] and b in okx_bars for b in last6):
        x["okx30"] = float(sum(v[0] - v[1] for b in last6 for v in okx_bars[b].values()))
    bars = inp.get("bars") or []
    if len(bars) >= 13 and bars[-1].get("close") and bars[-13].get("close"):
        x["move60"] = (bars[-1]["close"] / bars[-13]["close"] - 1) * 1e4
        ois = [b.get("oi_delta") for b in bars[-12:] if b.get("oi_delta") is not None]
        if len(ois) >= 10:
            x["oi60"] = float(sum(ois))
    # 🔴24h 분위는 **정확히 24h** 여야 연구 정의와 같다. 상황 카드의 candles 캐시는 ~135봉(11h)뿐이라(09-25 실측
    #   gate_pct 86/130) 따로 받은 5분봉 300개(candles_day)를 쓴다. 없으면 카드 캐시로 물러선다.
    day5 = situation_state.get("candles_day") or situation_state.get("candles") or []
    closes = np.array([float(c["close"]) for c in [c for c in day5 if int(c["time"]) < bar_start][-300:]], dtype=float)
    if len(closes) >= 100:
        x["move60_p75"] = float(np.percentile(np.abs(closes[12:] / closes[:-12] - 1) * 1e4, 75))
    # 융합 신호의 크기 관문: 완결 봉 30분 창 고저폭이 최근 24h(288창) 안 몇 분위인가(자기 포함). 연구와 같은 정의.
    cds = [c for c in day5 if int(c["time"]) < bar_start][-293:]
    if len(cds) >= 66:
        hh = np.array([float(c["high"]) for c in cds]); lo = np.array([float(c["low"]) for c in cds])
        cc = np.array([float(c["close"]) for c in cds])
        win = np.lib.stride_tricks.sliding_window_view
        rg = (win(hh, 6).max(1) - win(lo, 6).min(1)) / cc[5:]
        x["range30_pct"] = float(np.mean(rg <= rg[-1]))
    pr = _card30_pred(_card30_bars(now, situation_state), asset)   # ETH 모델(SOL·XRP 도 그대로 -- 검정 통과) · 보통은 compute_situation 이 스레드에서 미리 채운 캐시
    x["dir_p"], x["reach_p"] = pr.get("dir"), pr.get("reach")
    mp = micro_state["payload"] if micro_state["payload"].get("available") else {}
    imb = mp.get("imb40")
    if imb is not None and len(imb40_ring) >= 600:
        ring = np.fromiter(imb40_ring, dtype=float)
        x["imb40"], x["imb40_pct"] = float(imb), float(np.count_nonzero(ring <= imb) / ring.size)
    x.update(liq60s=mp.get("liq60") or {}, agree=mp.get("agree"), trigger_live=bool((mp.get("trigger") or {}).get("live")),
             sr=mp.get("sr") or {}, act=mp.get("act"), vol_pct=mp.get("vol60_pct"))
    return x


def _read_only_rows(db: Path, sql: str, params: list) -> list:   # 대역: 읽기 전용 sqlite
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def bybit_bars(symbol: str, t0: int, t1: int, db: Path) -> dict[int, tuple[float, float, float, float]]:   # server.py 759~774 옮겨 적음
    if not db.exists():
        return {}
    try:
        rows = _read_only_rows(db, """SELECT ts_sec - ts_sec % 300, sum(buy_qty), sum(sell_qty), sum(whale_buy_qty - whale_sell_qty),
                                             sum(retail_buy_qty - retail_sell_qty) FROM trade_tape_1s
                                      WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? GROUP BY 1""", [symbol, t0, t1])
        gaps = _read_only_rows(db, "SELECT from_ms, to_ms FROM gaps WHERE symbol = ? AND from_ms < ? AND to_ms > ?",
                               [symbol, t1 * 1000, t0 * 1000])
    except (sqlite3.Error, OSError):
        return {}
    return {int(b): (float(bu), float(se), float(w or 0.0), float(rt or 0.0)) for b, bu, se, w, rt in rows
            if not any(f < (int(b) + 300) * 1000 and t > int(b) * 1000 for f, t in gaps)}


def fused_line(now, read, inp, card30):   # server.py 2494~2513 의 줄 만들기 옮겨 적음(쓰기·fused_bar 는 Rust 시험이 따로)
    fu = read.get("fused") or {}
    o3 = fu.get("outcome") or {}
    bar_now = int(now) // FOOTPRINT_BAR_SECONDS * FOOTPRINT_BAR_SECONDS
    if not o3:
        return None
    return json.dumps({"ts": int(now), "bar": bar_now, "side": fu.get("side"), "mid": inp.get("mid"),
                       "votes": fu.get("votes"), "score": fu.get("score"), "gate_pct": fu.get("gate_pct"),
                       "cell": [o3["reg"], o3["a"], o3["g"]],
                       "p": {c["key"]: c["p"] for c in o3["cols"]}, "dir_p": card30["p"].get("dir") if str(o3.get("dir_src", "")).startswith("model") else None,
                       "reach_p": card30["p"].get("reach") if o3.get("reach_src") == "model" else None,
                       "feat_bar": card30.get("bar"),   # 모델 피쳐의 마지막 봉 -- bar−300 이 아니면 낡은 값
                       "dir_src": o3.get("dir_src"), "up_share": o3.get("up_share"),
                       "up": o3["cols"][0].get("target"), "dn": o3["cols"][1].get("target")},
                      ensure_ascii=False) + "\n"


def _last_logged_bar(path: Path) -> int | None:   # server.py 2408~2416 옮겨 적음
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 16384))
            tail = fh.read().decode("utf-8", "ignore").strip().splitlines()
        return int(json.loads(tail[-1])["bar"]) if tail else None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


# ── 합성 ──
def cell():
    q = lambda: R.choice([0.0, round(R.uniform(0, 30), R.choice([3, 4])), R.uniform(0, 5)])   # noqa: E731
    return [q() for _ in range(6)]


def bar_cells(n=None):
    d = {}
    for k in R.sample(range(10000, 10040), n or R.randint(1, 5)):   # 원본 dict 차례 = 칸이 생긴 차례(가격순 아님)
        d[k] = cell()
    return d


def scales(p_missing=0.1, p_no30=0.2):
    out = {}
    for w in (0, 1):
        for h in range(24):
            if R.random() < p_missing:
                continue
            d = {k: R.uniform(5, 200) for k in ("mid", "whale", "retail")}
            if R.random() > p_no30:
                d.update({k + "30": R.uniform(3, 150) for k in ("mid", "whale", "retail")})
            out[(w, h)] = d
    return out


def run(name, steps, *, okx_from_step=0, n_bars=110, holes=3, bybit=True, late_every=5):
    global CARD
    cctx.clear(); flows.clear(); okx_bar_net_by.clear(); flow_all_hold.clear()
    now = 1_763_700_000.0 + R.randint(0, 86400 * 7)
    b0 = int(now) // 300 * 300 - n_bars * 300
    fb = {b0 + 300 * k: bar_cells() for k in range(n_bars) if k not in R.sample(range(5, n_bars - 20), holes)}
    ob: dict[int, dict] = {}
    fs, fsa = scales(), {VENUE_SETS[0]: scales(0.3), VENUE_SETS[1]: scales(0.1)}
    st = {"bybit_bars": {}, "candles_day": [], "candles": []}
    ms = {"payload": {"available": False}, "flow_scales": fs, "flow_scales_all": fsa}
    ring: list[float] = []
    cctx["eth"] = SimpleNamespace(situation_state=st, micro_state=ms, fr_bar_net={}, imb40_ring=ring)
    okx = {"first_bar": 0}
    flows["eth"] = SimpleNamespace(footprint_state={"bars": fb}, okx_fp=okx, okx_bars=ob)
    out = []
    px = R.uniform(2000, 4000)
    for s in range(steps):
        now += R.choice([1.0, 1.0, 3.5, 17.25, 60.0, 300.0, 299.0])
        cur = int(now) // 300 * 300
        if cur not in fb and R.random() < 0.9:
            fb[cur] = bar_cells()
        for b in sorted(fb)[-3:]:              # 최근 봉 늦은 칸(백필)
            if R.random() < 0.5:
                fb[b][R.randrange(9990, 10050)] = cell()
        if R.random() < 0.2:                   # 굳은 봉이 바뀌어도 원본은 다시 안 센다
            b = R.choice(sorted(fb)[:-5])
            fb[b][R.randrange(9990, 10050)] = cell()
        while len(fb) > n_bars + 4:            # 창 자르기 → fr_bar_net 지움
            del fb[min(fb)]
        if s == okx_from_step and okx_from_step >= 0:
            okx["first_bar"] = sorted(fb)[R.randrange(0, min(40, len(fb) - 8))]
        if okx["first_bar"]:
            for b in fb:
                if b > okx["first_bar"] - 600 and (b not in ob or R.random() < 0.1):
                    ob[b] = bar_cells()
            for b in [b for b in ob if b not in fb]:
                del ob[b]
        if bybit:
            done = [b for b in sorted(fb) if b < cur]
            st["bybit_bars"] = {b: (R.uniform(0, 300), R.uniform(0, 300), R.gauss(0, 50), R.gauss(0, 20)) for b in done
                                if R.random() < 0.97 and not (b == done[-1] and s % late_every == 0)}
        if R.random() < 0.1:
            ms["flow_scales"] = {} if R.random() < 0.5 else scales()
        n5 = R.choice([300, 300, 150, 155, 320, 90, 50])   # 299·154·319 마감봉 → 가상 위치 t = .5·.75·.0
        cs, c = [], px
        for k in range(n5):
            c *= 1 + R.gauss(0, 0.002)
            hi, lo = c * (1 + abs(R.gauss(0, 0.001))), c * (1 - abs(R.gauss(0, 0.001)))
            cs.append({"time": cur - 300 * (n5 - k - 1), "high": hi, "low": lo, "close": c, **({"taker": 1.0} if R.random() < 0.9 else {})})
        st["candles_day"] = cs if R.random() < 0.8 else []
        st["candles"] = cs[-135:]
        CARD = {k: v for k, v in (("dir", R.random()), ("reach", R.random())) if R.random() < 0.8}
        ib = [{"close": R.choice([None, 0.0, px * (1 + R.gauss(0, 0.003))]) if R.random() < 0.1 else px * (1 + R.gauss(0, 0.003)),
               "oi_delta": None if R.random() < 0.1 else R.gauss(0, 100)} for _ in range(R.choice([25, 13, 12]))]
        ring[:] = [R.uniform(-1, 1) for _ in range(R.choice([700, 600, 599]))]
        ms["payload"] = {"available": R.random() < 0.9, "ts": int(now), "mid": px, "agree": R.choice(["중립", "동조매수", "기준없음"]),
                         "trigger": R.choice([{"side": None, "ts": None, "age": None, "live": False}, {"side": "동조매수", "ts": 1, "age": 2, "live": True}]),
                         "imb40": R.choice([None, ring[R.randrange(len(ring))], R.uniform(-1, 1)]), "vol60_pct": R.choice([None, R.random()]),
                         "act": R.choice([None, "보통"]), "sr": R.choice([{}, {"sup": 1.0, "sup_bp": 3.0, "near": "없음"}]),
                         "liq60": R.choice([{}, {"long": 1.5, "short": 0.0, "n": 1}]), "spread": 0.01, "sweep": None}
        x = _flow_read_ctx(now, {"bars": ib}, "eth")
        out.append({"now": now,
                    "fbars": [[b, list(c.values())] for b, c in fb.items()],
                    "okx_first_bar": okx["first_bar"], "okx_bars": [[b, list(c.values())] for b, c in ob.items()],
                    "bybit": [[b, list(v)] for b, v in st["bybit_bars"].items()],
                    "flow_scales": [[list(k), v] for k, v in ms["flow_scales"].items()],
                    "flow_scales_all": [[",".join(vs), [[list(k), v] for k, v in m.items()]] for vs, m in fsa.items()],
                    "bars": [[b["close"], b["oi_delta"]] for b in ib],
                    "day5": [[c["time"], c["high"], c["low"], c["close"]] for c in (st.get("candles_day") or st.get("candles") or [])],
                    "card30": [CARD.get("dir"), CARD.get("reach")] if _card30_bars(now, st) else [None, None],
                    "micro": ms["payload"], "ring": list(ring), "x": json.dumps(x)})
    (OUT / f"x_{name}.json").write_text(json.dumps(out))
    return len(out)


n = run("okx_late", 60, okx_from_step=0, n_bars=150, holes=2)
n += run("okx_mid", 50, okx_from_step=20, holes=6)
n += run("no_okx", 30, okx_from_step=-1, bybit=False, n_bars=150, holes=0)
n += run("short", 20, n_bars=30, holes=1)
n += run("twelve", 12, n_bars=11, holes=0)
n += run("grow", 30, n_bars=83, holes=0)

# 장부 한 줄: flow_read 결과 위에서
lines = []
for i in range(600):
    ev = {"dir": R.choice([-1, 0, 1]), "veto": R.choice([-1, 0, 1]), "reject": R.random() < 0.3, "mid": 3000.0 + i, "range_bp": R.choice([0.0, 40.5])}
    x = {"z60": {"whale": R.gauss(0, 1), "mid": R.gauss(0, 1), "retail": R.gauss(0, 1)}, "move60": R.gauss(0, 30), "oi60": R.gauss(0, 500),
         "move60_p75": 20.0, "range30_pct": R.choice([None, R.random(), 0.9]), "dir_p": R.choice([None, R.random()]), "reach_p": R.choice([None, R.random()])}
    rd = fr.read(ev, x) if R.random() < 0.95 else {"lines": [], "summary": "", "error": "x"}
    card = {"p": {k: R.random() for k in ("dir", "reach") if R.random() < 0.8}, "bar": R.choice([None, 1763700000])}
    now = 1_763_700_000 + R.uniform(0, 1e6)
    inp = {"mid": R.choice([None, 3000.25])}
    lines.append({"now": now, "read": rd, "mid": inp["mid"], "card_p": card["p"], "feat_bar": card["bar"], "line": fused_line(now, rd, inp, card)})
(OUT / "fused.json").write_text(json.dumps(lines, ensure_ascii=False))

# _last_logged_bar 꼬리
tails = []
for k, body in enumerate([b"", b"\n\n", b'{"bar": 1763700300}\n', b'{"bar": 17}\n{"bar": 1763700600}\n   \n', ("x" * 20000 + '\n{"bar": 5, "t": "한글"}\n').encode(),
                          b'{"bar": "12"}\n', b'{"bar": null}\n', b'{"bar": 3.9}\n', b'{"nobar": 1}\n', b'not json\n',
                          ('{"a": "' + "가" * 6000 + '"}\n{"bar": 99}').encode(), b'{"bar": true}\n', b'{"bar": 7}\xff\n']):
    p = OUT / f"tail_{k}.jsonl"
    p.write_bytes(body)
    tails.append({"file": p.name, "bar": _last_logged_bar(p)})
tails.append({"file": "missing.jsonl", "bar": _last_logged_bar(OUT / "missing.jsonl")})
(OUT / "tails.json").write_text(json.dumps(tails))

# bybit_bars · 기준선(SQLite)
db = OUT / "tape.sqlite"
con = sqlite3.connect(db)
con.execute("CREATE TABLE trade_tape_1s (ts_sec INTEGER, symbol TEXT, buy_qty REAL, sell_qty REAL, whale_buy_qty REAL, whale_sell_qty REAL, "
            "retail_buy_qty REAL, retail_sell_qty REAL)")
con.execute("CREATE TABLE gaps (symbol TEXT, from_ms INTEGER, to_ms INTEGER)")
NOW = 1_763_740_000
today = NOW // 86400 * 86400
rows = []
for sym in ("ETHUSDT", "ethusdt", "ETH-USDT-SWAP"):
    for b in range(today - 16 * 86400, NOW, 300):
        if R.random() < 0.002:
            continue
        for _ in range(R.choice([1, 2, 3])):
            t = b + R.randrange(300)
            w = None if (b < today - 15 * 86400 and R.random() < 0.5) else R.uniform(0, 20)
            rows.append((t, sym, R.uniform(0, 50), R.uniform(0, 50), w, None if w is None else R.uniform(0, 20), R.uniform(0, 10), R.uniform(0, 10)))
con.executemany("INSERT INTO trade_tape_1s VALUES (?,?,?,?,?,?,?,?)", rows)
con.executemany("INSERT INTO gaps VALUES (?,?,?)", [("ETHUSDT", (NOW - 3000) * 1000, (NOW - 2950) * 1000), ("ETHUSDT", (NOW - 9000) * 1000 + 1, (NOW - 8700) * 1000)])
con.commit(); con.close()
bb = {"t0": NOW - 26 * 3600, "t1": (NOW - 10) // 300 * 300, "out": [[b, list(v)] for b, v in sorted(bybit_bars("ETHUSDT", NOW - 26 * 3600, (NOW - 10) // 300 * 300, db).items())]}
time.time = lambda: float(NOW)   # 원본 함수들이 «오늘»을 time.time() 으로 잰다 -- 고정
fs1 = mref.flow_hour_scales(db, symbol="ethusdt", since_ts=0)
fs2 = mref.flow_hour_scales(db, symbol="ethusdt", since_ts=today - 9 * 86400)
combo = mref.flow_hour_scales_combos({"binance": (db, "ethusdt"), "okx": (db, "ETH-USDT-SWAP"), "bybit": (db, "ETHUSDT")}, list(VENUE_SETS))
combo_nb = mref.flow_hour_scales_combos({"binance": (db, "ethusdt"), "okx": (db, "ETH-USDT-SWAP"), "bybit": (OUT / "none.sqlite", "ETHUSDT")}, list(VENUE_SETS))
enc = lambda d: [[list(k), v] for k, v in d.items()]   # noqa: E731
(OUT / "tape.json").write_text(json.dumps({"now": NOW, "bybit": bb, "fs": enc(fs1), "fs_since": enc(fs2), "since": today - 9 * 86400,
                                           "combo": [[",".join(c), enc(m)] for c, m in combo.items()],
                                           "combo_nb": [[",".join(c), enc(m)] for c, m in combo_nb.items()]}))
print(f"x 걸음 {n} · 장부 {len(lines)} · 꼬리 {len(tails)} · 기준선 칸 {len(fs1)}/{len(fs2)} · 조합 {[len(m) for m in combo.values()]} → {OUT}")
