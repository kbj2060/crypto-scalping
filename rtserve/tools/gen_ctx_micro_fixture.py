#!/usr/bin/env python3
"""rtserve 미시 참고 payload(src/ctx_micro.rs) 일치 시험의 입력·정답지 (2026-11-22).

정답 = 원본 함수. import 해서 그대로 부르는 것: scripts/live_orderflow_raster_collector_20260914.py read_window(파일 형식도 그 모듈의
HEADER·ROW_HEAD·_hour_path 로 쓴다) · dashboard/micro_ref.py raster_flow·qi·agree_state·pct_rank·act_label·sr_context·baseline_from_tape ·
dashboard/market_ctx.py sweep_depth.
server.py collect_micro_ref 의 한 틱(1955~2008)은 make_app 안이라 **옮겨 적었다**(아래 micro_tick -- Read 로 본 줄 그대로).
바꾼 곳 셋만: ① REST bookTicker(1955 fetch_binance_json) 자리에 정해 둔 호가 dict · ② read_window 에 root= (시험 폴더) ·
③ load_liquidation_map(1991) 자리에 정해 둔 레벨 dict · vol60·liq60 은 원본 식(1985~1987 · 1996~1999)으로 합성 1초 칸·청산에서 센다.
기준선 갱신 분기(1926~1941)도 옮겨 적었다(baseline_tick).
🔴서버 판 파이썬으로(numpy 2.2.5 -- float32 합 차례):
    source <(bash rtmodel/tools/server_env.sh <venv> --env) && $PY rtserve/tools/gen_ctx_micro_fixture.py   # 저장소 루트에서
출력: rtserve/tests/fixtures/gen/ctx_micro/ (커밋 안 함)
"""
from __future__ import annotations

import json
import math
import random
import shutil
import sqlite3
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rtserve" / "tests" / "fixtures" / "gen" / "ctx_micro"
shutil.rmtree(OUT, ignore_errors=True)
(OUT / "raster").mkdir(parents=True)
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import dashboard.micro_ref as mref  # noqa: E402
import dashboard.market_ctx as mctx  # noqa: E402
import scripts.live_orderflow_raster_collector_20260914 as rc  # noqa: E402

R = random.Random(20261124)
RASTER = OUT / "raster"
SYM = "ETHUSDT"
MICRO_BASELINE_SECONDS = 3600   # server.py 467 «1시간마다»


def read_window(symbol, to_ms, cols, agg=1):   # 원본 호출 모양(1962) + 시험 폴더
    return rc.read_window(symbol, to_ms, cols, agg, root=RASTER)


def micro_tick(now, micro_state, mid_ring, ofi_hist, imb40_ring, depth25_ring, book, levels, vol60, liq60):
    """server.py collect_micro_ref 1955~2008 옮겨 적음(book·levels·vol60·liq60 는 인자로)."""
    now_sec = int(now)
    FOOTPRINT_SYMBOL = SYM   # noqa: N806
    qi_val = mref.qi(float(book["bidQty"]), float(book["askQty"]))
    mid = (float(book["bidPrice"]) + float(book["askPrice"])) / 2.0
    mid_ring[now_sec] = mid
    for old in [sc for sc in mid_ring if sc < now_sec - 600]:   # HL 표본(30초 수집 + 15초 캐시)보다 넉넉히 10분
        del mid_ring[old]
    w = read_window(FOOTPRINT_SYMBOL.lower(), int(now * 1000), 12, 1)
    flow = mref.raster_flow(w)
    sweep = None
    if flow.get("ok"):
        ofi_hist.append(abs(flow["ofi10"]))
        imb40_ring.append(float(flow["imb40"]))
        # 시장 맥락: 미드에서 ±25/50/100bp 까지 걸린 호가 = «그만큼 밀려면 먹어야 할 양». 같은 마지막 유효 초.
        q = np.asarray(w["qty"]); mids = np.asarray(w["mid"], dtype=float)
        last = int(np.flatnonzero(np.isfinite(mids))[-1])
        sweep = mctx.sweep_depth(q[last], (int(w["bin_lo"]) + np.arange(q.shape[1])) * float(w["bin_size"]), flow["mid"],
                                 top=(float(book["bidPrice"]), float(book["bidQty"]), float(book["askPrice"]), float(book["askQty"])))
        depth25_ring.append((sweep["bid"][0], sweep["ask"][0]))       # ±25bp = SWEEP_BPS[0]
    ofi_thr = float(np.median(ofi_hist)) if len(ofi_hist) >= 30 else None
    if flow.get("ok") and ofi_thr:
        agree, micro_state["qi_prev"] = mref.agree_state(
            qi_val, flow["ofi10"], ofi_thr, micro_state["qi_prev"])
    else:
        agree = "기준없음"
    if agree in ("동조매수", "동조매도"):
        micro_state["trigger"] = {"side": agree, "ts": now_sec}
    trig = dict(micro_state["trigger"])
    trig["age"] = (now_sec - trig["ts"]) if trig["ts"] else None
    trig["live"] = bool(trig["age"] is not None and trig["age"] <= mref.TRIGGER_LIVE_S)
    hour = int((now_sec % 86400) // 3600)
    base = micro_state["baseline"]
    vol60_pct = mref.pct_rank(vol60, base["minute_vol_sorted"][hour]) if base else None
    sr = mref.sr_context(levels if levels.get("warmed_up") else None, mid, flow.get("imb40") if flow.get("ok") else None)
    if levels.get("warmed_up"):   # 상황 읽기의 플러시 목표(청산 군집)용 -- 가까운 순 [{price, weight_pct}]
        sr["sup_levels"] = [{"price": lv["price"], "weight_pct": lv.get("weight_pct")} for lv in levels.get("support_levels") or []]
        sr["res_levels"] = [{"price": lv["price"], "weight_pct": lv.get("weight_pct")} for lv in levels.get("resistance_levels") or []]
    micro_state["tick"] = micro_state.get("tick", 0) + 1
    micro_state["payload"] = {
        "available": True, "ts": now_sec, "mid": mid,
        "agree": agree, "trigger": trig,
        "imb40": flow.get("imb40") if flow.get("ok") else None,
        "vol60_pct": vol60_pct, "act": mref.act_label(vol60_pct),
        "sr": sr, "liq60": liq60,
        "spread": float(book["askPrice"]) - float(book["bidPrice"]), "sweep": sweep,
    }
    return micro_state["payload"]


def baseline_tick(now, micro_state, got):   # server.py 1926~1941 옮겨 적음(got = baseline_from_tape 결과 · 예외 = None)
    if now - micro_state["baseline_at"] >= (MICRO_BASELINE_SECONDS if micro_state["baseline"] else 60.0):
        micro_state["baseline_at"] = now
        if got is not None:
            micro_state["baseline"] = got
        else:
            if micro_state["baseline"]:
                micro_state["baseline_at"] = now - MICRO_BASELINE_SECONDS + 60.0
        return True
    return False


# ── 래스터 파일(수집기 형식 그대로 · 모듈의 HEADER/ROW_HEAD) ──
T0 = 1_763_701_200_000 - 600_000   # 시 경계 10분 전부터(두 파일에 걸친다)
NB, BIN = 240, 0.5
files: dict[Path, tuple[int, dict[int, bytes]]] = {}
px0 = 3000.0
for s in range(1500):
    t = T0 + s * 1000
    if R.random() < 0.03:
        continue                                  # 안 쓴 초 = 0 으로 읽힘(구멍)
    px0 += R.gauss(0, 0.4)
    lo = int(px0 // BIN) - NB // 2 + R.randint(-3, 3)
    mid = float("nan") if (R.random() < 0.05 or 700 <= s < 730) else px0 + R.uniform(-0.25, 0.25)
    qty = np.zeros(NB, dtype=np.float32)
    for k in range(NB):
        p = (lo + k) * BIN
        if R.random() < 0.7:
            v = np.float32(R.choice([R.uniform(0, 50), round(R.uniform(0, 400), 3), 1e-3]))
            qty[k] = v if p < px0 else -v
    hour = t - t % 3_600_000
    path = rc._hour_path(SYM, hour, RASTER)
    row = rc.ROW_HEAD.pack(t, lo, np.float32(mid)) + qty.tobytes()
    files.setdefault(path, (hour, {}))[1][(t - hour) // 1000] = row
for path, (hour_ms, rows) in files.items():
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(rc.HEADER.pack(rc.MAGIC, rc.VERSION, NB, BIN, 0, hour_ms, 1000, 0))
        rb = rc.row_bytes(NB)
        for j in sorted(rows):
            fh.seek(rc.HEADER.size + j * rb)
            fh.write(rows[j])
# 한 시간은 칸 수가 다른 파일(원본은 첫 파일 칸 수와 다르면 건너뛴다)·깨진 머리 파일도 하나씩
odd = rc._hour_path(SYM, T0 - T0 % 3_600_000 + 2 * 3_600_000, RASTER)
odd.write_bytes(rc.HEADER.pack(rc.MAGIC, rc.VERSION, NB + 8, BIN, 0, 0, 1000, 0) + rc.ROW_HEAD.pack(T0, 1, 1.0) + b"\0" * 4 * (NB + 8))
bad = rc._hour_path(SYM, T0 - T0 % 3_600_000 + 3 * 3_600_000, RASTER)
bad.write_bytes(b"XXXX" + b"\0" * 28)
# BTC: 앞 시는 보통 파일 · 뒤 시는 칸 수가 다른 파일(원본은 첫 파일 칸 수와 다르면 건너뛴다) -- read_window 직접 대조용
H0 = T0 - T0 % 3_600_000
for hh, nb in ((H0, NB), (H0 + 3_600_000, NB + 8)):
    p_ = rc._hour_path("BTCUSDT", hh, RASTER); p_.parent.mkdir(parents=True, exist_ok=True)
    with open(p_, "wb") as fh:
        fh.write(rc.HEADER.pack(rc.MAGIC, rc.VERSION, nb, BIN, 0, hh, 1000, 0))
        for j in range(3590 if hh == H0 else 0, 3600 if hh == H0 else 10):
            q_ = np.array([R.uniform(-5, 5) for _ in range(nb)], dtype=np.float32)
            fh.seek(rc.HEADER.size + j * rc.row_bytes(nb)); fh.write(rc.ROW_HEAD.pack(hh + j * 1000, 6000 + j % 3, np.float32(3000.25)) + q_.tobytes())

# ── 기준선 테이프 ──
db = OUT / "tape.sqlite"
con = sqlite3.connect(db)
con.execute("CREATE TABLE trade_tape_1s (ts_sec INTEGER, symbol TEXT, buy_qty REAL, sell_qty REAL)")
NOW0 = T0 // 1000
rows = [(t, "ethusdt", R.uniform(0, 30) / 7, R.uniform(0, 30) / 3) for t in range(NOW0 - 7 * 86400 + 5, NOW0, 7) if R.random() < 0.5]
rows += [(t, "ethusdt", 0.1, 0.2) for t in range(NOW0 - 3000, NOW0, 61)]
con.executemany("INSERT INTO trade_tape_1s VALUES (?,?,?,?)", rows)
con.commit(); con.close()
edge = OUT / "edge.sqlite"   # 분 599개(원본 None) · 600개는 tape 가 넉넉히 넘는다
con = sqlite3.connect(edge); con.execute("CREATE TABLE trade_tape_1s (ts_sec INTEGER, symbol TEXT, buy_qty REAL, sell_qty REAL)")
con.executemany("INSERT INTO trade_tape_1s VALUES (?,?,?,?)", [(NOW0 - 60 * k - 1, "ethusdt", 1.0, 1.0) for k in range(599)]); con.commit(); con.close()
small = OUT / "small.sqlite"
con = sqlite3.connect(small); con.execute("CREATE TABLE trade_tape_1s (ts_sec INTEGER, symbol TEXT, buy_qty REAL, sell_qty REAL)")
con.executemany("INSERT INTO trade_tape_1s VALUES (?,?,?,?)", [(NOW0 - 100 * k, "ethusdt", 1.0, 1.0) for k in range(50)]); con.commit(); con.close()
time.time = lambda: float(NOW0)
base = mref.baseline_from_tape(db, symbol="ethusdt")
base_small = mref.baseline_from_tape(small, symbol="ethusdt")
base_other = mref.baseline_from_tape(db, symbol="solusdt")
base_edge = mref.baseline_from_tape(edge, symbol="ethusdt")

# ── 틱 ──
levels_set = [{}, {"warmed_up": False, "support_levels": [{"price": 2990.0}]},
              {"warmed_up": True, "support_levels": [{"price": 3005.0, "weight_pct": 12.5}, {"price": 2995.5, "weight_pct": 3.0}, {"price": "2980.0"}],
               "resistance_levels": [{"price": 2999.0}, {"price": 3008.25, "weight_pct": 40.0}]},
              {"warmed_up": True, "support_levels": [], "resistance_levels": [{"price": 3100.0, "weight_pct": 1.0}]}]
ms = {"payload": {"available": False}, "baseline": None, "baseline_at": 0.0, "qi_prev": "중립", "trigger": {"side": None, "ts": None}}
mid_ring: dict = {}
ofi_hist, imb40_ring, depth25_ring = deque(maxlen=600), deque(maxlen=6 * 3600), deque(maxlen=6 * 3600)
ticks = []
now = T0 / 1000 + 3.0
gots = [None, base, None, base, base]
gi = 0
while len(ticks) < 300:
    got = gots[gi % len(gots)] if R.random() < 0.3 else None
    did = baseline_tick(now, ms, got)
    if did:
        gi += 1
    m = 3000.0 + R.gauss(0, 5)
    spread = R.choice([0.01, 0.5, 1.0])
    bk = {"bidPrice": f"{m:.2f}", "bidQty": f"{R.uniform(0, 300):.3f}", "askPrice": f"{m + spread:.2f}", "askQty": f"{R.uniform(0, 300):.3f}"}
    lv = R.choice(levels_set)
    vol60 = R.choice([R.uniform(0, 3000), 0.0])
    liq60 = {"long": R.choice([0.0, R.uniform(0, 1e5)]), "short": 0.0, "n": R.randint(0, 3)}
    p = micro_tick(now, ms, mid_ring, ofi_hist, imb40_ring, depth25_ring, bk, lv, vol60, liq60)
    ticks.append({"now": now, "book": [float(bk["bidPrice"]), float(bk["bidQty"]), float(bk["askPrice"]), float(bk["askQty"])], "levels": lv,
                  "vol60": vol60, "liq60": liq60, "baseline_did": did, "baseline_got": None if got is None else ("base" if got is base else "?"),
                  "payload": json.dumps(p), "depth25_last": list(depth25_ring[-1]) if depth25_ring else None, "imb40_n": len(imb40_ring),
                  "mid_ring_n": len(mid_ring), "baseline_at": ms["baseline_at"]})
    now += R.choice([1.0, 1.0, 1.0, 0.4, 3.7, 30.0]) + (3601.0 if len(ticks) in (240, 250, 255) else 0.0)   # 기준선 주기 넘김(있는 기준선 + 실패 → 60초 뒤)
enc = lambda b: None if b is None else [[h, v] for h, v in sorted(b["minute_vol_sorted"].items())]   # noqa: E731
(OUT / "micro.json").write_text(json.dumps({"t0": T0, "now0": NOW0, "ticks": ticks, "base": enc(base), "base_small": enc(base_small), "base_other": enc(base_other), "base_edge": enc(base_edge)}))
# 개별 함수: sr_context 동률·hyst·pct_rank 경계
cases = []
for _ in range(3000):
    mid = R.choice([3000.0, 0.0, -1.0, 2999.99])
    lv = R.choice(levels_set + [{"support_levels": [{"price": mid - 0.45}], "resistance_levels": [{"price": mid + 0.45}]}])
    imb = R.choice([None, 0.3, -0.2, 0.0])
    cases.append({"mid": mid, "levels": lv, "imb40": imb, "sr": json.dumps(mref.sr_context(lv, mid, imb))})
hy = []
prev = "중립"
for _ in range(3000):
    q = R.choice([0.75, -0.75, 0.45, -0.45, 0.449, R.uniform(-1, 1), float("nan")])
    o = R.choice([0.0, 5.0, -5.0, R.gauss(0, 10)])
    thr = R.choice([5.0, 3.3])
    a, prev2 = mref.agree_state(q, o, thr, prev)
    hy.append({"q": None if math.isnan(q) else q, "nan": math.isnan(q), "o": o, "thr": thr, "prev": prev, "out": [a, prev2]})
    prev = prev2
pr = []
for _ in range(500):
    n = R.choice([19, 20, 50])
    a = sorted(R.choice([1.0, 2.0, R.uniform(0, 10)]) for _ in range(n))
    x = R.choice(a + [R.uniform(-1, 11), float("inf")])
    pr.append({"a": a, "x": None if math.isinf(x) else x, "inf": math.isinf(x), "out": mref.pct_rank(x, a)})
wins = []
for sym, to in (("btcusdt", H0 + 3_600_000 + 5_000), ("btcusdt", H0 + 3_599_000), ("ethusdt", H0 + 3_600_000 + 2_000), ("ethusdt", H0 - 50_000_000), ("ethusdt", T0 + 715_000)):
    w = rc.read_window(sym, to, 12, 1, root=RASTER)
    wins.append({"sym": sym, "to": to, "bin_size": w["bin_size"], "bin_lo": w["bin_lo"], "mid": [None if math.isnan(x) else float(x) for x in w["mid"]],
                 "qty": np.asarray(w["qty"]).astype(float).tolist()})
(OUT / "funcs.json").write_text(json.dumps({"sr": cases, "agree": hy, "pct": pr, "windows": wins}))
print(f"틱 {len(ticks)} · 래스터 파일 {len(files) + 2} · sr {len(cases)} · agree {len(hy)} · pct {len(pr)} · 기준선 {'있음' if base else '없음'} → {OUT}")
