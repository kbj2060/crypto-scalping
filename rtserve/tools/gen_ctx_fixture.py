#!/usr/bin/env python3
"""rtserve 상황 카드(src/ctx.rs) 일치 시험의 입력·정답지 (2026-11-22).

정답 = **원본 dashboard/flow_read.py 를 그대로 부른 값**: fr.read(ev, x)(안에서 fuse · outcome3 · gauges) · fr.vol_size_mult(bars, asset).
server.py 는 import 할 수 없다(make_app 안 닫힌 함수) -- 옮겨 적은 것(Read 로 본 줄):
  · situation_payload  = server.py 2734~2738 (아래 situation_payload)
  · SSE 상황 한 건      = server.py api_stream 3175~3179 (`event: situation\\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\\n\\n`)
  · /api/situation 본문 = server.py 2740~2741 web.json_response(…) = json.dumps 기본(ensure_ascii=True · ', ' ': ')
  · 계산 전 상태        = server.py _coin_ctx 1777 {"now": {"ok": False, "reason": "계산 전"}, "computed_at": 0.0}
입력은 합성: ev = situation.classify evidence 꼴(rtcore situation.rs Evidence 와 같은 칸 · cvd·liq 는 정수) · x = server.py _flow_read_ctx 꼴.
경계값(문턱 정확히 · 동률 반올림 · None · NaN 없음)을 일부러 섞는다. 네트워크 없음.
🔴서버 판 파이썬으로 돌린다(numpy 2.2.5 -- vol_size_mult 의 np.sum·np.log 가 서버와 같아야 한다):
    source <(bash rtmodel/tools/server_env.sh <venv> --env) && $PY rtserve/tools/gen_ctx_fixture.py   # 저장소 루트에서
출력: rtserve/tests/fixtures/gen/ctx/ (커밋 안 함 · 시험 전에 만든다)
"""
from __future__ import annotations

import json
import math
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rtserve" / "tests" / "fixtures" / "gen" / "ctx"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
sys.path.insert(0, str(ROOT))
import dashboard.flow_read as fr  # noqa: E402

R = random.Random(20261122)


def situation_payload(situation_state, fo_state, mp_state, situation_by):   # server.py 2734~2738 옮겨 적음
    return {"now": situation_state["now"], "computed_at": situation_state["computed_at"],
            "streams": {"fo": dict(fo_state), "mp": dict(mp_state)},
            "read": situation_state.get("read"), "vol_mult": situation_state.get("vol_mult"),
            "vol_mult_by": {a: s.get("vol_mult") for a, s in situation_by.items()}}


def sse(data):   # server.py 3175~3179 옮겨 적음
    return f"event: situation\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def maybe(p, f):
    return f() if R.random() < p else None


def pick(*vals):
    return R.choice(vals)


def num(scale, edges=()):
    """보통 값 · 반올림 동률(x.5) · 정해진 경계 중 하나."""
    u = R.random()
    if edges and u < 0.25:
        return R.choice(edges)
    if u < 0.4:
        return R.randint(-20, 20) * 0.5 * scale / 10   # .5 동률
    if u < 0.45:
        return 0.0
    return R.gauss(0, scale)


def ev_case():
    d = pick(-1, 0, 1, 0)
    ev = {"mid": maybe(0.95, lambda: pick(3000.0, 2999.5, 3123.45, R.uniform(1000, 5000))),
          "move_bp": round(num(30, (-12.5, 12.5, 0.0)), 1), "range_bp": round(abs(num(40, (0.0, 50.0))), 1), "dir": d,
          "move_ratio": round(R.random(), 3), "cvd": R.choice([0, R.randint(-3000, 3000), R.randint(-90000, 90000), 1500, -999, 10000, -10000, 1000, 999999, 1000000]),
          "reject": R.random() < 0.3, "liq_long": R.choice([0, R.randint(0, 80000), R.randint(0, 3_000_000)]),
          "liq_short": R.choice([0, R.randint(0, 80000), R.randint(0, 3_000_000), 25000]),
          "veto": pick(-1, 0, 1), "obi": maybe(0.7, lambda: round(R.uniform(-1, 1), R.choice([2, 3, 6]))),
          "persist": maybe(0.6, lambda: pick(0.5, 0.125, R.random())), "funding": maybe(0.8, lambda: pick(0.0001, -0.00005, R.gauss(0, 3e-4))),
          "basis_bp": maybe(0.8, lambda: num(5)), "basis_d_bp": maybe(0.7, lambda: num(3, (0.25, -0.25))),
          "basis_thr_bp": maybe(0.8, lambda: pick(0.0, 1.5, R.uniform(0.5, 3))), "crowd": pick(-1, 0, 1), "lead": pick(-1, 0, 1, 0),
          "btc_move_bp": maybe(0.8, lambda: round(num(20), 1)), "btc_rel": pick(None, "동행", "단독")}
    if R.random() < 0.1:
        for k in ("liq_long", "liq_short"):   # 청산 키 없음(워밍업)
            ev.pop(k)
    if R.random() < 0.05:
        ev.pop("move_bp")
    if R.random() < 0.05:
        ev.pop("dir")
    return ev


def x_case():
    x = {}
    if R.random() < 0.8:
        x["okx30"] = maybe(0.8, lambda: num(800))
        x["cvd30_z"] = maybe(0.8, lambda: num(1.2, (1.0, -1.0, 3.5)))
    if R.random() < 0.8:
        x["net60"] = R.choice([{}, {"whale": num(2000), "mid": num(1500), "retail": num(1000)}, {"whale": 0.0, "mid": 1.0, "retail": 0.0}])
        x["z60"] = R.choice([None, {}, {"whale": num(1, (0.5, -0.5)), "mid": num(1, (0.5, -0.5)), "retail": num(1, (0.5, -0.5))}])
    if R.random() < 0.85:
        x["move60"] = maybe(0.9, lambda: num(40, (-30.0, 30.0)))
        x["oi60"] = maybe(0.9, lambda: R.choice([0.0, num(3000), 12345.5]))
        x["move60_p75"] = maybe(0.85, lambda: pick(0.0, 30.0, R.uniform(5, 60)))
    if R.random() < 0.7:
        x["liq60s"] = R.choice([{}, {"long": R.uniform(0, 1e5), "short": 0.0}, {"long": 0.0, "short": 0.0}, {"long": 1234.5, "short": 2500.5}])
    if R.random() < 0.85:
        x["imb40_pct"] = maybe(0.8, lambda: pick(0.8, 0.2, 0.5, R.random(), 0.995, 0.005))
        x["imb40"] = maybe(0.9, lambda: round(R.uniform(-0.6, 0.6), R.choice([2, 3, 4])))
    x["agree"] = pick(None, "동조매수", "동조매도", "엇갈림")
    x["trigger_live"] = pick(True, False)
    if R.random() < 0.75:
        sup_bp, res_bp = maybe(0.7, lambda: pick(15.0, 2.5, R.uniform(0, 120))), maybe(0.7, lambda: pick(15.0, 7.5, R.uniform(0, 120)))
        x["sr"] = {"sup": 2950.25 if sup_bp is not None else None, "sup_bp": sup_bp, "res": 3040.75 if res_bp is not None else None,
                   "res_bp": res_bp, "near": pick(None, "", "없음", "지지근접", "저항근접"), "deep_wall": pick(None, "", "레벨쪽", "반대쪽")}
    x["act"] = pick(None, "", "활발", "조용")
    x["vol_pct"] = maybe(0.8, lambda: pick(0.8, 0.2, 0.5, R.random(), 0.125))
    x["range30_pct"] = maybe(0.9, lambda: pick(2 / 3, 1 / 3, 0.66, 0.5, R.random(), 0.995))
    x["dir_p"] = maybe(0.6, lambda: pick(0.0, 1.0, 0.5, R.random()))
    x["reach_p"] = maybe(0.6, lambda: pick(0.0, 1.0, 0.25, R.random()))
    return x


def bars_case():
    n = R.choice([200, 289, 289, 300, 400])
    t0, c = 1_700_000_100 // 300 * 300, R.uniform(1000, 4000)
    out = []
    for i in range(n):
        c *= math.exp(R.gauss(0, R.choice([0.0005, 0.002, 0.01])))
        out.append({"time": t0 + 300 * i, "close": c})
    if R.random() < 0.15:
        out[R.randrange(n - 289, n) if n >= 289 else 0]["time"] += 300   # 끊김
    return out, R.choice(["eth", "eth", "xrp", "sol"])


reads, vols, pays = [], [], []
for i in range(4000):
    ev, x = ev_case(), x_case()
    out = fr.read(ev, x)
    reads.append({"ev": ev, "x": x, "dflt": json.dumps(out), "sse": json.dumps(out, ensure_ascii=False, separators=(",", ":"))})
for i in range(400):
    b, a = bars_case()
    vols.append({"bars": [[x["time"], x["close"]] for x in b], "asset": a, "out": json.dumps(fr.vol_size_mult(b, a))})
for i in range(300):
    ev, x = ev_case(), x_case()
    st = {"now": {"ok": False, "reason": "계산 전"}, "computed_at": 0.0} if i == 0 else {
        "now": {"ok": True, "reason": None, "dir": ev.get("dir"), "labels": ["상승 +12bp"], "evidence": ev,
                "regime": {"ratio": 0.5, "thr": 0.35, "margin": 0.15, "enter": 0.5, "exit": 0.35}, "computed_at": 1763700000.25 + i},
        "computed_at": 1763700000.25 + i}
    if i % 3 == 1:
        st["read"] = fr.read(ev, x)
    elif i % 3 == 2:
        st["read"] = {"lines": [], "summary": "", "error": "KeyError('whale')"}
    if i % 2:
        b, a = bars_case()
        st["vol_mult"] = fr.vol_size_mult(b, "eth")
    fo = {"connected": i % 2 == 0, "since": None if i % 5 == 0 else 1763699000.5, "last_event_ms": 1763700000123, "events": i,
          "errors": 0, "last_error": None if i % 7 else "ClientError('x \"q\"')"}
    mp = {"connected": True, "since": 1763699000.5, "last_ms": 1763700000000, "events": 10 * i, "errors": 1, "last_error": None,
          "next_funding_ms": 1763712000000}
    by = {"eth": st, "xrp": {"vol_mult": None if i % 2 else {"sigma_bp": 1.5, "mult": 0.5}}}
    p = situation_payload(st, fo, mp, by)
    pays.append({"state": st, "fo": fo, "mp": mp, "by": [[a, s.get("vol_mult")] for a, s in by.items()],
                 "dflt": json.dumps(p), "sse": sse(p)})
for name, v in (("read", reads), ("vol", vols), ("payload", pays)):
    (OUT / f"{name}.json").write_text(json.dumps(v, ensure_ascii=False))
print(f"read {len(reads)} · vol {len(vols)} · payload {len(pays)} → {OUT}")
