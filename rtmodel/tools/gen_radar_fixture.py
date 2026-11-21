"""실시간 위험 레이더(scripts/live_risk_radar_20261115.py) «1분봉 캐시·DVOL 캐시 → 4코인 σ̂·띠·앞 1시간·급변 → latest.json·forward.jsonl» 정답지
(2026-11-21 · tests/risk_radar.rs).

정답은 원본 워커의 grid·dvol_series·compute 를 그대로 부른다(연구 research_realtime_risk_radar_20261115 의 features·design·wsum·iv_* 포함).
안전(소스 확인): 워커는 import 때 binance_ban_guard(requests 를 감싸기만 · 접속 없음)와 연구 모듈 둘만 부른다. REST(rest·dvol_rest)는 함수 안 --
여기서는 update_cache·update_dvol·main 을 부르지 않는다. main() 의 «코인 결과 → 파일 두 개» 조립(:221-240)만 아래 assemble() 에 같은 식으로 적었다.
입력 = 워커 캐시 꼴(1분봉 t·h·l·c·v, DVOL ts·close·got) 합성. 계수 = 저장소 dashboard/risk_radar_params_20261116.json. SOXL(세션 식)은 아직 안 옮겨 뺐다.

    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL" python rtmodel/tools/gen_radar_fixture.py     # → tests/fixtures/radar_fixture.json (커밋 안 함)
"""
import datetime as dt
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_risk_radar_20261115 as W  # noqa: E402

OUT = ROOT / "rtmodel" / "tests" / "fixtures"
MIN = 60_000
PX = {"ETH": 3000.0, "BTC": 90000.0, "SOL": 150.0, "XRP": 1.3, "SOXL": 30.0}
TICK = {"ETH": 0.01, "BTC": 0.1, "SOL": 0.01, "XRP": 0.0001, "SOXL": 0.01}


def bars(rng, coin, end_ms, n, sig, gaps=0, zero_v=0, spikes=(), flat=0, gap_day=0):
    t = end_ms - MIN * np.arange(n, 0, -1)
    r = rng.normal(0, sig, n) * np.exp(rng.normal(0, .4, n))
    for k, s in spikes:
        r[n - k] += s
    c = np.round(PX[coin] * np.exp(np.cumsum(r)) / TICK[coin]) * TICK[coin]
    if flat:
        c[-flat:] = c[-flat - 1]
    h = np.round(c * (1 + np.abs(rng.normal(0, sig, n))) / TICK[coin]) * TICK[coin]; l = np.round(c * (1 - np.abs(rng.normal(0, sig, n))) / TICK[coin]) * TICK[coin]
    v = np.round(np.exp(rng.normal(6, 1, n)), 3)
    if zero_v:
        v[rng.choice(n - 5, zero_v, replace=False)] = 0.0
    d = pd.DataFrame({"t": t.astype(np.int64), "h": h, "l": l, "c": c, "v": v})
    if gaps:
        d = d.drop(index=rng.choice(np.arange(10, n - 2), gaps, replace=False)).reset_index(drop=True)
    if gap_day:                                                      # 마지막 1일 안(마지막 4시간 밖)에 빠진 분 gap_day 개 -- 긴 창 «5% 까지 허용» 경계
        d = d.drop(index=rng.choice(np.arange(n - 1440, n - 300), gap_day, replace=False)).reset_index(drop=True)
    return d


def dvol(rng, now_ms, n=900, formed=False, stale_h=0, formed_prev=False):
    end = now_ms // 3600_000 * 3600_000 - stale_h * 3600_000
    ts = end - 3600_000 * np.arange(n - 1, -1, -1)
    x = np.round(55 * np.exp(np.cumsum(rng.normal(0, .02, n))), 2)
    got = np.maximum(ts + 3600_000 + 5000, now_ms - 10 * 3600_000).astype(np.int64)
    if formed:
        got[-1] = ts[-1] + 1800_000                                   # 끝나기 전에 받은 봉(형성 중 값) -- dvol_series 가 뺀다
    if formed_prev:                                                   # 이미 끝난 직전 시간봉을 형성 중에 받아 두고(새로 받기 실패 → 캐시 그대로 읽는 길) -- 빼고 그 앞 봉을 써야
        ts, x, got = ts[:-1], x[:-1], got[:-1]; got[-1] = ts[-1] + 1800_000
    return pd.DataFrame({"ts": ts.astype(np.int64), "close": x, "got": got})


def assemble(now_ms, res, params_name):
    """main() :221-240 과 같은 조립(코인 결과 → latest.json 글자 · forward.jsonl 줄)."""
    payload = dict(updated=dt.datetime.fromtimestamp(now_ms / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), params=params_name, coins=res)
    latest = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    log = [json.dumps(dict(coin=c, **{k: o[k] for k in ("ts", "price", "sigma15_pct", "baseline7d_sigma15_pct", "sigma60_pct", "usual60_pct", "state60", "data_age_s")}, alert=o["alert"]["active"]),
                      ensure_ascii=False, allow_nan=False) for c, o in res.items() if "error" not in o]
    return latest, log


def one(rng, name, P):
    now = int(pd.Timestamp("2026-11-21 10:00", tz="UTC").timestamp() * 1000) + int(rng.integers(0, 30 * 1440)) * MIN + int(rng.integers(1000, 50000))
    if name == "weekend":
        now = int(pd.Timestamp("2026-11-22 23:58:30", tz="UTC").timestamp() * 1000)
    res, inp = {}, {}
    for coin, p in P["coins"].items():
        n = {"short": 6 * 1440, "mid": 26 * 1440}.get(name, W.KEEP)
        lag = {"stale": 5, "stale_edge": 3}.get(name, 0) if coin == "SOL" else 0      # SOL 만 REST 가 새 봉을 안 줌 → 지연 무효
        spikes = {"spikes": [(30, .02), (25, -.015), (3, .03)], "spikes30": [(41, .025), (11, -.025)]}.get(name, [])   # 41·11 = 30분 간격(중복 제거 경계)
        d = bars(rng, coin, (now // MIN - lag) * MIN, n, 4e-4, gaps=300 if name == "gaps" else 0, zero_v=200 if name == "zero_vol" else 0, spikes=spikes,
                 flat=2000 if name == "flat" and coin == "XRP" else 0, gap_day=60 if name == "gap_day" else 0)
        iv = dv = None
        if "beta60_iv" in p and name != "no_dvol":
            dv = dvol(rng, now, formed=name == "dvol_formed", stale_h={"dvol_stale": 3, "dvol_stale2": 2}.get(name, 0), formed_prev=name == "dvol_formed_prev")
            iv = W.dvol_series(dv)
        try:
            cl, hi, lo, m0, qv = W.grid(d)
            out, _ = W.compute(cl, hi, lo, m0, p, P, qv, iv)
            out["data_age_s"] = round((now - int(d.t.iloc[-1]) - MIN) / 1000, 1)
            if out["data_age_s"] > W.STALE_DATA_S:
                out.update(valid=False, error=f"데이터 지연 {out['data_age_s']:.0f}초")
        except Exception as e:
            out = dict(error=f"{type(e).__name__}: {str(e)[:200]}", valid=False)
        res[coin] = out
        inp[coin] = {"t": d.t.tolist(), "h": d.h.tolist(), "l": d.l.tolist(), "c": d.c.tolist(), "v": d.v.tolist(),
                     "dvol": None if dv is None else {"ts": dv.ts.tolist(), "close": dv.close.tolist(), "got": dv.got.tolist()}}
    # SOXL(주식 세션 HAR-S · 원본 main 은 계수 파일 두 개를 합쳐 4코인 뒤에 돈다)
    ps = json.loads(W.PARAMS_SESS.read_text())["coins"]["SOXL"]
    d = bars(rng, "SOXL", (now // MIN - (2 if name == "stale_edge" else 0)) * MIN, {"short": 6 * 1440}.get(name, 12 * 1440), 5e-4, gaps=300 if name == "gaps" else 0)
    try:
        out = W.compute_session(d, ps)
        out["data_age_s"] = round((now - int(d.t.iloc[-1]) - MIN) / 1000, 1)
        if out["data_age_s"] > W.STALE_DATA_S:
            out.update(valid=False, error=f"데이터 지연 {out['data_age_s']:.0f}초")
    except Exception as e:
        out = dict(error=f"{type(e).__name__}: {str(e)[:200]}", valid=False)
    res["SOXL"] = out
    inp["SOXL"] = {"t": d.t.tolist(), "h": d.h.tolist(), "l": d.l.tolist(), "c": d.c.tolist(), "v": d.v.tolist(), "dvol": None}
    latest, log = assemble(now, res, W.PARAMS.name)
    return {"name": name, "now_ms": now, "inputs": inp, "latest": latest, "log": log}


def main():
    warnings.filterwarnings("ignore")
    P = json.loads(W.PARAMS.read_text())
    rng = np.random.default_rng(20261121)
    names = ["normal", "normal2", "gaps", "zero_vol", "spikes", "flat", "stale", "short", "mid", "no_dvol", "dvol_formed", "dvol_stale", "weekend", "gap_day", "spikes30", "dvol_stale2", "dvol_formed_prev", "stale_edge"]
    fx = {"npy_disable": os.environ.get("NPY_DISABLE_CPU_FEATURES", ""), "params": W.PARAMS.name, "sets": []}
    for nm in names:
        s = one(rng, nm, P); fx["sets"].append(s)
        c = json.loads(s["latest"])["coins"]
        print(nm, {k: (o.get("error") or f"σ15 {o['sigma15_pct']} {o['state60']} valid={o['valid']} iv60={o.get('iv60')} {o.get('session') or ''} 급변={o['alert']['active']}") for k, o in c.items()}, flush=True)
    # 세션 시계(연구 clock) 표: 서머타임 전환·휴장·개장/마감 경계 -- 2026-03 ~ 2027-12 무작위 + 경계 근처
    S_ = W.S; taus = list(rng.integers(int(pd.Timestamp("2026-03-01", tz="UTC").timestamp()), int(pd.Timestamp("2027-12-31", tz="UTC").timestamp()), 3000) // 60 * 60)
    for day in ("2026-03-08", "2026-11-01", "2026-11-26", "2026-12-24", "2027-03-14", "2027-11-07", "2026-11-23", "2026-11-27"):
        for hh in range(0, 24):
            for mm in (0, 29, 30, 31, 59):
                taus.append(int(pd.Timestamp(f"{day} {hh:02d}:{mm:02d}", tz="UTC").timestamp()))
    fx["clock"] = []
    for H in (15, 60):
        ck = S_.clock(np.array(taus), H)
        fx["clock"] += [[int(t), H, int(r.sess), float(r.open_in), float(r.first60), float(r.last30)] for t, r in zip(taus, ck.itertuples())]
    p = OUT / "radar_fixture.json"; p.write_text(json.dumps(fx)); print(p.name, f"{p.stat().st_size:,} 바이트 · 벌 {len(fx['sets'])}")


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
