#!/usr/bin/env python3
"""실시간 위험 레이더 1분 워커 (2026-11-15) -- 배포 대기 A10 · 결과 docs/experiments/realtime_risk_radar_20261115.md.

매분 마감 직후: 4코인 선물 1분봉(닫힌 봉만)을 캐시에 이어 붙이고, 연구와 같은 함수(research_realtime_risk_radar_20261115.features·design)와
동결 계수(dashboard/risk_radar_params_20261115.json)로 마지막 분의
  · 다음 15분 σ̂ · «15분 안 머물 확률 80%·95%» 띠 · 현행 화면 σ(7일 1시간 고정, 비교용)
  · 앞 1시간 조용/보통/격함(예측 ÷ 직전 30일 1시간 중앙값)
  · 급변 경보(1분 |r| ≥ 4σ̂ 또는 5분 RV ≥ 3배 · 30분 중복 제거 · 15분 동안 켜 둠 -- 불통과 항목이라 «경고»만)
을 data/live/risk_radar_latest.json 에 쓰고 data/live/risk_radar_forward.jsonl 에 한 줄씩 남긴다(전진 확인용).
읽기 전용 fapi(주문 호출 없음). ⚠️서버에서만 실행(로컬 REST = IP 공유). 로컬 확인은 --selftest(네트워크 없음, 연구 배열과 대조).
  python scripts/live_risk_radar_20261115.py            # cron 매분
  python scripts/live_risk_radar_20261115.py --selftest
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import binance_ban_guard  # noqa: E402,F401 -- 공용 밴 가드: requests 를 감싸 418/429 뒤엔 바이낸스 REST 를 내보내지 않는다
import research_realtime_risk_radar_20261115 as R  # noqa: E402

PARAMS = ROOT / "dashboard/risk_radar_params_20261115.json"
CACHE = ROOT / "data/live/risk_radar"
LATEST = ROOT / "data/live/risk_radar_latest.json"
LOG = ROOT / "data/live/risk_radar_forward.jsonl"
KEEP = 31 * 1440 + 60                     # «평소»(720 완결 시간 중 600 이상) + 7일 RV 를 채우는 길이
MIN = 60_000


def rest(symbol: str, start_ms: int, limit: int) -> list:
    import requests                                                   # 밴 가드가 감싼 requests(차단 중이면 보내지 않고 예외)
    r = requests.get("https://fapi.binance.com/fapi/v1/klines", params=dict(symbol=symbol, interval="1m", startTime=start_ms, limit=limit), timeout=20)
    r.raise_for_status()
    return r.json()


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet"); df.to_parquet(tmp); tmp.replace(path)          # 원자적 교체(쓰다 죽어도 옛 캐시 유지)


def read_cache(path: Path) -> pd.DataFrame:
    empty = pd.DataFrame({"t": pd.Series(dtype="int64"), **{k: pd.Series(dtype="float64") for k in "hlc"}})
    if not path.exists():
        return empty
    try:
        return pd.read_parquet(path)
    except Exception:                                                 # 깨진 캐시는 치워 두고 다시 백필
        path.replace(path.with_suffix(".bad"))
        return empty


def update_cache(coin: str, now_ms: int, fetch=rest, path: Path | None = None) -> pd.DataFrame:
    """캐시(t·h·l·c)에 닫힌 1분봉을 이어 붙인다. 처음엔 31일 백필(limit 1000·0.5초 간격 = 가중치 초당 10 이하)."""
    path = path or CACHE / f"{coin}_1m.parquet"
    df = read_cache(path)
    start = int(df.t.iloc[-1]) if len(df) else (now_ms // MIN - KEEP) * MIN   # 마지막 봉을 한 번 더 받아 덮는다(경계 직후 덜 닫힌 봉 교정)
    while start < now_ms - MIN:
        lim = 1000 if now_ms - start > 99 * MIN else 99            # 평소엔 1분 몇 개(가중치 1)
        k = fetch(f"{coin}USDT", start, lim)
        k = [x for x in k if x[6] < now_ms]                         # 닫힌 봉만
        if not k:
            break
        new = pd.DataFrame({"t": [x[0] for x in k], "h": [float(x[2]) for x in k], "l": [float(x[3]) for x in k], "c": [float(x[4]) for x in k]})
        df = pd.concat([df, new]).drop_duplicates("t", keep="last").sort_values("t").tail(KEEP).reset_index(drop=True)
        save(df, path)                                              # 쪽마다 저장(백필 중 실패해도 받은 만큼 남음)
        if k[-1][0] + MIN <= start:
            break
        start = k[-1][0] + MIN
        if lim == 1000:
            time.sleep(0.5)
    return df


def grid(df: pd.DataFrame):
    m0, m1 = int(df.t.iloc[0]) // MIN, int(df.t.iloc[-1]) // MIN
    g = df.assign(m=df.t // MIN).set_index("m").reindex(range(m0, m1 + 1))
    return g.c.to_numpy(float), g.h.to_numpy(float), g.l.to_numpy(float), m0


def usual60(F: dict) -> np.ndarray:
    """직전 완결 60분 RV 의 720개 굴림 중앙값(연구 coin_run 과 같은 식)."""
    mins = F["mins"]; s60, k60 = R.wsum(F["r2"], 60)
    past = np.where((k60 == 0) & ((mins + 1) % 60 == 0), np.sqrt(s60), np.nan)
    pidx = np.flatnonzero(np.isfinite(past))
    uh = pd.Series(past[pidx]).rolling(720, min_periods=600).median().to_numpy()
    last = np.searchsorted(pidx, np.arange(len(mins)), "right") - 1
    return np.where(last >= 0, uh[np.maximum(last, 0)], np.nan) if len(uh) else np.full(len(mins), np.nan)


def compute(cl, hi, lo, m0, p: dict, P: dict) -> dict:
    """배열 전체에 대해 예측(인과) → 마지막 분 결과 + 전체 배열(대조용)."""
    F = R.features(cl, m0); X = R.design(F, P["cols"])
    f15 = p["c15"] * np.exp(2 * X @ np.array(p["beta15"])); f60 = p["c60"] * np.exp(2 * X @ np.array(p["beta60"]))
    us = usual60(F)
    sig1 = np.sqrt(np.r_[np.nan, f15[:-1]]) / np.sqrt(15); f5 = np.r_[np.full(5, np.nan), f15[:-5]]
    raw = (np.abs(F["r"]) >= P["alert_k"] * sig1) | (np.exp(F["lrv5"]) >= P["alert_k5"] * np.sqrt(f5) * np.sqrt(5 / 15))
    raw &= np.isfinite(f15)
    last_alert, prev = None, -10**9                                 # 30분 중복 제거(과거만, 연구 alerts() 와 같은 사슬 -- 캐시 전체를 훑는다)
    for i in np.flatnonzero(raw):
        if i - prev > P["alert_dedupe_min"]:
            last_alert, prev = i, i
    i = len(cl) - 1
    fin = lambda x: float(x) if np.isfinite(x) else None             # NaN 은 None(json 에 NaN 을 쓰지 않는다)
    s15 = fin(np.sqrt(f15[i])); s60 = fin(np.sqrt(f60[i]))
    ratio = fin(s60 / us[i]) if s60 is not None and np.isfinite(us[i]) and us[i] > 0 else None
    state = None if ratio is None else ("조용" if ratio <= P["quiet"] else "격함" if ratio >= P["loud"] else "보통")
    ago = None if last_alert is None else int(i - last_alert)
    pc = lambda x, k=1.0: None if x is None else x * k * 100
    b0 = fin(F["b0sd"][i] * np.sqrt(15 / 60 * p["cb0"]))             # 기준선(7일 1시간 σ 고정 × 학습 눈금) -- 연구 B0 와 같은 값
    out = dict(ts=pd.Timestamp((m0 + i + 1) * MIN, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), price=fin(cl[i]),
               sigma15_pct=pc(s15), band80_pct=pc(s15, p["q80"]), band95_pct=pc(s15, p["q95"]), baseline7d_sigma15_pct=pc(b0),
               sigma60_pct=pc(s60), usual60_pct=pc(fin(us[i])), ratio60=ratio, state60=state,
               alert=dict(active=ago is not None and ago < 15, minutes_ago=ago), valid=s15 is not None and ratio is not None)
    return out, dict(f15=f15, f60=f60, usual=us, raw=raw)


def main(now_ms: int | None = None, fetch=rest) -> int:
    now_ms = now_ms or int(time.time() * 1000)
    P = json.loads(PARAMS.read_text()); res = {}
    for coin, p in P["coins"].items():
        try:                                                        # 한 코인 실패가 나머지를 막지 않게
            df = update_cache(coin, now_ms, fetch)
            cl, hi, lo, m0 = grid(df)
            out, _ = compute(cl, hi, lo, m0, p, P)
            out["data_age_s"] = round(now_ms / 1000 - (m0 + len(cl)) * 60, 1)
        except Exception as e:
            out = dict(error=f"{type(e).__name__}: {str(e)[:200]}", valid=False)
        res[coin] = out
    payload = dict(updated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), params=PARAMS.name, coins=res)
    LATEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = LATEST.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False)); tmp.replace(LATEST)   # 원자적 교체
    with LOG.open("a") as f:
        for c, o in res.items():
            if "error" not in o:
                f.write(json.dumps(dict(coin=c, **{k: o[k] for k in ("ts", "price", "sigma15_pct", "baseline7d_sigma15_pct", "sigma60_pct", "usual60_pct", "state60", "data_age_s")}, alert=o["alert"]["active"]), ensure_ascii=False, allow_nan=False) + "\n")
    print(" · ".join(f"{c} " + (o["error"] if "error" in o else f"σ15 {o['sigma15_pct'] or 0:.3f}% (기준 {o['baseline7d_sigma15_pct'] or 0:.3f}) {o['state60']}{' 급변' if o['alert']['active'] else ''}") for c, o in res.items()))
    return 0 if all("error" not in o for o in res.values()) else 1


def selftest():
    """네트워크 없음: (1) 연구 배열과 같은 예측 (2) 잘라 계산 = 마지막 분 값 (3) 캐시 이어 붙이기·닫힌 봉만."""
    P = json.loads(PARAMS.read_text())
    if (R.VIS / "klines1m").exists():
        for coin in ("ETH", "XRP"):
            cl, hi, lo, m0 = R.load(coin); p = P["coins"][coin]
            _, full = compute(cl, hi, lo, m0, p, P)
            F = R.features(cl, m0); X = R.design(F, P["cols"])
            ref = p["c15"] * np.exp(2 * X @ np.array(p["beta15"]))
            assert np.allclose(full["f15"], ref, equal_nan=True)
            nu = 0
            for i in (len(cl) - 120000, len(cl) - 60000, len(cl) - 5000):   # 서버처럼 31일만 들고 있을 때 마지막 분 값
                a = i - KEEP + 1
                o, _ = compute(cl[a:i + 1], hi[a:i + 1], lo[a:i + 1], m0 + a, p, P)
                if np.isfinite(full["f15"][i]):
                    assert np.isclose(o["sigma15_pct"], np.sqrt(full["f15"][i]) * 100), coin
                if not np.isnan(cl[a:i + 1]).any():                     # 구멍 없는 31일이면 연구 값과 같아야 함(구멍이 있으면 600~719시간 중앙값 -- 서버 장애 때만)
                    assert np.isclose(o["usual60_pct"], full["usual"][i] * 100), coin; nu += 1
            assert nu >= 1, coin
            print(f"  {coin}: 연구 배열과 일치 · 31일 창 마지막 분 일치")
    import tempfile
    base = 1_790_000_000_000 // MIN * MIN
    def fake(sym, start, lim):
        return [[t, 0, 101.0, 99.0, 100.0 + (t // MIN) % 7, 0, t + MIN - 1] for t in range(start, start + lim * MIN, MIN)]
    with tempfile.TemporaryDirectory() as td:
        pth = Path(td) / "x.parquet"
        d1 = update_cache("ETH", base, fake, pth); assert len(d1) <= KEEP and d1.t.iloc[-1] + MIN - 1 < base
        d2 = update_cache("ETH", base + 3 * MIN, fake, pth); assert d2.t.iloc[-1] == d1.t.iloc[-1] + 3 * MIN and d2.t.is_monotonic_increasing and not d2.t.duplicated().any()
        pth.write_bytes(b"broken"); d3 = update_cache("ETH", base, fake, pth); assert len(d3) > 0 and pth.with_suffix(".bad").exists()   # 깨진 캐시 → 치우고 다시 백필
        g = globals(); keep = (g["CACHE"], g["LATEST"], g["LOG"])                                       # main: 한 코인 실패가 나머지를 막지 않음 · NaN 없는 json
        g["CACHE"], g["LATEST"], g["LOG"] = Path(td) / "c", Path(td) / "l.json", Path(td) / "f.jsonl"
        def half(sym, start, lim):
            if sym == "SOLUSDT":
                raise ConnectionError("x")
            return fake(sym, start, lim)
        rc = main(base, half); g["CACHE"], g["LATEST"], g["LOG"] = keep
        out = json.loads((Path(td) / "l.json").read_text())
        assert rc == 1 and "error" in out["coins"]["SOL"] and "sigma15_pct" in out["coins"]["ETH"]
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
    else:
        sys.exit(main())
