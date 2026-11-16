#!/usr/bin/env python3
"""실시간 위험 레이더 1분 워커 (2026-11-15) -- 배포 대기 A10 · 결과 docs/experiments/realtime_risk_radar_20261115.md.

매분 마감 직후: 4코인 선물 1분봉(닫힌 봉만)을 캐시에 이어 붙이고, 연구와 같은 함수(research_realtime_risk_radar_20261115.features·design)와
동결 계수(dashboard/risk_radar_params_20261116.json, B2 거래대금 포함)로 마지막 분의
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
import research_soxl_session_vol_20261112 as S  # noqa: E402 -- 2026-11-17 SOXL 세션 HAR-S(연구 frame·har_x 그대로)

STALE_DATA_S = 180                                                 # 마지막 닫힌 봉이 3분 넘게 옛것이면 그 코인은 무효
PARAMS = ROOT / "dashboard/risk_radar_params_20261116.json"   # B2(11-16): + 거래대금 5창 · 옛 계수 _20261115.json 은 되돌리기용
PARAMS_SESS = ROOT / "dashboard/risk_radar_params_soxl_20261117.json"   # 2026-11-17 주식(SOXL) 세션 식 · 되돌리기 = 이 파일 삭제(그 종목만 빠진다)
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


DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"   # 2026-11-16 IV(1시간 식, ETH·BTC) -- 바이낸스 IP 와 무관
IV_KEEP_H = 900                                                                 # 30일 분위(720시간) + 여유


def dvol_rest(coin: str, start_ms: int, end_ms: int) -> list:
    import requests
    r = requests.get(DVOL_URL, params=dict(currency=coin, start_timestamp=start_ms, end_timestamp=end_ms, resolution="3600"), timeout=20)
    r.raise_for_status()
    return r.json()["result"]["data"]                                          # [[봉 시작 ms, o, h, l, c], ...]


def update_dvol(coin: str, now_ms: int, fetch=dvol_rest, path: Path | None = None) -> pd.DataFrame:
    """DVOL 시간봉 캐시(ts·close·got). 마지막 봉이 끝나면(또는 비었으면) 그 앞 3시간부터 다시 받아 덮는다 -- 시간에 한 번꼴.
    got = 받은 시각: 끝나기 전에 받은 봉(형성 중 값)은 아래 dvol_series 가 안 쓴다."""
    path = path or CACHE / f"{coin}_dvol.parquet"
    try:
        df = pd.read_parquet(path) if path.exists() else None
    except Exception:
        df = None
    if df is None or not len(df):
        df = pd.DataFrame({"ts": pd.Series(dtype="int64"), "close": pd.Series(dtype="float64"), "got": pd.Series(dtype="int64")})
    if len(df) and now_ms < int(df.ts.max()) + 3600_000:
        return df
    start = int(df.ts.max()) - 3 * 3600_000 if len(df) else now_ms - IV_KEEP_H * 3600_000
    rows = []
    while start < now_ms:                                                        # 500시간씩
        end = min(now_ms, start + 500 * 3600_000)
        rows += [(int(x[0]), float(x[4])) for x in fetch(coin, start, end)]
        start = end
    if rows:
        new = pd.DataFrame(rows, columns=["ts", "close"]).assign(got=now_ms)
        df = pd.concat([df, new]).drop_duplicates("ts", keep="last").sort_values("ts")
        df = df[df.ts >= now_ms - IV_KEEP_H * 3600_000].reset_index(drop=True)
        save(df, path)
    return df


def dvol_series(df: pd.DataFrame):
    """캐시 → 연구 iv_hourly 와 같은 (아는 시각, log, 30일 분위). 끝나기 전에 받은 봉은 뺀다."""
    d = df[df.got >= df.ts + 3600_000]
    return R.iv_hourly(d.ts.to_numpy() // 1000, d.close.to_numpy(float)) if len(d) else None


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet"); df.to_parquet(tmp); tmp.replace(path)          # 원자적 교체(쓰다 죽어도 옛 캐시 유지)


def read_cache(path: Path) -> pd.DataFrame:
    empty = pd.DataFrame({"t": pd.Series(dtype="int64"), **{k: pd.Series(dtype="float64") for k in ("h", "l", "c", "v")}})
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
    if len(df) and ("v" not in df or df.v.tail(1441).isna().any()):          # 거래대금 없는 옛 캐시 → 1일+ 다시 받아 덮는다(VWIN 최장 1440분)
        start = min(start, int(df.t.iloc[-1]) - 1500 * MIN)
    while start < now_ms - MIN:
        lim = 1000 if now_ms - start > 99 * MIN else 99            # 평소엔 1분 몇 개(가중치 1)
        k = fetch(f"{coin}USDT", start, lim)
        k = [x for x in k if x[6] + 1000 < now_ms]                  # 닫힌 봉만 -- 1초 여유(서버 시계가 조금 빨라도 형성 봉이 안 들어오게 · 검증 11-16)
        if not k:
            break
        new = pd.DataFrame({"t": [x[0] for x in k], "h": [float(x[2]) for x in k], "l": [float(x[3]) for x in k], "c": [float(x[4]) for x in k], "v": [float(x[5]) for x in k]})
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
    return g.c.to_numpy(float), g.h.to_numpy(float), g.l.to_numpy(float), m0, (g.v * g.c).to_numpy(float)   # 거래대금 = 연구 load_qv 와 같은 v×c


def usual60(F: dict) -> np.ndarray:
    """직전 완결 60분 RV 의 720개 굴림 중앙값(연구 coin_run 과 같은 식)."""
    mins = F["mins"]; s60, k60 = R.wsum(F["r2"], 60)
    past = np.where((k60 == 0) & ((mins + 1) % 60 == 0), np.sqrt(s60), np.nan)
    pidx = np.flatnonzero(np.isfinite(past))
    uh = pd.Series(past[pidx]).rolling(720, min_periods=600).median().to_numpy()
    last = np.searchsorted(pidx, np.arange(len(mins)), "right") - 1
    return np.where(last >= 0, uh[np.maximum(last, 0)], np.nan) if len(uh) else np.full(len(mins), np.nan)


def compute(cl, hi, lo, m0, p: dict, P: dict, qv=None, iv=None) -> dict:
    """배열 전체에 대해 예측(인과) → 마지막 분 결과 + 전체 배열(대조용). iv = dvol_series 결과(없으면 1시간도 B2 식)."""
    F = R.features(cl, m0)
    if any(c in R.VWIN for c in P["cols"]):
        R.vol_features(F, qv, p.get("vclip"))   # 창별 하한 = 학습과 같은 값(검증 11-16 ①)
    X = R.design(F, P["cols"])
    f15 = p["c15"] * np.exp(2 * X @ np.array(p["beta15"])); f60 = p["c60"] * np.exp(2 * X @ np.array(p["beta60"]))
    f60i = np.full(len(cl), np.nan)
    if iv is not None and "beta60_iv" in p:                                     # 1시간 = IV 판(B2 위 −0.53%) · IV 가 낡았으면 그 분은 B2 식
        R.iv_features(F, *iv); f60i = p["c60_iv"] * np.exp(2 * R.design(F, p["cols60_iv"]) @ np.array(p["beta60_iv"]))
        f60 = np.where(np.isfinite(f60i), f60i, f60)
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
               alert=dict(active=ago is not None and ago < 15, minutes_ago=ago), valid=s15 is not None and ratio is not None, iv60=bool(np.isfinite(f60i[i])))
    return out, dict(f15=f15, f60=f60, usual=us, raw=raw)


def compute_session(df: pd.DataFrame, p: dict) -> dict:
    """주식 선물(SOXL): 세션 HAR-S -- 연구 frame(매분 τ)·har_x 와 동결 계수. τ = 마지막 닫힌 봉 다음 분(형성 중 봉 자리를 빈 행으로 붙여 τ 행을 만든다).
    «앞 1시간 조용/격함»·급변 경고는 코인에서만 검정했다 -- 여기는 세션 이름과 1시간 σ 만 낸다.
    ponytail: 개장 더미(open_in = 개장까지 15분 미만)는 5분 경계에서만 배웠다 -- 매분 쓰면 개장 11~14분 전(동부 09:16~09:19)엔 σ̂ 가 1.6~2.3배 크게 나온다
    (시험 29일 · 시계 더미를 5분 내림하면 반대로 1.3~2배 작아져 더 나쁨). 고치려면 «창 안 정규장 분 비율» 피쳐로 새 검정."""
    k = pd.concat([df.tail(9 * 1440), pd.DataFrame({"t": [int(df.t.iloc[-1]) + MIN]})], ignore_index=True)   # 7일 창 + 여유
    r15, r60 = S.frame(k, 15, every=1).iloc[-1:], S.frame(k, 60, every=1).iloc[-1:]
    v15 = p["c15"] * np.exp(2 * (S.har_x(r15) @ np.array(p["beta15"]))[0]); v60 = p["c60"] * np.exp(2 * (S.har_x(r60) @ np.array(p["beta60"]))[0])
    fin = lambda x, m=1.0: float(x) * m * 100 if np.isfinite(x) else None
    return dict(ts=pd.Timestamp(int(r15.tau.iloc[0]), unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), price=float(df.c.iloc[-1]),
                sigma15_pct=fin(np.sqrt(v15)), band80_pct=fin(np.sqrt(v15), p["q80"]), band95_pct=fin(np.sqrt(v15), p["q95"]),
                baseline7d_sigma15_pct=fin(r15.b0.iloc[0] * np.sqrt(p["cb0"])), sigma60_pct=fin(np.sqrt(v60)), usual60_pct=None, ratio60=None, state60=None,
                session=S.SESS[int(r15.sess.iloc[0])], alert=dict(active=False, minutes_ago=None), valid=bool(np.isfinite(v15) and np.isfinite(v60)), touch=p["touch"])


def main(now_ms: int | None = None, fetch=rest, dvol_fetch=dvol_rest) -> int:
    now_ms = now_ms or int(time.time() * 1000)
    P = json.loads(PARAMS.read_text()); res = {}
    try:                                                            # 주식 계수 파일이 없거나 깨져도 4코인은 그대로
        sess = json.loads(PARAMS_SESS.read_text())["coins"] if PARAMS_SESS.exists() else {}
    except Exception as e:
        print(f"{PARAMS_SESS.name} {type(e).__name__}: {str(e)[:120]}", file=sys.stderr); sess = {}
    for coin, p in {**P["coins"], **sess}.items():
        try:                                                        # 한 코인 실패가 나머지를 막지 않게
            df = update_cache(coin, now_ms, fetch)
            if p.get("kind") == "session":
                out = compute_session(df, p)
            else:
                cl, hi, lo, m0, qv = grid(df)
                iv = None
                if "beta60_iv" in p:
                    try:                                                        # IV 실패는 1시간을 B2 식으로 떨어뜨릴 뿐(코인 출력은 그대로)
                        iv = dvol_series(update_dvol(coin, now_ms, dvol_fetch))
                    except Exception as e:
                        print(f"{coin} DVOL {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                        try:
                            iv = dvol_series(pd.read_parquet(CACHE / f"{coin}_dvol.parquet"))
                        except Exception:
                            iv = None
                out, _ = compute(cl, hi, lo, m0, p, P, qv, iv)
            out["data_age_s"] = round((now_ms - int(df.t.iloc[-1]) - MIN) / 1000, 1)
            if out["data_age_s"] > STALE_DATA_S:                   # REST 가 새 봉을 안 줘도 옛 값이 지금 값처럼 보이던 것(검증 11-16 ②) -- 화면은 24h 로 떨어진다
                out.update(valid=False, error=f"데이터 지연 {out['data_age_s']:.0f}초")
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
    print(" · ".join(f"{c} " + (o["error"] if "error" in o else f"σ15 {o['sigma15_pct'] or 0:.3f}% (기준 {o['baseline7d_sigma15_pct'] or 0:.3f}) {o.get('session') or o['state60']}{' 급변' if o['alert']['active'] else ''}") for c, o in res.items()))
    return 0 if all("error" not in o for o in res.values()) else 1


def selftest():
    """네트워크 없음: (1) 연구 배열과 같은 예측 (2) 잘라 계산 = 마지막 분 값 (3) 캐시 이어 붙이기·닫힌 봉만."""
    P = json.loads(PARAMS.read_text())
    if (R.VIS / "klines1m").exists():
        for coin in ("ETH", "XRP"):
            cl, hi, lo, m0 = R.load(coin); qv = R.load_qv(coin); p = P["coins"][coin]
            _, full = compute(cl, hi, lo, m0, p, P, qv)
            F = R.vol_features(R.features(cl, m0), qv, p.get("vclip")); X = R.design(F, P["cols"])
            ref = p["c15"] * np.exp(2 * X @ np.array(p["beta15"]))
            assert np.allclose(full["f15"], ref, equal_nan=True)
            if "beta60_iv" in p:                                         # 1시간 IV 판 = 연구 식(같은 DVOL 시간봉)
                iv = R.load_iv(coin); _, fi = compute(cl, hi, lo, m0, p, P, qv, iv)
                R.iv_features(F, *iv); ri = p["c60_iv"] * np.exp(2 * R.design(F, p["cols60_iv"]) @ np.array(p["beta60_iv"]))
                ok = np.isfinite(ri); assert ok.mean() > 0.5 and np.allclose(fi["f60"][ok], ri[ok]), coin
            nu = 0
            for i in (len(cl) - 120000, len(cl) - 60000, len(cl) - 5000):   # 서버처럼 31일만 들고 있을 때 마지막 분 값
                a = i - KEEP + 1
                o, _ = compute(cl[a:i + 1], hi[a:i + 1], lo[a:i + 1], m0 + a, p, P, qv[a:i + 1])
                if np.isfinite(full["f15"][i]):
                    assert np.isclose(o["sigma15_pct"], np.sqrt(full["f15"][i]) * 100), coin
                if not np.isnan(cl[a:i + 1]).any():                     # 구멍 없는 31일이면 연구 값과 같아야 함(구멍이 있으면 600~719시간 중앙값 -- 서버 장애 때만)
                    assert np.isclose(o["usual60_pct"], full["usual"][i] * 100), coin; nu += 1
            assert nu >= 1, coin
            print(f"  {coin}: 연구 배열과 일치 · 31일 창 마지막 분 일치")
    if PARAMS_SESS.exists() and (S.OUT / "klines").exists():                # SOXL: 그 시각까지만 든 캐시의 서빙 값 = 연구가 시험에 쓴 HAR-S 값(같은 τ)
        ps, k = json.loads(PARAMS_SESS.read_text())["coins"]["SOXL"], S.load("SOXLUSDT")
        for H, key in ((15, "sigma15_pct"), (60, "sigma60_pct")):
            d = S.frame(k, H).dropna(subset=[*S.HGB_COLS, "b0", "rv", "open_in"]).reset_index(drop=True)
            te = S.cut(d, "test"); ref = np.sqrt(S.fit_predict(S.cut(d, "refit"), te)["HAR"]) * 100
            for i in (7, len(te) // 3, len(te) // 2 + 1, len(te) - 9):       # 세션이 다른 네 시각
                tau = int(te.tau.iloc[i]); o = compute_session(k[k.t < tau * 1000], ps)
                assert o["valid"] and np.isclose(o[key], ref[i], rtol=1e-9) and o["ts"] == pd.Timestamp(tau, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), (H, i)
                assert o["session"] == S.SESS[int(te.sess.iloc[i])]
        mid = int(te.tau.iloc[len(te) // 2]) + 120                           # 5분 경계가 아닌 분에도 값이 난다(매분 갱신)
        assert compute_session(k[k.t < mid * 1000], ps)["valid"]
        print("  SOXL: 시험에 쓴 HAR-S 값과 일치(15분·1시간 · 미래 봉 없이)")
    import tempfile
    with tempfile.TemporaryDirectory() as td:                            # DVOL 캐시: 형성 봉 안 씀 · 한 시간 안엔 다시 안 받음 · 봉 끝나면 받음
        bh = 1_790_000_000_000 // 3600_000 * 3600_000
        fdv = lambda coin, s, e: [[t, 0, 0, 0, 50.0 + (t // 3600_000) % 7] for t in range(s - s % 3600_000, e, 3600_000)]
        def nofetch(*a):
            raise AssertionError("한 시간 안 재요청")
        pv = Path(td) / "dv.parquet"
        d1 = update_dvol("ETH", bh + 30 * MIN, fdv, pv); assert len(d1) >= 899 and d1.ts.max() == bh and dvol_series(d1)[0].max() == bh // 1000
        update_dvol("ETH", bh + 40 * MIN, nofetch, pv)
        d3 = update_dvol("ETH", bh + 3600_000 + 4000, fdv, pv); assert dvol_series(d3)[0].max() == (bh + 3600_000) // 1000 and not d3.ts.duplicated().any()
    base = 1_790_000_000_000 // MIN * MIN
    def fake(sym, start, lim):
        return [[t, 0, 101.0, 99.0, 100.0 + (t // MIN) % 7, 0, t + MIN - 1] for t in range(start, start + lim * MIN, MIN)]
    with tempfile.TemporaryDirectory() as td:
        pth = Path(td) / "x.parquet"
        d1 = update_cache("ETH", base, fake, pth); assert len(d1) <= KEEP and d1.t.iloc[-1] + MIN - 1 < base
        d2 = update_cache("ETH", base + 3 * MIN, fake, pth); assert d2.t.iloc[-1] == d1.t.iloc[-1] + 3 * MIN and d2.t.is_monotonic_increasing and not d2.t.duplicated().any()
        d1.drop(columns="v").to_parquet(pth); d4 = update_cache("ETH", base + 3 * MIN, fake, pth)              # 거래대금 없는 옛 캐시 → 꼬리 다시 받아 채움
        assert d4.v.tail(1441).notna().all() and not d4.t.duplicated().any()
        pth.write_bytes(b"broken"); d3 = update_cache("ETH", base, fake, pth); assert len(d3) > 0 and pth.with_suffix(".bad").exists()   # 깨진 캐시 → 치우고 다시 백필
        g = globals(); keep = (g["CACHE"], g["LATEST"], g["LOG"])                                       # main: 한 코인 실패가 나머지를 막지 않음 · NaN 없는 json
        g["CACHE"], g["LATEST"], g["LOG"] = Path(td) / "c", Path(td) / "l.json", Path(td) / "f.jsonl"
        def half(sym, start, lim):
            if sym == "SOLUSDT":
                raise ConnectionError("x")
            return fake(sym, start, lim)
        rc = main(base, half, lambda *a: []); g["CACHE"], g["LATEST"], g["LOG"] = keep
        out = json.loads((Path(td) / "l.json").read_text())
        assert rc == 1 and "error" in out["coins"]["SOL"] and "sigma15_pct" in out["coins"]["ETH"]
        assert not PARAMS_SESS.exists() or ("SOXL" in out["coins"] and "session" in out["coins"]["SOXL"]), out["coins"].get("SOXL")   # 주식 종목도 같은 파일에
        g["CACHE"], g["LATEST"], g["LOG"] = Path(td) / "c", Path(td) / "l.json", Path(td) / "f.jsonl"   # REST 가 10분 동안 새 봉을 안 주면 그 코인은 무효(검증 11-16 ②)
        main(base + 10 * MIN, lambda sym, start, lim: [], lambda *a: []); g["CACHE"], g["LATEST"], g["LOG"] = keep
        o2 = json.loads((Path(td) / "l.json").read_text())["coins"]["ETH"]
        assert o2["valid"] is False and "지연" in o2["error"] and o2["data_age_s"] > STALE_DATA_S
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
    else:
        sys.exit(main())
