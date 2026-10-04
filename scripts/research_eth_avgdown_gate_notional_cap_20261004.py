"""물타기 게이트 + 명목 상한 검토 (2026-10-04, 사용자 «물타기 게이트와 명목 상한 검토 진행해줘»).

배경: 10-02 20:00 KST 롱 1.706@2742 → 급락 중 4번 물타기(전부 추세 veto «숏만» z −5.8~−7.2 ATR) →
최대 6.66 ETH ≈ 자본 7.7배. 사용자 «미리 알았더라면 물타기하지 않고 스위칭해서 리스크관리를 했을텐데».

세 부분
  1 실원장 반사실(주) — 체결 단위 포지션 경로에서 게이트 G 가 막은 «추가»를 빼고, 이후 청산은 비례 축소.
    명목 상한 C 는 총 ETH 명목 ≤ k × 그 시점 순자산(지갑+미실현)으로 증가 체결을 자른다.
  2 테이프 일반화(보조) — 09-13 물타기 시뮬레이터 모양(무작위 진입 + 사다리)에 G1/G2 를 건 판과 안 건 판.
  3 구현 검토는 코드 수정 없이 보고서에서 한다.

실행
  서버(체결·income 수집, 로컬 금지 = 같은 공인 IP):  python scripts/research_eth_avgdown_gate_notional_cap_20261004.py --fetch
  로컬 분석:  python scripts/research_eth_avgdown_gate_notional_cap_20261004.py --ledger | --tape | --weekend
  자체점검:   --selftest
산출물: tmp/avgdown_gate_cap_20261004/
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/avgdown_gate_cap_20261004"

# ── 판정 기준 (결과 보기 전에 고정, 2026-10-04) ─────────────────────────────────────
CRITERIA = {
    # 추세 veto = 대시보드 규약 그대로(dashboard/server.py::trend_veto_rows, ETHUSDT 5m 마감봉)
    "veto": "SMA144(5m 종가) ± K·ATR144(Wilder ewm alpha=1/144) 히스테리시스, K=1.0 · z=(종가−SMA)/ATR · "
            "추가 시점 직전 **마감된** 5m 봉 값(형성 중 봉 점선은 미확정이라 안 씀)",
    "add": "같은 (심볼, positionSide) 포지션이 이미 있을 때 같은 방향 증가 체결. 직전 증가 체결과 180초 안이면 "
           "같은 이벤트(peg 재호가·폴백 묶음) — 왕복 첫 이벤트는 «진입», 이후 이벤트가 «추가»",
    "gates": {"G0": "모든 추가 금지(대조)", "G1": "veto 가 포지션 반대면 추가 금지",
              "G2": "veto 반대 · |z|≥2 일 때만 금지", "Gr": "G1 이 막은 개수만큼 무작위 추가 금지, 시드 50개(위약)"},
    "cap": "총 ETH 명목(양 심볼·양 측면 합) ≤ k × 순자산(지갑+미실현), k∈{3,4,5,6,8,10}. 초과분은 잘라서 체결, "
           "진입 첫 체결에도 적용. 10 = 현행(증거금 50% × 화면 20배)",
    "counterfactual": "막힌/잘린 증가는 안 한 것으로, 이후 감소 체결은 «실제 포지션 대비 같은 비율»로 청산. "
                      "수수료는 체결별 실제 요율을 반사실 수량에 곱한다. 펀딩은 양쪽 모두 제외",
    "equity": "실제 = 서버 income 전 유형 누적(USDT+USDC, 입출금 포함) + ETH 미실현(USDT 1분봉 종가). "
              "반사실 = 실제 + (반사실 누적손익 − 실제 누적손익)",
    "metrics": "총손익 · NAV 지수 MDD(입출금 중립) · 최악 왕복 · 왕복 중 최대 평가손/진입 시 순자산(1분 고저 역행) · "
               "청산가까지 최소 거리(교차증거금 근사 MMR 0.5%) · 손익/|MDD| · 왕복 수",
    "stats": "일(UTC 진입일) 클러스터 부트스트랩 4000회(주), 왕복 부트스트랩(보조), «규칙 − 실제» 짝지은 차이",
    "verdict": {
        "risk": "MDD 축소와 최대 평가손 축소 둘 다 일 클러스터 95% CI 가 0 을 배제(위험이 유의하게 줄어든다)",
        "pnl_noninferior": "총손익 차이의 일 클러스터 95% CI 하한 > −10% × |실제 총손익|",
        "selectivity": "G1 손익 차이가 Gr 50시드 분포의 몇 분위인가(보고만 — 95분위 이상이면 «정보 있음»)",
    },
    "tape": {"periods": {"main": ("2025-01-01", "2026-10-04"), "aux": ("2022-01-01", "2024-12-31")},
             "aux_note": "로컬 1분봉이 2022-01 부터라 보조는 2022~24(지시는 2021~24)",
             "trial": "무작위 시작·무작위 방향(50/50, 신탁 없음) · 사다리 L/5 진입 + d,2d,3d,4d 역행마다 L/5 추가 · "
                      "1440분 보유 · L=8 · 비용 5.88bp/명목 · 파산 = 1분 고저 역행 평가손 ≥ 자본",
             "d": (0.005, 0.01), "n": 12000,
             "metrics": "노출당 bp · 파산율 · 로그성장 · 짝 차이(일 클러스터 CI)"},
}
EVENT_GAP_MS = 180_000
MMR = 0.005
SYMBOLS = ("ETHUSDT", "ETHUSDC")
CAP_KS = (3, 4, 5, 6, 8, 10)
N_BOOT = 4000
N_PLACEBO = 50
SEED = 20261004
VETO_N, VETO_K = 144, 1.0
TAPE_COST_BP = 5.88
DAY = 86_400_000


# ════════════════════════════════════════════════════════════════════════════════
# 0. 수집 (서버 전용)
# ════════════════════════════════════════════════════════════════════════════════
async def _fetch() -> None:
    sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import aiohttp
    import live_binance_account_20260910 as A
    from scripts.binance_ban_guard import ban_remaining
    key, sec = os.getenv("BINANCE_API_KEY", ""), os.getenv("BINANCE_SECRET_KEY", "")
    assert key and sec, "키 없음 -- 서버에서 실행"
    OUT.mkdir(parents=True, exist_ok=True)

    async def get(s, path, params, off, gap):
        if ban_remaining() > 0:
            raise SystemExit("ban 기록 -- 중단")
        page = await A._get(s, path, params, key, sec, off)
        await asyncio.sleep(gap)
        if isinstance(page, dict) and "__error__" in page:
            print(path, params.get("symbol"), page["__error__"], flush=True)
            if page["__error__"].startswith(("418", "429")):
                raise SystemExit("rate limit -- 중단")
        return page

    async with aiohttp.ClientSession() as s:
        off = await A._clock_offset(s)
        now = int(time.time() * 1000) + off
        fills = []
        for sym in SYMBOLS:          # 가능한 오래: API 한도(~180일)부터 7일 창
            cur = now - 180 * DAY + DAY
            while cur < now:
                e = min(cur + 7 * DAY - 60_000, now)
                page = await get(s, "/fapi/v1/userTrades", {"symbol": sym, "startTime": cur, "endTime": e,
                                                              "limit": 1000}, off, 1.5)
                page = page if isinstance(page, list) else []
                fills += page
                print(sym, A._iso(cur), len(page), flush=True)
                cur = int(page[-1]["time"]) + 1 if len(page) >= 1000 else e + 1
        old_path = next(p for p in (ROOT / "tmp/kelly_20261004/income_full.jsonl",
                                    ROOT / "tmp/ledger_fetch_20260927/out/income.jsonl") if p.exists())
        income = [json.loads(x) for x in open(old_path)]
        cur = max(int(r["time"]) for r in income) + 1
        while cur < now:
            e = min(cur + 7 * DAY - 60_000, now)
            page = await get(s, "/fapi/v1/income", {"startTime": cur, "endTime": e, "limit": 1000}, off, 3.0)
            page = page if isinstance(page, list) else []
            income += page
            cur = int(page[-1]["time"]) + 1 if len(page) >= 1000 else e + 1
        bal = await get(s, "/fapi/v2/balance", {}, off, 3.0)
        pos = await get(s, "/fapi/v2/positionRisk", {}, off, 3.0)
    seen, inc = set(), []
    for r in income:
        k = (r.get("tranId"), r.get("incomeType"), r.get("asset"), r.get("time"))
        if k not in seen:
            seen.add(k); inc.append(r)
    with open(OUT / "fills_api.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in fills)
    with open(OUT / "income_full.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in inc)
    json.dump({"balance": bal, "positions": [p for p in (pos if isinstance(pos, list) else [])
                                             if float(p.get("positionAmt", 0)) != 0],
               "fetched_ms": now}, open(OUT / "account_now.json", "w"))
    print(f"fills {len(fills)} · income {len(inc)} (기존 {old_path.name})", flush=True)


# ════════════════════════════════════════════════════════════════════════════════
# 1. 1분봉 · 추세 veto (대시보드 규약)
# ════════════════════════════════════════════════════════════════════════════════
VISION_MONTHLY = Path("/home/kbj20/crypto-scalping/data/binance_vision/klines1m")


def _vision_day(day: str) -> pd.DataFrame:
    p = OUT / "k1m" / f"ETHUSDT-{day}.parquet"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://data.binance.vision/data/futures/um/daily/klines/ETHUSDT/1m/ETHUSDT-1m-{day}.zip"
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=60).read()))
        rows = [r.split(",") for r in z.read(z.namelist()[0]).decode().splitlines() if r and r[0].isdigit()]
        pd.DataFrame({"t": [int(r[0]) for r in rows], "h": [float(r[2]) for r in rows],
                      "l": [float(r[3]) for r in rows], "c": [float(r[4]) for r in rows]}).to_parquet(p)
    return pd.read_parquet(p)


def _deribit_tail(start_ms: int) -> pd.DataFrame:
    """vision 일별 파일이 아직 없는 마지막 날 -- Deribit ETH-PERPETUAL 공개 1분봉(바이낸스 REST 아님)."""
    now = int(time.time() * 1000)
    u = ("https://www.deribit.com/api/v2/public/get_tradingview_chart_data?instrument_name=ETH-PERPETUAL"
         f"&resolution=1&start_timestamp={start_ms}&end_timestamp={now}")
    r = json.loads(urllib.request.urlopen(u, timeout=30).read())["result"]
    return pd.DataFrame({"t": r["ticks"], "h": r["high"], "l": r["low"], "c": r["close"]})


def load_k1m(start: str, end: str) -> pd.DataFrame:
    """ETHUSDT 1분봉 [start, end]. 월 파일(로컬) → 일 파일(vision) → 오늘(Deribit) 순."""
    parts, have = [], -1
    for m in pd.period_range(start[:7], end[:7], freq="M"):
        f = VISION_MONTHLY / f"ETHUSDT-1m-{m}.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["t", "h", "l", "c"]); parts.append(d); have = int(d.t.max())
    day = pd.Timestamp(have + 60_000, unit="ms", tz="UTC").normalize() if have > 0 else pd.Timestamp(start, tz="UTC")
    today = pd.Timestamp.now(tz="UTC").normalize()
    while day <= min(pd.Timestamp(end, tz="UTC"), today):
        try:
            parts.append(_vision_day(day.strftime("%Y-%m-%d")))
        except Exception:                        # 아직 안 올라온 날(어제·오늘) -> Deribit
            parts.append(_deribit_tail(int(day.timestamp() * 1000))); break
        day += pd.Timedelta(days=1)
    k = pd.concat(parts).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    a, b = pd.Timestamp(start, tz="UTC").value // 10**6, (pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).value // 10**6
    return k[(k.t >= a) & (k.t < b)].reset_index(drop=True)


def veto_5m(k1: pd.DataFrame) -> pd.DataFrame:
    """dashboard/server.py::trend_veto_rows 를 그대로 -- 5m 봉, 마감 시각 tc 에서 확정."""
    g = k1.assign(b=k1.t // 300_000 * 300_000).groupby("b")
    b = pd.DataFrame({"h": g.h.max(), "l": g.l.min(), "c": g.c.last()}).reset_index()
    c, h, l = b.c, b.h, b.l
    sma = c.rolling(VETO_N).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / VETO_N, adjust=False).mean()
    dev, eps = ((c - sma) / sma).to_numpy(), (VETO_K * atr / c).to_numpy()
    v, cur = np.zeros(len(b), int), 0
    for i in range(len(b)):
        if np.isfinite(dev[i]) and np.isfinite(eps[i]):
            cur = 1 if dev[i] > eps[i] else -1 if dev[i] < -eps[i] else cur
        v[i] = cur
    return pd.DataFrame({"tc": b.b + 300_000, "veto": v, "z": ((c - sma) / atr).to_numpy(), "close": c})


def veto_at(vf: pd.DataFrame, ms) -> tuple[np.ndarray, np.ndarray]:
    """시각 ms 에 알려진(마감된) 마지막 5m 봉의 (veto, z)."""
    i = np.searchsorted(vf.tc.to_numpy(), np.asarray(ms), side="right") - 1
    ok = i >= 0
    v = np.where(ok, vf.veto.to_numpy()[np.maximum(i, 0)], 0)
    z = np.where(ok, vf.z.to_numpy()[np.maximum(i, 0)], np.nan)
    return v, z


# ════════════════════════════════════════════════════════════════════════════════
# 2. 원장 반사실
# ════════════════════════════════════════════════════════════════════════════════
def load_fills() -> pd.DataFrame:
    rows = {}
    for p in (ROOT / "tmp/ledger_fetch_20260927/out/fills.jsonl", OUT / "fills_api.jsonl"):
        if p.exists():
            for r in map(json.loads, open(p)):
                if r["symbol"] in SYMBOLS:
                    rows[(r["symbol"], int(r["id"]))] = r
    f = pd.DataFrame(rows.values())
    for c in ("price", "qty", "realizedPnl", "commission"):
        f[c] = f[c].astype(float)
    f["time"] = f.time.astype(int)
    f["sgn"] = np.where(f.positionSide == "LONG", 1, -1)
    f["inc"] = ((f.side == "BUY") & (f.positionSide == "LONG")) | ((f.side == "SELL") & (f.positionSide == "SHORT"))
    f = f.sort_values(["time", "id"]).reset_index(drop=True)
    # 스트림 시작이 잘렸으면(API 창 밖 진입) 처음 0 으로 돌아오는 지점까지 버린다
    keep = np.ones(len(f), bool)
    for _, g in f.groupby(["symbol", "positionSide"]):
        s = np.cumsum(np.where(g.inc, g.qty, -g.qty))
        p0 = max(0.0, -s.min())
        if p0 > 1e-9:
            j = int(np.argmax(np.abs(p0 + s) < 1e-6))
            keep[g.index[: j + 1]] = False
    return f[keep].reset_index(drop=True)


def tag_events(f: pd.DataFrame) -> pd.DataFrame:
    """왕복 번호·이벤트 번호(0 = 진입, ≥1 = 추가)를 붙인다. 180초 안 증가 체결은 한 이벤트."""
    f = f.copy(); f["trip"] = -1; f["ev"] = -1
    tid = 0
    for _, g in f.groupby(["symbol", "positionSide"]):
        q, ev, last_inc, cur = 0.0, -1, -10**15, -1
        for i, r in g.iterrows():
            if r.inc:
                if q < 1e-9:
                    cur, ev, tid = tid, 0, tid + 1
                elif r.time - last_inc > EVENT_GAP_MS:
                    ev += 1
                last_inc = r.time
                q += r.qty
            else:
                q -= r.qty
            f.at[i, "trip"], f.at[i, "ev"] = cur, ev
            if q < 1e-9:
                q = 0.0
    return f


def flows_wallet(income: pd.DataFrame):
    """(시각, 지갑 누적, 입출금). USDT<->USDC 지갑 이체 쌍(1시간 안·금액 1% 안 반대 부호)은 같은 시각으로
    묶는다 -- 75초 벌어진 쌍(09-19)이 순자산을 0 근처로 떨궈 NAV 수익률이 발산했다."""
    inc = income.assign(time=income.time.astype(int), income=income.income.astype(float)).sort_values("time")
    tr = inc[inc.incomeType == "TRANSFER"]
    for i, r in tr.iterrows():
        m = tr[(tr.index != i) & ((tr.time - r.time).abs() <= 3_600_000)
               & ((tr.income + r.income).abs() <= 0.01 * abs(r.income))]
        if len(m):
            inc.loc[[i, *m.index], "time"] = max(r.time, int(m.time.max()))
    inc = inc.sort_values("time")
    return inc.time.to_numpy(), np.cumsum(inc.income.to_numpy()), \
        np.where(inc.incomeType.to_numpy() == "TRANSFER", inc.income.to_numpy(), 0.0)


def replay(f: pd.DataFrame, wallet_t, wallet_cum, *, block_ev: set | None = None, cap_k: float | None = None):
    """체결 순서대로 실제/반사실 포지션을 같이 굴린다. 반환: 체결별 상태 배열 + 왕복별 손익."""
    block_ev = block_ev or set()
    n = len(f)
    qa, qc, aa, ac = {}, {}, {}, {}
    cum_a = cum_c = 0.0          # 실현 − 수수료 누적
    st = np.zeros((n, 4, 4))     # (스트림 4개) x (qa, avg_a, qc, avg_c)
    keys = [(s, p) for s in SYMBOLS for p in ("LONG", "SHORT")]
    kidx = {k: i for i, k in enumerate(keys)}
    cum = np.zeros((n, 2))
    trip_a, trip_c = {}, {}
    T, P, Q, INC, SG = f.time.to_numpy(), f.price.to_numpy(), f.qty.to_numpy(), f.inc.to_numpy(), f.sgn.to_numpy()
    RP, CM, TR, EV = f.realizedPnl.to_numpy(), f.commission.to_numpy(), f.trip.to_numpy(), f.ev.to_numpy()
    SYM, PS = f.symbol.to_numpy(), f.positionSide.to_numpy()
    for i in range(n):
        k = (SYM[i], PS[i]); s = SG[i]; px = P[i]; tr = TR[i]
        q_a, q_c = qa.get(k, 0.0), qc.get(k, 0.0)
        rate = CM[i] / (px * Q[i]) if Q[i] > 0 else 0.0
        if INC[i]:
            keep = 0.0 if (tr, EV[i]) in block_ev else Q[i]
            if cap_k is not None and keep > 0:
                w = wallet_cum[np.searchsorted(wallet_t, T[i], side="right") - 1] + (cum_c - cum_a)
                unrl = sum(qc[kk] * (px - ac.get(kk, px)) * (1 if kk[1] == "LONG" else -1) for kk in qc)
                gross = sum(qc[kk] * px for kk in qc)
                keep = min(keep, max(0.0, cap_k * (w + unrl) - gross) / px)
            aa[k] = (aa.get(k, 0.0) * q_a + px * Q[i]) / (q_a + Q[i]); qa[k] = q_a + Q[i]
            if keep > 0:
                ac[k] = (ac.get(k, 0.0) * q_c + px * keep) / (q_c + keep)
            qc[k] = q_c + keep
            da, dc = -CM[i], -rate * px * keep
        else:
            frac = min(1.0, Q[i] / q_a) if q_a > 1e-12 else 1.0
            close_c = frac * q_c
            da = (px - aa.get(k, px)) * Q[i] * s - CM[i]
            dc = (px - ac.get(k, px)) * close_c * s - rate * px * close_c
            qa[k] = max(0.0, q_a - Q[i]); qc[k] = q_c - close_c
            if qa[k] < 1e-9:
                qa[k] = qc[k] = 0.0
        cum_a += da; cum_c += dc
        trip_a[tr] = trip_a.get(tr, 0.0) + da; trip_c[tr] = trip_c.get(tr, 0.0) + dc
        cum[i] = cum_a, cum_c
        for kk, j in kidx.items():
            st[i, j] = qa.get(kk, 0.0), aa.get(kk, 0.0), qc.get(kk, 0.0), ac.get(kk, 0.0)
    return st, cum, trip_a, trip_c


def minute_path(f, st, cum, k1, wallet_t, wallet_cum, flow, t0, t1):
    """1분 격자에서 실제/반사실 순자산·역행 평가손·명목·청산거리."""
    k = k1[(k1.t >= t0) & (k1.t <= t1)]
    te = k.t.to_numpy() + 60_000                         # 분 마감
    j = np.searchsorted(f.time.to_numpy(), te, side="right") - 1
    S = np.where(j[:, None, None] >= 0, st[np.maximum(j, 0)], 0.0)    # (m, 4, 4)
    C = np.where(j[:, None] >= 0, cum[np.maximum(j, 0)], 0.0)
    sg = np.array([1, -1, 1, -1.0])
    c, h, l = k.c.to_numpy()[:, None], k.h.to_numpy()[:, None], k.l.to_numpy()[:, None]
    adv = np.where(sg > 0, l, h)
    out = {}
    wi = np.searchsorted(wallet_t, te, side="right") - 1
    wal = np.where(wi >= 0, wallet_cum[np.maximum(wi, 0)], 0.0)
    F = np.concatenate([[0.0], np.cumsum(flow)])[wi + 1]
    fl = np.diff(F, prepend=F[0])
    for tag, qi, ai, ci in (("a", 0, 1, 0), ("c", 2, 3, 1)):
        q, avg = S[:, :, qi], S[:, :, ai]
        U = (q * (c - avg) * sg).sum(1); Uadv = (q * (adv - avg) * sg).sum(1)
        E = wal + (C[:, ci] - C[:, 0]) + U                # 반사실 지갑 = 실제 지갑 + 누적 차이
        gross = (q * c).sum(1); net = (q * c * sg).sum(1)
        out[tag] = dict(E=E, Eadv=E - U + Uadv, gross=gross, Ustream=q * (adv - avg) * sg, Uend=q[-1] * (c[-1] - avg[-1]) * sg,
                        liq=np.where(np.abs(net) > 1, (E - MMR * gross) / np.maximum(np.abs(net), 1e-9), np.inf))
    out["t"], out["flow"] = te, fl
    return out


def nav_mdd(E: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, float]:
    r = np.zeros_like(E)
    r[1:] = (E[1:] - E[:-1] - flow[1:]) / np.maximum(E[:-1], 1e-9)
    nav = np.cumprod(1 + r)
    return r, float((nav / np.maximum.accumulate(nav) - 1).min())


def mdd_of(r: np.ndarray) -> float:
    nav = np.cumprod(1 + r)
    return float((nav / np.maximum.accumulate(nav) - 1).min())


RULES = {"G0": ("G0", None), "G1": ("G1", None), "G2": ("G2", None),
         **{f"C{k}": (None, k) for k in CAP_KS},
         "G1+C4": ("G1", 4), "G1+C6": ("G1", 6), "G2+C4": ("G2", 4), "G2+C6": ("G2", 6)}   # 조합은 사전 고정
STREAMS = [(s, p) for s in SYMBOLS for p in ("LONG", "SHORT")]


def _ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


def evaluate(f, trips, path_args, block, cap):
    st, cum, ta, tc = replay(f, path_args[1], path_args[2], block_ev=block, cap_k=cap)
    mp = minute_path(f, st, cum, *path_args)
    last_trip = {k: f[(f.symbol == k[0]) & (f.positionSide == k[1])].trip.max() for k in STREAMS}
    last_trip = {k: (-1 if pd.isna(v) else int(v)) for k, v in last_trip.items()}
    res = {}
    for tag, tp in (("a", ta), ("c", tc)):
        m = mp[tag]
        tp = dict(tp)
        for j, k in enumerate(STREAMS):           # 열린 왕복은 마지막 분 종가로 평가
            if m["Uend"][j] != 0 and last_trip[k] >= 0:
                tp[last_trip[k]] = tp.get(last_trip[k], 0.0) + m["Uend"][j]
        r, mdd = nav_mdd(m["E"], mp["flow"])
        mae = {}
        for _, t in trips.iterrows():
            ii = slice(np.searchsorted(mp["t"], t.t0, "right"), np.searchsorted(mp["t"], t.t1, "right") + 1)
            u = m["Ustream"][ii, t.j]
            e0 = m["E"][max(ii.start - 1, 0)]
            mae[t.trip] = float(u.min() / e0) if len(u) else 0.0
        res[tag] = dict(pnl=float(sum(tp.values())), trip=tp, r=r, mdd=mdd, mae=mae,
                        lev=float(np.max(m["gross"] / np.maximum(m["E"], 1e-9))),
                        liq=float(np.min(m["liq"])), nav=float(np.prod(1 + r) - 1))
    return res, st, mp


def run_ledger() -> dict:
    f = tag_events(load_fills())
    inc = pd.DataFrame(map(json.loads, open(OUT / "income_full.jsonl")))
    wt, wc, flow = flows_wallet(inc)
    now = json.load(open(OUT / "account_now.json"))["fetched_ms"]
    t0 = int(f.time.min()) // 60_000 * 60_000
    k1 = load_k1m(pd.Timestamp(t0 - 5 * DAY, unit="ms").strftime("%Y-%m-%d"),
                  pd.Timestamp(now, unit="ms").strftime("%Y-%m-%d"))
    vf = veto_5m(k1)
    k1 = k1[k1.t + 60_000 <= now]
    path_args = (k1, wt, wc, flow, t0, int(k1.t.max()))

    g = f[f.inc].groupby(["trip", "ev"]).agg(time=("time", "first"), sgn=("sgn", "first"), px=("price", "first"),
                                     qty=("qty", "sum"), inc=("inc", "first"), sym=("symbol", "first")).reset_index()
    ev = g[g.inc].copy()
    ev["veto"], ev["z"] = veto_at(vf, ev.time.to_numpy())
    ev["against"] = ev.veto == -ev.sgn
    adds = ev[ev.ev >= 1]
    sets = {"G0": set(zip(adds.trip, adds.ev)),
            "G1": set(zip(adds[adds.against].trip, adds[adds.against].ev)),
            "G2": set(zip(adds[adds.against & (adds.z.abs() >= 2)].trip, adds[adds.against & (adds.z.abs() >= 2)].ev))}
    trips = f.groupby("trip").agg(t0=("time", "min"), t1=("time", "max"), sym=("symbol", "first"),
                                   ps=("positionSide", "first")).reset_index()
    trips["j"] = [STREAMS.index((a, b)) for a, b in zip(trips.sym, trips.ps)]
    trips["day"] = trips.t0 // DAY
    open_tr = {f[(f.symbol == k[0]) & (f.positionSide == k[1])].trip.max() for k in STREAMS}
    # 열린 왕복은 지금까지
    trips.loc[trips.trip.isin(open_tr), "t1"] = trips.loc[trips.trip.isin(open_tr), "t1"].clip(lower=int(k1.t.max()))
    # 회계 항등식: 재생한 실제 손익 == 거래소 realizedPnl − 수수료
    base, st, mp = evaluate(f, trips, path_args, set(), None)
    ident = float((f.realizedPnl - f.commission).sum()) - float(replay(f, wt, wc)[1][-1, 0])
    out = {"data": {"fills": len(f), "trips": int(trips.trip.nunique()), "add_events": len(adds),
                    "adds_against_veto": int(adds.against.sum()), "adds_G2": len(sets["G2"]),
                    "first": pd.Timestamp(f.time.min(), unit="ms", tz="UTC").isoformat(),
                    "last": pd.Timestamp(f.time.max(), unit="ms", tz="UTC").isoformat(),
                    "days": int(trips.day.nunique()),
                    "equity_start": float(mp["a"]["E"][0]), "equity_end": float(mp["a"]["E"][-1]),
                    "transfers": float(flow.sum())}}
    rng = np.random.default_rng(SEED)
    day_of = trips.set_index("trip").day
    days = np.array(sorted(trips.day.unique()))
    bidx = rng.integers(0, len(days), (N_BOOT, len(days)))
    tidx = rng.integers(0, len(trips), (N_BOOT, len(trips)))
    mday = mp["t"] // DAY
    q15 = (mp["t"] - mp["t"][0]) // 900_000

    def r15(r):          # 15분 복리 수익률(부트스트랩 가볍게)
        return pd.Series(np.log1p(r)).groupby(q15).sum().pipe(np.expm1).to_numpy()
    d15 = pd.Series(mday).groupby(q15).first().to_numpy()
    blocks = [np.where(d15 == d)[0] for d in np.unique(d15)]

    def compare(res):
        tr = trips.trip.to_numpy()
        da = np.array([res["c"]["trip"].get(t, 0.0) - res["a"]["trip"].get(t, 0.0) for t in tr])
        ma = np.array([res["a"]["mae"][t] for t in tr]); mc = np.array([res["c"]["mae"][t] for t in tr])
        dsum = pd.Series(da).groupby(trips.day.to_numpy()).sum().reindex(days).to_numpy()
        pnl_day = dsum[bidx].sum(1)
        pnl_trip = da[tidx].sum(1)
        # 최대 평가손: 일 클러스터로 왕복을 뽑아 «최악 값» 차이
        tday = trips.day.to_numpy()
        by_day = [np.where(tday == d)[0] for d in days]
        w = np.array([[ma[np.concatenate([by_day[i] for i in row])].min(),
                       mc[np.concatenate([by_day[i] for i in row])].min()] for row in bidx[:1000]])
        ra, rc = r15(res["a"]["r"]), r15(res["c"]["r"])
        bl = rng.integers(0, len(blocks), (1000, len(blocks)))
        mm = np.array([[mdd_of(ra[np.concatenate([blocks[i] for i in row])]),
                        mdd_of(rc[np.concatenate([blocks[i] for i in row])])] for row in bl])
        a, c = res["a"], res["c"]
        return {"pnl_a": a["pnl"], "pnl_c": c["pnl"], "dpnl": c["pnl"] - a["pnl"],
                "dpnl_ci_day": _ci(pnl_day), "dpnl_ci_trip": _ci(pnl_trip),
                "mdd_a": a["mdd"], "mdd_c": c["mdd"], "dmdd_ci_day": _ci(mm[:, 1] - mm[:, 0]),
                "worst_trip_a": min(a["trip"].values()), "worst_trip_c": min(c["trip"].values()),
                "mae_a": float(ma.min()), "mae_c": float(mc.min()), "dmae_ci_day": _ci(w[:, 1] - w[:, 0]),
                "lev_a": a["lev"], "lev_c": c["lev"], "liq_a": a["liq"], "liq_c": c["liq"],
                "ret_mdd_a": a["nav"] / abs(a["mdd"]), "ret_mdd_c": c["nav"] / abs(c["mdd"]),
                "trips_c": int(sum(1 for t in tr if abs(c["trip"].get(t, 0.0)) > 1e-9))}

    rules = {}
    for name, (gname, cap) in RULES.items():
        res, _, _ = evaluate(f, trips, path_args, sets[gname] if gname else set(), cap)
        cmp_ = compare(res)
        risk = cmp_["dmdd_ci_day"][0] > 0 and cmp_["dmae_ci_day"][0] > 0
        noninf = cmp_["dpnl_ci_day"][0] > -0.10 * abs(cmp_["pnl_a"])
        cmp_["verdict"] = {"risk_sig": bool(risk), "pnl_noninferior": bool(noninf), "pass": bool(risk and noninf)}
        cmp_["blocked_events"] = len(sets[gname]) if gname else 0
        rules[name] = cmp_
        print(f"{name:7s} 실제 {cmp_['pnl_a']:+.0f} Δ손익 {cmp_['dpnl']:+8.1f} CI{np.round(cmp_['dpnl_ci_day'],1)} · MDD {cmp_['mdd_a']:+.3f}→{cmp_['mdd_c']:+.3f} "
              f"CI{np.round(cmp_['dmdd_ci_day'],3)} · 최대평가손 {cmp_['mae_a']:+.3f}→{cmp_['mae_c']:+.3f} "
              f"CI{np.round(cmp_['dmae_ci_day'],3)} · 판정 {cmp_['verdict']}", flush=True)
    # 위약: G1 이 막은 개수만큼 무작위
    allk = sorted(sets["G0"]); n1 = len(sets["G1"])
    plac = []
    for sd in range(N_PLACEBO):
        r2 = np.random.default_rng(SEED + 1 + sd)
        pick = {allk[i] for i in r2.choice(len(allk), n1, replace=False)}
        res, _, _ = evaluate(f, trips, path_args, pick, None)
        plac.append([res["c"]["pnl"] - res["a"]["pnl"], res["c"]["mdd"], min(res["c"]["mae"].values())])
    plac = np.array(plac)
    g1 = rules["G1"]
    out["placebo"] = {"dpnl_q": np.quantile(plac[:, 0], [0.05, 0.5, 0.95]).tolist(),
                      "mdd_q": np.quantile(plac[:, 1], [0.05, 0.5, 0.95]).tolist(),
                      "mae_q": np.quantile(plac[:, 2], [0.05, 0.5, 0.95]).tolist(),
                      "G1_dpnl_pct": float((plac[:, 0] < g1["dpnl"]).mean()),
                      "G1_mdd_pct": float((plac[:, 1] < g1["mdd_c"]).mean()),
                      "G1_mae_pct": float((plac[:, 2] < g1["mae_c"]).mean())}
    out["rules"] = rules
    out["identity_gap_usd"] = ident
    out["open_now"] = {"|".join(k): (float(st[-1, j, 0]), float(st[-1, j, 1])) for j, k in enumerate(STREAMS) if st[-1, j, 0] > 0}
    out["events"] = ev.assign(t=pd.to_datetime(ev.time, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").astype(str)
                              ).drop(columns=["inc"]).to_dict("records")
    # 추가 이벤트 요약: 측면 x veto
    out["adds_table"] = adds.groupby(["sgn", "against"]).size().rename("n").reset_index().to_dict("records")
    return out, f, trips, path_args, sets, ev


def weekend(f, trips, path_args, sets, ev, t_from="2026-10-02T10:00:00Z") -> list[dict]:
    """이번 주말 왕복(10-02 KST 저녁 진입) — 규칙별 경로 요약."""
    a = pd.Timestamp(t_from).value // 10**6
    sel = trips[trips.t0 >= a]
    # 이번 주말 체결만 따로 굴린다 -- 전 구간으로 굴리면 08-01 부터 쌓인 손익 차이가 반사실 순자산을 바꿔
    # 상한이 «그때 실제 순자산»이 아닌 값으로 걸린다. 여기서는 규칙을 이 왕복에만 적용한다.
    f = f[f.trip.isin(sel.trip)].reset_index(drop=True)
    rows = []
    for name, (gname, cap) in {"실제": (None, None), **RULES}.items():
        res, st, mp = evaluate(f, sel, path_args, sets[gname] if gname else set(), cap)
        key = "a" if name == "실제" else "c"
        for _, t in sel.iterrows():
            ii = slice(np.searchsorted(mp["t"], t.t0, "right"), np.searchsorted(mp["t"], t.t1, "right") + 1)
            fi = np.where(f.trip.to_numpy() == t.trip)[0]
            qcol = 0 if key == "a" else 2
            rows.append({"rule": name, "trip": int(t.trip), "side": t.ps, "sym": t.sym,
                         "peak_qty": float(st[fi, t.j, qcol].max()),
                         "peak_lev": float(np.max(mp[key]["gross"][ii] / mp[key]["E"][ii])),
                         "worst_unrl": float(mp[key]["Ustream"][ii, t.j].min()),
                         "worst_unrl_pct": res[key]["mae"][t.trip],
                         "worst_unrl_pct_of_actual_E": float(mp[key]["Ustream"][ii, t.j].min()
                                                             / mp["a"]["E"][max(ii.start - 1, 0)]),
                         "pnl_incl_open": res[key]["trip"].get(t.trip, 0.0)})
    return rows



# ════════════════════════════════════════════════════════════════════════════════
# 3. 테이프 일반화 (09-13 시뮬레이터 모양 + 게이트)
# ════════════════════════════════════════════════════════════════════════════════
def ladder_trial(px, adv, s, d, H, L, kept_fn):
    """한 시행: L/5 진입(px[0]) + 역행 d,2d,3d,4d 터치마다 L/5 추가. adv = 1..H 분의 역행 극값(롱=저가).
    kept_fn(k, j) -> 이 추가를 하나. 반환 (수익/자본, 평균 노출, 파산, 추가 시도 수, 막힌 수)."""
    u, e = L / 5, px[0]
    lots_p, lots_j = [e], [0]
    tried = blocked = 0
    for k in range(1, 5):
        lvl = e * (1 - s * k * d)
        hit = adv <= lvl if s > 0 else adv >= lvl
        if not hit.any():
            break
        j = int(np.argmax(hit)); tried += 1
        if kept_fn(k, j):
            lots_p.append(lvl); lots_j.append(j)
        else:
            blocked += 1
    m = np.arange(H)
    held = (m[None, :] >= np.array(lots_j)[:, None])                  # (lots, H)
    pnl_adv = (held * (s * (adv[None, :] / np.array(lots_p)[:, None] - 1))).sum(0) * u
    expo = held.sum(0).mean() * u
    if pnl_adv.min() <= -1.0:
        return -1.0, expo, 1, tried, blocked
    ret = sum(u * (s * (px[-1] / p - 1) - TAPE_COST_BP / 1e4) for p in lots_p)
    return ret, expo, 0, tried, blocked


def run_tape() -> dict:
    T = CRITERIA["tape"]
    k1 = load_k1m("2021-12-20", pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    vf = veto_5m(k1)
    t, c, h, l = k1.t.to_numpy(), k1.c.to_numpy(), k1.h.to_numpy(), k1.l.to_numpy()
    v_open, z_open = veto_at(vf, t)                       # 분 시작에 알려진 마감 5m 봉
    H, L = 1440, 8.0
    out = {}
    for per, (a0, a1) in T["periods"].items():
        ia = int(np.searchsorted(t, pd.Timestamp(a0, tz="UTC").value // 10**6))
        ib = int(np.searchsorted(t, (pd.Timestamp(a1, tz="UTC") + pd.Timedelta(days=1)).value // 10**6))
        ia = max(ia, 3000)                                   # veto 워밍업
        for d in T["d"]:
            rng = np.random.default_rng(SEED + int(d * 1e4) + (0 if per == "main" else 7))
            n = T["n"]
            starts = rng.integers(ia, ib - H - 2, n); sides = rng.choice([-1.0, 1.0], n)
            arms = {k: np.zeros((n, 5)) for k in ("none", "G1", "G2")}   # Gr 은 G1 차단률을 안 뒤에
            for i in range(n):
                a, sd = int(starts[i]), float(sides[i])
                px = c[a:a + H + 1]; adv = (l if sd > 0 else h)[a + 1:a + 1 + H]
                vv, zz = v_open[a + 1:a + 1 + H], z_open[a + 1:a + 1 + H]
                arms["none"][i] = ladder_trial(px, adv, sd, d, H, L, lambda k, j: True)
                arms["G1"][i] = ladder_trial(px, adv, sd, d, H, L, lambda k, j: vv[j] != -sd)
                arms["G2"][i] = ladder_trial(px, adv, sd, d, H, L,
                                             lambda k, j: not (vv[j] == -sd and abs(zz[j]) >= 2))
            p_block = arms["G1"][:, 4].sum() / max(arms["G1"][:, 3].sum(), 1)
            r2 = np.random.default_rng(SEED + 99)
            arms["Gr"] = np.zeros((n, 5))
            for i in range(n):
                a, sd = int(starts[i]), float(sides[i])
                px = c[a:a + H + 1]; adv = (l if sd > 0 else h)[a + 1:a + 1 + H]
                draws = r2.random(4)
                arms["Gr"][i] = ladder_trial(px, adv, sd, d, H, L, lambda k, j: draws[k - 1] >= p_block)
            day = t[starts] // DAY
            udays = np.unique(day); bi = rng.integers(0, len(udays), (N_BOOT, len(udays)))
            grp = [np.where(day == x)[0] for x in udays]
            sums = {k: np.array([[v[g, 0].sum(), v[g, 1].sum(), np.log1p(np.maximum(v[g, 0], -0.999999)).sum(), len(g)]
                                 for g in grp]) for k, v in arms.items()}
            res = {}
            for k, v in arms.items():
                ret, ex = v[:, 0], v[:, 1]
                res[k] = {"per_expo_bp": float(ret.mean() / ex.mean() * 1e4), "ret_bp": float(ret.mean() * 1e4),
                          "expo": float(ex.mean()), "ruin": float(v[:, 2].mean()),
                          "growth": float(np.mean(np.log1p(np.maximum(ret, -0.999999)))),
                          "adds_tried": int(v[:, 3].sum()), "adds_blocked": int(v[:, 4].sum())}
                if k != "none":
                    S0, S1 = sums["none"][bi].sum(1), sums[k][bi].sum(1)
                    res[k]["d_per_expo_ci"] = _ci((S1[:, 0] / S1[:, 1] - S0[:, 0] / S0[:, 1]) * 1e4)
                    res[k]["d_growth_ci"] = _ci((S1[:, 2] - S0[:, 2]) / S0[:, 3])
                    res[k]["d_ruin"] = float(v[:, 2].mean() - arms["none"][:, 2].mean())
                if k in ("G1", "G2"):                # 선택성: 같은 비율 무작위 차단(Gr) 대비
                    S0, S1 = sums["Gr"][bi].sum(1), sums[k][bi].sum(1)
                    res[k]["vs_Gr_per_expo_ci"] = _ci((S1[:, 0] / S1[:, 1] - S0[:, 0] / S0[:, 1]) * 1e4)
                    res[k]["vs_Gr_growth_ci"] = _ci((S1[:, 2] - S0[:, 2]) / S0[:, 3])
            res["p_block_G1"] = float(p_block)
            out[f"{per}_d{d}"] = res
            print(per, d, json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in x.items()}
                                     if isinstance(x, dict) else x for k, x in res.items()}, default=_json), flush=True)
    return out



def selftest() -> None:
    def mk(rows):   # (time_s, side, price, qty)
        f = pd.DataFrame([{"symbol": "ETHUSDT", "positionSide": "LONG", "side": sd, "price": px, "qty": q,
                           "realizedPnl": 0.0, "commission": 0.0, "time": int(t * 1000), "id": i}
                          for i, (t, sd, px, q) in enumerate(rows)])
        f["sgn"] = 1; f["inc"] = f.side == "BUY"
        return tag_events(f)
    W = (np.array([0]), np.array([100.0]))
    # ① 추가 차단 + 비례 청산: 실제 avg 95 로 2개 청산 = 0 / 반사실 1개 @100 -> 95 = −5
    f = mk([(0, "BUY", 100, 1), (1000, "BUY", 90, 1), (2000, "SELL", 95, 2)])
    assert list(f.ev) == [0, 1, 1], f.ev
    _, cum, _, _ = replay(f, *W, block_ev={(0, 1)})
    assert abs(cum[-1, 0]) < 1e-9 and abs(cum[-1, 1] + 5) < 1e-9, cum[-1]
    # ② 분할 청산 비율: 반사실 0.5 씩 두 번 -> +5 +10
    f = mk([(0, "BUY", 100, 1), (1000, "BUY", 100, 1), (2000, "SELL", 110, 1), (3000, "SELL", 120, 1)])
    _, cum, _, _ = replay(f, *W, block_ev={(0, 1)})
    assert abs(cum[-1, 0] - 30) < 1e-9 and abs(cum[-1, 1] - 15) < 1e-9, cum[-1]
    # ③ 180초 안 증가 체결은 한 이벤트(진입), 넘으면 추가
    f = mk([(0, "BUY", 100, 1), (170, "BUY", 100, 1), (400, "BUY", 100, 1), (500, "SELL", 100, 3)])
    assert list(f.ev) == [0, 0, 1, 1], f.ev
    # ④ 명목 상한: 순자산 100 · k=3 -> 5개 요청 중 3개만
    f = mk([(0, "BUY", 100, 5), (1000, "SELL", 100, 5)])
    st, _, _, _ = replay(f, *W, cap_k=3)
    assert abs(st[0, 0, 2] - 3) < 1e-9 and abs(st[0, 0, 0] - 5) < 1e-9, st[0, 0]
    # ⑤ veto 히스테리시스: 오르면 +1, 밴드 안으로 돌아와도 유지, 크게 빠지면 −1
    n = 300 * 5
    px = np.r_[np.full(n, 100.0), np.linspace(100, 110, n), np.full(200, 109.9), np.linspace(109.9, 90, n)]
    k1 = pd.DataFrame({"t": np.arange(len(px)) * 60_000, "h": px + 0.05, "l": px - 0.05, "c": px})
    vf = veto_5m(k1)
    v = vf.veto.to_numpy()
    assert v[600] == 1 and v[len(v) - 1] == -1 and set(np.unique(v)) <= {-1, 0, 1}
    assert veto_at(vf, [0])[0][0] == 0                               # 첫 봉 마감 전 = 모름
    # ⑥ 사다리: 1% 역행 4단 모두 터치 후 100 으로 회복
    adv = np.r_[np.linspace(100, 95.9, 100), np.full(1340, 100)]
    pxs = np.r_[100, adv]
    r, ex, ruin, tried, blk = ladder_trial(pxs, adv, 1.0, 0.01, 1440, 8.0, lambda k, j: True)
    want = sum(1.6 * (100 / p - 1) for p in (100, 99, 98, 97, 96)) - 8.0 * TAPE_COST_BP / 1e4
    assert tried == 4 and ruin == 0 and abs(r - want) < 1e-9, (r, want)
    r2, _, _, _, b2 = ladder_trial(pxs, adv, 1.0, 0.01, 1440, 8.0, lambda k, j: False)
    assert b2 == 4 and abs(r2 - (-1.6 * TAPE_COST_BP / 1e4)) < 1e-12, r2
    # ⑦ 지갑 이체 쌍은 같은 시각으로 묶인다
    inc = pd.DataFrame({"time": [0, 75_000, 200_000], "income": [-100.0, 100.0, 5.0],
                        "incomeType": ["TRANSFER", "TRANSFER", "REALIZED_PNL"], "asset": ["USDT", "USDC", "USDT"]})
    t, w, fl = flows_wallet(inc)
    assert t[0] == t[1] == 75_000 and abs(w[-1] - 5) < 1e-12
    print("selftest OK -- 비례 청산·이벤트 묶음·명목 상한·veto 히스테리시스·사다리·이체 쌍")


def _json(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, tuple): return list(o)
    return str(o)


def main() -> int:
    ap = argparse.ArgumentParser()
    for m in ("--fetch", "--ledger", "--tape", "--selftest"):
        ap.add_argument(m, action="store_true")
    a = ap.parse_args()
    if a.fetch:
        asyncio.run(_fetch()); return 0
    if a.selftest:
        selftest(); return 0
    OUT.mkdir(parents=True, exist_ok=True)
    if a.ledger:
        out, f, trips, pa, sets, ev = run_ledger()
        out["weekend"] = weekend(f, trips, pa, sets, ev)
        json.dump(out, open(OUT / "ledger_result.json", "w"), ensure_ascii=False, indent=1, default=_json)
        print(json.dumps({k: out[k] for k in ("data", "placebo", "identity_gap_usd", "open_now")},
                         ensure_ascii=False, default=_json, indent=1))
    if a.tape:
        res = run_tape()
        json.dump(res, open(OUT / "tape_result.json", "w"), ensure_ascii=False, indent=1, default=_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
