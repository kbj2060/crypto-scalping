#!/usr/bin/env python3
"""추세 롱 코어(V0) **전진 기록기** -- docs/operating_logic_20261114.md 1층 · 규칙 = 11-10 trend_voltarget 규칙 D(조율 없음).

매일 UTC 00:00 에 닫힌 일봉 D 까지의 종가로 «D+1 에 들 비중»을 계산해 data/live/trend_core_forward.jsonl 에 한 줄 남긴다.
그날의 종가·펀딩도 같이 남겨 장부만으로 손익을 다시 셀 수 있다. 늦은 실행(일봉 마감 26시간 뒤)·중복 날짜·코인 빠짐은 기록 거부.
원천: 바이낸스 USDⓈ-M 선물 1d klines·fundingRate (읽기 전용, 주문 호출 없음). **서버에서만 실행**(로컬 REST = IP 공유).
  python scripts/live_trend_core_forward_20261114.py            # cron(서버, 매일 09:30·12:30 KST -- 두 번째는 첫 번째가 실패했을 때만 씀)
  python scripts/live_trend_core_forward_20261114.py --status   # 누적 손익 vs 20코인 그냥 보유 · 낙폭 · 마지막 비중
  python scripts/live_trend_core_forward_20261114.py --selftest # 네트워크 없음
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
LEDGER = ROOT / "data/live/trend_core_forward.jsonl"
COINS = ["ADA", "APT", "ARB", "ATOM", "AVAX", "BNB", "BTC", "DOGE", "DOT", "ETH", "FIL", "INJ", "LINK", "LTC", "NEAR", "OP", "SOL", "SUI", "TRX", "XRP"]
TARGET, COST, LATE_H, HALT_DD = 0.5, 5e-4, 26, -0.25
DAY = 86_400_000


def weights(C: pd.DataFrame):
    """행 D = D 종가까지로 정한 비중(다음 날 보유). 11-10 규칙 D 와 같은 식."""
    ret = C.pct_change(fill_method=None)
    score = sum(np.sign(C / C.shift(k) - 1) for k in (7, 14, 21, 28)) / 4
    vol = ret.rolling(28).std() * np.sqrt(365)
    vw = (TARGET / vol).clip(upper=1)
    ok = (score.notna() & vw.notna()).astype(float)
    n = ok.sum(1).replace(0, np.nan)
    return (score.clip(lower=0) * vw * ok).fillna(0).div(n, axis=0).fillna(0), score, vol


def fetch(now_ms: int):
    import requests
    C, F = {}, {}
    for c in COINS:
        k = requests.get("https://fapi.binance.com/fapi/v1/klines", params={"symbol": f"{c}USDT", "interval": "1d", "limit": 60}, timeout=30).json()
        k = [x for x in k if x[6] < now_ms]                                 # 닫힌 봉만
        C[c] = pd.Series([float(x[4]) for x in k], index=pd.to_datetime([x[0] for x in k], unit="ms"))
        time.sleep(0.2)
    C = pd.DataFrame(C).sort_index()
    d_ms = int(C.index[-1].timestamp() * 1000)
    for c in COINS:                                                         # 일봉 D 의 펀딩 = (D 00:00, D+1 00:00] 정산 합(연구 규약)
        f = requests.get("https://fapi.binance.com/fapi/v1/fundingRate", params={"symbol": f"{c}USDT", "startTime": d_ms + 1, "endTime": d_ms + DAY, "limit": 10}, timeout=30).json()
        F[c] = float(sum(float(x["fundingRate"]) for x in f))
        time.sleep(0.2)
    return C, F


def make_record(C: pd.DataFrame, F: dict, now: dt.datetime) -> dict:
    W, score, vol = weights(C)
    d = C.index[-1]
    if C.loc[d].isna().any() or len(C) < 30:
        raise ValueError(f"코인 빠짐·길이 부족: {C.loc[d][C.loc[d].isna()].index.tolist()} · {len(C)}일")
    r = lambda s: {c: round(float(s[c]), 8) for c in COINS}
    return dict(tau=str(d.date()), recorded_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"), rule="V0 max(score,0)*min(0.5/vol28,1)/n",
                weights=r(W.loc[d]), gross=round(float(W.loc[d].sum()), 8), score=r(score.loc[d]), vol28=r(vol.loc[d]), close=r(C.loc[d]), funding=r(pd.Series(F)))


def too_late(now: dt.datetime, tau: str) -> bool:
    close = dt.datetime.fromisoformat(tau).replace(tzinfo=dt.timezone.utc) + dt.timedelta(days=1)
    return (now - close) > dt.timedelta(hours=LATE_H)


def read(ledger: Path = LEDGER) -> list[dict]:
    return [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()] if ledger.exists() else []


def append(rec: dict, ledger: Path = LEDGER) -> bool:
    if any(r["tau"] == rec["tau"] for r in read(ledger)):
        return False                                                        # 덮어쓰기·사후 메움 금지
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return True


def pnl(rows: list[dict]) -> pd.DataFrame:
    """날짜 D 손익 = Σ w_{D-1}·(종가 수익 − 펀딩_D) − 편도 5bp × |w_{D-1} − w_{D-2}|. 연속 날짜만 센다(빠진 날 뒤 하루는 건너뜀)."""
    by = {pd.Timestamp(r["tau"]): r for r in rows}
    out = []
    for d in sorted(by):
        p, pp = by.get(d - pd.Timedelta(days=1)), by.get(d - pd.Timedelta(days=2))
        if p is None:
            continue
        cur = by[d]
        ret = {c: cur["close"][c] / p["close"][c] - 1 for c in COINS}
        w = p["weights"]; w0 = pp["weights"] if pp else {c: 0.0 for c in COINS}
        core = sum(w[c] * (ret[c] - cur["funding"][c]) for c in COINS) - COST * sum(abs(w[c] - w0[c]) for c in COINS)
        out.append(dict(day=d, core=core, hold=float(np.mean([ret[c] - cur["funding"][c] for c in COINS])), gross=p["gross"]))
    return pd.DataFrame(out)


def status(ledger: Path = LEDGER) -> dict:
    rows = read(ledger)
    if not rows:
        print("기록 없음"); return {}
    x = pnl(rows)
    last = rows[-1]
    s = dict(records=len(rows), first=rows[0]["tau"], last=last["tau"], pnl_days=int(len(x)))
    if len(x):
        for k in ("core", "hold"):
            eq = (1 + x[k]).cumprod()
            s[k] = dict(cum_pct=float((eq.iloc[-1] - 1) * 100), mdd_pct=float((eq / eq.cummax() - 1).min() * 100))
        s["mean_gross"] = float(x.gross.mean())
        s["halt"] = s["core"]["mdd_pct"] / 100 <= HALT_DD
    held = {c: w for c, w in last["weights"].items() if w > 0}
    print(json.dumps(s, ensure_ascii=False))
    print(f"{last['tau']} 종가 기준 다음 날 비중 · 총 {last['gross']:.3f}:", {c: round(w, 4) for c, w in sorted(held.items(), key=lambda t: -t[1])} or "전부 현금")
    if s.get("halt"):
        print(f"⚠ 코어 낙폭 {s['core']['mdd_pct']:.1f}% ≤ {HALT_DD * 100:.0f}% -- 운영 로직 4층 규칙: 멈추고 다시 볼 것")
    return s


def main() -> int:
    now = dt.datetime.now(dt.timezone.utc)
    C, F = fetch(int(now.timestamp() * 1000))
    rec = make_record(C, F, now)
    if too_late(now, rec["tau"]):
        print(f"{now:%F %T} 늦은 실행: {rec['tau']} 일봉 마감 뒤 {LATE_H}시간 초과 -- 기록 안 함", file=sys.stderr); return 2
    ok = append(rec)
    print(f"{now:%F %T} tau {rec['tau']} {'기록' if ok else '이미 있음'} · 총 비중 {rec['gross']:.3f}")
    return 0


def selftest():
    import tempfile
    # 1) 연구 규칙과 같은 비중(로컬 일봉 사본이 있을 때)
    tv = Path("/home/kbj20/crypto-scalping/.claude/worktrees/direction-prediction-model-test-085284/scripts")
    if (tv / "research_trend_voltarget_long_20261110.py").exists():
        sys.path.insert(0, str(tv)); import research_trend_voltarget_long_20261110 as T
        C, _ = T.load()
        assert list(C.columns) == COINS
        assert np.allclose(weights(C)[0].shift(1).fillna(0).to_numpy(), T.weights(C)["D"].fillna(0).to_numpy(), atol=1e-12)
        W, _, _ = weights(C); W2, _, _ = weights(C.iloc[-60:])              # 서버는 60일만 받는다 → 마지막 날 비중이 같아야 함
        assert np.allclose(W.iloc[-1].to_numpy(), W2.iloc[-1].to_numpy(), atol=1e-12)
        print("  연구 규칙 D 와 일치 · 60일 창 마지막 비중 일치")
    # 2) 장부: 합성 3일 → 손익·중복·늦은 실행
    idx = pd.date_range("2026-01-01", periods=40, freq="D")
    C = pd.DataFrame({c: 100 * np.exp(np.cumsum(np.full(40, .01 if i % 2 else -.01))) for i, c in enumerate(COINS)}, index=idx)
    F = {c: 0.0001 for c in COINS}
    with tempfile.TemporaryDirectory() as td:
        lg = Path(td) / "l.jsonl"; now = dt.datetime(2026, 2, 10, 1, tzinfo=dt.timezone.utc)
        recs = [make_record(C.iloc[:k], F, now) for k in (38, 39, 40)]
        assert all(append(r, lg) for r in recs) and not append(recs[-1], lg)
        x = pnl(read(lg)); assert len(x) == 2
        w = recs[1]["weights"]; exp = sum(w[c] * (C.iloc[39][c] / C.iloc[38][c] - 1 - 1e-4) for c in COINS) - COST * sum(abs(w[c] - recs[0]["weights"][c]) for c in COINS)
        assert abs(x.core.iloc[-1] - exp) < 1e-9
        assert sum(recs[-1]["weights"].values()) > 0 and all(v == 0 for c, v in recs[-1]["weights"].items() if COINS.index(c) % 2 == 0)   # 하락 코인 0
    assert not too_late(dt.datetime(2026, 2, 11, 1, tzinfo=dt.timezone.utc), "2026-02-10") and too_late(dt.datetime(2026, 2, 12, 3, tzinfo=dt.timezone.utc), "2026-02-10")
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--status", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.status:
        status()
    else:
        sys.exit(main())
