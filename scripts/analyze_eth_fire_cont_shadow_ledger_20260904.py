#!/usr/bin/env python3
"""지속 규칙 섀도우 원장 점검 -- 사전등록(§2) 30일 계측 점검 + 90일 판정 지표를 한 번에 낸다 (2026-09-04).

입력: 서버에서 pull한 `data/live/fire_cont_shadow_state.json` (또는 --state), 바이낸스 선물 5m klines(공개 API).
검사:
  ① 계측 무결성: 진입 지연 중앙값(≤30s), 슬리피지 평균(|·|≤3bp), 놓친 봉 비율(<2%), 첫발동 빈도(24/일 ±30%)
  ② 재구성 파리티: 원장의 (발동 봉, 진입가, ATR, 방향)으로 `sim_exit`(5.0/1.5/0.1, 200봉)을 실제 봉에서 재계산 → 청산봉·손익 일치율(≥95%)
  ③ 성과: 마감 원장 / 오픈 포지션 시가평가 포함, 측면별·신호별, 일군집 CI, 최근 30일 킬 스위치 지표
  ④ 백테스트 참조와의 위치: VAL +4.44 / OOS +6.78bp, 24건/일
사전등록 문서: docs/experiments/eth_fire_cont_shadow_prereg_20260904.md. 판단은 사람이 한다 -- 이 스크립트는 숫자만 낸다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


V2 = _load("hev2_led", "scripts/research_homer_entry_v2_20260904.py")
sim_exit, day_boot = V2.sim_exit, V2.day_boot
CELL, FWD, COST = (5.0, 1.5, 0.1), 200, 10.0
KL_URL = "https://fapi.binance.com/fapi/v1/klines"


def fetch_klines(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []; t = int(start.timestamp() * 1000); end_ms = int(end.timestamp() * 1000)
    while t < end_ms:
        r = requests.get(KL_URL, params={"symbol": "ETHUSDT", "interval": "5m", "startTime": t, "limit": 1500}, timeout=20); r.raise_for_status()
        raw = r.json()
        if not raw:
            break
        rows += raw; t = raw[-1][0] + 300_000
    k = pd.DataFrame(rows, columns=["ot", "open", "high", "low", "close", "v", "ct", "qv", "n", "tb", "tq", "i"])
    for c in ("open", "high", "low", "close"):
        k[c] = k[c].astype(float)
    k["ts"] = pd.to_datetime(k["ot"], unit="ms", utc=True).dt.tz_localize(None)
    return k.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--state", default=str(ROOT / "data/live/fire_cont_shadow_state.json")); a = ap.parse_args()
    s = json.loads(Path(a.state).read_text()); led = pd.DataFrame(s["ledger"]); pos = pd.DataFrame(s["positions"])
    start = pd.Timestamp(s["started_utc"]); now = pd.Timestamp.now(tz="UTC"); days = max((now - start).total_seconds() / 86400, 1e-9)
    n_bars = int(days * 288); print(f"가동 {days:.2f}일 · 마감 {len(led)}건 · 오픈 {len(pos)}건 · 놓친 봉 {s.get('missed_bars')} ({s.get('missed_bars', 0)/max(n_bars,1)*100:.2f}%) · 스킵 {s.get('skipped')}")
    if not len(led) and not len(pos):
        return
    kl = fetch_klines(start - pd.Timedelta(hours=2), now + pd.Timedelta(minutes=5)); kts = kl["ts"].to_numpy(); o, h, l, c = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    # ① 계측
    allrec = pd.concat([led, pos], ignore_index=True) if len(pos) else led
    n_entries = len(allrec); per_day = n_entries / days
    print(f"① 진입 {n_entries}건 = {per_day:.1f}/일 (백테스트 24.3±30% → [17, 32]) · 지연 중앙 {allrec['decision_lag_sec'].median():.0f}s (≤30) · "
          f"슬리피지 평균 {allrec['entry_slip_bp'].mean():+.2f}bp (|·|≤3) · 슬리피지 |·| 중앙 {allrec['entry_slip_bp'].abs().median():.2f}")
    # ② 재구성 파리티 (마감 원장)
    if len(led):
        ok_exit = ok_pnl = 0; deltas = []
        for r in led.itertuples():
            i = int(np.searchsorted(kts, np.datetime64(pd.Timestamp(r.entry_utc).tz_localize(None))))          # 발동 봉 인덱스
            if i + 1 + FWD >= len(kts) or kts[i] != np.datetime64(pd.Timestamp(r.entry_utc).tz_localize(None)):
                continue
            H = h[i + 1:i + 1 + FWD][None]; L = l[i + 1:i + 1 + FWD][None]; C = c[i + 1:i + 1 + FWD][None]
            pn, ex = sim_exit(np.array([r.entry]), np.array([r.atr]), np.array([1.0 if r.side == "long" else -1.0]), H, L, C, *CELL)
            rec_bp = pn[0] * 1e4 - COST; d = rec_bp - r.pnl_bp; deltas.append(d)
            exit_ts = kts[i + 1 + int(ex[0])]; ok_exit += int(str(pd.Timestamp(exit_ts).tz_localize("UTC")) == r.exit_utc or abs(d) < 0.01); ok_pnl += int(abs(d) < 0.01)
        n = len(deltas)
        if n:
            print(f"② 재구성 파리티 n{n}: 손익 일치(|Δ|<0.01bp) {ok_pnl/n*100:.1f}% (≥95) · 평균 Δ {np.mean(deltas):+.2f}bp · max|Δ| {np.max(np.abs(deltas)):.2f}")
    # ③ 성과
    def block(d, name):
        if not len(d):
            return
        p = d["pnl_bp"].to_numpy(float); w = p > 0; eq = np.cumsum(p); dd = (eq - np.maximum.accumulate(eq)).min()
        ci = day_boot(p, pd.to_datetime(d["entry_utc"]).dt.tz_localize(None), 1000, np.random.default_rng(1)) if len(p) >= 10 else (np.nan, np.nan)
        po = p[w].mean() / -p[~w].mean() if w.any() and (~w).any() else float("nan")
        print(f"   {name:>14s} n {len(p):4d} 기대값 {p.mean():+.2f}bp 일CI [{ci[0]:+.1f}, {ci[1]:+.1f}] 메이커 {d['pnl_maker_bp'].mean():+.2f} 승률 {w.mean()*100:.1f}% 손익비 {po:.2f} 최대DD {dd:+.0f} 누적 {p.sum():+.0f}")
    if len(led):
        print("③ 마감 원장"); block(led, "전체"); block(led[led["side"] == "short"], "숏(바닥발동)"); block(led[led["side"] == "long"], "롱(천장발동)")
        for sg in sorted({x for xs in led["signals"] for x in (xs or [])}):
            block(led[led["signals"].apply(lambda xs: sg in (xs or []))], sg[:14])
        rec30 = led[pd.to_datetime(led["exit_utc"]) >= (now - pd.Timedelta(days=30)).tz_localize(None).tz_localize("UTC")] if len(led) else led
        if len(rec30) >= 10:
            print("   최근 30일(킬 스위치: 평균<−10 ∧ CI상한<0)"); block(rec30, "최근30일")
    if len(pos):
        last_close = float(c[-1]); mtm = []
        for r in pos.itertuples():
            sgn = 1.0 if r.side == "long" else -1.0; mtm.append(sgn * (last_close - r.entry) / r.entry * 1e4 - COST)
        mtm = np.array(mtm); allp = np.r_[led["pnl_bp"].to_numpy(float) if len(led) else [], mtm]
        print(f"   오픈 {len(pos)}건 시가평가 평균 {mtm.mean():+.2f}bp · 마감+오픈 평균 {allp.mean():+.2f}bp (n {len(allp)})")
    print("④ 백테스트 참조: VAL +4.44 [0.7, 8.1] · OOS +6.78 [2.6, 11.4] bp · 24/일 · 승률 75% · 손익비 0.35~0.39")


if __name__ == "__main__":
    main()
