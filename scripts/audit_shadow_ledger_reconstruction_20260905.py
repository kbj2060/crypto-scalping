#!/usr/bin/env python3
"""섀도우 원장 **전건 재구성 파리티** (2026-09-05) -- R 지속 규칙·F0 V자반등 경제라벨 원장의 마감 트레이드 전부를
실제 5분봉으로 `sim_exit`(5.0/1.5/0.1 ATR, 비관 순서, 봉 고가/저가) 재계산해 러너 회계(청산봉·손익)와 대조한다.

기존 analyze_eth_fire_cont_shadow_ledger 의 ② 파리티는 진입 후 200봉이 다 찬 트레이드만 재구성해 초기엔 n이 1~2다.
여기서는 **청산 시각까지의 봉만으로** 재구성한다(스톱/트레일 청산은 그 안에서 결정되므로 충분; 타임아웃 청산만 제외).
입력: --r-state / --f0-state (서버 원장 JSON), 바이낸스 선물 5m klines(공개 REST). 출력: 트레이드별 Δ청산봉·Δ손익, 일치율.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
_s = importlib.util.spec_from_file_location("hev2_rec", ROOT / "scripts/research_homer_entry_v2_20260904.py"); hev2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(hev2)
sim_exit = hev2.sim_exit; CELL = (5.0, 1.5, 0.1); COST = 10.0


def klines(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []; t = int(start.timestamp() * 1000); end_ms = int(end.timestamp() * 1000)
    while t < end_ms:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines", params={"symbol": "ETHUSDT", "interval": "5m", "startTime": t, "limit": 1500}, timeout=20); r.raise_for_status(); raw = r.json()
        if not raw: break
        rows += raw; t = raw[-1][0] + 300_000
    k = pd.DataFrame(rows, columns=["ot", "open", "high", "low", "close", "v", "ct", "qv", "n", "tb", "tq", "i"])
    for c in ("open", "high", "low", "close"): k[c] = k[c].astype(float)
    k["ts"] = pd.to_datetime(k["ot"], unit="ms", utc=True).dt.tz_localize(None)
    k = k.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    if len(k) and int(k["ct"].iloc[-1]) >= int(pd.Timestamp.now(tz="UTC").timestamp() * 1000): k = k.iloc[:-1]
    return k


def reconstruct(name: str, ledger: list[dict], kl: pd.DataFrame, entry_key: str = "entry_utc") -> pd.DataFrame:
    pos = {t: i for i, t in enumerate(kl["ts"])}; h, l, c = (kl[x].to_numpy(float) for x in ("high", "low", "close")); out = []
    for r in ledger:
        e_ts = pd.to_datetime(r[entry_key], utc=True).tz_localize(None); x_ts = pd.to_datetime(r["exit_utc"], utc=True).tz_localize(None)
        i = pos.get(e_ts); j = pos.get(x_ts)
        if i is None or j is None or j <= i: out.append({"entry_utc": str(e_ts), "status": "bars_missing"}); continue
        sgn = 1.0 if r["side"] == "long" else -1.0; H = h[i + 1:j + 1][None]; L = l[i + 1:j + 1][None]; C = c[i + 1:j + 1][None]
        pn, ex = sim_exit(np.array([float(r["entry"])]), np.array([float(r["atr"])]), np.array([sgn]), H, L, C, *CELL)
        rec_off = j - i - 1; pnl_rec = float(pn[0]) * 1e4 - COST; closed_inside = int(ex[0]) < H.shape[1] - 1 or r.get("reason") == "timeout"
        # sim_exit 은 창 끝까지 미청산이면 마지막 종가로 강제청산 -> 창이 정확히 청산봉까지라 마지막 봉에서 스톱이면 ex == len-1 이 정상
        out.append({"entry_utc": str(e_ts), "side": r["side"], "reason": r.get("reason"), "bars_ledger": rec_off, "bars_recon": int(ex[0]),
                    "pnl_ledger": round(float(r["pnl_bp"]), 2), "pnl_recon": round(pnl_rec, 2), "d_pnl": round(pnl_rec - float(r["pnl_bp"]), 2),
                    "status": "ok" if abs(pnl_rec - float(r["pnl_bp"])) < 0.05 and int(ex[0]) == rec_off else "MISMATCH"})
    df = pd.DataFrame(out); ok = (df["status"] == "ok").sum(); n = (df["status"] != "bars_missing").sum()
    print(f"\n### {name}: 재구성 {n}건 · 일치 {ok} ({ok/max(n,1)*100:.1f}%) · 봉 결측 {(df['status']=='bars_missing').sum()}")
    bad = df[df["status"] == "MISMATCH"]
    if len(bad): print(bad.to_string(index=False))
    else: print("  전건 일치 (청산봉·손익 |Δ|<0.05bp)")
    return df


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--r-state"); ap.add_argument("--f0-state"); a = ap.parse_args()
    ledgers = {}
    if a.r_state: ledgers["R 지속 규칙"] = json.load(open(a.r_state))["ledger"]
    if a.f0_state: ledgers["F0 V자반등"] = json.load(open(a.f0_state))["ledger"]
    t0 = min(pd.to_datetime(r["entry_utc"], utc=True) for L in ledgers.values() for r in L) - pd.Timedelta(hours=2)
    kl = klines(t0, pd.Timestamp.now(tz="UTC")); print(f"klines {len(kl)} {kl['ts'].iloc[0]} ~ {kl['ts'].iloc[-1]}")
    for name, L in ledgers.items(): reconstruct(name, L, kl)


if __name__ == "__main__":
    main()
