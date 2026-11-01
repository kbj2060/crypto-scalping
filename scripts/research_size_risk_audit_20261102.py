#!/usr/bin/env python3
"""크기(배수) 위험 감사 (2026-11-02, 기술 감사 -- 합격/불합격 판정 없음): 서버 원장 배수 + 시장 최대역행폭(MAE) → 순자산 손실 환산.
  python scripts/research_size_risk_audit_20261102.py <ledger_full.jsonl>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = "/home/kbj20/crypto-scalping/tmp/meta_label_race_20261101/{}USDT_5m.parquet"
HOLD = {"1h": 12, "3h": 36, "5h": 60}
SL_PCT = (0.3, 0.5, 0.8, 1.2)               # 사용자 SL 거리 후보(%)
RISK_PCT = 1.0                               # 스톱 1회당 허용 순자산 손실(%)


def ledger(path):
    r = [json.loads(l) for l in open(path, errors="replace") if l.startswith("{")]
    d = pd.DataFrame(r)
    d = d[d.equity_at_record.notna() & d.peak_notional.notna()].copy()
    d["eq_before"] = d.equity_at_record - d.net_pnl
    d["mult"] = d.peak_notional / d.eq_before
    d["t"] = pd.to_datetime(d.entry_time, unit="ms")
    d["hold_h"] = (d.exit_time - d.entry_time) / 3.6e6
    d["pnl_pct"] = d.net_pnl / d.eq_before * 100
    return d


def mae(coin):
    """매 시 정각 진입(롱·숏)의 H봉 최대 역행폭 %. 반환 {보유: (n,2) 배열}."""
    from numpy.lib.stride_tricks import sliding_window_view as sw
    p = pd.read_parquet(PANEL.format(coin), columns=["o", "h", "l"])
    o, h, l = p.o.to_numpy(), p.h.to_numpy(), p.l.to_numpy()
    idx = np.arange(12, len(p) - 61, 12)
    out = {}
    for k, n in HOLD.items():
        e = o[idx + 1]
        hh, ll = sw(h, n)[idx + 1].max(1), sw(l, n)[idx + 1].min(1)
        out[k] = np.stack([(hh / e - 1) * 100, (1 - ll / e) * 100], 1).ravel()   # 숏 역행(위) · 롱 역행(아래)
    return out


def main(path):
    d = ledger(path)
    print("== 원장 (배수 계산 가능 %d건, %s ~ %s)" % (len(d), d.t.min().date(), d.t.max().date()))
    d["bucket"] = pd.cut(d.mult, [0, 1.5, 3, 6, 100], labels=["≤1.5", "1.5~3", "3~6", ">6"])
    print(d.groupby(["symbol", "bucket"], observed=True).agg(n=("mult", "size"), pnl_usd=("net_pnl", "sum"), mean_pct=("pnl_pct", "mean"), worst_pct=("pnl_pct", "min")).round(2))
    x = d[d.symbol.str.startswith("XRP")].sort_values("t")
    print("XRP 배수 추이:", " ".join(f"{m:.1f}" for m in x.mult))
    print("== 시장 최대역행폭 분위 (%, 매 시 정각 진입 롱·숏 합산)")
    qs = [50, 90, 95, 99, 99.9]
    res = {}
    for coin in ("XRP", "ETH"):
        m = mae(coin)
        res[coin] = m
        print(coin, pd.DataFrame({k: np.percentile(v, qs) for k, v in m.items()}, index=[f"p{q}" for q in qs]).round(2).to_dict("index"))
    print("== 배수별 순자산 손실(%) = 배수 × MAE (XRP, 보유 3h, SL 없음)")
    for mult in (2, 3, 4, 5, 7):
        print(f"{mult}배:", {f"p{q}": round(mult * np.percentile(res['XRP']['3h'], q), 1) for q in (90, 99, 99.9)})
    print("== SL 거리별 (XRP 3h): 스톱 닿을 확률 · 스톱 1회 손실 상한 배수(=%.0f%%/SL)" % RISK_PCT)
    for s in SL_PCT:
        print(f"SL {s}%: 닿을 확률 {np.mean(res['XRP']['3h'] >= s) * 100:.0f}% · 허용 배수 {RISK_PCT / s:.1f}배 · 현재 마지막 XRP 배수 {x.mult.iloc[-1]:.1f}배면 스톱 1회 {x.mult.iloc[-1] * s:.1f}% 손실")
    cap = x.assign(f=np.minimum(1, 3 / x.mult))
    print("== 3배 상한 반사실 (XRP 최근 14건 중 배수 있는 %d건): 실제 $%.1f → 상한 $%.1f (포기 이익 $%.1f)" % (len(cap), cap.net_pnl.sum(), (cap.net_pnl * cap.f).sum(), cap.net_pnl.sum() - (cap.net_pnl * cap.f).sum()))


if __name__ == "__main__":
    main(sys.argv[1])
