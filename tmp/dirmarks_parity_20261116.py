"""방향 표식(app.js DIRCALC) ↔ 연구 Python 사건 목록 대조 (2026-11-16).

같은 5분봉(ETH·XRP Vision 1분봉 → 5분)을 app.js 의 dirCalc(node) 와 연구 함수(후지모토 브랜치)에 넣어
A14 뉴욕 개장 범위 돌파 · A15 CCI 1h · A16 RSI2 눌림 15m · A12 RSI 다이버전스 15m 사건 (종류, 봉 시각, 방향)이 같은지 본다.
  python tmp/dirmarks_parity_20261116.py
"""
import json, re, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/home/kbj20/crypto-scalping/.claude/worktrees/fujimoto-126-trading-analysis-724dd5/scripts")
import research_orb_meaning_20261115 as ORB
import research_volume_osc_bundle_20261116 as VO
import research_classic_indicator_bundle_20261115 as CB
import research_rsi_divergence_meaning_20261115 as RD
import research_smc_ict_meaning_20261115 as SM
import importlib.util as _iu
_sp = _iu.spec_from_file_location("RR", ROOT / "scripts/research_realtime_risk_radar_20261115.py"); RR = _iu.module_from_spec(_sp); _sp.loader.exec_module(RR)


def five(coin, a, b):
    g, m0 = RR._kl(coin)
    g = g[(g.index * 60 >= pd.Timestamp(a).timestamp()) & (g.index * 60 < pd.Timestamp(b).timestamp())].dropna(subset=["c"])
    t = (g.index.to_numpy() * 60) // 300 * 300
    d = pd.DataFrame({"t": t, "h": g.h.values, "l": g.l.values, "c": g.c.values}).groupby("t").agg(h=("h", "max"), l=("l", "min"), c=("c", "last")).reset_index()
    return d[d.t < d.t.iloc[-1] // 3600 * 3600]          # 정시에서 끝냄(형성 봉이 모든 지평에서 새 묶음이 되게)


def agg(d5, M):
    g = d5.assign(T=d5.t // M * M).groupby("T").agg(h=("h", "max"), l=("l", "min"), c=("c", "last")).reset_index()
    return g                                             # d5 는 정시에서 끝나고 app 쪽엔 그 뒤 형성 봉 하나를 붙인다 → 두 쪽 모두 마감 봉 = d5 전체


def research(d5):
    ev = set()
    B5 = agg(d5, 300); B = dict(t=B5["T"].to_numpy() * 1000, h=B5.h.to_numpy(float), l=B5.l.to_numpy(float), c=B5.c.to_numpy(float))
    S = ORB.starts(int(B["t"][0]), int(B["t"][-1]))["NY"]
    ix = np.searchsorted(B["t"], S); ok = (ix < len(B["t"])) & (B["t"][np.clip(ix, 0, len(B["t"]) - 1)] == S)
    for s0, i0 in zip(S[ok], ix[ok]):
        for L in (30, 60):
            o = ORB.orb(B, int(s0), L, int(i0))
            if o is not None:
                b, dd, W = o
                ev.add((f"orb{L}", int(B["t"][b] // 1000), int(dd)))
    H = agg(d5, 3600); h, l, c = (H[x].to_numpy(float) for x in "hlc"); n = len(c)
    C = VO.cci(h, l, c)
    E1 = np.where(CB.cross(C, np.full(n, 100.0)) == 1, 1, np.where(CB.cross(C, np.full(n, -100.0)) == -1, -1, 0))
    for i in np.flatnonzero(E1):
        ev.add(("cci", int(H["T"].iloc[i]), int(E1[i])))
    Q = agg(d5, 900); h, l, c = (Q[x].to_numpy(float) for x in "hlc"); n = len(c)
    r2, tr = RD.rsi(c, 2), np.sign(c - CB.sma(c, 200))
    with np.errstate(invalid="ignore"):
        G1 = np.where((tr == 1) & (CB.cross(r2, np.full(n, 10.0)) == -1), 1, np.where((tr == -1) & (CB.cross(r2, np.full(n, 90.0)) == 1), -1, 0))
    for i in np.flatnonzero(G1):
        ev.add(("rsi2", int(Q["T"].iloc[i]), int(G1[i])))
    r = RD.rsi(c)
    for high in (True, False):
        x = h if high else l
        for p, kind in RD.classify(x, r, SM.pivots(x, RD.N, high), high):
            if kind == "D" and p + RD.N < n:
                ev.add(("rdiv", int(Q["T"].iloc[p + RD.N]), -1 if high else 1))
    return ev


def app(d5):
    src = (ROOT / "dashboard/live/app.js").read_text()
    js = re.search(r"// DIRCALC-BEGIN\n(.*?)// DIRCALC-END", src, re.S).group(1)
    c5 = [{"time": int(t), "high": h, "low": l, "close": c} for t, h, l, c in d5[["t", "h", "l", "c"]].itertuples(index=False)]
    c5.append({"time": int(d5.t.iloc[-1]) + 300, "high": 1, "low": 1, "close": 1})   # 형성 중 봉(app 은 마지막 묶음을 뺀다)
    prog = js + f"\nconst c5 = {json.dumps(c5)};\nconst ev = dirCalc(c5, c5[c5.length - 1].time);\nconsole.log(JSON.stringify(ev.filter((e) => e.tk != null).map((e) => [e.k, e.t, e.d])));"
    out = subprocess.run(["node", "-"], input=prog, capture_output=True, text=True, check=True).stdout
    return {tuple(x) for x in json.loads(out)}


if __name__ == "__main__":
    bad = 0
    for coin in ("ETH", "XRP"):
        d5 = five(coin, "2026-08-01", "2026-10-20")
        A, R = app(d5), research(d5)
        warm = int(d5.t.iloc[0]) + 14 * 86400                          # RSI·SMA 시작 차이(ewm 첫 값) 영향 구간은 뺀다
        A2, R2 = {e for e in A if e[1] >= warm}, {e for e in R if e[1] >= warm}
        for k in ("orb30", "orb60", "cci", "rsi2", "rdiv"):
            a, r = {e for e in A2 if e[0] == k}, {e for e in R2 if e[0] == k}
            print(coin, k, f"app {len(a)} · 연구 {len(r)} · 같음 {len(a & r)} · app만 {sorted(a - r)[:3]} · 연구만 {sorted(r - a)[:3]}")
            bad += len(a ^ r)
            assert len(r) > 0, (coin, k)                                 # 양성 대조: 사건이 실제로 있다
    print("PARITY", "OK" if bad == 0 else f"FAIL {bad}")
    sys.exit(1 if bad else 0)
