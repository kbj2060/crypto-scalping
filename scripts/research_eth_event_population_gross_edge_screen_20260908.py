#!/usr/bin/env python3
"""**모델 없는 총수익 스크린** -- 이 저장소의 모든 사건 모집단에 비용을 넘는 방향 엣지가 있는가.

## 배경 (2026-09-08)
같은 날 배리어/지평 확장(`research_eth_anchor_barrier_horizon_extension_20260908.py`)에서
`any3/Wc3` 앵커의 지속 진입은 **같은-측면 무작위 봉 대비 초과가 두 창 CI>0 인 셀 0/16** 이었다.
VAL 만 +8~11bp 이고 OOS 는 +0.2~−3.4. ⇒ 앵커 자체에 방향 정보가 없다(모델 이전의 문제).

그렇다면 질문을 넓힌다: **다른 사건 모집단에는 있는가?**
모델을 쓰지 않고(방향축은 ~120셀 기각) 사건 발동 자체를 신호로 보고 총수익을 잰다.

## 설계
모집단 = 앵커 정의 5종 + 개별 증거신호 8종(first_fire) × 측면(all/top/bottom)
격자 = H ∈ {48, 144} 봉 × 배리어 ∈ {±1.0, ±1.5, ±2.0}%
진입 `open[t+1]` · 1분봉 첫터치 · **커버리지 100%**(미터치는 H봉 뒤 종가)
방향 = 지속(bottom→롱, top→숏). 페이드는 부호 반전이므로 같은 표에서 읽는다.
비용 7.8bp. **승격 조건: |초과| − 7.8 > 0 이고 VAL·OOS 두 창 일군집 CI 하한 > 0.**

## 귀무 (측면 편향 제거)
같은 창의 무작위 봉 N=20,000 을 롱/숏 각각으로 채점 -> `null_long/null_short`.
모집단 초과 = y_i − (sgn_i>0 ? null_long : null_short).
[[feedback_side_asymmetry_needs_same_side_null_20260905]] -- VAL/OOS 는 하락장이라 필수.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
SRC = ROOT / "tmp/eth_anchor_label_dataset_20260907/anchors_labels.parquet"
OUT = ROOT / "tmp/eth_event_gross_screen_20260908"
H_GRID = (48, 144)
P_GRID = (1.0, 1.5, 2.0)
COST = 7.8
MAX_MIN = max(H_GRID) * 5
CHUNK = 3000
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
NRAND = 20000
BOOT = 2000
SEED = 20260908


def first_touch(hi1, lo1, start, up, dn, max_min=MAX_MIN):
    n = len(start)
    t_up = np.full(n, -1, np.int32); t_dn = np.full(n, -1, np.int32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        idx = start[a:b, None] + np.arange(max_min)[None, :]
        hu = hi1[idx] >= up[a:b, None]; hd = lo1[idx] <= dn[a:b, None]
        au = hu.any(1); ad = hd.any(1)
        t_up[a:b] = np.where(au, hu.argmax(1), -1)
        t_dn[a:b] = np.where(ad, hd.argmax(1), -1)
    return t_up, t_dn


def pnl(t_up, t_dn, entry, exit_close, sgn, H, P):
    lim = H * 5; big = 1 << 30
    uo = (t_up >= 0) & (t_up < lim); do = (t_dn >= 0) & (t_dn < lim)
    tu = np.where(uo, t_up, big); td = np.where(do, t_dn, big)
    y = (exit_close - entry) / entry * 1e4 * sgn
    y = np.where(uo & (tu < td), sgn * P * 100.0, y)
    y = np.where(do & (td < tu), -sgn * P * 100.0, y)
    y = np.where(uo & do & (tu == td), -P * 100.0, y)      # 동시터치 = 보수적 손실
    return y


def day_ci(v, day, rng, B=BOOT):
    u = np.unique(day)
    if len(u) < 8: return (np.nan, np.nan)
    idx = {x: np.flatnonzero(day == x) for x in u}
    o = [v[np.concatenate([idx[x] for x in rng.choice(u, len(u), True)])].mean() for _ in range(B)]
    return tuple(np.percentile(o, [2.5, 97.5]))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("[1/4] 로드 ...", flush=True)
    D = pd.read_parquet(SRC).reset_index(drop=True)
    eth = B._load_kl(B.ETH_KL)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    CAP = len(eth) - MAX_MIN // 5 - 2

    def touches(bidx):
        ei = np.minimum(bidx + 1, len(O5) - 1)
        entry = O5[ei]
        st = np.searchsorted(ts1, ts5[ei])
        ok = (st < len(ts1) - MAX_MIN) & (ts1[np.minimum(st, len(ts1) - 1)] == ts5[ei]) & (bidx <= CAP)
        s = np.where(ok, st, 0)
        return entry, ei, ok, {P: first_touch(hi1, lo1, s, entry * (1 + P / 100), entry * (1 - P / 100))
                               for P in P_GRID}

    print(f"[2/4] 사건 {len(D):,} 배리어 터치 ...", flush=True)
    e_a, ei_a, ok_a, tt_a = touches(D["bar_idx"].to_numpy())
    split = D["split"].to_numpy()
    day = pd.to_datetime(D["timestamp"]).dt.floor("D").to_numpy()
    sgn = np.where(D["side"].to_numpy() == "bottom", 1.0, -1.0)

    print(f"[3/4] 창별 무작위 봉 귀무 N={NRAND:,} ...", flush=True)
    NULL = {}
    for w in WINS:
        m = split == w
        if m.sum() == 0: continue
        a, b = int(D["bar_idx"].to_numpy()[m].min()), min(int(D["bar_idx"].to_numpy()[m].max()), CAP)
        rb = rng.integers(a, b + 1, NRAND)
        er, eir, okr, ttr = touches(rb)
        for H in H_GRID:
            xr = np.minimum(eir + H, len(C5) - 1)
            for P in P_GRID:
                tu, td = ttr[P]
                NULL[(w, H, P, +1)] = float(pnl(tu, td, er, C5[xr], +1.0, H, P)[okr].mean())
                NULL[(w, H, P, -1)] = float(pnl(tu, td, er, C5[xr], -1.0, H, P)[okr].mean())
        print(f"      {w}: null_long(H48,±1%) {NULL[(w,48,1.0,1)]:+.2f}bp "
              f"· null_short {NULL[(w,48,1.0,-1)]:+.2f}bp", flush=True)

    # ---- 모집단 정의 -------------------------------------------------------------------
    pops = {}
    for a in D["anchor"].unique():
        pops[f"A:{a}"] = (D["anchor"].to_numpy() == a)
    ff = D["anchor"].to_numpy() == "first_fire"
    sig = D["signal"].astype(str).to_numpy()
    for s in sorted(set(sig[ff])):
        if s in ("nan", "None"): continue
        pops[f"S:{s}"] = ff & (sig == s)

    print(f"[4/4] 모집단 {len(pops)} × 측면 3 × H{len(H_GRID)} × P{len(P_GRID)} 채점 ...", flush=True)
    rows = []
    for pname, pm in pops.items():
        for sd in ("all", "top", "bottom"):
            sm = pm if sd == "all" else (pm & (D["side"].to_numpy() == sd))
            for w in WINS:
                m = sm & (split == w) & ok_a
                if m.sum() < 40: continue
                for H in H_GRID:
                    xi = np.minimum(ei_a + H, len(C5) - 1)
                    for P in P_GRID:
                        tu, td = tt_a[P]
                        y = pnl(tu, td, e_a, C5[xi], sgn, H, P)[m]
                        nb = np.array([NULL[(w, H, P, int(np.sign(x)))] for x in sgn[m]])
                        exc = y - nb
                        lo, hi = day_ci(exc, day[m], rng)
                        rows.append(dict(pop=pname, side=sd, win=w, H=H, P=P, n=int(m.sum()),
                                         gross=float(y.mean()), excess=float(exc.mean()),
                                         lo=lo, hi=hi, ndays=int(len(np.unique(day[m])))))
    A = pd.DataFrame(rows); A.to_csv(OUT / "screen.csv", index=False)
    print(f"      셀 {len(A):,}", flush=True)

    # ---- 판정: 지속(초과>0) 또는 페이드(초과<0) 어느 쪽이든 |초과|−비용>0 & 두 창 CI 배제 ----
    key = ["pop", "side", "H", "P"]
    piv = A.pivot_table(index=key, columns="win", values=["excess", "lo", "hi"])
    winners = []
    for k, g in A.groupby(key):
        gg = {r.win: r for r in g.itertuples()}
        if "VAL" not in gg or "OOS" not in gg: continue
        cont = all(gg[w].lo - COST > 0 for w in ("VAL", "OOS"))
        fade = all(-gg[w].hi - COST > 0 for w in ("VAL", "OOS"))
        if cont or fade:
            winners.append((k, "지속" if cont else "페이드",
                            gg["VAL"].excess, gg["OOS"].excess,
                            gg.get("HOLDOUT_SPENT").excess if "HOLDOUT_SPENT" in gg else np.nan))
    print()
    print("=" * 100)
    print(f"⭐승격 후보(두 창 CI 가 비용 {COST}bp 를 배제): {len(winners)}건")
    for k, d_, v, o, h in winners:
        print(f"   {k} · {d_} · VAL {v:+.2f} · OOS {o:+.2f} · HOLDOUT {h:+.2f}")
    # 참고: 비용 무시하고 두 창 CI 부호만 일치하는 셀
    sign_ok = []
    for k, g in A.groupby(key):
        gg = {r.win: r for r in g.itertuples()}
        if "VAL" not in gg or "OOS" not in gg: continue
        if all(gg[w].lo > 0 for w in ("VAL", "OOS")) or all(gg[w].hi < 0 for w in ("VAL", "OOS")):
            sign_ok.append((k, gg["VAL"].excess, gg["OOS"].excess,
                            gg["HOLDOUT_SPENT"].excess if "HOLDOUT_SPENT" in gg else np.nan))
    print(f"\n참고 · 비용 무시하고 두 창 CI 부호만 0 배제: {len(sign_ok)}/{A.groupby(key).ngroups}셀")
    for k, v, o, h in sorted(sign_ok, key=lambda x: -abs(x[1] + x[2]))[:15]:
        print(f"   {k} · VAL {v:+.2f} · OOS {o:+.2f} · HOLDOUT {h:+.2f}")
    print("\n상위 |VAL+OOS| 초과 셀 (참고, CI 무관):")
    A2 = A[A.win.isin(("VAL", "OOS"))].pivot_table(index=key, columns="win", values="excess")
    A2["s"] = A2["VAL"] + A2["OOS"]
    for k, r in A2.reindex(A2.s.abs().sort_values(ascending=False).index).head(10).iterrows():
        print(f"   {k} · VAL {r['VAL']:+.2f} · OOS {r['OOS']:+.2f}")
    print(json.dumps({"cells": len(A), "promote": len(winners), "sign_ok": len(sign_ok)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
