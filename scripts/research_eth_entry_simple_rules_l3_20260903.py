#!/usr/bin/env python3
"""진입 **단순 규칙** 탐색 -- L3 정직 라벨 (2026-09-03).

사용자 제안: "딥러닝 모델 말고 간단한 알고리즘으로 진입하는 건 어때?"

⭐근거는 예측력이 아니라 **입증 부담**이다. 오늘 확인된 것은 모델이 나쁘다가 아니라
(선별력은 시드 15/15로 안정적이었다) **독립 일수 42~45일이 161피쳐 모델을 감당하지 못한다**는
것이다 -- 일 단위 군집 부트스트랩 CI 하한이 음수이고 dtype 하나에 VAL 부호가 뒤집혔다.
파라미터 1~2개짜리 규칙은 같은 표본으로도 확립이 가능하다.

⚠️기대치: 기계 자체가 이미 0 근처다(무필터 arm1 VAL +4.01 / OOS +6.02 / **HOLDOUT −0.99**).
단순 규칙이 이걸 양수로 만들어야 의미가 있다.

시험:
  A/B/E 단일 피쳐 임계값 -- L3 중요도 상위 12개, 백분위 컷 × 방향. **임계값은 TRAIN에서만** 고른다
  C     레짐 게이트      -- bull/bear/chop. L0에서 기각됐으나 재시험 필요
  D     트리거 제거      -- 신호 없이 **최근 3봉 수익률을 페이드**(ret3<0이면 아래 매수).
                           파라미터 0개. 무작위 봉이 트리거 봉보다 나았던 것의 직접 후속

⚠️**대조군 필수**: 12피쳐 × 9컷 × 2방향 = 216조합을 TRAIN에서 고르므로 선택 편의가 있다.
   무작위 피쳐/임계값 대조군을 같이 돌려 "이 정도는 우연히도 나온다"를 잰다.
⚠️VAL/OOS/HOLDOUT은 **진단**이다. 확립은 섀도우 전진 데이터로만 한다.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research_eth_regime_gated_costgate_ensemble_20260902 import KLINES_PATH  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402
from research_eth_entry_direction_oracle_ceiling_20260903 import (  # noqa: E402
    SL, ARM, TRAIL, NOTIONAL, COST)

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
L3D = ROOT / "tmp/eth_entry_1m_resolved_20260903/labels_1m_all.csv"
M1P = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
ETH_REGIME = ROOT / "tmp/eth_regime_s12k3_clean_20260902/predictions.parquet"
OUT = ROOT / "tmp/eth_entry_simple_rules_20260903"
DEPTH, WAIT, NSLOT = 3.0, 6, 4
W3 = ("VAL", "OOS", "HOLDOUT")
TOP = ["atr_pct", "vwap_dist_288", "bb_width", "btc_corr_60", "whale_retail_ratio",
       "realized_skewness", "parkinson_vol", "cvd_roll_roc_48", "bb_pctb", "rsi",
       "vol_z", "adx14"]
CUTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
RNG = np.random.default_rng(20260903)


def log(m): print(f"[simple] {m}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LAB = pd.read_csv(L3D, parse_dates=["timestamp"])
    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    A = D.merge(LAB[["timestamp", "signal", "arm", "depth", "btf", "y_L3"]],
                on=["timestamp", "signal", "arm", "depth", "btf"], how="left")
    A = A[np.isfinite(A.y_L3)].reset_index(drop=True)
    S = A[(A.depth == DEPTH) & (A.btf <= WAIT) & (A.arm == 1)].reset_index(drop=True)
    tr = (S.split == "TRAIN").to_numpy()
    M = {w: (S.split == w).to_numpy() for w in W3}
    log(f"arm1 후보 {len(S):,} · TRAIN {int(tr.sum()):,}")

    def perf(mask):
        d = S[mask]
        if not len(d): return np.nan, 0
        t = slotN(d.assign(y=d.y_L3), NSLOT)
        return (float(np.mean(t) * 1e4) if len(t) else 0.0), int(len(t))

    nf_tr = perf(tr)
    nf = {w: perf(M[w]) for w in W3}
    print(f"\n{'':34s}{'TRAIN':>13s}" + "".join(f"{w:>14s}" for w in W3))
    print(f"{'무필터 (기준선)':34s}{nf_tr[0]:+8.2f}(n{nf_tr[1]:4d})"
          + "".join(f"{nf[w][0]:+8.2f}(n{nf[w][1]:4d})" for w in W3))

    # ---- A/B/E 단일 피쳐 임계값 ----
    log("\nA/B/E 단일 피쳐 임계값 (TRAIN에서만 선택)...")
    cand = []
    for f in TOP:
        if f not in S.columns: continue
        v = pd.to_numeric(S[f], errors="coerce").to_numpy(float)
        if not np.isfinite(v[tr]).any(): continue
        for c in CUTS:
            thr = float(np.nanpercentile(v[tr], c))
            for dirn, m0 in (("<=", v <= thr), (">=", v >= thr)):
                if m0[tr].mean() < 0.10 or m0[tr].mean() > 0.90: continue
                p_tr = perf(tr & m0)
                cand.append({"feat": f, "cut": c, "dir": dirn, "thr": thr,
                             "train_bp": p_tr[0], "train_n": p_tr[1],
                             "keep_tr": float(m0[tr].mean()), "mask": m0})
    C = pd.DataFrame([{k: v for k, v in c.items() if k != "mask"} for c in cand])
    C = C[C.train_n >= 200]
    log(f"  유효 조합 {len(C):,} (TRAIN n>=200)")
    best = C.sort_values("train_bp", ascending=False).head(5)
    print(f"\n=== A/B/E TRAIN 상위 5 (TRAIN에서만 선택, 나머지는 진단) ===")
    print(f"{'피쳐':>20s}{'컷':>5s}{'방향':>5s}{'유지':>7s}{'TRAIN':>13s}"
          + "".join(f"{w:>14s}" for w in W3))
    for _, r in best.iterrows():
        m0 = next(c["mask"] for c in cand
                  if c["feat"] == r.feat and c["cut"] == r.cut and c["dir"] == r["dir"])
        rr = {w: perf(M[w] & m0) for w in W3}
        print(f"{r.feat:>20s}{int(r.cut):5d}{r['dir']:>5s}{r.keep_tr:7.2f}"
              f"{r.train_bp:+8.2f}(n{int(r.train_n):4d})"
              + "".join(f"{rr[w][0]:+8.2f}(n{rr[w][1]:4d})" for w in W3))

    # 대조군: 무작위 임계값 규칙이 TRAIN에서 이 정도를 내는 빈도
    log("\n  무작위 규칙 대조군...")
    rand_tr = []
    for _ in range(300):
        f = TOP[RNG.integers(len(TOP))]
        if f not in S.columns: continue
        v = pd.to_numeric(S[f], errors="coerce").to_numpy(float)
        c = CUTS[RNG.integers(len(CUTS))]
        thr = float(np.nanpercentile(v[tr], c))
        m0 = (v <= thr) if RNG.random() < 0.5 else (v >= thr)
        if m0[tr].sum() < 200: continue
        rand_tr.append(perf(tr & m0)[0])
    rand_tr = np.array(rand_tr)
    top_train = float(best.train_bp.iloc[0])
    print(f"  TRAIN 최고 {top_train:+.2f}bp vs 무작위 규칙 평균 {rand_tr.mean():+.2f} "
          f"(p95 {np.percentile(rand_tr,95):+.2f}) · "
          f"⭐백분위 **{(rand_tr < top_train).mean()*100:.1f}%**")

    # ---- C 레짐 게이트 ----
    if ETH_REGIME.exists():
        R = pd.read_parquet(ETH_REGIME).rename(columns={"regime": "reg"})
        S2 = S.merge(R, on="timestamp", how="left")
        print(f"\n=== C 레짐 게이트 (0 bull / 1 bear / 2 chop) ===")
        print(f"{'레짐':>10s}{'TRAIN':>13s}" + "".join(f"{w:>14s}" for w in W3))
        for rv, nm in ((0, "bull"), (1, "bear"), (2, "chop")):
            m0 = (S2.reg == rv).to_numpy()
            p_tr = perf(tr & m0); rr = {w: perf(M[w] & m0) for w in W3}
            print(f"{nm:>10s}{p_tr[0]:+8.2f}(n{p_tr[1]:4d})"
                  + "".join(f"{rr[w][0]:+8.2f}(n{rr[w][1]:4d})" for w in W3))

    # ---- D 트리거 제거: 최근 3봉 수익률을 페이드 (파라미터 0개) ----
    log("\nD 트리거 제거 (ret3 페이드, 파라미터 0개)...")
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values(
        "timestamp").reset_index(drop=True)
    h5, l5, c5 = (kl[x].to_numpy(float) for x in ("high", "low", "close"))
    ts5 = pd.DatetimeIndex(kl["timestamp"]); n5 = len(kl)
    m1 = pd.read_csv(M1P, parse_dates=["timestamp"]).sort_values(
        "timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    m1["b5"] = m1["timestamp"].dt.floor("5min")
    m1h, m1l = m1["high"].to_numpy(float), m1["low"].to_numpy(float)
    f0 = m1.groupby("b5", sort=True).apply(lambda d: d.index[0])
    cn = m1.groupby("b5", sort=True).size()
    IDX0 = {k: int(v) for k, v in f0.items()}; IDXN = {k: int(v) for k, v in cn.items()}
    # ATR은 트리거 표본에서 보간
    ai = S.i.to_numpy().astype(int); aa = S.atr_pct.to_numpy(float)
    o = np.argsort(ai); ai, aa = ai[o], aa[o]

    def trail(side, e, a, hi, lo, cl):
        if side > 0:
            stop = e * (1 - SL * a); peak = e; armed = False
            for k in range(len(cl)):
                if lo[k] <= stop: return stop / e - 1.0
                if hi[k] > peak:
                    peak = hi[k]
                    if not armed and (peak - e) / e >= ARM * a: armed = True
                    if armed: stop = max(stop, peak * (1 - TRAIL * a))
            return cl[-1] / e - 1.0
        stop = e * (1 + SL * a); peak = e; armed = False
        for k in range(len(cl)):
            if hi[k] >= stop: return 1.0 - stop / e
            if lo[k] < peak:
                peak = lo[k]
                if not armed and (e - peak) / e >= ARM * a: armed = True
                if armed: stop = min(stop, peak * (1 + TRAIL * a))
        return 1.0 - cl[-1] / e

    ret3 = np.concatenate([[np.nan] * 3, (c5[3:] - c5[:-3]) / c5[:-3]])
    lo_i, hi_i = int(ai.min()), int(ai.max())
    step = 3                                   # 3봉마다 = 후보 수를 트리거와 비슷하게
    rows = []
    for i in range(lo_i, min(hi_i, n5 - 80), step):
        if not np.isfinite(ret3[i]) or ret3[i] == 0: continue
        a = float(np.interp(i, ai, aa))
        if not np.isfinite(a) or a <= 0: continue
        sd = 1 if ret3[i] < 0 else -1          # ⭐최근 하락이면 아래 매수(페이드)
        lim = c5[i] * (1 - DEPTH * a) if sd > 0 else c5[i] * (1 + DEPTH * a)
        ff = -1
        for off in range(1, WAIT + 1):
            k = i + off
            if k >= n5: break
            if (l5[k] <= lim) if sd > 0 else (h5[k] >= lim): ff = k; break
        if ff < 0: continue
        bt = ts5[ff]
        if bt not in IDX0: continue
        s0, nn = IDX0[bt], IDXN[bt]
        sh, sl_ = m1h[s0:s0 + nn], m1l[s0:s0 + nn]
        hit = np.flatnonzero(sl_ <= lim) if sd > 0 else np.flatnonzero(sh >= lim)
        if not len(hit): continue
        k0 = int(hit[0]); ph, pl = sh[k0 + 1:], sl_[k0 + 1:]
        fh = float(ph.max()) if len(ph) else lim
        fl = float(pl.min()) if len(pl) else lim
        hz = 24
        if ff + hz > n5: continue
        H = np.concatenate([[fh], h5[ff + 1:ff + hz]])
        L = np.concatenate([[fl], l5[ff + 1:ff + hz]])
        Cc = np.concatenate([[c5[ff]], c5[ff + 1:ff + hz]])
        rows.append({"timestamp": ts5[i], "fi": ff, "ei": ff + hz,
                     "y": trail(sd, lim, a, H, L, Cc) * NOTIONAL - COST * NOTIONAL})
    Dd = pd.DataFrame(rows)
    Dd["split"] = np.where(Dd.timestamp < pd.Timestamp("2025-09-01"), "TRAIN",
                    np.where(Dd.timestamp < pd.Timestamp("2026-01-01"), "VAL",
                    np.where(Dd.timestamp < pd.Timestamp("2026-04-01"), "OOS", "HOLDOUT")))
    print(f"\n=== D 트리거 제거 · ret3 페이드 (파라미터 0개) ===")
    print(f"{'':34s}{'TRAIN':>13s}" + "".join(f"{w:>14s}" for w in W3))
    r0 = []
    for w in ("TRAIN",) + W3:
        d = Dd[Dd.split == w]
        t = slotN(d, NSLOT) if len(d) else np.array([])
        r0.append((float(np.mean(t) * 1e4) if len(t) else 0.0, int(len(t))))
    print(f"{'ret3 페이드 (신호 없음)':34s}" + "".join(f"{v:+8.2f}(n{n:4d})" for v, n in r0))
    print(f"{'  (참고) 트리거 arm1 무필터':34s}{nf_tr[0]:+8.2f}(n{nf_tr[1]:4d})"
          + "".join(f"{nf[w][0]:+8.2f}(n{nf[w][1]:4d})" for w in W3))
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
