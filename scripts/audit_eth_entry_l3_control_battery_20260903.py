#!/usr/bin/env python3
"""진입 모델 **정직 라벨(L3) 대조군 4종** (2026-09-03).

배경: L0 라벨이 체결 봉의 체결-이전 고가를 크레딧해 PF 2.86 → 0.95로 무너졌다
(`docs/experiments/eth_entry_intrabar_fill_bar_credit_artifact_20260903.md`).
정직한 L3로 재학습하니 TabPFN 분류가 3창 전부 무작위 필터를 이겼다
(VAL +9.65 p=0.003 · OOS +19.22 p=0.000 · HOLDOUT +23.63 p=0.000).

⚠️그러나 **대조군이 무작위 필터 하나뿐**이다. L0에서 통과했던 나머지는 오염된 라벨로 채점된
것이라 무효다. 여기서 정직한 라벨 위에 다시 세운다:

  ① 모멘텀 뒤집기 -- 지정가를 **반대편**에 건다(페이드 대신 추격). 같은 트리거·같은 피쳐·
     같은 모델·같은 임계값. 페이드 **방향성**이 진짜인지 가린다. 통과 = 3창 손실이어야 한다.
  ② 무작위 봉     -- 트리거가 아닌 **무작위 봉**에서 같은 기계를 굴린다(모델 없음).
     트리거가 기여하는지 가린다.
  ③ 시드 로버스트니스 -- TabPFN 컨텍스트 추출 시드를 **새로 5개** 뽑아 성과 분산을 본다.
  ④ 시간블록 부트스트랩 -- **일 단위 군집** 부트스트랩으로 (필터 − 무필터) CI를 낸다.
     같은 날 트레이드는 독립이 아니므로 행 단위 부트스트랩은 CI를 과소평가한다.

⚠️HOLDOUT은 이미 여러 번 소진됐다. 여기 숫자는 전부 진단이다.
⚠️1분봉이 2026-07-31까지라 HOLDOUT 후보가 줄어 있다.
"""
from __future__ import annotations

import json
import os
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
from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402
from research_eth_entry_direction_oracle_ceiling_20260903 import (  # noqa: E402
    SL, ARM, TRAIL, NOTIONAL, COST)

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
L3D = ROOT / "tmp/eth_entry_1m_resolved_20260903/labels_1m_all.csv"
M1P = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
OUT = ROOT / "tmp/eth_entry_l3_controls_20260903"
DEPTH, WAIT, NSLOT, KEEP0, SUB = 3.0, 6, 4, 0.2037, 18000
NEW_SEEDS = [418271, 902334, 155709, 640182, 373845]     # 무작위 추출(고정 증분 아님)
W3 = ("VAL", "OOS", "HOLDOUT")
RNG = np.random.default_rng(20260903)


def log(m): print(f"[ctrl] {m}", flush=True)


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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
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

    def l3_label(sd_, e_, a_, f_, hz_):
        """1분해상 라벨 -- audit_eth_entry_1m_resolved_labels와 동일 규약."""
        bt = ts5[f_]
        if bt not in IDX0 or hz_ <= 0 or f_ + hz_ > n5: return np.nan
        s0, nn = IDX0[bt], IDXN[bt]
        sh, sl_ = m1h[s0:s0 + nn], m1l[s0:s0 + nn]
        hit = np.flatnonzero(sl_ <= e_) if sd_ > 0 else np.flatnonzero(sh >= e_)
        if not len(hit): return np.nan
        k0 = int(hit[0])
        ph, pl = sh[k0 + 1:], sl_[k0 + 1:]
        fh = float(ph.max()) if len(ph) else float(e_)
        fl = float(pl.min()) if len(pl) else float(e_)
        H = np.concatenate([[fh], h5[f_ + 1:f_ + hz_]])
        L = np.concatenate([[fl], l5[f_ + 1:f_ + hz_]])
        C = np.concatenate([[c5[f_]], c5[f_ + 1:f_ + hz_]])
        return trail(sd_, e_, a_, H, L, C) * NOTIONAL - COST * NOTIONAL

    # ---- 기준 데이터 ----
    LAB = pd.read_csv(L3D, parse_dates=["timestamp"])
    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    A = D.merge(LAB[["timestamp", "signal", "arm", "depth", "btf", "y_L3"]],
                on=["timestamp", "signal", "arm", "depth", "btf"], how="left")
    A = A[np.isfinite(A.y_L3)].reset_index(drop=True)
    dsel = ((A.depth == DEPTH) & (A.btf <= WAIT)).to_numpy()
    tr = (A.split == "TRAIN").to_numpy()
    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    excl = set(base + ["arm", "sig_id", "atr_pct", "depth", "y", "y_L3", "split", "timestamp",
                       "i", "side", "signal", "fi", "ei", "btf", "lim", "sd", "pred"])
    R = [c for c in A.columns if c.endswith("_r136")] + \
        [c for c in A.columns if c not in excl and not c.endswith("_r136")]
    R = list(dict.fromkeys([c for c in R if A[c].dtype.kind in "fiub"]))
    FE = list(dict.fromkeys(base + ["arm", "sig_id", "atr_pct", "depth"] + R))
    X = A[FE].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X[tr].median()).to_numpy(np.float32)
    y = A["y_L3"].to_numpy(float)
    lab = (y > 0.0040).astype(int)
    itr = np.flatnonzero(tr); prow = np.flatnonzero(dsel)
    M = {w: ((A.split == w).to_numpy() & dsel) for w in W3}
    log(f"모집단 {len(A):,} · 후보 {len(prow):,} · TRAIN {len(itr):,}")

    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    from tabpfn import TabPFNClassifier

    def fit_score(seeds, Xq):
        ps = []
        for sd_ in seeds:
            rs = np.random.default_rng(sd_).choice(itr, size=min(SUB, len(itr)), replace=False)
            m = TabPFNClassifier(device="cuda", random_state=sd_)
            m.fit(X[rs], lab[rs])
            ps.append(m.predict_proba(Xq)[:, 1])
        return np.mean(ps, axis=0)

    def perf(mask, keep_arr, yv):
        d = A[mask & keep_arr]
        t = slotN(d.assign(y=yv[mask & keep_arr]), NSLOT)
        return (float(np.mean(t) * 1e4) if len(t) else 0.0), int(len(t))

    def expand(v):
        f = np.full(len(A), -np.inf); f[prow] = v; return f

    log("\n기준 모델 (원 시드 5개) 채점...")
    P = expand(fit_score(SEEDS, X[prow]))
    thr = float(np.quantile(P[tr & dsel], 1 - KEEP0))
    keep = P > thr
    ones = np.ones(len(A), bool)
    nf = {w: perf(M[w], ones, y) for w in W3}
    bl = {w: perf(M[w], keep, y) for w in W3}
    print(f"\n{'':24s}" + "".join(f"{w:>14s}" for w in W3))
    print(f"{'무필터':24s}" + "".join(f"{nf[w][0]:+9.2f}(n{nf[w][1]:4d})" for w in W3))
    print(f"{'기준 모델':24s}" + "".join(f"{bl[w][0]:+9.2f}(n{bl[w][1]:4d})" for w in W3))

    # ---- ③ 시드 로버스트니스 ----
    log("\n③ 시드 로버스트니스 (새 시드 5개, 개별 멤버)...")
    sr = {w: [] for w in W3}
    for sd_ in NEW_SEEDS:
        p1 = expand(fit_score([sd_], X[prow]))
        k1 = p1 > float(np.quantile(p1[tr & dsel], 1 - KEEP0))
        for w in W3:
            sr[w].append(perf(M[w], k1, y)[0])
    print(f"\n=== ③ 시드 로버스트니스 (단일멤버 × 5 새 시드) ===")
    for w in W3:
        v = np.array(sr[w])
        print(f"  {w:8s} " + " ".join(f"{x:+7.2f}" for x in v)
              + f"  | 평균 {v.mean():+7.2f} 표준편차 {v.std():5.2f} · "
                f"무필터 초과 **{int((v > nf[w][0]).sum())}/5**")

    # ---- ④ 시간블록(일) 군집 부트스트랩 ----
    print(f"\n=== ④ 시간블록(일) 군집 부트스트랩 (필터 − 무필터, B=2000) ===")
    for w in W3:
        d_k = A[M[w] & keep]; d_n = A[M[w]]
        tk = slotN(d_k.assign(y=y[M[w] & keep]), NSLOT)
        days_k = pd.to_datetime(d_k.timestamp).dt.date.to_numpy()
        # 채택된 트레이드만 남기려면 slotN 인덱스가 필요하므로, 날짜별 평균으로 근사한다
        dk = pd.DataFrame({"d": days_k, "bp": y[M[w] & keep] * 1e4})
        dn = pd.DataFrame({"d": pd.to_datetime(d_n.timestamp).dt.date.to_numpy(),
                           "bp": y[M[w]] * 1e4})
        uk = dk.d.unique()
        diffs = []
        for _ in range(2000):
            s = RNG.choice(uk, size=len(uk), replace=True)
            a1 = dk[dk.d.isin(s)].bp.mean(); a2 = dn[dn.d.isin(s)].bp.mean()
            if np.isfinite(a1) and np.isfinite(a2): diffs.append(a1 - a2)
        diffs = np.array(diffs)
        lo_, hi_ = np.percentile(diffs, [2.5, 97.5])
        print(f"  {w:8s} 일수 {len(uk):3d} · 차이 평균 {diffs.mean():+7.2f}bp · "
              f"95% CI [{lo_:+7.2f}, {hi_:+7.2f}] {'✅' if lo_ > 0 else '❌'}")

    # ---- ① 모멘텀 뒤집기 ----
    log("\n① 모멘텀 뒤집기 (지정가 반대편, L3 재시뮬)...")
    S = A[dsel].reset_index(drop=True)
    i0 = S.i.to_numpy().astype(int); a0 = S.atr_pct.to_numpy(float)
    sd0 = S.sd.to_numpy(int); hz0 = (S.ei - S.fi).to_numpy().astype(int)
    flip_lim = np.where(sd0 > 0, c5[i0] * (1 + DEPTH * a0), c5[i0] * (1 - DEPTH * a0))
    yf = np.full(len(S), np.nan)
    for j in range(len(S)):
        ff = -1
        for off in range(1, WAIT + 1):
            k = i0[j] + off
            if k >= n5: break
            if (h5[k] >= flip_lim[j]) if sd0[j] > 0 else (l5[k] <= flip_lim[j]):
                ff = k; break
        if ff < 0: continue
        yf[j] = l3_label(sd0[j], float(flip_lim[j]), float(a0[j]), ff, int(hz0[j]))
    S["yf"] = yf
    Sv = np.isfinite(yf)
    print(f"\n=== ① 모멘텀 뒤집기 (추격) -- 3창 손실이어야 통과 ===")
    kp = keep[dsel]
    for w in W3:
        mw = (S.split == w).to_numpy() & Sv
        for tag, kk in (("무필터", np.ones(len(S), bool)), ("모델필터", kp)):
            d = S[mw & kk]
            t = slotN(d.assign(y=d.yf), NSLOT)
            v = float(np.mean(t) * 1e4) if len(t) else 0.0
            print(f"  {w:8s} {tag:8s} {v:+8.2f} (n{len(t):4d})")

    # ---- ② 무작위 봉 ----
    log("\n② 무작위 봉 진입 (기계만, 모델 없음)...")
    print(f"\n=== ② 무작위 봉 vs 트리거 봉 (둘 다 무필터) ===")
    for w in W3:
        mw = M[w]
        n_want = int(mw.sum())
        lo_i, hi_i = int(A[mw].i.min()), int(A[mw].i.max())
        vals = []
        for _ in range(5):
            ridx = RNG.integers(lo_i, hi_i, size=n_want)
            ra = np.interp(ridx, i0, a0)                       # ATR은 인접 트리거에서 보간
            rs_ = RNG.choice([1, -1], size=n_want)
            rl = np.where(rs_ > 0, c5[ridx] * (1 - DEPTH * ra), c5[ridx] * (1 + DEPTH * ra))
            rec = []
            for j in range(n_want):
                ff = -1
                for off in range(1, WAIT + 1):
                    k = ridx[j] + off
                    if k >= n5: break
                    if (l5[k] <= rl[j]) if rs_[j] > 0 else (h5[k] >= rl[j]):
                        ff = k; break
                if ff < 0: continue
                yy = l3_label(rs_[j], float(rl[j]), float(ra[j]), ff, 24)
                if np.isfinite(yy): rec.append({"fi": ff, "ei": ff + 24, "y": yy})
            if rec:
                t = slotN(pd.DataFrame(rec), NSLOT)
                vals.append(float(np.mean(t) * 1e4) if len(t) else 0.0)
        v = np.array(vals)
        print(f"  {w:8s} 트리거 무필터 {nf[w][0]:+7.2f} · 무작위 봉 {v.mean():+7.2f} "
              f"(±{v.std():.2f}, {len(v)}회)")

    json.dump({"no_filter": nf, "model": bl, "seed_robust": sr},
              open(OUT / "result.json", "w"), ensure_ascii=False, indent=2, default=str)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
