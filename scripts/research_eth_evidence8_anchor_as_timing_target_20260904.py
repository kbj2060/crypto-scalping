#!/usr/bin/env python3
"""앵커를 **진입점이 아니라 타이밍 타깃**으로 -- 특권정보 학습(LUPI) 시험 (2026-09-04).

## 발상 (사용자)

`cluster_dedup` 앵커는 클러스터의 극단이라 결정 시점에 알 수 없다 -- 그래서 진입점으로 쓰면
미래참조다(5.16절, +11.73 -> −5.66). **그런데 학습 타깃으로 쓰는 건 다른 이야기다.**
사후 정보로 더 깨끗한 타깃을 만들고 추론은 인과적 입력만 쓰는 건 정식 기법이다(LUPI).

    라벨(사후)  이 발동에서 클러스터 극단까지 **얼마나 더 역행하는가**(ATR 단위)
    학습        인과 피쳐(Tier0)만으로 그 역행폭을 예측
    운용        예측 역행폭이 작을 때만 진입 -> "아직 더 간다" 구간을 회피

**왜 이게 방향 예측과 다른가**: 오늘 11축이 실패한 이유는 Tier0가 크기는 알고 방향은 모른다는
것이었다. "극단이 아직 안 왔나"는 방향이 아니라 **크기·타이밍** 질문이라 어긋나지 않는다.

## 구조 -- ⭐오라클을 먼저 본다

  **A 오라클 게이트**  *진짜* 역행폭 하위 분위만 진입 -> 경제성. **여기서 못 이기면
     예측력과 무관하게 죽은 아이디어**다(예측이 완벽해도 그 이상은 못 얻는다).
  **B 학습 게이트**    A가 통과할 때만 -- Tier0로 역행폭을 예측해 같은 게이트를 건다.

모집단은 **인과 첫발동이 아니라 raw 발동 전부**다 -- 라이브는 매 발동마다 "지금 들어갈까
기다릴까"를 결정하므로 그게 실제 결정 지점이다.

⚠️TRAIN/VAL만. OOS·HOLDOUT 미터치.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_pf = _load("pf_anc", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0, sim_exit = _pf.TIER0, _pf.sim_exit

SIGNALS = {"liquidity_sweep": 30, "taker_delta_z_climax": 24, "short_term_return_z": 12,
           "orthogonal_combo": 24, "smt_divergence": 72, "fib_extension_exhaustion": 20,
           "demarker_extreme": 8, "kalman_deviation_meanrev": 12}
CELL = (4.0, 1.0, 0.1)
GAP, COST_BP, SEED = 12, 10.0, 20260904
GATE_Q = [0.10, 0.25, 0.50]          # 예측 역행폭 하위 분위 게이트
OUT = ROOT / "data/research/eth_evidence8_anchor_timing_20260904/report.json"


def log(m): print(f"[anc] {m}", flush=True)


def cluster_t(vals, days):
    if len(vals) < 5:
        return np.nan
    dev = vals - vals.mean()
    s = sum(dev[days == d].sum() ** 2 for d in np.unique(days))
    se = np.sqrt(s) / len(vals)
    return vals.mean() / se if se > 0 else np.nan


def clusters_of(fire_idx, gap):
    """연속 발동을 gap 안에서 한 클러스터로 묶는다. [(멤버 인덱스 배열), ...]"""
    out, cur = [], []
    for i in fire_idx:
        if cur and i - cur[-1] > gap:
            out.append(np.array(cur)); cur = []
        cur.append(i)
    if cur:
        out.append(np.array(cur))
    return out


def main() -> int:
    t0 = time.time()
    log("프레임 빌드...")
    sig, feat, eth = _s1.build_sig()
    dummy = np.full(len(sig), "none", dtype=object)
    long = _s1.long_frame_for(sig, feat, dummy, dummy)
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    if kl["timestamp"].dt.tz is not None:
        kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    o, h, l, c = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(kl)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    ltsr = long["timestamp"]
    lts = (ltsr.dt.tz_localize(None) if ltsr.dt.tz is not None else ltsr).to_numpy()
    LONG = long.copy(); LONG["_ts"] = lts
    lpos_all = np.array([pos_of.get(np.datetime64(t), -1) for t in LONG["_ts"].to_numpy()])
    isd_all = LONG["is_downside"].to_numpy().astype(bool)
    T0 = [x for x in TIER0 if x in LONG.columns]
    # (pos, is_down) -> LONG 행 인덱스
    row_of = {(int(pp), bool(dd)): q for q, (pp, dd) in enumerate(zip(lpos_all, isd_all))
              if pp >= 0}

    parts = []
    for SIGNAL, HZ in SIGNALS.items():
        bcol, tcol = f"bottom_{SIGNAL}", f"top_{SIGNAL}"
        S = sig[["timestamp", bcol, tcol]].copy()
        if S["timestamp"].dt.tz is not None:
            S["timestamp"] = S["timestamp"].dt.tz_localize(None)
        S["pos"] = [pos_of.get(np.datetime64(t), -1) for t in S["timestamp"].to_numpy()]
        S = S.loc[S["pos"] >= 0]
        fbv = np.zeros(n, bool); ftv = np.zeros(n, bool)
        fbv[S["pos"].to_numpy()] = S[bcol].fillna(False).to_numpy(bool)
        ftv[S["pos"].to_numpy()] = S[tcol].fillna(False).to_numpy(bool)

        for fire, is_down in ((fbv, True), (ftv, False)):
            idx = np.flatnonzero(fire)
            if len(idx) < 50:
                continue
            for mem in clusters_of(idx, GAP):
                mem = mem[(mem + 1 + HZ < n) & (mem >= 1)]
                if not len(mem):
                    continue
                lo_hi = mem[0], mem[-1]
                span = np.arange(lo_hi[0], lo_hi[1] + 1)
                # ⭐클러스터 극단 = 앵커 (사후에만 알 수 있다)
                e = int(span[np.argmin(l[span])] if is_down else span[np.argmax(h[span])])
                for i in mem:
                    q = row_of.get((int(i), is_down))
                    if q is None:
                        continue
                    a_ = float(LONG["atr"].to_numpy()[q])
                    if not np.isfinite(a_) or a_ <= 0:
                        continue
                    # ⭐남은 역행폭(ATR): 이 발동에서 극단까지 얼마나 더 불리하게 가나
                    if e > i:
                        w = slice(i + 1, e + 1)
                        adv = (c[i] - l[w].min()) if is_down else (h[w].max() - c[i])
                    else:
                        adv = 0.0
                    parts.append({"row": q, "pos": int(i), "is_down": is_down,
                                  "signal": SIGNAL, "hz": HZ, "atr": a_,
                                  "adverse_atr": max(0.0, float(adv) / a_),
                                  "bars_to_extreme": int(max(0, e - i))})
    P = pd.DataFrame(parts)
    log(f"⭐raw 발동 전부 {len(P):,}건 (인과 첫발동만 쓰던 앞 실험과 다른 모집단)")

    # 라벨(경제성) -- 진입 o[i+1], 자기 HORIZON 트레일링
    net = np.empty(len(P))
    for HZ in sorted(P["hz"].unique()):
        m = (P["hz"] == HZ).to_numpy()
        ii = P.loc[m, "pos"].to_numpy().astype(int)
        sg = np.where(P.loc[m, "is_down"].to_numpy(), 1.0, -1.0)
        entry = o[ii + 1]
        H = np.stack([h[i + 1:i + 1 + HZ] for i in ii])
        L = np.stack([l[i + 1:i + 1 + HZ] for i in ii])
        C = np.stack([c[i + 1:i + 1 + HZ] for i in ii])
        pn, _ = sim_exit(entry, P.loc[m, "atr"].to_numpy(float), sg, H, L, C, *CELL)
        net[m] = pn * 1e4 - COST_BP
    P["net_bp"] = net
    P["split"] = LONG["split"].to_numpy()[P["row"].to_numpy()]
    P["_ts"] = LONG["_ts"].to_numpy()[P["row"].to_numpy()]
    days = pd.to_datetime(P["_ts"]).dt.floor("D").to_numpy()
    tr, va = (P["split"] == "TRAIN").to_numpy(), (P["split"] == "VAL").to_numpy()
    log(f"  TRAIN {tr.sum():,} / VAL {va.sum():,} · 전체 평균 {P['net_bp'].mean():+.2f}bp")
    log(f"  남은 역행폭(ATR) 중앙 {P['adverse_atr'].median():.2f} · "
        f"극단까지 봉수 중앙 {P['bars_to_extreme'].median():.0f}")

    # ========== A. 오라클 게이트 ==========
    log("\n=== ⭐A. 오라클 게이트 (진짜 역행폭을 안다고 가정) ===")
    print(f"{'신호':>24s}{'게이트':>8s}{'n':>7s}{'독립일':>7s}{'전체bp':>9s}{'게이트bp':>10s}{'일t':>7s}")
    print("-" * 74)
    orc = {}
    nv_all, dv_all = P["net_bp"].to_numpy()[va], days[va]
    for s_ in SIGNALS:
        mk = (P["signal"] == s_).to_numpy() & va
        if mk.sum() < 100:
            continue
        base = float(P["net_bp"].to_numpy()[mk].mean())
        for gq in GATE_Q:
            thr = np.quantile(P["adverse_atr"].to_numpy()[mk], gq)
            g = mk & (P["adverse_atr"].to_numpy() <= thr)
            if g.sum() < 30:
                continue
            gv, gd = P["net_bp"].to_numpy()[g], days[g]
            tt = float(cluster_t(gv, gd))
            print(f"{s_[:23]:>24s}{gq:8.2f}{int(g.sum()):7d}{len(np.unique(gd)):7d}"
                  f"{base:9.2f}{float(gv.mean()):10.2f}{tt:7.2f}"
                  f"{'  ⭐' if tt > 1.96 else ''}")
            orc[f"{s_}|q{gq}"] = {"n": int(g.sum()), "all_mean_bp": base,
                                  "gated_mean_bp": float(gv.mean()), "cluster_t": tt,
                                  "independent_days": int(len(np.unique(gd)))}
    npass = sum(1 for v in orc.values() if v["cluster_t"] > 1.96)
    sigs = sorted({k.split("|")[0] for k, v in orc.items() if v["cluster_t"] > 1.96})
    log(f"\n  ⭐오라클 통과 {npass}/{len(orc)} 조합 · 신호 {len(sigs)}/8: {sigs}")

    res = {"oracle": orc, "n_oracle_pass": npass, "oracle_signals": sigs}

    # ========== ⭐A2. 동어반복 분해 ==========
    # 오라클 게이트는 사실상 `역행폭==0` = "극단이 이미 들어왔다"로 수렴한다(분위 0.10/0.25/0.50이
    # 같은 결과). 저점이 들어왔으면 롱이 이기는 건 **부분적으로 정의상 참**이라, 믿기 전에 둘로
    # 나눠야 한다:
    #   C1 무작위 게이트   같은 선택 수를 무작위로 -> 오라클이 그 분포를 넘나
    #   C2 ⭐**후방 극단**  "이 봉이 직전 GAP봉 중 최저(롱)/최고(숏)인가" -- **100% 인과적**이다.
    #      오라클 엣지의 상당 부분을 이걸로 잡을 수 있으면 **학습 없이 배포 가능한 규칙**이 된다.
    log("\n=== ⭐A2. 동어반복 분해 (무작위 게이트 · 후방 극단) ===")
    rng2 = np.random.default_rng(SEED)
    pos_arr = P["pos"].to_numpy().astype(int)
    isd_arr = P["is_down"].to_numpy().astype(bool)
    # C2: 후방 GAP봉 극단 여부 (미래 미참조)
    back = np.zeros(len(P), bool)
    for q_ in range(len(P)):
        i = pos_arr[q_]
        w = slice(max(0, i - GAP), i + 1)
        back[q_] = (l[i] <= l[w].min() + 1e-12) if isd_arr[q_] else (h[i] >= h[w].max() - 1e-12)
    log(f"  후방 극단 비율 {back.mean():.3f} · 오라클(역행폭 0) 비율 "
        f"{(P['adverse_atr'].to_numpy() <= 1e-12).mean():.3f} · "
        f"둘 다 참 {(back & (P['adverse_atr'].to_numpy() <= 1e-12)).mean():.3f}")
    print(f"\n{'신호':>24s}{'전체bp':>9s}{'오라클bp':>10s}{'무작위bp':>10s}"
          f"{'후방극단bp':>11s}{'후방일t':>8s}{'n':>7s}")
    print("-" * 80)
    dec = {}
    netv = P["net_bp"].to_numpy()
    zero = P["adverse_atr"].to_numpy() <= 1e-12
    for s_ in SIGNALS:
        mk = (P["signal"] == s_).to_numpy() & va
        if mk.sum() < 100:
            continue
        g_o = mk & zero
        g_b = mk & back
        if g_o.sum() < 30 or g_b.sum() < 30:
            continue
        base = float(netv[mk].mean())
        ob = float(netv[g_o].mean())
        pool = netv[mk]
        rnd = np.array([pool[rng2.choice(len(pool), int(g_o.sum()), replace=False)].mean()
                        for _ in range(400)])
        bb, bt = float(netv[g_b].mean()), float(cluster_t(netv[g_b], days[g_b]))
        print(f"{s_[:23]:>24s}{base:9.2f}{ob:10.2f}{float(rnd.mean()):10.2f}"
              f"{bb:11.2f}{bt:8.2f}{int(g_b.sum()):7d}{'  ⭐' if bt > 1.96 else ''}")
        dec[s_] = {"all_bp": base, "oracle_bp": ob, "random_gate_bp": float(rnd.mean()),
                   "oracle_vs_random_p": float((rnd >= ob).mean()),
                   "backward_extreme_bp": bb, "backward_cluster_t": bt,
                   "n_backward": int(g_b.sum())}
    nb = sum(1 for v in dec.values() if v["backward_cluster_t"] > 1.96)
    log(f"\n  ⭐**후방 극단(100% 인과)** 통과 신호: **{nb}/{len(dec)}**")
    log(f"  오라클이 무작위 게이트를 넘은 신호: "
        f"{sum(1 for v in dec.values() if v['oracle_vs_random_p'] < 0.05)}/{len(dec)}")
    res["decomposition"] = dec
    res["n_backward_pass"] = nb
    if not sigs:
        log("  ❌**오라클도 못 이긴다** -- 역행폭을 완벽히 알아도 경제성이 안 난다.")
        log("     ⇒ 예측력과 무관하게 이 아이디어는 죽었다. B단계 생략.")
    else:
        # ========== B. 학습 게이트 ==========
        log("\n=== B. 학습 게이트 (Tier0로 역행폭 예측) ===")
        from sklearn.ensemble import HistGradientBoostingRegressor
        from scipy.stats import spearmanr
        X = np.nan_to_num(LONG.iloc[P["row"].to_numpy()][T0]
                          .apply(pd.to_numeric, errors="coerce").to_numpy(float),
                          nan=0.0, posinf=0.0, neginf=0.0)
        y = P["adverse_atr"].to_numpy(float)
        m = HistGradientBoostingRegressor(random_state=SEED, max_iter=300, learning_rate=0.05)
        m.fit(X[tr], y[tr])
        pv = m.predict(X[va])
        ic = float(spearmanr(pv, y[va]).correlation)
        log(f"  역행폭 예측 IC(VAL) = {ic:+.4f}")
        lrn = {}
        print(f"{'신호':>24s}{'게이트':>8s}{'n':>7s}{'전체bp':>9s}{'게이트bp':>10s}{'일t':>7s}")
        print("-" * 67)
        vidx = np.flatnonzero(va)
        for s_ in sigs:
            mk_v = (P["signal"].to_numpy()[vidx] == s_)
            if mk_v.sum() < 100:
                continue
            base = float(P["net_bp"].to_numpy()[vidx][mk_v].mean())
            for gq in GATE_Q:
                thr = np.quantile(pv[mk_v], gq)
                g = mk_v & (pv <= thr)
                if g.sum() < 30:
                    continue
                gv, gd = P["net_bp"].to_numpy()[vidx][g], days[vidx][g]
                tt = float(cluster_t(gv, gd))
                print(f"{s_[:23]:>24s}{gq:8.2f}{int(g.sum()):7d}{base:9.2f}"
                      f"{float(gv.mean()):10.2f}{tt:7.2f}{'  ⭐' if tt > 1.96 else ''}")
                lrn[f"{s_}|q{gq}"] = {"n": int(g.sum()), "gated_mean_bp": float(gv.mean()),
                                      "cluster_t": tt}
        res["learned"] = lrn
        res["adverse_pred_ic"] = ic
        res["n_learned_pass"] = sum(1 for v in lrn.values() if v["cluster_t"] > 1.96)
        log(f"\n  ⭐학습 게이트 통과 {res['n_learned_pass']}/{len(lrn)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.update({"n_pool": int(len(P)), "oos_touched": False,
                "runtime_sec": round(time.time() - t0, 1)})
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    log(f"산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
