#!/usr/bin/env python3
"""증거신호 8종 경제성 회귀 -- **지정가 진입**으로 모집단 자체를 개선 (목표: 8종 관문 통과).

## 왜 지정가인가

시장가(o[i+1]) 실험에서 8종 중 **fib만** 일단위 클러스터 t>1.96을 넘었다(TabPFN 회귀 t=4.15).
원인을 보니 **fib는 8종 중 유일하게 모집단 전체 평균이 양수**(+3.19bp)였고, 나머지 7종은
−1.18 ~ −10.47bp로 전부 음수였다. 음수 모집단에서는 모델이 상위 10%를 골라도 "덜 잃는" 것에
그친다. 피쳐 확장(재료텐서·klines 여분)은 둘 다 실패해 정보량 축은 이미 막혔다.

⇒ 남은 축은 **진입 가격**이다. 5.20절 결론이 정확히 *"문제는 트리거가 아니라 진입 가격"* 이고,
지정가는 그 8축 중 유일하게 생존 기미를 보였다(XRP 2종).

## 라벨

    지정가   신호 방향으로 유리한 쪽 depth x ATR (롱=아래, 숏=위)
    체결     i+1 .. i+WAIT 봉 중 처음 닿는 봉 j. 미체결이면 **후보에서 제외**(주문 취소)
    청산     ⚠️체결 봉 j **다음 봉부터** 트레일링 -- 체결 봉 자체를 크레딧하면 오늘 진입모델을
             무효화한 B버그(체결 이전 고가를 진입 후 이익으로 계상)가 그대로 재현된다
    net      (청산 - 지정가)/지정가 x 방향 x 1e4 - 비용

depth는 사전등록 격자 3개를 **전부 보고**한다(고르지 않는다 -- 고르면 선택편향).
생존은 depth 간 일관성으로 판정한다.

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


_pf = _load("pf_lim", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0, sim_exit = _pf.TIER0, _pf.sim_exit

SIGNALS = {"liquidity_sweep": 30, "taker_delta_z_climax": 24, "short_term_return_z": 12,
           "orthogonal_combo": 24, "smt_divergence": 72, "fib_extension_exhaustion": 20,
           "demarker_extreme": 8, "kalman_deviation_meanrev": 12}
CELL = (4.0, 1.0, 0.1)
DEPTHS = [0.5, 1.0, 1.5]        # 사전등록 격자 -- 전부 보고, 고르지 않는다
WAIT = 6
GAP, COST_BP, TOPQ, SEED = 12, 10.0, 0.10, 20260904
OUT = ROOT / "data/research/eth_evidence8_econ_limit_entry_20260904/report.json"


def log(m): print(f"[lim] {m}", flush=True)


def causal_first_fire(fire, gap):
    keep = np.zeros(len(fire), bool); last = -10**9
    for i in np.flatnonzero(fire):
        if i - last > gap:
            keep[i] = True
        last = i
    return keep


def cluster_t(vals, days):
    n = len(vals)
    if n < 5:
        return np.nan
    dev = vals - vals.mean()
    s = sum(dev[days == d].sum() ** 2 for d in np.unique(days))
    se = np.sqrt(s) / n
    return vals.mean() / se if se > 0 else np.nan


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

    from sklearn.ensemble import HistGradientBoostingRegressor
    from tabpfn import TabPFNRegressor

    print(f"\n{'신호':>22s}{'depth':>7s}{'모델':>10s}{'체결률':>8s}{'k':>5s}{'독립일':>7s}"
          f"{'전체bp':>9s}{'상위bp':>9s}{'일t':>7s}")
    print("-" * 88)
    rep = {}

    for SIGNAL, HZ in SIGNALS.items():
        bcol, tcol = f"bottom_{SIGNAL}", f"top_{SIGNAL}"
        S = sig[["timestamp", bcol, tcol]].copy()
        if S["timestamp"].dt.tz is not None:
            S["timestamp"] = S["timestamp"].dt.tz_localize(None)
        S["pos"] = [pos_of.get(np.datetime64(t), -1) for t in S["timestamp"].to_numpy()]
        S = S.loc[S["pos"] >= 0]
        fb = np.zeros(n, bool); ft = np.zeros(n, bool)
        fb[S["pos"].to_numpy()] = S[bcol].fillna(False).to_numpy(bool)
        ft[S["pos"].to_numpy()] = S[tcol].fillna(False).to_numpy(bool)
        kb, kt = causal_first_fire(fb, GAP), causal_first_fire(ft, GAP)

        lpos = np.array([pos_of.get(np.datetime64(t), -1) for t in LONG["_ts"].to_numpy()])
        isd = LONG["is_downside"].to_numpy().astype(bool)
        base_keep = (lpos >= 0) & (lpos + 1 + WAIT + HZ + 1 < n)
        base_keep &= np.where(isd, kb[np.clip(lpos, 0, n - 1)], kt[np.clip(lpos, 0, n - 1)])
        D0 = LONG.loc[base_keep].reset_index(drop=True)
        i0 = lpos[base_keep]
        if len(D0) < 400:
            continue
        sg0 = np.where(D0["is_downside"].to_numpy() == 1, 1.0, -1.0)
        atr0 = D0["atr"].to_numpy(float)
        X0 = D0[[x for x in TIER0 if x in D0.columns]].apply(
            pd.to_numeric, errors="coerce").to_numpy(float)
        X0 = np.nan_to_num(X0, nan=0.0, posinf=0.0, neginf=0.0)

        for depth in DEPTHS:
            lim = c[i0] - sg0 * depth * atr0        # 롱=아래, 숏=위 (유리한 쪽)
            fj = np.full(len(i0), -1)
            for q, i in enumerate(i0):
                w = slice(i + 1, i + 1 + WAIT)
                hit = (l[w] <= lim[q]) if sg0[q] > 0 else (h[w] >= lim[q])
                nz = np.flatnonzero(hit)
                if len(nz):
                    fj[q] = i + 1 + nz[0]
            fm = fj >= 0                              # 체결분만
            fill_rate = float(fm.mean())
            if fm.sum() < 400:
                continue
            D = D0.loc[fm].reset_index(drop=True)
            jj, sg, X = fj[fm], sg0[fm], X0[fm]
            entry = lim[fm]
            # ⚠️체결 봉 **다음 봉부터** -- 체결 봉 크레딧 금지(오늘 진입모델 B버그)
            H = np.stack([h[j + 1:j + 1 + HZ] for j in jj])
            L = np.stack([l[j + 1:j + 1 + HZ] for j in jj])
            C = np.stack([c[j + 1:j + 1 + HZ] for j in jj])
            pn, _ = sim_exit(entry, D["atr"].to_numpy(float), sg, H, L, C, *CELL)
            net = pn * 1e4 - COST_BP
            split = D["split"].to_numpy()
            tr, va = split == "TRAIN", split == "VAL"
            if tr.sum() < 300 or va.sum() < 100:
                continue
            days = pd.to_datetime(D["_ts"]).dt.floor("D").to_numpy()
            k = max(10, int(round(va.sum() * TOPQ)))
            nv, dv = net[va], days[va]
            base = float(nv.mean())

            for mname in ("HGB_reg", "TabPFN_reg"):
                try:
                    m = (HistGradientBoostingRegressor(random_state=SEED, max_iter=300,
                                                       learning_rate=0.05)
                         if mname == "HGB_reg" else
                         TabPFNRegressor(device="cuda", random_state=SEED,
                                         ignore_pretraining_limits=True))
                    m.fit(X[tr], net[tr])
                    pred = m.predict(X[va])
                    top = np.argsort(-pred)[:k]
                    tb, tt = float(nv[top].mean()), float(cluster_t(nv[top], dv[top]))
                    print(f"{SIGNAL[:21]:>22s}{depth:7.1f}{mname:>10s}{fill_rate:8.2f}{k:5d}"
                          f"{len(np.unique(dv[top])):7d}{base:9.2f}{tb:9.2f}{tt:7.2f}"
                          f"{'  ⭐' if tt > 1.96 else ''}")
                    rep[f"{SIGNAL}|d{depth}|{mname}"] = {
                        "depth": depth, "fill_rate": fill_rate, "k": k,
                        "all_mean_bp": base, "top_mean_bp": tb, "cluster_t": tt,
                        "independent_days": int(len(np.unique(dv[top]))),
                        "n_val": int(va.sum()), "n_filled": int(fm.sum())}
                except Exception as e:                            # noqa: BLE001
                    log(f"  ⚠️{SIGNAL}|d{depth}|{mname}: {type(e).__name__}")
        print("-" * 88)

    surv = {k_: v for k_, v in rep.items() if v["cluster_t"] > 1.96}
    posbase = {k_: v for k_, v in rep.items() if v["all_mean_bp"] > 0}
    print()
    log(f"⭐일단위 t>1.96 : **{len(surv)}/{len(rep)}** (노이즈 기대 {len(rep)*0.025:.1f})")
    log(f"⭐모집단 평균이 **양수**인 조합: {len(posbase)}/{len(rep)}  "
        f"(시장가에선 8종 중 fib 하나뿐이었다)")
    for k_, v in sorted(surv.items(), key=lambda x: -x[1]["cluster_t"])[:14]:
        log(f"    {k_:<50s} 상위{v['top_mean_bp']:+7.2f} t={v['cluster_t']:.2f} "
            f"(전체{v['all_mean_bp']:+.2f} 체결{v['fill_rate']:.2f} 일{v['independent_days']})")
    sigs = sorted({k_.split("|")[0] for k_ in surv})
    log(f"⭐통과 신호 {len(sigs)}/8: {sigs}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"combos": rep, "cell": list(CELL), "depths": DEPTHS,
                               "wait": WAIT, "cost_bp": COST_BP, "n_survived": len(surv),
                               "signals_passed": sigs, "oos_touched": False,
                               "runtime_sec": round(time.time() - t0, 1)},
                              ensure_ascii=False, indent=2))
    log(f"산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
