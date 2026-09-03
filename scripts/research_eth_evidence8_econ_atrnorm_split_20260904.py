#!/usr/bin/env python3
"""증거신호 8종 경제성 -- **ATR정규화 라벨 · 방향/크기 분리 · 방향반전** (3개 신규 축).

## 왜 이 셋인가 -- 앞선 8축 실패의 진단에서 직접 나온다

피쳐 제거 실험의 TRAIN IC 랭킹이 `atr` 0.326을 1위로 놓았고 방향성 피쳐는 IC≈0이었다.
그런데 **`atr`의 높은 IC는 정보가 아니라 비용 아티팩트일 수 있다**: 트레일링 배리어는 전부
ATR에 비례해 커지는데 **비용 10bp는 고정**이다. 저ATR 구간은 비용 비중이 커져 net_bp가
구조적으로 음수가 된다. 즉 모델이 배운 것은 *"고ATR을 골라라"*(비용이 덜 아픈 쪽)이지
방향이 아니다. 실제로 상위10%를 뽑아도 여전히 음수였다 -- 이미 소진된 축이다.

  ①**ATR 정규화 라벨** `net/atr`로 학습해 기계적 ATR 관계를 제거한다. 모델이 크기 대신
    방향 구조를 찾도록 강제된다. 평가는 여전히 **실제 net_bp(돈)** 로 한다.
    변동성 타깃 사이징(비중 ∝ 1/atr)도 같이 잰다 -- ATR정규화 예측의 자연스러운 짝이다.
  ②**방향/크기 분리** 하나의 모델에 부호붙은 net을 통째로 맡기면 변동성이 지배한다.
    `sign(net)` 분류와 `|net|` 회귀를 **따로** 학습해 기대값으로 결합한다.
  ③**방향 반전** 첫 실험에서 "발동이 무작위 진입보다 나쁜 조합"이 15/32였다. 체계적으로
    반대인 신호가 있다면 뒤집는 것만으로 양수가 된다. ⚠️사후선택이 되지 않도록 **8종 전부**를
    보고하고, 반전이 통하는지는 신호별 일단위 t로 판정한다.

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


_pf = _load("pf_an", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0, sim_exit = _pf.TIER0, _pf.sim_exit

SIGNALS = {"liquidity_sweep": 30, "taker_delta_z_climax": 24, "short_term_return_z": 12,
           "orthogonal_combo": 24, "smt_divergence": 72, "fib_extension_exhaustion": 20,
           "demarker_extreme": 8, "kalman_deviation_meanrev": 12}
CELL = (4.0, 1.0, 0.1)
GAP, COST_BP, TOPQ, SEED = 12, 10.0, 0.10, 20260904
OUT = ROOT / "data/research/eth_evidence8_econ_atrnorm_split_20260904/report.json"


def log(m): print(f"[an] {m}", flush=True)


def causal_first_fire(fire, gap):
    keep = np.zeros(len(fire), bool); last = -10**9
    for i in np.flatnonzero(fire):
        if i - last > gap:
            keep[i] = True
        last = i
    return keep


def cluster_t(vals, days):
    if len(vals) < 5:
        return np.nan
    dev = vals - vals.mean()
    s = sum(dev[days == d].sum() ** 2 for d in np.unique(days))
    se = np.sqrt(s) / len(vals)
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
    lpos_all = np.array([pos_of.get(np.datetime64(t), -1) for t in LONG["_ts"].to_numpy()])
    isd_all = LONG["is_downside"].to_numpy().astype(bool)
    T0 = [x for x in TIER0 if x in LONG.columns]

    parts = []
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
        keep = (lpos_all >= 0) & (lpos_all + 1 + HZ < n)
        keep &= np.where(isd_all, kb[np.clip(lpos_all, 0, n - 1)], kt[np.clip(lpos_all, 0, n - 1)])
        if keep.sum() < 300:
            continue
        cols = list(dict.fromkeys(["_ts", "split", "is_downside", "atr"] + T0))
        D = LONG.loc[keep, cols].reset_index(drop=True)
        ii = lpos_all[keep]
        sg = np.where(D["is_downside"].to_numpy() == 1, 1.0, -1.0)
        entry = o[ii + 1]
        a_ = D["atr"].to_numpy(float)
        H = np.stack([h[i + 1:i + 1 + HZ] for i in ii])
        L = np.stack([l[i + 1:i + 1 + HZ] for i in ii])
        C = np.stack([c[i + 1:i + 1 + HZ] for i in ii])
        pn, _ = sim_exit(entry, a_, sg, H, L, C, *CELL)
        D["net_bp"] = pn * 1e4 - COST_BP
        # ⭐③방향 반전: 같은 후보를 반대 방향으로 잡았을 때
        pnf, _ = sim_exit(entry, a_, -sg, H, L, C, *CELL)
        D["net_bp_flip"] = pnf * 1e4 - COST_BP
        D["atr_pct_bp"] = a_ / entry * 1e4          # ATR을 bp로
        D["signal"] = SIGNAL
        parts.append(D)
    P = pd.concat(parts, ignore_index=True)
    for s_ in SIGNALS:
        P[f"is_{s_}"] = (P["signal"] == s_).astype(np.int8)
    ONE = [f"is_{s_}" for s_ in SIGNALS]
    net = P["net_bp"].to_numpy(float)
    netf = P["net_bp_flip"].to_numpy(float)
    atrbp = np.clip(P["atr_pct_bp"].to_numpy(float), 1e-6, None)
    net_atr = net / atrbp                            # ⭐①ATR 정규화
    split = P["split"].to_numpy()
    tr, va = split == "TRAIN", split == "VAL"
    days = pd.to_datetime(P["_ts"]).dt.floor("D").to_numpy()
    X = np.nan_to_num(P[T0 + ONE].apply(pd.to_numeric, errors="coerce").to_numpy(float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    log(f"⭐풀 {len(P):,}건 (TRAIN {tr.sum():,} / VAL {va.sum():,})")

    # ============ ③방향 반전 -- 학습 없이 즉시 판정 ============
    log("\n=== ③방향 반전 (학습 불필요) ===")
    print(f"{'신호':>24s}{'VAL n':>8s}{'정방향bp':>10s}{'반전bp':>10s}{'반전 일t':>10s}")
    print("-" * 64)
    flip_rep = {}
    Pva = P.loc[va].reset_index(drop=True)
    for s_ in SIGNALS:
        mk = (Pva["signal"] == s_).to_numpy()
        if mk.sum() < 80:
            continue
        fwd, rev, dd = net[va][mk], netf[va][mk], days[va][mk]
        tt = float(cluster_t(rev, dd))
        print(f"{s_[:23]:>24s}{int(mk.sum()):8d}{float(fwd.mean()):10.2f}"
              f"{float(rev.mean()):10.2f}{tt:10.2f}{'  ⭐' if tt > 1.96 else ''}")
        flip_rep[s_] = {"fwd_mean_bp": float(fwd.mean()), "flip_mean_bp": float(rev.mean()),
                        "flip_cluster_t": tt, "n_val": int(mk.sum())}

    # ============ ①ATR정규화 · ②분리 예측 ============
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    rng = np.random.default_rng(SEED)
    tri = np.flatnonzero(tr)
    ctx = rng.choice(tri, size=min(18000, len(tri)), replace=False)
    preds = {}
    log("\n학습 중...")
    m = TabPFNRegressor(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
    m.fit(X[ctx], net[ctx]); preds["A_net_bp(기준)"] = m.predict(X[va])
    m = TabPFNRegressor(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
    m.fit(X[ctx], net_atr[ctx]); preds["B_net/atr(①정규화)"] = m.predict(X[va])
    # ②분리: 부호 분류 x 크기 회귀
    ysign = (net > 0).astype(int)
    mc = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
    mc.fit(X[ctx], ysign[ctx]); p_up = mc.predict_proba(X[va])[:, 1]
    mr = TabPFNRegressor(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
    mr.fit(X[ctx], np.abs(net[ctx])); mag = mr.predict(X[va])
    preds["C_분리(②부호x크기)"] = (2 * p_up - 1) * np.clip(mag, 0, None)

    nv, dv, av = net[va], days[va], atrbp[va]
    res = {}
    for pname, pred in preds.items():
        log(f"\n=== {pname} ===")
        print(f"{'신호':>24s}{'k':>5s}{'독립일':>7s}{'전체bp':>9s}{'상위bp':>9s}{'일t':>7s}"
              f"{'볼타깃bp':>10s}{'볼t':>7s}")
        print("-" * 79)
        per, npass, npass_v = {}, 0, 0
        for s_ in SIGNALS:
            mk = (Pva["signal"] == s_).to_numpy()
            if mk.sum() < 80:
                continue
            k = max(10, int(round(mk.sum() * TOPQ)))
            sp, sn, sd, sa = pred[mk], nv[mk], dv[mk], av[mk]
            top = np.argsort(-sp)[:k]
            tb, tt = float(sn[top].mean()), float(cluster_t(sn[top], sd[top]))
            # ⭐변동성 타깃 사이징: 비중 ∝ 1/atr (일정 리스크). 평균 비중 1로 정규화
            w = (1.0 / sa[top]); w = w / w.mean()
            vb, vt = float((sn[top] * w).mean()), float(cluster_t(sn[top] * w, sd[top]))
            npass += tt > 1.96; npass_v += vt > 1.96
            print(f"{s_[:23]:>24s}{k:5d}{len(np.unique(sd[top])):7d}{float(sn.mean()):9.2f}"
                  f"{tb:9.2f}{tt:7.2f}{vb:10.2f}{vt:7.2f}"
                  f"{'  ⭐' if max(tt, vt) > 1.96 else ''}")
            per[s_] = {"top_mean_bp": tb, "cluster_t": tt, "voltgt_bp": vb, "voltgt_t": vt,
                       "independent_days": int(len(np.unique(sd[top])))}
        kk = max(10, int(round(va.sum() * TOPQ)))
        tp = np.argsort(-pred)[:kk]
        ptb, ptt = float(nv[tp].mean()), float(cluster_t(nv[tp], dv[tp]))
        log(f"  풀 상위10%: {ptb:+.2f}bp t={ptt:.2f} · ⭐신호별 통과 **{npass}/8** "
            f"(볼타깃 {npass_v}/8)")
        res[pname] = {"per_signal": per, "pool_top_bp": ptb, "pool_top_t": ptt,
                      "n_passed": npass, "n_passed_voltgt": npass_v}

    best = max(res.items(), key=lambda x: max(x[1]["n_passed"], x[1]["n_passed_voltgt"]))
    nflip = sum(1 for v in flip_rep.values() if v["flip_cluster_t"] > 1.96)
    log(f"\n⭐최고: {best[0]} -> {best[1]['n_passed']}/8 (볼타깃 {best[1]['n_passed_voltgt']}/8)")
    log(f"⭐방향 반전이 유의한 신호: {nflip}/8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"predictors": res, "direction_flip": flip_rep,
                               "n_pool": int(len(P)), "oos_touched": False,
                               "runtime_sec": round(time.time() - t0, 1)},
                              ensure_ascii=False, indent=2))
    log(f"산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
