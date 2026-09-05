#!/usr/bin/env python3
"""축 14 -- **지속 방향 × 회귀 모델 × 신호별 평가** (2026-09-05).

## 왜 이게 새로운가

축 1~13은 전부 **페이드 방향**(bottom→롱, top→숏)을 못 박고 라벨·모델·피쳐만 바꿨다.
5.23이 그 전제가 틀렸음을 보였다 -- 발동 봉은 **지속** 시점이다(TRAIN 페이드 −3.0 vs
지속 +4.7bp, 8종·양측면·세 창 동일). 방향을 고치자 **모델 없는 자유도 0 규칙**이
VAL +4.44 [0.7, 8.1] · OOS +6.78 [2.6, 11.4]로 살아났다.

⇒ 아직 아무도 안 해본 조합: **지속 방향 위에서 모델이 신호별로 선별할 수 있는가.**
페이드 축의 14번째 변형이 아니다 -- 모집단의 부호가 반대인 **다른 모집단**에서의 첫 시험이고,
기저가 양수라 모델이 개선할 여지가 구조적으로 존재한다(페이드에선 음수 영역에서 "덜 잃기"
뿐이었다).

## 규격 -- 5.23과 **비트 단위로 같게** 맞춘다(비교 가능성)

    모집단  8종 raw 첫발동(GAP12)
    방향    ⭐**신호 반대**(지속)      ← 축 1~13과 유일하게 다른 점
    진입    다음 봉 시가
    청산    sim_exit(5.0/1.5/0.1), 200봉
    비용    10bp

## 판정

  기준선  평평한 지속 규칙(모델 없음)의 신호별 평균 -- 5.23이 낸 값
  모델    상위10% 선별 평균 + **일 단위 군집 CI**
  통과    신호별 CI 하한 > 0  **그리고** 평평한 규칙보다 개선

⚠️TRAIN/VAL만. OOS·HOLDOUT 미터치(5.23의 OOS는 이미 공개된 값이라 인용만).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# 로컬 실행용 스텁(서버 다운). 접근 이름을 기록해 사후 검증한다 -- 축 13에서 검증된 방식.
_STUB_USED: list[str] = []
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    class _Any:
        def __init__(self, p="torch"): self._p = p
        def __getattr__(self, n): _STUB_USED.append(f"{self._p}.{n}"); return _Any(f"{self._p}.{n}")
        def __call__(self, *a, **k): _STUB_USED.append(f"{self._p}()"); return _Any(self._p)
        def __enter__(self): return self
        def __exit__(self, *a): return False
    _t = types.ModuleType("torch"); _t.__getattr__ = lambda n: _Any(f"torch.{n}")
    _t.Tensor = type("Tensor", (), {})
    _nn = types.ModuleType("torch.nn"); _nn.__getattr__ = lambda n: _Any(f"torch.nn.{n}")
    _nn.Module = type("Module", (), {})
    _u = types.ModuleType("torch.utils"); _d = types.ModuleType("torch.utils.data")
    _d.__getattr__ = lambda n: _Any(f"torch.utils.data.{n}"); _u.data = _d
    _t.nn, _t.utils = _nn, _u
    for _n, _m in (("torch", _t), ("torch.nn", _nn), ("torch.utils", _u), ("torch.utils.data", _d)):
        sys.modules[_n] = _m
    try:
        import catboost  # noqa: F401
    except ModuleNotFoundError:
        _cb = types.ModuleType("catboost"); _cb.__getattr__ = lambda n: _Any(f"catboost.{n}")
        for _nm in ("CatBoostClassifier", "CatBoostRegressor", "Pool"):
            setattr(_cb, _nm, _Any(f"catboost.{_nm}"))
        sys.modules["catboost"] = _cb


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


_pf = _load("pf_cont", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0, sim_exit = _pf.TIER0, _pf.sim_exit

SIGNALS = ["liquidity_sweep", "taker_delta_z_climax", "short_term_return_z", "orthogonal_combo",
           "smt_divergence", "fib_extension_exhaustion", "demarker_extreme",
           "kalman_deviation_meanrev"]
CELL, HZ = (5.0, 1.5, 0.1), 200          # ⭐5.23 규격
GAP, COST_BP, TOPQ, SEED = 12, 10.0, 0.10, 20260905
OUT = ROOT / "data/research/eth_evidence8_continuation_model_20260905/report.json"


def log(m): print(f"[cont] {m}", flush=True)


def causal_first_fire(fire, gap):
    keep = np.zeros(len(fire), bool); last = -10**9
    for i in np.flatnonzero(fire):
        if i - last > gap:
            keep[i] = True
        last = i
    return keep


def cluster_t(v, d):
    if len(v) < 5: return np.nan
    dev = v - v.mean()
    se = np.sqrt(sum(dev[d == x].sum() ** 2 for x in np.unique(d))) / len(v)
    return v.mean() / se if se > 0 else np.nan


def day_ci(v, d, rng, B=2000):
    u = np.unique(d); idx = {x: np.flatnonzero(d == x) for x in u}
    bs = [v[np.concatenate([idx[x] for x in rng.choice(u, len(u), True)])].mean() for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


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
    lp = np.array([pos_of.get(np.datetime64(t), -1) for t in LONG["_ts"].to_numpy()])
    isd = LONG["is_downside"].to_numpy().astype(bool)
    T0 = [x for x in TIER0 if x in LONG.columns]

    parts = []
    for S_ in SIGNALS:
        bc, tc = f"bottom_{S_}", f"top_{S_}"
        SS = sig[["timestamp", bc, tc]].copy()
        if SS["timestamp"].dt.tz is not None:
            SS["timestamp"] = SS["timestamp"].dt.tz_localize(None)
        SS["pos"] = [pos_of.get(np.datetime64(t), -1) for t in SS["timestamp"].to_numpy()]
        SS = SS.loc[SS["pos"] >= 0]
        fb = np.zeros(n, bool); ft = np.zeros(n, bool)
        fb[SS["pos"].to_numpy()] = SS[bc].fillna(False).to_numpy(bool)
        ft[SS["pos"].to_numpy()] = SS[tc].fillna(False).to_numpy(bool)
        kb, kt = causal_first_fire(fb, GAP), causal_first_fire(ft, GAP)
        keep = (lp >= 0) & (lp + 1 + HZ < n)
        keep &= np.where(isd, kb[np.clip(lp, 0, n - 1)], kt[np.clip(lp, 0, n - 1)])
        if keep.sum() < 200:
            continue
        cols = list(dict.fromkeys(["_ts", "split", "is_downside", "atr"] + T0))
        D = LONG.loc[keep, cols].reset_index(drop=True)
        D["pos"] = lp[keep]; D["signal"] = S_
        parts.append(D)
    P = pd.concat(parts, ignore_index=True)
    ii = P["pos"].to_numpy().astype(int)
    # ⭐지속 = 신호 **반대** 방향. (bottom fire → 숏, top fire → 롱)
    sg_fade = np.where(P["is_downside"].to_numpy() == 1, 1.0, -1.0)
    sg = -sg_fade
    entry = o[ii + 1]
    H = np.stack([h[i + 1:i + 1 + HZ] for i in ii])
    L = np.stack([l[i + 1:i + 1 + HZ] for i in ii])
    C = np.stack([c[i + 1:i + 1 + HZ] for i in ii])
    pn, _ = sim_exit(entry, P["atr"].to_numpy(float), sg, H, L, C, *CELL)
    P["net_bp"] = pn * 1e4 - COST_BP
    pnf, _ = sim_exit(entry, P["atr"].to_numpy(float), sg_fade, H, L, C, *CELL)
    P["net_fade_bp"] = pnf * 1e4 - COST_BP
    split = P["split"].to_numpy(); tr, va = split == "TRAIN", split == "VAL"
    days = pd.to_datetime(P["_ts"]).dt.floor("D").to_numpy()
    log(f"⭐풀 {len(P):,}건 (TRAIN {tr.sum():,} / VAL {va.sum():,})")
    log(f"  ⭐지속 평균 TRAIN {P.loc[tr,'net_bp'].mean():+.2f} / VAL {P.loc[va,'net_bp'].mean():+.2f}bp"
        f"   (페이드 {P.loc[tr,'net_fade_bp'].mean():+.2f} / {P.loc[va,'net_fade_bp'].mean():+.2f})")
    log("  ⇒ 5.23 재현 확인용: TRAIN 지속 +4.87 · 페이드 −2.98 과 비교")

    # ⭐목표가 묻는 "8종 모두 통과"의 직접 답 -- **평평한 지속 규칙의 신호별 일CI**.
    # 5.23은 풀 단위 CI만 냈다. 모델 없이 자유도 0인 이 규칙이 신호별로 몇 종이나
    # CI 하한 > 0인지가 관문 통과 여부다.
    net = P["net_bp"].to_numpy(float)     # ⚠️아래 모델 블록보다 먼저 필요
    rng0 = np.random.default_rng(SEED)
    days_all = days
    log("\n=== ⭐평평한 지속 규칙(모델 없음)의 신호별 일CI ===")
    print(f"{'신호':>24s}{'VAL n':>8s}{'독립일':>7s}{'평균bp':>9s}{'일CI하한':>10s}"
          f"{'일CI상한':>10s}{'일t':>7s}")
    print("-" * 76)
    flat_rep, flat_pass = {}, 0
    Pva0 = P.loc[va].reset_index(drop=True)
    nv0, dv0 = net[va], days_all[va]
    for s_ in SIGNALS:
        mk = (Pva0["signal"] == s_).to_numpy()
        if mk.sum() < 80: continue
        v, d = nv0[mk], dv0[mk]
        lo, hi = day_ci(v, d, rng0)
        tt = float(cluster_t(v, d))
        ok = lo > 0
        flat_pass += ok
        print(f"{s_[:23]:>24s}{int(mk.sum()):8d}{len(np.unique(d)):7d}{float(v.mean()):9.2f}"
              f"{lo:10.2f}{hi:10.2f}{tt:7.2f}{'  ⭐' if ok else ''}")
        flat_rep[s_] = {"n_val": int(mk.sum()), "mean_bp": float(v.mean()),
                        "day_ci_lo": lo, "day_ci_hi": hi, "cluster_t": tt,
                        "independent_days": int(len(np.unique(d))), "passed": bool(ok)}
    lo_p, hi_p = day_ci(nv0, dv0, rng0)
    log(f"  풀 전체 {float(nv0.mean()):+.2f}bp [일CI {lo_p:+.2f}, {hi_p:+.2f}] "
        f"(독립일 {len(np.unique(dv0))})")
    log(f"  ⭐⭐평평한 규칙 신호별 통과(CI하한>0): **{flat_pass}/{len(flat_rep)}**")

    from sklearn.ensemble import HistGradientBoostingRegressor
    try:
        from tabpfn import TabPFNRegressor
        MODELS = ("HGB_reg", "TabPFN_reg")
    except Exception:
        TabPFNRegressor = None
        MODELS = ("HGB_reg",)
        log("  ⚠️TabPFN 없음(서버 다운, 로컬 CPU) -- HGB로 진행. 축 12에서 두 모델 동등 확인됨")

    X = np.nan_to_num(P[T0].apply(pd.to_numeric, errors="coerce").to_numpy(float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.random.default_rng(SEED)
    res = {}
    for mn in MODELS:
        if mn == "HGB_reg":
            m = HistGradientBoostingRegressor(random_state=SEED, max_iter=400, learning_rate=0.05)
            m.fit(X[tr], net[tr])
        else:
            tri = np.flatnonzero(tr)
            ctx = rng.choice(tri, size=min(18000, len(tri)), replace=False)
            m = TabPFNRegressor(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
            m.fit(X[ctx], net[ctx])
        pred = m.predict(X[va])
        Pva = P.loc[va].reset_index(drop=True)
        nv, dv = net[va], days[va]
        log(f"\n=== {mn} ===")
        print(f"{'신호':>24s}{'VAL n':>8s}{'평평한규칙':>11s}{'모델상위10%':>12s}{'개선':>8s}"
              f"{'일CI하한':>10s}{'일t':>7s}")
        print("-" * 82)
        per, npass = {}, 0
        for s_ in SIGNALS:
            mk = (Pva["signal"] == s_).to_numpy()
            if mk.sum() < 80: continue
            k = max(10, int(round(mk.sum() * TOPQ)))
            sp, sn, sd = pred[mk], nv[mk], dv[mk]
            flat = float(sn.mean())
            top = np.argsort(-sp)[:k]
            tb, tt = float(sn[top].mean()), float(cluster_t(sn[top], sd[top]))
            lo, hi = day_ci(sn[top], sd[top], rng)
            ok = lo > 0 and tb > flat
            npass += ok
            print(f"{s_[:23]:>24s}{int(mk.sum()):8d}{flat:11.2f}{tb:12.2f}{tb-flat:8.2f}"
                  f"{lo:10.2f}{tt:7.2f}{'  ⭐' if ok else ''}")
            per[s_] = {"n_val": int(mk.sum()), "flat_rule_bp": flat, "model_top_bp": tb,
                       "improvement_bp": tb - flat, "day_ci_lo": lo, "day_ci_hi": hi,
                       "cluster_t": tt, "passed": bool(ok)}
        pk = max(10, int(round(va.sum() * TOPQ)))
        tp = np.argsort(-pred)[:pk]
        plo, phi = day_ci(nv[tp], dv[tp], rng)
        log(f"  풀 전체: 평평 {float(nv.mean()):+.2f} → 모델상위10% {float(nv[tp].mean()):+.2f}bp "
            f"[일CI {plo:+.2f}, {phi:+.2f}]")
        log(f"  ⭐신호별 통과(CI하한>0 AND 평평한 규칙 개선): **{npass}/{len(per)}**")
        res[mn] = {"per_signal": per, "n_passed": npass,
                   "pool_flat_bp": float(nv.mean()), "pool_model_bp": float(nv[tp].mean()),
                   "pool_ci": [plo, phi]}
    if _STUB_USED:
        log(f"\n⚠️스텁 접근 고유 {len(set(_STUB_USED))}종: {sorted(set(_STUB_USED))[:6]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"flat_rule_per_signal": flat_rep,
                           "flat_rule_n_passed": flat_pass, "models": res, "cell": list(CELL), "horizon": HZ,
                               "direction": "continuation(opposite signal side)",
                               "n_pool": int(len(P)), "oos_touched": False,
                               "runtime_sec": round(time.time() - t0, 1)},
                              ensure_ascii=False, indent=2))
    log(f"산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
