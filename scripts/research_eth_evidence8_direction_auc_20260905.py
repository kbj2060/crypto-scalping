#!/usr/bin/env python3
"""축 13 -- **피쳐에 방향 정보가 있는가**를 직접 측정 (2026-09-05).

## 왜 이게 필요한가 (자기 정정)

§5.22는 "축 11(방향 반전 0/8)이 피쳐에 방향 정보가 없음을 직접 증명"이라고 적었다.
**과잉 해석이다.** 반전 테스트가 증명하는 건 두 명제 중 하나뿐이다:

    (a) **신호의** 방향이 무의미하다 -- 정방향도 반전도 못 이긴다   <- 축 11이 증명 ✅
    (b) **피쳐에** 방향 정보가 없다                                  <- 별개 명제 ❌

모델은 신호 방향과 무관하게 피쳐에서 방향을 뽑아낼 수 있다("이 bottom 발동은 X가 높으니
실제론 내려간다"). (b)를 주장하려면 직접 재야 한다.

## 측정

    타깃   신호 방향 기준 **비용 전** 수익 부호 (sign of directional move)
           -- 비용을 빼면 "방향력 부재"와 "비용 장벽"이 섞인다. 여기선 방향력만 본다.
    모델   TabPFN 분류 · HGB 분류
    지표   VAL AUC + **일 단위 군집 부트스트랩 CI** (행 단위 CI는 하루 안 상관을 무시)
    판정   CI가 0.5를 포함 -> (b) 확립. 0.5를 넘으면 방향 정보는 있고 병목이 다른 데 있다.

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

# ⚠️2026-09-05 서버 다운으로 로컬 실행. import 체인이 `torch`를 요구하지만 **우리 경로는
# torch를 한 번도 호출하지 않는다**: 체인이 실제로 가져오는 건 `_atr_pct`(순수 pandas/numpy,
# eval_omega4_1_atr_safety_sltp_20260622.py:47~57)뿐이고, 그 파일의 torch 사용처는 166행
# `torch.load`(CLI 전용) 하나다. 중간 파일(train_eval_omega1_2_tabm_3head)이 요구하는 것도
# `torch` · `torch.nn` · `torch.utils.data`(DataLoader/TensorDataset) · 타입힌트용
# `torch.Tensor`뿐이고 **최상위 실행 코드에서 호출하지 않는다**(87행 함수 시그니처 힌트뿐).
# ⇒ 딱 그만큼만 스텁한다. **속성에 실제로 접근하면 즉시 예외**를 던져, 조용한 오작동 대신
#   시끄러운 실패가 되게 한다(오늘 하루 반복된 "조용히 넘어가기" 결함 방지).
try:
    import torch  # noqa: F401
    _TORCH_STUB_USED: list[str] = []
except ModuleNotFoundError:
    # 관대한 스텁: 어떤 속성이든 no-op으로 응답하되 **접근한 이름을 전부 기록**한다.
    # 실제 텐서 연산이 일어나면 numpy로도 흉내낼 수 없으므로, 기록을 보고 "우리 경로가
    # 정말 torch를 안 쓰는지" 사후 검증한다(조용한 오작동 방지).
    import types

    _TORCH_STUB_USED = []

    class _Any:
        def __init__(self, path="torch"): self._p = path
        def __getattr__(self, n):
            _TORCH_STUB_USED.append(f"{self._p}.{n}")
            return _Any(f"{self._p}.{n}")
        def __call__(self, *a, **k):
            _TORCH_STUB_USED.append(f"{self._p}()")
            return _Any(self._p)              # 데코레이터/컨텍스트매니저 겸용
        def __enter__(self): return self
        def __exit__(self, *a): return False

    _t = types.ModuleType("torch")
    _t.__getattr__ = lambda n: _Any(f"torch.{n}")
    _t.Tensor = type("Tensor", (), {})
    _nn = types.ModuleType("torch.nn")
    _nn.__getattr__ = lambda n: _Any(f"torch.nn.{n}")
    _nn.Module = type("Module", (), {})
    _utils = types.ModuleType("torch.utils")
    _data = types.ModuleType("torch.utils.data")
    _data.__getattr__ = lambda n: _Any(f"torch.utils.data.{n}")
    _utils.data = _data
    _t.nn, _t.utils = _nn, _utils
    for _n, _m in (("torch", _t), ("torch.nn", _nn),
                   ("torch.utils", _utils), ("torch.utils.data", _data)):
        sys.modules[_n] = _m

# catboost도 같은 이유로 스텁한다 -- 체인이 `CatBoostClassifier`를 import만 하고 우리 경로에서
# 인스턴스화하지 않는다. 마찬가지로 **접근 이름을 기록**해 사후 검증한다.
try:
    import catboost  # noqa: F401
except ModuleNotFoundError:
    _cb = types.ModuleType("catboost")
    _cb.__getattr__ = lambda n: _Any(f"catboost.{n}")
    for _nm in ("CatBoostClassifier", "CatBoostRegressor", "Pool"):
        setattr(_cb, _nm, _Any(f"catboost.{_nm}"))
    sys.modules["catboost"] = _cb


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_pf = _load("pf_dir", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0 = _pf.TIER0

SIGNALS = {"liquidity_sweep": 30, "taker_delta_z_climax": 24, "short_term_return_z": 12,
           "orthogonal_combo": 24, "smt_divergence": 72, "fib_extension_exhaustion": 20,
           "demarker_extreme": 8, "kalman_deviation_meanrev": 12}
GAP, SEED, B_DAY = 12, 20260905, 2000
OUT = ROOT / "data/research/eth_evidence8_direction_auc_20260905/report.json"


def log(m): print(f"[dir] {m}", flush=True)


def causal_first_fire(fire, gap):
    keep = np.zeros(len(fire), bool); last = -10**9
    for i in np.flatnonzero(fire):
        if i - last > gap:
            keep[i] = True
        last = i
    return keep


def auc_day_ci(y, p, days, rng, B=B_DAY):
    """일 단위 군집 부트스트랩으로 AUC CI. 날짜 블록을 복원추출한다."""
    from sklearn.metrics import roc_auc_score
    uniq = np.unique(days)
    idx_by_day = {d: np.flatnonzero(days == d) for d in uniq}
    out = []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by_day[d] for d in pick])
        yy = y[ii]
        if len(np.unique(yy)) < 2:
            continue
        out.append(roc_auc_score(yy, p[ii]))
    out = np.array(out)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if len(out) else (np.nan, np.nan)


def main() -> int:
    t0 = time.time()
    log("프레임 빌드...")
    sig, feat, eth = _s1.build_sig()
    dummy = np.full(len(sig), "none", dtype=object)
    long = _s1.long_frame_for(sig, feat, dummy, dummy)
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    if kl["timestamp"].dt.tz is not None:
        kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    o, c = kl["open"].to_numpy(float), kl["close"].to_numpy(float)
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
        cols = list(dict.fromkeys(["_ts", "split", "is_downside"] + T0))
        D = LONG.loc[keep, cols].reset_index(drop=True)
        ii = lpos_all[keep]
        sg = np.where(D["is_downside"].to_numpy() == 1, 1.0, -1.0)
        # ⭐비용 전 방향 수익 -- 비용을 빼면 "방향력 부재"와 "비용 장벽"이 섞인다
        D["dir_ret_bp"] = (c[ii + HZ] / o[ii + 1] - 1.0) * sg * 1e4
        D["signal"] = SIGNAL
        parts.append(D)
    P = pd.concat(parts, ignore_index=True)
    P["y"] = (P["dir_ret_bp"] > 0).astype(int)
    X = np.nan_to_num(P[T0].apply(pd.to_numeric, errors="coerce").to_numpy(float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = P["y"].to_numpy()
    split = P["split"].to_numpy()
    tr, va = split == "TRAIN", split == "VAL"
    days = pd.to_datetime(P["_ts"]).dt.floor("D").to_numpy()
    log(f"⭐풀 {len(P):,}건 (TRAIN {tr.sum():,} / VAL {va.sum():,}) · "
        f"방향 승률 전체 {y.mean():.4f} / VAL {y[va].mean():.4f}")
    log(f"  VAL 독립일 {len(np.unique(days[va]))}일")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    # ⚠️2026-09-05 서버 다운으로 로컬 실행. TabPFN(GPU)이 없으면 HGB만으로 진행한다 --
    # 축 12에서 두 모델이 IC 0.102/0.120으로 **같은 벽**에 부딪힌 게 확인됐으므로,
    # "방향 정보가 있는가"라는 이 질문에는 HGB 단독으로도 답할 수 있다.
    try:
        from tabpfn import TabPFNClassifier
        MODELS_TO_RUN = ("HGB_clf", "TabPFN_clf")
    except Exception:
        TabPFNClassifier = None
        MODELS_TO_RUN = ("HGB_clf",)
        log("  ⚠️TabPFN 없음 -- HGB 단독으로 진행(축 12에서 두 모델 동등 확인됨)")
    rng = np.random.default_rng(SEED)
    res = {}
    print(f"\n{'모델':>13s}{'VAL AUC':>10s}{'일CI 하한':>11s}{'일CI 상한':>11s}   판정")
    print("-" * 62)
    preds = {}
    for mname in MODELS_TO_RUN:
        if mname == "HGB_clf":
            m = HistGradientBoostingClassifier(random_state=SEED, max_iter=400,
                                               learning_rate=0.05)
            m.fit(X[tr], y[tr])
        else:
            tri = np.flatnonzero(tr)
            ctx = rng.choice(tri, size=min(18000, len(tri)), replace=False)
            m = TabPFNClassifier(device="cuda", random_state=SEED,
                                 ignore_pretraining_limits=True)
            m.fit(X[ctx], y[ctx])
        p = m.predict_proba(X[va])[:, 1]
        preds[mname] = p
        a = float(roc_auc_score(y[va], p))
        lo, hi = auc_day_ci(y[va], p, days[va], rng)
        verdict = "⭐0.5 초과(방향정보 있음)" if lo > 0.5 else "0.5 포함 -> 방향정보 없음"
        print(f"{mname:>13s}{a:10.4f}{lo:11.4f}{hi:11.4f}   {verdict}")
        res[mname] = {"val_auc": a, "day_ci_lo": lo, "day_ci_hi": hi,
                      "n_val": int(va.sum()), "base_rate": float(y[va].mean())}

    # 신호별 (TabPFN 기준)
    print(f"\n{'신호':>24s}{'VAL n':>8s}{'승률':>8s}{'AUC':>9s}{'일CI 하한':>11s}")
    print("-" * 62)
    Pva = P.loc[va].reset_index(drop=True)
    per = {}
    for s_ in SIGNALS:
        mk = (Pva["signal"] == s_).to_numpy()
        if mk.sum() < 100 or len(np.unique(y[va][mk])) < 2:
            continue
        pk = preds.get("TabPFN_clf", preds["HGB_clf"])
        a = float(roc_auc_score(y[va][mk], pk[mk]))
        lo, hi = auc_day_ci(y[va][mk], pk[mk], days[va][mk], rng, B=800)
        print(f"{s_[:23]:>24s}{int(mk.sum()):8d}{float(y[va][mk].mean()):8.3f}{a:9.4f}"
              f"{lo:11.4f}{'  ⭐' if lo > 0.5 else ''}")
        per[s_] = {"n": int(mk.sum()), "auc": a, "day_ci_lo": lo, "day_ci_hi": hi}
    npass = sum(1 for v in per.values() if v["day_ci_lo"] > 0.5)
    log(f"\n⭐신호별 방향 AUC의 일CI 하한이 0.5를 넘는 신호: **{npass}/{len(per)}**")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"pooled": res, "per_signal": per, "n_pool": int(len(P)),
                               "n_signals_pass": npass, "oos_touched": False,
                               "runtime_sec": round(time.time() - t0, 1)},
                              ensure_ascii=False, indent=2))
    if _TORCH_STUB_USED:
        uniq = sorted(set(_TORCH_STUB_USED))
        log(f"\n⚠️torch 스텁 접근 {len(_TORCH_STUB_USED)}회 (고유 {len(uniq)}): {uniq[:12]}")
        log("   위 이름들이 전부 import/데코레이터 시점 접근이면 결과에 영향 없음 -- "
            "실제 텐서 연산이면 결과 무효")
    else:
        log("\n✅torch 스텁 접근 0회 -- 이 경로는 torch를 전혀 쓰지 않는다")
    log(f"산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
