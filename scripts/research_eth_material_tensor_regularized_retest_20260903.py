#!/usr/bin/env python3
"""재료 텐서 정규화 재시험 -- "학습이 값어치 있는가" (2026-09-03).

선행: `research_eth_material_tensor_resolved_variant_test_20260903.py`에서 저장소 표준 HGB
설정이 TRAIN IC 0.59 → VAL/HOLDOUT **음수**로 무너졌고, **모델 없는 원시 signed 합**
(VAL +0.0412 / OOS +0.0273 / HOLDOUT +0.0658)이 학습 모델을 3/4 창에서 이겼다.

⭐진단이 먼저다(메모리 `feedback_modern_dl_training_checklist`의 "diagnostic habit"):
   실패 양상 = **암기**(TRAIN만 높고 표본외 음수). 그러면 맞는 기법은 두 가지다 --
   ①용량 축소/정규화 ②정직한 검증 분할. 다른 기법을 기계적으로 얹지 않는다.

고치는 것:
  ① **시계열 분할 early stopping** -- sklearn HGB의 `validation_fraction`은 **무작위** 분할이라
     시계열에서 낙관적이다. `warm_start`로 한 iter씩 키우며 TRAIN의 **마지막 15%**(시간순)에서
     조기중단한다. 그 과정의 train/val 곡선을 **iter마다 로그**한다
     (메모리 `feedback_always_log_and_monitor_epoch_metrics`, 부스팅판 에폭곡선).
  ② **용량/정규화 격자** -- leaf {4,8,15,31} × min_samples_leaf {200,2000} × l2 {1,50}
  ③ **저용량 선형 대조(Ridge)** -- 선형이 HGB를 이기면 용량이 문제라는 직접 증거다.

⭐사전등록(데이터 보기 전 고정): 어떤 변형이든 **원시 합을 VAL·OOS 양 창에서 이겨야**
"학습이 값어치 있다"가 성립한다. 미달이면 재료의 사용 형태는 원시 합이다.
⚠️격자 승자는 2시드 스크린 → **5시드 확정**으로만 인정한다
(메모리 `tabm_hp_low_signal_pattern`: 단일시드 HP 승자는 대개 노이즈).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, OOS_START, VAL_START)

SRC = ROOT / "tmp/eth_causal_population_metalabel_20260902"
ETH_REGIME = ROOT / "tmp/eth_regime_s12k3_clean_20260902/predictions.parquet"
BTC_REGIME = ROOT / "tmp/btc_regime_s24k3_clean_20260902/predictions.parquet"
OUT = ROOT / "tmp/eth_material_regularized_20260903"
FWD = 24
INNER = 0.85                     # TRAIN의 앞 85%로 학습, 뒤 15%(시간순)로 조기중단
STEP, MAX_ITER, PATIENCE = 20, 600, 5
SEEDS5 = [76010, 130820, 194636, 331076, 703883]


def log(m): print(f"[reg] {m}", flush=True)


def build_tensor():
    from research_eth_kalman_demarker_gridscreen_20260831 import load_klines
    cfg = json.loads((SRC / "config.json").read_text())["cfg"]
    kl = load_klines()[["timestamp", "close"]].sort_values("timestamp").reset_index(drop=True)
    n = len(kl); ts = pd.DatetimeIndex(kl["timestamp"])
    pos = {t: i for i, t in enumerate(ts)}
    c = kl["close"].to_numpy(float)
    out = pd.DataFrame({"timestamp": kl["timestamp"]})
    cols, S = [], np.zeros(n)
    for name, cc in cfg.items():
        H = int(cc["horizon"])
        f = SRC / f"{name}_causal_proba_cal.csv"
        d = pd.read_csv(f if f.exists() else SRC / f"{name}_causal_proba.csv",
                        parse_dates=["timestamp"])
        if "proba_cal" not in d.columns:
            d["proba_cal"] = d["proba"]
        trp = np.sort(d.loc[d.split == "TRAIN", "proba"].to_numpy(float))
        d["pct"] = np.searchsorted(trp, d["proba"].to_numpy(float), side="right") / max(len(trp), 1)
        d = d[d.timestamp.isin(pos)].copy(); d["i"] = [pos[t] for t in d.timestamp]
        d["dir"] = np.where(d.is_bottom == 1, 1.0, -1.0)
        d = d.sort_values(["i", "proba"]).drop_duplicates("i", keep="last")
        fire = np.zeros(n); fire[d["i"].to_numpy()] = d["dir"].to_numpy()
        pi = {int(i): (float(a), float(b), float(x), float(y)) for i, a, b, x, y in
              zip(d["i"], d["proba"], d["proba_cal"], d["pct"], d["dir"])}
        P = np.zeros((n, 4)); age = np.ones(n)
        last, li = (0.0, 0.0, 0.0, 0.0), -10**9
        for i in range(n):
            if i in pi: last, li = pi[i], i
            el = i - li
            if el < H:
                P[i] = last; age[i] = el / H
        sgn = P[:, 2] * P[:, 3]
        S += sgn
        for tag, arr in (("fire", fire), ("proba", P[:, 0]), ("proba_cal", P[:, 1]),
                         ("pct", P[:, 2]), ("signed", sgn), ("age", age)):
            out[f"{name}_{tag}"] = arr; cols.append(f"{name}_{tag}")
    for tag, p in (("eth", ETH_REGIME), ("btc", BTC_REGIME)):
        if p.exists():
            r = pd.read_parquet(p)
            out = out.merge(r.rename(columns={"regime": f"regime_{tag}"}), on="timestamp", how="left")
            out[f"regime_{tag}"] = out[f"regime_{tag}"].ffill().fillna(-1).astype(int)
            cols.append(f"regime_{tag}")
    y = np.concatenate([(c[FWD:] - c[:-FWD]) / c[:-FWD], np.full(FWD, np.nan)])
    split = np.where(ts < VAL_START, "TRAIN", np.where(ts < OOS_START, "VAL",
                     np.where(ts < HOLDOUT_START, "OOS", "HOLDOUT")))
    return out, cols, y, split, S


def ic(p, y, m):
    return float(spearmanr(p[m], y[m])[0]) if m.sum() > 200 else np.nan


def fit_ts_early(X, y, itr, ival, seed, hp, verbose=False):
    """⭐시계열 조기중단: warm_start로 STEP씩 키우며 뒤 15%(시간순) IC가 꺾이면 멈춘다."""
    m = HistGradientBoostingRegressor(random_state=seed, loss="squared_error",
                                      early_stopping=False, warm_start=True,
                                      max_iter=STEP, **hp)
    best, best_it, bad = -np.inf, STEP, 0
    curve = []
    for it in range(STEP, MAX_ITER + 1, STEP):
        m.set_params(max_iter=it)
        m.fit(X[itr], y[itr])
        pi_ = m.predict(X[itr]); pv = m.predict(X[ival])
        tr_ic = float(spearmanr(pi_, y[itr])[0]); va_ic = float(spearmanr(pv, y[ival])[0])
        curve.append((it, tr_ic, va_ic))
        if verbose:
            log(f"    iter={it:4d} train_IC={tr_ic:+.4f} inner_val_IC={va_ic:+.4f}"
                f"{'  ← best' if va_ic > best else ''}")
        if va_ic > best:
            best, best_it, bad = va_ic, it, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    final = HistGradientBoostingRegressor(random_state=seed, loss="squared_error",
                                          early_stopping=False, max_iter=best_it, **hp)
    final.fit(X[itr | ival], y[itr | ival])
    return final, best_it, best, curve


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    D, cols, y, split, S = build_tensor()
    ok = np.isfinite(y)
    tr_all = (split == "TRAIN") & ok
    idx = np.flatnonzero(tr_all)
    cut = idx[int(len(idx) * INNER)]
    itr = tr_all & (np.arange(len(y)) < cut)
    ival = tr_all & (np.arange(len(y)) >= cut)
    log(f"봉 {len(y):,} · TRAIN {int(tr_all.sum()):,} (내부학습 {int(itr.sum()):,} / "
        f"내부검증 {int(ival.sum()):,}, 시간순) · 피쳐 {len(cols)}")

    W = ("VAL", "OOS", "HOLDOUT")
    base = {w: ic(S, y, (split == w) & ok) for w in W}
    log(f"⭐기준선(원시 signed 합, 모델 없음)  " + " ".join(f"{w} {base[w]:+.4f}" for w in W))

    X = D[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X[tr_all].median()).to_numpy(float)

    # --- Ridge 저용량 대조 ---
    log("\n=== 저용량 선형 대조 (Ridge) ===")
    sc = StandardScaler().fit(X[tr_all])
    Xs = sc.transform(X)
    for a in (1.0, 100.0, 10000.0):
        r = Ridge(alpha=a).fit(Xs[tr_all], y[tr_all])
        p = r.predict(Xs)
        log(f"  alpha={a:>7.0f}  " + " ".join(f"{w} {ic(p,y,(split==w)&ok):+.4f}" for w in W))

    # --- 정규화 격자 (2시드 스크린) ---
    GRID = [{"max_leaf_nodes": L, "min_samples_leaf": ms, "l2_regularization": l2,
             "learning_rate": 0.03}
            for L in (4, 8, 15, 31) for ms in (200, 2000) for l2 in (1.0, 50.0)]
    log(f"\n=== 정규화 격자 2시드 스크린 ({len(GRID)}조합) ===")
    print(f"{'leaf':>5s}{'minleaf':>9s}{'l2':>7s}{'iter':>6s}{'innerVAL':>10s}"
          + "".join(f"{w:>10s}" for w in W))
    rows = []
    for hp in GRID:
        ps, its, ivs = [], [], []
        for sd in SEEDS5[:2]:
            mdl, bit, biv, _ = fit_ts_early(X, y, itr, ival, sd, hp)
            ps.append(mdl.predict(X)); its.append(bit); ivs.append(biv)
        p = np.mean(ps, axis=0)
        r = {**hp, "iter": int(np.mean(its)), "innerVAL": float(np.mean(ivs)),
             **{w: ic(p, y, (split == w) & ok) for w in W}}
        rows.append(r)
        print(f"{hp['max_leaf_nodes']:5d}{hp['min_samples_leaf']:9d}{hp['l2_regularization']:7.0f}"
              f"{r['iter']:6d}{r['innerVAL']:+10.4f}" + "".join(f"{r[w]:+10.4f}" for w in W))
    R = pd.DataFrame(rows); R.to_csv(OUT / "grid.csv", index=False)

    # ⭐선택은 표본외를 보지 않고 **내부검증**으로만 한다
    best = R.sort_values("innerVAL", ascending=False).iloc[0]
    hp = {k: best[k] for k in ("max_leaf_nodes", "min_samples_leaf", "l2_regularization",
                               "learning_rate")}
    hp["max_leaf_nodes"] = int(hp["max_leaf_nodes"]); hp["min_samples_leaf"] = int(hp["min_samples_leaf"])
    log(f"\n⭐내부검증 최선: leaf={hp['max_leaf_nodes']} minleaf={hp['min_samples_leaf']} "
        f"l2={hp['l2_regularization']:.0f} (innerVAL {best['innerVAL']:+.4f})")

    log("\n=== 5시드 확정 (곡선 로그 포함, 첫 시드만 상세) ===")
    ps = []
    for k, sd in enumerate(SEEDS5):
        mdl, bit, biv, curve = fit_ts_early(X, y, itr, ival, sd, hp, verbose=(k == 0))
        ps.append(mdl.predict(X))
        log(f"  seed={sd} 중단 iter={bit} innerVAL={biv:+.4f}")
    p = np.mean(ps, axis=0)
    fin = {w: ic(p, y, (split == w) & ok) for w in W}
    print(f"\n{'':22s}" + "".join(f"{w:>10s}" for w in W))
    print(f"{'원시 합(모델 없음)':22s}" + "".join(f"{base[w]:+10.4f}" for w in W))
    print(f"{'정규화 HGB 5시드':22s}" + "".join(f"{fin[w]:+10.4f}" for w in W))
    win = all(fin[w] > base[w] for w in ("VAL", "OOS"))
    log(f"\n⭐사전등록 판정: 원시 합을 VAL·OOS 양 창에서 이김 -- {'✅' if win else '❌'}")
    log(f"  → {'**학습이 값어치 있다**' if win else '**재료의 사용 형태는 원시 합이다**'}")
    json.dump({"baseline": base, "final": fin, "hp": hp, "win": bool(win)},
              open(OUT / "result.json", "w"), ensure_ascii=False, indent=2, default=str)
    log(f"\n{time.time()-t0:.0f}초 · 산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
