#!/usr/bin/env python3
"""페이드/지속 — **출구·비용과 무관한 순수 레짐 라벨** (2026-09-06).

사용자: *"수익률이 목적이 아니라 이후 페이드냐 지속이냐를 목적으로 해야 해."*

## 문제 인식 (사용자 지적)
지금까지 라벨이 전부 **출구 구조에 오염**돼 있었다:
  y_dec / margin  = `pnl_fade > pnl_cont`  -> 트레일링 5.0/1.5/0.1 + 비용 10bp 가 라벨을 결정
  y_order         = 2.0ATR vs 1.5ATR 터치 순서 -> 배리어 배수 선택에 의존
즉 "이후가 페이드였나 지속이었나"를 물은 게 아니라 "그 출구 규칙으로 어느 쪽이 벌었나"를 물었다.

## 여기서 바꾸는 것
**목적 = 이후 레짐 분류.** 라벨을 출구·비용·배리어 없이 **가격 경로만으로** 정의한다.
경제성은 목적이 아니라 **2차 확인**으로 분리한다(분류가 되면 돈이 되는지 따로 본다).

  R1 경로우세도(주)  r = MFE_cont / (MFE_cont + MFE_fade)  in [0,1]
                     MFE_* = 발동 봉 종가 대비 그 방향 최대 유리이탈(H봉). 배리어·출구 없음.
                     분류: r > 0.5 (지속 우세) · 회귀: r 자체
  R2 시간우세도      H봉 중 종가가 지속 쪽에 있던 봉 비율  (경로 점유, 크기 무관)
  R3 순수 부호       sign(지속방향 순수익 over H)          (끝점만 — 대조군)
  R4 진폭비 로그     log(MFE_cont / MFE_fade)              (R1의 무경계 변형)
H = 12 / 24 / 48 봉 (1h / 2h / 4h) — 레짐이지 거래결과가 아니므로 200봉보다 짧게.

## 평가 (두 층으로 분리)
  1차(목적)  라벨 자체의 AUC/ρ. 판정 대역은 실측 일군집 CI 반폭 **VAL ±0.021 · OOS ±0.026**
             (`eth_fade_cont_uniqueness_embargo_20260906.md`). 대역 밖 + 두 창 같은 부호여야 "분류 성공".
  2차(부수)  1차를 통과한 라벨에 한해, 그 예측으로 배포 셀 라우팅 -> cont_all 대비 일별 짝비교.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


M = _load("order_mod", "scripts/research_eth_fire_ordering_model_20260906.py")
DS = ROOT / "tmp/eth_fade_vs_cont_dataset_20260906/dataset.parquet"
OUT = ROOT / "tmp/eth_regime_outcome_label_20260906"
SEEDS = [11, 23, 47, 71, 97]
HORIZONS = [12, 24, 48]
BAND = {"VAL": 0.0214, "OOS": 0.0257}          # 실측 일군집 CI 반폭
NON = {"y_order", "y_dec", "margin", "pnl_fade", "pnl_cont", "cls3", "split", "timestamp", "pos"}
RNG = np.random.default_rng(20260906)


def log(m): print(f"[regime] {m}", flush=True)


def day_ci(v, d, B=1500):
    ud = np.unique(d); idx = {x: np.flatnonzero(d == x) for x in ud}
    o = np.empty(B)
    for b in range(B):
        p = RNG.choice(ud, len(ud), replace=True)
        o[b] = np.concatenate([v[idx[x]] for x in p]).mean()
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def auc_day_ci(y, p, days, B=1500):
    ud = np.unique(days); idx = {x: np.flatnonzero(days == x) for x in ud}
    o = []
    for b in range(B):
        pick = np.concatenate([idx[x] for x in RNG.choice(ud, len(ud), replace=True)])
        if len(np.unique(y[pick])) < 2:
            continue
        o.append(roc_auc_score(y[pick], p[pick]))
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(DS).reset_index(drop=True)
    feats = [c for c in D.columns if c not in NON]
    sp = D["split"].to_numpy(); tr, va, oo = sp == "TRAIN", sp == "VAL", sp == "OOS"
    ts = pd.to_datetime(D["timestamp"].to_numpy())
    days = {"VAL": pd.Series(ts[va]).dt.floor("D").to_numpy(), "OOS": pd.Series(ts[oo]).dt.floor("D").to_numpy()}
    pf, pc = D["pnl_fade"].to_numpy(), D["pnl_cont"].to_numpy()

    bar, seg, p0 = M.build_bars()
    H_, L_, C_ = (seg[c].to_numpy(float) for c in ("high", "low", "close"))
    kp = D["pos"].to_numpy() - p0
    sgn_fade = np.where(D["is_downside"].to_numpy() == 1, 1.0, -1.0)   # 페이드 방향
    c0 = C_[kp]                                                        # 발동 봉 종가 (기준점)

    labels = {}
    for H in HORIZONS:
        off = np.arange(1, H + 1)
        Hm = H_[kp[:, None] + off[None, :]]; Lm = L_[kp[:, None] + off[None, :]]
        Cm = C_[kp[:, None] + off[None, :]]
        up = (Hm.max(axis=1) - c0) / c0                                # 위쪽 최대이탈
        dn = (c0 - Lm.min(axis=1)) / c0                                # 아래쪽 최대이탈
        # 페이드 방향 sgn_fade: +1(롱) 이면 유리=위 / -1(숏) 이면 유리=아래
        mfe_fade = np.where(sgn_fade > 0, up, dn)
        mfe_cont = np.where(sgn_fade > 0, dn, up)
        denom = mfe_cont + mfe_fade
        r1 = np.where(denom > 0, mfe_cont / np.maximum(denom, 1e-12), 0.5)
        side = np.sign((Cm - c0[:, None]) * (-sgn_fade)[:, None])       # 지속 쪽이면 +1
        r2 = (side > 0).mean(axis=1)
        r3 = (-sgn_fade) * (Cm[:, -1] - c0) / c0
        r4 = np.log(np.maximum(mfe_cont, 1e-9) / np.maximum(mfe_fade, 1e-9))
        labels[f"R1 경로우세도 H{H}"] = ("clf", (r1 > 0.5).astype(int), r1)
        labels[f"R1r 경로우세도(회귀) H{H}"] = ("reg", r1, r1)
        labels[f"R2 시간우세도 H{H}"] = ("clf", (r2 > 0.5).astype(int), r2)
        labels[f"R3 순수부호 H{H}"] = ("clf", (r3 > 0).astype(int), r3)
        labels[f"R4 진폭비로그 H{H}"] = ("reg", r4, r4)

    res = {}
    print(f"\n{'라벨':<26}{'양성률/평균':>12}{'VAL':>9}{'[일군집 CI]':>22}{'OOS':>9}{'[일군집 CI]':>22}  1차")
    for name, (task, y, raw) in labels.items():
        y = np.asarray(y)
        if task == "clf" and len(np.unique(y[tr])) < 2:
            continue
        P = np.zeros(len(y))
        for sd in SEEDS:
            m = (HistGradientBoostingClassifier if task == "clf" else HistGradientBoostingRegressor)(
                max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=40,
                l2_regularization=1.0, random_state=sd)
            m.fit(D.loc[tr, feats], y[tr])
            for msk in (va, oo):
                P[msk] += (m.predict_proba(D.loc[msk, feats])[:, 1] if task == "clf"
                           else m.predict(D.loc[msk, feats])) / len(SEEDS)
        r = {"task": task, "base": float(y[tr].mean())}
        ok = []
        for w, msk in (("VAL", va), ("OOS", oo)):
            if task == "clf":
                s = float(roc_auc_score(y[msk], P[msk])); lo, hi = auc_day_ci(y[msk], P[msk], days[w])
                out_band = abs(s - 0.5) > BAND[w]
            else:
                s = float(spearmanr(y[msk], P[msk]).statistic); lo = hi = float("nan")
                out_band = abs(s) > BAND[w]
            r[w] = {"score": s, "ci": [lo, hi], "out_of_band": bool(out_band)}
            ok.append(out_band and (s - 0.5 if task == "clf" else s))
        r["stage1_pass"] = bool(r["VAL"]["out_of_band"] and r["OOS"]["out_of_band"]
                                and np.sign(ok[0]) == np.sign(ok[1]))
        # 2차: 경제성 (1차 통과 여부와 무관하게 기록)
        arms = {}
        for w, msk in (("VAL", va), ("OOS", oo)):
            for q in (0.10, 0.20, 0.30):
                use = P[msk] <= np.quantile(P[msk], q)     # 지속 우세 예측이 낮은 쪽 = 페이드 후보
                d = np.where(use, pf[msk], pc[msk]) - pc[msk]
                lo, hi = day_ci(d, days[w])
                arms[f"{w}_fadeq{q}"] = (float(d.mean()), lo, hi)
        r["arms"] = arms
        r["stage2_pass"] = any(arms[f"VAL_fadeq{q}"][1] > 0 and arms[f"OOS_fadeq{q}"][1] > 0
                               for q in (0.10, 0.20, 0.30))
        res[name] = r
        f = lambda w: (f"[{r[w]['ci'][0]:.3f}, {r[w]['ci'][1]:.3f}]" if task == "clf" else "-")
        print(f"{name:<26}{r['base']:>12.3f}{r['VAL']['score']:>9.4f}{f('VAL'):>22}"
              f"{r['OOS']['score']:>9.4f}{f('OOS'):>22}  {'⭐통과' if r['stage1_pass'] else ''}")

    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    p1 = [k for k, v in res.items() if v["stage1_pass"]]
    print(f"\n1차(레짐 분류) 통과: {len(p1)} / {len(res)}   {p1}")
    if p1:
        print("\n2차 경제성 (1차 통과 라벨만):")
        for k in p1:
            for q in (0.10, 0.20, 0.30):
                a, b = res[k]["arms"][f"VAL_fadeq{q}"], res[k]["arms"][f"OOS_fadeq{q}"]
                print(f"  {k:<26} q{q}  VAL {a[0]:+6.2f}[{a[1]:+6.2f},{a[2]:+6.2f}]  "
                      f"OOS {b[0]:+6.2f}[{b[1]:+6.2f},{b[2]:+6.2f}]")
    print(f"\n2차 통과: {[k for k,v in res.items() if v['stage2_pass']]}")
    print(f"\n산출물: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
