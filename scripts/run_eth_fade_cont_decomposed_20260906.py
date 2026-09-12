#!/usr/bin/env python3
"""페이드/지속 모델 v2 — **분해**(신호별·측면별) + 그룹내 순위 정규화 (2026-09-06). 서버 실행.

사용자: *"우선 나는 페이드냐 지속이냐를 맞추는 모델을 우선적으로 만들고 싶어."*

지금까지는 8종 × 2측면을 **한 모델에 풀링**하고 signal_id를 피쳐로만 줬다. 저장소가 스스로
원인 후보로 지목한 것을 정면으로 검정한다:
  §12(1)b — *"8종 확률이 신호별로 다르게 캘리브레이션돼 있어 합치면 순위가 깨진다"*

## 팔 (사전 고정)
  A0 pooled            1 모델 (기준, 지금까지의 구성)
  A1 per-side          2 모델 (바닥 / 천장)
  A2 per-signal        8 모델
  A3 per-signal×side  16 모델
각 팔 × 라벨 3종(y_order·y_dec·margin) × 엔진 2종(TabPFN·HGB).
그룹 TRAIN n<100 또는 단일 클래스면 그 그룹은 **무정보**(TRAIN 기저율)로 채운다.

## 라우팅 두 방식 (⭐이게 §12(1)b 검정의 핵심)
  raw   : 조립한 점수 그대로 상위 q% -> 서로 다른 모델의 확률을 직접 비교(= 순위가 깨지는 쪽)
  rank  : **그룹 내 백분위**로 변환한 뒤 상위 q% -> 캘리브레이션 차이를 제거
분해가 효과가 있다면 rank 라우팅에서 나와야 한다.

평가 경제성은 배포 셀 고정(진입 open[pos+1] · sim_exit 5.0/1.5/0.1 · 200봉 · 10bp),
cont_all 대비 일별 짝비교, VAL·OOS 두 창 CI 하한 > 0 이 통과.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "tmp/eth_fade_vs_cont_dataset_20260906/dataset.parquet"
OUT = ROOT / "tmp/eth_fade_cont_decomposed_20260906"
SEEDS = [11, 23, 47, 71, 97]
DEVICE = "cuda"
MIN_FIT = 100
RNG = np.random.default_rng(20260906)
NON_FEAT = {"y_order", "y_dec", "margin", "pnl_fade", "pnl_cont", "cls3", "split", "timestamp", "pos"}


def log(m): print(f"[decomp] {m}", flush=True)


def day_ci(v, d, B=1500):
    ud = np.unique(d); idx = {x: np.flatnonzero(d == x) for x in ud}
    o = np.empty(B)
    for b in range(B):
        p = RNG.choice(ud, len(ud), replace=True)
        o[b] = np.concatenate([v[idx[x]] for x in p]).mean()
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def fit_group(engine, task, Xtr, ytr, Xpred, cat_idx):
    P = np.zeros(len(Xpred))
    for sd in SEEDS:
        if engine == "tabpfn":
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            m = (TabPFNClassifier if task == "clf" else TabPFNRegressor)(
                device=DEVICE, random_state=sd, ignore_pretraining_limits=True,
                categorical_features_indices=cat_idx or None)
        else:
            m = (HistGradientBoostingClassifier if task == "clf" else HistGradientBoostingRegressor)(
                max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=40,
                l2_regularization=1.0, random_state=sd)
        m.fit(Xtr, ytr)
        P += (m.predict_proba(Xpred)[:, 1] if task == "clf" else m.predict(Xpred)) / len(SEEDS)
    return P


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(DS).reset_index(drop=True)
    feats = [c for c in D.columns if c not in NON_FEAT]
    cat = [feats.index("signal_id")] if "signal_id" in feats else []
    split = D["split"].to_numpy()
    tr, va, oo = split == "TRAIN", split == "VAL", split == "OOS"
    tstamp = pd.to_datetime(D["timestamp"].to_numpy())
    pf, pc = D["pnl_fade"].to_numpy(), D["pnl_cont"].to_numpy()
    days = {"VAL": pd.Series(tstamp[va]).dt.floor("D").to_numpy(),
            "OOS": pd.Series(tstamp[oo]).dt.floor("D").to_numpy()}
    sid = D["signal_id"].to_numpy(); isd = D["is_downside"].to_numpy()
    GROUPS = {"A0 pooled": np.zeros(len(D), int),
              "A1 per-side": isd.astype(int),
              "A2 per-signal": sid.astype(int),
              "A3 per-sig×side": (sid.astype(int) * 2 + isd.astype(int))}
    log(f"{len(D):,}행 · 피쳐 {len(feats)} · TRAIN {tr.sum():,}/VAL {va.sum():,}/OOS {oo.sum():,}")

    res = {}
    for arm, g in GROUPS.items():
        for task, label in (("clf", "y_order"), ("clf", "y_dec"), ("reg", "margin")):
            y = D[label].to_numpy()
            for eng in ("tabpfn", "hgb"):
                t0 = time.time()
                P = np.full(len(D), np.nan); used, skipped = 0, 0
                for gv in np.unique(g):
                    gm = g == gv
                    ftm = gm & tr; prm = gm & (va | oo)
                    if prm.sum() == 0:
                        continue
                    ok = ftm.sum() >= MIN_FIT and (task == "reg" or len(np.unique(y[ftm])) > 1)
                    if not ok:
                        P[prm] = float(np.mean(y[ftm])) if ftm.sum() else float(np.mean(y[tr])); skipped += 1
                        continue
                    try:
                        P[prm] = fit_group(eng, task, D.loc[ftm, feats], y[ftm], D.loc[prm, feats], cat)
                        used += 1
                    except Exception as ex:                                       # noqa: BLE001
                        log(f"{arm}|{label}|{eng} g={gv} 실패 {ex}"); P[prm] = float(np.mean(y[ftm])); skipped += 1
                r = {"arm": arm, "label": label, "engine": eng, "n_groups": int(len(np.unique(g))),
                     "fitted": used, "fallback": skipped, "sec": round(time.time() - t0, 1)}
                # 그룹내 백분위 (캘리브레이션 차이 제거)
                Pr = np.full(len(D), np.nan)
                for gv in np.unique(g):
                    for w, msk in (("VAL", va), ("OOS", oo)):
                        m2 = (g == gv) & msk
                        if m2.sum() > 1:
                            Pr[m2] = pd.Series(P[m2]).rank(pct=True).to_numpy()
                        elif m2.sum() == 1:
                            Pr[m2] = 0.5
                for w, msk in (("VAL", va), ("OOS", oo)):
                    yy, pp = y[msk], P[msk]
                    r[f"score_{w}"] = (float(roc_auc_score(yy, pp)) if task == "clf" and len(np.unique(yy)) > 1
                                       else float(spearmanr(yy, pp).statistic))
                    for mode, S in (("raw", P), ("rank", Pr)):
                        s = S[msk]
                        for q in (0.10, 0.20, 0.30):
                            use = s >= np.nanquantile(s, 1 - q)
                            d = np.where(use, pf[msk], pc[msk]) - pc[msk]
                            lo, hi = day_ci(d, days[w])
                            r[f"{w}_{mode}_q{q}"] = (float(d.mean()), lo, hi)
                r["pass"] = [f"{mode}_q{q}" for mode in ("raw", "rank") for q in (0.10, 0.20, 0.30)
                             if r[f"VAL_{mode}_q{q}"][1] > 0 and r[f"OOS_{mode}_q{q}"][1] > 0]
                res[f"{arm}|{label}|{eng}"] = r
                log(f"{arm:<16}|{label:<8}|{eng:<7} {r['score_VAL']:+.4f}/{r['score_OOS']:+.4f} "
                    f"(적합 {used}/{used+skipped}) 통과={r['pass']}  ({r['sec']}s)")

    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    print("\n" + "=" * 118)
    print(f"{'팔':<16}{'라벨':<9}{'엔진':<8}{'VAL':>8}{'OOS':>8} | {'raw q0.2 VAL/OOS':>26} | {'rank q0.2 VAL/OOS':>26}  통과")
    for k, r in res.items():
        f = lambda t: f"{t[0]:+6.2f}[{t[1]:+6.2f},{t[2]:+6.2f}]"
        print(f"{r['arm']:<16}{r['label']:<9}{r['engine']:<8}{r['score_VAL']:>8.4f}{r['score_OOS']:>8.4f} | "
              f"{f(r['VAL_raw_q0.2'])}{f(r['OOS_raw_q0.2'])} | "
              f"{f(r['VAL_rank_q0.2'])}{f(r['OOS_rank_q0.2'])}  {r['pass'] or ''}")
    npass = sum(1 for r in res.values() if r["pass"])
    print(f"\n두 창 동시 통과 구성: {npass} / {len(res)}  (24구성 × 6검정 = 144, 우연 기대치 ~0.36개)")
    for k, r in res.items():
        if r["pass"]:
            print(f"  ⭐{k}  {r['pass']}")
            for p in r["pass"]:
                print(f"      VAL {f(r['VAL_'+p])}   OOS {f(r['OOS_'+p])}")
    print(f"\n산출물: {OUT/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
