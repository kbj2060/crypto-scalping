#!/usr/bin/env python3
"""페이드 vs 지속 — TabPFN 분류/회귀 (2026-09-06). **서버에서 실행**(GPU + 캐시된 체크포인트).

입력 하나: tmp/eth_fade_vs_cont_dataset_20260906/dataset.parquet (로컬에서 동결해 push)
의존성: tabpfn, sklearn, pandas 뿐 — 저장소 모듈을 임포트하지 않는다(동기화 회피).

왜 TabPFN인가: 지금까지 실패는 전부 **과적합 서명**이다(HGB TRAIN 0.62~0.66 -> 표본외 0.49~0.51).
TabPFN은 데이터셋별 학습이 없는 in-context 추론이라 다른 귀납 편향을 준다. 모델 효과만 보려고
피쳐·라벨·분할을 동결하고 **HGB 매칭 대조군**을 같은 입력에 나란히 돌린다.

라벨: y_order(분류) · y_dec(분류) · margin(회귀, bp)
피쳐셋: compact(24, 사전 큐레이션) · f5m(테이프 제외) · all
통제: 5시드 · TRAIN에서만 적합 · VAL/OOS 1회 · 누수가드(VAL AUC>=0.99) · 라우팅 일군집 CI
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
OUT = ROOT / "tmp/eth_fade_vs_cont_tabpfn_20260906"
SEEDS = [11, 23, 47, 71, 97]
DEVICE = "cuda"
RNG = np.random.default_rng(20260906)

COMPACT = ["votes", "is_downside", "atr_pct", "signal_id",
           "w_ret_atr", "w_vol_z_end", "w_delta_z_aligned", "w_range_atr_end", "w_trades_z_end",
           "w_oi_chg", "w_forced_flow", "w_lsr_retail_chg", "w_taker_ratio_chg", "w_funding_z",
           "w_basis_chg", "w_depth_imb", "w_btc_ret_aligned",
           "tp_imb", "tp_lg_imb", "tp_xl_imb", "tp_kyle", "tp_tpm", "tp_runlen", "tp_imb_chg"]
NON_FEAT = {"y_order", "y_dec", "margin", "pnl_fade", "pnl_cont", "cls3", "split", "timestamp", "pos"}


def log(m): print(f"[tpfn] {m}", flush=True)


def day_ci(v, d, B=2000):
    ud = np.unique(d); idx = {x: np.flatnonzero(d == x) for x in ud}
    o = np.empty(B)
    for b in range(B):
        p = RNG.choice(ud, len(ud), replace=True)
        o[b] = np.concatenate([v[idx[x]] for x in p]).mean()
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def predict_batched(model, X, proba: bool, bs=8000):
    outs = []
    for k in range(0, len(X), bs):
        chunk = X.iloc[k:k + bs]
        outs.append(model.predict_proba(chunk)[:, 1] if proba else model.predict(chunk))
    return np.concatenate(outs)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(DS)
    tr, va, oo = (D["split"] == "TRAIN").to_numpy(), (D["split"] == "VAL").to_numpy(), (D["split"] == "OOS").to_numpy()
    log(f"{len(D):,}행  TRAIN {tr.sum():,} / VAL {va.sum():,} / OOS {oo.sum():,}")
    allf = [c for c in D.columns if c not in NON_FEAT]
    f5m = [c for c in allf if not c.startswith("tp_")]
    compact = [c for c in COMPACT if c in D.columns]
    log(f"피쳐셋 compact {len(compact)} / f5m {len(f5m)} / all {len(allf)}")
    if len(compact) != len(COMPACT):
        log(f"⚠️ compact 누락: {sorted(set(COMPACT) - set(D.columns))}")

    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        have_tp = True
    except Exception as ex:                                                # noqa: BLE001
        log(f"TabPFN 임포트 실패: {ex}"); have_tp = False

    days_va = pd.Series(D.loc[va, "timestamp"]).dt.floor("D").to_numpy()
    days_oo = pd.Series(D.loc[oo, "timestamp"]).dt.floor("D").to_numpy()
    pf, pc = D["pnl_fade"].to_numpy(), D["pnl_cont"].to_numpy()
    res = {}

    # TRAIN 진단 지표는 3,000행 고정 표본으로만 — GPU 점유 축소(대시보드와 공유). 선택/판정에는 안 쓴다.
    tr_idx = np.flatnonzero(tr)
    tr_sub = np.sort(np.random.default_rng(7).choice(tr_idx, size=min(3000, len(tr_idx)), replace=False))
    for fs_name, cols in (("compact", compact), ("f5m", f5m), ("all", allf)):
        Xtr, Xva, Xoo = D.loc[tr, cols], D.loc[va, cols], D.loc[oo, cols]
        Xtrs = D.loc[tr_sub, cols]
        cat_idx = [cols.index("signal_id")] if "signal_id" in cols else []
        for task, label in (("clf", "y_order"), ("clf", "y_dec"), ("reg", "margin")):
            ytr = D.loc[tr, label].to_numpy()
            yva, yoo = D.loc[va, label].to_numpy(), D.loc[oo, label].to_numpy()
            for engine in (["tabpfn", "hgb"] if have_tp else ["hgb"]):
                key = f"{fs_name}|{label}|{engine}"
                t0 = time.time()
                Pva = np.zeros(va.sum()); Poo = np.zeros(oo.sum()); Ptr = np.zeros(len(tr_sub))
                try:
                    for sd in SEEDS:
                        if engine == "tabpfn":
                            m = (TabPFNClassifier if task == "clf" else TabPFNRegressor)(
                                device=DEVICE, random_state=sd, ignore_pretraining_limits=True,
                                categorical_features_indices=cat_idx or None)
                        else:
                            m = (HistGradientBoostingClassifier if task == "clf" else HistGradientBoostingRegressor)(
                                max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=40,
                                l2_regularization=1.0, random_state=sd)
                        m.fit(Xtr, ytr)
                        Ptr += predict_batched(m, Xtrs, task == "clf") / len(SEEDS)
                        Pva += predict_batched(m, Xva, task == "clf") / len(SEEDS)
                        Poo += predict_batched(m, Xoo, task == "clf") / len(SEEDS)
                except Exception as ex:                                     # noqa: BLE001
                    log(f"{key}: 실패 {type(ex).__name__}: {ex}"); res[key] = {"error": str(ex)}; continue

                r = {"engine": engine, "task": task, "label": label, "featureset": fs_name,
                     "n_feat": len(cols), "sec": round(time.time() - t0, 1)}
                if task == "clf":
                    r["auc"] = {"train": float(roc_auc_score(D.loc[tr_sub, label].to_numpy(), Ptr)), "val": float(roc_auc_score(yva, Pva)),
                                "oos": float(roc_auc_score(yoo, Poo))}
                    r["leak_fail"] = bool(r["auc"]["val"] >= 0.99)
                    score_va, score_oo = Pva, Poo
                else:
                    ytrs = D.loc[tr_sub, label].to_numpy()
                    r["spearman"] = {w: float(spearmanr(a, b).statistic) for w, a, b in
                                     (("train", ytrs, Ptr), ("val", yva, Pva), ("oos", yoo, Poo))}
                    r["sign_acc"] = {w: float(((a > 0) == (b > 0)).mean()) for w, a, b in
                                     (("train", ytrs, Ptr), ("val", yva, Pva), ("oos", yoo, Poo))}
                    r["pred_pos_rate"] = {"val": float((Pva > 0).mean()), "oos": float((Poo > 0).mean())}
                    score_va, score_oo = Pva, Poo
                # 라우팅: 점수 상위 q% 를 페이드로 (회귀는 margin>0 규칙도 별도)
                arms = {}
                for wn, msk, sc, dd in (("VAL", va, score_va, days_va), ("OOS", oo, score_oo, days_oo)):
                    for q in (0.10, 0.20, 0.30):
                        thr = np.quantile(sc, 1 - q); use = sc >= thr
                        d = np.where(use, pf[msk], pc[msk]) - pc[msk]
                        lo, hi = day_ci(d, dd)
                        arms[f"{wn}_q{q}"] = {"n": int(use.sum()), "delta": float(d.mean()), "ci": [lo, hi]}
                    if task == "reg":
                        use = sc > 0
                        d = np.where(use, pf[msk], pc[msk]) - pc[msk]
                        lo, hi = day_ci(d, dd) if use.any() else (0.0, 0.0)
                        arms[f"{wn}_margin>0"] = {"n": int(use.sum()), "delta": float(d.mean()), "ci": [lo, hi]}
                r["arms"] = arms
                res[key] = r
                s = (f"AUC {r['auc']['train']:.4f}/{r['auc']['val']:.4f}/{r['auc']['oos']:.4f}" if task == "clf"
                     else f"rho {r['spearman']['train']:+.4f}/{r['spearman']['val']:+.4f}/{r['spearman']['oos']:+.4f}")
                log(f"{key:<28} {s}  ({r['sec']}s)")

    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    print("\n" + "=" * 104)
    print(f"{'피쳐셋':<9}{'라벨':<9}{'엔진':<8}{'지표 TRAIN':>12}{'VAL':>10}{'OOS':>10}   {'OOS 라우팅 q0.2 Δ [CI]':>30}")
    for k, r in res.items():
        if "error" in r:
            print(f"{k:<28} 실패"); continue
        a = r.get("auc") or r.get("spearman")
        arm = r["arms"]["OOS_q0.2"]
        print(f"{r['featureset']:<9}{r['label']:<9}{r['engine']:<8}"
              f"{a['train']:>12.4f}{a['val']:>10.4f}{a['oos']:>10.4f}   "
              f"{arm['delta']:>+8.2f} [{arm['ci'][0]:+7.2f},{arm['ci'][1]:+7.2f}]"
              + ("  ⛔누수" if r.get("leak_fail") else ""))
    print(f"\n산출물: {OUT / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
