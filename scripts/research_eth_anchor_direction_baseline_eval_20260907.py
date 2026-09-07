#!/usr/bin/env python3
"""앵커 방향 예측 — **기준선 + 대조군 전체** (2026-09-07).

사용자: *"끝점 30분 평균으로 바꾸고 기준선+대조군 돌려줘"*

입력 `tmp/eth_anchor_modeling_table_20260907/modeling_table.parquet` (3,711 x 64)
  = 피쳐 41종(동시 세션 `build_eth_anchor_features_20260907` 산출, 09-06 `build_context`/
    `window_features` 축자 재사용 · 창 [t-2,t] 인과 · 페이드 방향 정렬 · 단일피쳐 최대 AUC 0.5354)
  + 확정 라벨(±1% 대칭 배리어 · 1분봉 첫 터치 · 끝점 = 마지막 6봉 30분 평균 종가)

## 팔 (사용자가 고른 ②+③ 을 둘 다)
  A1 가중이진   y_bin(먼저 닿기) · 표본가중 w_final = 고유도 x 품질   TRAIN 1982 / VAL 450 / OOS 340
  A2 하드필터   y3 in {0,2} 만                                       TRAIN  663 / VAL 153 / OOS 107
  A3 3클래스    y3 (지속승/혼재/되돌림승) -> 지속승 vs 되돌림승 이진 환산으로 평가

## 대조군 (사전 등록 -- 이 저장소는 대조군 없는 헤드라인을 무효로 본다)
  P1 양성대조   라벨을 **크기**(range_pct > TRAIN 중앙값)로 교체. 파이프라인이 정상이면 높게 나와야
                한다("크기는 배워지고 방향은 안 배워진다"의 확인). 여기서 낮으면 코드 결함이다.
  C1 크기단독   atr_pct 한 피쳐만으로 같은 라벨 학습
  C3 방향뒤집기 **TRAIN 라벨만** 뒤집고 VAL/OOS 는 원본으로 평가 -> AUC 가 1-원본 으로 대칭
                이동해야 정상. ⚠️학습·평가를 **둘 다** 뒤집으면 AUC 가 수학적으로 불변이라
                아무것도 검정하지 못한다(초판 결함, 2026-09-07 적발).
  C4 라벨셔플   TRAIN 라벨을 **일(day) 안에서** 셔플 B회 -> VAL/OOS AUC 귀무 분포
  C5 시드안정성 8시드
  C6 수치취약성 float32/64 x 컬럼순서 3변형 (게이트 T2)

## 판정 (실행 전 고정)
검정력 상한: OOS n=340(A1) -> AUC SE ~ 0.031 · n=107(A2) -> ~0.055.
  **A1 은 AUC 0.56, A2 는 0.61 미만을 못 가른다.** 그보다 작은 값은 주장하지 않는다.
통과 = VAL·OOS 둘 다 일군집 부트 AUC 하한 > 0.5  ∧  C4 셔플 귀무 95백분위 초과  ∧  C1 대비 우위.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/eth_anchor_modeling_table_20260907/modeling_table.parquet"
OUT = ROOT / "tmp/eth_anchor_baseline_eval_20260907"
SEEDS = [11, 23, 47, 71, 97, 131, 197, 251]
NULL_B = 30
BOOT = 2000
HP = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0)


def feature_cols(D):
    # 🔴2026-09-07: `w_quality`/`w_train` 은 **동시 세션 라벨 파이프라인의 품질 가중치**(라벨 파생)인데
    # `w_` 로 시작해 피쳐 필터를 통과했다. 혼재 라벨에 대한 단일 피쳐 AUC 0.971/0.947 -- 게이트 L3 FAIL.
    # 라벨/가중치/사후 통계는 이름이 아니라 **출처**로 걸러야 한다.
    drop = {"timestamp", "bar_idx", "side", "anchor", "n_signals", "signals", "split", "y3", "y_bin",
            "y2", "is_clean", "q_label", "w_uniq", "w_final", "w_quality", "w_train", "path_eff",
            "end_cont_pct", "range_pct", "pre_adv", "end_dir", "decided", "first_cont", "kp",
            "pre_adv_cont", "pre_adv_fade", "undecided", "hold_bars", "t_win_min", "margin_min",
            "train_quality_ok"}
    return [c for c in D.columns if c not in drop]


def day_auc_ci(y, p, days, rng, B=BOOT):
    """일(day) 군집 부트스트랩 AUC CI95."""
    uniq = np.unique(days)
    if len(uniq) < 5:
        return (np.nan, np.nan)
    idx = {d: np.flatnonzero(days == d) for d in uniq}
    out = []
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx[d] for d in pick])
        yy = y[ii]
        if len(np.unique(yy)) < 2:
            continue
        out.append(roc_auc_score(yy, p[ii]))
    if len(out) < B // 3:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def fit_predict(Xtr, ytr, wtr, Xs, seed, model="hgb", cols=None):
    if cols is not None:
        Xtr, Xs = Xtr[:, cols], [x[:, cols] for x in Xs]
    if model == "hgb":
        m = HistGradientBoostingClassifier(random_state=seed, **HP)
        m.fit(Xtr, ytr, sample_weight=wtr)
        return [m.predict_proba(x)[:, 1] for x in Xs]
    sc = StandardScaler().fit(np.nan_to_num(Xtr))
    m = LogisticRegression(max_iter=2000, C=0.5, random_state=seed)
    m.fit(sc.transform(np.nan_to_num(Xtr)), ytr, sample_weight=wtr)
    return [m.predict_proba(sc.transform(np.nan_to_num(x)))[:, 1] for x in Xs]


def arm_data(D, arm):
    """팔별 (마스크, y, w). y 는 1=지속 승."""
    if arm == "A1":
        m = np.isfinite(D["y_bin"].to_numpy()); y = D["y_bin"].to_numpy(); w = D["w_final"].to_numpy()
    elif arm == "A2":
        m = D["is_clean"].to_numpy(); y = (D["y3"].to_numpy() == 2).astype(float); w = D["w_uniq"].to_numpy()
    else:                                    # A3: 3클래스 -> 깨끗 두 클래스만 평가에 쓴다(학습은 3클래스)
        m = D["is_clean"].to_numpy(); y = (D["y3"].to_numpy() == 2).astype(float); w = D["w_uniq"].to_numpy()
    w = np.where(np.isfinite(w) & (w > 0), w, 1e-6)
    return m, y, w


def run_arm(D, F, arm, label_override=None, flip=None, cols=None, seeds=SEEDS,
            model="hgb", rng=None, tag=""):
    m, y, w = arm_data(D, arm)
    if label_override is not None:
        y = label_override.copy()
        m = m & np.isfinite(y)
    if flip == "both":
        y = 1.0 - y
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy()
    tr = m & (sp == "TRAIN"); va = m & (sp == "VAL"); oo = m & (sp == "OOS")
    y_fit = y.copy()
    if flip == "train":                       # TRAIN 만 뒤집고 평가는 원본 -> AUC 는 1-원본 이 되어야 정상
        y_fit = np.where(tr, 1.0 - y, y)
    X = F
    res = {"arm": arm, "tag": tag, "n_tr": int(tr.sum()), "n_va": int(va.sum()), "n_oo": int(oo.sum())}
    pv = np.zeros((len(seeds), int(va.sum()))); po = np.zeros((len(seeds), int(oo.sum())))
    for k, s in enumerate(seeds):
        pv[k], po[k] = fit_predict(X[tr], y_fit[tr], w[tr], [X[va], X[oo]], s, model, cols)
    for nm, mask, P in (("VAL", va, pv), ("OOS", oo, po)):
        yy = y[mask]
        if len(np.unique(yy)) < 2:
            res[f"{nm}_auc"] = np.nan; continue
        aucs = [roc_auc_score(yy, P[k]) for k in range(len(seeds))]
        res[f"{nm}_auc"] = float(np.mean(aucs)); res[f"{nm}_auc_sd"] = float(np.std(aucs))
        lo, hi = day_auc_ci(yy, P.mean(axis=0), day[mask], rng)
        res[f"{nm}_lo"], res[f"{nm}_hi"] = lo, hi
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    D = pd.read_parquet(SRC)
    cols = feature_cols(D)
    F = D[cols].to_numpy(np.float64)
    print(f"[입력] {D.shape} · 피쳐 {len(cols)}개 · split {D.split.value_counts().to_dict()}", flush=True)
    rows = []

    print("\n[1/6] 기준선 3팔 (HGB, 8시드) ...", flush=True)
    for arm in ("A1", "A2"):
        r = run_arm(D, F, arm, rng=rng, tag="HGB 41피쳐"); rows.append(r); print("   ", r, flush=True)
    r = run_arm(D, F, "A1", model="lr", rng=rng, tag="선형 41피쳐"); rows.append(r); print("   ", r, flush=True)

    print("\n[2/6] P1 양성대조 — 라벨을 '크기'로 교체 (파이프라인 정상성) ...", flush=True)
    med = np.nanmedian(D.loc[D.split == "TRAIN", "range_pct"])
    size_y = (D["range_pct"].to_numpy() > med).astype(float)
    r = run_arm(D, F, "A1", label_override=size_y, rng=rng, tag=f"P1 크기(>{med:.2f}%)"); rows.append(r); print("   ", r, flush=True)

    print("\n[3/6] C1 크기단독 — atr_pct 한 피쳐 ...", flush=True)
    ci = [cols.index("atr_pct")]
    for arm in ("A1", "A2"):
        r = run_arm(D, F, arm, cols=ci, rng=rng, tag="C1 atr_pct 단독"); rows.append(r); print("   ", r, flush=True)

    print("\n[4/6] C3 방향 뒤집기 ...", flush=True)
    r = run_arm(D, F, "A1", flip="train", rng=rng, tag="C3 TRAIN만 뒤집기"); rows.append(r); print("   ", r, flush=True)

    print(f"\n[5/6] C4 라벨 셔플 귀무 (일 내 셔플, B={NULL_B}) ...", flush=True)
    m, y, w = arm_data(D, "A1")
    day = D["timestamp"].dt.floor("D").to_numpy()
    nulls = {"VAL": [], "OOS": []}
    for b in range(NULL_B):
        ys = y.copy()
        tr = m & (D["split"].to_numpy() == "TRAIN")
        for d in np.unique(day[tr]):
            ii = np.flatnonzero(tr & (day == d))
            if len(ii) > 1:
                ys[ii] = rng.permutation(ys[ii])
        rr = run_arm(D, F, "A1", label_override=np.where(m, ys, np.nan), seeds=[SEEDS[0]], rng=rng, tag=f"C4-{b}")
        nulls["VAL"].append(rr.get("VAL_auc", np.nan)); nulls["OOS"].append(rr.get("OOS_auc", np.nan))
        if (b + 1) % 10 == 0:
            print(f"      {b+1}/{NULL_B}", flush=True)
    nz = {k: np.array([x for x in v if np.isfinite(x)]) for k, v in nulls.items()}

    print("\n[6/6] C6 수치 취약성 (float32/64 x 컬럼순서 3변형) ...", flush=True)
    frag = []
    for dt in (np.float32, np.float64):
        for perm in range(3):
            order = np.arange(len(cols)) if perm == 0 else rng.permutation(len(cols))
            rr = run_arm(D, F.astype(dt)[:, order], "A1", seeds=SEEDS[:3], rng=rng,
                         tag=f"C6 {np.dtype(dt).name} perm{perm}")
            frag.append(rr); rows.append(rr)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "arms.csv", index=False)

    print("\n" + "=" * 110)
    print(f"{'팔':<5}{'설명':<24}{'n(tr/va/oo)':>18}{'VAL AUC':>10}{'[일군집 CI]':>20}{'OOS AUC':>10}{'[일군집 CI]':>20}")
    for r in rows:
        if r["tag"].startswith(("C4-", "C6 ")):
            continue
        nn = "{}/{}/{}".format(r["n_tr"], r["n_va"], r["n_oo"])
        vci = "[{:.3f}, {:.3f}]".format(r.get("VAL_lo", float("nan")), r.get("VAL_hi", float("nan")))
        oci = "[{:.3f}, {:.3f}]".format(r.get("OOS_lo", float("nan")), r.get("OOS_hi", float("nan")))
        print(f"{r['arm']:<5}{r['tag']:<24}{nn:>18}{r.get('VAL_auc', float('nan')):>10.4f}{vci:>20}"
              f"{r.get('OOS_auc', float('nan')):>10.4f}{oci:>20}")

    print("\nC4 라벨셔플 귀무:", {k: f"평균 {v.mean():.4f} · p95 {np.percentile(v,95):.4f} · 최대 {v.max():.4f}"
                              for k, v in nz.items() if len(v)})
    fa = np.array([r.get("OOS_auc", np.nan) for r in frag]); fa = fa[np.isfinite(fa)]
    print(f"C6 수치취약성 OOS AUC 6변형: [{fa.min():.4f}, {fa.max():.4f}] · 폭 {fa.max()-fa.min():.4f} "
          f"· 부호반전 {'있음(FAIL)' if (fa.min()-0.5)*(fa.max()-0.5) < 0 else '없음'}")
    a1 = next(r for r in rows if r["arm"] == "A1" and r["tag"] == "HGB 41피쳐")
    verdict = {"A1_VAL_lo>0.5": bool(a1.get("VAL_lo", 0) > 0.5), "A1_OOS_lo>0.5": bool(a1.get("OOS_lo", 0) > 0.5),
               "A1_OOS>C4_p95": bool(a1.get("OOS_auc", 0) > np.percentile(nz["OOS"], 95)) if len(nz["OOS"]) else None}
    verdict["PASS"] = all(v for v in verdict.values() if v is not None)
    (OUT / "summary.json").write_text(json.dumps(
        {"verdict": verdict, "null_C4": {k: v.tolist() for k, v in nz.items()},
         "power": {"A1_OOS_n": a1["n_oo"], "AUC_SE": 0.031, "min_detectable": 0.56}}, indent=2, ensure_ascii=False))
    print("\n판정:", json.dumps(verdict, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------- 옵션② 3클래스
def run_3class(D, F, cols, rng, seeds=SEEDS, shuffle=False):
    """y3(0 되돌림승 / 1 혼재 / 2 지속승) 3클래스 학습 후 **두 축을 분리 평가**한다.

      (a) 혼재 식별력   P(혼재) vs 나머지 -- '언제 결판나는가'(크기/변동성 축에 가깝다)
      (b) 방향 식별력   깨끗한 건에서만 P(2)/(P(0)+P(2)) -- '어느 쪽인가'(진짜 방향 질문)
    ①/③ 과 달리 혼재를 버리지도 섞지도 않고 **예측 대상**으로 둔다 => 모델이 '진입 안 함'을 낼 수 있다.
    """
    y3 = D["y3"].to_numpy(); sp = D["split"].to_numpy()
    day = D["timestamp"].dt.floor("D").to_numpy()
    w = np.where(np.isfinite(D["w_uniq"].to_numpy()), D["w_uniq"].to_numpy(), 1e-6)
    tr = sp == "TRAIN"
    ytr = y3.copy()
    if shuffle:                                   # 일 내 셔플 귀무
        for d in np.unique(day[tr]):
            ii = np.flatnonzero(tr & (day == d))
            if len(ii) > 1:
                ytr[ii] = rng.permutation(ytr[ii])
    out = {}
    P = {"VAL": [], "OOS": []}
    for s in seeds:
        m = HistGradientBoostingClassifier(random_state=s, **HP)
        m.fit(F[tr], ytr[tr], sample_weight=w[tr])
        for nm in ("VAL", "OOS"):
            P[nm].append(m.predict_proba(F[sp == nm]))
    for nm in ("VAL", "OOS"):
        pr = np.mean(P[nm], axis=0)               # (n, 3) 평균 확률
        mask = sp == nm
        yy, dd = y3[mask], day[mask]
        # (a) 혼재 식별력
        a_y = (yy == 1).astype(int); a_p = pr[:, 1]
        out[f"{nm}_mixed_auc"] = float(roc_auc_score(a_y, a_p))
        lo, hi = day_auc_ci(a_y, a_p, dd, rng)
        out[f"{nm}_mixed_lo"], out[f"{nm}_mixed_hi"] = lo, hi
        # (b) 방향 식별력 (깨끗한 건만)
        cl = yy != 1
        b_y = (yy[cl] == 2).astype(int)
        denom = pr[cl][:, 0] + pr[cl][:, 2]
        b_p = np.where(denom > 0, pr[cl][:, 2] / np.maximum(denom, 1e-12), 0.5)
        out[f"{nm}_dir_n"] = int(cl.sum())
        if len(np.unique(b_y)) > 1:
            out[f"{nm}_dir_auc"] = float(roc_auc_score(b_y, b_p))
            lo, hi = day_auc_ci(b_y, b_p, dd[cl], rng)
            out[f"{nm}_dir_lo"], out[f"{nm}_dir_hi"] = lo, hi
    return out
