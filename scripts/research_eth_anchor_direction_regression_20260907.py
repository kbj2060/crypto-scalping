#!/usr/bin/env python3
"""앵커 되돌림/지속 — **회귀 접근** (2026-09-07).

사용자: *"우린 방향이 아니라 되돌림과 지속을 예측하는데 이걸 회귀로 해보는건 어때"*

⚠️먼저 정정: 지금까지의 라벨도 **원시 상승/하락이 아니라 되돌림/지속**이다(라벨·피쳐 모두 앵커
측면 기준 정렬 -- 바닥이면 지속=숏, 천장이면 지속=롱). 바뀌는 것은 **분류 -> 회귀** 하나다.

## 회귀가 새로운 이유
분류는 경계(0.5) 근처에서 잡음이 지배하고 **크기 정보를 버린다**. 회귀는 큰 결과에 자연히
가중을 둔다. 09-06 `margin` 회귀가 있었으나 그건 **first-fire 앵커 + 출구 오염 라벨**이었다.

## ⭐설계상 가장 중요한 함정
회귀 타깃은 **크기와 강하게 상관**된다(|끝점|이 크면 변동성이 큰 구간). 그래서 rho 가 높아도
**부호를 못 맞히면 크기만 배운 것**이다. 이 저장소의 반복 검출이 정확히 그것이다.
=> **rho 와 부호 정확도를 분리 평가**하고, 크기 회귀를 양성대조로 병기한다.

## 타깃 4종 (전부 앵커 측면 정렬, 지속 방향이 +)
  R1 end_cont      판정창 끝점(마지막 30분 평균 종가)의 지속 방향 수익 %
  R2 r1_path       지속MFE / (지속MFE + 페이드MFE)  in [0,1] -- 0.5 가 중립 (경로우세도)
  R3 t_gap         (페이드 터치분 - 지속 터치분) / 240, 미터치는 창끝(240분) 검열 -- 속도 차
  R4 end_log       sign(end_cont) * log1p(|end_cont|) -- 꼬리 압축, 부호 보존
  P  size(양성대조) |end_cont| -- 방향 없는 크기. 파이프라인 정상성 확인용.

## 평가 (사전 고정)
  (a) Spearman rho(예측, 실제) · 일군집 부트 CI
  (b) **부호 정확도** -- 예측 부호가 실제 부호와 맞는 비율(중립 제외). 이게 진짜 되돌림/지속 질문
  (c) **선별 이득** -- 예측 상위 30% 의 실제 평균 - 전체 평균 (타깃 단위)
  대조군: 날 블록 셔플 귀무 B=20 · atr_pct 단독 · 양성대조
  튜닝/피쳐선택은 하지 않는다 -- 부록 M 에서 이월성 rho 가 음수(-0.937)로 나와 의미가 없다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import research_eth_anchor_direction_baseline_eval_20260907 as BE  # noqa: E402

DIR = ROOT / "tmp/eth_anchor_direction_labels_20260907/direction_labels.parquet"
OUT = ROOT / "tmp/eth_anchor_regression_20260907"
SEEDS = [11, 23, 47, 71, 97]
NULL_B = 20
BOOT = 1500
HP = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0)
LIM_MIN = 240.0                      # H=48봉 = 240분


def build_targets(D):
    L = pd.read_parquet(DIR)[["timestamp", "side", "hit_cont_min_P1", "hit_fade_min_P1"]]
    D = D.merge(L, on=["timestamp", "side"], how="left")
    tc = D["hit_cont_min_P1"].to_numpy(float); tf = D["hit_fade_min_P1"].to_numpy(float)
    tc = np.where(np.isfinite(tc) & (tc < LIM_MIN), tc, LIM_MIN)
    tf = np.where(np.isfinite(tf) & (tf < LIM_MIN), tf, LIM_MIN)
    e = D["end_cont_pct"].to_numpy(float)
    up = D["mfe_cont"].to_numpy(float) if "mfe_cont" in D else None
    T = {"R1_end_cont": e,
         "R3_t_gap": (tf - tc) / LIM_MIN,
         "R4_end_log": np.sign(e) * np.log1p(np.abs(e)),
         "P_size": np.abs(e)}
    # R2 경로우세도: 창 진폭과 끝점으로는 못 만든다 -> 터치 시각으로 대용(둘 다 미터치면 0.5)
    both = (tc < LIM_MIN) | (tf < LIM_MIN)
    r2 = np.where(both, tf / np.maximum(tf + tc, 1e-9), 0.5)
    T["R2_r1_path"] = r2
    return D, T


def day_ci_stat(fn, y, p, days, rng, B=BOOT):
    uniq = np.unique(days)
    if len(uniq) < 5:
        return (np.nan, np.nan)
    idx = {d: np.flatnonzero(days == d) for d in uniq}
    out = []
    for _ in range(B):
        ii = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        v = fn(y[ii], p[ii])
        if np.isfinite(v):
            out.append(v)
    if len(out) < B // 3:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def sign_acc(y, p, neutral=None):
    m = np.isfinite(y) & np.isfinite(p) & (y != 0)
    if neutral is not None:
        m &= np.abs(y - neutral) > 1e-12
        return float((np.sign(p[m] - neutral) == np.sign(y[m] - neutral)).mean()) if m.sum() else np.nan
    return float((np.sign(p[m]) == np.sign(y[m])).mean()) if m.sum() else np.nan


def run(D, F, y, rng, shuffle=False, cols=None, seeds=SEEDS, neutral=None):
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy()
    w = np.where(np.isfinite(D["w_uniq"].to_numpy()), D["w_uniq"].to_numpy(), 1e-6)
    ok = np.isfinite(y)
    tr = ok & (sp == "TRAIN")
    yt = y.copy()
    if shuffle:                                   # 날 블록 셔플 (부록 L4)
        days = np.unique(day[tr]); perm = rng.permutation(days)
        src = {d: np.flatnonzero(tr & (day == d)) for d in days}
        for d, d2 in zip(days, perm):
            if len(src[d]):
                yt[src[d]] = np.resize(y[src[d2]], len(src[d]))
    X = F if cols is None else F[:, cols]
    out = {}
    for nm in ("VAL", "OOS"):
        te = ok & (sp == nm)
        p = np.mean([HistGradientBoostingRegressor(random_state=s, **HP)
                     .fit(X[tr], yt[tr], sample_weight=w[tr]).predict(X[te]) for s in seeds], axis=0)
        yy = y[te]; dd = day[te]
        out[f"{nm}_n"] = int(te.sum())
        out[f"{nm}_rho"] = float(spearmanr(yy, p)[0])
        out[f"{nm}_rho_lo"], out[f"{nm}_rho_hi"] = day_ci_stat(lambda a, b: spearmanr(a, b)[0], yy, p, dd, rng)
        out[f"{nm}_sign"] = sign_acc(yy, p, neutral)
        out[f"{nm}_sign_lo"], out[f"{nm}_sign_hi"] = day_ci_stat(
            lambda a, b: sign_acc(a, b, neutral), yy, p, dd, rng)
        k = max(10, int(0.3 * te.sum()))
        top = np.argsort(-p)[:k]
        out[f"{nm}_lift_top30"] = float(np.mean(yy[top]) - np.mean(yy))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    D = pd.read_parquet(BE.SRC)
    cols = BE.feature_cols(D)
    D, T = build_targets(D)
    F = D[cols].to_numpy(np.float64)
    ai = [cols.index("atr_pct")]
    print(f"[입력] {D.shape} · 피쳐 {len(cols)} · 타깃 {list(T)}", flush=True)
    for k, v in T.items():
        print(f"   {k:<14} 유효 {int(np.isfinite(v).sum()):>5} · 중앙 {np.nanmedian(v):+.4f} · "
              f"sd {np.nanstd(v):.4f} · 양성률 {float(np.nanmean(v > (0.5 if k=='R2_r1_path' else 0))):.3f}", flush=True)
    rows, nulls = [], []
    for name, y in T.items():
        neu = 0.5 if name == "R2_r1_path" else None
        print(f"\n[{name}] ...", flush=True)
        r = run(D, F, y, rng, neutral=neu); r["target"] = name; r["model"] = "39피쳐"
        rows.append(r); print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        ra = run(D, F, y, rng, cols=ai, neutral=neu); ra["target"] = name; ra["model"] = "atr 단독"
        rows.append(ra)
        nv, no, sv, so = [], [], [], []
        for _ in range(NULL_B):
            rr = run(D, F, y, rng, shuffle=True, seeds=SEEDS[:2], neutral=neu)
            nv.append(rr["VAL_rho"]); no.append(rr["OOS_rho"])
            sv.append(rr["VAL_sign"]); so.append(rr["OOS_sign"])
        nulls.append({"target": name,
                      "rho_VAL_p95": float(np.nanpercentile(nv, 95)), "rho_OOS_p95": float(np.nanpercentile(no, 95)),
                      "sign_VAL_p95": float(np.nanpercentile(sv, 95)), "sign_OOS_p95": float(np.nanpercentile(so, 95))})
        print("   귀무:", {k: round(v, 4) for k, v in nulls[-1].items() if k != "target"}, flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "regression.csv", index=False)
    N = pd.DataFrame(nulls); N.to_csv(OUT / "nulls.csv", index=False)

    print("\n" + "=" * 118)
    print(f"{'타깃':<14}{'모델':<9}{'VAL rho':>9}{'[CI]':>18}{'VAL 부호':>9}{'[CI]':>18}"
          f"{'OOS rho':>9}{'[CI]':>18}{'OOS 부호':>9}{'[CI]':>18}")
    for r in rows:
        f = lambda a, b: "[{:+.3f},{:+.3f}]".format(r.get(a, np.nan), r.get(b, np.nan))
        print(f"{r['target']:<14}{r['model']:<9}{r['VAL_rho']:>9.4f}{f('VAL_rho_lo','VAL_rho_hi'):>18}"
              f"{r['VAL_sign']:>9.4f}{f('VAL_sign_lo','VAL_sign_hi'):>18}"
              f"{r['OOS_rho']:>9.4f}{f('OOS_rho_lo','OOS_rho_hi'):>18}"
              f"{r['OOS_sign']:>9.4f}{f('OOS_sign_lo','OOS_sign_hi'):>18}")
    print()
    print(f"{'타깃':<14}{'귀무 rho p95 (V/O)':>24}{'귀무 부호 p95 (V/O)':>24}{'선별이득 상위30% (V/O)':>26}")
    for n, r in zip(nulls, [x for x in rows if x["model"] == "39피쳐"]):
        c1 = "{:+.4f} / {:+.4f}".format(n["rho_VAL_p95"], n["rho_OOS_p95"])
        c2 = "{:.4f} / {:.4f}".format(n["sign_VAL_p95"], n["sign_OOS_p95"])
        c3 = "{:+.4f} / {:+.4f}".format(r["VAL_lift_top30"], r["OOS_lift_top30"])
        print(f"{n['target']:<14}{c1:>24}{c2:>24}{c3:>26}")
    (OUT / "summary.json").write_text(json.dumps({"rows": rows, "nulls": nulls}, indent=2, ensure_ascii=False, default=float))
    print(f"\n저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
