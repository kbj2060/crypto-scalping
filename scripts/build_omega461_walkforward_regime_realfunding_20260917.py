#!/usr/bin/env python3
"""C단계 — walk-forward 레짐 6열 생성 (2026-09-17).

2022-01~2023-12 → B단계 재적합본 · 2024-01~ → 배포 balnobb.
6열 산출식은 라이브(`omega4_6_2_source_parent_live.py:230-236`)를 그대로 따른다:
  {prefix}{bull,bear,chop}_prob · _confidence=max · _margin=top1-top2
  _entropy = -(p·log p).sum() / log(3)          ← **log(3) 정규화**. 밑수 하나 틀리면 부모 입력이 어긋난다.
🔴라이브는 HMM(filter_proba + state_class_matrix), 여기는 GBM(predict_proba) -- 확률 산출 경로만
   다르고 6열 변환은 동일하다. 그 차이를 이 주석으로 명시한다.

C2 판 (2026-09-17 재실행): 입력이 **펀딩 복구본** 프레임으로 바뀌었다.
옛 프레임은 2022~2024 펀딩이 전량 중앙값이라 파생 9열이 같이 상수였다(D단계 §3).
모델 config·라벨식·검증 절차는 한 글자도 바꾸지 않았다 -- 바뀐 건 입력뿐이다.
"""
import sys, numpy as np, pandas as pd, joblib
from pathlib import Path
ROOT = Path("/home/llewyn/crypto-scalping"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"trading_bot_modules"))
from omega4_6_2_source_parent_live import CURRENT_PREFIX

OUT = ROOT/"tmp/omega461_longwindow_20260917"
SEAM = pd.Timestamp("2024-01-01")

F = pd.read_parquet(OUT/"features_136_2022_2026_realfunding.parquet")
F["timestamp"] = pd.to_datetime(F["timestamp"])
old = joblib.load(ROOT/"tmp/eth_regime_balnobb_20260910/model.joblib")
new = joblib.load(OUT/"regime_balnobb_refit2022_realfunding.joblib")
FC, CLS = old["feature_cols"], old["classes"]
assert new["feature_cols"] == FC and new["classes"] == CLS, "피쳐/클래스 계약 불일치"
assert int(F[FC].isna().sum().sum()) == 0, "입력에 NaN"

def six(proba):
    p = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    sp = np.sort(p, axis=1)
    return {**{f"{CURRENT_PREFIX}{c}_prob": p[:, i] for i, c in enumerate(CLS)},
            f"{CURRENT_PREFIX}confidence": p.max(axis=1),
            f"{CURRENT_PREFIX}margin": sp[:, -1] - sp[:, -2],
            f"{CURRENT_PREFIX}entropy": -(p*np.log(np.clip(p,1e-12,None))).sum(axis=1)/np.log(3.0)}

pre, post = F.timestamp < SEAM, F.timestamp >= SEAM
print(f"2022~2023 {int(pre.sum()):,}봉 → 재적합본 · 2024~ {int(post.sum()):,}봉 → 배포본", flush=True)
cols = [f"{CURRENT_PREFIX}{c}_prob" for c in CLS] + [f"{CURRENT_PREFIX}{k}" for k in ("confidence","margin","entropy")]
for c in cols: F[c] = np.nan
for mask, art, tag in ((pre, new, "refit2022"), (post, old, "deployed")):
    X = pd.DataFrame(F.loc[mask, FC].to_numpy(np.float64), columns=FC)
    for k, v in six(art["model"].predict_proba(X)).items():
        F.loc[mask, k] = v
    print(f"  {tag}: {int(mask.sum()):,}봉 완료", flush=True)

assert int(F[cols].isna().sum().sum()) == 0, "레짐 6열에 NaN 잔존"
print("\n=== 연도별 레짐 분포·안정성 ===")
print(f"{'연도':>6s} {'bull%':>7s} {'bear%':>7s} {'chop%':>7s} {'conf중앙':>9s} {'ent중앙':>8s} {'flip%':>7s}")
F["arg"] = F[[f"{CURRENT_PREFIX}{c}_prob" for c in CLS]].to_numpy().argmax(1)
for y, g in F.groupby(F.timestamp.dt.year):
    sh = np.bincount(g.arg, minlength=3)/len(g)
    flip = float((g.arg.to_numpy()[1:] != g.arg.to_numpy()[:-1]).mean())
    print(f"{y:6d} {sh[0]*100:7.1f} {sh[1]*100:7.1f} {sh[2]*100:7.1f} "
          f"{g[f'{CURRENT_PREFIX}confidence'].median():9.4f} {g[f'{CURRENT_PREFIX}entropy'].median():8.4f} {flip*100:7.2f}")

# ── 🔴이음매 검증: 경계 ±3일에서 출력이 튀는가 ──
w = F[(F.timestamp >= SEAM - pd.Timedelta(days=3)) & (F.timestamp < SEAM + pd.Timedelta(days=3))]
a, b = w[w.timestamp < SEAM], w[w.timestamp >= SEAM]
print(f"\n=== 이음매 (2023-12-29~2024-01-03) ===")
for c in cols:
    print(f"  {c.replace(CURRENT_PREFIX,''):12s} 직전3일 {a[c].mean():+.4f}  직후3일 {b[c].mean():+.4f}  Δ {b[c].mean()-a[c].mean():+.4f}")
fa = float((a.arg.to_numpy()[1:] != a.arg.to_numpy()[:-1]).mean()); fb = float((b.arg.to_numpy()[1:] != b.arg.to_numpy()[:-1]).mean())
print(f"  전이율 직전 {fa*100:.2f}% · 직후 {fb*100:.2f}%")

F.drop(columns=["arg"]).to_parquet(OUT/"features_with_regime_2022_2026_realfunding.parquet", index=False)
print(f"\n저장 {OUT/'features_with_regime_2022_2026_realfunding.parquet'} · {len(F):,}행 × {F.shape[1]-1}열")
