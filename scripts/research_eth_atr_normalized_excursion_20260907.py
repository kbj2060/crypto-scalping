#!/usr/bin/env python3
"""**ATR 정규화 단기 크기** 축 탐색 (2026-09-07, 서버 GPU).

사용자: *"ATR 정규화 단기 크기 파보자"*

## 왜 이 축인가
부록 V 에서 세 크기 라벨의 **모델 증분**(atr 단독 대비)이 이렇게 갈렸다:
    앵커 `range_pct`(정규화 없음)  atr 0.7635 → 모델 0.7959   증분 **+0.032**
    앵커 `range/atr`               atr 0.665  → 모델 0.7305   증분 +0.066
    V자 `move >= 1.5*atr`(15분)    atr 0.59~0.63 → 모델 0.7708 증분 **+0.15~0.18**
ATR 로 정규화하고 지평을 짧게 할수록 **모델이 실제로 더하는 몫이 커진다**.
정규화 없는 라벨은 변동성 지속성이라 atr 한 줄로 대부분 설명되는 반면, 정규화 라벨은 그렇지 않다.

다만 V자 수치는 **스윕 사건** 모집단에서 잰 것이다. 여기서는
  (1) **앵커 모집단**에서도 성립하는가
  (2) 지평 H · 문턱 K 를 격자로 훑어 축의 모양을 그린다
  (3) 증분이 어디서 나오는가(피쳐 카테고리)
를 본다.

## 라벨 (전부 인과적: 진입 이후만 본다)
진입 = `open[t+1]` (라벨 규약 그대로). 이후 H 봉 동안
    up = max(high)/entry - 1 · dn = 1 - min(low)/entry
    excursion = max(up, dn) / atr_pct[t]      ← **ATR 정규화**, 방향 무관
    y = excursion >= K
`atr_pct[t]` 는 앵커 봉까지의 값이라 미래를 안 본다.

## 격자 (사전 지정 15셀)
H ∈ {3, 6, 12, 24, 48} 봉 (15분~4시간) × K ∈ {1.0, 1.5, 2.0}

## 판정 -- 절대 AUC 가 아니라 **atr 대비 증분**
`atr_pct` 단독은 분모에 있어 **반대 부호**로 정보를 낸다. 기준선은 `max(auc, 1-auc)` 로 잡는다
(부록 V 에서 "0.335 니까 정보 없음"으로 잘못 읽은 실수를 반복하지 않기 위해).
통과 = 세 창 모두 일군집 CI 하한 > 0.5 ∧ **증분 짝비교 CI 하한 > 0** ∧ 날 블록 귀무 초과.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/eth_anchor_features154_20260907"
OUT = ROOT / "tmp/eth_atr_excursion_20260907"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H_GRID = (3, 6, 12, 24, 48)
K_GRID = (1.0, 1.5, 2.0)
SEED, N_EST, BOOT = 20260907, 4, 1200


def day_ci(y, p, d, rng, B=BOOT):
    u = np.unique(d)
    if len(u) < 5: return (np.nan, np.nan)
    idx = {x: np.flatnonzero(d == x) for x in u}; o = []
    for _ in range(B):
        i = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        if len(np.unique(y[i])) > 1: o.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))) if len(o) > B // 3 else (np.nan, np.nan)


def diff_ci(y, p1, p2, d, rng, B=BOOT):
    u = np.unique(d); idx = {x: np.flatnonzero(d == x) for x in u}; o = []
    for _ in range(B):
        i = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        if len(np.unique(y[i])) > 1:
            o.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))) if len(o) > B // 3 else (np.nan, np.nan)


def dayperm(y, d, rng):
    u = np.unique(d); src = {x: np.flatnonzero(d == x) for x in u}
    pm = rng.permutation(u); z = y.copy()
    for a, b in zip(u, pm):
        if len(src[a]): z[src[a]] = np.resize(y[src[b]], len(src[a]))
    return z


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    import importlib.util, torch
    from tabpfn import TabPFNClassifier
    s = importlib.util.spec_from_file_location("ab", ROOT / "scripts/build_eth_anchor_label_dataset_20260907.py")
    B = importlib.util.module_from_spec(s); s.loader.exec_module(B)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    D = pd.read_parquet(SRC / "features154.parquet")
    meta = json.loads((SRC / "meta.json").read_text())
    cols = [c for c in meta["feature_cols"] if c in D.columns]
    X = D[cols].to_numpy(np.float64)
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy()
    atr = D["atr_pct"].to_numpy(float)

    kl = B._load_kl(B.ETH_KL)
    pos = pd.Series(np.arange(len(kl)), index=pd.to_datetime(kl["timestamp"]))
    idx = pos.reindex(D["timestamp"]).to_numpy()
    assert not np.isnan(idx).any(), "앵커가 klines 에 없음"
    idx = idx.astype(int)
    O = kl["open"].to_numpy(float); HI = kl["high"].to_numpy(float); LO = kl["low"].to_numpy(float)
    print(f"[입력] 앵커 {len(D):,} · 피쳐 {len(cols)} · klines {len(kl):,} · {dev}", flush=True)

    # 진입 = open[t+1], 이후 H 봉의 최대 이탈폭 / atr
    exc = {}
    for H in H_GRID:
        e = np.full(len(D), np.nan)
        for j, i in enumerate(idx):
            a, b = i + 1, i + 1 + H
            if b > len(kl) or not np.isfinite(atr[j]) or atr[j] <= 0: continue
            entry = O[a]
            up = HI[a:b].max() / entry - 1.0
            dn = 1.0 - LO[a:b].min() / entry
            e[j] = max(up, dn) / atr[j]
        exc[H] = e
        ok = np.isfinite(e)
        print(f"   H={H:>2}봉 유효 {ok.sum():,} · 이탈폭/ATR 중앙 {np.nanmedian(e):.3f} "
              f"· 상관(atr) {np.corrcoef(e[ok], atr[ok])[0,1]:+.3f}", flush=True)

    rows = []
    print("\n" + "=" * 118, flush=True)
    print(f"{'H':>3}{'K':>6}{'양성률':>8}" + "".join(f"{w[:3]+' 모델':>16}" for w in WINS)
          + f"{'atr기준선':>12}{'평균증분':>10}{'통과':>6}", flush=True)
    print("=" * 118, flush=True)
    for H in H_GRID:
        e = exc[H]
        for K in K_GRID:
            y = np.where(np.isfinite(e), (e >= K).astype(float), np.nan)
            ok = np.isfinite(y)
            tr = ok & (sp == "TRAIN")
            if tr.sum() < 300 or len(np.unique(y[tr])) < 2: continue
            pr = y[tr].mean()
            if not (0.05 < pr < 0.95):        # 극단 불균형 셀은 건너뛴다
                rows.append({"H": H, "K": K, "pos_rate": float(y[ok].mean()), "skipped": "불균형"})
                print(f"{H:>3}{K:>6.1f}{y[ok].mean():>8.3f}   (TRAIN 양성률 {pr:.3f} -- 불균형, 건너뜀)", flush=True)
                continue
            clf = TabPFNClassifier(device=dev, n_estimators=N_EST, random_state=SEED,
                                   ignore_pretraining_limits=True, memory_saving_mode=True)
            clf.fit(np.nan_to_num(X[tr]).astype(np.float32), y[tr].astype(int))
            rec = {"H": H, "K": K, "pos_rate": float(y[ok].mean()), "n_train": int(tr.sum())}
            incs, base_aucs, ci_ok = [], [], []
            line = f"{H:>3}{K:>6.1f}{y[ok].mean():>8.3f}"
            for w in WINS:
                te = ok & (sp == w)
                if te.sum() < 40 or len(np.unique(y[te])) < 2:
                    line += f"{'-':>16}"; continue
                yy = y[te].astype(int)
                p = clf.predict_proba(np.nan_to_num(X[te]).astype(np.float32))[:, 1]
                a_m = roc_auc_score(yy, p)
                a_a = roc_auc_score(yy, atr[te])
                # ⚠️atr 은 분모라 반대 부호로 정보를 낸다 -- 기준선은 유리한 쪽으로 잡는다
                pa = atr[te] if a_a >= 0.5 else -atr[te]
                a_b = max(a_a, 1 - a_a)
                lo, _ = day_ci(yy, p, day[te], rng)
                dlo, dhi = diff_ci(yy, p, pa, day[te], rng)
                rec[f"{w}_auc"], rec[f"{w}_lo"] = a_m, lo
                rec[f"{w}_atr"] = a_b
                rec[f"{w}_d"], rec[f"{w}_dlo"], rec[f"{w}_dhi"] = a_m - a_b, dlo, dhi
                incs.append(a_m - a_b); base_aucs.append(a_b)
                ci_ok.append(bool(lo > 0.5 and dlo > 0))
                line += f"  {a_m:.4f}[{lo:.3f}]"
            rec["mean_inc"] = float(np.mean(incs)) if incs else np.nan
            rec["mean_atr"] = float(np.mean(base_aucs)) if base_aucs else np.nan
            rec["pass3"] = bool(len(ci_ok) == 3 and all(ci_ok))
            rows.append(rec)
            print(line + f"{rec['mean_atr']:>12.4f}{rec['mean_inc']:>+10.4f}"
                  f"{'  ✅' if rec['pass3'] else '  ❌':>6}", flush=True)

    A = pd.DataFrame(rows); A.to_csv(OUT / "grid.csv", index=False)
    live = A[A.get("pass3", False) == True] if "pass3" in A else A.iloc[0:0]
    print("\n" + "=" * 118, flush=True)
    print(f"세 창 CI + 증분 CI 통과: {len(live)}/{len(A[A.get('skipped').isna()]) if 'skipped' in A else len(A)}셀", flush=True)
    if len(live):
        b = live.loc[live.mean_inc.idxmax()]
        print(f"⭐최고 증분 셀: H={int(b.H)}봉({int(b.H)*5}분) K={b.K} · "
              f"모델 {np.mean([b[f'{w}_auc'] for w in WINS]):.4f} vs atr {b.mean_atr:.4f} "
              f"· 증분 {b.mean_inc:+.4f}", flush=True)
        # 귀무
        H, K = int(b.H), float(b.K)
        y = np.where(np.isfinite(exc[H]), (exc[H] >= K).astype(float), np.nan)
        ok = np.isfinite(y); tr = ok & (sp == "TRAIN")
        clf = TabPFNClassifier(device=dev, n_estimators=2, random_state=SEED,
                               ignore_pretraining_limits=True, memory_saving_mode=True)
        print("\n날 블록 셔플 귀무 B=10 (최고 셀)", flush=True)
        nl = {w: [] for w in WINS}
        for _ in range(10):
            yt = y.copy(); yt[tr] = dayperm(y[tr], day[tr], rng)
            clf.fit(np.nan_to_num(X[tr]).astype(np.float32), yt[tr].astype(int))
            for w in WINS:
                te = ok & (sp == w)
                if te.sum() >= 40 and len(np.unique(y[te])) > 1:
                    nl[w].append(roc_auc_score(y[te].astype(int),
                                               clf.predict_proba(np.nan_to_num(X[te]).astype(np.float32))[:, 1]))
        for w in WINS:
            if nl[w]:
                p95 = float(np.percentile(nl[w], 95))
                print(f"   {w:<14} 관측 {b[f'{w}_auc']:.4f} vs 귀무 p95 {p95:.4f} "
                      f"→ {'통과' if b[f'{w}_auc'] > p95 else '🔴미달'}", flush=True)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
