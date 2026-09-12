#!/usr/bin/env python3
"""ATR 정규화 이탈폭 — **새 축인가, 더 나은 변동성 추정인가** (2026-09-07).

사용자: *"ATR 정규화 단기 크기 파보자"* 의 후속.

## 무엇이 문제인가
H=3봉(15분)/K=1.0 에서 모델 0.7496 vs `atr_pct` 0.5781, 증분 **+0.1715** 가 나왔다.
지평이 짧을수록 증분이 커지는 단조 구조(+0.17/+0.13/+0.08/+0.05)도 일관된다.

그런데 `atr_pct` 는 **14봉 평균**이라 15분 이탈폭에는 애초에 약한 대리변수다.
150피쳐 안에는 다른 변동성 측정치가 많다(parkinson_vol · garch_vol_z · realized_vol_ratio ·
bb_width · atr_pct_rank_288 …). 그렇다면 증분은 "새 축"이 아니라
**"atr_pct 가 나쁜 추정치였다"** 일 수 있다.

## 검정 — 기준선을 올린다
  (a) `atr_pct` 단독            -- 지금까지의 기준선
  (b) **TRAIN 최고 단일피쳐**    -- TRAIN 에서만 골라 평가창에 적용(선택 편향 없음)
  (c) 변동성 계열 전체 + TabPFN  -- 이름에 vol/atr/range/width/parkinson/garch/rv 가 든 피쳐만
  (d) 전체 150 + TabPFN          -- 현재 모델
증분을 (d)−(a), (d)−(b), (d)−(c) 세 가지로 낸다.
  (d)−(c) 가 0 근처면 → **변동성 계열 안에서 끝난다**(= 더 나은 변동성 추정).
  (d)−(c) 가 유의하게 양수면 → 변동성 밖의 정보가 있다.

## 덧붙여
TRAIN 순열중요도로 무엇이 증분을 만드는지도 본다.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/eth_anchor_features154_20260907"
OUT = ROOT / "tmp/eth_excursion_volcheck_20260907"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H, K = 3, 1.0
SEED, N_EST, BOOT = 20260907, 4, 1200
VOL_PAT = re.compile(r"vol|atr|range|width|parkinson|garch|rv_|realized|bb_|std|sigma", re.I)


def day_ci(y, p, d, rng, B=BOOT):
    u = np.unique(d); idx = {x: np.flatnonzero(d == x) for x in u}; o = []
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
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy()
    atr = D["atr_pct"].to_numpy(float)
    kl = B._load_kl(B.ETH_KL)
    pos = pd.Series(np.arange(len(kl)), index=pd.to_datetime(kl["timestamp"]))
    idx = pos.reindex(D["timestamp"]).to_numpy().astype(int)
    O = kl["open"].to_numpy(float); HI = kl["high"].to_numpy(float); LO = kl["low"].to_numpy(float)
    e = np.full(len(D), np.nan)
    for j, i in enumerate(idx):
        a, b = i + 1, i + 1 + H
        if b <= len(kl) and np.isfinite(atr[j]) and atr[j] > 0:
            e[j] = max(HI[a:b].max() / O[a] - 1.0, 1.0 - LO[a:b].min() / O[a]) / atr[j]
    y = np.where(np.isfinite(e), (e >= K).astype(float), np.nan)
    ok = np.isfinite(y); tr = ok & (sp == "TRAIN")
    volc = [c for c in cols if VOL_PAT.search(c)]
    print(f"[입력] H={H}봉 K={K} · 유효 {ok.sum():,} · 양성률 {y[ok].mean():.3f} "
          f"· 변동성 계열 {len(volc)}/{len(cols)} · {dev}", flush=True)

    # (b) TRAIN 최고 단일피쳐 -- TRAIN 에서만 고른다(선택 편향 제거)
    yt = y[tr].astype(int); best = (0.5, None)
    for c in cols:
        v = D[c].to_numpy(float)[tr]; m = np.isfinite(v)
        if m.sum() < 100 or len(np.unique(yt[m])) < 2: continue
        a = roc_auc_score(yt[m], v[m]); sc = max(a, 1 - a)
        if sc > best[0]: best = (sc, c, a >= 0.5)
    bcol, bsign = best[1], best[2]
    print(f"[기준선] TRAIN 최고 단일피쳐 = {bcol} (TRAIN |AUC| {best[0]:.4f}, "
          f"부호 {'정' if bsign else '역'})", flush=True)

    def fit_pred(feats):
        Xm = D[feats].to_numpy(np.float64)
        clf = TabPFNClassifier(device=dev, n_estimators=N_EST, random_state=SEED,
                               ignore_pretraining_limits=True, memory_saving_mode=True)
        clf.fit(np.nan_to_num(Xm[tr]).astype(np.float32), yt)
        out = np.full(len(D), np.nan)
        for w in WINS:
            te = ok & (sp == w)
            if te.sum() >= 40:
                out[te] = clf.predict_proba(np.nan_to_num(Xm[te]).astype(np.float32))[:, 1]
        return out

    P = {"a_atr": np.where(True, atr, atr),
         "b_best1": D[bcol].to_numpy(float) * (1.0 if bsign else -1.0),
         "c_vol계열": fit_pred(volc),
         "d_전체150": fit_pred(cols)}
    print("\n" + "=" * 104, flush=True)
    print(f"{'모델':<14}" + "".join(f"{w[:3]:>17}" for w in WINS) + "   mean3", flush=True)
    print("=" * 104, flush=True)
    A = {}
    for k, p in P.items():
        line = f"{k:<14}"; aucs = []
        for w in WINS:
            te = ok & (sp == w) & np.isfinite(p)
            if te.sum() < 40: line += f"{'-':>17}"; continue
            yy = y[te].astype(int); a = roc_auc_score(yy, p[te])
            if k == "a_atr": a = max(a, 1 - a)          # atr 은 역부호 정보
            lo, _ = day_ci(yy, p[te] if a >= 0.5 or k != "a_atr" else -p[te], day[te], rng)
            aucs.append(a); line += f"  {a:.4f}[{lo:.3f}]"
        A[k] = float(np.mean(aucs))
        print(line + f"  {A[k]:.4f}", flush=True)

    print("\n" + "=" * 104, flush=True)
    print("⭐증분 — 전체150 대비 각 기준선 (같은 날 표집 짝비교 CI)", flush=True)
    print("=" * 104, flush=True)
    res = []
    for k in ("a_atr", "b_best1", "c_vol계열"):
        line = f"  d − {k:<12}"; n_gt = 0
        for w in WINS:
            te = ok & (sp == w) & np.isfinite(P["d_전체150"]) & np.isfinite(P[k])
            if te.sum() < 40: continue
            yy = y[te].astype(int)
            pb = P[k][te]
            if k == "a_atr" and roc_auc_score(yy, pb) < 0.5: pb = -pb
            d = roc_auc_score(yy, P["d_전체150"][te]) - roc_auc_score(yy, pb)
            lo, hi = diff_ci(yy, P["d_전체150"][te], pb, day[te], rng)
            n_gt += int(lo > 0)
            line += f"  {w[:3]} {d:+.4f}[{lo:+.3f},{hi:+.3f}]"
        res.append({"base": k, "n_win_gt0": n_gt})
        print(line + f"   창>0 {n_gt}/3", flush=True)

    print("\n" + "=" * 104, flush=True)
    g = {r["base"]: r["n_win_gt0"] for r in res}
    if g.get("c_vol계열", 0) >= 2:
        print("  ⇒ ✅변동성 계열 밖의 정보가 있다 -- 새 축으로 볼 수 있다", flush=True)
    else:
        print(f"  ⇒ 🔴변동성 계열 안에서 끝난다 -- 이 축의 정체는 **더 나은 변동성 추정**이다\n"
              f"     (atr_pct 0.578 → 최고단일 {A['b_best1']:.3f} → 변동성계열 {A['c_vol계열']:.3f} "
              f"→ 전체 {A['d_전체150']:.3f})", flush=True)
    json.dump({"auc": A, "increments": res, "best_single": bcol, "H": H, "K": K,
               "n_vol_features": len(volc)},
              open(OUT / "result.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
