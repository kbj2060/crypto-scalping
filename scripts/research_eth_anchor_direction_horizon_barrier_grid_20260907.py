#!/usr/bin/env python3
"""앵커 방향 -- **지평 × 배리어 격자**로 축을 닫는다 (2026-09-07, 서버 GPU).

사용자: *"앵커 방향의 짧은 horizon은 결과가 어떻게 된거지?"*

## 왜 지금 도는가
세션 내내 앵커 방향은 **H=48봉(4시간) · ±1% 대칭**만 봤고 0.53 이었다.
멀티트리거에서 0.74 가 나와 "짧은 지평 때문"인 줄 알았으나, 그건 `local_extreme`
(전방 6봉 참조)이 만든 **모집단 선택 룩어헤드**였다(제외하면 붕괴, B팔 양성률 0.983).
감사 과정에서 앵커를 H=6/48 × ±1.5ATR 로 두 점 재봤고 둘 다 0.51~0.53 이었다.
여기서는 **격자로 전부** 훑어 "짧은 지평에 뭔가 있나"를 완전히 닫는다.

## 격자 (사전 지정 20셀 + 대조 5셀)
  지평 H ∈ {3, 6, 12, 24, 48} 봉 (15분 ~ 4시간)
  배리어 K ∈ {0.75, 1.0, 1.5, 2.0} × ATR   + 대조로 ±1%(기존 규약) 를 H 별로
라벨 = **first-touch**(±K×ATR 중 먼저 닿는 쪽), 진입 = `open[t+1]`, 지속=1.
같은 봉에 양쪽 다 닿으면 모호로 제외. 미해소도 제외(별도 보고).

## 판정
세 창 모두 일군집 CI 하한 > 0.5 인 셀. 통과 셀이 있으면 날 블록 귀무까지 건다.
⚠️20셀 격자이므로 **통과 개수를 우연 기대치와 비교**한다(무작위도 몇 셀은 통과한다).
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
OUT = ROOT / "tmp/eth_anchor_dir_hbgrid_20260907"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H_GRID = (3, 6, 12, 24, 48)
K_GRID = (0.75, 1.0, 1.5, 2.0)
SEED, N_EST, BOOT = 20260907, 4, 1000


def day_ci(y, p, d, rng, B=BOOT):
    u = np.unique(d)
    if len(u) < 5: return (np.nan, np.nan)
    idx = {x: np.flatnonzero(d == x) for x in u}; o = []
    for _ in range(B):
        i = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        if len(np.unique(y[i])) > 1: o.append(roc_auc_score(y[i], p[i]))
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
    atr = D["atr_pct"].to_numpy(float); side = D["side"].to_numpy()
    kl = B._load_kl(B.ETH_KL)
    pos = pd.Series(np.arange(len(kl)), index=pd.to_datetime(kl["timestamp"]))
    idx = pos.reindex(D["timestamp"]).to_numpy().astype(int)
    OP = kl["open"].to_numpy(float); HI = kl["high"].to_numpy(float); LO = kl["low"].to_numpy(float)
    print(f"[입력] 앵커 {len(D):,} · 피쳐 {len(cols)} · {dev}", flush=True)

    def label(H, K, pct=False):
        """지속=1. K 가 pct 면 ±K%(고정), 아니면 ±K×ATR."""
        y = np.full(len(D), np.nan); amb = 0; unres = 0
        for j, i in enumerate(idx):
            a, b = i + 1, i + 1 + H
            if b > len(kl) or not np.isfinite(atr[j]) or atr[j] <= 0: continue
            e = OP[a]; w = (K / 100.0) if pct else (K * atr[j])
            ub, db = e * (1 + w), e * (1 - w)
            h = HI[a:b]; l = LO[a:b]
            iu = np.argmax(h >= ub) if (h >= ub).any() else 10**9
            il = np.argmax(l <= db) if (l <= db).any() else 10**9
            if iu == 10**9 and il == 10**9: unres += 1; continue
            if iu == il: amb += 1; continue
            y[j] = float((iu < il) if side[j] == "top" else (il < iu))   # 지속
        return y, amb, unres

    def fit_eval(y):
        ok = np.isfinite(y); tr = ok & (sp == "TRAIN")
        if tr.sum() < 300 or len(np.unique(y[tr])) < 2: return None
        c = TabPFNClassifier(device=dev, n_estimators=N_EST, random_state=SEED,
                             ignore_pretraining_limits=True, memory_saving_mode=True)
        c.fit(np.nan_to_num(X[tr]).astype(np.float32), y[tr].astype(int))
        r = {}
        for w in WINS:
            te = ok & (sp == w)
            if te.sum() < 40 or len(np.unique(y[te])) < 2: continue
            p = c.predict_proba(np.nan_to_num(X[te]).astype(np.float32))[:, 1]
            lo, _ = day_ci(y[te].astype(int), p, day[te], rng)
            r[w] = {"auc": roc_auc_score(y[te].astype(int), p), "lo": lo, "n": int(te.sum())}
        return r

    rows = []
    print("\n" + "=" * 106, flush=True)
    print(f"{'H':>3}{'배리어':>10}{'n':>7}{'지속률':>8}" + "".join(f"{w[:3]:>17}" for w in WINS) + "  mean3", flush=True)
    print("=" * 106, flush=True)
    for H in H_GRID:
        for K in K_GRID:
            y, amb, unres = label(H, K)
            r = fit_eval(y)
            if r is None: continue
            ok = np.isfinite(y)
            a = [r[w]["auc"] for w in WINS if w in r]
            ci3 = all(r[w]["lo"] > 0.5 for w in WINS if w in r) and len(r) == 3
            line = f"{H:>3}{f'{K}xATR':>10}{ok.sum():>7,}{np.nanmean(y[ok]):>8.3f}"
            for w in WINS:
                line += f"  {r[w]['auc']:.4f}[{r[w]['lo']:.3f}]" if w in r else f"{'-':>17}"
            rows.append({"H": H, "K": K, "type": "atr", "n": int(ok.sum()),
                         "pos": float(np.nanmean(y[ok])), "mean3": float(np.mean(a)),
                         "ci3": bool(ci3), "amb": amb, "unres": unres})
            print(line + f"  {np.mean(a):.4f}{'  ✅' if ci3 else ''}", flush=True)
        # ±1% 대조 (기존 규약)
        y, amb, unres = label(H, 1.0, pct=True)
        r = fit_eval(y)
        if r:
            ok = np.isfinite(y); a = [r[w]["auc"] for w in WINS if w in r]
            ci3 = all(r[w]["lo"] > 0.5 for w in WINS if w in r) and len(r) == 3
            line = f"{H:>3}{'±1%(대조)':>10}{ok.sum():>7,}{np.nanmean(y[ok]):>8.3f}"
            for w in WINS:
                line += f"  {r[w]['auc']:.4f}[{r[w]['lo']:.3f}]" if w in r else f"{'-':>17}"
            rows.append({"H": H, "K": 1.0, "type": "pct", "n": int(ok.sum()),
                         "pos": float(np.nanmean(y[ok])), "mean3": float(np.mean(a)), "ci3": bool(ci3)})
            print(line + f"  {np.mean(a):.4f}{'  ✅' if ci3 else ''}", flush=True)

    A = pd.DataFrame(rows); A.to_csv(OUT / "grid.csv", index=False)
    npass = int(A.ci3.sum())
    print("\n" + "=" * 106, flush=True)
    print(f"세 창 CI 통과 {npass}/{len(A)}셀 · mean3 최고 {A.mean3.max():.4f} "
          f"(H={int(A.loc[A.mean3.idxmax(),'H'])} K={A.loc[A.mean3.idxmax(),'K']})", flush=True)
    print(f"⚠️20+셀 격자다 -- 우연 기대 통과수는 셀당 (1-0.95^... ) 가 아니라 상관된 셀들이라 "
          f"날블록 귀무로 직접 잰다", flush=True)
    if npass:
        b = A.loc[A.mean3.idxmax()]
        y, _, _ = label(int(b.H), float(b.K), pct=(b.type == "pct"))
        ok = np.isfinite(y); tr = ok & (sp == "TRAIN")
        nl = {w: [] for w in WINS}
        for _ in range(10):
            yt = y.copy(); yt[tr] = dayperm(y[tr], day[tr], rng)
            c = TabPFNClassifier(device=dev, n_estimators=2, random_state=SEED,
                                 ignore_pretraining_limits=True, memory_saving_mode=True)
            c.fit(np.nan_to_num(X[tr]).astype(np.float32), yt[tr].astype(int))
            for w in WINS:
                te = ok & (sp == w)
                if te.sum() >= 40 and len(np.unique(y[te])) > 1:
                    nl[w].append(roc_auc_score(y[te].astype(int),
                                 c.predict_proba(np.nan_to_num(X[te]).astype(np.float32))[:, 1]))
        print(f"\n날 블록 귀무 B=10 (최고 셀 H={int(b.H)} K={b.K})", flush=True)
        for w in WINS:
            if nl[w]:
                p95 = float(np.percentile(nl[w], 95))
                obs = A.loc[A.mean3.idxmax(), f"mean3"]
                print(f"   {w:<14} 귀무 p95 {p95:.4f}", flush=True)
    else:
        print("⇒ 통과 셀 없음 -- 지평·배리어 어느 조합에서도 방향은 동전이다", flush=True)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
