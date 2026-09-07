#!/usr/bin/env python3
"""단기 지평 방향 결과 **감사 4종** (2026-09-07, 서버 GPU).

사용자: *"1번부터 4번까지 다 돌려줘"*

앞선 결과: 멀티트리거 매봉 · first-touch(±1.5×ATR) · H=6봉
    AUC 0.7398/0.7339/0.7361 (세 창 CI 하한 >0.5)
    대조군 3종 전부 통과(뒤집기 합 1.0021 · 날블록 귀무 · 무작위 direction 0.495~0.511)
    비용 후 상위30% +13.57/+16.76/+8.46bp ('항상 되돌림' -0.47/+1.89/-2.33bp)
그러나 앵커 축(H=48, ±1%)은 0.53 이었다. 왜 다른지 · 진짜인지를 넷으로 가른다.

## 1. `atr` 인과성 ⭐최우선
배리어를 `entry ± 1.5*atr` 로 놓는데, 그 `atr` 이 미래를 담고 있으면 라벨 자체가 오염된다.
저장된 `atr` 을 **klines 에서 인과적으로 재계산**한 값과 대조한다
(tr = max(h-l, |h-c_prev|, |l-c_prev|), atr = tr.rolling(14).mean(), 봉 i 까지).
일치하지 않으면 나머지 셋은 무의미하다.

## 2. 기존 배포 메타라벨과 겹치는가
"새 발견"이 아니라 배포 칩의 재도출일 수 있다. 배포 라벨(L0=V자반등)로 학습한 모델과
내 first-touch 모델의 **예측 상관**, 그리고 L0 모델 대비 **짝비교 증분**을 본다.

## 3. 배리어 폭 가설 -- 앵커 축을 좁은 배리어로 재측정
앵커(any3/Wc3)에서 ±1% 대신 **±1.5×ATR** 로 바꾸면 0.53 이 0.65 로 오르는가.
오르면 "지평·모집단이 아니라 **배리어 폭**이 갈랐다"가 확정된다.

## 4. 청산 구조 -- 트레일링 vs 브래킷 짝비교
이 저장소의 기존 경제성 기각은 **트레일링** 기준이었다. 같은 진입에 두 청산을 걸어 비교한다
(트레일링: SL 5×ATR · ARM 1.5×ATR · trail 0.1×ATR, F0 셀 상속).
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
LAB = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/eth_5m_v_rebound_multitrigger_labels.csv"
FEA = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/eth_5m_v_rebound_multitrigger_features_tier0.csv"
ANC = ROOT / "tmp/eth_anchor_features154_20260907"
OUT = ROOT / "tmp/eth_short_horizon_audit_20260907"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H, K_BAR, COST = 6, 1.5, 7.8
SL_M, ARM_M, TRAIL_M = 5.0, 1.5, 0.1
SEED, N_EST, MAX_CTX = 20260907, 4, 12000


def split_of(ts):
    return np.where(ts < "2025-09-01", "TRAIN", np.where(ts < "2026-01-01", "VAL",
           np.where(ts < "2026-04-01", "OOS", "HOLDOUT_SPENT")))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    import importlib.util, torch
    from tabpfn import TabPFNClassifier
    s = importlib.util.spec_from_file_location("ab", ROOT / "scripts/build_eth_anchor_label_dataset_20260907.py")
    B = importlib.util.module_from_spec(s); s.loader.exec_module(B)
    s2 = importlib.util.spec_from_file_location("vr", ROOT / "scripts/live_eth_sweep_v_rebound_signal_20260829.py")
    VR = importlib.util.module_from_spec(s2); s2.loader.exec_module(VR)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    L = pd.read_csv(LAB); F = pd.read_csv(FEA)
    for d in (L, F):
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_localize(None)
    feats = [c for c in VR.FEATURES if c in F.columns]
    D = L.merge(F[["timestamp", "direction"] + feats], on=["timestamp", "direction"]).sort_values("timestamp").reset_index(drop=True)
    kl = B._load_kl(B.ETH_KL)
    pos = pd.Series(np.arange(len(kl)), index=pd.to_datetime(kl["timestamp"]))
    idx = pos.reindex(D["timestamp"]).to_numpy(); keep = ~pd.isna(idx)
    D = D[keep].reset_index(drop=True); idx = idx[keep].astype(int)
    OPk = kl["open"].to_numpy(float); HIk = kl["high"].to_numpy(float)
    LOk = kl["low"].to_numpy(float); CLk = kl["close"].to_numpy(float)

    # ── 1. atr 인과성
    print("=" * 100, flush=True)
    print("[1] atr 인과성 -- 저장값 vs klines 인과 재계산", flush=True)
    print("=" * 100, flush=True)
    pc = np.concatenate([[CLk[0]], CLk[:-1]])
    tr_ = np.maximum.reduce([HIk - LOk, np.abs(HIk - pc), np.abs(LOk - pc)])
    atr_causal_full = pd.Series(tr_).rolling(14, min_periods=14).mean().to_numpy()
    a_stored = D["atr"].to_numpy(float)
    for off, tag in ((0, "봉 i 까지(인과)"), (1, "봉 i+1 까지(미래 1봉)"), (-1, "봉 i-1 까지")):
        a_c = atr_causal_full[np.clip(idx + off, 0, len(kl) - 1)]
        m = np.isfinite(a_c) & np.isfinite(a_stored)
        rel = np.abs(a_c[m] - a_stored[m]) / np.maximum(a_stored[m], 1e-12)
        print(f"   {tag:<22} 최대상대오차 {rel.max():.3e} · 중앙 {np.median(rel):.3e} "
              f"· 1e-6 이내 비율 {(rel < 1e-6).mean():.4f}", flush=True)
    a_c0 = atr_causal_full[idx]
    m0 = np.isfinite(a_c0) & np.isfinite(a_stored)
    causal_ok = bool((np.abs(a_c0[m0] - a_stored[m0]) / np.maximum(a_stored[m0], 1e-12) < 1e-6).mean() > 0.99)
    print(f"   ⇒ {'✅인과적 -- 봉 i 까지의 값과 일치' if causal_ok else '🔴불일치 -- 아래 결과 무효'}", flush=True)

    # 라벨/피쳐
    atr = a_stored; isdn = (D["direction"].to_numpy() == "downside")
    ts = D["timestamp"]; sp = split_of(ts); day = ts.dt.floor("D").to_numpy()
    X = D[feats].to_numpy(np.float64)
    ft = np.full(len(D), np.nan); bp = np.full(len(D), np.nan)
    tout = np.full(len(D), np.nan); trail = np.full(len(D), np.nan)
    for j, i in enumerate(idx):
        a, b = i + 1, i + 1 + H
        if b > len(kl) or not np.isfinite(atr[j]) or atr[j] <= 0: continue
        e = OPk[a]; h = HIk[a:b]; l = LOk[a:b]
        bp[j] = K_BAR * atr[j] / e * 1e4
        ub, db = e + K_BAR * atr[j], e - K_BAR * atr[j]
        iu = np.argmax(h >= ub) if (h >= ub).any() else 10**9
        il = np.argmax(l <= db) if (l <= db).any() else 10**9
        if iu != 10**9 or il != 10**9:
            if iu != il:
                ft[j] = float((iu < il) if isdn[j] else (il < iu))
        else:
            tout[j] = (CLk[b - 1] - e) / e * 1e4 * (1 if isdn[j] else -1)
        # 4) 트레일링 (되돌림 방향 진입 기준, 같은 H 창)
        sgn = 1.0 if isdn[j] else -1.0
        fav = (h - e) * sgn if sgn > 0 else (e - l) * (-sgn)
        adv = (e - l) * sgn if sgn > 0 else (h - e) * (-sgn)
        stop = -SL_M * atr[j]; armed = False; peak = 0.0; px = None
        for t in range(len(h)):
            hi_m = (h[t] - e) * sgn; lo_m = (l[t] - e) * sgn
            if sgn < 0: hi_m, lo_m = (e - l[t]) * 1.0, (e - h[t]) * 1.0
            if lo_m <= stop: px = stop; break
            peak = max(peak, hi_m)
            if not armed and peak >= ARM_M * atr[j]: armed = True
            if armed:
                stop = max(stop, peak - TRAIL_M * atr[j])
                if lo_m <= stop: px = stop; break
        if px is None: px = (CLk[b - 1] - e) * sgn
        trail[j] = px / e * 1e4
    ok = np.isfinite(ft); trm_ = ok & (sp == "TRAIN")
    sel = rng.choice(np.flatnonzero(trm_), min(MAX_CTX, trm_.sum()), replace=False)
    trm = np.zeros(len(D), bool); trm[sel] = True

    def fit(y, mask):
        t = mask & (sp == "TRAIN")
        if t.sum() > MAX_CTX:
            s_ = rng.choice(np.flatnonzero(t), MAX_CTX, replace=False)
            t = np.zeros(len(D), bool); t[s_] = True
        c = TabPFNClassifier(device=dev, n_estimators=N_EST, random_state=SEED,
                             ignore_pretraining_limits=True, memory_saving_mode=True)
        c.fit(np.nan_to_num(X[t]).astype(np.float32), y[t].astype(int))
        p = np.full(len(D), np.nan)
        am = np.isfinite(X).all(axis=1)
        p[am] = c.predict_proba(np.nan_to_num(X[am]).astype(np.float32))[:, 1]
        return p

    P_ft = fit(ft, ok)
    L0 = (D["outcome"].to_numpy() == "V자반등").astype(float)
    P_l0 = fit(L0, np.isfinite(L0))

    print("\n" + "=" * 100, flush=True)
    print("[2] 기존 배포 라벨(L0=V자반등) 모델과 겹치는가", flush=True)
    print("=" * 100, flush=True)
    mm = np.isfinite(P_ft) & np.isfinite(P_l0)
    print(f"   예측 상관 pearson {np.corrcoef(P_ft[mm], P_l0[mm])[0,1]:+.4f} · "
          f"spearman {pd.Series(P_ft[mm]).corr(pd.Series(P_l0[mm]), method='spearman'):+.4f}", flush=True)
    for w in WINS:
        te = ok & (sp == w)
        if te.sum() < 40: continue
        a1 = roc_auc_score(ft[te].astype(int), P_ft[te]); a2 = roc_auc_score(ft[te].astype(int), P_l0[te])
        print(f"   {w:<14} first-touch 라벨에서  내 모델 {a1:.4f} · L0 모델 {a2:.4f} · 증분 {a1-a2:+.4f}", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("[3] 배리어 폭 가설 -- 앵커(any3/Wc3)를 ±1.5×ATR 로 재측정", flush=True)
    print("=" * 100, flush=True)
    AD = pd.read_parquet(ANC / "features154.parquet")
    meta = json.loads((ANC / "meta.json").read_text())
    acols = [c for c in meta["feature_cols"] if c in AD.columns]
    aidx = pos.reindex(AD["timestamp"]).to_numpy().astype(int)
    aatr = AD["atr_pct"].to_numpy(float)
    aside = AD["side"].to_numpy(); asp = AD["split"].to_numpy()
    for Ha, tag in ((6, "H=6봉(30분)"), (48, "H=48봉(4시간)")):
        yft = np.full(len(AD), np.nan)
        for j, i in enumerate(aidx):
            a, b = i + 1, i + 1 + Ha
            if b > len(kl) or not np.isfinite(aatr[j]) or aatr[j] <= 0: continue
            e = OPk[a]; ub = e * (1 + K_BAR * aatr[j]); db = e * (1 - K_BAR * aatr[j])
            h = HIk[a:b]; l = LOk[a:b]
            iu = np.argmax(h >= ub) if (h >= ub).any() else 10**9
            il = np.argmax(l <= db) if (l <= db).any() else 10**9
            if iu == 10**9 and il == 10**9 or iu == il: continue
            # 지속 = 천장앵커면 위 / 바닥앵커면 아래
            yft[j] = float((iu < il) if aside[j] == "top" else (il < iu))
        okA = np.isfinite(yft); tA = okA & (asp == "TRAIN")
        if tA.sum() < 300: print(f"   {tag}: TRAIN 부족"); continue
        XA = AD[acols].to_numpy(np.float64)
        c = TabPFNClassifier(device=dev, n_estimators=N_EST, random_state=SEED,
                             ignore_pretraining_limits=True, memory_saving_mode=True)
        c.fit(np.nan_to_num(XA[tA]).astype(np.float32), yft[tA].astype(int))
        line = f"   {tag} n={okA.sum():>5} 양성 {np.nanmean(yft[okA]):.3f} ·"
        aus = []
        for w in WINS:
            te = okA & (asp == w)
            if te.sum() < 40 or len(np.unique(yft[te])) < 2: continue
            a_ = roc_auc_score(yft[te].astype(int), c.predict_proba(np.nan_to_num(XA[te]).astype(np.float32))[:, 1])
            aus.append(a_); line += f" {w[:3]} {a_:.4f}"
        print(line + f"  mean {np.mean(aus):.4f}   (기존 ±1% H=48 측정 0.53~0.56)", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("[4] 청산 구조 -- 브래킷(±1.5ATR) vs 트레일링(SL5/ARM1.5/trail0.1)", flush=True)
    print("=" * 100, flush=True)
    am = np.isfinite(bp) & np.isfinite(P_ft)
    for q in (1.00, 0.30):
        for nm in ("브래킷", "트레일링"):
            line = f"   {nm:<7} 상위{q:.0%}"
            for w in WINS:
                m = am & (sp == w)
                if m.sum() < 50: continue
                p = P_ft[m]; k = max(10, int(len(p) * q)); s_ = np.argsort(-np.abs(p - 0.5))[:k]
                lr = p[s_] > 0.5
                if nm == "브래킷":
                    r_ft = ft[m][s_]; r_bp = bp[m][s_]; r_to = tout[m][s_]
                    g = np.where(np.isfinite(r_ft),
                                 np.where(np.where(lr, r_ft, 1 - r_ft) > 0.5, r_bp, -r_bp),
                                 np.where(lr, r_to, -r_to))
                else:
                    g = np.where(lr, trail[m][s_], -trail[m][s_])
                line += f"  {w[:3]} {np.nanmean(g) - COST:+.2f}bp"
            print(line, flush=True)
    json.dump({"atr_causal": causal_ok}, open(OUT / "audit.json", "w"), indent=1)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
