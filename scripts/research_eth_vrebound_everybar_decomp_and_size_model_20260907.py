#!/usr/bin/env python3
"""매봉 V자 분해 검증 + **크기 모델 동결** (2026-09-07, 서버 GPU).

사용자: *"1번으로 진행"* (전체 봉 모집단 분해 검증) →
       *"v자반등과 앵커방향을 방향과 크기 둘 다 대시보드에 표시 · 라벨과 확률까지 모두"*

## 두 가지를 한 번에
(A) **검증** -- 앞선 분해는 스윕 사건(14,259)에서 쟀다. 배포 칩은 멀티트리거 매봉 모집단
    (66,395행/64,956봉)을 쓴다. 라벨 공식은 동일(`fast_move_atr_mult>=1.5 within 30min AND
    giveback_ratio<=0.20 within 60min`, 라이브 스크립트 16행 "EXACT SAME v7b formula").
    같은 분해가 이 모집단에서도 나오는지 본다.
(B) **크기 모델 동결** -- 대시보드에 크기 확률을 띄우려면 실제 모델이 필요하다. 지금은 없다.
    라이브가 이미 계산하는 **TIER0+rsi 23피쳐**로 학습해 동결하면 추가 데이터 없이 서빙된다.

## 라벨 (klines 에서 재계산 -- 저장 파일은 행의 direction 쪽만 담아 크기를 못 구한다)
진입 = 그 봉의 종가. 이후 6봉(30분):
    up = max(high)/c0 - 1 · dn = 1 - min(low)/c0
    S  = max(up, dn)/atr >= 1.5            ← **크기**, 방향 무관 ⇒ 이 모델을 동결한다
    D  = (그 봉 direction 쪽이 더 멀리 갔나), S==1 인 행에서만   ← **방향**, 크기 통제
    L0 = 저장된 outcome == 'V자반등'        ← 배포 라벨(대조)
`atr` 은 그 봉까지의 값이라 미래를 안 본다.

## 산출
`data/live/vrebound_size_artifact/` (context.npz · meta.json) -- 라이브가 in-context 로 쓴다.
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
ART = ROOT / "data/live/vrebound_size_artifact"
OUT = ROOT / "tmp/eth_vrebound_everybar_decomp_20260907"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H, K = 6, 1.5                 # 30분 · 1.5×ATR (배포 라벨과 동일)
SEED, N_EST, BOOT = 20260907, 4, 800
MAX_CTX = 12000               # TabPFN 문맥 상한 (동결 아티팩트 크기 관리)


def day_ci(y, p, d, rng, B=BOOT):
    u = np.unique(d)
    if len(u) < 5: return (np.nan, np.nan)
    idx = {x: np.flatnonzero(d == x) for x in u}; o = []
    for _ in range(B):
        i = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        if len(np.unique(y[i])) > 1: o.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))) if len(o) > B // 3 else (np.nan, np.nan)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
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
    D = L.merge(F[["timestamp", "direction"] + feats], on=["timestamp", "direction"], how="inner")
    D = D.sort_values("timestamp").reset_index(drop=True)
    print(f"[1/5] 멀티트리거 매봉 {len(D):,} · 피쳐 {len(feats)}/{len(VR.FEATURES)} · {dev}", flush=True)

    kl = B._load_kl(B.ETH_KL)
    pos = pd.Series(np.arange(len(kl)), index=pd.to_datetime(kl["timestamp"]))
    idx = pos.reindex(D["timestamp"]).to_numpy()
    keep = ~pd.isna(idx)
    D = D[keep].reset_index(drop=True); idx = idx[keep].astype(int)
    C = kl["close"].to_numpy(float); HI = kl["high"].to_numpy(float); LO = kl["low"].to_numpy(float)
    atr = D["atr"].to_numpy(float)
    up = np.full(len(D), np.nan); dn = np.full(len(D), np.nan)
    for j, i in enumerate(idx):
        a, b = i + 1, i + 1 + H
        if b > len(kl) or not np.isfinite(atr[j]) or atr[j] <= 0: continue
        c0 = C[i]
        up[j] = (HI[a:b].max() / c0 - 1.0) * c0 / atr[j]      # atr 은 가격 단위
        dn[j] = (1.0 - LO[a:b].min() / c0) * c0 / atr[j]
    isdn = (D["direction"].to_numpy() == "downside")           # 하락스윕 → 되돌림은 위
    reb = np.where(isdn, up, dn); cont = np.where(isdn, dn, up)
    S = np.where(np.isfinite(reb) & np.isfinite(cont), (np.maximum(reb, cont) >= K).astype(float), np.nan)
    Dl = np.where(S == 1, (reb > cont).astype(float), np.nan)
    L0 = (D["outcome"].to_numpy() == "V자반등").astype(float)
    print(f"[2/5] 라벨 · S {np.nanmean(S):.4f} · D {np.nanmean(Dl):.4f} (S==1 {int(np.nansum(S)):,}) "
          f"· L0 {L0.mean():.4f}", flush=True)

    ts = D["timestamp"]
    sp = np.where(ts < "2025-09-01", "TRAIN", np.where(ts < "2026-01-01", "VAL",
         np.where(ts < "2026-04-01", "OOS", "HOLDOUT_SPENT")))
    day = ts.dt.floor("D").to_numpy()
    X = D[feats].to_numpy(np.float64)

    print("\n[3/5] 분해 — 같은 모집단·같은 라이브 피쳐, 라벨만 교체", flush=True)
    print("=" * 100, flush=True)
    res = {}
    for name, y, desc in [("L0", L0, "배포 라벨 (V자반등)"),
                          ("S", S, "⭐크기 (양방향 중 ≥1.5ATR)"),
                          ("D", Dl, "⭐방향 (크기 통제)")]:
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        tr = ok & (sp == "TRAIN")
        if tr.sum() > MAX_CTX:                       # 문맥 상한 -- 무작위 부분표집(고정 시드)
            sel = rng.choice(np.flatnonzero(tr), MAX_CTX, replace=False)
            trm = np.zeros(len(D), bool); trm[sel] = True
        else:
            trm = tr
        clf = TabPFNClassifier(device=dev, n_estimators=N_EST, random_state=SEED,
                               ignore_pretraining_limits=True, memory_saving_mode=True)
        clf.fit(np.nan_to_num(X[trm]).astype(np.float32), y[trm].astype(int))
        line = f"  {name:<3}{desc:<30}"; aucs = []
        for w in WINS:
            te = ok & (sp == w)
            if te.sum() < 40 or len(np.unique(y[te])) < 2: line += f"{'-':>18}"; continue
            p = clf.predict_proba(np.nan_to_num(X[te]).astype(np.float32))[:, 1]
            a = roc_auc_score(y[te].astype(int), p); lo, _ = day_ci(y[te].astype(int), p, day[te], rng)
            aucs.append(a); line += f"  {a:.4f}[{lo:.3f}]"
        res[name] = {"mean3": float(np.mean(aucs)), "n_train": int(trm.sum()),
                     "pos_rate": float(np.nanmean(y[ok]))}
        print(line + f"  mean3 {res[name]['mean3']:.4f} (n_tr {trm.sum():,}, 양성 {res[name]['pos_rate']:.3f})",
              flush=True)
        if name == "S":
            np.savez_compressed(ART / "context.npz",
                                X=np.nan_to_num(X[trm]).astype(np.float32), y=y[trm].astype(np.int8))
            size_clf_train = int(trm.sum())

    print("\n[4/5] " + "=" * 96, flush=True)
    print(f"  스윕 모집단(앞선 측정): 크기 0.7708 · 방향 0.5277", flush=True)
    print(f"  매봉 모집단(이번):      크기 {res['S']['mean3']:.4f} · 방향 {res['D']['mean3']:.4f}", flush=True)
    print(f"  ⇒ {'✅재현 -- 같은 결론' if res['D']['mean3'] < 0.56 <= res['S']['mean3'] else '⚠️불일치 -- 재검토 필요'}",
          flush=True)

    (ART / "meta.json").write_text(json.dumps({
        "rule_id": "vrebound_size_20260907",
        "what": "30분 안 max(up,dn)/atr >= 1.5 확률 (방향 무관)",
        "features": feats, "H_bars": H, "K_atr": K,
        "n_context": size_clf_train, "base_rate": res["S"]["pos_rate"],
        "measured": {"mean3_auc": res["S"]["mean3"],
                     "direction_mean3_auc": res["D"]["mean3"],
                     "deployed_label_mean3_auc": res["L0"]["mean3"],
                     "sweep_population_ref": {"size": 0.7708, "direction": 0.5277}},
        "n_estimators": N_EST, "seed": SEED,
        "note": "대시보드 크기 확률 표시용. 방향은 별개 모델(배포 V자 칩)이 담당한다.",
    }, indent=1, ensure_ascii=False))
    json.dump(res, open(OUT / "decomp.json", "w"), indent=1, ensure_ascii=False)
    sz = sum(f.stat().st_size for f in ART.iterdir()) / 1e6
    print(f"\n[5/5] 크기 모델 동결: {ART} ({sz:.1f}MB, 문맥 {size_clf_train:,})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
