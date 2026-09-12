#!/usr/bin/env python3
"""배포 구성(MASHT·±1%)에서 **H 만 바꿔** 재학습 (2026-09-08, 서버 GPU).

사용자: *"배포(MASHT, ±1%, H=48, 워크포워드 풀링)에서 h를 60분으로 해서 라벨을 다시 만들어서
       재학습하면 어떻게 되지?"*

## 왜 재계산이 필요 없나
`build_eth_anchor_direction_labels_20260907.py` 가 **1분봉 first-touch 시각(분)** 을 이미
저장했다(`hit_cont_min_P1` / `hit_fade_min_P1`). H 를 바꾸는 것은 그 시각을 **잘라내는 것**뿐이다:
    y(H) = 1  if hit_cont < hit_fade  and  hit_cont <= H*5분      (지속)
         = 0  if hit_fade < hit_cont  and  hit_fade <= H*5분      (되돌림)
         = NaN 둘 다 H 안에 없으면(미해소)
라벨 정의·배리어(±1%)·1분봉 해상도·진입(open[t+1])은 **배포와 완전히 동일**하고 H 만 다르다.

## 격자 (사전 지정)
H ∈ {12(60분), 24(2시간), 48(4시간, 배포)} × 팔 wbin(배포 팔)
피쳐 = MASHT 2,784 (동결 변환 아님 -- 각 실행에서 TRAIN fit)
프로토콜 = **월 1회 재학습 walk-forward** (배포 측정과 동일, 엠바고 4h)

## 판정 -- 배포 잣대 그대로
세 창 풀링 AUC · **상위 30% 진입 정확도**(일군집 CI) · 날블록 귀무 · 비용 후 건당 bp.
±1% 배리어·왕복 7.8bp 이므로 **손익분기 정확도 53.90%** 는 H 와 무관하게 동일하다.
⚠️H 를 줄이면 미해소가 늘어 n 이 준다 -- 해소율을 함께 보고한다.
🔴미해소는 경제성에서 **반드시 포함**한다(시간청산 = 총수익 0 − 비용, 보수적).
   부록 W 에서 미해소를 빼고 세어 +22bp -> +13bp 로 정정했는데, 이 스크립트 1차판이
   docstring 에만 적고 코드에 반영하지 않아 **같은 실수를 반복했다**(H=12 는 미해소가 57%).
   해소분 기준과 전체 기준을 **둘 다** 낸다.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "tmp/eth_anchor_window_tensor_20260907"
DIR = ROOT / "tmp/eth_anchor_direction_labels_20260907/direction_labels.parquet"
OUT = ROOT / "tmp/eth_masht_horizon_recut_20260908"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H_GRID = (12, 24, 48)
COST, BARRIER_BP = 7.8, 100.0
EMBARGO = pd.Timedelta(hours=4)
SEED, N_EST, BOOT = 20260907, 4, 1200


def day_ci(fn, d, rng, B=BOOT):
    u = np.unique(d)
    if len(u) < 5: return (np.nan, np.nan)
    idx = {x: np.flatnonzero(d == x) for x in u}; o = []
    for _ in range(B):
        i = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        v = fn(i)
        if v is not None: o.append(v)
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
    import torch
    from tabpfn import TabPFNClassifier
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    M = np.load(WT / "masht.npy").astype(np.float32)
    v = np.load(WT / "valid.npy")
    I = pd.read_parquet(WT / "index.parquet")[v].reset_index(drop=True)
    L = pd.read_parquet(DIR)
    L["timestamp"] = pd.to_datetime(L["timestamp"], utc=True).dt.tz_localize(None) \
        if L["timestamp"].dt.tz is not None else pd.to_datetime(L["timestamp"])
    if "anchor" in L.columns:
        L = L[L["anchor"] == "any3/Wc3"]
    # ⚠️앵커의 키는 **(timestamp, side)** 다. timestamp 만으로 dedup 하면 같은 봉의 bottom/top
    #   중 하나가 통째로 사라진다 -- 1차 실행이 그래서 4,743 중 2,634 만 조인됐다(사용자 지적).
    J = I[["timestamp", "side", "split"]].merge(
        L[["timestamp", "side", "hit_cont_min_P1", "hit_fade_min_P1"]],
        on=["timestamp", "side"], how="left")
    assert len(J) == len(I), "조인 후 행수 변화"
    # 조인 자체의 성공은 "L 에 그 (timestamp, side) 행이 있었나"로 본다.
    # 터치시각이 NaN 인 것은 조인 실패가 아니라 **라벨 빌더 최대 지평 안에 양쪽 다 안 닿은**
    # 정당한 미해소다(약 11%).
    L_key = set(map(tuple, L[["timestamp", "side"]].to_numpy()))
    hit_key = np.array([tuple(x) in L_key for x in I[["timestamp", "side"]].to_numpy()])
    hc = J["hit_cont_min_P1"].to_numpy(float); hf = J["hit_fade_min_P1"].to_numpy(float)
    ts = I["timestamp"]; sp = I["split"].to_numpy(); day = ts.dt.floor("D").to_numpy()
    resolved = int((np.isfinite(hc) | np.isfinite(hf)).sum())
    print(f"[입력] MASHT {M.shape} · (timestamp,side) 조인 {hit_key.sum():,}/{len(J):,} "
          f"({hit_key.mean():.1%}) · 그중 터치 있음 {resolved:,} ({resolved/len(J):.1%}) · {dev}", flush=True)
    assert hit_key.mean() > 0.95, f"조인률 {hit_key.mean():.1%} -- 키가 틀렸다"

    months, cur = [], pd.Timestamp("2025-09-01")
    while cur <= ts.max():
        months.append(cur); cur = cur + pd.offsets.MonthBegin(1)

    rows = []
    for H in H_GRID:
        lim = H * 5.0
        c_ok = np.isfinite(hc) & (hc <= lim); f_ok = np.isfinite(hf) & (hf <= lim)
        y = np.full(len(J), np.nan)
        y[c_ok & (~f_ok | (hc < hf))] = 1.0                 # 지속 먼저
        y[f_ok & (~c_ok | (hf < hc))] = 0.0                 # 되돌림 먼저
        tie = c_ok & f_ok & (hc == hf)
        y[tie] = np.nan
        mask = np.isfinite(y)
        # 월간 재학습 walk-forward
        preds = np.full(len(J), np.nan)
        for m0 in months:
            m1 = m0 + pd.offsets.MonthBegin(1)
            te = mask & (ts >= m0).to_numpy() & (ts < m1).to_numpy()
            tr = mask & (ts < (m0 - EMBARGO)).to_numpy()
            if te.sum() < 20 or tr.sum() < 300 or len(np.unique(y[tr])) < 2: continue
            c = TabPFNClassifier(device=dev, n_estimators=N_EST, random_state=SEED,
                                 ignore_pretraining_limits=True, memory_saving_mode=True)
            c.fit(M[tr], y[tr].astype(int))
            preds[te] = c.predict_proba(M[te])[:, 1]
        s = mask & np.isfinite(preds) & np.isin(sp, WINS)
        yy = y[s].astype(int); pv = preds[s]; dv = day[s]
        auc = roc_auc_score(yy, pv)
        alo, _ = day_ci(lambda i: roc_auc_score(yy[i], pv[i]) if len(np.unique(yy[i])) > 1 else None, dv, rng)
        k = max(10, int(len(pv) * 0.30)); top = np.argsort(-pv)[:k]
        acc = float(yy[top].mean())
        def _acc(i):
            kk = max(10, int(len(i) * 0.30))
            return float(yy[i][np.argsort(-pv[i])[:kk]].mean())
        clo, chi = day_ci(_acc, dv, rng)
        # 🔴경제성 -- **미해소를 반드시 포함**한다.
        # 부록 W 에서 미해소를 빼고 세어 +22bp -> +13bp 로 정정한 적이 있는데,
        # 이 스크립트 1차판이 docstring 에만 적고 코드에는 반영하지 않아 **같은 실수를 반복**했다.
        # H 를 줄이면 미해소가 급증하므로(H=12 는 57%) 이 차이가 결정적이다.
        n_all = int(mask.sum()); n_res = int(s.sum())
        ev_res = acc * BARRIER_BP - (1 - acc) * BARRIER_BP - COST      # 해소분만(참고)
        # 전체 앵커 대비: 해소 안 된 것은 시간청산(배리어 미도달 -> 총수익 0 가정, 보수적)
        n_total_anchor = int(np.isfinite(hc).sum() | np.isfinite(hf).sum()) if False else int(len(J))
        resolve_frac = n_all / max(len(J), 1)
        ev_all = ev_res * resolve_frac + (0.0 - COST) * (1 - resolve_frac)
        # 귀무
        nl = []
        for _ in range(8):
            yt = y.copy()
            trm = mask & (ts < pd.Timestamp("2025-09-01")).to_numpy()
            yt[trm] = dayperm(y[trm], day[trm], rng)
            c = TabPFNClassifier(device=dev, n_estimators=2, random_state=SEED,
                                 ignore_pretraining_limits=True, memory_saving_mode=True)
            c.fit(M[trm], yt[trm].astype(int))
            te = mask & np.isin(sp, WINS)
            nl.append(roc_auc_score(y[te].astype(int), c.predict_proba(M[te])[:, 1]))
        p95 = float(np.percentile(nl, 95)); pval = float(np.mean(np.array(nl) >= auc))
        rows.append({"H": H, "min": H * 5, "n_label": n_all, "n_eval": n_res,
                     "resolve_rate": float(np.isfinite(y[mask | ~mask]).mean()),
                     "pos_rate": float(np.nanmean(y[mask])), "pooled_auc": auc, "auc_lo": alo,
                     "top30_acc": acc, "acc_lo": clo, "acc_hi": chi,
                     "ev_bp_resolved": ev_res, "ev_bp_all": ev_all,
                     "resolve_frac": resolve_frac,
                     "null_p95": p95, "null_p": pval, "n_entries": int(k)})
        print(f"  H={H:>2}봉({H*5:>3}분) 라벨 {n_all:>5,} 평가 {n_res:>5,} 지속률 {np.nanmean(y[mask]):.3f} · "
              f"AUC {auc:.4f}[{alo:.3f}] · 상위30% {acc:.2%}[{clo:.2%},{chi:.2%}] "
              f"· 건당 해소분 {ev_res:+.1f}bp / **전체 {ev_all:+.1f}bp** "
              f"(해소율 {resolve_frac:.1%}) · 귀무 p={pval:.3f}", flush=True)

    A = pd.DataFrame(rows); A.to_csv(OUT / "recut.csv", index=False)
    BE = (BARRIER_BP + COST) / (2 * BARRIER_BP)
    print("\n" + "=" * 104, flush=True)
    print(f"손익분기 정확도 {BE:.2%} (±1% 배리어·왕복 {COST}bp, H 와 무관)", flush=True)
    for _, r in A.iterrows():
        ok = r.acc_lo > BE and r.pooled_auc > r.null_p95
        print(f"  H={int(r.H):>2}봉({int(r['min']):>3}분) 상위30% CI 하한 {r.acc_lo:.2%} vs 손익분기 {BE:.2%} "
              f"· 귀무 {'초과' if r.pooled_auc > r.null_p95 else '미달'} → {'✅' if ok else '❌'}", flush=True)
    b = A.loc[A.top30_acc.idxmax()]
    print(f"\n⭐최고: H={int(b.H)}봉({int(b['min'])}분) 상위30% {b.top30_acc:.2%} "
          f"(배포 H=48 은 59.88%)", flush=True)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
