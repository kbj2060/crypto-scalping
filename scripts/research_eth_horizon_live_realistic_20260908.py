#!/usr/bin/env python3
"""지평 선택 -- **라이브 현실 평가** (2026-09-08, 서버 GPU).

사용자: *"이전처럼 룩어헤드 문제가 있는건 아니지?"*

## 확인한 것
앵커 모집단은 인과적이다 -- `_dedup` 은 클러스터의 **첫** 발동을 남기며 앞으로만 진행하고
(`i - last > gap`), `_within` 은 pandas `rolling(W)` 로 뒤만 본다. L1 게이트도 통과했다.
MASHT 창은 `[t-47, t]` 이고 `path_atr[t]==0` 트립와이어로 확인했다.

## 🔴그러나 남은 구조적 문제 -- **해소 선택**
H 를 줄이면 라벨이 붙는 행이 줄어든다(H=12 는 38.1%, H=24 는 54.7%). 그리고
**어느 앵커에 라벨이 붙는지는 미래가 결정한다**(그 창 안에 배리어를 쳤는가).
`local_extreme` 과 구조가 같다 -- 다만 그것과 달리 **모집단(앵커)은 그대로**이고
**평가 대상만** 미래로 걸러진다.

라이브는 전건 진입하고 시간청산을 받는다. 그런데 지금까지의 AUC·적중률은 **해소된 행만**
보고 있다. H 가 짧을수록 이 괴리가 커진다(H=12 는 62% 를 안 보고 있다).

## 라이브 현실 평가
학습은 지금처럼 **해소된 행**으로 한다(배포자가 할 수 있는 최선). 그러나 **예측과 채점은
전 앵커**에 대고, 미해소는 **시간청산 실현손익**(H 봉 뒤 종가 기준, 진입 방향)으로 센다.
    라이브 건당 = 평균( 해소면 ±배리어bp, 미해소면 종가손익bp ) − 비용
비교: 해소분만 본 값 vs 라이브 현실 값. 갭이 크면 지금까지의 H 비교가 왜곡된 것이다.

⭐추가 진단: **상위30% 의 해소율이 전체와 다른가**. 모델이 해소되는 쪽을 골라내면
   시간청산이 줄어 유리하고, 반대면 불리하다. 이 편향은 해소분만 보면 안 보인다.
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
WT = ROOT / "tmp/eth_anchor_window_tensor_20260907"
DIR = ROOT / "tmp/eth_anchor_direction_labels_20260907/direction_labels.parquet"
OUT = ROOT / "tmp/eth_horizon_live_realistic_20260908"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H_GRID = (12, 24, 48)
COST, BARRIER_PCT = 7.8, 1.0
EMBARGO = pd.Timedelta(hours=4)
SEED, N_EST, BOOT = 20260907, 4, 1000


def boot_ci(fn, d, rng, B=BOOT):
    u = np.unique(d)
    if len(u) < 5: return (np.nan, np.nan)
    idx = {x: np.flatnonzero(d == x) for x in u}; o = []
    for _ in range(B):
        i = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        v = fn(i)
        if v is not None: o.append(v)
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))) if len(o) > B // 3 else (np.nan, np.nan)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    import importlib.util, torch
    from tabpfn import TabPFNClassifier
    s = importlib.util.spec_from_file_location("ab", ROOT / "scripts/build_eth_anchor_label_dataset_20260907.py")
    B = importlib.util.module_from_spec(s); s.loader.exec_module(B)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    M = np.load(WT / "masht.npy").astype(np.float32)
    v = np.load(WT / "valid.npy")
    I = pd.read_parquet(WT / "index.parquet")[v].reset_index(drop=True)
    L = pd.read_parquet(DIR)
    L["timestamp"] = pd.to_datetime(L["timestamp"])
    if "anchor" in L.columns:
        L = L[L["anchor"] == "any3/Wc3"]
    J = I[["timestamp", "side", "split"]].merge(
        L[["timestamp", "side", "hit_cont_min_P1", "hit_fade_min_P1"]], on=["timestamp", "side"], how="left")
    hc = J["hit_cont_min_P1"].to_numpy(float); hf = J["hit_fade_min_P1"].to_numpy(float)
    ts = I["timestamp"]; sp = I["split"].to_numpy(); day = ts.dt.floor("D").to_numpy()
    side = I["side"].to_numpy()

    kl = B._load_kl(B.ETH_KL)
    pos = pd.Series(np.arange(len(kl)), index=pd.to_datetime(kl["timestamp"]))
    idx = pos.reindex(ts).to_numpy().astype(int)
    OP = kl["open"].to_numpy(float); CL = kl["close"].to_numpy(float)
    print(f"[입력] 앵커 {len(I):,} · {dev}", flush=True)

    months, cur = [], pd.Timestamp("2025-09-01")
    while cur <= ts.max():
        months.append(cur); cur = cur + pd.offsets.MonthBegin(1)

    rows = []
    for H in H_GRID:
        lim = H * 5.0
        c_ok = np.isfinite(hc) & (hc <= lim); f_ok = np.isfinite(hf) & (hf <= lim)
        y = np.full(len(J), np.nan)
        y[c_ok & (~f_ok | (hc < hf))] = 1.0
        y[f_ok & (~c_ok | (hf < hc))] = 0.0
        y[c_ok & f_ok & (hc == hf)] = np.nan
        lab = np.isfinite(y)                       # 해소(라벨 있음)
        # 시간청산 실현손익(bp): 지속 방향 기준. 지속 = 천장이면 상승 · 바닥이면 하락
        tout = np.full(len(J), np.nan)
        for j, i in enumerate(idx):
            a, b = i + 1, i + H
            if b >= len(kl): continue
            e = OP[a]
            tout[j] = (CL[b] - e) / e * 1e4 * (1.0 if side[j] == "top" else -1.0)
        # 월간 재학습: 학습은 해소된 행으로, 예측은 **전 앵커**에
        preds = np.full(len(J), np.nan)
        for m0 in months:
            m1 = m0 + pd.offsets.MonthBegin(1)
            te = (ts >= m0).to_numpy() & (ts < m1).to_numpy() & np.isfinite(tout)
            tr = lab & (ts < (m0 - EMBARGO)).to_numpy()
            if te.sum() < 20 or tr.sum() < 300 or len(np.unique(y[tr])) < 2: continue
            c = TabPFNClassifier(device=dev, n_estimators=N_EST, random_state=SEED,
                                 ignore_pretraining_limits=True, memory_saving_mode=True)
            c.fit(M[tr], y[tr].astype(int))
            preds[te] = c.predict_proba(M[te])[:, 1]
        ev = np.isin(sp, WINS) & np.isfinite(preds) & np.isfinite(tout)
        pv = preds[ev]; dv = day[ev]; yv = y[ev]; lv = lab[ev]; tv = tout[ev]
        k = max(10, int(len(pv) * 0.30)); top = np.argsort(-pv)[:k]
        # (a) 해소분만 (지금까지의 잣대)
        sel_lab = top[lv[top]]
        acc_res = float(yv[sel_lab].mean()) if len(sel_lab) else np.nan
        ev_res = acc_res * 100 - (1 - acc_res) * 100 - COST
        # (b) 라이브 현실: 전 앵커, 미해소는 시간청산 실현손익
        gross = np.where(lv[top], np.where(yv[top] > 0.5, 100.0, -100.0), tv[top])
        net = gross - COST
        lo, hi = boot_ci(lambda i: float((np.where(lv[i][np.argsort(-pv[i])[:max(10, int(len(i)*0.30))]],
                          np.where(yv[i][np.argsort(-pv[i])[:max(10, int(len(i)*0.30))]] > 0.5, 100.0, -100.0),
                          tv[i][np.argsort(-pv[i])[:max(10, int(len(i)*0.30))]]) - COST).mean()), dv, rng)
        rows.append({"H": H, "min": H * 5, "n_all": int(ev.sum()), "n_entries": int(k),
                     "resolve_all": float(lv.mean()), "resolve_top30": float(lv[top].mean()),
                     "acc_resolved": acc_res, "ev_resolved": ev_res,
                     "ev_live": float(net.mean()), "ev_live_lo": lo, "ev_live_hi": hi,
                     "timeout_mean_bp": float(tv[top][~lv[top]].mean()) if (~lv[top]).any() else np.nan})
        print(f"  H={H:>2}봉({H*5:>3}분) 진입 {k:>4}/{ev.sum():>4} · 해소율 전체 {lv.mean():.1%} / "
              f"상위30% {lv[top].mean():.1%} · 해소분 적중 {acc_res:.2%}({ev_res:+.1f}bp) → "
              f"**라이브 {net.mean():+.2f}bp** [{lo:+.2f},{hi:+.2f}] "
              f"· 시간청산 평균 {rows[-1]['timeout_mean_bp']:+.1f}bp", flush=True)

    A = pd.DataFrame(rows); A.to_csv(OUT / "live_realistic.csv", index=False)
    print("\n" + "=" * 100, flush=True)
    print("해소분 잣대 vs 라이브 현실 -- 갭이 크면 지금까지의 H 비교가 왜곡된 것", flush=True)
    print("=" * 100, flush=True)
    for _, r in A.iterrows():
        print(f"  H={int(r.H):>2}봉  해소분 {r.ev_resolved:+6.1f}bp → 라이브 {r.ev_live:+6.2f}bp "
              f"(갭 {r.ev_live - r.ev_resolved:+.1f}bp) · 상위30% 해소율 {r.resolve_top30:.1%} "
              f"vs 전체 {r.resolve_all:.1%} ({'유리' if r.resolve_top30 > r.resolve_all else '불리'})", flush=True)
    b = A.loc[A.ev_live.idxmax()]
    print(f"\n⭐라이브 기준 최고: H={int(b.H)}봉({int(b['min'])}분) {b.ev_live:+.2f}bp "
          f"[{b.ev_live_lo:+.2f}, {b.ev_live_hi:+.2f}] → "
          f"{'✅CI 하한 > 0' if b.ev_live_lo > 0 else '❌CI 가 0 포함'}", flush=True)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
