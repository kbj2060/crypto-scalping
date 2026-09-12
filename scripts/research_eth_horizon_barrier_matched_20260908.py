#!/usr/bin/env python3
"""지평 × 배리어 **동시 조정** -- 라이브 현실 잣대 (2026-09-08, 서버 GPU).

사용자: *"horizon이 줄어들수록 배리어 간격을 줄여야 하는거 아니야?"*

## 왜 맞는 지적인가
부록 X 는 배리어를 ±1% 로 **고정한 채** H 만 줄였다. 그래서 해소율이 붕괴하고
(H=12 는 38.1%, 상위30% 는 17.1%) 시간청산 손실이 이득을 삼켰다(라이브 −3.56bp).
가격 이탈폭은 대략 √H 로 커지므로 H 를 1/4 로 줄이면 배리어도 1/2 로 줄여야 해소율이 유지된다.
    H=48 ±1.00%  ·  H=24 ±0.707%(≈0.75)  ·  H=12 ±0.500%
저장된 1분봉 터치 시각에 `P0.5`/`P0.75`/`P1` 이 모두 있어 재계산 없이 격자를 짤 수 있다.

## 🔴그러나 공짜가 아니다 -- 손익분기가 오른다
    ±1.00%(100bp) 손익분기 (100+7.8)/200 = **53.90%**
    ±0.75%( 75bp)            (75+7.8)/150 = **55.20%**
    ±0.50%( 50bp)            (50+7.8)/100 = **57.80%**
좁은 배리어 = 해소율↑(시간청산 노출↓) 이지만 = 건당 이익↓(정확도 문턱↑).
**어느 쪽이 이기는지는 라이브 건당 bp 로만 답할 수 있다.**

## 격자 (사전 지정 9셀)
H ∈ {12, 24, 48} × 배리어 ∈ {0.5%, 0.75%, 1.0%}
학습 = 해소된 행(배포자가 할 수 있는 최선) · 월 1회 재학습 walk-forward · 엠바고 4h
채점 = **전 앵커**, 미해소는 시간청산 실현손익(H봉 뒤 종가, 지속 방향)

## 보고 규칙 (부록 X4 신설)
모든 셀에 **커버리지(해소율)** 와 **그 커버리지가 결과로 정해졌는지** 를 함께 낸다.
헤드라인은 **라이브 건당 bp** 다 -- 해소분 수치는 참고로만 병기한다.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "tmp/eth_anchor_window_tensor_20260907"
DIR = ROOT / "tmp/eth_anchor_direction_labels_20260907/direction_labels.parquet"
OUT = ROOT / "tmp/eth_horizon_barrier_matched_20260908"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H_GRID = (12, 24, 48)
P_GRID = (0.5, 0.75, 1.0)
COST = 7.8
EMBARGO = pd.Timedelta(hours=4)
SEED, N_EST, BOOT = 20260907, 4, 1000


def boot_ci(vals_fn, d, rng, B=BOOT):
    u = np.unique(d)
    if len(u) < 5: return (np.nan, np.nan)
    idx = {x: np.flatnonzero(d == x) for x in u}; o = []
    for _ in range(B):
        i = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        v = vals_fn(i)
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
    L = pd.read_parquet(DIR); L["timestamp"] = pd.to_datetime(L["timestamp"])
    if "anchor" in L.columns:
        L = L[L["anchor"] == "any3/Wc3"]
    cols = ["timestamp", "side"] + [f"hit_{k}_min_P{p:g}" for p in P_GRID for k in ("cont", "fade")]
    cols = [c for c in cols if c in L.columns]
    J = I[["timestamp", "side", "split"]].merge(L[cols], on=["timestamp", "side"], how="left")
    assert len(J) == len(I)
    ts = I["timestamp"]; sp = I["split"].to_numpy(); day = ts.dt.floor("D").to_numpy()
    side = I["side"].to_numpy()
    kl = B._load_kl(B.ETH_KL)
    pos = pd.Series(np.arange(len(kl)), index=pd.to_datetime(kl["timestamp"]))
    idx = pos.reindex(ts).to_numpy().astype(int)
    OP = kl["open"].to_numpy(float); CL = kl["close"].to_numpy(float)
    print(f"[입력] 앵커 {len(I):,} · 배리어 {P_GRID} · {dev}", flush=True)

    months, cur = [], pd.Timestamp("2025-09-01")
    while cur <= ts.max():
        months.append(cur); cur = cur + pd.offsets.MonthBegin(1)
    # 시간청산 실현손익(지속 방향 기준) -- H 별로
    TOUT = {}
    for H in H_GRID:
        t = np.full(len(J), np.nan)
        for j, i in enumerate(idx):
            a, b = i + 1, i + H
            if b >= len(kl): continue
            e = OP[a]
            t[j] = (CL[b] - e) / e * 1e4 * (1.0 if side[j] == "top" else -1.0)
        TOUT[H] = t

    rows = []
    print("\n" + "=" * 114, flush=True)
    print(f"{'H':>3}{'배리어':>8}{'BE':>8}{'해소율 전체/상위30%':>20}{'해소분 적중':>12}{'해소분bp':>10}"
          f"{'  ⭐라이브 건당':>16}{'[CI]':>20}", flush=True)
    print("=" * 114, flush=True)
    for H in H_GRID:
        lim = H * 5.0
        for P in P_GRID:
            hc = J[f"hit_cont_min_P{P:g}"].to_numpy(float)
            hf = J[f"hit_fade_min_P{P:g}"].to_numpy(float)
            c_ok = np.isfinite(hc) & (hc <= lim); f_ok = np.isfinite(hf) & (hf <= lim)
            y = np.full(len(J), np.nan)
            y[c_ok & (~f_ok | (hc < hf))] = 1.0
            y[f_ok & (~c_ok | (hf < hc))] = 0.0
            y[c_ok & f_ok & (hc == hf)] = np.nan
            lab = np.isfinite(y); tout = TOUT[H]
            bar_bp = P * 100.0
            be = (bar_bp + COST) / (2 * bar_bp)
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
            if ev.sum() < 100: continue
            pv = preds[ev]; dv = day[ev]; yv = y[ev]; lv = lab[ev]; tv = tout[ev]
            k = max(10, int(len(pv) * 0.30)); top = np.argsort(-pv)[:k]
            sl = top[lv[top]]
            acc = float(yv[sl].mean()) if len(sl) else np.nan
            ev_res = acc * bar_bp - (1 - acc) * bar_bp - COST
            gross = np.where(lv[top], np.where(yv[top] > 0.5, bar_bp, -bar_bp), tv[top])
            live = float((gross - COST).mean())

            def _live(i):
                kk = max(10, int(len(i) * 0.30)); t2 = np.argsort(-pv[i])[:kk]
                g = np.where(lv[i][t2], np.where(yv[i][t2] > 0.5, bar_bp, -bar_bp), tv[i][t2])
                return float((g - COST).mean())
            lo, hi = boot_ci(_live, dv, rng)
            rows.append({"H": H, "P": P, "breakeven": be, "n_eval": int(ev.sum()), "n_entry": int(k),
                         "resolve_all": float(lv.mean()), "resolve_top30": float(lv[top].mean()),
                         "acc_resolved": acc, "ev_resolved": ev_res,
                         "ev_live": live, "lo": lo, "hi": hi})
            print(f"{H:>3}{f'±{P}%':>8}{be:>8.2%}"
                  f"{f'{lv.mean():.1%} / {lv[top].mean():.1%}':>20}"
                  f"{acc:>12.2%}{ev_res:>+10.1f}{live:>+16.2f}"
                  f"{f'[{lo:+.2f}, {hi:+.2f}]':>20}"
                  f"{'  ✅' if lo > 0 else ''}", flush=True)

    A = pd.DataFrame(rows); A.to_csv(OUT / "grid.csv", index=False)
    print("\n" + "=" * 114, flush=True)
    npass = int((A.lo > 0).sum())
    print(f"라이브 건당 CI 하한 > 0 인 셀: {npass}/{len(A)}", flush=True)
    b = A.loc[A.ev_live.idxmax()]
    print(f"⭐라이브 최고: H={int(b.H)}봉({int(b.H)*5}분) ±{b.P}% · {b.ev_live:+.2f}bp "
          f"[{b.lo:+.2f}, {b.hi:+.2f}] · 해소율 {b.resolve_all:.1%}(상위30% {b.resolve_top30:.1%}) "
          f"· 손익분기 {b.breakeven:.2%} vs 해소분 적중 {b.acc_resolved:.2%}", flush=True)
    print("\n⚠️해소율은 결과로 정해진다 -- 해소분 수치는 참고이고 판정은 라이브 건당 bp 로 한다.", flush=True)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
