#!/usr/bin/env python3
"""페이드/지속 — **목적함수 직접 최적화 Tier 0** (2026-09-06). 사용자 승인 후 착수.

문헌: Deep Momentum Networks(arXiv 1904.04912) — 추세추정과 사이징을 동시에 학습하고
네트워크를 **샤프 직접 최적화**로 훈련. Slow Momentum with Fast Reversion(2105.13727)이
같은 프레임에 변화점탐지를 얹는다. 조사: docs/homer/external_literature_fade_vs_continuation_20260906.md

## 왜 Tier 0인가
설계를 지배하는 제약은 **유효표본 700(TRAIN)** 이다(평균 고유도 0.055). DMN 논문은 88개 선물 ×
수십 년 일봉으로 우리의 ~100배 규모다. 그래서 **파라미터 59개**(선형 1층)에서 시작한다.
이 크기여야 "분류 손실 vs 목적함수 손실"의 차이만 순수하게 분리된다 — 같은 선형 모델을
로지스틱으로 학습한 C1이 대조군이므로, 차이가 나오면 원인이 손실함수임이 확정된다.

## 모델
    position p = tanh(w·x + b) ∈ [-1, +1]      # w: 58, b: 1  -> 파라미터 59
    실현손익  pnl(p) = relu(p)·r_fade + relu(-p)·r_cont
      p=+1 -> 페이드 전량 · p=-1 -> **지속 전량(= cont_all 과 동일)** · p=0 -> 무포지션
      ⇒ 모델이 "항상 지속"을 **스스로 발견할 수 있다**(p≡-1). 그게 정답이면 그렇게 수렴해야 한다.
    손실 = -Sharpe( 일별 집계 pnl ),  거래별 기여에 **고유도 가중** 적용(AFML Ch.4)

## 훈련 (체크리스트에서 이 실패모드에 맞는 것만)
  전배치 Adam(lr 1e-2, wd 1e-3) · 최대 500 epoch · **VAL 샤프 기준 조기종료(patience 20)**
  ⭐**전 에폭 train/val 곡선 전량 로깅**(규칙: 곡선 모양이 과적합/LR불안정/검증잡음을 가른다)
  5시드 앙상블 · 결측은 TRAIN 중앙값 대체 · 표준화는 TRAIN에서만 적합

## 대조군 (필수)
  C1 같은 선형 × **로지스틱 손실**(y_dec) -> position = 2·prob-1   [손실함수 효과만 분리]
  C2 방향 뒤집기 (r_fade <-> r_cont 교환)
  C3 라벨 셔플 귀무 20회 (TRAIN에서 (r_fade,r_cont) 쌍을 일 내 셔플)

## 사전 판정 기준 (결과 보기 전 고정)
  VAL·OOS **두 창** 일손익 CI 하한 > 0  ∧  cont_all 대비 **일별 짝비교**도 두 창 CI 하한 > 0.
  아니면 기각. (샤프·AUC 단독으로는 판정하지 않는다.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "tmp/eth_fade_vs_cont_dataset_20260906/dataset.parquet"
OUT = ROOT / "tmp/eth_fade_cont_sharpe_t0_20260906"
SEEDS = [11, 23, 47, 71, 97]
EPOCHS, PATIENCE, LR, WD = 500, 20, 1e-2, 1e-3
FWD, N_NULL = 200, 20
NON = {"y_order", "y_dec", "margin", "pnl_fade", "pnl_cont", "cls3", "split", "timestamp", "pos"}
RNG = np.random.default_rng(20260906)
torch.set_num_threads(8)


def log(m): print(f"[t0] {m}", flush=True)


def avg_uniqueness(pos, fwd=FWD):
    p = np.asarray(pos); lo, hi = p.min(), p.max() + fwd + 1
    d = np.zeros(hi - lo + 2, dtype=np.int64)
    for s in p:
        d[s - lo] += 1; d[s - lo + fwd + 1] -= 1
    conc = np.cumsum(d)[: hi - lo + 1]
    return np.array([np.mean(1.0 / np.maximum(conc[s - lo: s - lo + fwd + 1], 1)) for s in p])


def day_ci(v, d, B=1500):
    ud = np.unique(d); idx = {x: np.flatnonzero(d == x) for x in ud}
    o = np.empty(B)
    for b in range(B):
        p = RNG.choice(ud, len(ud), replace=True)
        o[b] = np.concatenate([v[idx[x]] for x in p]).mean()
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def daily_sharpe(pnl: torch.Tensor, day_idx: torch.Tensor, n_days: int) -> torch.Tensor:
    d = torch.zeros(n_days, dtype=pnl.dtype).index_add_(0, day_idx, pnl)
    return d.mean() / (d.std() + 1e-8) * np.sqrt(365.0)


def fit_one(Xtr, rf_tr, rc_tr, dtr, ndtr, w_tr, Xva, rf_va, rc_va, dva, ndva, seed, mode="sharpe", ytr=None):
    torch.manual_seed(seed)
    lin = nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=LR, weight_decay=WD)
    bce = nn.BCEWithLogitsLoss(weight=w_tr.squeeze() if mode == "logit" else None)
    best, best_state, bad, curve = -1e9, None, 0, []
    for ep in range(EPOCHS):
        lin.train(); opt.zero_grad()
        z = lin(Xtr).squeeze(-1)
        if mode == "sharpe":
            p = torch.tanh(z)
            pnl = (torch.relu(p) * rf_tr + torch.relu(-p) * rc_tr) * w_tr.squeeze()
            loss = -daily_sharpe(pnl, dtr, ndtr)
        else:                                    # C1: 로지스틱 손실 (손실함수 효과 분리용)
            loss = bce(z, ytr)
        loss.backward(); opt.step()
        lin.eval()
        with torch.no_grad():
            ptr = torch.tanh(lin(Xtr).squeeze(-1))
            s_tr = daily_sharpe(torch.relu(ptr) * rf_tr + torch.relu(-ptr) * rc_tr, dtr, ndtr).item()
            pva = torch.tanh(lin(Xva).squeeze(-1))
            s_va = daily_sharpe(torch.relu(pva) * rf_va + torch.relu(-pva) * rc_va, dva, ndva).item()
        curve.append({"epoch": ep, "loss": float(loss.item()), "train_sharpe": s_tr, "val_sharpe": s_va})
        if s_va > best + 1e-6:
            best, bad = s_va, 0
            best_state = {k: v.clone() for k, v in lin.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    lin.load_state_dict(best_state)
    return lin, curve, best


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(DS).reset_index(drop=True)
    feats = [c for c in D.columns if c not in NON]
    sp = D["split"].to_numpy(); tr, va, oo = sp == "TRAIN", sp == "VAL", sp == "OOS"
    ts = pd.to_datetime(D["timestamp"].to_numpy())
    rf, rc = D["pnl_fade"].to_numpy(np.float32), D["pnl_cont"].to_numpy(np.float32)
    med = D.loc[tr, feats].median()
    X = D[feats].fillna(med).to_numpy(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    X = (X - mu) / sd
    u = np.zeros(len(D), np.float32); u[tr] = avg_uniqueness(D.loc[tr, "pos"].to_numpy())
    u[tr] /= u[tr].mean()
    day = pd.Series(ts).dt.floor("D")
    packs = {}
    for w, m in (("TRAIN", tr), ("VAL", va), ("OOS", oo)):
        di, uniq = pd.factorize(day[m], sort=True)
        packs[w] = dict(X=torch.tensor(X[m]), rf=torch.tensor(rf[m]), rc=torch.tensor(rc[m]),
                        d=torch.tensor(di), nd=len(uniq), w=torch.tensor(u[m] if w == "TRAIN" else np.ones(m.sum(), np.float32)),
                        y=torch.tensor(D.loc[m, "y_dec"].to_numpy(np.float32)), mask=m, days=day[m].to_numpy())
    log(f"TRAIN {tr.sum():,}/{packs['TRAIN']['nd']}일 · VAL {va.sum():,}/{packs['VAL']['nd']}일 · OOS {oo.sum():,}/{packs['OOS']['nd']}일 · 피쳐 {len(feats)}")
    T, V, O = packs["TRAIN"], packs["VAL"], packs["OOS"]

    def evaluate(models, tag, flip=False):
        res = {"arm": tag}
        for w, P in (("VAL", V), ("OOS", O)):
            RF, RC = (P["rc"], P["rf"]) if flip else (P["rf"], P["rc"])
            with torch.no_grad():
                p = torch.stack([torch.tanh(m(P["X"]).squeeze(-1)) for m in models]).mean(0)
            pnl = (torch.relu(p) * RF + torch.relu(-p) * RC).numpy()
            base = RC.numpy()                                     # cont_all
            dd = pd.Series(pnl).groupby(P["days"]).sum()
            res[w] = {"sharpe": float(dd.mean() / dd.std() * np.sqrt(365)) if dd.std() > 0 else np.nan,
                      "bp_trade": float(pnl.mean()),
                      "ci_bp": list(day_ci(pnl, P["days"])),
                      "vs_cont_bp": float((pnl - base).mean()),
                      "vs_cont_ci": list(day_ci(pnl - base, P["days"])),
                      "frac_fade": float((p.numpy() > 0).mean()), "mean_pos": float(p.numpy().mean())}
        res["pass"] = bool(res["VAL"]["ci_bp"][0] > 0 and res["OOS"]["ci_bp"][0] > 0
                           and res["VAL"]["vs_cont_ci"][0] > 0 and res["OOS"]["vs_cont_ci"][0] > 0)
        return res

    out = {}
    # --- T0: 샤프 목적함수 ---
    models, curves = [], []
    for sd_ in SEEDS:
        m, c, bs = fit_one(T["X"], T["rf"], T["rc"], T["d"], T["nd"], T["w"],
                           V["X"], V["rf"], V["rc"], V["d"], V["nd"], sd_)
        models.append(m); curves.append(c)
        log(f"  seed {sd_}: {len(c)} epoch, best VAL 샤프 {bs:.3f}, 마지막 TRAIN 샤프 {c[-1]['train_sharpe']:.3f}")
    json.dump(curves, open(OUT / "epoch_curves.json", "w"))
    out["T0 샤프 목적함수"] = evaluate(models, "T0")
    out["C2 방향뒤집기"] = evaluate(models, "C2", flip=True)
    # --- C1: 로지스틱 손실 (같은 선형 모델) ---
    lmodels = []
    for sd_ in SEEDS:
        m, c, _ = fit_one(T["X"], T["rf"], T["rc"], T["d"], T["nd"], T["w"],
                          V["X"], V["rf"], V["rc"], V["d"], V["nd"], sd_, mode="logit", ytr=T["y"])
        lmodels.append(m)
    out["C1 로지스틱 손실"] = evaluate(lmodels, "C1")
    # --- 기준: 항상 지속 ---
    for w, P in (("VAL", V), ("OOS", O)):
        base = P["rc"].numpy()
        dd = pd.Series(base).groupby(P["days"]).sum()
        out.setdefault("cont_all 기준", {})[w] = {
            "sharpe": float(dd.mean() / dd.std() * np.sqrt(365)), "bp_trade": float(base.mean()),
            "ci_bp": list(day_ci(base, P["days"]))}
    # --- C3: 라벨 셔플 귀무 ---
    nulls = []
    for i in range(N_NULL):
        idx = np.arange(len(T["rf"]))
        for dv in np.unique(T["d"].numpy()):
            mm = np.flatnonzero(T["d"].numpy() == dv); idx[mm] = RNG.permutation(mm)
        idx = RNG.permutation(idx)
        m, _, _ = fit_one(T["X"], T["rf"][idx], T["rc"][idx], T["d"], T["nd"], T["w"],
                          V["X"], V["rf"], V["rc"], V["d"], V["nd"], SEEDS[0])
        with torch.no_grad():
            p = torch.tanh(m(O["X"]).squeeze(-1))
        pnl = (torch.relu(p) * O["rf"] + torch.relu(-p) * O["rc"]).numpy()
        nulls.append(float((pnl - O["rc"].numpy()).mean()))
    obs = out["T0 샤프 목적함수"]["OOS"]["vs_cont_bp"]
    out["C3 셔플귀무"] = {"n": len(nulls), "mean": float(np.mean(nulls)), "p95": float(np.percentile(nulls, 95)),
                       "max": float(np.max(nulls)), "obs": obs,
                       "obs_pctile": float((np.array(nulls) < obs).mean() * 100)}
    (OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))

    print("\n" + "=" * 104)
    print(f"{'팔':<20}{'창':<5}{'샤프':>8}{'건당bp':>9}{'일CI95':>24}{'cont대비':>10}{'짝비교 CI95':>24}{'페이드%':>8}")
    for k, r in out.items():
        if k.startswith("C3"):
            continue
        for w in ("VAL", "OOS"):
            if w not in r:
                continue
            v = r[w]
            vc = "[%+.2f, %+.2f]" % tuple(v["vs_cont_ci"]) if "vs_cont_ci" in v else ""
            ci = "[%+.2f, %+.2f]" % tuple(v["ci_bp"])
            fr = v.get("frac_fade", float("nan")) * 100
            print(f"{k:<20}{w:<5}{v['sharpe']:>8.2f}{v['bp_trade']:>9.2f}{ci:>24}"
                  f"{v.get('vs_cont_bp', float('nan')):>10.2f}{vc:>24}{fr:>7.1f}%")
    c3 = out["C3 셔플귀무"]
    print(f"\nC3 셔플귀무 {c3['n']}회: 평균 {c3['mean']:+.2f} · p95 {c3['p95']:+.2f} · 최대 {c3['max']:+.2f}"
          f"  -> 관측 {c3['obs']:+.2f} 은 {c3['obs_pctile']:.0f} 백분위")
    print(f"\n사전 판정 통과: {[k for k,r in out.items() if isinstance(r,dict) and r.get('pass')]}")
    print(f"\n산출물: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
