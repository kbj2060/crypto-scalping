#!/usr/bin/env python3
"""최종 후보 4구성을 **TabPFN 회귀**로 재학습·검정 (2026-09-08, 서버 GPU).

분류판(`..._tabpfn_candidates_20260908.py`)과 **평가 라벨 y 는 완전히 동일**하고 학습 타깃만 바꾼다.
  분류: 이진 y(돌파=1/되돌림=0) 를 배우고 p>0.5 로 판정
  회귀: 방향조정 수익률 clo(bp) 를 배우고 **부호**로 판정, 확신도는 |예측bp|
같은 y 로 채점하므로 두 결과는 직접 비교된다.

## 타깃 선택 근거
`clo = (close[bt+H] - entry)/entry * 1e4 * sgn` -- H=1시간 시점의 방향조정 수익률(bp).
배리어 해소 시점 수익률을 쓰면 해소된 사건은 사실상 ±P 두 값이라 회귀 타깃으로 퇴화한다.
clo 는 전 사건에서 연속이고, 타임아웃 사건에서는 라벨 정의와 부호가 정확히 일치한다.
TRAIN 1/99 분위로 윈저화한다(꼬리가 적합을 지배하는 것을 막되 정보는 남긴다).

## 회귀 고유 지표도 같이 낸다
부호 정확도만 보면 회귀를 분류로 축소해 버리므로, 예측-실현 스피어만 IC 를 병기한다.
"""
from __future__ import annotations
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

MY = ROOT / "tmp/eth_breakout_atr_state_20260908_s1"
KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
H = 12
COV = (1.0, 0.5)
EMB = pd.Timedelta(hours=4)
CHUNK, SEED, SUB = 4000, 20260908, 18000
PFN_SEEDS = [11, 23, 47]

CAND = [
    ("A 후보1·4", "first_fire", 0.75, 0.0025, 21719, 0.450),
    ("B 후보2",   "first_fire", 0.75, 0.0030, 21719, 0.472),
    ("C 후보3",   "first_fire", 1.00, 0.0025, 18157, 0.435),
    ("D 후보5",   "any3/Wc3",   0.75, 0.0025,  4568, 0.430),
]


def first_touch(hi1, lo1, start, up, dn, nmin):
    n = len(start)
    tu = np.full(n, -1, np.int32); td = np.full(n, -1, np.int32)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        idx = start[a:b, None] + np.arange(nmin)[None, :]
        hu = hi1[idx] >= up[a:b, None]; hd = lo1[idx] <= dn[a:b, None]
        au = hu.any(axis=1); ad = hd.any(axis=1)
        tu[a:b] = np.where(au, hu.argmax(axis=1), -1); td[a:b] = np.where(ad, hd.argmax(axis=1), -1)
    return tu, td


def main() -> int:
    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingRegressor
    from tabpfn import TabPFNRegressor
    import torch
    print(f"cuda {torch.cuda.is_available()} · "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}", flush=True)

    A = pd.read_parquet(MY / "dataset_v2.parquet")
    A["timestamp"] = pd.to_datetime(A["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    print("=" * 130, flush=True)

    rows = []
    for tag, anch, TM, P, n_exp, up_exp in CAND:
        d = A[(A.anchor == anch) & (A.T_mult == TM)].sort_values("timestamp").reset_index(drop=True)
        sgn = np.where(d["dir_up"].to_numpy() > 0, 1.0, -1.0)
        bi = d["bar_idx"].to_numpy(); T = d["T_atr"].to_numpy(float)
        entry = O5[np.minimum(bi + 1, len(O5) - 1)] * (1 + sgn * T)
        s1 = np.searchsorted(ts1, d["timestamp"].to_numpy())
        bt = np.searchsorted(ts5, ts1[np.clip(s1, 0, len(ts1) - 1)], side="right") - 1
        okm = (s1 > 0) & (s1 + H * 5 < len(ts1)) & (bt + H < len(C5)) & (bt >= 1)
        tu, td = first_touch(hi1, lo1, np.where(okm, s1, 0), entry * (1 + P), entry * (1 - P), H * 5)
        big = 1 << 30
        uo, do_ = tu >= 0, td >= 0
        au, ad = np.where(uo, tu, big), np.where(do_, td, big)
        cont = np.where(sgn > 0, uo & (au < ad), do_ & (ad < au))
        rev = np.where(sgn > 0, do_ & (ad < au), uo & (au < ad))
        x5 = np.minimum(bt + H, len(C5) - 1)
        clo = (C5[x5] - entry) / entry * 1e4 * sgn          # ⭐회귀 타깃 (bp)
        y = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))   # ⭐평가 라벨(분류판과 동일)

        n_got, up_got = int(okm.sum()), float(y[okm].mean())
        ok_align = (n_got == n_exp) and abs(up_got - up_exp) < 0.002
        print(f"\n### {tag} · {anch} T={TM} P={P*100:.2f}%  n={n_got:,} 돌파율={up_got:.3f}  "
              f"정렬 {'OK' if ok_align else '❌불일치'}", flush=True)
        if not ok_align:
            print("   ⚠️정렬 불일치 -- 건너뜀", flush=True); continue

        ts = d["timestamp"]; sp = d["split"].to_numpy()
        months = ts.dt.to_period("M"); uniq = sorted(months.unique())
        base = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
               [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] + \
               ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
        X = np.nan_to_num(d[base].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        trm = sp == "TRAIN"; cut = ts[trm].quantile(0.7)
        tr_fit = np.flatnonzero(trm & (ts <= cut).to_numpy())
        tr_hold = trm & (ts > cut).to_numpy()
        lo_w, hi_w = np.percentile(clo[trm & okm], [1, 99])
        print(f"   타깃 clo(bp): TRAIN 중앙 {np.median(clo[trm&okm]):+.1f} · "
              f"윈저 [{lo_w:+.0f}, {hi_w:+.0f}] · 표준편차 {clo[trm&okm].std():.0f}", flush=True)

        def fit_pred(kind, itr, Xte):
            tgt = np.clip(clo[itr], lo_w, hi_w)
            if kind == "hgbr":
                m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                  l2_regularization=1.0, early_stopping=True,
                                                  validation_fraction=0.15, random_state=SEED)
                return m.fit(X[itr], tgt).predict(Xte)
            ps = []
            for s in PFN_SEEDS:
                if len(itr) <= SUB:
                    sel = np.arange(len(itr))
                else:
                    sel = np.random.default_rng(s).choice(len(itr), SUB, replace=False)
                m = TabPFNRegressor(device="cuda", random_state=s, ignore_pretraining_limits=True)
                ps.append(m.fit(X[itr[sel]], tgt[sel]).predict(Xte))
            return np.mean(ps, axis=0)

        for kind in ("hgbr", "tabpfn_reg"):
            t0 = time.time()
            pred = np.full(len(y), np.nan)
            for i, mo in enumerate(uniq):
                if i < 5: continue
                te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
                if tr.sum() < 1500 or te.sum() < 30: continue
                pred[te] = fit_pred(kind, np.flatnonzero(tr), X[te])
            conf = np.abs(fit_pred(kind, tr_fit, X[tr_hold]))     # 확신도 = |예측 bp|
            el = time.time() - t0
            for cv in COV:
                thr = -np.inf if cv >= 1.0 else float(np.quantile(conf, 1 - cv))
                rec = {"cand": tag, "anchor": anch, "T": TM, "P": P, "model": kind,
                       "cov": int(cv * 100), "sec": el}
                line = f"   {kind:>11} 커버{int(cv*100):>4}% | "
                for w in WINS:
                    m = np.isfinite(pred) & (sp == w) & okm & (np.abs(pred) >= thr)
                    acc = float(((pred[m] > 0).astype(int) == y[m]).mean()) if m.sum() >= 30 else np.nan
                    ic = (float(spearmanr(pred[m], clo[m]).statistic) if m.sum() >= 30 else np.nan)
                    rec[f"{w}_acc"] = acc; rec[f"{w}_ic"] = ic; rec[f"{w}_n"] = int(m.sum())
                    line += f"{w[:4]} {acc:.4f} IC{ic:+.3f} n{m.sum():>5} | "
                rows.append(rec); print(line + f"{el:.0f}초", flush=True)
            pd.DataFrame(rows).to_csv(MY / "h1h_tabpfn_regression.csv", index=False)

    R = pd.DataFrame(rows)
    print("\n" + "=" * 130)
    print("TabPFN회귀 − HGB회귀 (부호 정확도, +면 TabPFN 우세)")
    for tag in R["cand"].unique():
        for cv in (100, 50):
            a = R[(R["cand"] == tag) & (R["model"] == "tabpfn_reg") & (R["cov"] == cv)]
            b = R[(R["cand"] == tag) & (R["model"] == "hgbr") & (R["cov"] == cv)]
            if a.empty or b.empty: continue
            a, b = a.iloc[0], b.iloc[0]
            print(f"   {tag:>10} 커버{cv:>4}% | " +
                  " ".join(f"{w[:4]} {(a[w+'_acc']-b[w+'_acc'])*100:+6.2f}pp" for w in WINS))
    print(json.dumps({"done": True, "cells": len(R)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
