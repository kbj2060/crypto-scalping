#!/usr/bin/env python3
"""최종 후보 5개를 **TabPFN** 으로 재학습·검정 (2026-09-08, 서버 GPU).

후보 5개 중 1·4번은 같은 모델(커버만 100/50 으로 다름)이라 실제 학습은 4개다:
  A  first_fire T=0.75 P=0.25%   -> 후보 1(커버50, 8.0건/일) · 후보 4(커버100, 22.1건/일)
  B  first_fire T=0.75 P=0.30%   -> 후보 2
  C  first_fire T=1.00 P=0.25%   -> 후보 3
  D  any3/Wc3   T=0.75 P=0.25%   -> 후보 5 (현 배포 앵커)

프로토콜은 HGB 후보표와 완전히 동일: 월간 재학습 워크포워드(4시간 엠바고) →
임계값은 TRAIN 앞70% 적합 → 뒤30% |p-0.5| 분위 → 커버 100%/50% 채점.
같은 실행 안에서 HGB 를 **대조**로 돌린다(데이터 처리 차이를 배제하기 위해 필수).

⚠️`bar_idx` 는 5분봉 배열 인덱스다. 서버 5분봉 파일이 로컬보다 3일 길어도 앞쪽 인덱스는
   같아야 한다 -- 어긋나면 조용히 틀린 라벨이 나오므로 n·돌파율을 로컬 실측치와 대조한다.

⚠️이 박스는 라이브 매매가 도는 공유 GPU다(과거 대시보드 타임아웃 유발 이력).
   한 번에 한 잡만, 시드 3개로 제한한다.
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

# (이름, 앵커, T, 배리어, 로컬 실측 n, 로컬 실측 돌파율) -- 뒤 둘은 정렬 검증용
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
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    from tabpfn import TabPFNClassifier
    import torch
    print(f"cuda {torch.cuda.is_available()} · {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}", flush=True)

    A = pd.read_parquet(MY / "dataset_v2.parquet")
    A["timestamp"] = pd.to_datetime(A["timestamp"])
    eth = B._load_kl(B.ETH_KL)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    print(f"5분봉 {len(ts5):,}행 · 1분봉 {len(ts1):,}행", flush=True)
    print("=" * 122, flush=True)

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
        clo = (C5[x5] - entry) / entry * 1e4 * sgn
        y = np.where(cont, 1, np.where(rev, 0, (clo > 0).astype(int)))

        n_got, up_got = int(okm.sum()), float(y[okm].mean())
        ok_align = (n_got == n_exp) and abs(up_got - up_exp) < 0.002
        print(f"\n### {tag} · {anch} T={TM} P={P*100:.2f}%  n={n_got:,}(기대 {n_exp:,}) "
              f"돌파율={up_got:.3f}(기대 {up_exp:.3f})  정렬 {'OK' if ok_align else '❌불일치'}", flush=True)
        if not ok_align:
            print("   ⚠️5분봉 인덱스 정렬 불일치 -- 이 셀은 건너뛴다(조용히 틀린 라벨 방지)", flush=True)
            continue

        ts = d["timestamp"]; sp = d["split"].to_numpy()
        months = ts.dt.to_period("M"); uniq = sorted(months.unique())
        base = [c for c in d.columns if c.startswith(("f_", "sig_", "v2_"))] + \
               [c for c in d.columns if c.startswith(("x_m_tkv", "x_m_ttp", "x_m_ttc", "x_m_retail"))] + \
               ["dir_up", "trig_min", "T_atr", "atr_at_anchor", "n_signals", "side_bottom"]
        X = np.nan_to_num(d[base].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        trm = sp == "TRAIN"; cut = ts[trm].quantile(0.7)
        tr_fit = np.flatnonzero(trm & (ts <= cut).to_numpy())
        tr_hold = trm & (ts > cut).to_numpy()

        def fit_pred(kind, itr, Xte):
            if kind == "hgb":
                c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                   l2_regularization=1.0, early_stopping=True,
                                                   validation_fraction=0.15, random_state=SEED)
                return c.fit(X[itr], y[itr]).predict_proba(Xte)[:, 1]
            ps = []
            for s in PFN_SEEDS:
                sel = itr if len(itr) <= SUB else np.random.default_rng(s).choice(itr, SUB, replace=False)
                c = TabPFNClassifier(device="cuda", random_state=s, ignore_pretraining_limits=True)
                ps.append(c.fit(X[sel], y[sel]).predict_proba(Xte)[:, 1])
            return np.mean(ps, axis=0)

        for kind in ("hgb", "tabpfn"):
            t0 = time.time()
            pred = np.full(len(y), np.nan)
            for i, mo in enumerate(uniq):
                if i < 5: continue
                te = (months == mo).to_numpy(); tr = (ts < ts[te].min() - EMB).to_numpy()
                if tr.sum() < 1500 or te.sum() < 30: continue
                pred[te] = fit_pred(kind, np.flatnonzero(tr), X[te])
            conf = np.abs(fit_pred(kind, tr_fit, X[tr_hold]) - 0.5)
            el = time.time() - t0
            for cv in COV:
                thr = 0.0 if cv >= 1.0 else float(np.quantile(conf, 1 - cv))
                rec = {"cand": tag, "anchor": anch, "T": TM, "P": P, "model": kind,
                       "cov": int(cv * 100), "sec": el}
                line = f"   {kind:>7} 커버{int(cv*100):>4}% | "
                for w in WINS:
                    m = np.isfinite(pred) & (sp == w) & okm & (np.abs(pred - 0.5) >= thr)
                    acc = float(((pred[m] > 0.5).astype(int) == y[m]).mean()) if m.sum() >= 30 else np.nan
                    auc = (float(roc_auc_score(y[m], pred[m]))
                           if m.sum() >= 30 and len(np.unique(y[m])) > 1 else np.nan)
                    rec[f"{w}_acc"] = acc; rec[f"{w}_auc"] = auc; rec[f"{w}_n"] = int(m.sum())
                    line += f"{w[:4]} {acc:.4f} A{auc:.3f} n{m.sum():>5} | "
                rows.append(rec); print(line + f"{el:.0f}초", flush=True)
            pd.DataFrame(rows).to_csv(MY / "h1h_tabpfn_candidates.csv", index=False)

    R = pd.DataFrame(rows)
    print("\n" + "=" * 122)
    print("TabPFN − HGB (같은 후보·같은 커버, +면 TabPFN 우세)")
    for tag in R["cand"].unique():
        for cv in (100, 50):
            a = R[(R["cand"] == tag) & (R["model"] == "tabpfn") & (R["cov"] == cv)]
            b = R[(R["cand"] == tag) & (R["model"] == "hgb") & (R["cov"] == cv)]
            if a.empty or b.empty: continue
            a, b = a.iloc[0], b.iloc[0]
            cv_a = " ".join(f"{w[:4]} 실현커버{a[w+'_n']/max(R[(R['cand']==tag)&(R['model']=='tabpfn')&(R['cov']==100)].iloc[0][w+'_n'],1)*100:>3.0f}%" for w in WINS) if cv == 50 else ""
            print(f"   {tag:>10} 커버{cv:>4}% | " +
                  " ".join(f"{w[:4]} {(a[w+'_acc']-b[w+'_acc'])*100:+6.2f}pp" for w in WINS) + "  " + cv_a)
    print(json.dumps({"done": True, "cells": len(R)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
