#!/usr/bin/env python3
"""B2: 진입 필터 피쳐 선별 -- 162개를 줄인다 (2026-09-03).

배경: 136 레짐패널을 넣으니 OOS 예측상관이 0.103 -> 0.204로 두 배가 됐다(펀딩 9 · OI 3 ·
BTC 교차자산 11 등 klines에 없는 정보). 다만 162피쳐 / TRAIN 16,826행은 과하고,
F만(136) arm의 VAL 0.344 vs OOS 0.205 격차가 과적합 신호다.

절차 (⚠️VAL을 선별에 쓰지 않는다)
--------------------------------
  1. **TRAIN 내부 분할**로 순열중요도 계산 (3시드 평균). VAL/OOS는 건드리지 않는다.
  2. 중요도 순으로 그리디 선택하되, 이미 선택된 피쳐와 |corr|>0.95면 **중복으로 버린다**.
  3. k ∈ {10,20,30,50,80,162} 스윕. **k는 VAL에서 고르고 OOS를 보고**한다.
  4. anti-stable 진단 -- 선택된 집합의 TRAIN-VAL 중요도 상관. SOL에서 −0.38이었던 그 지표.

이 저장소는 "TRAIN 상위 피쳐가 VAL에서 부호를 안 지킨다"로 여러 번 죽었다. 그래서 선별을
TRAIN 안에서만 하고, VAL 유지 여부는 **판정이 아니라 진단**으로 본다.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat, slot_sim  # noqa: E402
from research_eth_regime_s12k3_label_train_20260902 import GBM3_MODEL_PATH, load_frame  # noqa: E402

OUT = ROOT / "tmp/eth_entry_b2_featsel_20260903"
KS = [10, 20, 30, 50, 80, 162]
TAUS = [0.0005, 0.0010, 0.0020]
CORR_CUT = 0.95


def log(m): print(f"[b2] {m}", flush=True)


def main() -> int:
    d = pd.read_csv(ROOT / "tmp/eth_entry_b1_20260903/arm_rows.csv", parse_dates=["ts"])
    src = joblib.load(GBM3_MODEL_PATH); cols = src["feature_cols"]; med = src["feature_medians"]
    rf = load_frame()
    x = rf[["timestamp"] + cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med.get(c, 0.0))
    x = x.rename(columns={"timestamp": "ts"})
    dup = [c for c in cols if c in d.columns]
    x = x.rename(columns={c: c + "_r136" for c in dup})
    R136 = [(c + "_r136" if c in dup else c) for c in cols]
    d = d.merge(x, on="ts", how="left").dropna(subset=R136).reset_index(drop=True)

    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    FEATS = base + ["arm", "sig_id", "sig_dir", "atr"] + R136
    X = d[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (d.split == "TRAIN").to_numpy()
    X = X.fillna(X[tr].median())
    y = d["y"].to_numpy()
    log(f"행 {len(d):,} | TRAIN {int(tr.sum()):,} | 후보 피쳐 {len(FEATS)}")

    # ---- 1. TRAIN 내부 분할로 순열중요도 (VAL 미사용) ----
    idx = np.flatnonzero(tr)
    cut = int(len(idx) * 0.8)
    itr, iva = idx[:cut], idx[cut:]          # 시간순 내부 분할
    imps = []
    for s in SEEDS[:3]:
        m = HistGradientBoostingRegressor(random_state=s, **HP).fit(X.iloc[itr], y[itr])
        r = permutation_importance(m, X.iloc[iva], y[iva], n_repeats=3, random_state=s, n_jobs=-1)
        imps.append(r.importances_mean)
    imp = np.mean(imps, axis=0)
    stab = np.mean([np.argsort(np.argsort(-i)) for i in imps], axis=0)   # 평균 순위(낮을수록 상위)
    log(f"TRAIN 내부 분할 {len(itr):,}/{len(iva):,} · 순열중요도 3시드 완료")

    # ---- 2. 그리디 선택 + 중복 제거 ----
    order = np.argsort(-imp)
    C = X.iloc[itr].corr().abs().to_numpy()
    sel, dropped = [], 0
    for i in order:
        if len(sel) >= max(KS): break
        if any(C[i, j] > CORR_CUT for j in sel):
            dropped += 1; continue
        sel.append(int(i))
    log(f"중복(|r|>{CORR_CUT}) 제거 {dropped}개 → 선택 후보 {len(sel)}개")
    log(f"상위 15: {[FEATS[i] for i in sel[:15]]}")

    # ---- 3. k 스윕 ----
    log("\n=== k 스윕 (k는 VAL에서 고르고 OOS 보고) ===")
    print(f"{'k':>4s} {'상관 VAL/OOS':>17s} | " + " | ".join(f"τ={t*1e4:.0f}bp 1슬롯 VAL/OOS" for t in TAUS))
    rows = []
    for k in KS:
        fs = [FEATS[i] for i in sel[:k]]
        pr = np.mean([HistGradientBoostingRegressor(random_state=s, **HP)
                      .fit(X[fs][tr], y[tr]).predict(X[fs]) for s in SEEDS], axis=0)
        d["p"] = pr
        cv = float(np.corrcoef(d.loc[d.split == "VAL", "p"], d.loc[d.split == "VAL", "y"])[0, 1])
        co = float(np.corrcoef(d.loc[d.split == "OOS", "p"], d.loc[d.split == "OOS", "y"])[0, 1])
        row = {"k": k, "corr_val": round(cv, 4), "corr_oos": round(co, 4)}
        cells = []
        for tau in TAUS:
            o = []
            for wn in ("VAL", "OOS"):
                w = d[d.split == wn]
                v = slot_sim(w, (w.p > tau).to_numpy(), 1)
                _, m, _ = stat(v)
                row[f"{wn}_t{int(tau*1e4)}"] = round(m, 2); row[f"{wn}_t{int(tau*1e4)}_n"] = len(v)
                o.append(f"{m:+6.2f}(n{len(v)})")
            cells.append(" ".join(o))
        rows.append(row)
        print(f"{k:4d} {cv:+8.4f}/{co:+8.4f} | " + " | ".join(f"{c:>22s}" for c in cells))

    # ---- 4. anti-stable 진단 (선택집합, TRAIN vs VAL) ----
    r = pd.DataFrame(rows)
    bestk = int(r.loc[r[[c for c in r.columns if c.startswith("VAL_t")]].max(axis=1).idxmax(), "k"])
    fs = [FEATS[i] for i in sel[:bestk]]
    m = HistGradientBoostingRegressor(random_state=SEEDS[0], **HP).fit(X[fs][tr], y[tr])
    va = np.flatnonzero((d.split == "VAL").to_numpy())
    it = permutation_importance(m, X[fs].iloc[itr[-4000:]], y[itr[-4000:]], n_repeats=3, random_state=0, n_jobs=-1)
    iv = permutation_importance(m, X[fs].iloc[va[:4000]], y[va[:4000]], n_repeats=3, random_state=0, n_jobs=-1)
    corr = float(np.corrcoef(it.importances_mean, iv.importances_mean)[0, 1])
    log(f"\n=== anti-stable 진단 (k={bestk}) ===")
    log(f"  TRAIN-VAL 중요도 상관 {corr:+.4f}  (음수면 중단 사유. SOL에서 −0.38이었음)")
    log(f"  TRAIN 상위 10 중 VAL에서도 양의 중요도: "
        f"{int((iv.importances_mean[np.argsort(-it.importances_mean)[:10]]>0).sum())}/10")

    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "k_sweep.csv", index=False)
    pd.DataFrame({"feature": [FEATS[i] for i in sel], "imp": imp[sel],
                  "mean_rank": stab[sel]}).to_csv(OUT / "selected_features.csv", index=False)
    json.dump({"best_k_by_val": bestk, "corr_cut": CORR_CUT, "seeds": SEEDS,
               "selection": "TRAIN-internal permutation importance, VAL never used for selection",
               "antistable_corr": corr, "features": fs}, open(OUT / "b2_report.json", "w"), indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
