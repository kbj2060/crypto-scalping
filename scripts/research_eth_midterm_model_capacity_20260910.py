#!/usr/bin/env python3
"""중기 저하는 모델 탓인가 표본 탓인가 (2026-09-10) -- 사용자: *"딥러닝 모델이라 학습을 중기 데이터로 하지 않아서 그런 거 아니야?"*

선행 비교(`research_eth_signals_midterm_timeframe_20260910.py`)는 각 TF 의 데이터로 **재학습**했지만
모델 용량·계열은 5분봉용 설정 하나를 그대로 썼다(HGB 250회·15잎·min_leaf 40). 그 지적을 세 축으로 검정한다.

축 A **용량·계열 튜닝**: 각 TF 에서 내부 검증창으로 HP 를 고른다.
   arms = 원본HGB · 튜닝HGB(격자) · 로지스틱(C 격자) · RandomForest · ExtraTrees · **TabPFN**(격리 venv 서브프로세스,
   배포판이 쓰는 계열이자 소표본 특화 -- 이 가설의 정확한 대조군).
축 B **중기 데이터 늘리기**: 2021-12~2023-12 아카이브(ETH `eth_5m_2021_2023_archive.csv` + BTC 재수집)를 붙여
   같은 TF 의 학습 행을 늘린다. 4h 는 513 → 약 1,200 행.
축 C **표본 수 일치**: 모든 TF·모든 계열을 같은 학습 행 수(513)로 맞춰 다시 잰다.

분할  TRAIN ≤2025-05-31 · INNER-VAL 2025-06-01~08-31(HP 선택 전용) · TEST 2025-09-01~ (선행 실험과 같은 TEST).
      HP 는 INNER-VAL 에서만 고르고, 고른 뒤 TRAIN+INNER-VAL 로 재적합해 TEST 를 한 번 본다.
라벨·피쳐는 선행 실험과 **동일**(`X.build_rows`, `X.FEATS`) -- 비교 가능성이 목적이다.
출력 tmp/eth_midterm_capacity_20260910/report.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_exit_synth_dataset_20260910 as BD  # noqa: E402
import research_eth_signals_midterm_timeframe_20260910 as M  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
import live_eth_extreme_detector_20260909 as X  # noqa: E402

OUT = ROOT / "tmp/eth_midterm_capacity_20260910"
HIST = ROOT / "tmp/eth_signal_midterm_20260910"
TFS = {"5m": 1, "15m": 3, "1h": 12, "4h": 48}
TRAIN_END, VAL_END = "2025-05-31 23:59", "2025-08-31 23:59"
SEEDS = (11, 22, 33)
MATCH_N = 513                      # 4h 전체 학습 행 수(선행 실험) -- 표본 일치 기준점
TABPFN_PY = Path("/tmp/tabpfn_venv/bin/python")
TABPFN_MAX_TRAIN = 4000            # CPU 예산 -- 넘으면 무작위 부분표집(그 사실을 기록한다)


def log(m):
    print(f"[cap {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- 데이터
def load_klines_extended() -> tuple[pd.DataFrame, pd.DataFrame]:
    """2021-12~ 아카이브를 현행 프레임 앞에 붙인다. 없으면 현행만."""
    eth, btc = BD.load_klines("eth", "ETHUSDT"), BD.load_klines("btc", "BTCUSDT")
    ea = ROOT / "data/eth_5m_2021_2023_archive.csv"
    ba = HIST / "btc_5m_2021_2023.parquet"
    if not (ea.exists() and ba.exists()):
        log("⚠️아카이브 없음 -- 축 B 생략")
        return eth, btc
    e0 = pd.read_csv(ea)
    e0["timestamp"] = pd.to_datetime(e0["open_time"], unit="ms")
    b0 = pd.read_parquet(ba)
    cols = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base", "quote_volume", "trades", "close_time"]
    out = []
    for old, new in ((e0, eth), (b0, btc)):
        old = old.copy()
        for c in ("open", "high", "low", "close", "volume", "taker_buy_base", "quote_volume"):
            old[c] = old[c].astype(float)
        old["trades"] = pd.to_numeric(old["trades"], errors="coerce")
        cat = pd.concat([old[cols], new[cols]], ignore_index=True)
        out.append(cat.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True))
    log(f"확장: eth {len(out[0]):,}봉 {out[0].timestamp.iloc[0]} → {out[0].timestamp.iloc[-1]}")
    return out[0], out[1]


def rows_for(eth: pd.DataFrame, btc: pd.DataFrame, mult: int) -> pd.DataFrame:
    e, b = M.resample(eth, mult), M.resample(btc, mult)
    A = X.build_rows(compute_signals(e, btc_df=b, funding_df=None), b, with_label=True)
    return A[A["_y"] >= 0].reset_index(drop=True)


# ---------------------------------------------------------------- 모델
def hgb(**kw):
    base = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                l2_regularization=1.0, early_stopping=False)
    base.update(kw)
    return lambda sd: HistGradientBoostingClassifier(random_state=sd, **base)


def logit(C):
    return lambda sd: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    LogisticRegression(C=C, max_iter=3000, random_state=sd))


def forest(kind, **kw):
    cls = RandomForestClassifier if kind == "rf" else ExtraTreesClassifier
    return lambda sd: make_pipeline(SimpleImputer(strategy="median"),
                                    cls(n_estimators=400, n_jobs=6, random_state=sd, **kw))


GRIDS = {
    "hgb": {f"hgb_it{it}_lf{lf}_ml{ml}": hgb(max_iter=it, max_leaf_nodes=lf, min_samples_leaf=ml, learning_rate=lr)
            for it, lf, ml, lr in [(250, 15, 40, 0.05), (100, 7, 60, 0.05), (400, 31, 20, 0.03),
                                   (60, 3, 100, 0.08), (150, 7, 100, 0.03)]},
    "logit": {f"logit_C{C}": logit(C) for C in (0.01, 0.1, 1.0)},
    "forest": {"rf_d6": forest("rf", max_depth=6, min_samples_leaf=20),
               "rf_full": forest("rf", min_samples_leaf=10),
               "et_d8": forest("et", max_depth=8, min_samples_leaf=20)},
}


def fit_auc(make, Xtr, ytr, Xev, yev) -> float:
    P = np.zeros(len(yev))
    for sd in SEEDS:
        P += make(sd).fit(Xtr, ytr).predict_proba(Xev)[:, 1] / len(SEEDS)
    return float(roc_auc_score(yev, P))


def tabpfn_auc(Xtr, ytr, Xev, yev, tag: str) -> dict | None:
    """격리 venv 에서 실행(공유 파이썬 환경을 건드리지 않는다)."""
    if not TABPFN_PY.exists():
        return None
    n = len(ytr)
    if n > TABPFN_MAX_TRAIN:
        idx = np.random.default_rng(0).choice(n, TABPFN_MAX_TRAIN, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    f = OUT / f"_tabpfn_{tag}.npz"
    np.savez(f, Xtr=np.nan_to_num(Xtr, nan=0.0), ytr=ytr, Xev=np.nan_to_num(Xev, nan=0.0))
    code = (f"import numpy as np;from tabpfn import TabPFNClassifier;d=np.load(r'{f}');"
            "m=TabPFNClassifier(device='cpu',random_state=11).fit(d['Xtr'],d['ytr']);"
            f"np.save(r'{f}.out.npy', m.predict_proba(d['Xev'])[:,1])")
    try:
        subprocess.run([str(TABPFN_PY), "-c", code], check=True, capture_output=True, timeout=1800)
        p = np.load(f"{f}.out.npy")
        return {"auc": float(roc_auc_score(yev, p)), "n_train_used": int(len(ytr)), "subsampled": bool(n > TABPFN_MAX_TRAIN)}
    except Exception as e:  # noqa: BLE001
        log(f"⚠️tabpfn {tag}: {type(e).__name__} {str(getattr(e, 'stderr', b''))[-200:]}")
        return None


def evaluate_tf(A: pd.DataFrame, tag: str, match_n: int | None = None) -> dict:
    ts = pd.to_datetime(A["_ts"])
    tr = (ts <= TRAIN_END).to_numpy()
    va = ((ts > TRAIN_END) & (ts <= VAL_END)).to_numpy()
    te = (ts > VAL_END).to_numpy()
    Xm = A[X.FEATS].to_numpy(np.float32); y = A["_y"].to_numpy()
    if match_n is not None:
        keep = np.random.default_rng(7).choice(np.flatnonzero(tr), min(match_n, int(tr.sum())), replace=False)
        tr = np.zeros(len(A), bool); tr[keep] = True
    if tr.sum() < 100 or va.sum() < 40 or te.sum() < 80:
        return {"error": "too_small", "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())}
    res = {"n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
           "base_rate_test": float(y[te].mean()), "families": {}}
    fit_all = tr | va                       # HP 선택 후 재적합에 쓰는 창
    for fam, grid in GRIDS.items():
        picks = {nm: fit_auc(mk, Xm[tr], y[tr], Xm[va], y[va]) for nm, mk in grid.items()}
        best = max(picks, key=picks.get)
        auc_te = fit_auc(grid[best], Xm[fit_all], y[fit_all], Xm[te], y[te])
        res["families"][fam] = {"picked": best, "val_auc": picks[best], "test_auc": auc_te, "val_grid": picks}
        log(f"  {tag:22s} {fam:7s} → {best:24s} VAL {picks[best]:.3f} TEST {auc_te:.3f}")
    tp_val = tabpfn_auc(Xm[tr], y[tr], Xm[va], y[va], f"{tag}_val")
    tp_te = tabpfn_auc(Xm[fit_all], y[fit_all], Xm[te], y[te], f"{tag}_test")
    if tp_te:
        res["families"]["tabpfn"] = {"picked": "tabpfn_default", "val_auc": (tp_val or {}).get("auc"),
                                     "test_auc": tp_te["auc"], **{k: v for k, v in tp_te.items() if k != "auc"}}
        log(f"  {tag:22s} tabpfn  → VAL {(tp_val or {}).get('auc')} TEST {tp_te['auc']:.3f} (학습 {tp_te['n_train_used']}행)")
    res["best_family"] = max(res["families"], key=lambda k: res["families"][k]["val_auc"] or 0)
    res["best_test_auc"] = res["families"][res["best_family"]]["test_auc"]
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    eth_x, btc_x = load_klines_extended()
    eth_c, btc_c = BD.load_klines("eth", "ETHUSDT"), BD.load_klines("btc", "BTCUSDT")
    rep = {"train_end": TRAIN_END, "val_end": VAL_END, "match_n": MATCH_N,
           "tabpfn_available": TABPFN_PY.exists(), "arms": {}}
    for tf, mult in TFS.items():
        A_cur = rows_for(eth_c, btc_c, mult)
        log(f"[{tf}] 현행 데이터 {len(A_cur):,}행")
        rep["arms"][f"{tf}|current"] = evaluate_tf(A_cur, f"{tf}|current")
        rep["arms"][f"{tf}|current|n{MATCH_N}"] = evaluate_tf(A_cur, f"{tf}|match{MATCH_N}", match_n=MATCH_N)
        if len(eth_x) > len(eth_c):
            A_ext = rows_for(eth_x, btc_x, mult)
            log(f"[{tf}] 확장 데이터 {len(A_ext):,}행")
            rep["arms"][f"{tf}|extended"] = evaluate_tf(A_ext, f"{tf}|extended")
        (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    # 요약
    log("=" * 100)
    for k, v in rep["arms"].items():
        if "error" in v:
            log(f"{k:22s} {v}")
            continue
        fams = " · ".join(f"{f} {d['test_auc']:.3f}" for f, d in v["families"].items())
        log(f"{k:22s} n_tr={v['n_train']:6,d} 최고={v['best_family']}({v['best_test_auc']:.3f}) | {fams}")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
