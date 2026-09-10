#!/usr/bin/env python3
"""중기 팔에 **TabPFN**(배포 계열·소표본 특화)을 붙인 대조 (2026-09-10 후속).

`research_eth_midterm_model_capacity_20260910.py` 의 TabPFN 팔은 pip 최신판(8.5)이 API 토큰을 요구해 실패했다.
로컬 실행되는 **TabPFN v2.2.1** 을 격리 venv(`/tmp/tp2`)에 두고 여기서 서브프로세스로 부른다
(공유 파이썬 환경은 건드리지 않는다 -- pydantic 1.10 유지 확인).

CPU 라 비용이 n_train × n_test 에 비례한다. 그래서 **동일 부분표집 위에서 HGB 와 1:1 로 비교**한다:
학습 최대 3,000행 · 평가 최대 1,500행(고정 시드). HGB 는 같은 행으로 다시 적합해 같은 잣대로 잰다.
분할·라벨·피쳐는 본 실험과 동일(TRAIN+VAL ≤2025-08-31 / TEST 2025-09-01~).
출력 tmp/eth_midterm_capacity_20260910/tabpfn_arm.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_exit_synth_dataset_20260910 as BD  # noqa: E402
import research_eth_midterm_model_capacity_20260910 as C  # noqa: E402
import live_eth_extreme_detector_20260909 as X  # noqa: E402

TP = Path("/tmp/tp2/bin/python")
OUT = C.OUT
MAX_TR, MAX_TE = 2000, 1200   # CPU 예산 -- TabPFN 은 n_train×n_test 에 비례한다
ARMS = [("4h", "current"), ("4h", "extended"), ("1h", "current"), ("1h", "extended"),
        ("15m", "match513"), ("5m", "match513")]


def log(m):
    print(f"[tp {time.strftime('%H:%M:%S')}] {m}", flush=True)


def tabpfn_proba(Xtr, ytr, Xte, tag) -> np.ndarray | None:
    f = OUT / f"_tp_{tag}.npz"
    np.savez(f, Xtr=np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0),
             ytr=ytr, Xte=np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0))
    code = ("import numpy as np;from tabpfn import TabPFNClassifier;"
            f"d=np.load(r'{f}');m=TabPFNClassifier(device='cpu',random_state=11).fit(d['Xtr'],d['ytr']);"
            f"np.save(r'{f}.out.npy', m.predict_proba(d['Xte'])[:,1])")
    try:
        env = {**os.environ, "TABPFN_ALLOW_CPU_LARGE_DATASET": "1"}   # CPU 대용량 기본 차단 해제(느릴 뿐 정확도 무관)
        subprocess.run([str(TP), "-c", code], check=True, capture_output=True, timeout=5400, env=env)
        return np.load(f"{f}.out.npy")
    except Exception as e:  # noqa: BLE001
        log(f"⚠️{tag}: {type(e).__name__} {str(getattr(e, 'stderr', b''))[-300:]}")
        return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    eth_x, btc_x = C.load_klines_extended()
    eth_c, btc_c = BD.load_klines("eth", "ETHUSDT"), BD.load_klines("btc", "BTCUSDT")
    rows = {}
    out = {"max_train": MAX_TR, "max_test": MAX_TE, "tabpfn": "2.2.1 (local, cpu, isolated venv)", "arms": {}}
    for tf, mode in ARMS:
        key = f"{tf}|{mode}"
        src = "extended" if mode == "extended" else "current"
        if (tf, src) not in rows:
            e, b = (eth_x, btc_x) if src == "extended" else (eth_c, btc_c)
            rows[(tf, src)] = C.rows_for(e, b, C.TFS[tf])
        A = rows[(tf, src)]
        ts = pd.to_datetime(A["_ts"])
        tr = (ts <= C.VAL_END).to_numpy(); te = (ts > C.VAL_END).to_numpy()
        Xm = A[X.FEATS].to_numpy(np.float32); y = A["_y"].to_numpy()
        rng = np.random.default_rng(7)
        itr = np.flatnonzero(tr)
        if mode == "match513":
            itr = rng.choice(itr, min(C.MATCH_N, len(itr)), replace=False)
        elif len(itr) > MAX_TR:
            itr = rng.choice(itr, MAX_TR, replace=False)
        ite = np.flatnonzero(te)
        if len(ite) > MAX_TE:
            ite = np.sort(rng.choice(ite, MAX_TE, replace=False))
        Xtr, ytr, Xte, yte = Xm[itr], y[itr], Xm[ite], y[ite]
        t0 = time.time()
        p = tabpfn_proba(Xtr, ytr, Xte, key.replace("|", "_"))
        rec = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "base_rate": float(yte.mean()),
               "seconds": round(time.time() - t0, 1)}
        if p is not None:
            rec["tabpfn_auc"] = float(roc_auc_score(yte, p))
        # 같은 행 위에서 HGB 두 설정(원본 · 소표본 튜닝판)
        for nm, kw in (("hgb_orig", {}), ("hgb_small", dict(max_iter=150, max_leaf_nodes=7, min_samples_leaf=100, learning_rate=0.03))):
            P = np.zeros(len(yte))
            for sd in C.SEEDS:
                P += C.hgb(**kw)(sd).fit(Xtr, ytr).predict_proba(Xte)[:, 1] / len(C.SEEDS)
            rec[f"{nm}_auc"] = float(roc_auc_score(yte, P))
        out["arms"][key] = rec
        log(f"{key:16s} n_tr={rec['n_train']:5d} n_te={rec['n_test']:5d} · tabpfn {rec.get('tabpfn_auc', float('nan')):.3f} "
            f"· hgb_orig {rec['hgb_orig_auc']:.3f} · hgb_small {rec['hgb_small_auc']:.3f} ({rec['seconds']}s)")
        (OUT / "tabpfn_arm.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
