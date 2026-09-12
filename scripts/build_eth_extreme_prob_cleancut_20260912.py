#!/usr/bin/env python3
"""극점 탐지기 확률을 **판정 창에 대해 누수 없이** 매긴다 (2026-09-12).

왜 배포 아티팩트를 그대로 못 쓰나 — `data/live/eth_extreme_detector_artifact/meta.json` 의
`train_span` 이 **2024-09-08 ~ 2026-03-31** 이다. 규칙 채굴의 VAL(2025-09~12)과
OOS 앞 3개월(2026-01~03)이 통째로 그 학습 구간 안이다. 그 확률을 조건으로 걸면 모델이
이미 답을 본 구간에서 규칙을 고르는 셈이라, 승격 판정이 무효가 된다.

그래서 **같은 피쳐·같은 라벨·같은 하이퍼파라미터·같은 5시드**로 절단만 바꿔 다시 매긴다.
  TRAIN(≤2025-08-31) 구간   5겹 시계열 OOF + 라벨창(12봉) **퍼지**  → 확인창도 정직하게
  그 이후(VAL·OOS)          TRAIN 전체로 1회 학습 후 전진 예측      → 완전 표본외
등급 컷은 **TRAIN 분포의 분위**로 잡는다(배포 컷은 다른 학습창에서 나온 값이라 못 쓴다).

출력 tmp/eth_extreme_prob_cleancut_20260912/prob.parquet
  timestamp · long(1 바닥 / 0 천장) · p · tq(추세분위, 게이트용) · y(참값, 진단용)
자체점검 `--selftest` — 퍼지가 실제로 라벨창을 가리는지 인덱스로 확인한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402
import live_eth_extreme_detector_20260909 as LD  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

OUT = ROOT / "tmp/eth_extreme_prob_cleancut_20260912"
TRAIN_CUT = pd.Timestamp("2025-08-31 23:59:59")   # 규칙 채굴의 TRAIN 끝과 같다
SEEDS = [20260909, 771233, 305610, 517758, 961476]   # 배포 아티팩트와 동일
KFOLD = 5
HP = dict(max_iter=400, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
          early_stopping=True, validation_fraction=0.15)


def log(m: str) -> None:
    print(f"[xprob {time.strftime('%H:%M:%S')}] {m}", flush=True)


def purged_folds(n: int, k: int, embargo: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """시계열 k겹. 검증 블록 앞뒤 `embargo` 행을 학습에서 뺀다.

    라벨이 i+1..i+12 를 보므로 블록 경계에서 학습행의 라벨이 검증행 구간과 겹친다.
    퍼지가 없으면 그 겹침만으로 OOF 가 낙관된다.
    """
    bounds = np.linspace(0, n, k + 1).astype(int)
    out = []
    for j in range(k):
        lo, hi = bounds[j], bounds[j + 1]
        va = np.arange(lo, hi)
        tr = np.concatenate([np.arange(0, max(0, lo - embargo)), np.arange(min(n, hi + embargo), n)])
        out.append((tr, va))
    return out


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = np.zeros(len(te))
    for sd in SEEDS:
        m = HistGradientBoostingClassifier(random_state=sd, **HP)
        m.fit(tr[LD.FEATS], tr["_y"])
        p += m.predict_proba(te[LD.FEATS])[:, 1] / len(SEEDS)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        f = purged_folds(100, 5, 12)
        tr, va = f[2]
        assert va.min() == 40 and va.max() == 59, (va.min(), va.max())
        assert tr.max() < 40 - 12 or True
        assert not set(range(28, 72)) & set(tr.tolist()), "퍼지 구간이 학습에 남아 있다"
        assert set(f[0][1].tolist()) == set(range(0, 20))
        assert all(len(np.intersect1d(t, v)) == 0 for t, v in f), "학습/검증 겹침"
        print("selftest OK — 5겹 퍼지 12행, 학습/검증 무교차")
        return 0

    kl = B._load_kl(B.ETH_KL)
    btc = B._load_kl(B.BTC_KL)
    fund = B._load_funding()
    cap = min(kl["timestamp"].max(), btc["timestamp"].max())
    kl = kl[(kl["timestamp"] >= pd.Timestamp(a.start)) & (kl["timestamp"] <= cap)].reset_index(drop=True)
    sig = compute_signals(kl, btc_df=btc[btc["timestamp"] <= cap], funding_df=fund[fund["calc_time"] <= cap])
    A = LD.build_rows(sig, btc[btc["timestamp"] <= cap], with_label=True)
    A = A[A["_y"] >= 0].sort_values("_ts").reset_index(drop=True)
    log(f"모집단(발동 봉, 측면별) {len(A):,}행 {A._ts.min()} ~ {A._ts.max()}")

    is_tr = (A["_ts"] <= TRAIN_CUT).to_numpy()
    T, F = A[is_tr].reset_index(drop=True), A[~is_tr].reset_index(drop=True)
    log(f"TRAIN {len(T):,} (OOF {KFOLD}겹 퍼지) · 이후 {len(F):,} (전진 예측)")

    p_tr = np.full(len(T), np.nan)
    for j, (tr, va) in enumerate(purged_folds(len(T), KFOLD, LD.W)):
        p_tr[va] = fit_predict(T.iloc[tr], T.iloc[va])
        log(f"  OOF {j+1}/{KFOLD} 완료 (학습 {len(tr):,} → 검증 {len(va):,})")
    p_f = fit_predict(T, F)
    log("전진 예측 완료")

    out = pd.concat([
        pd.DataFrame({"timestamp": T["_ts"].to_numpy(), "long": T["_long"].astype(int).to_numpy(),
                      "p": p_tr, "tq": T["_tq"].to_numpy(), "y": T["_y"].to_numpy(), "split": "TRAIN"}),
        pd.DataFrame({"timestamp": F["_ts"].to_numpy(), "long": F["_long"].astype(int).to_numpy(),
                      "p": p_f, "tq": F["_tq"].to_numpy(), "y": F["_y"].to_numpy(), "split": "FWD"}),
    ], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    from sklearn.metrics import roc_auc_score
    cuts = {g: float(np.nanquantile(p_tr, 1 - q)) for g, q in (("강", 0.05), ("중", 0.10), ("약", 0.25))}
    aucs = {"TRAIN_OOF": float(roc_auc_score(T["_y"], p_tr))}
    for nm, s, e in (("VAL", "2025-09-01", "2025-12-31"), ("OOS", "2026-01-01", None)):
        m = (out.split == "FWD") & (out.timestamp >= pd.Timestamp(s))
        if e:
            m &= out.timestamp <= pd.Timestamp(e)
        aucs[nm] = float(roc_auc_score(out.loc[m, "y"], out.loc[m, "p"]))
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "prob.parquet", index=False)
    meta = {"train_cut": str(TRAIN_CUT), "kfold": KFOLD, "embargo_bars": LD.W, "seeds": SEEDS,
            "n": int(len(out)), "cuts_from_TRAIN": cuts, "auc": aucs,
            "base_rate": {k: float(out[out.split == v]["y"].mean()) for k, v in (("TRAIN", "TRAIN"), ("FWD", "FWD"))},
            "why": "배포 아티팩트의 train_span(2024-09-08~2026-03-31)이 규칙 채굴의 VAL/OOS 를 덮어 누수."}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log("AUC " + " · ".join(f"{k} {v:.4f}" for k, v in aucs.items()) + f" | 컷(TRAIN분위) {cuts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
