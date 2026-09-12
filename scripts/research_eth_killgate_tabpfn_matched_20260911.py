#!/usr/bin/env python3
"""청산 킬게이트 — **TabPFN** 계열 추가, 표본 매칭 대조 (2026-09-11).

사용자 지적: *"tabpfn 모델도 해봐야하는거 아니야?"* — 맞다. 지금까지 시험한 셋
(HistGradientBoosting · GBM · 원시시퀀스 TCN)은 전부 **경사 기반 학습기**라 귀납 편향을
공유한다. TabPFN 은 사전학습 트랜스포머의 **문맥 내 학습**이라 계열이 다르고, 무엇보다
**이 저장소가 실제로 배포하는 모델**이다(증거신호 8종 메타라벨 전부 TabPFN).

🔴표본 매칭 필수: TabPFN 은 학습 표본 상한이 있어(이 저장소 경험치 ~18,000행) IS 전체
175k 행을 쓰는 GBM 과 정면 비교하면 불공정하다. 그래서 **같은 부분표본으로 학습한 GBM**을
짝대조군으로 같이 낸다 — [[eth_regime_gbm3_vs_tabpfn_matched_control_20260902]] /
[[feedback_gbm_proxy_fails_when_sample_size_is_the_driver_20260902]] 가 남긴 규율이다.

판정은 앞선 게이트와 같다: ★청산 타깃이 VAL·OOS 둘 다 AUC >= 0.55 인가.
양성대조(이벤트 트리거 4종)가 같이 돌아가므로 «모델이 망가졌다»와 «정보가 없다»를 가를 수 있다.

실행 환경: **서버 GPU**(RTX 3070 Ti) · tabpfn 8.5.0 · **v3 기본 분류기**
(v3 가중치와 auth_token 이 서버에 이미 있다 — 로컬은 v2 캐시뿐이라 서버로 옮겼다).
⚠️이 GPU 는 대시보드의 TabPFN 모델들과 공유된다. 경합이 대시보드 타임아웃을 낸 전례가 있어
(feedback_shared_gpu_contention_causes_dashboard_timeouts_20260903) 작업을 짧게 끊고
추론을 배치로 쪼갠다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_eth_raw_sequence_killgate_with_triggers_20260911 import (  # noqa: E402
    DATA, SPLITS, labels,
)
from scripts.live_evidence_signal_dashboard_20260823 import compute_signals, SIGNAL_ORDER  # noqa: E402

import os
N_FIT = int(os.environ.get("N_FIT", 10_000))    # TabPFN 문맥 표본 (이 저장소 경험 상한 ~18,000)
N_EVAL = int(os.environ.get("N_EVAL", 20_000))  # 평가 부분표본 — AUC 표준오차 ~0.004
N_EST = int(os.environ.get("N_EST", 4))
DEVICE = os.environ.get("TABPFN_DEVICE", "cuda")
PRED_BATCH = 4_000    # GPU 를 오래 점유하지 않도록 추론을 쪼갠다(대시보드 공유)
RNG = np.random.default_rng(20260911)


def proba(model, X: np.ndarray) -> np.ndarray:
    """배치 추론 — 한 번에 다 밀면 VRAM 을 오래 잡아 대시보드와 경합한다."""
    return np.concatenate([model.predict_proba(X[i:i + PRED_BATCH])[:, 1]
                           for i in range(0, len(X), PRED_BATCH)])


def sub(idx: np.ndarray, k: int) -> np.ndarray:
    """시간축 균일 부분표집 — 무작위로 뽑으면 특정 구간에 몰릴 수 있다."""
    return idx if len(idx) <= k else idx[np.linspace(0, len(idx) - 1, k).astype(int)]


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier

    eth = pd.read_csv(DATA / "eth_5m_1year.csv", parse_dates=["timestamp"])
    btc = pd.read_csv(DATA / "btc_5m_1year.csv", parse_dates=["timestamp"])
    eth = eth[eth.timestamp >= "2024-01-01"].reset_index(drop=True)
    sig = compute_signals(eth, btc_df=btc)

    tab = ["p_fast", "p_slow", "delta_z", "vol_z", "ret3_z", "atr_pct",
           "lower_wick_ratio", "upper_wick_ratio", "dem", "kalman_dev_z"]
    X = np.column_stack([sig[tab].to_numpy(float)] +
                        [(sig[f"bottom_{s}"].fillna(False).to_numpy().astype(float)
                          - sig[f"top_{s}"].fillna(False).to_numpy().astype(float))
                         for s, _ in SIGNAL_ORDER])
    ts = eth["timestamp"].to_numpy()
    ys = labels(eth, sig, len(eth))
    finite = np.isfinite(X).all(1)

    print(f"# 피쳐 {X.shape[1]} · TabPFN v3 문맥 {N_FIT:,} · 평가 {N_EVAL:,} · "
          f"n_estimators {N_EST} · device {DEVICE}")
    print(f"# 짝대조군 = 같은 {N_FIT:,}행으로 학습한 HGB · 참조 = IS 전체로 학습한 HGB\n")
    hdr = (f"{'타깃':22s} {'양성률':>6s} | {'TabPFN VAL':>10s} {'TabPFN OOS':>10s} | "
           f"{'짝GBM VAL':>9s} {'짝GBM OOS':>9s} | {'전체GBM VAL':>11s} {'전체GBM OOS':>11s}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for name, y in ys.items():
        ok = finite & np.isfinite(y)
        m = {k: np.flatnonzero(ok & (ts >= np.datetime64(a)) & (ts <= np.datetime64(b + "T23:59")))
             for k, (a, b) in SPLITS.items()}
        if min(len(v) for v in m.values()) < 500:
            print(f"{name:22s} 표본 부족 — 건너뜀")
            continue
        yy = np.nan_to_num(y).astype(int)
        fit = sub(m["IS"], N_FIT)
        ev = {k: sub(m[k], N_EVAL) for k in ("VAL", "OOS")}

        # model_path 를 주지 않는다 → 서버에 이미 있는 **v3 기본 분류기**를 쓴다
        tp = TabPFNClassifier(n_estimators=N_EST, device=DEVICE, ignore_pretraining_limits=True)
        tp.fit(X[fit], yy[fit])
        tp_auc = {k: roc_auc_score(yy[v], proba(tp, X[v])) for k, v in ev.items()}

        g_m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                             l2_regularization=1.0, random_state=0,
                                             early_stopping=True).fit(X[fit], yy[fit])
        gm_auc = {k: roc_auc_score(yy[v], g_m.predict_proba(X[v])[:, 1]) for k, v in ev.items()}

        g_f = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                             l2_regularization=1.0, random_state=0,
                                             early_stopping=True).fit(X[m["IS"]], yy[m["IS"]])
        gf_auc = {k: roc_auc_score(yy[v], g_f.predict_proba(X[v])[:, 1]) for k, v in ev.items()}

        print(f"{name:22s} {yy[fit].mean():6.3f} | {tp_auc['VAL']:10.3f} {tp_auc['OOS']:10.3f} | "
              f"{gm_auc['VAL']:9.3f} {gm_auc['OOS']:9.3f} | {gf_auc['VAL']:11.3f} {gf_auc['OOS']:11.3f}",
              flush=True)
        rows.append(dict(target=name, pos=yy[fit].mean(), n_fit=len(fit),
                         tabpfn_VAL=tp_auc["VAL"], tabpfn_OOS=tp_auc["OOS"],
                         gbm_matched_VAL=gm_auc["VAL"], gbm_matched_OOS=gm_auc["OOS"],
                         gbm_full_VAL=gf_auc["VAL"], gbm_full_OOS=gf_auc["OOS"]))
        pd.DataFrame(rows).to_csv(ROOT / "tmp/evidence_entry_peak_exit_20260911/killgate_tabpfn.csv",
                                  index=False)
    df = pd.DataFrame(rows)
    gate = df[df.target.str.startswith("★")]
    ctrl = df[~df.target.str.startswith("★")]
    print(f"\n=== 판정 ===")
    print(f"양성대조 TabPFN VAL {ctrl.tabpfn_VAL.min():.3f}~{ctrl.tabpfn_VAL.max():.3f} "
          f"(최고 {ctrl.loc[ctrl.tabpfn_VAL.idxmax(), 'target']})")
    print(f"킬게이트 TabPFN VAL {gate.tabpfn_VAL.min():.3f}~{gate.tabpfn_VAL.max():.3f} · "
          f"OOS {gate.tabpfn_OOS.min():.3f}~{gate.tabpfn_OOS.max():.3f}")
    print(f"킬게이트 통과(VAL·OOS >= 0.55): "
          f"{int(((gate.tabpfn_VAL >= .55) & (gate.tabpfn_OOS >= .55)).sum())}/{len(gate)}")
    print(f"\n소요 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
