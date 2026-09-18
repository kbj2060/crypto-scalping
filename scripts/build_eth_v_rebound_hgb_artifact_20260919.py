#!/usr/bin/env python3
"""V자 급등락 **HGB 아티팩트** 빌더 — 라이브 TabPFN 을 대체한다 (2026-09-19, 사용자 지시).

근거 실측: `scripts/research_eth_v_rebound_hgb_vs_tabpfn_20260919.py` (같은 45,000행 맞대결)
    TabPFN 18k ctx   VAL .7025 · OOS .7102 · FWD .7029 · 라이브 1사이클 **6.62s**
    HGB 184k TRAIN   VAL .7027 · OOS .7125 · FWD .7036 · 라이브 1사이클 **12.7ms**
    HGB 같은 18k ctx VAL .6791 · OOS .6987 · FWD .6807      <- 컨텍스트를 맞추면 HGB 가 **진다**
⭐우위의 정체는 모델이 아니라 TRAIN 크기다. TabPFN 은 컨텍스트 상한(=6.6초의 원인) 때문에
  18,000행밖에 못 쓰고, HGB 는 1회 학습이라 184,207행을 다 쓴다.
🔴+0.0002~+0.0023 은 **앙상블 분산감소**다(시드폭 VAL .6976~.7018). 「AUC 가 이겨서」가 아니라
  「동률인데 575배 싸서」 바꾸는 것이다.

## 아티팩트 규약 — 극점 탐지기와 **같은 모양**
`data/live/eth_v_rebound_hgb_artifact/{model.joblib, meta.json}`.
`meta["features"]` 가 라이브의 FEATURES 와 다르면 라이브는 아티팩트를 **먹지 않는다**
(극점 탐지기 `load_art()` 와 같은 가드 -- 옛 형상을 조용히 삼키지 않는다).

## 임계값 — 확률값이 아니라 **발동률**로 옮긴다
`PROBA_THRESHOLD=0.60` 은 TabPFN 확률의 운영점이다. 모델이 바뀌면 같은 0.60 이 전혀 다른
발동률이 된다([[feedback_alert_threshold_must_be_declared_as_percentile_20260919]]).
배포본이 실측으로 남긴 **하루 13.25 발동봉**(0.60 기준)을 재현하는 HGB 확률을 찾아
`meta["proba_threshold"]` 에 적는다.

🔴**보정 모집단은 «모든 봉»이다 -- 학습 모집단이 아니다.**
  학습은 3상태(ambiguous 제외, 전체의 ~53%) 위에서 하지만, 라이브 칩은 **라벨 유무와 무관하게
  모든 봉**을 채점하고 봉마다 `max(bottom, top)` 하나만 보여준다(`best_by_pos`). 분모와 집계가
  둘 다 다르므로 «라벨 있는 행»에서 잡은 분위는 화면 발동률이 아니다.
  집계 정의는 2026-09-01 의 `research_eth_v_rebound_every_bar_threshold_signal_frequency_
  20260901.py`(f2eb2377 에서 삭제, `git show f2eb2377^:` 로 복원)를 그대로 옮겼다 --
  봉당 max · 연속 발동은 한 사건 · 48봉 스트립 평균 칠해진 칸 · 신호 없는 날 비율.
  재현 확인: 이 빌더의 VAL 채점봉 수가 그 리포트의 `n_bars=35,136` 과 일치해야 한다(assert).

🔴이건 **분류 운영점을 보존**하는 것이지 경제성 게이트를 다시 통과했다는 뜻이 아니다 --
  0.60 이 통과했던 그 시험(data/research/eth_v_rebound_every_bar_tabpfn_costgate_20260901)은
  HGB 확률로 다시 돌려야 한다. 그 전까지 이 교체는 «같은 빈도·같은 순위»까지만 보증한다.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
for _p in (CODE, CODE / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import research_eth_v_rebound_hgb_vs_tabpfn_20260919 as X                    # noqa: E402
import research_eth_v_rebound_close_anchor_retrain_20260912 as R12           # noqa: E402
import live_eth_sweep_v_rebound_signal_20260829 as LIVE                      # noqa: E402

ART = X.ROOT / "data/live/eth_v_rebound_hgb_artifact"
DEPLOYED_FIRES_PER_DAY = 13.25   # 배포 TabPFN 의 thr=0.60 VAL 실측(발동봉/일). 이걸 재현한다.
# 같은 리포트의 나머지 0.60 프로필 -- 재보정이 여기에 얼마나 붙는지 같이 낸다.
DEPLOYED_060_VAL = {"n_bars_fired": 1616, "pct_of_bars": 4.60, "per_day": 13.25,
                    "events_per_day": 11.01, "median_gap_hours": 1.1,
                    "strip_avg_colored_of_48": 2.2, "dry_day_pct": 0.0, "n_bars": 35136}
FREQ_REPORT = ("data/research/eth_v_rebound_every_bar_tabpfn_costgate_20260901/"
               "signal_frequency.json")
HISTORY_BARS = 48


def frequency(per_bar: pd.Series, thr: float, days: float) -> dict:
    """2026-09-01 `..._signal_frequency_20260901.py` 의 집계를 **그대로** 옮긴 것."""
    fired = per_bar >= thr
    n_fire = int(fired.sum())
    idx = np.flatnonzero(fired.to_numpy())
    if len(idx) >= 2:
        gaps = np.diff(idx)
        gap_h, n_events = float(np.median(gaps) * 5 / 60), 1 + int((gaps > 1).sum())
    else:
        gap_h, n_events = float("nan"), n_fire
    by_day = fired.groupby(fired.index.date).any()
    return {"n_bars_fired": n_fire, "pct_of_bars": round(n_fire / len(per_bar) * 100, 2),
            "per_day": round(n_fire / days, 2), "n_events": n_events,
            "events_per_day": round(n_events / days, 2),
            "median_gap_hours": round(gap_h, 1) if gap_h == gap_h else None,
            "strip_avg_colored_of_48": round(float(fired.rolling(HISTORY_BARS).sum().mean()), 1),
            "dry_day_pct": round(float((~by_day).mean() * 100), 1)}


def main() -> int:
    # 🔴모집단은 «live»(모든 봉 x 양측면) 하나로 뽑는다. 학습용 3상태 부분집합은 여기서 마스크로
    #   떼어낸다 -- assemble 을 두 번 돌리면 같은 걸 두 번 만든다.
    D, win, agree = X.assemble("live")
    X_all = D[LIVE.FEATURES].to_numpy(float)
    y3 = D.y3.to_numpy()
    tr_fit = (win["TRAIN"] & np.isfinite(y3)
              & (D.timestamp >= pd.Timestamp(X.MATCHED_START, tz="UTC")).to_numpy())
    print(f"[학습] 3상태 TRAIN {int(tr_fit.sum()):,}행 · 라벨률 {y3[tr_fit].mean():.2%} "
          f"(기록 182,969 @14.63%)", flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib
    models = []
    for sd in R12.SEEDS:
        m = HistGradientBoostingClassifier(          # 측정 스크립트와 **같은** 하이퍼파라미터
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=sd)
        m.fit(X_all[tr_fit], y3[tr_fit].astype(int))
        models.append(m)
    P = np.mean([m.predict_proba(X_all)[:, 1] for m in models], axis=0)

    # ── 임계값: 배포본의 «발동봉/일» 을 **화면 모집단에서** 재현하는 확률 ─────────
    # 라이브는 봉마다 양측면을 채점하고 높은 쪽을 그 봉의 그림으로 쓴다(best_by_pos).
    def per_bar_of(mask) -> tuple[pd.Series, float]:
        s = (pd.DataFrame({"t": D.timestamp[mask], "p": P[mask]})
             .groupby("t", sort=True)["p"].max().sort_index())
        d = (s.index.max() - s.index.min()).total_seconds() / 86400
        return s, d

    pb_val, days_val = per_bar_of(win["VAL"])
    assert len(pb_val) == DEPLOYED_060_VAL["n_bars"], (
        f"VAL 채점봉 {len(pb_val):,} != 09-01 리포트 {DEPLOYED_060_VAL['n_bars']:,} "
        "-- 모집단이 그때와 다르다. 아래 빈도 비교가 성립하지 않는다.")
    target = DEPLOYED_FIRES_PER_DAY / (len(pb_val) / days_val)
    thr = float(np.quantile(pb_val.to_numpy(), 1.0 - target))
    print(f"[임계값] VAL {len(pb_val):,}봉({days_val:.0f}일) 목표 {DEPLOYED_FIRES_PER_DAY}/일 "
          f"-> proba {thr:.4f} (분위 {1 - target:.4f})", flush=True)

    freq = {}
    print(f"\n  {'창':>4} {'발동봉':>7} {'%':>6} {'/일':>6} {'사건/일':>7} {'간격h':>6} "
          f"{'스트립/48':>9} {'무신호일%':>8}", flush=True)
    for nm in ("VAL", "OOS", "FWD"):
        pb, dd = per_bar_of(win[nm])
        freq[nm] = frequency(pb, thr, dd)
        f = freq[nm]
        print(f"  {nm:>4} {f['n_bars_fired']:>7,} {f['pct_of_bars']:>5.2f}% {f['per_day']:>6.2f} "
              f"{f['events_per_day']:>7.2f} {str(f['median_gap_hours']):>6} "
              f"{f['strip_avg_colored_of_48']:>9} {f['dry_day_pct']:>7.1f}%", flush=True)
    d0 = DEPLOYED_060_VAL
    print(f"  배포 {d0['n_bars_fired']:>7,} {d0['pct_of_bars']:>5.2f}% {d0['per_day']:>6.2f} "
          f"{d0['events_per_day']:>7.2f} {d0['median_gap_hours']:>6} "
          f"{d0['strip_avg_colored_of_48']:>9} {d0['dry_day_pct']:>7.1f}%  <- TabPFN@0.60 VAL",
          flush=True)

    # ── AUC: 학습 모집단(3상태)에서 -- 빈도와 모집단이 다르므로 따로 낸다 ────────
    scores = {}
    for nm in ("VAL", "OOS", "FWD"):
        m = win[nm] & np.isfinite(y3)
        scores[nm] = {"n": int(m.sum()), "base_rate": float(y3[m].mean()),
                      "model_auc": R12.auc(P[m], y3[m].astype(int)),
                      "single_feature_auc": R12.auc(D.rng_atr.to_numpy()[m], y3[m].astype(int))}
        print(f"  [AUC·3상태] {nm:>4} n={m.sum():>7,} {scores[nm]['model_auc']:.4f} "
              f"(단일피쳐 {scores[nm]['single_feature_auc']:.4f})", flush=True)

    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, ART / "model.joblib")
    (ART / "meta.json").write_text(json.dumps({
        "rule_id": "eth_v_rebound_hgb_allbars_20260919",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "replaces": "TabPFN in-context (frozen 18,000-row context, device=cuda)",
        "features": LIVE.FEATURES,
        "seeds": R12.SEEDS,
        "model": "HistGradientBoostingClassifier x5 (max_iter=300, lr=0.05, leaves=31, l2=1.0)",
        "train": {"n": int(tr_fit.sum()), "label_rate": float(y3[tr_fit].mean()),
                  "first": str(D.timestamp[tr_fit].min()), "last": str(D.timestamp[tr_fit].max()),
                  "population": "3-state (v_rebound / chop / ambiguous 제외), 2026-09-01 모집단 복원"},
        "label_parity_vs_frozen_context": agree,
        "proba_threshold": thr,
        "proba_threshold_percentile": 1.0 - target,
        "threshold_basis": {
            "target": f"배포 TabPFN thr=0.60 의 VAL {DEPLOYED_FIRES_PER_DAY} 발동봉/일",
            "population": "ALL bars x per-bar max(bottom, top) -- 라이브 best_by_pos 와 동일 "
                          "(학습 모집단이 아니다)",
            "aggregation_source": FREQ_REPORT,
            "deployed_060_val": DEPLOYED_060_VAL},
        "frequency": freq,
        "scores_on_3state_population": scores,
        "tabpfn_head_to_head_same_rows": {
            "source": "scripts/research_eth_v_rebound_hgb_vs_tabpfn_20260919.py",
            "VAL": {"tabpfn": 0.7025, "hgb": 0.7027},
            "OOS": {"tabpfn": 0.7102, "hgb": 0.7125},
            "FWD": {"tabpfn": 0.7029, "hgb": 0.7036},
            "live_cycle_sec": {"tabpfn": 6.62, "hgb": 0.0127},
            "caveat": "차이는 앙상블 분산감소 범위."},
        # 🔴🔴경제성 게이트는 **통과하지 못했다**. 사용자가 부하 이득을 우선해 그대로 배포하기로
        # 결정(2026-09-19). 이 칩은 trading_bot.py 에 배선돼 있지 않은 «재량 참고용»이라 주문
        # 손실은 없지만, **화면 신호의 질은 아래만큼 나빠진 채로 서빙된다**.
        # 근거: scripts/backtest_eth_v_rebound_hgb_costgate_20260919.py
        "costgate_FAILED": {
            "verdict": "FAIL -- 분류는 재현하는데 경제성이 4배 작다",
            "same_config_same_call_count": {
                "config": "SL4.0/ARM1.5/TR0.1 (배포판이 VAL 에서 고른 그 설정)",
                "VAL": {"hgb_pess_bp": 3.95, "tabpfn_pess_bp": 16.22, "n": [487, 488]},
                "OOS": {"hgb_pess_bp": 1.87, "tabpfn_pess_bp": 11.44, "n": [346, 385]}},
            "classification_is_fine": {
                "VAL_precision": [0.7064, 0.7131], "OOS_precision": [0.6994, 0.6831],
                "note": "호출률 맞춤(HGB thr 0.5386 = 배포 0.60). OOS 는 HGB 가 오히려 높다."},
            "flip_control": "통과 -- VAL 정방향 +4.26 vs 뒤집기 -38.62, 수익격자 정23/뒤0 "
                            "(배포판 OOS 는 뒤집기가 157칸이라 오히려 지저분했다). 노이즈수확 아님.",
            "mechanism": "같은 AUC·같은 정밀도인데 **고르는 봉이 다르다**. 2026-09-01 원본이 "
                         "예고한 실패 양식: '경제성은 AUC 가 아니라 상위 확률 꼬리의 순위가 결정한다'.",
            "unresolved": "비교 기준(labeled)은 커밋 aac1805c 에서 이미 철회된 숫자다. 철회되지 "
                          "않은 all_bars 기준에서는 HGB 가 전 임계값 음수인데(수익격자 정0/뒤0) "
                          "배포 TabPFN 의 all_bars 숫자는 존재하지 않아 우열을 못 가린다.",
            "next_diagnostic": "호출된 봉의 실현 |이동폭| 분포를 두 모델에서 비교 -- TabPFN 상위 "
                               "꼬리가 «큰 움직임»에 걸리고 HGB 는 «확률은 맞지만 작은 움직임»에 "
                               "걸리는지."},
    }, ensure_ascii=False, indent=1))
    print(f"\n산출물 {ART}/model.joblib "
          f"({(ART / 'model.joblib').stat().st_size / 1e6:.1f}MB) · meta.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
