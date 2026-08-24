#!/usr/bin/env python3
"""SOL "상승추세 중 zig075 SHORT" 진입 -- 5시드 seed-variance 저감 효과 저비용 사후진단 (2026-08-20).

## 배경 / 질문
ETH에서 확정된 오디세이4 섀도우 entry-veto(SustainedUptrendDetector로 "상승추세 중 zig075 SHORT
진입"을 규칙기반, 자유파라미터 0개로 스킵) + exit-guard 메커니즘이 h48qual/zig075 인코더를 진짜로
재시드해도 6/6창 포트폴리오 부호를 안정시킨다는 게 확정됐다
(docs/experiments/eth_odyssey4_shadow_full_reseed_causal_isolation_20260820.md). 유력 메커니즘
가설은 "상승추세 중 SHORT 진입"이 모델 확신도/캘리브레이션이 시드마다 가장 크게 갈리는 트레이드
유형이라 이걸 규칙으로 제거하면 시드 분산의 핵심 원천이 사라진다는 것.

이 스크립트는 그 메커니즘을 SOL에 이식하기 "전에" -- 새 detector/메커니즘을 구축하지 않고 --
SOL 자신의 이미 평가된 N=5 시드 trade ledger에 대해 순수 사후(post-hoc) 재집계만 수행해,
"SOL의 SHORT 진입도 SOL 자신의 상승추세 구간에서 유독 시드간 분산이 큰가"를 저렴하게 진단한다.
이 저장소는 ETH 리스크 메커니즘을 BTC로 이식하려다 두 번 실패한 전례(drawdown governor)가 있어
"ETH서 됐으니 SOL도"라는 가정을 검증 없이 믿지 않는다 -- 반드시 대조군(무작위 동일개수 제외,
비상승추세 SHORT 동일개수 제외)과 비교해서만 결론 낸다.

## 입력 (전부 기존 산출물 재사용 -- 신규 학습/신규 백테스트 없음)
- 5시드 SOL zig075 v2(adaptive_squeeze) trade ledger: tmp/causal_regen_20260516/
  sol_live_promotion_seed_robustness_20260819/portfolio_ledger_<seed>_<window>.csv
  (docs/experiments/sol_live_promotion_seed_robustness_5seed_20260819.md,
  scripts/sol_live_promotion_seed_robustness_eval_5seed_20260819.py 로 생성됨. 로컬엔
  summary_report.json만 있었고 ledger CSV 15개는 서버에만 있어 `handoff.sh pull`로 그대로
  가져왔다 -- 재실행/재훈련 아님, 이미 서버에 존재하던 산출물 그대로).
- SOL dual_momentum 피쳐: data/splits/year_oos_adaptive_squeeze_sol_20260720/
  sol_features_{2025,2026}.csv. SOL도 ETH와 동일하게 dual_momentum이 {-1,0,1} 카테고리컬
  피쳐임을 실측 확인(둘 다 nunique=3, 분포도 유사) -- ETH SustainedUptrendDetector의
  `dual_momentum > 0` 임계 로직이 그대로 이식 가능함을 확인했다. 두 CSV를 이어붙이면
  2025-01-01~2026-07-21까지 5분봉 gap/중복 0개로 완전 연속(실측 확인) -- rolling(2016) 같은
  row-count 기반 창이 안전하게 성립한다.

## 방법
1. dual_momentum>0의 rolling(2016bar=1주) 평균 = "상승추세 점수". ETH의
   scripts/live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py::SustainedUptrendDetector와
   동일한 causal 로직을 pandas rolling으로 벡터화 재구현(deque 기반 라이브 구현과 수학적으로
   동일 -- 매 bar에서 자신을 포함한 직전 2016bar 중 dual_momentum>0 비율).
2. threshold = ETH 고정값(0.8025793650793651)을 쓰지 않고, **SOL 2025년 전체 데이터에서 이
   점수 분포의 p90**으로 새로 잡는다. 이는 진단용 placeholder이며 프로덕션 캘리브레이션이
   아니다(ETH는 2025 H1만으로 캘리브레이션했으나, 이 진단은 사용자 지시대로 "SOL 자신의
   2025년 데이터 분포" 전체를 쓴다 -- 방법론 차이를 명시).
3. 15개(5시드 x 3창) ledger 각각에서 entry_timestamp로 이 활성여부를 매칭해 "상승추세중
   SHORT" 트레이드를 식별한다.
4. 그 트레이드를 뺀 뒤 잔여 trade_return을 원래 순서대로 다시 복리 재계산(cumprod) -- 새
   백테스트/새 예측이 전혀 아니고, 이미 계산된 per-trade trade_return의 부분합 재구성이다.
   `research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py::_ledger_metrics`와
   동일한 PnL% 공식을 재사용(첫 트레이드부터 실측 baseline_pnl이 summary_report.json의 no_gate
   pnl과 일치하는지 대조해 방법론이 맞는지 자체 검증한다).
5. 대조군 2종을 각 2000회 부트스트랩으로 비교한다:
   - Control A: "상승추세중 SHORT"와 무관하게, 같은 개수만큼 무작위로 아무 트레이드나 제외.
   - Control B: SHORT이긴 하되 상승추세가 아닌 구간의 SHORT를 같은 개수만큼 무작위로 제외.
   목적: "트레이드 수를 줄이면 분산이 준다"는 트리비얼 효과와 "상승추세중 SHORT를 특정해서
   제거하는 게 유독 효과적"이라는 진짜 신호를 구분한다. Treatment(결정론적, 무작위성 0)의
   std가 대조군 부트스트랩 분포의 낮은 백분위(예: <10%)에 위치하면 "특정성 있는 신호",
   대조군 분포 중앙 근처면 "트레이드 수 감소 자체의 효과와 구분 불가"로 판정한다.

## 출력
tmp/causal_regen_20260516/sol_uptrend_short_seed_variance_diagnostic_20260820/report.json +
stdout 표. fresh_forward_bar_by_bar=true였던 원 ledger를 그대로 재사용하는 순수 사후 재집계이므로
trade_ledgers_used_as_input에 해당하지만, 이 진단 자체가 "promotion/model selection 근거"가
아니라 "메커니즘 이식이 유망한가"를 보는 저비용 탐색이라는 점을 report.json에 명시한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = ROOT / "tmp/causal_regen_20260516/sol_live_promotion_seed_robustness_20260819"
SUMMARY_REPORT_PATH = LEDGER_DIR / "summary_report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sol_uptrend_short_seed_variance_diagnostic_20260820"

SEED_LABELS = ["seed1_live_original", "848498120", "732130789", "193749676", "534479280"]
WINDOWS = ["val", "oos_q1", "oos_q2"]

DETECTOR_WEEK_BARS = 2016  # 1 week of 5m bars -- identical to ETH SustainedUptrendDetector
DETECTOR_PERCENTILE = 0.90  # matches ETH's own calibration percentile choice; the THRESHOLD VALUE itself is recalibrated on SOL's own 2025 data below (see module docstring) -- NOT the ETH fixed constant

N_BOOTSTRAP = 2000
RNG_SEED = 20260820


def load_uptrend_score() -> tuple[pd.Series, float]:
    """Causal rolling(2016).mean(dual_momentum>0) score, indexed by timestamp, spanning SOL's
    full 2025+2026 feature history (contiguous 5-min grid, verified no gaps/dupes below).
    Returns (score_series_indexed_by_timestamp, sol_own_2025_p90_threshold)."""
    frames = []
    for yr in (2025, 2026):
        df = pd.read_csv(
            ROOT / f"data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_{yr}.csv",
            usecols=["timestamp", "dual_momentum"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["_year"] = yr
        frames.append(df)
    full = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if not full["timestamp"].is_unique:
        raise RuntimeError("duplicate timestamps in concatenated SOL feature history")
    gaps = full["timestamp"].diff().dropna()
    if not (gaps == pd.Timedelta("5min")).all():
        raise RuntimeError(f"gap in concatenated SOL feature history ({(gaps != pd.Timedelta('5min')).sum()} non-5min steps) -- rolling(2016) row-count window would be invalid")

    up_bar = (full["dual_momentum"] > 0.0).astype(float)
    score = up_bar.rolling(DETECTOR_WEEK_BARS, min_periods=DETECTOR_WEEK_BARS).mean()

    threshold = float(score.loc[full["_year"] == 2025].quantile(DETECTOR_PERCENTILE))
    score.index = full["timestamp"]
    return score, threshold


def compound_pnl_pct(trade_returns: np.ndarray) -> float:
    """Identical formula to research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py::
    _ledger_metrics -- compound curve over ordered trade_return, pnl = (final - 1) * 100."""
    if len(trade_returns) == 0:
        return 0.0
    curve = np.cumprod(1.0 + trade_returns)
    return float((curve[-1] - 1.0) * 100.0)


def sign_consistent(values) -> bool:
    return len({v >= 0 for v in values}) == 1


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    official = json.loads(SUMMARY_REPORT_PATH.read_text(encoding="utf-8"))

    score, threshold = load_uptrend_score()
    active = score > threshold  # NaN (insufficient warm-up history) compares False, matching the live detector's own "len(window) < week_bars -> False" behavior
    print(f"uptrend threshold (SOL 2025 own p{int(DETECTOR_PERCENTILE*100)}, diagnostic placeholder, NOT ETH's 0.8025793650793651) = {threshold:.6f}")
    print(f"active-bar fraction over full 2025+2026 history = {float(active.mean()):.4f} (n={int(active.notna().sum())})")
    print()

    cells: dict[tuple[str, str], dict] = {}
    mismatches = []
    for seed in SEED_LABELS:
        for window in WINDOWS:
            path = LEDGER_DIR / f"portfolio_ledger_{seed}_{window}.csv"
            ledger = pd.read_csv(path)
            ledger["entry_timestamp"] = pd.to_datetime(ledger["entry_timestamp"])
            ledger = ledger.sort_values("entry_i").reset_index(drop=True)  # already sorted on disk; explicit for safety since compounding is order-sensitive

            mapped_active = ledger["entry_timestamp"].map(active)
            n_unmapped = int(mapped_active.isna().sum())
            ledger["uptrend_active"] = mapped_active.fillna(False).astype(bool)
            is_short = ledger["side"] == -1
            treat_mask = (is_short & ledger["uptrend_active"]).to_numpy()
            controlB_pool_mask = (is_short & (~ledger["uptrend_active"])).to_numpy()  # SHORT trades NOT in an uptrend

            returns = ledger["trade_return"].to_numpy(dtype=np.float64)
            n = len(returns)
            n_removed = int(treat_mask.sum())

            baseline_pnl = compound_pnl_pct(returns)
            treatment_pnl = compound_pnl_pct(returns[~treat_mask])

            official_pnl = official["results"][seed][window]["no_gate"]["pnl"]
            if abs(baseline_pnl - official_pnl) > 1e-6 * max(1.0, abs(official_pnl)):
                mismatches.append((seed, window, baseline_pnl, official_pnl))

            # Control A: remove n_removed uniformly random trades (any side/regime)
            controlA_pnls = np.full(N_BOOTSTRAP, baseline_pnl)
            if n_removed > 0 and n > 0:
                k = min(n_removed, n)
                for i in range(N_BOOTSTRAP):
                    drop_idx = rng.choice(n, size=k, replace=False)
                    keep = np.ones(n, dtype=bool)
                    keep[drop_idx] = False
                    controlA_pnls[i] = compound_pnl_pct(returns[keep])

            # Control B: remove n_removed random trades from the "SHORT but NOT in an uptrend" pool
            poolB_idx = np.flatnonzero(controlB_pool_mask)
            controlB_pnls = np.full(N_BOOTSTRAP, baseline_pnl)
            controlB_note = None
            if n_removed > 0 and len(poolB_idx) > 0:
                k = min(n_removed, len(poolB_idx))
                if k < n_removed:
                    controlB_note = f"pool too small ({len(poolB_idx)} < {n_removed}); sampled all {k} available instead"
                for i in range(N_BOOTSTRAP):
                    drop_idx = rng.choice(poolB_idx, size=k, replace=False)
                    keep = np.ones(n, dtype=bool)
                    keep[drop_idx] = False
                    controlB_pnls[i] = compound_pnl_pct(returns[keep])
            elif n_removed > 0:
                controlB_note = "no non-uptrend SHORT trades available in this cell -- control B == baseline"

            cells[(seed, window)] = dict(
                n_trades=n, n_removed_uptrend_short=n_removed, n_unmapped_timestamps=n_unmapped,
                baseline_pnl=baseline_pnl, treatment_pnl=treatment_pnl,
                controlA_pnls=controlA_pnls, controlB_pnls=controlB_pnls, controlB_note=controlB_note,
            )
            print(f"{seed:22} {window:8} n_trades={n:3d} n_removed_uptrend_short={n_removed:2d} "
                  f"baseline={baseline_pnl:8.2f}%  treatment(minus uptrend-SHORT)={treatment_pnl:8.2f}%")

    if mismatches:
        print("\nWARNING -- baseline recompute did not match summary_report.json's no_gate pnl for:")
        for seed, window, mine, official_v in mismatches:
            print(f"  {seed} {window}: mine={mine} official={official_v}")
    else:
        print("\nOK: recomputed baseline PnL matches summary_report.json's no_gate pnl exactly for all 15 cells (methodology cross-check passed).")

    # ---- per-window aggregation across 5 seeds ----
    summary = {}
    print()
    for window in WINDOWS:
        baseline_v = np.array([cells[(s, window)]["baseline_pnl"] for s in SEED_LABELS])
        treatment_v = np.array([cells[(s, window)]["treatment_pnl"] for s in SEED_LABELS])
        n_removed_v = [cells[(s, window)]["n_removed_uptrend_short"] for s in SEED_LABELS]

        controlA_signcon = np.empty(N_BOOTSTRAP, dtype=bool)
        controlA_std = np.empty(N_BOOTSTRAP)
        controlB_signcon = np.empty(N_BOOTSTRAP, dtype=bool)
        controlB_std = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            va = np.array([cells[(s, window)]["controlA_pnls"][i] for s in SEED_LABELS])
            vb = np.array([cells[(s, window)]["controlB_pnls"][i] for s in SEED_LABELS])
            controlA_signcon[i] = sign_consistent(va)
            controlA_std[i] = va.std(ddof=0)
            controlB_signcon[i] = sign_consistent(vb)
            controlB_std[i] = vb.std(ddof=0)

        treatment_std = float(treatment_v.std(ddof=0))
        summary[window] = dict(
            n_removed_per_seed=n_removed_v,
            baseline_pnls=baseline_v.tolist(), baseline_std=float(baseline_v.std(ddof=0)), baseline_sign_consistent=sign_consistent(baseline_v),
            treatment_pnls=treatment_v.tolist(), treatment_std=treatment_std, treatment_sign_consistent=sign_consistent(treatment_v),
            controlA_std_mean=float(controlA_std.mean()), controlA_std_p10=float(np.percentile(controlA_std, 10)), controlA_std_p50=float(np.percentile(controlA_std, 50)),
            controlA_prob_sign_consistent=float(controlA_signcon.mean()),
            controlA_treatment_std_percentile_rank=float((controlA_std <= treatment_std).mean()),
            controlB_std_mean=float(controlB_std.mean()), controlB_std_p10=float(np.percentile(controlB_std, 10)), controlB_std_p50=float(np.percentile(controlB_std, 50)),
            controlB_prob_sign_consistent=float(controlB_signcon.mean()),
            controlB_treatment_std_percentile_rank=float((controlB_std <= treatment_std).mean()),
        )
        s = summary[window]
        print(f"=== window={window} ===")
        print(f"  n_removed per seed ({SEED_LABELS}): {n_removed_v}")
        print(f"  baseline   pnls={np.round(baseline_v, 2).tolist()} std={s['baseline_std']:.2f} sign_consistent={s['baseline_sign_consistent']}")
        print(f"  treatment  pnls={np.round(treatment_v, 2).tolist()} std={s['treatment_std']:.2f} sign_consistent={s['treatment_sign_consistent']}")
        print(f"  controlA(random N)          std_mean={s['controlA_std_mean']:.2f} (p10={s['controlA_std_p10']:.2f}, p50={s['controlA_std_p50']:.2f})  P(sign_consistent)={s['controlA_prob_sign_consistent']:.3f}  treatment_std_percentile_within_controlA={s['controlA_treatment_std_percentile_rank']:.3f}")
        print(f"  controlB(non-uptrend SHORT) std_mean={s['controlB_std_mean']:.2f} (p10={s['controlB_std_p10']:.2f}, p50={s['controlB_std_p50']:.2f})  P(sign_consistent)={s['controlB_prob_sign_consistent']:.3f}  treatment_std_percentile_within_controlB={s['controlB_treatment_std_percentile_rank']:.3f}")
        print()

    report = dict(
        purpose="cheap post-hoc diagnostic: does excluding SOL SHORT-during-own-uptrend trades reduce cross-seed PnL variance/sign-flips more than excluding a similar count of random or non-uptrend-SHORT trades? NOT a promotion/model-selection claim, NOT a new mechanism build.",
        purely_post_hoc_reaggregation=True, no_new_training=True, no_new_backtest=True, no_new_predictions=True,
        note="all PnLs recomputed by simple cumprod re-aggregation over EXISTING per-trade trade_return values from the already-evaluated N=5 SOL ledgers (scripts/sol_live_promotion_seed_robustness_eval_5seed_20260819.py output, pulled from server via handoff.sh -- not regenerated) -- no new predictions, no new replay. This methodology is diagnostic-only per this repo's Fresh-Forward Validation/OOS/Test Rule (saved trade-ledger replays are diagnostic, not promotion/test evidence).",
        baseline_cross_check="recomputed baseline (no exclusion) PnL compared against summary_report.json's no_gate pnl for all 15 (seed,window) cells" + (" -- MISMATCHES FOUND, see stdout" if mismatches else " -- exact match, all 15 cells"),
        uptrend_detector="rolling(2016bar=1wk).mean(dual_momentum>0), vectorized-identical causal logic to scripts/live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py::SustainedUptrendDetector",
        threshold_definition=f"SOL 2025 full-year p{int(DETECTOR_PERCENTILE * 100)} of the score's own distribution -- DIAGNOSTIC PLACEHOLDER, NOT a production calibration, NOT the ETH fixed constant (0.8025793650793651). ETH itself calibrated on 2025 H1 only; this diagnostic uses SOL's full 2025 per the task instruction.",
        threshold_value=threshold,
        active_bar_fraction_full_history=float(active.mean()),
        n_bootstrap=N_BOOTSTRAP, rng_seed=RNG_SEED,
        seed_labels=SEED_LABELS, windows=WINDOWS,
        ledger_source_dir=str(LEDGER_DIR),
        summary=summary,
    )
    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
