#!/usr/bin/env python3
"""Live TabPFN meta-label inference for the evidence signals upgraded so far
(taker_delta_z_climax, short_term_return_z, liquidity_sweep, orthogonal_combo, smt_divergence,
fib_extension_exhaustion) -- 2026-08-30, user decision: REPLACE these evidence-signal chips in-place (they live in the
"증거 신호" row, not the "모델 내부 지표" row V_REBOUND was added to) with the trained model's own
live probability, rather than adding new standalone indicator-group chips the way
live_eth_sweep_v_rebound_signal_20260829.py did.

2026-08-31: dalton_rule2_balance_edge REMOVED (both its metalabel entry here and its dashboard
chip entirely, per docs/homer/README.md -- consistent negative evidence: lift<1x, 0/96 trailing-
stop cost-gate FAILED). volume_wick_climax was never added here (its rule-based chip was removed
from the dashboard directly, since its Homer upgrade also failed 0/12).

Architecture deliberately differs from that V_REBOUND precedent in one way: this module does NOT
fetch its own klines. dashboard/server.py's load_evidence_signals() already fetches
EVIDENCE_FETCH_LIMIT(1500) closed 5m bars and already calls compute_signals() on them every cache
cycle -- this module is called right after that, reusing both the same klines frame (`df`) and the
same compute_signals() output's raw bottom_/top_ boolean columns (`sig_latest`) as its single
source of truth for "did it fire", so there is no risk of the live model ever disagreeing with the
dashboard's own fired/not-fired state.

Model: same TabPFNClassifier(device="cuda") in-context inference against a FROZEN TRAIN-only
context (ts < 2025-09-01, never extended with VAL/OOS/HOLDOUT, matching V_REBOUND's own
frozen-context design decision for traceability against the validated numbers) for every signal
here. taker_delta_z_climax/short_term_return_z/liquidity_sweep/smt_divergence all share the same
Tier0-style 23 features (build_indicator_frame imported verbatim, not reimplemented) via the
module-level FEATURE_COLUMNS; orthogonal_combo instead uses a 20-feature subset (Tier0 23 minus
the 3 session-timing features nyse_open_flag/hour_utc/weekday, dropped after group ablation showed
they hurt OOS/HOLDOUT) -- see each METALABEL_SIGNALS entry's own `feature_columns` override.
taker_delta_z_climax/liquidity_sweep/orthogonal_combo/smt_divergence have each independently
passed this repo's trailing-stop cost-gate (see their entries below); short_term_return_z has not
yet been tested -- shown for statistical information only until then, per this repo's
dashboard-exposure convention (IC/informativeness, not economic edge; see
feedback_dashboard_indicators_ic_bar_not_pnl_bar).

Fire semantics: the RAW single-bar bottom_/top_ column (not compute_signals()'s "_active" rolling
window) is what triggers a fresh TabPFN call -- the model was trained on cluster-anchored RAW fire
bars, not on an arbitrary bar within a multi-bar "still lit" afterglow window. While the
dashboard's "_active" flag keeps a chip lit for a few bars after the true raw fire (so a
single-bar blink isn't missed), this module keeps returning the SAME cached probability for that
same original fire (keyed by its timestamp) instead of either re-running TabPFN needlessly every
cycle or going blank -- see `_LAST_FIRE_CACHE`.

2026-08-30: the cached proba's validity window (how long it keeps being shown before reverting to
"not fired") now matches each signal's own trained HORIZON in bars, not a flat constant -- the
proba is a real prediction of an outcome up to that many bars forward, so it should stay "live"
for exactly as long as that prediction is actually still unresolved, not an arbitrary shorter
window that goes dark while the thing it predicted hasn't happened yet (user-identified
2026-08-30). See each entry's `horizon_bars` in METALABEL_SIGNALS below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research_eth_candidate_pool_raw_lift_check_20260831 import (  # noqa: E402
    kalman_level_and_velocity,
    rolling_zscore,
)
from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

DEMARKER_FEATURE_COLUMNS = FEATURE_COLUMNS + ["dem"]
KALMAN_FEATURE_COLUMNS = FEATURE_COLUMNS + ["kalman_dev_z"]

# orthogonal_combo v2's final feature set (2026-08-30 ablation): Tier0 23 minus the 3
# session-timing features, which group-ablation showed actively hurt OOS/HOLDOUT (VAL-only
# overfit) -- see research_eth_orthogonal_combo_metalabel_ablation_20260830.py. Every other signal
# below still uses the full FEATURE_COLUMNS (23).
ORTHOGONAL_COMBO_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in ("nyse_open_flag", "hour_utc", "weekday")]

METALABEL_SIGNALS = {
    "taker_delta_z_climax": {
        # 2026-08-30: upgraded v4(CLUSTER_GAP_MERGE=3) -> v5(=12) -- v5 improved VAL/OOS/HOLDOUT
        # AUC (0.622/0.608/0.650 -> 0.633/0.645/0.667) AND trailing-stop economics together, then
        # independently survived its own single HOLDOUT touch (win_rate 64.7%, avg_trade +2.17bp,
        # vs v4's FAILED -0.98bp) -- see eth_taker_delta_climax_trailing_stop_costgate_
        # breakthrough_20260830.md. Only the frozen TRAIN context changes here; features/HORIZON
        # unchanged, so nothing else in this module needs to change.
        "train_context": ROOT / "data/labels/eth_5m_taker_delta_climax_metalabel_v5_gap12_20260830/tabpfn_train_context_frozen_v5_gap12_20260830.csv",
        "seed": 20260829,
        "horizon_bars": 24,  # research_eth_taker_delta_climax_metalabel_tabpfn_20260829.HORIZON (2h) --
                             # must match live_evidence_signal_dashboard_20260823.py's SUSTAIN_BARS_OVERRIDE
        "k": 2.00,  # 2026-08-31: the K actually baked into this train_context CSV's hit column (verified
        "atr_median_bp": 31.5,  # 발동시 ATR 중앙값(2026-09-01 실측, HOLDOUT 이전) -- 저ATR 경고 기준선
                    # empirically: hit==1 move_atr_mult min 2.001 / hit==0 max 1.9998) -- NOT the 2.4 the
                    # cross-signal K-calibration audit found to be the "correct" re-balanced value after
                    # the v4->v5 GAP change; that correction was never redeployed (see
                    # eth_evidence_signal_cross_signal_k_calibration_audit_20260831), so 2.00 is what this
                    # live model's own TP-price math must match to stay consistent with what it was trained on.
    },
    "short_term_return_z": {
        # 2026-09-04 인과 모집단 컨텍스트로 교체 -- 라이브는 raw 단일봉 발동에서 호출되는데 이전 컨텍스트는 클러스터 앵커 봉 학습이라
        # 확률이 과신(캘리브레이션 기울기 <0.6)이었다. 라이브 결정 모집단(같은 측면 raw 발동이 직전 horizon_bars 안에 없는 봉)의
        # TRAIN(<2025-09-01)만으로 재학습. 근거/수치: docs/experiments/eth_evidence_chip_accuracy_upgrade_20260904.md
        # 이전: data/labels/eth_5m_short_term_return_z_metalabel_20260829/tabpfn_train_context_frozen_20260829.csv
        "train_context": ROOT / "data/labels/eth_5m_evidence_chip_causal_20260904/short_term_return_z_train_context_causal_F0_live_20260904.csv",
        "seed": 20260829,
        "horizon_bars": 12,  # research_eth_short_term_return_z_metalabel_tabpfn_20260829.HORIZON (1h) --
                             # must match live_evidence_signal_dashboard_20260823.py's SUSTAIN_BARS_OVERRIDE
        "k": 1.75,  # verified empirically against this train_context CSV: hit==1 move_atr_mult min 1.7522 / hit==0 max 1.7499
        "atr_median_bp": 37.0,  # 발동시 ATR 중앙값(2026-09-01 실측, HOLDOUT 이전) -- 저ATR 경고 기준선
    },
    "liquidity_sweep": {
        # 2026-08-30: standard touch-based-MFE redo (research_eth_liquidity_sweep_topdown_
        # metalabel_{gridscreen,ksweep_tabpfn_confirm,final,holdout}_20260830.py), REPLACING the
        # V_REBOUND-model bridge (live_evidence_signal_liquidity_sweep_metalabel_20260830.py,
        # deleted) that just relayed the specialized "V자 반등락" giveback/confirmed-window model's
        # own probability without retraining. This uses the SAME Tier0+rsi 23-feature schema/
        # FEATURE_COLUMNS as taker/short_term_return_z (unlike V_REBOUND's different is_downside/
        # sweep_penetration_atr/... schema), so it fits this shared build_indicator_frame/
        # row["is_bottom"] path directly -- no other code changes needed, unlike V_REBOUND which
        # required its own standalone module.
        # Label: cluster-anchor bottom_/top_liquidity_sweep (unchanged 48-bar swing-sweep trigger)
        # by deepest sweep penetration (causal, non-circular), HORIZON=30(150min)/GAP=12/K=4.0xATR
        # touch-based MFE (no persistence check, matching taker/short_term_return_z's template, not
        # V_REBOUND's giveback-ratio/excluded-middle design). VAL AUC 0.6587/OOS AUC 0.6372/HOLDOUT
        # AUC 0.6612 (4 seeds each, single HOLDOUT touch -- HOLDOUT the strongest of the three, no
        # degradation) -- beats taker's own 0.622/0.608/0.650. Chart-verified (10 HIT/10 NO_HIT).
        # Trailing-stop cost-gate (SL=4.0x/ARM=2.0x/Trail=0.1x, picked from a 96-combo grid where
        # 91/96 passed VAL+OOS both positive net of 10bp cost) gave VAL +10.70bp/OOS +14.49bp,
        # robust to optimistic/pessimistic intrabar-ordering (~1-1.4bp spread), and survived its
        # single HOLDOUT exposure (+1.97bp, win 67.7%, n=913 trades) -- the strongest economic
        # result of any Homer signal so far. V_REBOUND itself (this signal's predecessor model)
        # FAILED this same gate 0/205 across three different exit structures.
        "train_context": ROOT / "data/labels/eth_5m_liquidity_sweep_topdown_metalabel_20260830/tabpfn_train_context_frozen_liquidity_sweep_topdown_20260830.csv",
        "seed": 20260829,
        "horizon_bars": 30,  # research_eth_liquidity_sweep_topdown_metalabel_final_20260830.HORIZON (150min) --
                             # must match live_evidence_signal_dashboard_20260823.py's SUSTAIN_BARS_OVERRIDE
        "k": 4.00,  # per this signal's own module docstring above (K=4.0xATR) -- this train_context CSV
        "atr_median_bp": 26.4,  # 발동시 ATR 중앙값(2026-09-01 실측, HOLDOUT 이전) -- 저ATR 경고 기준선
                    # doesn't carry a move_atr_mult column to re-derive it from, unlike the other 5 signals.
    },
    "orthogonal_combo": {
        # 2026-08-31: v2, HORIZON=24(2h)/GAP=12, exclude-middle labeling (HIT>=3.571xATR,
        # MISS<=1.786xATR, middle dropped from train/eval -- liquidity_sweep v7b principle, first
        # applied to an evidence signal here after the user found v1's NO_HIT population was
        # concentrated just below the single-K threshold rather than near zero). 20-feature set
        # (ORTHOGONAL_COMBO_FEATURE_COLUMNS, Tier0 23 minus 3 session-timing features that group
        # ablation showed hurt OOS/HOLDOUT) -- see this module's own "feature_columns" override
        # below and research_eth_orthogonal_combo_metalabel_ablation_20260830.py.
        # VAL/OOS/HOLDOUT AUC 0.6844/0.7274/0.7245 (4 seeds, single HOLDOUT touch) -- the best
        # classification result of any Homer signal so far, OOS~=HOLDOUT both above VAL (same
        # trustworthy pattern as taker). Confirmation-leg lift 1.232x (>1, unlike volume_wick_climax/
        # dalton's gate-adds-nothing pattern) -- the delta_z/funding_z OR-condition genuinely adds
        # information on top of the oscillator extremity alone.
        # Trailing-stop cost-gate (SL=4.0x/ARM=0.5x/Trail=0.1x, picked from a 96-combo grid where
        # 73/96 passed VAL+OOS both positive net of 10bp cost, tested against the RAW H=24/GAP=12
        # fire population BEFORE exclude-middle filtering -- see build_eth_orthogonal_combo_raw_
        # fires_H24_GAP12_20260831.py -- since exclude-middle is a classifier-training device, not a
        # property of what actually fires live) gave VAL +9.36bp(win91.5%)/OOS +15.13bp(win96.0%),
        # robust to optimistic/pessimistic intrabar-ordering (~1.0-1.3bp spread), and survived its
        # single HOLDOUT exposure (+3.78bp, win91.5%, n=343 trades) -- beats liquidity_sweep in all
        # three windows, the strongest economic result of any Homer signal so far.
        # 2026-09-04 인과 모집단 컨텍스트로 교체 -- 라이브는 raw 단일봉 발동에서 호출되는데 이전 컨텍스트는 클러스터 앵커 봉 학습이라
        # 확률이 과신(캘리브레이션 기울기 <0.6)이었다. 라이브 결정 모집단(같은 측면 raw 발동이 직전 horizon_bars 안에 없는 봉)의
        # TRAIN(<2025-09-01)만으로 재학습. 근거/수치: docs/experiments/eth_evidence_chip_accuracy_upgrade_20260904.md
        # 이전: data/labels/eth_5m_orthogonal_combo_metalabel_20260830/tabpfn_train_context_frozen_orthogonal_combo_20260831.csv
        "train_context": ROOT / "data/labels/eth_5m_evidence_chip_causal_20260904/orthogonal_combo_train_context_causal_F0_live_20260904.csv",
        "seed": 20260829,
        # 2026-09-04: 인과 컨텍스트(첫 발동 봉 학습)에서는 세션 피쳐 3개를 뺀 20피쳐가 라이브 모집단 AUC 0.585/0.622로 배포 컨텍스트
        # (0.611/0.650)보다 나빴고, Tier0 23피쳐가 0.619/0.667·Brier 0.188/0.211로 유일하게 교체 규칙을 통과했다. 08-30 ablation의
        # "세션 피쳐 = VAL 과적합" 판정은 앵커 모집단 기준이었다. 근거: docs/experiments/eth_evidence_chip_accuracy_upgrade_20260904.md
        "feature_columns": FEATURE_COLUMNS,
        "horizon_bars": 24,  # research_eth_orthogonal_combo_metalabel_tabpfn_20260830.HORIZON (2h) --
                             # must match live_evidence_signal_dashboard_20260823.py's SUSTAIN_BARS_OVERRIDE
        "k": 3.571,  # exclude-middle HIT threshold (K_hi) -- verified empirically against this
        "atr_median_bp": 32.3,  # 발동시 ATR 중앙값(2026-09-01 실측, HOLDOUT 이전) -- 저ATR 경고 기준선
                     # train_context CSV: hit==1 move_atr_mult min 3.5775 / hit==0 max 1.7838. The lower
                     # (K_lo=1.786) excluded-middle boundary has no TP-price meaning, only K_hi does.
    },
    "smt_divergence": {
        # 2026-08-31: v1, HORIZON=72(6h)/GAP=12/K=4.20, plain single-K (NOT exclude-middle -- the
        # ambiguous-middle-concentration check that found orthogonal_combo's problem was re-run
        # here and came back healthy, 24.8% clear-miss fraction). Tier0 23 features unchanged --
        # group ablation found BOTH the vol-regime and session-timing groups are genuine
        # contributors here (removing either hurts VAL/OOS/HOLDOUT), the opposite of
        # orthogonal_combo's session-timing-is-an-overfit-trap finding, so all 23 are kept.
        # VAL/OOS/HOLDOUT AUC 0.6613/0.6253/0.6823 -- the best HOLDOUT AUC of any Homer signal so
        # far, OOS/HOLDOUT both above VAL (same trustworthy pattern as taker/orthogonal_combo).
        # Trailing-stop cost-gate (SL=4.0x/ARM=2.0x/Trail=0.1x, 71/96 grid combos passed VAL+OOS)
        # gave VAL +7.00bp(win72.4%)/OOS +6.18bp(win69.6%), robust to optimistic/pessimistic
        # intrabar-ordering (~1.0-1.1bp spread) and validated against a random-entry baseline
        # (win 64.8-65.2%, avg bp near zero -- confirms the edge is genuine, not an exit-structure
        # artifact, since ARM=2.0 here is much larger than orthogonal_combo's unusually tight 0.5).
        # Survived its single HOLDOUT exposure (+3.24bp, win70.3%, n=646 trades) -- win rate is the
        # most stable across VAL/OOS/HOLDOUT (~70% throughout) of any Homer signal so far.
        "train_context": ROOT / "data/labels/eth_5m_smt_divergence_metalabel_20260831/tabpfn_train_context_frozen_smt_divergence_20260831.csv",
        "seed": 20260829,
        "horizon_bars": 72,  # research_eth_smt_divergence_metalabel_tabpfn_20260831.HORIZON (6h) --
                             # must match live_evidence_signal_dashboard_20260823.py's SUSTAIN_BARS_OVERRIDE
        "k": 4.20,  # verified empirically against this train_context CSV: hit==1 move_atr_mult min
        "atr_median_bp": 24.4,  # 발동시 ATR 중앙값(2026-09-01 실측, HOLDOUT 이전) -- 저ATR 경고 기준선
                    # 4.2059 / hit==0 max 4.1978.
    },
    "fib_extension_exhaustion": {
        # 2026-08-31: v1, HORIZON=20(100min)/GAP=18/K=2.35, Tier0 23 features unchanged (ablation
        # confirmed vol-regime is a real, if unusually dominant, contributor -- kept as-is). Hit
        # label is NOT plain touch-based MFE like every other signal here -- it also caps the
        # maximum adverse excursion in the same window (MFE>=K AND MAE<2.0xK, order-blind) after a
        # user-found example where a fleeting MFE touch coexisted with a much larger same-window
        # reversal. A K-calibration bug (the joint hit_rate(K) curve has an interior peak and
        # crosses 0.5 twice; the fix always takes the larger-K/declining branch) was found and
        # fixed during grid screening -- see research_eth_fib_extension_exhaustion_metalabel_
        # tabpfn_20260831.py.
        # VAL/OOS/HOLDOUT AUC 0.6054/0.6201/0.6210 (4 seeds, single HOLDOUT touch).
        # Trailing-stop cost-gate (SL=3.5x/ARM=0.5x/Trail=0.1x, 9/96 grid combos passed VAL+OOS,
        # this repo's lowest pass rate) gave VAL +15.15bp(win93.2%)/OOS +3.00bp(win87.6%) -- the
        # high win rate is an ARM=0.5 exit-structure artifact (random entries with the same exit
        # also win 82-84%, confirmed via a random-entry baseline run BEFORE trusting the grid
        # result, per the orthogonal_combo lesson), but the avg-bp edge over random (which is
        # NEGATIVE, -2.89/-2.11bp) is genuine. Survived its single HOLDOUT exposure (+2.54bp,
        # win90.6%, n=288 trades).
        "train_context": ROOT / "data/labels/eth_5m_fib_extension_exhaustion_metalabel_20260831/tabpfn_train_context_frozen_fib_extension_exhaustion_20260831.csv",
        "seed": 20260829,
        "horizon_bars": 20,  # research_eth_fib_extension_exhaustion_metalabel_tabpfn_20260831.HORIZON (100min) --
                             # must match live_evidence_signal_dashboard_20260823.py's SUSTAIN_BARS_OVERRIDE
        "k": 2.35,  # verified empirically against this train_context CSV: hit==1 move_atr_mult min
        "atr_median_bp": 26.2,  # 발동시 ATR 중앙값(2026-09-01 실측, HOLDOUT 이전) -- 저ATR 경고 기준선
                    # 2.3510. This signal's label ALSO has a MAE<4.70xATR (2.0xK) cap, but that's a
                    # disqualifying condition on the label, not a second price target -- TP price math
                    # below only ever uses the single profit-target K.
    },
    "demarker_extreme": {
        # 2026-08-31: Homer candidate-pool signal (docs/homer/README.md "후보 풀" table), NOT one
        # of the original 8 -- new evidence-signal slot, not an in-place upgrade. Trigger:
        # DeMarker(14) >= 0.90 (top) / <= 0.10 (bottom), the plain oscillator-extreme reading only
        # -- the original 3-rule design (SMC-divergence/Wyckoff-spring/VP-edge compounds) was
        # abandoned after component-decomposition showed "DeMarker alone" already explained most of
        # each rule's raw lift (eth_demarker_evidence_signal_lift_check_20260831). HORIZON=8(40min)/
        # GAP=12/K=0.70 (touch-based MFE, no persistence -- plain, NOT exclude-middle: a v2
        # ambiguous-middle design was tried and dropped after a GBM re-check showed it doesn't beat
        # plain at either signal's optimum, see eth_kalman_demarker_horizon_gap_k_screening_20260831).
        # VAL/OOS/HOLDOUT AUC 0.7527/0.7157/0.7464 (4 seeds, single HOLDOUT touch) -- the best
        # classification result of any Homer signal so far, holdout barely below VAL. Permutation
        # importance found bb_pctb (Bollinger %B), NOT dem itself (alone: AUC 0.51, uninformative),
        # is the actual classification driver (bb_pctb alone: 0.7333) -- confirmed NOT a lookahead/
        # contamination bug via a dedicated line-by-line audit (eth_kalman_demarker_lookahead_
        # audit_20260831) -- the trigger condition itself remains independently valid (raw lift +
        # unconditional-on-proba economics gate below), same "trigger valid, confidence driven by a
        # different feature" pattern as dalton_rule2_balance_edge.
        # Trailing-stop cost-gate (SL=2.0x/ARM=1.5x/Trail=0.1x, 96/96 grid combos passed VAL+OOS --
        # this repo's best pass rate) gave VAL +12.14bp(win70.7%)/OOS +20.20bp(win80.0%, the best OOS
        # bp of any Homer signal so far), robust to optimistic/pessimistic intrabar-ordering (<0.3bp
        # spread), and survived its single HOLDOUT exposure (+11.53bp, win77.9%, n=420 trades) with
        # almost no shrinkage from VAL/OOS -- the most stable economic result of any Homer signal so far.
        "train_context": ROOT / "data/labels/eth_5m_kalman_demarker_metalabel_20260831/tabpfn_train_context_frozen_demarker_extreme_20260831.csv",
        "seed": 20260829,
        "feature_columns": DEMARKER_FEATURE_COLUMNS,
        "horizon_bars": 8,  # research_eth_kalman_demarker_gridscreen_20260831.SIGNAL_CONFIG["demarker_extreme"]["horizon"] (40min) --
                            # must match live_evidence_signal_dashboard_20260823.py's SUSTAIN_BARS_OVERRIDE
        "k": 0.70,  # this train_context CSV's own hit column threshold (peak>=0.70xATR) -- see
        "atr_median_bp": 32.6,  # 발동시 ATR 중앙값(2026-09-01 실측, HOLDOUT 이전) -- 저ATR 경고 기준선
                    # eth_kalman_demarker_horizon_gap_k_screening_20260831 for the full K-sweep history.
    },
    "kalman_deviation_meanrev": {
        # 2026-08-31: Homer candidate-pool signal, NOT one of the original 8. Trigger: (close -
        # kalman-filtered trend level)/level, rolling-288-bar z-scored (same recipe as delta_z/
        # vol_z/ret3_z), >= 2.0 (top) / <= -2.0 (bottom) -- features/engineering.py::
        # _kalman_trend_velocity's exact state-space model (F/H/Q/R, obs_noise=1e-3/proc_noise=1e-5),
        # extended to also keep the level state x[0] (the live feature only returns velocity x[1]).
        # HORIZON=12(1h)/GAP=12(wider than the 3-bar draft -- this signal's own clustering is much
        # tighter than DeMarker's, confirmed by the HORIZON x GAP grid screen)/K=2.5 (touch-based
        # MFE, no persistence, plain not exclude-middle -- same v2-dropped history as demarker_extreme
        # above). VAL/OOS/HOLDOUT AUC 0.6569/0.6311/0.6284 (4 seeds, single HOLDOUT touch), holdout
        # close to OOS. Permutation importance here is broadly distributed (nyse_open_flag top at
        # only +0.018, kalman_dev_z itself low but positive) -- the healthy pattern, unlike
        # demarker_extreme's single-feature dominance.
        # Trailing-stop cost-gate (SL=4.0x/ARM=1.5x/Trail=0.1x, 96/96 grid combos passed VAL+OOS) gave
        # VAL +10.26bp(win71.2%)/OOS +11.00bp(win71.0%), robust to optimistic/pessimistic
        # intrabar-ordering (~1bp spread), and survived its single HOLDOUT exposure (+5.80bp,
        # win71.8%, n=925 trades) -- roughly half of VAL/OOS but clearly positive.
        # 2026-09-04 인과 모집단 컨텍스트로 교체 -- 라이브는 raw 단일봉 발동에서 호출되는데 이전 컨텍스트는 클러스터 앵커 봉 학습이라
        # 확률이 과신(캘리브레이션 기울기 <0.6)이었다. 라이브 결정 모집단(같은 측면 raw 발동이 직전 horizon_bars 안에 없는 봉)의
        # TRAIN(<2025-09-01)만으로 재학습. 근거/수치: docs/experiments/eth_evidence_chip_accuracy_upgrade_20260904.md
        # 이전: data/labels/eth_5m_kalman_demarker_metalabel_20260831/tabpfn_train_context_frozen_kalman_deviation_meanrev_20260831.csv
        "train_context": ROOT / "data/labels/eth_5m_evidence_chip_causal_20260904/kalman_deviation_meanrev_train_context_causal_F0_live_20260904.csv",
        "seed": 20260829,
        "feature_columns": KALMAN_FEATURE_COLUMNS,
        "horizon_bars": 12,  # research_eth_kalman_demarker_gridscreen_20260831.SIGNAL_CONFIG["kalman_deviation_meanrev"]["horizon"] (1h) --
                             # must match live_evidence_signal_dashboard_20260823.py's SUSTAIN_BARS_OVERRIDE
        "k": 2.5,  # this train_context CSV's own hit column threshold (peak>=2.5xATR).
        "atr_median_bp": 36.6,  # 발동시 ATR 중앙값(2026-09-01 실측, HOLDOUT 이전) -- 저ATR 경고 기준선
    },
}

_TRAIN_CACHE: dict[str, pd.DataFrame] = {}
_LAST_FIRE_CACHE: dict[str, dict] = {}  # signal_name -> {"bar_ts": Timestamp, "side": str, "proba": float, "tp_price": float|None}


def _load_train_context(signal_name: str) -> pd.DataFrame:
    if signal_name not in _TRAIN_CACHE:
        _TRAIN_CACHE[signal_name] = pd.read_csv(METALABEL_SIGNALS[signal_name]["train_context"], parse_dates=["timestamp"])
    return _TRAIN_CACHE[signal_name]


def _tp_price(entry_price: float, atr_pct: float, k: float, side: str) -> float | None:
    """Take-profit price implied by this signal's OWN trained label definition: entry (the fire
    bar's own close, exactly as every offline label script uses) moved by k*atr_pct (the fire bar's
    own atr_pct, exactly as move_atr_mult = pred_dir_ret/atr_pct was computed at training time) --
    the same price level the model's "hit" target represents, not this repo's separate trailing-
    stop economics config (SL/ARM/Trail), which is a different, later-added execution layer."""
    if pd.isna(entry_price) or pd.isna(atr_pct):
        return None
    move = k * atr_pct
    return float(entry_price * (1 + move)) if side == "bottom" else float(entry_price * (1 - move))


def _tp_touched(df: pd.DataFrame, fire_pos: int, tp_price: float | None,
                side: str) -> bool | None:
    """발동 이후 **익절가에 닿았는가**(봉 고가/저가 기준).

    ⚠️2026-09-03 신설. 그 전까지 칩은 발동 후 `horizon_bars` 동안 무조건 유지됐고
    (smt_divergence는 **6시간**, taker/orthogonal은 2시간), **익절가 도달 여부를 전혀 보지
    않았다.** 그래서 목표가 이미 달성된 뒤에도 "신호 활성 · 익절 XXXX"를 계속 띄웠다 --
    이걸 보고 뒤늦게 진입하면 손해다.

    ⚠️이건 **표시 전용 사실**이다. 각 신호의 학습 라벨 HIT 정의와 반드시 같지는 않다
    (일부 신호는 종가 기준으로 라벨링됐다). 여기서는 사람이 화면을 보고 판단할 때 의미 있는
    "가격이 목표에 닿았나"를 고가/저가로 답한다 -- 닿음을 **더 이르게** 잡는 쪽이므로
    늦은 진입을 막는 안전한 방향이다. 모델 확률도 발동 여부도 바꾸지 않는다.
    """
    if tp_price is None or fire_pos is None or not np.isfinite(tp_price):
        return None
    seg = df.iloc[fire_pos + 1:]                      # 발동봉 다음 봉부터(라벨 컨벤션과 동일)
    if seg.empty:
        return False
    if side == "bottom":
        return bool((seg["high"].to_numpy(dtype=float) >= tp_price).any())
    return bool((seg["low"].to_numpy(dtype=float) <= tp_price).any())


def _find_recent_raw_fire_pos(col: pd.Series, horizon_bars: int, n: int) -> int | None:
    """Most recent True position (0-based, positional) in `col` within the trailing horizon_bars
    window ending at n-1 (inclusive) -- used to recover a raw fire this process never saw fresh
    (see compute_evidence_signal_metalabels's restart-recovery note)."""
    lo = max(0, n - horizon_bars)
    arr = col.iloc[lo:n].fillna(False).to_numpy(dtype=bool)
    true_idx = np.flatnonzero(arr)
    return int(lo + true_idx[-1]) if len(true_idx) else None


def compute_evidence_signal_metalabels(df: pd.DataFrame, sig: pd.DataFrame) -> dict[str, dict]:
    """`df`: the same closed-bar ETH klines frame load_evidence_signals() already fetched this
    cycle (timestamp/open/high/low/close/volume/taker_buy_base -- confirmed sufficient, the Tier0
    indicator chain reads only these raw columns). `sig`: the FULL frame from the SAME
    compute_signals() call already made this cycle (2026-08-31: widened from just sig.iloc[-1] to
    the whole frame -- see restart-recovery note below, which needs to look backward through the
    raw bottom_/top_ history, not just the latest bar). Returns
    {signal_name: {"fired": bool, "side": "bottom"|"top"|None, "proba": float|None,
    "tp_price": float|None}}. tp_price (2026-08-31) is the price level implied by this signal's own
    trained K*ATR% target (see _tp_price) -- frozen at the original fire bar's entry/ATR, same
    caching lifecycle as proba, not recomputed as price moves.

    2026-08-30 (dalton_rule2_balance_edge addition): restructured to check cache VALIDITY first,
    before deciding whether to run fresh inference -- previously, any bar where the raw fire
    condition was still true triggered a brand-new TabPFN call, even on the 2nd/3rd/... bar of the
    same still-firing streak. That's a harmless approximation for taker/short_term_return_z (whose
    raw fires are rarely more than a couple of consecutive bars) but was never correct: none of
    these signals' offline labels were built that way -- cluster-anchoring always collapses a
    same-side streak to ONE representative bar (taker/short_term_return_z: the most extreme
    z-score bar; dalton: the run's first bar) and trains on that bar's features alone. Re-inferring
    on every bar of an ongoing streak evaluates the model on bars it was never trained to
    represent. dalton_rule2_balance_edge makes this concrete: its raw condition is a regime STATE
    that can stay true for up to 32-40 bars (160-200min) at a stretch, far longer than taker/
    short_term_return_z's typical streaks -- re-inferring on every one of those bars would be both
    wasteful (dozens of needless GPU calls per episode) and a real train/serve mismatch. Fix: only
    run fresh inference when there is no still-valid (same-side, not yet past horizon_bars) cached
    proba for this signal -- covers a genuine new fire, a side flip, AND the case where an unusually
    long-lived state outlasts the model's own horizon_bars (treated as a fresh decision point once
    the original anchor's prediction window has fully elapsed). Otherwise reuse the cached proba,
    whether this bar is mid-streak (raw condition still true) or in the post-fire afterglow window
    (raw condition already false again). Backward-compatible for taker/short_term_return_z -- this
    only changes behavior on the rare bars where their raw condition was already true on the
    immediately preceding bar too, and in that direction it strictly improves train/serve fidelity.

    2026-08-31 (user-reported gap): `_LAST_FIRE_CACHE` lives only in process memory -- a dashboard
    restart wipes it. dashboard/server.py's own `bottom_{name}_active`/`top_{name}_active` fields
    (a rolling-max of the RAW column over horizon_bars, computed fresh from stored kline history
    every cycle) keep a chip lit correctly across a restart, but this function's OWN "is this bar's
    raw condition true" check only looks at the CURRENT bar -- if the true raw fire happened before
    this process started and has since rolled off (current bar's raw condition is already False
    again), a restarted process has no memory of it and reports fired=False/proba=None/tp_price=None
    even while the dashboard chip still correctly shows "발동" from the persisted history. Observed
    live for kalman_deviation_meanrev (and earlier taker_delta_z_climax) shortly after a restart.
    Fix: when the current bar isn't raw-firing and no valid cache exists, scan BACKWARD through the
    raw bottom_/top_ column itself (available now that this function receives the full `sig` frame,
    not just its last row) for the most recent True bar within horizon_bars, and if found, run
    inference AT THAT bar's own features (matching how every offline label script defines entry --
    the anchor bar's own close/atr_pct, not "now") exactly as if it had just fired fresh. This is a
    read-only recovery (same NaN/inference/caching code as a fresh fire), not a new firing rule."""
    n = len(sig)
    if n == 0:
        return {name: {"fired": False, "side": None, "proba": None, "tp_price": None, "atr_bp": None,
                       "atr_median_bp": cfg.get("atr_median_bp"), "low_atr": None,
                       "tp_touched": None, "bars_since_fire": None}
                for name, cfg in METALABEL_SIGNALS.items()}
    latest_ts = pd.Timestamp(sig["timestamp"].iloc[-1])
    _ts_index = {pd.Timestamp(t): i for i, t in enumerate(sig["timestamp"])}

    def _pos_of_ts(t) -> int | None:
        return _ts_index.get(pd.Timestamp(t))

    indicator_frame = None  # built lazily, only if at least one signal actually needs a fresh call

    def _ensure_indicator_frame():
        nonlocal indicator_frame
        if indicator_frame is None:
            indicator_frame = build_indicator_frame(df)
            # demarker_extreme/kalman_deviation_meanrev's own 24th feature -- not part of the
            # shared Tier0 FEATURE_COLUMNS bank, computed once here (cheap, pure numpy/pandas,
            # no GPU) regardless of which signal actually triggered this cycle's fresh call.
            indicator_frame["dem"] = compute_demarker(df["high"], df["low"]).to_numpy()
            levels, _ = kalman_level_and_velocity(df["close"].to_numpy())
            kalman_dev = (df["close"].to_numpy() - levels) / levels
            indicator_frame["kalman_dev_z"] = rolling_zscore(pd.Series(kalman_dev)).to_numpy()
        return indicator_frame

    def _low_atr_fields(signal_name: str, atr_pct: float) -> dict:
        """저ATR 경고 (2026-09-01 신설). 발동봉의 ATR이 이 신호 자신의 발동시 ATR 중앙값보다
        낮으면 low_atr=True. 근거: 저변동성 구간에서는 SL/ARM/Trail이 전부 ATR 배수로 줄어드는데
        왕복비용 10bp는 고정이라, 방향이 맞아도 수수료를 못 넘기는 비율이 커진다 -- 실측상
        '방향정확도 - 수익승률' 격차가 fib 23.0pp / kalman 5.3pp / demarker 5.1pp였다
        (docs/homer/evidence_signal_economics_tuning_protocol.md).

        ⚠️이건 표시 전용 정보다. 모델 확률(proba)은 전혀 건드리지 않으며, 발동 여부도 안 바꾼다."""
        median_bp = METALABEL_SIGNALS[signal_name].get("atr_median_bp")
        if atr_pct is None or not np.isfinite(atr_pct) or median_bp is None:
            return {"atr_bp": None, "atr_median_bp": median_bp, "low_atr": None}
        atr_bp = float(atr_pct) * 1e4
        return {"atr_bp": round(atr_bp, 1), "atr_median_bp": median_bp, "low_atr": bool(atr_bp < median_bp)}

    def _infer_at(signal_name: str, pos: int, side: str, bar_ts: pd.Timestamp) -> dict:
        frame = _ensure_indicator_frame()
        row = frame.iloc[pos].copy()
        row["is_bottom"] = 1 if side == "bottom" else 0
        tp_price = _tp_price(row["close"], row["atr_pct"], METALABEL_SIGNALS[signal_name]["k"], side)
        atr_fields = _low_atr_fields(signal_name, row["atr_pct"])
        feature_cols = METALABEL_SIGNALS[signal_name].get("feature_columns", FEATURE_COLUMNS)
        if row[feature_cols].isna().any():
            return {"fired": True, "side": side, "proba": None, "tp_price": tp_price,
                    "fire_pos": pos, **atr_fields}
        proba = _predict_proba(signal_name, row)
        _LAST_FIRE_CACHE[signal_name] = {"bar_ts": bar_ts, "side": side, "proba": proba,
                                          "tp_price": tp_price, "fire_pos": pos, **atr_fields}
        return {"fired": True, "side": side, "proba": proba, "tp_price": tp_price,
                "fire_pos": pos, **atr_fields}

    out: dict[str, dict] = {}
    for signal_name in METALABEL_SIGNALS:
        bcol, tcol = f"bottom_{signal_name}", f"top_{signal_name}"
        bottom_fired_now = bool(sig[bcol].iloc[-1]) if bcol in sig.columns else False
        top_fired_now = bool(sig[tcol].iloc[-1]) if tcol in sig.columns else False
        horizon_bars = METALABEL_SIGNALS[signal_name]["horizon_bars"]
        cached = _LAST_FIRE_CACHE.get(signal_name)

        if bottom_fired_now or top_fired_now:
            side = "bottom" if bottom_fired_now else "top"
            cache_valid = (
                cached is not None and cached["side"] == side
                and 0 <= (latest_ts - cached["bar_ts"]).total_seconds() / 300 < horizon_bars
            )
            if cache_valid:
                out[signal_name] = {"fired": True, "side": side, "proba": cached["proba"], "tp_price": cached.get("tp_price"), "fire_pos": _pos_of_ts(cached["bar_ts"]), "atr_bp": cached.get("atr_bp"), "atr_median_bp": cached.get("atr_median_bp"), "low_atr": cached.get("low_atr")}
                continue
            out[signal_name] = _infer_at(signal_name, n - 1, side, latest_ts)
            continue

        # not currently firing on THIS bar -- afterglow window from a previous fire this process
        # itself already saw, if still within horizon_bars
        if cached is not None and 0 <= (latest_ts - cached["bar_ts"]).total_seconds() / 300 < horizon_bars:
            out[signal_name] = {"fired": True, "side": cached["side"], "proba": cached["proba"], "tp_price": cached.get("tp_price"), "fire_pos": _pos_of_ts(cached["bar_ts"]), "atr_bp": cached.get("atr_bp"), "atr_median_bp": cached.get("atr_median_bp"), "low_atr": cached.get("low_atr")}
            continue

        # restart-recovery: no live memory of a fire, but one may still be sitting in the raw
        # history within horizon_bars (see docstring above) -- recover it exactly like a fresh fire.
        bottom_pos = _find_recent_raw_fire_pos(sig[bcol], horizon_bars, n) if bcol in sig.columns else None
        top_pos = _find_recent_raw_fire_pos(sig[tcol], horizon_bars, n) if tcol in sig.columns else None
        if bottom_pos is None and top_pos is None:
            out[signal_name] = {"fired": False, "side": None, "proba": None, "tp_price": None, "atr_bp": None, "atr_median_bp": METALABEL_SIGNALS[signal_name].get("atr_median_bp"), "low_atr": None}
            continue
        if top_pos is None or (bottom_pos is not None and bottom_pos >= top_pos):
            anchor_pos, anchor_side = bottom_pos, "bottom"
        else:
            anchor_pos, anchor_side = top_pos, "top"
        anchor_ts = pd.Timestamp(sig["timestamp"].iloc[anchor_pos])
        out[signal_name] = _infer_at(signal_name, anchor_pos, anchor_side, anchor_ts)

    # ── 라벨 확정 후처리 (2026-09-03) ──
    # 출력 경로가 6개(워밍업/신규발동/캐시재사용/afterglow/재시작복구/미발동)라 각각에 넣지 않고
    # 여기서 한 번에 채운다. 경로가 늘어도 빠뜨리지 않는다.
    #
    # ⭐`fired`는 `compute_signals()`의 `_active` 컬럼을 **그대로 따른다**. 그 컬럼은
    # 2026-09-03부터 각 신호 자신의 라벨 확정 규칙(터치 / 터치+MAE)을 쓴다 -- 예전처럼
    # 고정 봉수가 아니다. 여기서 독립적으로 판단하면 칩 점등과 이 확률 표시가 어긋날 수 있는데,
    # 그건 화면에서 가장 헷갈리는 종류의 불일치다. 단일 출처를 강제한다.
    # ⚠️`fired=False`가 되어도 tp_price/tp_touched/bars_since_fire는 남긴다 -- 칩이 왜 꺼졌는지
    #   ("목표 도달") 화면에 설명할 수 있어야 하기 때문이다.
    for name, o in out.items():
        # 2026-09-06: 잔여 호라이즌 표시용. 발동 봉 확률은 호라이즌 내내 재사용되는데(애프터글로우 캐시)
        # 실측상 순위 AUC가 0.70 -> 0.60~0.64로 낡고 수준은 9~12봉 뒤 실제의 1.5배로 과대해진다
        # (docs/experiments/eth_composite_direction_trend_pullback_results_20260905.md 부록 22).
        # 모델은 그대로 두고, 화면이 "언제 값인지 · 얼마나 남았는지"를 말할 수 있게 상수만 실어 보낸다.
        o["horizon_bars"] = METALABEL_SIGNALS[name]["horizon_bars"]
        fp = o.pop("fire_pos", None)
        if not o.get("fired"):
            o["tp_touched"] = None
            o["bars_since_fire"] = None
            continue
        o["tp_touched"] = _tp_touched(df, fp, o.get("tp_price"), o.get("side"))
        o["bars_since_fire"] = (n - 1 - fp) if fp is not None else None
        side = o.get("side")
        acol = f"{'bottom' if side == 'bottom' else 'top'}_{name}_active"
        if acol in sig.columns and not bool(sig[acol].iloc[-1]):
            o["fired"] = False        # 라벨이 확정됨 -- 더 이상 "지금 유효한 증거"가 아니다
    return out


def _predict_proba(signal_name: str, feature_row: pd.Series) -> float:
    from tabpfn import TabPFNClassifier

    cfg = METALABEL_SIGNALS[signal_name]
    feature_cols = cfg.get("feature_columns", FEATURE_COLUMNS)
    train = _load_train_context(signal_name)
    clf = TabPFNClassifier(device="cuda", random_state=cfg["seed"])
    clf.fit(train[feature_cols], train["hit"].to_numpy().astype(int))
    X = feature_row[feature_cols].to_frame().T
    return float(clf.predict_proba(X)[:, 1][0])
