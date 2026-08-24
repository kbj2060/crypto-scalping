# ETH Omega4.6.1 dedup 78-feature N=5 시드 direction_head 스킬 재검증 (h48qual + zig075, 2026-08-15)

## 배경

`docs/experiments/eth_omega461_live_102feature_redundancy_audit_20260815.md`가 live h48qual/
zig075 3-Head TabM 번들의 102개 공유 `base_cols`(두 컴포넌트 완전 동일, 순서까지 일치) 중
38개(14개 클러스터, |corr|>0.9 Spearman)가 사실상 중복 정보임을 확인했다(`smart_money_flow`≡
`oi_change_rate`, `funding_z_score`≡`ou_funding_z` 등 r=1.0000 완전중복 포함). 같은 세션에서
`docs/experiments/eth_omega461_zig075_direction_head_skill_formal_nseed_20260815.md`가 N=5
진짜 다양시드(`random.SystemRandom()`)로 zig075의 ungated `direction_head`가 정확히 102개
feature 세트에서 VAL/OOS 10칸 중 10칸 모두 always_short에 진다는 것을 formal 확정했다.
h48qual의 동일한 no-skill 벽은 Odyssey1(30개 이상 변형, 38/40 손)에서 이미 확정된 바 있다.

이 문서는 사용자 요청에 따라 "중복 feature를 제거하면 결론이 바뀌는가?"를 직접 테스트한다:
14개 클러스터에서 대표 1개씩만 남겨 102→78 feature로 줄인 뒤, h48qual/zig075 둘 다 동일한
5개 시드로 재학습하고 동일한 스킬 테스트를 재실행한다.

**주의**: Odyssey1에서 이미 FINAL12/h384/REL11/alt-data 등 30개 이상의 feature-set 변형이 같은
no-skill 벽을 넘지 못했으므로, 이번 결과도 그 패턴을 재확인할 가능성이 높다고 사전에 예상했다
(기대치 조정). 이번 실험의 고유 가치는 "오늘 감사에서 확인된 정확한 live-102-feature-minus-
중복" 조합이 지금까지 한 번도 테스트된 적이 없었다는 점이다.

## 1. 78-feature 축소 목록 도출 (사전 확정 규칙, 커닝 없음)

### 규칙

각 클러스터에서 **live 번들 `base_cols`의 원래 순서상 가장 먼저 등장하는 멤버 1개만 남긴다**
(단순·결정론적·사전 확정 — 어떤 학습/backtest 결과도 보기 전에 정한 규칙). 클러스터 소속이
아닌 나머지 64개는 그대로 유지.

`redundancy_analysis.py`(감사 문서가 재현용으로 남긴 스크립트, 이 세션 scratchpad에 그대로
존재)를 재실행해 14개 클러스터/38개 feature를 다시 계산했다(감사 문서 본문 표는 "나머지 8개
클러스터"라고 서술했으나 실제로는 9개였음 — 이 문서에서 스크립트 재실행으로 직접 확인한 값을
사용, 클러스터 개수 14/포함 feature 38 자체는 감사 문서 요약과 일치).

### 클러스터별 대표 선정 (14개, base_cols 인덱스 기준)

| 클러스터 | 멤버(전체) | 대표(유지, 굵게=최소 index) | 제거 |
|---|---|---|---|
| OHLC+OI | close(3), high(1), low(2), open(0), sum_open_interest_value(9) | **open**(0) | high, low, close, sum_open_interest_value |
| 거래량군 | quote_volume(5), taker_buy_base(7), taker_buy_quote(8), trades(6), volume(4) | **volume**(4) | quote_volume, taker_buy_base, taker_buy_quote, trades |
| 펀딩군 | funding_abs(79), last_funding_rate(12), long_squeeze_risk(66), squeeze_power(19) | **last_funding_rate**(12) | funding_abs, long_squeeze_risk, squeeze_power |
| 변동성 추정량 3종 | garman_klass_vol(33), parkinson_vol(38), rogers_satchell_vol(37) | **garman_klass_vol**(33) | parkinson_vol, rogers_satchell_vol |
| mean-reversion 계열 | fibonacci_level(61), mean_reversion_z(58), volume_profile_signal(60) | **mean_reversion_z**(58) | fibonacci_level, volume_profile_signal |
| btc거래량 | quote_volume_btc(15), volume_btc(14) | **volume_btc**(14) | quote_volume_btc |
| OI변화 | oi_change_rate(20), smart_money_flow(18) | **smart_money_flow**(18) | oi_change_rate |
| 수익률 | jump_z(85), log_return(25) | **log_return**(25) | jump_z |
| 추세 | mtf_trend_4h(36), rsi(27) | **rsi**(27) | mtf_trend_4h |
| 기울기 | hma_slope(31), kalman_velocity(73) | **hma_slope**(31) | kalman_velocity |
| 브레이크아웃 | breakout_strength(59), cvp_volume_imbalance(54) | **cvp_volume_imbalance**(54) | breakout_strength |
| 펀딩z | funding_z_score(65), ou_funding_z(82) | **funding_z_score**(65) | ou_funding_z |
| EVT | evt_excess_z(87), evt_tail_flag(86) | **evt_tail_flag**(86) | evt_excess_z |
| regime3 confidence | regime3_..._confidence(99), regime3_..._margin(101) | **regime3_..._confidence**(99) | regime3_..._margin |

14개 대표 유지 + 24개 제거(38−14=24) → **102−24=78 feature**.

### 전체 목록 (재현 검증용, 순서는 원 base_cols 순서 유지)

**제거된 24개** (알파벳순):
```
breakout_strength, close, evt_excess_z, fibonacci_level, funding_abs, high, jump_z,
kalman_velocity, long_squeeze_risk, low, mtf_trend_4h, oi_change_rate, ou_funding_z,
parkinson_vol, quote_volume, quote_volume_btc,
regime3_current_sensitive_wide24_margin, rogers_satchell_vol, squeeze_power,
sum_open_interest_value, taker_buy_base, taker_buy_quote, trades, volume_profile_signal
```

**유지된 78개** (base_cols 원 순서):
```
open, volume, sum_toptrader_long_short_ratio, count_long_short_ratio, last_funding_rate,
close_btc, volume_btc, whale_retail_ratio, whale_conviction, smart_money_flow,
net_taker_ratio, taker_acceleration, trade_intensity, big_trade_ratio, log_return,
volatility_z, rsi, macd_hist, bb_width, bb_width_z, hma_slope, wick_ratio,
garman_klass_vol, realized_vol_ratio, mtf_trend_1h, amihud_illiquidity_z, btc_corr_60,
eth_btc_ratio_change, fvg_dist, chop_index, hour_sin, hour_cos, minute_sin, minute_cos,
session_europe, session_us, is_hour_open, cvp_poc_dist, cvp_vah_val_width,
cvp_cluster_position, cvp_volume_imbalance, cvp_regime, turtle_signal, dual_momentum,
mean_reversion_z, funding_roc_12, funding_roc_48, funding_roc_288, funding_z_score,
short_squeeze_risk, funding_price_divergence, hurst_48, hurst_288, regime_trending,
ofi_acceleration, realized_skewness, ofti, kel, mta_funding, svps, funding_pressure,
garch_vol_z, ou_halflife, jump_flag, evt_tail_flag, sig_volume_confirm,
sig_liquidity_trap, sig_trend_health, regime_persistence, cross_scale_curvature,
liquidity_vacuum, crowding_pressure, execution_quality,
regime3_current_sensitive_wide24_bull_prob, regime3_current_sensitive_wide24_bear_prob,
regime3_current_sensitive_wide24_chop_prob, regime3_current_sensitive_wide24_confidence,
regime3_current_sensitive_wide24_entropy
```

두 컴포넌트(h48qual/zig075)가 원래 102개를 완전히 공유했으므로, 이 78개 목록도 두 컴포넌트에
동일하게 적용된다(따로 계산하지 않음).

## 2. 레시피 fork

`scripts/train_eval_omega4_3head_parent72_pinned102_20260727.py`(기존 zig075 5-시드 formal
테스트가 쓴 wrapper, 라이브 번들 `base_cols`를 그대로 pin)를 참고해
`scripts/train_eval_omega4_3head_parent72_pinned78_20260815.py`를 새로 작성했다. 핵심 차이는
`_numeric_feature_cols` 몽키패치가 라이브 번들에서 읽은 102개가 아니라 위 78개
`REDUCED_78_COLS`(스크립트에 하드코딩, 이 문서 표와 동일 값)를 반환한다는 점뿐이다.

2025 학습 프레임 컬럼 복구 로직(`_load_omega_frames` 몽키패치)도 그대로 재사용하되, pinned102가
복구하던 7개(`fibonacci_level, funding_roc_12, funding_roc_48, funding_z_score,
short_squeeze_risk, hurst_288, regime_persistence`) 중 `fibonacci_level`은 이번 dedup으로
자체가 제거 대상이라 더 이상 필요 없어 REPAIR_COLS에서 뺐다(6개만 복구). 그 외
아키텍처(k=8/hidden=192/layers=3/dropout=0.08), 라벨, epoch, `quality_mode`, `exit_label_mode`
등은 원본 트레이너(`scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`)
그대로, 인자로만 제어 — feature 목록 하나만 바뀐 단일변수 비교.

**학습 커맨드**(컴포넌트별 정확히 배포 레시피 그대로, `--seed`만 5개 값으로 교체):

zig075:
```
python scripts/train_eval_omega4_3head_parent72_pinned78_20260815.py \
  --epochs 2 --quality-mode same_as_direction \
  --direction-label-dir tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531 \
  --quality-thresholds 0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 \
  --max-exit-samples 30000 --max-train-rows 0 \
  --exit-label-mode entry_label_terminal_giveback \
  --out-suffix pinned78_zig075_dedup_seed<SEED> --device cpu --seed <SEED>
```

h48qual:
```
python scripts/train_eval_omega4_3head_parent72_pinned78_20260815.py \
  --epochs 2 --max-train-rows 0 --max-exit-samples 30000 \
  --quality-thresholds 0.50 --exit-label-mode entry_label_terminal_giveback \
  --direction-label-dir tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531 \
  --quality-mode quality_label_action \
  --quality-label-dir tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps \
  --out-suffix pinned78_h48qual_dedup_seed<SEED> --device cpu --seed <SEED>
```

(h48qual 레시피는 `.../pinned102_20260727.py` 자체 docstring의 h48qual-control 사용례,
그리고 라이브 h48qual 번들 `report.json`의 `label_contract`/`exit_label`과 대조해 확인 —
`quality_threshold=0.50`도 `trading_bot_modules/omega4_6_1_live.py:289`의 live 값과 일치.)

드라이버: `scripts/run_dedup78_nseed_dev_20260815.sh`(10개 run 순차 — zig075/h48qual 페어를
시드마다 번갈아 실행).

## 3. 시드: 기존 5개 재사용 (신규 추첨 아님)

`946043153, 932925759, 74851798, 975176982, 542143953` — zig075 102-feature formal 테스트가
이미 `random.SystemRandom().sample(range(1, 1_000_000_000), 5)`로 뽑은 바로 그 5개를 그대로
재사용했다.

**왜 재사용이 여기서는 적절한가**: 이번 실험의 목적은 "102개 vs 78개, feature 세트만 바꿨을 때
결론이 달라지는가"를 보는 **paired/matched-seed ablation**이지, 독립적인 새 Seed-Diversity
Gate 통과 여부를 처음부터 심사하는 것이 아니다. 같은 시드를 재사용하면 초기화/배치 순서
노이즈가 두 조건(102 vs 78) 사이에서 최대한 상쇄돼 feature-세트 효과만 더 깨끗하게 드러난다
(신규 무작위 시드를 또 뽑으면 시드 노이즈와 feature-세트 효과가 섞여 버림). CLAUDE.md
Seed-Diversity Ensemble Promotion Gate가 요구하는 "N≥5 진짜 다양시드"라는 표본 자체는 이미
102-feature 테스트에서 정식으로 만족되었고, 이번 78-feature 결과도 동일 시드 집합이므로 그
표본 크기·다양성 조건을 상속한다.

## 4. dev/server 용량 및 실행 노트

작업 시작 전 `nproc`=12, `uptime` load average 0.37/0.17/0.42 — 유휴 상태. 선례(zig075
5-시드 formal 테스트, 시드당 3.2분)에 견줘 이번엔 10개 run(2 컴포넌트×5시드) 예상 총
30~50분으로 dev 단독 순차 실행이 합리적이라고 판단, 서버(`llewyn@192.168.0.232`)는 건드리지
않았다(상태 조회조차 하지 않음 — dev 하나로 충분히 짧아 분배 이득이 없었음).

**실측**: `scripts/run_dedup78_nseed_dev_20260815.sh` 10개 run 전부 성공(`ALL 10 RUNS OK`),
13:48:19~14:17:59(KST), 총 **29분 40초**, run당 평균 2분 58초 — zig075/h48qual 교대로 순차
실행. 각 번들 `report.json`에서 `input_contract.base_feature_count=78`,
`total_features=91`(78+position 13)을 10개 전부 재확인해 feature 오염이 없음을 검증했다.

## 5. 평가 방법

zig075: `scripts/diagnose_eth_zig075_ungated_direction_vs_always_short_20260815.py`를
`--bundle-dir`/`--out-csv`만 바꿔 그대로 재사용(이미 일반화돼 있음, 로직 변경 없음).

h48qual: `scripts/diagnose_eth_h48qual_ungated_direction_vs_always_short_20260812.py`가 기존엔
`--bundle-dir`/`--out-csv` 인자가 없었으므로 zig075 쪽과 동일 패턴으로 이번에 일반화했다
(하드코딩 `BUNDLE_DIR` → argparse 기본값, 로직 변경 없음). 인자 없이 실행하면 기존 배포 번들
결과(VAL ungated +10.80 vs always_short +10.86, OOS ungated -4.76 vs always_short +13.53)를
그대로 재현하는 것으로 회귀 확인 완료.

둘 다 `quality_threshold`를 완전히 무시하고 `direction_head`의 원본 argmax(`dir_action`)만
사용, `always_short`/`always_long`은 같은 ungated active set에서 방향만 강제. `cost_mult=3.0`,
`max_hold=0`/`cooldown=0`. 구간은 저장소 관행대로 VAL=2025-10-01~12-31,
OOS=2026-01-01~03-31.

## 6. 결과 (10 run × VAL/OOS = 20칸, cherry-picking 없음)

### zig075 (78-feature, 5시드)

| seed | split | ungated pnl | trades | wr | always_short pnl | trades | wr | always_long pnl | ungated이 always_short 이김? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 946043153 | VAL | -1.38 | 39 | 35.9% | +9.70 | 37 | 43.2% | -16.36 | **NO** |
| 946043153 | OOS | +2.98 | 28 | 39.3% | +21.84 | 25 | 56.0% | -19.31 | **NO** |
| 932925759 | VAL | -15.91 | 43 | 27.9% | +9.66 | 40 | 42.5% | -22.40 | **NO** |
| 932925759 | OOS | +0.60 | 25 | 36.0% | +24.91 | 24 | 58.3% | -18.91 | **NO** |
| 74851798 | VAL | +10.65 | 35 | 42.9% | +12.64 | 38 | 44.7% | -15.06 | **NO** |
| 74851798 | OOS | +5.58 | 30 | 43.3% | +24.24 | 29 | 55.2% | -17.77 | **NO** |
| 975176982 | VAL | -9.23 | 34 | 32.4% | +10.87 | 36 | 44.4% | -17.34 | **NO** |
| 975176982 | OOS | -3.61 | 28 | 32.1% | +20.71 | 25 | 56.0% | -19.08 | **NO** |
| 542143953 | VAL | +9.37 | 43 | 39.5% | +9.72 | 39 | 41.0% | -16.06 | **NO** |
| 542143953 | OOS | -3.50 | 29 | 34.5% | +19.92 | 26 | 53.8% | -16.37 | **NO** |

**10/10칸 always_short에 패배.**

### h48qual (78-feature, 5시드, 동일 시드)

| seed | split | ungated pnl | trades | wr | always_short pnl | trades | always_long pnl | ungated이 always_short 이김? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 946043153 | VAL | +2.93 | 30 | 36.7% | +9.41 | 40 | -13.87 | **NO** |
| 946043153 | OOS | -11.40 | 31 | 25.8% | +19.23 | 26 | -16.34 | **NO** |
| 932925759 | VAL | -5.86 | 44 | 34.1% | +10.04 | 43 | -18.14 | **NO** |
| 932925759 | OOS | +5.63 | 22 | 40.9% | +26.68 | 23 | -18.61 | **NO** |
| 74851798 | VAL | +7.93 | 42 | 40.5% | +11.45 | 44 | -14.93 | **NO** |
| 74851798 | OOS | -2.04 | 29 | 34.5% | +23.99 | 29 | -14.49 | **NO** |
| 975176982 | VAL | -18.80 | 47 | 25.5% | +9.61 | 40 | -18.61 | **NO** |
| 975176982 | OOS | -6.03 | 30 | 33.3% | +17.33 | 27 | -18.10 | **NO** |
| 542143953 | VAL | -22.00 | 44 | 22.7% | +6.75 | 39 | -15.91 | **NO** |
| 542143953 | OOS | +15.99 | 25 | 52.0% | +19.65 | 26 | -14.92 | **NO** |

**10/10칸 always_short에 패배**(마지막 셀 542143953/OOS가 가장 근접했지만 +15.99 < +19.65로도
여전히 짐).

**두 컴포넌트 합산: 20/20칸 전부 always_short에 패배.**

## 7. 102-feature 대비 비교 및 최종 판정

| | 102-feature (기존 formal) | 78-feature (본 문서) |
|---|---|---|
| zig075 | 10/10 손실 (5시드 formal, `..._zig075_direction_head_skill_formal_nseed_20260815.md`) | **10/10 손실** |
| h48qual | 38/40 손실 (Odyssey1, 30+ 변형 누적) | **10/10 손실** (같은 5시드 paired 재확인) |

24개 중복 feature 제거는 **어느 컴포넌트에서도 부호-일관성 그림을 전혀 바꾸지 않았다**. 78개로
줄여도 두 컴포넌트 모두 시드에 무관하게 always_short에 완패하는 패턴이 정확히 유지된다 —
5개 시드 중 VAL에서 근소하게 앞선 경우(zig075 74851798/542143953, h48qual 946043153/74851798)
는 있지만 always_short를 실제로 이긴 칸은 20칸 중 0칸이다.

**최종 판정: dedup78 재검증도 REJECTED.** feature 중복 제거는 direction_head no-skill 벽의
원인이 아니었다 — 감사 문서가 이미 별개 사실로만 기록했던 추정("이 중복 자체가 no-skill의
원인이라는 근거는 아니다")이 이번 직접 테스트로 확인됐다. 78-feature 세트는 promotion
후보로서 102-feature와 동일하게 기각한다.

## Fresh-Forward 체크리스트

`fresh_forward_bar_by_bar=true`(VAL/OOS 고정 구간, `omega._to_fixed_decisions`/`omega._metrics`가
매 bar 인과적으로 TP/SL/time-exit 확정), `trade_ledgers_used_as_input=false`(이번 재학습이
자체 생성한 예측 CSV만 사용), `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`. 라이브 파일(`trading_bot_modules/omega4_6_1_live.py`,
`trading_bot.py`, `runtime_config.py`, `.env`, 배포 체크포인트)은 전혀 건드리지 않았다 —
`git status`로 확인 완료. 이 세션이 수정/생성한 것은 연구용 스크립트 3개
(`scripts/train_eval_omega4_3head_parent72_pinned78_20260815.py` 신규,
`scripts/run_dedup78_nseed_dev_20260815.sh` 신규,
`scripts/diagnose_eth_h48qual_ungated_direction_vs_always_short_20260812.py` 일반화)와 이
문서뿐이다. 서버에는 어떤 파일도 push/수정하지 않았다(이번엔 조회조차 하지 않음).

## 산출물

- 78-feature 도출/검증 스크립트: `/tmp/claude-1000/-home-kbj20-crypto-scalping/930e2f78-37fd-47e0-bb45-601fad343923/scratchpad/{redundancy_analysis.py,dedup78_cols.json}`(세션 scratchpad, 재현용 로직은 이 문서 §1에 값으로 고정 기록)
- 재학습 wrapper: `scripts/train_eval_omega4_3head_parent72_pinned78_20260815.py`
- 재학습 드라이버: `scripts/run_dedup78_nseed_dev_20260815.sh`
- 10개 번들: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned78_{zig075,h48qual}_dedup_seed{946043153,932925759,74851798,975176982,542143953}/`
- 학습 로그: `tmp/eth_dedup78_nseed_skill_retest_20260815/pinned78_{zig075,h48qual}_seed<SEED>.log`, 드라이버 로그 `tmp/eth_dedup78_nseed_skill_retest_20260815_driver.log`
- 평가 스크립트: `scripts/diagnose_eth_zig075_ungated_direction_vs_always_short_20260815.py`(재사용),
  `scripts/diagnose_eth_h48qual_ungated_direction_vs_always_short_20260812.py`(일반화)
- 시드별 진단 결과 CSV: `tmp/eth_dedup78_nseed_skill_retest_20260815/diag_out/{zig075,h48qual}_dedup78_seed<SEED>.csv`

## 다음 단계

이 문서는 verdict만 확정한다 — `docs/model_contracts/`의 Odyssey/h48qual/zig075 관련 계약
문서 갱신은 사용자가 직접 처리한다(작업 지시에 따라 이 세션은 contract 문서를 스스로 편집하지
않음).
