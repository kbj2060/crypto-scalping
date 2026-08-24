# ETH 캐시 슬리브 EV-HGB 후보 — cheap_gate (오라클/사후확인 측정) (2026-08-16)

상태: **Stage 1(실제 HGB 학습 + purged CV + 라벨 순열 대조군) 완료 — 사전등록한 결합 기준 미충족,
N≥5 시드/fresh-forward walk-forward로 진행하지 않음을 권고.** (cheap_gate/IC-check 단계는 완료.)

## 배경 — BTC 메커니즘 이식

이전 조사에서 BTC에 "캐시 슬리브"라는, 라이브 검증은 됐지만 현재 배선되지 않은(dead code) 메커니즘이
발견됐다: PRIMARY 모델이 CASH(무포지션) 상태일 때만 작동하는 별도의 EV 회귀 모델 2개(`long_model`,
`short_model`, 둘 다 `sklearn.ensemble.HistGradientBoostingRegressor`)가 그 시점의 피처로부터
"지금 폴백 트레이드를 넣으면 얻을 net EV"를 예측하는 구조다. 라벨/시뮬레이션 로직은:

- CASH인 모든 bar에서 LONG/SHORT 폴백 트레이드를 둘 다 시뮬레이션: 다음 bar 시가에 슬리피지 포함
  진입, TP/SL/max_hold_bars/"primary_takeover"(가격 경로상 PRIMARY가 다시 active가 되는 시점) 중
  먼저 도달하는 조건으로 청산.
- 비용은 정상치의 3배로 스트레스(`fee_eff=fee*3`, `slip_eff=slip*3`).
- 고정 리스크 프로파일(그대로 이식): `take_profit=0.026`, `stop_loss=0.014`(둘 다 가격 변동률),
  `notional=0.405`, `leverage=2.0`(→ `margin_fraction=notional/leverage=0.2025`,
  CLAUDE.md Futures Risk Sizing Contract와 일치 확인), `max_hold_bars=192`.
- `ev_min=0.002`(0.2%)가 프로덕션 채택 임계값.

이 문서는 그 구조를 ETH의 PRIMARY(h48qual/zig075 3-Head TabM + Odyssey4 잠금 베이스라인)에 이식하기
전에, **학습 없이** 빠르게 "CASH 구간에 실제로 잡을 만한 기회가 있는가"만 확인하는 cheap_gate 결과다.

## 반드시 먼저 읽을 것 — 오라클/사후확인(hindsight) 프레이밍

아래 모든 `long_net`/`short_net`/"oracle" 수치는 **그 bar 자신의 실제 미래 가격 경로를 완전히 알고
있다는 가정**으로 계산됐다. 어떤 모델도, 인과적 방식으로도 이 숫자를 만들어낼 수 없다 — 이건 오직
"실제로 학습할 만한 엣지가 존재하는가"의 상한선을 재는 용도다. 이 문서의 어떤 숫자도 승격 근거, 전략
성과, 또는 이 저장소의 Fresh-Forward Validation/OOS/Test Rule이 요구하는 causal walk-forward 테스트의
대체물이 아니다 — 실제 EV-HGB 후보가 나온다면 그 자체로 다시 인과적 bar-by-bar 학습/추론 및 자체
walk-forward 테스트를 거쳐야 승격 주장을 할 수 있다.

## PRIMARY(CASH 상태) 정의 — 재구현이 아니라 재사용

"bar i에서 PRIMARY가 CASH"는 raw 모델 출력에서 새로 유도하지 않고, **Odyssey4 잠금 베이스라인의
실제 계정 레벨 greedy replay 렛저**(entry_i/exit_i 정수 bar 인덱스)에서 직접 읽는다 — 그래서 단일
포지션 슬롯, h48qual>zig075 우선순위, 마진/레버리지 캡, h48qual의 잠긴 regime-aware exit guard,
zig075의 잠긴 지속상승장 SHORT 진입거부(`docs/model_contracts/odyssey4_eth_entry_veto_baseline_
contract_20260814.md`의 G0 베이스라인 그 자체)가 전부 자동으로 반영된다. 아래 모듈을 **수정 없이**
import만 해서 사용:

- `eth_omega461_multiwindow_confirmation_gate_20260814`(윈도우 정의/로딩)
- `research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814`(h48qual regime-aware exit
  guard, 컴포넌트 준비, 탐지기)
- `research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814`
  (`greedy_replay_entry_veto` — Odyssey4 잠금 베이스라인 replay 함수 그 자체)

렛저의 `entry_i`/`exit_i`(둘 다 inclusive)로부터 `held` boolean 배열을 재구성한다 — 이는 계정
레벨 시뮬레이션을 bar-by-bar로 다시 도는 것과 정확히 동일한 결과를 더 싸게 얻는 것이다. `entry_signal_
i`(신호가 난 bar 자신)는 아직 포지션이 열리지 않은 상태이므로 올바르게 CASH bar로 분류된다(진입은
`entry_i = entry_signal_i + 1`의 시가에 체결).

## 폴백 트레이드 오라클 시뮬레이션 — 이번에 새로 짠 유일한 로직

CASH인 모든 bar `i`에 대해 LONG/SHORT 둘 다 시뮬레이션:

- 진입: bar `i+1`의 시가, 슬리피지 포함.
- TP/SL 체크는 **매 bar의 종가 기준**(intrabar high/low 아님) — `replay_omega4_6_1_greedy_router_
  20260706.greedy_replay` 자신의 배리어 체크 컨벤션을 그대로 따른 것으로, 라이브 시스템의 실제 TP/SL
  판정 규칙과 동일한 규칙으로 폴백 트레이드도 평가받는다(더 유리한 intrabar-touch 규칙을 쓰지 않음).
- `primary_takeover`: `held` 배열이 다시 True가 되기 **직전** bar에서 강제 청산(계정에 포지션 슬롯이
  하나뿐이므로, PRIMARY가 슬롯을 다시 쓰기 전에 폴백은 반드시 빠져 있어야 한다).
- `max_hold_bars=192` 도달 시 강제 청산.
- 비용은 3배 스트레스. fee/slip 상수(`FEE_RATE=0.0005`, `SLIP_RATE=0.0002`)는
  `train_eval_omega1_2_tabm_diffusion_risk_20260603._load_fee_slip()`에서 그대로 읽음 — 이 계보의
  다른 모든 ETH Omega4.6.1/Odyssey 스크립트가 쓰는 것과 동일한 소스, 새로 정의하지 않음.

수익 계산은 `greedy_replay`의 자체 산식을 그대로 복제:
`cash_after_entry = 1 - fee_eff*notional`, `cash_final = cash_after_entry*(1+raw_exit*notional) -
cash_after_entry*fee_eff*notional`, `net_return = cash_final - 1`(진입/청산 슬리피지는 `raw_exit`
계산에 이미 내재).

## 윈도우 경계 — CLAUDE.md 기본값과의 편차, 명시적으로 기록

CLAUDE.md의 일반 Fresh-Forward 기본값은 VAL=2025-09-01~12-31 / OOS=2026-01-01~03-31이다. 이 실험은
**VAL=2025-10-01~12-31 / OOS(=oos_q1)=2026-01-01~03-31**을 사용했다 — OOS는 기본값과 정확히
일치하지만 VAL 시작월이 한 달 늦다. 이유: h48qual/zig075 ThreeHeadTabM parent 아티팩트 자신의
train/validation split 경계가 `SPLIT_TS=2025-10-01`이다(`train_eval_omega1_2_tabm_3head_20260603.py:
33`) — 2025년 9월은 그 모델의 **TRAIN split 내부**라 `validation_predictions_*.csv` 자체가 9월
데이터를 갖고 있지 않다. 9월을 억지로 포함시키면 "validation"이라는 이름으로 in-sample(train-split)
예측을 조용히 섞어 쓰는 셈이 되는데, 이는 이 저장소 자체 컨벤션이 이미 context 전용 티어(2025q3)로
분리해둔 것보다 더 나쁜 Fresh-Forward 위반이다. 그래서 이 서브프로젝트 전체가 이미 쓰고 있는, 이미
검증된 val/oos_q1 경계(`eth_omega461_multiwindow_confirmation_gate_20260814.WINDOW_DEFS`)를 그대로
따랐다.

## G0 정합성 체크 (환경/데이터 드리프트 없음을 확인)

Odyssey4 잠금 베이스라인 계약(`docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_
20260814.md`)의 G0 표에서 no_gate 수치를 그대로 재현하는지 확인:

| 윈도우 | 기준값(no_gate pnl/mdd/trades) | 실제 재현값 | 일치 |
|---|---|---|---|
| val | 41.13% / −21.70% / 35 | 41.13% / −21.70% / 35 | ✅ |
| oos_q1 | 93.27% / −15.48% / 24 | 93.27% / −15.48% / 24 | ✅ |

정확히 일치 — Odyssey4 잠금 베이스라인 엔진을 올바르게 재사용했다는 근거.

## 실제 수치 결과

| 항목 | VAL(2025-10~12) | OOS-Q1(2026-01~03) |
|---|---:|---:|
| 전체 유효 bar 수 | 26,207 | 25,536 |
| PRIMARY 보유(비-CASH) bar 수 | 19,221 (73.3%) | 18,637 (73.0%) |
| **PRIMARY CASH bar 수 / 비율** | **6,986 / 26.66%** | **6,899 / 27.02%** |
| CASH bar 중 `max(long_net,short_net) > ev_min` 비율 | **45.78%** | **56.21%** |
| 전체 유효 bar 대비 위 비율 | 12.20% | 15.19% |
| 조건 충족 bar의 평균 net edge | **+0.843%** | **+0.899%** |
| 오라클 상한 누적 PnL(가산, **달성 불가능**) | +2696.23% | +3487.73% |
| always_short(모든 CASH bar, 게이트 없음) 합/평균 | −1136.88% / −0.163% | −1013.36% / −0.147% |
| always_long(모든 CASH bar, 게이트 없음) 합/평균 | −1378.05% / −0.197% | −1147.96% / −0.166% |
| 순수 방향성 베이스라인(max(always_long,always_short)) | −1136.88%(short) | −1013.36%(short) |
| 오라클 상한 − 베이스라인 | +3833.11pp | +4501.08pp |

**합산(VAL+OOS-Q1)**: 유효 bar 51,743개 중 CASH 13,885개(26.83%), 오라클 상한 누적 합
+6183.96%, 방향성 베이스라인 합 −2150.23%.

exit reason 분포(VAL, LONG/SHORT 각각): `stop_loss` 2833/2709, `primary_takeover` 1757/1482,
`max_hold_bars` 1478/1603, `take_profit` 918/1192. TP:SL 가격변동 배율이 2.6:1.4≈1.86:1이라 무비용
기준 손익분기 승률은 약 35%인데, 고정 한 방향(always_long)의 실제 TP 적중률은 13.1%(VAL)에 불과 —
always_long/always_short 둘 다 뚜렷하게 마이너스인 것과 정합.

원본 CSV/report: `tmp/causal_regen_20260516/eth_candidate_cash_sleeve_ev_hgb_cheap_gate_20260816/
report.json`, `cash_sleeve_oracle_bars_val.csv`, `cash_sleeve_oracle_bars_oos_q1.csv`.

## 해석과 정직한 경고

1. **"CASH 시간"은 결코 사소하지 않다** — VAL/OOS-Q1 둘 다 약 27%가 PRIMARY 무포지션 구간이다.
   idle capital이 실제로 존재한다는 스펙 항목 1의 답은 명확히 "그렇다".
2. **오라클 조건 충족 비율(46~56%)과 평균 엣지(0.84~0.90%, ev_min의 4배 이상)는 이 실험의 사전
   등록된 close-negative 기준("<5~10%이거나 평균이 작으면 학습 없이 종료")을 명백히 충족하지
   못한다** — 즉 이 cheap_gate 기준으로는 "닫을" 근거가 없다.
3. **그러나 이 큰 숫자를 곧이곧대로 신뢰하면 안 된다.** 오라클 상한(+2696%/+3487%)은 약
   3,200~3,900개의 서로 시간적으로 크게 겹치는 가상 트레이드를 단순 가산한 것일 뿐, 복리도 아니고
   자본 제약도 없다 — "달성 불가능"이라는 라벨 그대로 읽어야 한다. 더 중요한 건 **"조건 충족 비율이
   높다"는 사실 자체가, 오라클이 매 bar마다 사후에 이긴 쪽(long/short)만 골라 쓰기 때문**이라는
   점이다: TP:SL 배율이 1.86:1인 배리어에서 한 방향으로 고정하면 손익분기 승률(~35%)에 크게 못
   미치는데(always_long TP 적중률 13.1%), "그날 어느 쪽이 이겼는지 사후에 안다"는 가정만으로 조건
   충족률이 46~56%까지 뛴다. 이건 변동성이 큰 5분봉 자산에서 최대 192bar(16시간) 보유를 허용하면
   어느 한쪽으로건 2.6% 움직임이 자주 나온다는 사실을 반영하는 것에 가깝고, **실제 예측 모델이 이
   상한의 상당 부분을 잡을 수 있다는 근거는 전혀 아니다.**
4. **가장 중요한 사전 정보(prior)**: 이 저장소의 ETH 방향 예측 축은 이미 여러 차례 정면으로
   실패했다 — h48qual/zig075 direction head가 always_short/always_long 대비 no-skill이라는 것은
   contract-level로 확정된 사실이고(`eth_odyssey3_zig075_short_entry_veto_uptrend_confirmed_
   20260815` 등 메모리 참고), evidence-signal 22종 스코어카드·오실레이터 confluence·AMT/VSA/iFVG
   4종 등 방향성/타이밍 신호 다수가 CLOSED 상태다. EV-HGB는 방향을 직접 분류하진 않지만 net
   realized return을 회귀하는 것은 사실상 방향 예측과 강하게 얽혀 있다 — 같은 시장·같은
   타임프레임에서 방향 신호가 반복적으로 no-skill이었다는 사실은, 실제 EV-HGB가 이 오라클 상한의
   유의미한 부분을 인과적으로 포착할 수 있다는 주장에 상당한 회의를 갖게 하는 강한 사전 정보다.

## 판정

**cheap_gate 자체 기준(비율<5~10%, 평균 엣지 작음 → 종료)으로는 종료 근거가 없다** — 헤드룸이
이론상 존재한다. 그러나 위 3번·4번의 경고 때문에 **이 결과만으로 실제 학습에 착수하지 않는다** —
과업 지시(오케스트레이팅 세션)에 따라 여기서 멈추고 수치를 보고한다. `docs/model_contracts/
research_line_registry.json`에도 항목을 추가하지 않는다(결정적 부정 결과가 아니므로).

## 권장 다음 단계 (실제 학습 착수 여부는 사용자/오케스트레이팅 세션 판단)

1. **학습 전에 더 싼 확인**: 실제 HGB 학습 전에, CASH bar에서 이미 계산 가능한 기존 causal
   피처(dual_momentum, ATR, 최근 realized vol 등 h48qual/zig075가 이미 쓰는 피처)와
   `max(long_net,short_net)` 또는 `long_net−short_net`의 순위상관(IC)만 먼저 재는 게, 전체 HGB
   purged-CV 학습보다 훨씬 싸다 — "raw feature price-trend contamination 체크"처럼 이 저장소가
   이미 쓰는 방법론과 일치한다.
2. 실제 학습을 한다면: `long_model`/`short_model` 각각 CASH bar에서만 학습(라벨은 이 cheap_gate와
   동일한 3배 비용 스트레스 net_return 시뮬레이션, causal하게 그 시점 피처만 사용), purged/embargo
   CV, 그리고 라벨 순열(permutation) 대조군 없이 개선을 채택하지 않는다(`research_line_registry.
   json`의 falsification_audit 취지).
3. N≥5개 진짜 랜덤 시드로 OOS 부호 일치를 확인하기 전에는(Seed-Diversity Ensemble Promotion
   Gate) 어떤 형태로도 승격 주장을 하지 않는다.
4. 실제 causal walk-forward(Fresh-Forward Rule)로 VAL 통과 후 OOS 단일터치 확인 전에는 "이 정도
   PnL을 기대할 수 있다"는 주장을 하지 않는다 — 이번 cheap_gate 수치는 그 근거가 될 수 없다.

## 준수 사항

`fresh_forward_bar_by_bar=true`(PRIMARY 렛저는 수정 없는 단일 causal bar-by-bar replay; 폴백
시뮬레이션 자체도 bar-by-bar 전진이지만, 설계상 각 bar 자신의 **실현된 미래 경로**를 사용하는
오라클임을 명시). `trade_ledgers_used_as_input=false`(PRIMARY 렛저는 오직 "이 bar가 CASH인가"라는
계정 구조적 사실만 제공 — 개별 트레이드 신호로 사용하지 않음). `saved_parent_exit_timestamps_
used=false`. `future_rows_used_for_entry=false`(PRIMARY 기준으로는 참, 폴백 오라클 시뮬레이션은
스펙상 의도적으로 미래 가격 경로를 사용 — 처음부터 그렇다고 명시). `trading_bot.py` /
`trading_bot_modules/omega4_6_1_live.py` / `runtime_config.py` / `.env` 미변경. 재학습 없음, GPU
없음(`DEVICE=cpu`), conda env `quant_ai`.

## 후속 조사 — 학습 전 더 싼 확인: 기존 causal 피처 vs 오라클 폴백 엣지의 IC (순위상관) (2026-08-16)

위 "권장 다음 단계" 1번을 그대로 실행한 결과다. **HGB 학습은 하지 않았다** — 순수 상관관계 분석.

### 방법

- 구현: `scripts/research_eth_candidate_cash_sleeve_ev_hgb_ic_check_20260816.py` (신규, 읽기 전용
  연구 스크립트, `trading_bot.py`/live 코드 미접촉, GPU 미사용).
- 피처 프레임은 **재구현이 아니라 재사용** — cheap_gate 스크립트 자신이 오라클 CSV를 만들 때 쓴 것과
  동일한 `aligned_frame`을, 동일한 미수정 로더(`eth_omega461_multiwindow_confirmation_gate_20260814.
  load_all_windows` + `research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.
  prepare_regime_aware_components`/`build_detector`)로 다시 만들어 썼다. 오라클 CSV의 `i` 컬럼이
  가리키는 `aligned_frame` row의 timestamp가 CSV 자신의 `timestamp` 컬럼과 **bit-exact 일치**하는지
  먼저 assert로 확인한 뒤에만 조인했다(VAL 6,986행 / OOS-Q1 6,899행 전부 일치 — cheap_gate 리포트의
  CASH bar 수와도 정확히 같다).
- `aligned_frame`은 `data/splits/year_oos/training_features_2025.csv`(2026은 `..._2026_rebuilt.csv`,
  142개 원본 컬럼) + `regime3_current_sensitive_wide24_*` 오버레이(6개 컬럼)로 구성된, h48qual/zig075
  계보 전체가 쓰는 base+wide24 피처 패널이다(`train_eval_omega1_2_tabm_3head_20260603.py`가 최종
  학습에 쓰는 `feature_cols`도 같은 계보의 다른 소스 CSV에서 온 것이지만 사실상 동일한 피처 카탈로그
  ─ dual_momentum, ATR류, realized-vol류, regime3 라우터 확률 등 ─ 을 공유한다).
- 지시받은 대로 전수조사가 아니라 **선별한 후보 25개**만 테스트: 모멘텀/추세 10개(`dual_momentum`,
  `mtf_trend_1h`, `mtf_trend_4h`, `mean_reversion_z`, `breakout_strength`, `macd_hist`, `hma_slope`,
  `turtle_signal`, `kalman_velocity`, `rsi`), 변동성 9개(`garman_klass_vol`, `realized_vol_ratio`,
  `rogers_satchell_vol`, `parkinson_vol`, `atr_pct_rank_288`, `garch_vol_z`, `volatility_z`,
  `bb_width_z`, `compression_score`), 레짐/라우터 6개(`regime3_current_sensitive_wide24_{bull,bear,
  chop}_prob`, `_confidence`, `_entropy`, `_margin`).
- 타겟 2개: `max_net = max(long_net, short_net)`(오라클의 최선 방향 엣지), `net_diff = long_net −
  short_net`(오라클의 방향 선호도 — 실제 `long_ev >= short_ev` 분기 결정에 대응).
- 각 (피처, 타겟, 윈도우) 조합마다: Spearman IC, IC의 bootstrap 95% CI(쌍 리샘플링 500회), 라벨
  셔플 null 분포(200회, 평균/표준편차), 그리고 이 저장소 자체 방법론(`feedback_raw_feature_price_
  trend_contamination`)에 따른 가격추세 오염 체크 — 같은 bar들에서 피처 vs raw ETH `close`의
  Spearman IC(디스퀄리파잉 기준 |r|>0.5, 이 실험에서는 적용 안 함).
- 중요한 차이점: 여기서 쓰는 타겟(`long_net`/`short_net`)은 cheap_gate 스크립트가 이미 3배 비용
  스트레스(수수료·슬리피지)를 적용해 계산한 **net** 값이다 — 즉 이 IC는 "raw 가격움직임과의 상관"이
  아니라 "이미 비용을 뺀 결과와의 상관"이라, evidence-signal 서브프로젝트가 별도로 걱정했던
  "비용을 반영 안 한 IC가 cost-gate에서 사라지는" 문제를 이 시점에서 이미 어느 정도 통제한다(그렇다고
  이 IC 자체가 실제 트레이딩 가능한 엣지라는 뜻은 아니다 — 아래 해석 참고).

### 결과 — 피처 × 타겟 × 윈도우 전수 표 (25피처 × 2타겟 = 50행, VAL/OOS-Q1 나란히 표시)

원본 CSV: `tmp/causal_regen_20260516/eth_candidate_cash_sleeve_ev_hgb_ic_check_20260816/ic_results.csv`
(100행, 윈도우별로 분리된 형태), `val_oos_consistency.csv`, `ic_check_summary.json`.

| 그룹 | 피처 | 타겟 | VAL IC | VAL 가격오염IC | VAL 판정 | OOS-Q1 IC | OOS-Q1 가격오염IC | OOS-Q1 판정 |
|---|---|---|---:|---:|---|---:|---:|---|
| 레짐/라우터 | `regime3_..._bear_prob` | `max_net` | +0.065 | -0.110 | 보통(0.05~0.10) | +0.034 | -0.056 | 약함(0.02~0.05) |
| 레짐/라우터 | `regime3_..._bear_prob` | `net_diff` | +0.130 | -0.110 | 강함(≥0.10) | +0.033 | -0.056 | 약함(0.02~0.05) |
| 레짐/라우터 | `regime3_..._bull_prob` | `max_net` | +0.009 | +0.221 | 노이즈와구분불가 | +0.028 | +0.100 | 약함(0.02~0.05) |
| 레짐/라우터 | `regime3_..._bull_prob` | `net_diff` | -0.091 | +0.221 | 보통(0.05~0.10) | +0.037 | +0.100 | 약함(0.02~0.05) |
| 레짐/라우터 | `regime3_..._chop_prob` | `max_net` | -0.129 | -0.195 | 강함(≥0.10) | -0.022 | -0.103 | 노이즈와구분불가 |
| 레짐/라우터 | `regime3_..._chop_prob` | `net_diff` | +0.003 | -0.195 | 노이즈와구분불가 | -0.100 | -0.103 | 강함(≥0.10) |
| 레짐/라우터 | `regime3_..._confidence` | `max_net` | -0.005 | -0.074 | 노이즈와구분불가 | -0.084 | -0.022 | 보통(0.05~0.10) |
| 레짐/라우터 | `regime3_..._confidence` | `net_diff` | -0.006 | -0.074 | 노이즈와구분불가 | -0.005 | -0.022 | 노이즈와구분불가 |
| 레짐/라우터 | `regime3_..._entropy` | `max_net` | +0.005 | +0.046 | 노이즈와구분불가 | +0.090 | +0.007 | 보통(0.05~0.10) |
| 레짐/라우터 | `regime3_..._entropy` | `net_diff` | +0.033 | +0.046 | 약함(0.02~0.05) | +0.016 | +0.007 | 노이즈와구분불가 |
| 레짐/라우터 | `regime3_..._margin` | `max_net` | -0.004 | -0.075 | 노이즈와구분불가 | -0.082 | -0.023 | 보통(0.05~0.10) |
| 레짐/라우터 | `regime3_..._margin` | `net_diff` | -0.006 | -0.075 | 노이즈와구분불가 | -0.004 | -0.023 | 노이즈와구분불가 |
| 모멘텀/추세 | `breakout_strength` | `max_net` | +0.012 | +0.243 | 노이즈와구분불가 | +0.001 | +0.099 | 노이즈와구분불가 |
| 모멘텀/추세 | `breakout_strength` | `net_diff` | -0.161 | +0.243 | 강함(≥0.10) | +0.015 | +0.099 | 노이즈와구분불가 |
| 모멘텀/추세 | `dual_momentum` | `max_net` | -0.041 | +0.001 | 약함(0.02~0.05) | -0.105 | -0.016 | 강함(≥0.10) |
| 모멘텀/추세 | `dual_momentum` | `net_diff` | -0.206 | +0.001 | 강함(≥0.10) | +0.127 | -0.016 | 강함(≥0.10) |
| 모멘텀/추세 | `hma_slope` | `max_net` | -0.019 | +0.068 | 노이즈와구분불가 | +0.005 | +0.045 | 노이즈와구분불가 |
| 모멘텀/추세 | `hma_slope` | `net_diff` | -0.036 | +0.068 | 약함(0.02~0.05) | +0.016 | +0.045 | 노이즈와구분불가 |
| 모멘텀/추세 | `kalman_velocity` | `max_net` | -0.029 | +0.100 | 약함(0.02~0.05) | +0.022 | +0.062 | 노이즈와구분불가 |
| 모멘텀/추세 | `kalman_velocity` | `net_diff` | -0.048 | +0.100 | 약함(0.02~0.05) | +0.019 | +0.062 | 노이즈와구분불가 |
| 모멘텀/추세 | `macd_hist` | `max_net` | -0.025 | +0.013 | 노이즈와구분불가 | +0.029 | +0.033 | 약함(0.02~0.05) |
| 모멘텀/추세 | `macd_hist` | `net_diff` | +0.022 | +0.013 | 노이즈와구분불가 | +0.026 | +0.033 | 약함(0.02~0.05) |
| 모멘텀/추세 | `mean_reversion_z` | `max_net` | -0.013 | -0.116 | 노이즈와구분불가 | +0.106 | -0.101 | 강함(≥0.10) |
| 모멘텀/추세 | `mean_reversion_z` | `net_diff` | +0.157 | -0.116 | 강함(≥0.10) | +0.016 | -0.101 | 노이즈와구분불가 |
| 모멘텀/추세 | `mtf_trend_1h` | `max_net` | -0.030 | +0.127 | 약함(0.02~0.05) | +0.028 | +0.068 | 약함(0.02~0.05) |
| 모멘텀/추세 | `mtf_trend_1h` | `net_diff` | -0.059 | +0.127 | 보통(0.05~0.10) | +0.015 | +0.068 | 노이즈와구분불가 |
| 모멘텀/추세 | `mtf_trend_4h` | `max_net` | -0.014 | +0.192 | 노이즈와구분불가 | +0.034 | +0.089 | 약함(0.02~0.05) |
| 모멘텀/추세 | `mtf_trend_4h` | `net_diff` | -0.123 | +0.192 | 강함(≥0.10) | +0.011 | +0.089 | 노이즈와구분불가 |
| 모멘텀/추세 | `rsi` | `max_net` | -0.031 | +0.166 | 약함(0.02~0.05) | +0.041 | +0.086 | 약함(0.02~0.05) |
| 모멘텀/추세 | `rsi` | `net_diff` | -0.095 | +0.166 | 보통(0.05~0.10) | +0.005 | +0.086 | 노이즈와구분불가 |
| 모멘텀/추세 | `turtle_signal` | `max_net` | -0.002 | +0.157 | 노이즈와구분불가 | -0.086 | +0.083 | 보통(0.05~0.10) |
| 모멘텀/추세 | `turtle_signal` | `net_diff` | -0.166 | +0.157 | 강함(≥0.10) | +0.005 | +0.083 | 노이즈와구분불가 |
| 변동성 | `atr_pct_rank_288` | `max_net` | +0.120 | +0.104 | 강함(≥0.10) | +0.004 | +0.045 | 노이즈와구분불가 |
| 변동성 | `atr_pct_rank_288` | `net_diff` | +0.082 | +0.104 | 보통(0.05~0.10) | +0.121 | +0.045 | 강함(≥0.10) |
| 변동성 | `bb_width_z` | `max_net` | +0.072 | +0.033 | 보통(0.05~0.10) | +0.027 | +0.022 | 약함(0.02~0.05) |
| 변동성 | `bb_width_z` | `net_diff` | +0.048 | +0.033 | 약함(0.02~0.05) | +0.007 | +0.022 | 노이즈와구분불가 |
| 변동성 | `compression_score` | `max_net` | -0.108 | -0.100 | 강함(≥0.10) | +0.007 | -0.026 | 노이즈와구분불가 |
| 변동성 | `compression_score` | `net_diff` | -0.057 | -0.100 | 보통(0.05~0.10) | -0.120 | -0.026 | 강함(≥0.10) |
| 변동성 | `garch_vol_z` | `max_net` | +0.120 | +0.071 | 강함(≥0.10) | -0.004 | +0.063 | 노이즈와구분불가 |
| 변동성 | `garch_vol_z` | `net_diff` | +0.055 | +0.071 | 보통(0.05~0.10) | +0.082 | +0.063 | 보통(0.05~0.10) |
| 변동성 | `garman_klass_vol` | `max_net` | +0.157 | +0.322 | 강함(≥0.10) | +0.146 | +0.185 | 강함(≥0.10) |
| 변동성 | `garman_klass_vol` | `net_diff` | -0.017 | +0.322 | 노이즈와구분불가 | +0.068 | +0.185 | 보통(0.05~0.10) |
| 변동성 | `parkinson_vol` | `max_net` | +0.158 | +0.322 | 강함(≥0.10) | +0.147 | +0.191 | 강함(≥0.10) |
| 변동성 | `parkinson_vol` | `net_diff` | -0.018 | +0.322 | 노이즈와구분불가 | +0.062 | +0.191 | 보통(0.05~0.10) |
| 변동성 | `realized_vol_ratio` | `max_net` | +0.127 | +0.012 | 강함(≥0.10) | +0.017 | +0.068 | 노이즈와구분불가 |
| 변동성 | `realized_vol_ratio` | `net_diff` | +0.072 | +0.012 | 보통(0.05~0.10) | +0.091 | +0.068 | 보통(0.05~0.10) |
| 변동성 | `rogers_satchell_vol` | `max_net` | +0.157 | +0.322 | 강함(≥0.10) | +0.142 | +0.186 | 강함(≥0.10) |
| 변동성 | `rogers_satchell_vol` | `net_diff` | -0.015 | +0.322 | 노이즈와구분불가 | +0.071 | +0.186 | 보통(0.05~0.10) |
| 변동성 | `volatility_z` | `max_net` | +0.133 | +0.105 | 강함(≥0.10) | -0.007 | +0.055 | 노이즈와구분불가 |
| 변동성 | `volatility_z` | `net_diff` | +0.056 | +0.105 | 보통(0.05~0.10) | +0.134 | +0.055 | 강함(≥0.10) |

("판정"은 이 스크립트 자체 규칙: `|price_contam_ic|>0.5`면 가격오염의심; bootstrap 95% CI가 0을
포함하거나 `|IC|`가 셔플 null 표준편차의 2배 미만이면 노이즈와구분불가; 그 외 `|IC|` 구간별로
무시가능/약함/보통/강함. 셔플 null 표준편차는 n≈6,900~6,990에서 전 피처·타겟·윈도우 공통으로
0.011~0.013 — 이론적 기댓값 1/√n≈0.012와 정합.)

### VAL/OOS-Q1 둘 다에서 "노이즈 이상 + 가격오염 아님" 판정이고 부호까지 같은 조합 (12/50)

전체 `val_oos_consistency.csv` 중 `promising=true`인 행만:

| 피처 | 타겟 | VAL IC (판정) | OOS-Q1 IC (판정) |
|---|---|---:|---:|
| `garman_klass_vol` | `max_net` | +0.157 (강함) | +0.146 (강함) |
| `parkinson_vol` | `max_net` | +0.158 (강함) | +0.147 (강함) |
| `rogers_satchell_vol` | `max_net` | +0.157 (강함) | +0.142 (강함) |
| `bb_width_z` | `max_net` | +0.072 (보통) | +0.027 (약함) |
| `dual_momentum` | `max_net` | -0.041 (약함) | -0.105 (강함) |
| `regime3_..._bear_prob` | `max_net` | +0.065 (보통) | +0.034 (약함) |
| `atr_pct_rank_288` | `net_diff` | +0.082 (보통) | +0.121 (강함) |
| `compression_score` | `net_diff` | -0.057 (보통) | -0.120 (강함) |
| `garch_vol_z` | `net_diff` | +0.055 (보통) | +0.082 (보통) |
| `realized_vol_ratio` | `net_diff` | +0.072 (보통) | +0.091 (보통) |
| `regime3_..._bear_prob` | `net_diff` | +0.130 (강함) | +0.033 (약함) |
| `volatility_z` | `net_diff` | +0.056 (보통) | +0.134 (강함) |

### 해석과 정직한 경고

1. **이 IC 체크는 결정적 음성(negative)이 아니다.** 50개 (피처×타겟) 조합 중 12개가 VAL/OOS-Q1
   양쪽에서 부호 일치 + 노이즈 초과 + 가격오염 임계값(0.5) 미만을 동시에 통과했다. 관측된
   `|price_contam_ic|` 최댓값은 0.322(`garman_klass_vol`/`parkinson_vol`/`rogers_satchell_vol`의
   VAL `max_net`)로, 지시받은 디스퀄리파잉 기준 0.5보다는 낮지만 결코 작지 않다 — 아래 4번 참고.
2. **`max_net`(크기) 쪽 유망 후보는 대부분 변동성 피처(`garman_klass_vol`/`parkinson_vol`/
   `rogers_satchell_vol`)다.** 이건 본 실험 문서의 "해석과 정직한 경고" 3번이 이미 지적한 오라클의
   본질(최대 192bar 보유에서 TP:SL 배리어 중 **둘 중 하나라도** 먼저 닿을 확률은 변동성이 클수록
   커진다는, 거의 정의상 성립하는 관계)과 정확히 들어맞는다 — 이 IC가 "어느 방향이 이기는지"를
   맞히는 능력이 아니라 "둘 중 하나가 크게 움직일 가능성"을 반영할 가능성이 높고, 이 세 변동성
   피처의 가격오염 IC(0.19~0.32)가 이 그룹 전체에서 가장 크다는 사실도 같은 이야기다 — 변동성이
   높은 구간은 종종 방향성 있는 가격 이동(=가격 자체와의 상관) 구간과 겹치기 때문이다. `max_net`
   유망 후보는 신중하게, 가치가 있다면 "언제 폴백 트레이드를 아예 고려할지"의 게이트 신호로만 취급
   해야지, EV 크기를 예측하는 신호로 곧이곧대로 받아들이면 안 된다.
3. **`net_diff`(방향) 쪽 유망 후보(`atr_pct_rank_288`, `compression_score`, `garch_vol_z`,
   `realized_vol_ratio`, `regime3_..._bear_prob`, `volatility_z`)가 더 의미 있다** — 이건 실제
   `long_ev >= short_ev` 분기, 즉 EV-HGB가 학습해야 할 진짜 결정에 대응하는 타겟이고, 가격오염
   IC가 전부 0.11 미만으로 낮다(즉 가격 자체와의 공선형성으로 설명되는 부분이 작다). IC 크기는
   0.055~0.134로, 이 저장소의 evidence-signal quant-use 서브프로젝트가 참고한 "0.02~0.05는 종종
   cost-gate를 못 버틴다"는 구간보다는 위이고, 다른 CLOSED 연구에서 "강한 신호"로 취급했던 0.3+
   구간에는 한참 못 미친다 — **애매한 중간 지대**다. 다만 이 타겟(`net_diff`)은 이미 3배 비용
   스트레스를 반영한 net 값이라, 이 IC가 살아남는다는 것 자체는 순수 raw-price IC보다는 조금 더
   의미가 있다(위 "방법" 절 마지막 문단 참고).
4. **`|price_contam_ic|`가 0.5 미만이라고 해서 오염이 0이라는 뜻은 아니다.** 특히 세 변동성 피처의
   0.19~0.32는 무시할 수 없는 수준이고, 레짐 라우터 확률들의 0.05~0.22도 마찬가지다. 이 수치들은
   "디스퀄리파잉"이 아니라 "완전히 깨끗하지는 않다"는 경고로 읽어야 한다.
5. **통계적으로 노이즈와 구분된다는 것과 트레이딩 가능한 엣지라는 것은 다르다.** n≈6,900~6,990에서
   셔플 null 표준편차가 0.011~0.013이므로 `|IC|>0.03` 정도만 돼도 이미 "2-표준편차 이상"으로
   찍힌다 — 표본이 크면 통계적 유의성 문턱은 낮다. 12개 유망 후보의 IC(0.03~0.16)는 이 기준을
   가볍게 넘지만, 이건 "이 rank correlation이 우연이 아니다"라는 것만 말해줄 뿐 "이 rank
   correlation을 실제 트레이딩 임계값/HGB 비선형 결합으로 바꿨을 때 비용을 넘는 EV가 나온다"는
   것까지 보장하지 않는다 — 그건 실제 학습 + purged CV + 라벨 순열 대조군 + N≥5 시드 + causal
   walk-forward를 거쳐야만 알 수 있다(이 문서의 "권장 다음 단계" 2~4번 그대로 유효).
6. **가장 중요한 사전 정보는 여전히 유효하다** — 이 저장소의 ETH 방향 예측 축은 여러 차례 no-skill로
   CLOSED됐다(메모리 `eth_odyssey_dl_rl_architecture_axis_closed_20260816` 등). 다만 이번 `net_diff`
   후보는 그 실패한 축들(직접적인 상승/하락 분류)과는 다른, 더 좁은 질문(비용 스트레스를 반영한
   TP:SL 배리어 트레이드의 방향 선호도 순위)을 묻고 있어 그 prior가 그대로 전이된다고 단정할 수는
   없다 — 그렇다고 반증되는 것도 아니다.

### 판정

**이 IC 체크는 결정적 음성이 아니다 — `docs/model_contracts/research_line_registry.json`에 CLOSED
항목을 추가하지 않았고, 계약 문서 상태도 CLOSED로 바꾸지 않았다.** 지시받은 대로, 신호가 보이는
경우 학습에 직접 착수하지 않고 아래처럼 구체적으로 보고한다:

- **가장 유망한 조합**: `net_diff` 타겟에 대한 `atr_pct_rank_288`(VAL +0.082/OOS +0.121),
  `compression_score`(-0.057/-0.120), `volatility_z`(+0.056/+0.134) — 셋 다 가격오염 IC가 낮고
  (<0.11), VAL보다 OOS-Q1에서 IC가 더 크다(과최적화 방향이 아님).
- **다음으로 유망**: `garch_vol_z`/`realized_vol_ratio`(`net_diff`, 둘 다 양 윈도우 "보통" 등급,
  0.055~0.091), `regime3_current_sensitive_wide24_bear_prob`(`net_diff`와 `max_net` 둘 다에서 유망,
  다만 가격오염 IC가 -0.11/-0.056으로 위 그룹보다 약간 높음).
- **`max_net`의 강한 변동성 신호(`garman_klass_vol`/`parkinson_vol`/`rogers_satchell_vol`,
  0.14~0.16)는 실재하지만, 위 2번 해석대로 "둘 중 하나가 이길 확률"에 가까운 신호일 가능성이 높고
  가격오염 IC도 세 그룹 중 가장 크다(0.19~0.32) — 액면 그대로 신뢰하지 말 것.**

실제 HGB 학습 착수 여부(그리고 착수한다면 이 6~7개 피처 중 어느 것을 포함할지)는 이 문서의 기존
"권장 다음 단계"(purged/embargo CV, 라벨 순열 대조군, N≥5 진짜 랜덤 시드, causal walk-forward)를
전제로 오케스트레이팅 세션/사용자 판단에 맡긴다.

### 준수 사항 (이 후속 조사)

`fresh_forward_bar_by_bar=true`(피처 값은 각 bar 자신의 이미 계산된 causal 피처, PRIMARY 파이프라인의
실시간 계약과 동일; 오라클 타겟은 cheap_gate 스크립트가 이미 "달성 불가능"으로 명시한 그 값 그대로).
`trade_ledgers_used_as_input=false`. `saved_parent_exit_timestamps_used=false`.
`future_rows_used_for_entry=false`(피처 기준; 타겟은 cheap_gate 자신의 오라클 정의를 그대로 재사용).
HGB 학습 없음, `trading_bot.py`/live 코드/`.env` 미변경, GPU 없음(`DEVICE=cpu`), conda env
`quant_ai`. 스크립트: `scripts/research_eth_candidate_cash_sleeve_ev_hgb_ic_check_20260816.py`.
출력: `tmp/causal_regen_20260516/eth_candidate_cash_sleeve_ev_hgb_ic_check_20260816/{ic_results.csv,
val_oos_consistency.csv, ic_check_summary.json, feature_target_join_{val,oos_q1}.csv}`.

## Stage 1 — 실제 HGB 학습 + purged CV + 라벨 순열(permutation) 대조군 (2026-08-16)

사용자가 실제 학습 착수를 명시적으로 승인해 진행한 첫 학습 단계다. **VAL 윈도우(2025-10-01~12-31)만
사용했고 OOS-Q1은 이 스크립트 어디에서도 로드/사용하지 않았다** — OOS-Q1은 이후 별도 단계(fresh-forward
walk-forward)를 위해 그대로 보존한다(지금 사용하면 모델/하이퍼파라미터 선택에 대한 누수가 된다).

구현: `scripts/research_eth_candidate_cash_sleeve_ev_hgb_train_stage1_20260816.py` (신규, CPU 전용
`DEVICE=cpu`, `trading_bot.py`/`trading_bot_modules/`/`runtime_config.py`/`.env` 미접촉, 어떤 import
모듈도 수정하지 않음).

### 방법론

**데이터/라벨 재사용(재구현 아님)**: `research_eth_candidate_cash_sleeve_ev_hgb_cheap_gate_20260816.
build_primary_ledger`(→G0 정합성 재확인, val: 41.13%/−21.70%/35 트레이드로 기존 표와 정확히 일치)로
PRIMARY 렛저를 얻고, 그 `_held_mask`로 CASH bar를 식별(6,986개, cheap_gate 문서와 정확히 일치)한 뒤,
`run_cash_sleeve_oracle`을 **직접 다시 호출**해 `long_net`/`short_net` 라벨(3배 비용 스트레스 시뮬레이션)
을 재생성했다. 이 재생성 결과를 이미 발행된 `cash_sleeve_oracle_bars_val.csv`와 `i` 인덱스·수치
(`np.allclose`, rtol=1e-9) 양쪽 다 바이트 단위로 대조해 divergence가 없음을 스크립트 자체가 assert로
검증한다(재사용 원칙 위반 방지).

**피처 패널(74개, 3그룹 — BTC 캐시 슬리브 `trading_bot_modules/omega1_2_3_cash_sleeve.py`
`Omega123CashSleeveAdapter`의 market/primary-trace/cash-state-history 3그룹 설계를 미러링, 이 파이프라인에
실재하는 것만 사용)**:

1. **market(25개)** — `research_eth_candidate_cash_sleeve_ev_hgb_ic_check_20260816.ALL_FEATURES`를
   그대로 재사용(모멘텀/추세 10 + 변동성 9 + 레짐/라우터 6, IC-check와 동일한 후보군).
2. **primary-trace(43개)** — h48qual/zig075 각각의 원본 예측 CSV(`gate.align_frame_and_predictions`로
   정렬)에 `train_eval_omega1_2_tabm_diffusion_risk_20260603._source_state`(이 계보가 이미 쓰는,
   raw ThreeHeadTabM 예측 CSV → `tabm_*` 트레이스 피처 변환 함수 — BTC 라이브 어댑터가 자기
   `_trace_features`를 만들 때 쓰는 것과 **정확히 같은 `tabm_` prefix 명명**, ETH에서 가장 충실한
   BTC-동등물)를 그대로 적용해 컴포넌트별 prefix(`h48qual_`/`zig075_`)를 붙였다(각 21개: 라우터
   bull/bear/chop 3 + router_confidence/margin 2 + dir_p_cash/long/short 3 + dir_confidence/side_edge/
   trade_prob/action 4 + quality_p_cash/long/short 3 + quality_for_action/threshold 2 + final_action 1 +
   long/short_quality_edge + abs_side_edge 3) + `sustained_uptrend_detector_active`(h48qual regime-aware
   exit guard와 zig075 SHORT 진입거부가 공유하는 바로 그 탐지기 마스크, 1개) = 43.
3. **cash-state-history(6개, 이번에 새로 짠 유일한 로직)** — `held`/`ledger`에서만 causal하게 유도:
   `primary_cash_streak`/`time_since_primary_exit`(tanh(streak/144), BTC 어댑터와 동일 스케일링 —
   BTC 자체 구현에서도 이 둘은 같은 값을 이름만 다르게 노출하므로 여기서도 동일하게 따름),
   `primary_active_roll_12`/`_48`(직전 12/48bar `held` 평균, bar i 자신은 제외해 causal 보장),
   `last_primary_active_len`(tanh(직전에 이미 종료된 포지션의 보유bar수/288), `exit_i<i`인 렛저 행만
   사용), `last_primary_side`(그 포지션의 side). 74개 중 2개는 거의 상수(near-constant, std<1e-12) —
   HGB가 자동으로 무시하므로 그대로 둠.

**모델**: `sklearn.ensemble.HistGradientBoostingRegressor` 2개(`long_model`/`short_model`), BTC
프로덕션 하이퍼파라미터를 그대로 이식(`max_iter=140, learning_rate=0.035, max_leaf_nodes=9,
l2_regularization=2.0` — `train_eval_omega1_2_3_cash_sleeve_upgrade_20260615.py:213`과 동일, 미수정).
단일 시드 `SEED=20260816`(N≥5 시드 재현은 이 스테이지의 결과에 따라 결정될 별도 후속 단계).

**Purged CV**: `core/event_label_engine.py`의 `purged_kfold_splits`(AFML Ch.7 purge+embargo)를
그대로 재사용 — 새로 작성하지 않았다. 이 함수는 이벤트별 `(event_idx, t1_idx)` 구간이 test fold와
겹치는 train 표본을 제거하도록 설계돼 있어, "CASH bar i의 라벨이 최대 `max_hold_bars`(=192)bar
앞선 가격 경로에 의존한다"는 이 실험의 이벤트 구조와 정확히 맞아떨어진다. `t1_idx`는 각 이벤트의
실제 청산 시점(TP/SL로 조기 청산될 수 있음) 대신 **보수적 상한** `min(event_idx + 192, n_bars-1)`을
모든 이벤트에 균일하게 사용했다 — 오케스트레이팅 세션의 지시("경계에서 192bar 이내 train 표본은
전부 제거")를 문자 그대로 만족시키기 위함이며, `embargo_frac=0`으로 뒀다(192가 이미 `t1_idx`에
내장돼 있으므로 이중 계산 방지). 5-fold, 실제 각 fold의 "test 구간 경계 − 남은 train 표본" 간격을
스크립트가 직접 계산해 192bar 미만이면 예외를 던지도록 assert했고, 실행 결과 5개 fold 전부
192~846bar 간격(양끝 fold는 한쪽 방향에 인접 fold가 없어 해당 방향 간격 없음, `nan`)으로 확인됐다 —
아래 표 참고.

**라벨 순열(permutation) 대조군**: fold마다 30회 반복 — 매 반복 train 구간의 `(long_net, short_net)`
을 **같은 행-순열**로 함께 섞고(둘의 실제 페어링을 보존), 실제 피처로 재학습한 뒤 held-out test fold에
예측, **섞이지 않은 실제 라벨**과 비교해 채점한다(귀무가설: "피처와 라벨 사이에 진짜 관계가 없다"를
정확히 구현). 30회 각각 5개 fold의 test 예측을 실제 모델과 동일하게 pooling해 반복당 하나의
pooled null 값을 얻고, 실제 모델의 pooled 값 1개와 직접 비교한다(30개 null 표본 vs 실측 1개, z-score
와 percentile rank 둘 다 계산).

**지표(사전 명시, 사후 선택 아님)**: (1) 순위 품질 — OOF pooled Spearman IC(예측 EV vs 실현
net_return), R²/MAE는 보조. (2) 결정 관련 지표 — `max(long_pred, short_pred) - ev_offset(=0, 이
스테이지는 calibration 안 함) > ev_min(=0.002)`으로 선택된 bar에서: 실제로 `ev_min`을 넘긴 비율,
선택된 bar들의 실현 평균 net edge(모델이 고른 쪽의 실제 결과), 이를 "무조건(모델이 고른 쪽을 전체
CASH bar에 적용했을 때의) 평균"과 비교한 값(`selected_minus_unconditional_pp`). **사전 등록한
합격 기준**: `long_ic`, `short_ic`, `selected_minus_unconditional_pp` **셋 다** 순열 null 대비
z≥2.0이어야 통과(셋 중 사후에 제일 좋아 보이는 것만 골라 보고하는 것을 방지하기 위해 스크립트 코드
자체에 이 규칙을 실행 전에 박아뒀다).

### 결과 — fold별 purge 간격 검증

| fold | n_train | n_test | test event 범위(bar idx) | purge_gap_before(bar) | purge_gap_after(bar) |
|---|---:|---:|---|---:|---:|
| 0 | 5,519 | 1,397 | [0, 8346] | (인접 train 없음) | 192.0 |
| 1 | 5,339 | 1,397 | [8347, 14017] | 846.0 | 192.0 |
| 2 | 5,370 | 1,397 | [14018, 20087] | 193.0 | 840.0 |
| 3 | 5,373 | 1,397 | [20088, 24808] | 480.0 | 192.0 |
| 4 | 5,396 | 1,398 | [24809, 26206] | 193.0 | (인접 train 없음) |

모든 fold에서 실측 purge 간격이 요구치(192bar) 이상 — 설계가 문자 그대로 지켜졌음을 실행 결과로도
확인.

### 결과 — 순위 품질(OOF pooled Spearman IC) 및 fold별 분해

| 항목 | 실제(pooled, n=6,986) | 순열 null 평균 | 순열 null std | z-score | percentile rank |
|---|---:|---:|---:|---:|---:|
| `long_model` IC | **−0.182** | −0.069 | 0.034 | **−3.35** | 0.00 |
| `short_model` IC | **+0.056** | −0.013 | 0.032 | **+2.13** | 0.97 |

R²/MAE(보조 지표, pooled): long R²=−0.347, MAE=0.00588 / short R²=−0.231, MAE=0.00566 (둘 다 음의
R² — 평균 예측보다도 못함, HGB가 held-out에서 사실상 아무 것도 설명하지 못한다는 뜻).

fold별 분해(라벨 순열과 무관, 각 fold의 실제 모델 OOF IC만):

| fold | n | long IC | short IC |
|---|---:|---:|---:|
| 0 | 1,397 | −0.426 | +0.026 |
| 1 | 1,397 | −0.343 | −0.139 |
| 2 | 1,397 | −0.170 | −0.015 |
| 3 | 1,397 | +0.243 | +0.328 |
| 4 | 1,398 | +0.271 | +0.407 |

### 결과 — 결정 관련 지표(ev_min 필터의 실제 트레이딩 유의미성)

| 항목 | 실제 | 순열 null 평균 | 순열 null std | z-score | percentile rank |
|---|---:|---:|---:|---:|---:|
| 선택된 bar 수 / 비율 | 1,641 / 23.49% | — | — | — | — |
| 선택된 bar 중 실현 `>ev_min` 비율 | 19.56% | 17.96% | 0.178 | +0.09 | 0.56 |
| 선택된 bar 평균 실현 net edge | **−0.309%** | −0.249% | 0.267 | −0.22 | 0.40 |
| 무조건(모델 선택 방향, 전체 CASH bar) 평균 | −0.210% | — | — | — | — |
| 선택 − 무조건(pp) | **−0.099pp** | −0.052pp | 0.263 | −0.18 | 0.40 |

**사전 등록 합격 기준(`long_ic`, `short_ic`, `selected_minus_unconditional_pp` 셋 다 z≥2.0) 결과:
FAIL** (`beats_permutation_null=false`) — `short_ic`만 개별적으로 z=2.13을 충족했을 뿐, `long_ic`는
null보다 훨씬 나쁘고(z=−3.35), 실제 트레이딩 의사결정에 대응하는 지표(`selected_minus_
unconditional_pp`)는 null과 통계적으로 구분되지 않는다(z=−0.18, 부호도 음수).

### 해석과 정직한 경고

1. **`long_model`은 노이즈보다도 나쁘다.** IC=−0.182, z=−3.35는 "예측력이 없다"가 아니라 "held-out에서
   체계적으로 틀린 방향으로 순위를 매긴다"는 뜻이다 — 순수 라벨 순열조차 이보다는 낫다(null 평균
   −0.069). 이는 흔한 과적합(순수 노이즈 학습)보다 더 나쁜 실패 양상으로, 74개 피처 중 다수가 서로
   상관된 변동성/레짐 피처라 특정 방향으로 체계적 바이어스를 학습했을 가능성을 시사한다.
2. **`short_model`의 IC=+0.056, z=+2.13은 통계적으로는 null과 구분되지만 fold별로 완전히 불안정하다.**
   fold 0~2(대략 10~11월)는 음수(−0.14~+0.03), fold 3~4(대략 12월)는 강한 양수(+0.33~+0.41)로
   **부호 자체가 VAL 윈도우 내부에서 뒤집힌다.** pooled z-score 하나만 보면 "합격선을 넘었다"고 오독할
   수 있지만, 이 fold-to-fold 분산은 이 저장소가 이미 반복적으로 확인한 패턴(메모리
   `tabm_hp_low_signal_pattern`: "cross-seed std가 전형적 효과 크기를 넘으면 노이즈") — 여기서는 시드가
   아니라 시간 구간(fold)에 대해서지만 같은 교훈이다: 하나의 pooled 숫자가 좋아 보여도, 그 밑에 있는
   구성요소가 이렇게 불안정하면 신뢰할 수 없다.
3. **가장 결정적인 것은 결정 관련 지표다.** `short_model`에 아주 약한 순위 신호가 있다 해도, 실제로
   "EV>ev_min일 때만 트레이드한다"는 트레이딩 규칙을 적용했을 때는 **아무 이득이 없다** —
   선택된 bar들의 평균 실현 edge(−0.309%)가 무조건 평균(−0.210%)보다 오히려 더 나쁘고, 이 차이
   자체가 순열 null 분포(평균 −0.052pp, std 0.263)와 통계적으로 전혀 구분되지 않는다(z=−0.18). 순위
   상관(IC)이 "약하지만 0은 아니다"에서 "실제 비용을 반영한 진입 결정 기준에서 이득이 된다"로 이어지지
   않는다는, 이 서브프로젝트가 IC-check 문서에서 이미 경고했던 바로 그 간극이 실측으로 재확인됐다.
4. **cheap_gate/IC-check 단계의 "애매한 중간 지대" 판단은 사후적으로 정당했다** — 결정적 음성이라고
   부르기엔 이르다고 봤던 판단은 옳았지만(무의미한 신호였다면 학습 자체가 시도할 가치가 없다고 판단해
   조기 종료했을 것), 실제 학습·purged CV·라벨 순열 대조군을 거치자 이 지점의 애매함은 "실제로 붙잡을
   수 있는 엣지"가 아니라 "노이즈에 가까운 신호가 학습 파이프라인의 자유도(74피처, HGB 비선형결합)
   앞에서 안정적으로 재현되지 않는다"는 쪽으로 해소됐다. 이는 이 저장소 이번 주 다른 조사들(GCE loss,
   전체 학습 레시피 번들)에서 이미 반복된 "cheap check에서는 유망해 보였다가 공정한 paired/permutation
   테스트에서 실패" 패턴과 정확히 같다.

### 판정

**Stage 1 사전 등록 결합 기준 FAIL — N≥5 시드 재현 및 fresh-forward walk-forward로 진행하지 않을 것을
권고한다.** `docs/model_contracts/research_line_registry.json`에 `eth_candidate_cash_sleeve_ev_hgb_
stage1_train_20260816` 항목을 추가했다(falsification_audit 취지 — 사전 등록 기준을 실패한 실제 학습
결과를 CLOSED로 기록). 이 후보는 공식 Odyssey 계보에 속한 적이 없으므로("Odyssey5" 명명 금지 원칙,
이 문서 최상단 참고) 계보 번호에는 영향 없다.

이유를 명확히 하면: 셋 중 하나(`short_ic`)만 개별적으로 통계 문턱을 넘었고, 그마저도 fold별로
부호가 뒤집힐 만큼 불안정했으며, 실제 트레이딩 판단에 대응하는 결정 관련 지표는 완전히 null과
구분되지 않았다(오히려 음수). "하나라도 좋아 보이는 지표를 골라 보고"하지 않기 위해 셋 다 통과해야
한다는 기준을 사전에 코드에 박아뒀고, 그 기준으로 명백히 실패했다.

### 준수 사항 (Stage 1)

`fresh_forward_bar_by_bar=true`(PRIMARY 렛저·`held`는 미수정 단일 causal bar-by-bar replay; 피처는
각 bar 자신의 이미 계산된 causal 컬럼/이미 학습이 끝난 고정 TabM 출력). `trade_ledgers_used_as_
input=false`(렛저는 CASH 구조적 사실과, causal하게 "이미 종료된" 트레이드의 cash-state-history
피처에만 쓰임). `saved_parent_exit_timestamps_used=false`. `future_rows_used_for_entry=false`(피처
기준; 라벨은 cheap_gate 자신의 오라클 정의를 그대로 재사용 — 오라클이 미래 가격 경로를 쓴다는 점은
처음부터 명시된 설계). OOS-Q1은 이 스크립트 어디에서도 로드/사용되지 않음. `trading_bot.py`/
`trading_bot_modules/`/`runtime_config.py`/`.env` 미변경, import 모듈 미수정, GPU 없음
(`DEVICE=cpu`), 단일 시드(`SEED=20260816`), conda env `quant_ai`. 스크립트:
`scripts/research_eth_candidate_cash_sleeve_ev_hgb_train_stage1_20260816.py`. 출력:
`tmp/causal_regen_20260516/eth_candidate_cash_sleeve_ev_hgb_train_stage1_20260816/{report.json,
oof_predictions.csv, fold_purge_diagnostics.csv, permutation_null.csv}`.
