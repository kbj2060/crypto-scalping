# 일리아스 — 방향 품질 반응형 exit 신호 베이스라인 구현·테스트 (2026-08-17)

이 문서는 `docs/model_contracts/ilias_eth_human_direction_risk_management_contract_20260817.md`의
1차 연구 질문(`docs/experiments/ilias_eth_adaptive_exit_direction_quality_signal_design_20260817.md`
설계, `docs/experiments/ilias_eth_exit_head_passivity_root_cause_20260817.md` 근본원인 진단)을
**실제로 구현하고 학습·백테스트한** 세션의 결과다. 설계문서의 성공/킬 기준은 **그대로 인용**하며
사후 변경하지 않는다.

## 1. 라벨 재정의 — 순환논리 함정 해결

### 1.1 함정 요약 (설계문서 함정 4, 근본원인 진단 §결론)

기존 h48qual exit_head가 방향 품질과 무관한 이유는 라벨 설계 자체가 방향 품질을 반영하지 않기
때문이다(근본원인 진단: 원본 라벨의 99.86%가 오라클 세그먼트 경계 신호, liveATR 재라벨의
75.7~79.8%가 국소 MFE-되돌림 노이즈). 새 신호가 exit_head의 **실제 발동 이력**을 라벨에 다시
섞으면("exit_head가 발동한 거래는 그 시점 손익 부호로 라벨링" 등) 기존 exit_head의 방향-무관
패턴을 재현하도록 학습되어 처음부터 결론이 오염된다.

### 1.2 채택한 해법 — 반사실적(counterfactual) TP/SL 배리어 재구성

계약이 지시한 대로: **exit_head의 실제 발동 여부를 완전히 무시**하고, 각 진입마다 그 포지션의
**실제 TP/SL 가격 배리어**만을 기준으로 가격 데이터를 순수하게 따라가 "SL을 먼저 건드리는가
TP를 먼저 건드리는가"를 판정해 라벨로 썼다. 구현: `scripts/research_ilias_eth_adaptive_exit_signal_common_20260817.py`의
`simulate_private_barrier_trades` — h48qual 컴포넌트 하나만의 **독립(private) 단일슬롯**
시뮬레이션(포트폴리오 슬롯 공유 없음)으로, 진입은 실배포 h48qual의 quality 게이트·ATR TP/SL·
리스크사이드카 사이징을 그대로 쓰되(방향은 실모델 그대로, 오버라이드 없음 — 라벨 생성 단계는
override 대상 아님), 보유 구간은 exit_head/레짐가드/trailing-stop을 전부 제거하고 **TP/SL 가격
배리어 도달만으로** 트레이드를 종결한다. 이는 표준 triple-barrier 라벨 생성 관행(오프라인 라벨
구성 단계에서 미래 가격 경로를 쓰는 것)이며, `.claude/CLAUDE.md` Fresh-Forward 규칙이 금지하는
"라이브 causal 결정"이 아니다 — 라벨(학습 타겟) 구성에만 미래 bar를 사용하고, feature(진입 시
quality, 미실현손익, MFE-so-far, 보유bar수 등, 전부 `pos_*` POS_COLS 계열 + entry quality 1개
파생)와 실제 fresh-forward replay 판단은 그 시점까지 causal한 정보만 쓴다(§2/§3에서 확인).

피처 행은 "실제 exit 결정이 평가됐을 bar"에서만 생성한다 — 원본 `greedy_replay_entry_veto`의
`if not reason:` 게이트(TP/SL이 그 bar의 종가에서 이미 트리거됐으면 exit_head 자체가 호출되지
않는다)를 그대로 재현해, TP/SL이 트리거된 바로 그 bar는 feature 행에서 제외한다(그 bar는
이미 결과가 확정된 bar이므로 "조기 개입 여부를 결정하는 시점"이 아니다).

### 1.3 엣지케이스 처리 (사전에 규칙을 정하고 데이터를 봄)

- **max_hold 강제청산**: 코드 확인 결과(`research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.greedy_replay_entry_veto` 직접 재확인)
  이 리플레이 엔진에는 max_hold_bars 강제청산 로직이 아예 없다(TP/SL/exit_head/trailing_stop
  4가지 사유만 존재) — 따라서 이 엣지케이스는 실제로 발생하지 않는다.
- **데이터 끝단 잘림**: 윈도우가 끝날 때까지 TP도 SL도 건드리지 않은 포지션(각 학습/평가 윈도우당
  보통 1건, 표본 대비 크지 않음)은 **완전히 드롭**한다(라벨을 추측하지 않음) — 정확한 카운트는
  각 스테이지의 `diag`에 `n_trades_truncated_open_at_frame_end`/`n_rows_truncated_dropped`로
  기록된다.

## 2. 베이스라인 신호 학습

### 2.1 라벨 데이터셋

- 스크립트: `scripts/research_ilias_eth_adaptive_exit_signal_labels_20260817.py`.
- 대상: **h48qual만**(계약 Open Issue (a) — zig075는 exit_head 관여 0/86이라 범위 밖, 근본원인
  진단이 이미 h48qual을 1순위로 확정). 실배포 `asymmetric_tabm_liveatr` 번들(`quality_threshold=0.5`,
  `atr_window=192`, `tp_mult=12.0`, `sl_mult=6.0`, `min_tp=0.075`, `min_sl=0.04`) 방향은
  오버라이드 없이 그대로.
- **TRAIN split 범위 주의사항**: 계약 Dataset Split 표의 명목 TRAIN 범위는 2024-01-01~2025-09-30
  (183,936행)이지만, 이 저장소에 실제로 존재하는 h48qual OOF 예측 CSV
  (`tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/h48qual/train_predictions_q050.csv`)는
  2025-01-01 04:55 ~ 2025-09-30 23:55(78,509행)만 커버한다(직접 확인, 2024년분 예측 파일 없음).
  이 스크립트는 따라서 **2025-01-01~2025-09-30**을 라벨 구성 윈도우로 썼다 — 명목 TRAIN 범위의
  진부분집합이지만 VAL(2025-10-01~)/OOS(2026-01-01~)는 전혀 건드리지 않으므로 causal split 규칙은
  준수한다. 게이트 모듈의 분기별(2025q1/q2/q3) 윈도우 대신 **연속된 단일 프레임**으로 로드해,
  분기 경계에서 트레이드가 인위적으로 잘리는 아티팩트를 피했다.
- 결과: 65건 해소(resolved) 트레이드, 1건 윈도우 끝단 잘림(드롭), 60,694개 라벨 행(라벨 SL
  양성률 63.9%). 트레이드당 평균 보유 bar 수가 매우 김(exit_head를 완전히 제거했으므로 실배포보다
  훨씬 김 — `min_tp=0.075`가 도달하기 어려운 넓은 배리어라 TP/SL 어느 쪽이든 도달까지 오래 걸림) —
  이는 §5 한계에서 다시 논의.
- 산출물: `tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/train_labels_h48qual_2025q1q3.csv`,
  `labels_report.json`.

### 2.2 Feature/모델

- Feature 14개(전부 재사용, 신규 원시 피처 없음 — Shared Feature Contract 준수): POS_COLS 13개
  (`pos_side`/`pos_hold_bars`/`pos_unrealized`/`pos_mfe`/`pos_mae`/`pos_giveback`/`pos_dist_to_tp`/
  `pos_dist_to_sl`/`pos_notional`/`pos_leverage`/`pos_exposure`/`pos_tp`/`pos_sl`,
  `trading_bot_modules/odyssey_tabm_core.py:45-59`) + 근본원인 진단이 유일하게 신규 파생 필요로
  지목한 `entry_quality_for_action`(진입 시 `quality_for_action`, 트레이드 내내 상수).
- 모델: 로지스틱 회귀(StandardScaler+LogisticRegression) vs 얕은 HistGradientBoostingClassifier
  (max_depth=4, max_iter=200, early_stopping) — **딥러닝/GPU 불필요**(`ml-adoption-playbook`
  베이스라인 우선 원칙).
- **사전등록 결정**(6개 평가 윈도우를 보기 전에 확정, 스크립트 자체에 문서화):
  - 결정 임계값 = 0.5(고정, 윈도우별 튜닝 없음).
  - 1차 모델 선택 = GroupKFold(5, trade_id 그룹화) 교차검증 ROC-AUC가 더 높은 쪽.
  - 배포 모델(arm-eval에서 실제 사용)은 전체 TRAIN 라벨셋(65 트레이드 전부)으로 재학습한 것 —
    CV 점수는 일반화 진단 전용, 실제 모델 자체는 아님.
- 결과(`train_report.json`): 로지스틱 회귀 CV AUC 평균 0.7296(폴드별 0.37~0.99, 폴드 표준편차
  큼 — 트레이드 수 65건/5폴드=폴드당 ~13트레이드로 표본이 작기 때문), HGB CV AUC 평균 0.6479.
  **1차 모델 = 로지스틱 회귀**(사전등록 규칙 그대로 적용).
- 산출물: `scripts/research_ilias_eth_adaptive_exit_signal_train_20260817.py`,
  `tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/{new_exit_signal_bundle.pkl,new_exit_signal_bundle_secondary.pkl,train_report.json}`.

## 3. "사람 방향 입력" 프록시 테스트 — 6개 윈도우

### 3.1 방법

- 방법 A(설계문서 1차 추천) 재사용: always_long/always_short/random(N=30, `np.random.SeedSequence(20260817)`,
  대규모 재검정 스크립트와 완전히 동일한 시드셋) — `prepare_component_direction_override`,
  `build_ablation_components`, `run_arm`을 `scripts/research_eth_odyssey4_random_direction_risk_management_ablation_20260817.py`/
  `scripts/research_eth_odyssey4_random_direction_large_n_reverification_20260817.py`에서 **무수정 재사용**.
- 6개 윈도우 전부(오늘 어블레이션이 이미 식별한 정확한 구간): VAL(2025-10-01~12-31)/OOS-Q1(2026-01-01~03-31)/
  OOS-Q2(2026-04-01~06-30, 전부 하락추세) + 레인지 3구간(2025-05-12~07-07, 2025-03-10~05-05,
  2026-02-09~04-06).
- **성공조건 1 측정 방식**: 발동률/precision을 **트레이드 단위**로 측정한다(오늘 어블레이션의
  "exit_head 비중" 관례와 동일 개념) — h48qual **단독**(포트폴리오 zig075 혼입 없음, 근본원인
  진단이 지적한 희석 문제 회피) 반사실적 배리어 시뮬레이션(§1.2와 동일 함수, 이번엔 방향
  오버라이드 arm에 적용)으로 각 트레이드의 실제 배리어 귀결(SL=1/TP=0)을 구하고, 학습된 분류기를
  같은 트레이드의 매 causal 보유-bar에 스코어링해 "이 트레이드 안에서 한 번이라도 P(SL)≥0.5를
  건드렸는가"(발동)와 "발동한 트레이드 중 실제로 SL로 끝난 비율"(precision)을 집계한다.
- **N=30/t-검정 해석 결정(설계문서 원문이 결정론적 arm 쌍에 이 공식을 어떻게 적용할지 명시하지
  않아 여기서 명시적으로 확정)**: `gap = firing_rate(always_long) − firing_rate(always_short)`
  (두 결정론적 arm의 점추정 차이), `std = std(N=30 random arm들의 같은 지표)`(같은 윈도우·같은
  분류기에서 방향만 무작위화했을 때 자연 변동폭의 노이즈-바닥 추정치 — 대규모 재검정 스크립트가
  이미 확립한 것과 동일한 "결정론적 앵커 vs random 분포" 구조를 재사용), `t = gap/(std/√30)`.
  이 해석은 6개 윈도우 결과를 보기 전에 스크립트에 고정했다(사후 조정 아님).
- 스크립트: `scripts/research_ilias_eth_adaptive_exit_signal_arm_eval_20260817.py`.

### 3.2 결과 — 성공조건 1 (발동률/precision, N=30, |t|>2)

**6/6 윈도우 전부 통과** — 대부분 매우 큰 |t| (6.1~42.4, 하나는 firing_rate가 random N=30 전부
1.0000으로 분산이 0이라 t가 정의되지 않지만(NaN) precision t=42.4로 통과).

| 윈도우 | 지표 | AL(always_long) | AS(always_short) | random(N=30) 평균±표준편차 | gap | t-stat | 유의 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| VAL(하락) | firing_rate | 1.0000 | 0.8519 | 0.8282±0.0392 | +0.148 | **20.70** | O |
| VAL(하락) | precision | 0.7857 | 0.5652 | 0.5904±0.0535 | +0.221 | **22.58** | O |
| OOS-Q1(하락) | firing_rate | 1.0000 | 0.7778 | 0.9122±0.0796 | +0.222 | **15.28** | O |
| OOS-Q1(하락) | precision | 0.7000 | 0.4286 | 0.6111±0.0958 | +0.271 | **15.51** | O |
| OOS-Q2(하락) | firing_rate | 1.0000 | 1.0000 | 0.9947±0.0203 | 0.000 | 0.00 | X |
| OOS-Q2(하락) | precision | 0.8750 | 0.5000 | 0.6832±0.0897 | +0.375 | **22.89** | O |
| 레인지①(2025-05-12~07-07) | firing_rate | 1.0000 | 0.9286 | 0.9296±0.0643 | +0.071 | **6.09** | O |
| 레인지①(2025-05-12~07-07) | precision | 0.5000 | 0.6923 | 0.8160±0.0816 | **−0.192** | **−12.90** | O(역방향) |
| 레인지②(2025-03-10~05-05) | firing_rate | 0.6667 | 1.0000 | 1.0000±0.0000 | −0.333 | NaN(σ=0) | — |
| 레인지②(2025-03-10~05-05) | precision | 1.0000 | 0.6250 | 0.6859±0.0484 | +0.375 | **42.42** | O |
| 레인지③(2026-02-09~04-06) | firing_rate | 1.0000 | 0.8000 | 0.9350±0.0724 | +0.200 | **15.12** | O |
| 레인지③(2026-02-09~04-06) | precision | 0.6667 | 0.5000 | 0.5639±0.1276 | +0.167 | **7.15** | O |

**주목할 예외 — 레인지①에서 precision 방향이 역전됨**: 다른 5개 윈도우 전부 "always_long(틀린
방향)의 precision이 always_short(옳은 방향)보다 높다"(양의 gap, 기대한 방향 — 틀린 방향일수록
SL로 끝날 확률이 높아 분류기가 발동했을 때 더 자주 맞음)는 패턴을 보이는데, 레인지①만 **반대**
(AL precision 0.50 < AS precision 0.69, t=−12.90, 매우 유의). 이 윈도우는 오늘 어블레이션에서도
이미 "실제모델이 무작위보다 나쁨"(gap/σ=−1.01~−3.36) 규모로 방향편향이 뒤집힌 곳으로 지목된
구간과 동일하다 — 레짐의존성이 exit 신호에도 그대로 전이됨을 보여주는 근거로, §3.4에서 다시
다룬다.

### 3.3 결과 — 성공조건 2 (fresh-forward replay, MDD 완화 + PnL 가드레일)

성공조건 1을 통과한 6개 윈도우 전부에서 성공조건 2를 검증했다 — **3/6 윈도우만 통과**
(OOS-Q1, OOS-Q2, 레인지③).

| 윈도우 | baseline(real exit_head) AL pnl/mdd | new_signal AL pnl/mdd | MDD 완화? | baseline AS pnl | new_signal AS pnl | 가드레일(≤50%악화·부호반전금지) | 조건2 |
|---|---|---|---:|---:|---:|:---:|:---:|
| VAL | −14.46% / −36.56% | −37.26% / **−46.18%** | **X**(악화) | +58.48% | +59.46% | O | **X** |
| OOS-Q1 | −41.61% / −50.83% | −37.45% / **−47.32%** | O | +82.13% | +76.50%(−6.9%) | O | **O** |
| OOS-Q2 | −26.96% / −35.94% | −26.35% / **−35.40%** | O | +48.89% | +36.26%(−25.8%) | O | **O** |
| 레인지①(2025-05-12~07-07) | −10.70% / −31.44% | −10.70% / −31.44%(변화없음) | O(동률) | −6.18% | −16.10%(160.5%악화) | **X** | **X** |
| 레인지②(2025-03-10~05-05) | −3.47% / −14.90% | −3.47% / −14.90%(변화없음) | O(동률) | +4.09% | −5.50%(부호반전) | **X** | **X** |
| 레인지③(2026-02-09~04-06) | −15.49% / −25.45% | **−10.27% / −20.88%** | O | +4.58% | +3.00%(−34.5%) | O | **O** |

가드레일 함수는 `research_eth_omega461_gittins_index_exit_head_20260814._guardrail_pass`의
"부호반전 금지·50% 상대악화 이내" 관례를 재사용하되, 이 실험의 always_short-type 베이스라인이
음수인 레인지 윈도우가 있어 부호에 무관하게 동작하도록 일반화했다(`_guardrail_pass`,
`scripts/research_ilias_eth_adaptive_exit_signal_arm_eval_20260817.py`) — 최초 구현에 음수
베이스라인 분기의 부호 버그가 있어 레인지①이 잘못 통과 판정됐던 것을 재실행 전에 발견·수정하고
전체를 재실행했다(정직성 기록).

레인지①·②에서 "new_signal AL pnl/mdd가 baseline과 완전히 동일"한 것은 버그가 아니다. 두
윈도우 모두 always_long 트레이드 표본이 매우 작다(레인지① with_gate 3건/raw 6건, 레인지② with_gate
3건/raw 6건 — `arm_eval_report.json`의 `baseline_always_long_with_gate.trades`/`skipped` 직접
재확인). 레인지②는 새 신호가 raw 6건 중 **한 번도 발동하지 않아**(reason_counts에 `new_exit_signal`
키 자체가 없음, TP/SL만 3/3) baseline과 완전히 같은 결과가 나온 것이 자명하다. 레인지①은 raw
6건 중 2건이 `new_exit_signal`로 조기청산됐지만(reason_counts `{take_profit:4, stop_loss:8,
new_exit_signal:2}` — 합 14로 range의 raw trades와 다른 것은 no_gate 전체 창 기준 누적이기
때문), with_gate 필터(duration gate)가 그 2건을 정확히 걸러낸(`skipped=3`) 나머지 3건에는
포함되지 않아 **with_gate 지표에는 반영되지 않았다** — 우연의 일치이지, 새 신호가 무해했다는
뜻은 아니다.

### 3.4 레짐의존성 — 하나만 보고 결론내지 않음

- **하락추세 3창**: 성공조건 1은 3/3 통과, 성공조건 2는 **2/3 통과**(OOS-Q1, OOS-Q2 O / VAL X).
  VAL이 유일하게 조건2에서 탈락한 하락추세 창이다 — MDD가 오히려 −36.56%→−46.18%로 악화됐다.
- **레인지 3구간**: 성공조건 1은 3/3 통과, 성공조건 2는 **1/3 통과**(레인지③ O / 레인지①·② X).
  통과한 레인지③(2026-02-09~04-06)은 오늘 어블레이션에서 이미 "잔여방향성"(AL-AS 스프레드
  20.06pp, 순수 레인지로 보기 어려움)으로 분류된 구간이고, 탈락한 레인지①·②(스프레드 4.52pp/
  7.56pp, 더 순수한 무방향 구간)에서는 always_short-type PnL 자체가 가드레일을 어겼다(레인지①
  160.5% 악화, 레인지② 부호반전).
- **종합**: 조건2 통과 3창(OOS-Q1/OOS-Q2/레인지③)은 **하락추세 2개 + 레인지 1개**로, 특정
  레짐 하나에 몰려있지 않다 — 그러나 "순수 레인지"(스프레드가 가장 작은 두 구간)에서는 조건2가
  **일관되게 실패**했다는 점이 명확한 패턴이다. 이는 오늘 어블레이션의 기존 발견("저스프레드
  레인지에서 실제 모델이 무작위보다 유의하게 나쁨", t=−3.36/−5.05)과 방향이 일치한다 — 새 exit
  신호도 같은 저스프레드-레인지 조건에서 방향 정보 자체가 가장 빈약해지는 것으로 보인다.

## 4. 정직한 판정

**성공조건 1(발동률/precision이 방향 품질에 따라 유의하게 다르다) — 명확히 통과, 6/6 윈도우, 강한
근거.** 이 로지스틱 회귀 베이스라인 1개 config만으로도, 기존 exit_head의 21.8~27.7% flat 패턴과
달리, 방향 품질(always_long-type vs always_short-type)에 따라 precision이 통계적으로 매우
유의하게 갈린다(|t|=6.1~42.4, 대부분 |t|>15). 이는 "post-entry causal 상태에는 방향 품질을
판별할 정보가 전혀 없다"는 킬 가설을 명확히 기각한다 — **근본원인 진단의 예측(라벨만 재정의하면
방향 품질 반응성이 생긴다)이 실증적으로 확인됐다.**

**성공조건 2(fresh-forward replay가 실제 MDD/PnL을 개선한다) — 부분 통과, 3/6 윈도우.**
OOS-Q1/OOS-Q2(하락추세)와 레인지③(2026-02-09~04-06, 잔여방향성 구간)에서 always_long-type MDD가
실제로 완화되면서(각각 -50.83%→-47.32%, -35.94%→-35.40%, -25.45%→-20.88%) always_short-type
PnL은 가드레일 안에서 유지됐다(각각 -6.9%/-25.8%/-34.5% 상대변화, 전부 ≤50%). 반면 VAL(하락추세)과
순수 레인지 2구간(레인지①·②)에서는 조건2가 실패했다 — VAL은 MDD 자체가 악화, 레인지①·②는
always_short-type PnL이 가드레일(50%악화/부호반전 금지)을 위반.

**종합 판정: 조건부 성공(Partial success) — 킬 아님, 그러나 전면 승격 근거도 아님.** 설계문서의
성공조건은 "성공조건 1을 충족 못하면 킬"이라고만 명시했고 조건2가 몇 개 창에서 통과해야
"성공"인지는 명시하지 않았다(계약이 이미 레짐의존성을 Open Issue (d)로 열어둔 것과 정합) —
그래서 이 문서는 사후에 기준을 새로 만들지 않고, **윈도우별 결과를 있는 그대로 보고**한다:
이 베이스라인 신호는 exit_head를 능동형으로 바꾸는 메커니즘 자체는 실증했지만(조건1),
그 메커니즘을 실제 PnL/MDD 개선으로 전환하는 데는 레짐에 따라 성패가 갈린다(조건2, 순수
레인지에서 특히 약함).

**증거 강도**: 조건1/조건2 전부 **단일 config**(로지스틱 회귀 1회, 임계값 0.5 고정, 하이퍼파라미터
스윕 없음) 결과다. [[feedback_dl_needs_optimization_before_failure_verdict]] 원칙에 따라, 조건2가
실패한 3개 윈도우를 "이 축이 원리적으로 안 된다"로 확대해석하지 않는다 — 임계값 튜닝(현재 고정
0.5), HGB 대안(2차 후보로 저장됨, CV AUC는 로지스틱보다 낮았으나 임계값 조정 시 다를 수 있음),
발동 시점을 늦추는 하한(예: 최소 hold_bars 이후에만 발동 허용, 지금은 즉시 발동 가능) 등은
아직 탐색하지 않은 여지다. 반대로 조건1의 통과(6/6, 대부분 |t|>15)는 여러 독립 윈도우·매우 큰
효과크기로 반복 확인됐으므로 "방향 품질을 반영하는 라벨로 재정의하면 exit 신호가 능동형이 된다"는
근본원인 진단의 핵심 주장 자체는 강한 근거로 지지된다.

**다음 단계 제안(승격 판단 아님, 관찰)**: 조건2가 통과한 3창(OOS-Q1/Q2, 레인지③)과 실패한
3창(VAL, 레인지①·②)을 가르는 변수가 무엇인지(레짐, 트레이드 표본 수, 발동 타이밍)는 이번
세션에서 분리하지 못했다 — 후속 세션이 있다면 우선 확인할 질문으로 남긴다.

## 5. 한계

- **작은 표본**: TRAIN 라벨 트레이드가 65건뿐(exit_head를 완전 제거했기 때문에 트레이드당 평균
  보유기간이 매우 길어져, 같은 9개월 구간에서도 실배포보다 트레이드 수 자체가 적다). GroupKFold
  AUC의 폴드간 분산이 커서(0.37~0.99) 일반화 성능 추정 자체의 불확실성이 크다.
- **단일 config**: 로지스틱회귀/얕은 HGB 각 1회 학습(하이퍼파라미터 스윕 없음). 이번 결과는
  "이 베이스라인 config로는 [성공/킬]"이지, "이 방향의 신호가 원리적으로 불가능/가능하다"는
  강한 주장이 아니다([[feedback_dl_needs_optimization_before_failure_verdict]] 원칙).
- **트레이드 보유기간 분포 이동**: 반사실적 라벨 생성이 exit_head를 완전히 제거하므로, 학습·평가
  양쪽 모두에서 관측되는 보유-bar 상태는 실배포보다 "더 깊은" 단계(hold_bars가 큰 상태)를 더 많이
  포함한다 — 학습/평가가 동일 가정을 공유하므로 성공조건 1 비교 자체는 내적으로 일관되지만, 성공
  조건 2의 실제 라이브 배포 시 발동 시점 분포는 이 연구가 직접 관찰한 반사실적 분포와 다를 수 있다.
- **발동률이 시간에 따라 포화하는 경향**: 임계값 0.5의 분류기는 트레이드가 충분히 길게 지속되면
  거의 항상 한 번은 P(SL)≥0.5를 넘는 경향이 있어(§3.2 표 참고), "발동률" 지표 자체가 방향 품질보다
  "얼마나 오래 버텼는가"에 더 민감할 수 있다 — precision 지표가 이를 보완하지만, 두 지표의 상대적
  정보량 차이는 이번 세션에서 완전히 분해하지 못했다.
- Fresh-Forward 준수: `fresh_forward_bar_by_bar=true`(criterion 2 replay), 라벨 생성 자체는 오프라인
  triple-barrier 구성(§1.2 설명). `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
  `future_rows_used_for_entry=false`.

## 6. 산출물 전체 목록

- `scripts/research_ilias_eth_adaptive_exit_signal_common_20260817.py` — 공유 반사실적 시뮬레이션 +
  신규 replay 함수.
- `scripts/research_ilias_eth_adaptive_exit_signal_labels_20260817.py` — TRAIN 라벨 생성.
- `scripts/research_ilias_eth_adaptive_exit_signal_train_20260817.py` — 베이스라인 분류기 학습.
- `scripts/research_ilias_eth_adaptive_exit_signal_arm_eval_20260817.py` — 6윈도우 성공/킬 평가.
- `tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/` — 전체 산출물
  디렉토리(라벨 CSV, 모델 pkl, 리포트 json, arm 평가 CSV).

**이 문서의 §1~§6은 이후 §7이 정정한 오염된 학습 결과를 그대로 보존한 원본 기록이다 — 삭제/수정하지
않았다.** §7을 먼저 읽고 §1~§6을 "정정 전 실행 로그"로 참고할 것.

## 7. 정정 — `pos_side` quasi-separation 발견 및 side-blind 재검증 (2026-08-17, 같은 날 후속 세션)

### 7.1 발견 경위

사용자가 §2.2의 로지스틱 회귀 번들(`new_exit_signal_bundle.pkl`)을 직접 열어 표준화 계수를
확인할 것을 요청했다. 결과(절대값 순):

| feature | 표준화 계수 |
|---|---:|
| `pos_side` | **-27.14** |
| `pos_leverage` | **-25.52** |
| `pos_exposure` | **-22.65** |
| `pos_notional` | **+21.32** |
| `pos_tp` | -3.67 |
| `pos_giveback` | +0.50 |
| `pos_mfe` | -0.47 |
| `pos_hold_bars` | +0.42 |
| `entry_quality_for_action` | -0.40 |
| `pos_mae` | -0.31 |
| `pos_dist_to_tp` | -0.017 |
| `pos_dist_to_sl` | -0.0011 |
| `pos_unrealized` | -0.0011 |

상위 4개(`pos_side`/`pos_leverage`/`pos_exposure`/`pos_notional`)가 나머지 전부보다 한 자릿수 이상
크다 — 표준화 로지스틱 회귀에서 계수가 20~27에 달하는 것은 quasi-separation(거의 결정론적 분리)의
전형적 신호다. `pos_side`는 always_long/always_short arm을 정의하는 상수(+1/-1) 그 자체이고,
TRAIN 라벨 구간(2025-01-01~09-30)은 실제로 SHORT가 유리했던 하락추세였다(`docs/model_contracts/
odyssey4_eth_entry_veto_baseline_contract_20260814.md` 등 상위 서사가 반복 확인). 즉 이 모델은
"이 트레이드가 잘 되고 있는가"(`pos_unrealized`/`pos_dist_to_sl`, 계수 사실상 0)를 배운 게 아니라
**"이 TRAIN 구간에서는 SHORT가 이겼다"를 `pos_side`(+상관된 사이징 출력)로 암기**했을 가능성이 높다.
이것이 §3.2의 "성공조건 1 6/6 윈도우, 대부분 \|t\|>15" 결과의 (적어도 부분적인) 진짜 원인이었을
수 있다는 것이 본 절의 재검증 대상이다.

### 7.2 재학습 — side-blind feature 목록

- 제거: `pos_side`, `pos_leverage`, `pos_notional`, `pos_exposure`(방향/사이징 노출 4개).
- 유지(10개): `pos_hold_bars`, `pos_unrealized`, `pos_mfe`, `pos_mae`, `pos_giveback`,
  `pos_dist_to_tp`, `pos_dist_to_sl`, `pos_tp`, `pos_sl`, `entry_quality_for_action` — 전부
  경로의존적이거나 side-정규화된 값(방향 자체를 직접 노출하지 않음).
- **라벨 CSV 재사용**: `train_labels_h48qual_2025q1q3.csv`(65트레이드/60,694행)를 그대로 재사용했다
  — `label_sl`은 반사실적 TP/SL 배리어 귀결로 정의되며(§1.2), 실배포 h48qual의 실제 방향으로
  시뮬레이션됐을 뿐 오버라이드된 적이 없으므로 side와 무관하게 이미 정의돼 있다. 재생성 불필요.
- 스크립트: `scripts/research_ilias_eth_adaptive_exit_signal_train_sideblind_20260817.py`(원본
  `..._train_20260817.py`의 문서화된 복사본, feature 목록만 변경 — 원본은 무수정 보존).
- 사전등록 결정(결정임계값 0.5, GroupKFold(5) CV AUC로 모델 패밀리 선택, 전체 TRAIN으로 재학습)은
  전부 원본과 동일하게 유지했다.
- 공유 모듈: `scripts/research_ilias_eth_adaptive_exit_signal_common_sideblind_20260817.py` — 라벨/
  피처 생성 함수(`simulate_private_barrier_trades`)는 원본에서 무수정 재수입(피처 부분집합과
  무관하게 항상 전체 raw 컬럼을 출력하므로 재구현 불필요), `score_new_exit_signal`과
  `greedy_replay_new_exit_signal`만 일반화(고정 14컬럼 순서 가정 대신 `bundle["feature_columns"]`를
  이름으로 조회 — 원본 그대로 쓰면 10컬럼 모델에 14컬럼 벡터를 넣어 sklearn shape mismatch가 남).

### 7.3 재학습 결과 — 계수 재확인 (요청대로, 숨기지 않고 보고)

GroupKFold(5) CV AUC: 로지스틱 회귀 0.6488(원본 0.7296에서 하락, 폴드별 0.459~0.830으로 여전히
분산 큼), HGB 0.4210(원본 0.6479에서 큰 폭 하락, 무작위 수준 이하 폴드 존재). **1차 모델 = 로지스틱
회귀**(사전등록 규칙 그대로, CV AUC 더 높음).

재학습된 로지스틱 회귀의 표준화 계수(절대값 순, 10개 전부):

| feature | 표준화 계수 |
|---|---:|
| `pos_tp` | -0.4414 |
| `pos_mfe` | -0.4012 |
| `pos_hold_bars` | +0.3641 |
| `pos_giveback` | +0.3202 |
| `entry_quality_for_action` | -0.2339 |
| `pos_mae` | -0.1488 |
| `pos_dist_to_sl` | -0.0972 |
| `pos_unrealized` | -0.0972 |
| `pos_dist_to_tp` | +0.0951 |
| `pos_sl` | +0.0000 |

**quasi-separation 재발 없음** — 최대 절대값이 0.4414(`pos_tp`)로, 원본 모델의 상위 4개(21~27)와
비교 불가할 만큼 정상 범위다. `pos_sl`은 사실상 0(다른 피처와의 공선성으로 보임), `pos_dist_to_sl`과
`pos_unrealized`가 완전히 동일한 값(-0.0972)을 보이는 것도 quasi-separation이 아니라 두 피처가
서로 거의 공선(`pos_dist_to_sl = move + |stop_loss|`, `pos_unrealized = move`)이라 로지스틱
회귀가 계수를 균등 분배한 것으로 해석된다. `pos_tp`/`pos_sl`이 우연히 TRAIN 구간 방향과 상관됐을
가능성도 배제하지 않고 확인했으나, 재발 징후는 없었다.

산출물: `scripts/research_ilias_eth_adaptive_exit_signal_train_sideblind_20260817.py`,
`tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/
{new_exit_signal_bundle_sideblind.pkl,new_exit_signal_bundle_sideblind_secondary.pkl,
train_report_sideblind.json}`.

### 7.4 6윈도우 재검증 — 정확히 같은 절차/기준

스크립트: `scripts/research_ilias_eth_adaptive_exit_signal_arm_eval_sideblind_20260817.py`(원본
`..._arm_eval_20260817.py`의 문서화된 복사본 — 성공/킬 기준, 6개 윈도우, N=30 시드셋
(`np.random.SeedSequence(20260817)`, 원본과 완전히 동일), t-검정 해석, 가드레일 함수는 전부
**변경 없이 그대로**. 바뀐 것은 오직: side-blind 번들 로드, criterion1의 feature 선택을
`bundle["feature_columns"]`(10개)로, criterion2 리플레이를
`greedy_replay_new_exit_signal_sideblind`로 교체한 것뿐. 원본 스크립트/산출물은 무수정 보존.

`fresh_forward_bar_by_bar=true`(criterion 2 replay), `trade_ledgers_used_as_input=false`,
`saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false` — 원본과 동일.

### 7.5 결과 — 성공조건 1 (발동률/precision, N=30, \|t\|>2) — **6/6 윈도우 전부 통과, 여전히**

| 윈도우 | 지표 | 원본(오염) \|t\| | side-blind \|t\| | side-blind AL | side-blind AS | side-blind random 평균±표준편차 |
|---|---:|---:|---:|---:|---:|---:|
| VAL | firing | 20.70 | 13.97 | 1.0000 | 0.9630 | 0.9684±0.0145 |
| VAL | precision | 22.58 | **35.84** | 0.7857 | 0.5385 | 0.5609±0.0378 |
| OOS-Q1 | firing | 15.28 | NaN(σ=0) | 1.0000 | 1.0000 | 1.0000±0.0000 |
| OOS-Q1 | precision | 15.51 | **16.21** | 0.7000 | 0.4444 | 0.5768±0.0863 |
| OOS-Q2 | firing | 0.00 | NaN(σ=0) | 1.0000 | 1.0000 | 1.0000±0.0000 |
| OOS-Q2 | precision | 22.89 | 23.04 | 0.8750 | 0.5000 | 0.6850±0.0892 |
| 레인지①(05-12~07-07) | firing | 6.09 | NaN(σ=0) | 1.0000 | 1.0000 | 1.0000±0.0000 |
| 레인지①(05-12~07-07) | precision | -12.90 | -9.05 | 0.5000 | 0.6429 | 0.7579±0.0865 |
| 레인지②(03-10~05-05) | firing | NaN(σ=0) | NaN(σ=0) | 1.0000 | 1.0000 | 1.0000±0.0000 |
| 레인지②(03-10~05-05) | precision | 42.42 | 42.42 | 1.0000 | 0.6250 | 0.6859±0.0484 |
| 레인지③(02-09~04-06) | firing | 15.12 | NaN(σ=0) | 1.0000 | 1.0000 | 1.0000±0.0000 |
| 레인지③(02-09~04-06) | precision | 7.15 | 11.92 | 0.6667 | 0.4000 | 0.5264±0.1225 |

**핵심 관찰(예상과 다름, 숨기지 않고 보고)**: pos_side 등 4개 방향/사이징 컬럼을 제거했음에도 \|t\|가
**전반적으로 작아지지 않았다** — 6개 윈도우 중 4개(VAL/OOS-Q1/OOS-Q2/레인지③)에서 precision \|t\|가
오히려 비슷하거나 커졌고(22.58→35.84, 15.51→16.21, 22.89→23.04, 7.15→11.92), 1개(레인지②)는
완전히 동일(42.42→42.42, AL/AS precision 점추정치 자체가 정확히 일치), 1개(레인지①)만 작아졌다
(-12.90→-9.05, 여전히 매우 유의). firing_rate는 대부분 윈도우에서 AL/AS 둘 다 1.0000으로 포화돼
random(N=30) 분산이 0이 되며 t가 정의되지 않는 경우가 늘었다(원본 2개→side-blind 5개) — 그러나
criterion1_pass는 firing_rate OR precision 중 하나만 유의하면 통과이므로(사전등록 기준 그대로),
6/6 전부 precision으로 통과했다.

**해석**: 제거된 4개 컬럼이 표준화 계수 크기로는 압도적이었지만, 성공조건 1의 통계적 유의성 대부분은
그 컬럼들에 의존하지 않았다 — 남은 10개 피처(특히 `pos_unrealized`/`pos_mfe`/`pos_dist_to_sl`)
자체가 방향 품질에 따라 **실제로 다르게 움직인다**(틀린 방향 포지션은 가격이 실제로 불리하게
움직이므로 미실현손익/MFE/SL까지거리 궤적이 진짜로 더 나쁘다) — 이는 `pos_side`를 암기하는 것과
질적으로 다른, 포지션 자체의 causal 상태에 근거한 반응성이다. **성공조건 1은 side-blind 재검증을
통과했다 — 진짜 방향-품질-반응 신호로 재확인됐다.**

### 7.6 결과 — 성공조건 2 (fresh-forward replay, MDD 완화 + PnL 가드레일) — **3/6 윈도우 통과(원본과 동일 개수, 다른 구성)**

| 윈도우 | 원본(오염) 조건2 | side-blind 조건2 | side-blind AL mdd(base→new) | side-blind AS pnl(base→new) | side-blind 가드레일 |
|---|:---:|:---:|---|---|:---:|
| VAL | X | **X**(동일 실패) | -36.56%→-41.65%(악화) | +58.48%→+37.03% | O |
| OOS-Q1 | O | **O**(동일 통과) | -50.83%→-47.27%(개선) | +82.13%→+75.86%(-7.6%) | O |
| OOS-Q2 | O | **X (반전!)** | -35.94%→**-40.23%(악화)** | +48.89%→+36.21%(-25.9%) | O이나 MDD 조건 자체 실패 |
| 레인지①(05-12~07-07) | X | **X**(동일 실패) | -31.44%→-31.44%(동률) | -6.18%→-17.93%(190%악화) | X |
| 레인지②(03-10~05-05) | X | **O (반전!)** | -14.90%→-14.90%(동률) | +4.09%→+2.77%(-32.3%, 부호유지) | O |
| 레인지③(02-09~04-06) | O | **O**(동일 통과) | -25.45%→-20.82%(개선) | +4.58%→+5.46%(개선) | O |

**두 곳에서 판정이 뒤집혔다**:
- **OOS-Q2(하락추세) — 통과→실패**: 원본 모델은 이 윈도우에서 AL MDD를 -35.94%→-35.40%로
  완화시켰지만, side-blind 모델은 오히려 -35.94%→-40.23%로 **악화**시켰다. `pos_side` 등이 남아
  있을 때는 "이 트레이드가 always_long(구조적으로 불리한 방향)"이라는 정보를 직접 사용해 더 빨리/
  똑똑하게 청산할 수 있었지만, 그 정보를 제거하자 이 윈도우에서는 순수 post-entry 상태만으로는
  실질적 MDD 개선을 못 만들어냈다 — 이 윈도우의 원본 조건2 통과는 부분적으로 방향 정보 누출
  덕분이었을 가능성이 있다.
- **레인지②(순수 레인지) — 실패→통과**: 원본 모델은 이 윈도우에서 always_short-type PnL이
  부호반전(+4.09%→-5.50%)해 가드레일을 위반했지만, side-blind 모델은 부호를 유지(+4.09%→+2.77%,
  -32.3%로 가드레일 이내)해 통과했다. 방향 정보를 제거하자 오히려 이 순수 레인지 구간에서는 더
  안정적으로 작동했다 — 원본 모델의 이 윈도우 실패가 방향 정보에 대한 과적합(레인지장에서는
  방향 정보 자체가 오도적일 수 있음, §3.4의 레짐의존성 논의와 일치)이었을 가능성을 시사한다.
- VAL/레인지①(실패)과 OOS-Q1/레인지③(통과)은 원본과 side-blind가 동일하게 판정했다.

**레짐별 재집계**: 하락추세 3창 통과 1/3(원본 2/3에서 감소, OOS-Q2 상실) — 레인지 3창 통과 2/3
(원본 1/3에서 증가, 레인지② 획득). 총합은 3/6으로 원본과 같지만 구성이 달라졌다 — "하락추세에서
더 잘 통한다"는 원본의 인상은 side-blind 재검증에서 유지되지 않는다.

### 7.7 정정된 정직한 판정

**성공조건 1 — 재확인, side-blind에서도 명확히 통과(6/6 윈도우, \|t\|=9.05~42.4, 대부분 \|t\|>15).**
`pos_side`/`pos_leverage`/`pos_notional`/`pos_exposure`를 제거해도 발동률/precision이 방향 품질에
따라 통계적으로 매우 유의하게 갈린다 — §7.5의 관찰대로 \|t\|가 전반적으로 줄지 않았다는 점은
"원본 결과가 순전히 `pos_side` 암기였다"는 우려와 반대되는 증거다. **이것은 진짜
방향-품질-반응(post-entry causal 상태 기반) 신호다** — pos_side quasi-separation은 실재했지만,
성공조건 1의 결론 자체를 무효화하지는 않았다.

**성공조건 2 — 부분 통과 유지(3/6), 그러나 어느 창이 통과하는지는 바뀌었다.** 원본이 "하락추세에
더 잘 듣는다"는 인상을 줬다면, side-blind 재검증은 그 인상이 방향 정보 누출에 일부 의존했음을
보여준다(OOS-Q2 상실) — 반대로 순수 레인지에서의 실패도 일부는 방향 정보 과적합이었을 가능성을
보여준다(레인지② 회복). 3/6이라는 헤드라인 숫자는 우연히 같지만, **레짐별 해석은 원본 문서(§3.4)를
그대로 신뢰할 수 없다** — 이 정정 이후에는 "어느 레짐에서 통하는가"를 다시 열린 질문으로 취급해야
한다.

**최종 판정: 진짜 신호 확인(Confirmed genuine, 부분) — 킬 아님.** exit_head를 방향-무관 배경신호에서
방향-품질-반응형으로 바꾸는 메커니즘 자체(성공조건 1)는 `pos_side` 오염 없이도 강한 근거로 재확인
됐다. 이를 실제 PnL/MDD 개선으로 전환하는 능력(성공조건 2)은 여전히 부분적이고(3/6), 원본이
암시했던 "하락추세 우위" 패턴은 side-blind 재검증에서 사라졌다 — 다음 단계 결정에 있어 레짐별
재분석이 필요하다는 점이 이번 정정의 새로운 발견이다.

**증거 강도**: 여전히 단일 config(로지스틱 회귀 1회, 임계값 0.5 고정) 결과다
([[feedback_dl_needs_optimization_before_failure_verdict]] 원칙 재적용). N=65 트레이드라는
작은 표본 캐비어트도 §5와 동일하게 적용된다.

### 7.8 이 절의 산출물

- `scripts/research_ilias_eth_adaptive_exit_signal_common_sideblind_20260817.py` — side-blind
  feature 목록 + 일반화된 `score_new_exit_signal`/`greedy_replay_new_exit_signal_sideblind`
  (원본 `..._common_20260817.py`는 무수정 보존, `simulate_private_barrier_trades`만 재수입).
- `scripts/research_ilias_eth_adaptive_exit_signal_train_sideblind_20260817.py` — side-blind
  재학습(계수 재확인 로직 포함, 원본 `..._train_20260817.py`는 무수정 보존).
- `scripts/research_ilias_eth_adaptive_exit_signal_arm_eval_sideblind_20260817.py` — side-blind
  6윈도우 재검증(원본 `..._arm_eval_20260817.py`는 무수정 보존).
- `tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/
  {new_exit_signal_bundle_sideblind.pkl,new_exit_signal_bundle_sideblind_secondary.pkl,
  train_report_sideblind.json,arm_eval_criterion1_by_arm_sideblind.csv,
  arm_eval_report_sideblind.json}` — 전부 원본 파일과 다른 파일명(`_sideblind` 접미사)으로 저장,
  원본 산출물과 나란히 보존.

## 8. 후속 정정 — "3/6 통과" 중 1개는 트리비얼 통과였다 (2026-08-17, 같은 날 3차 세션, 메인스레드 직접 분석)

### 8.1 방법

§7.6이 남긴 열린 질문("조건2 통과/실패를 가르는 변수는 무엇인가")을 조사하기 위해
`arm_eval_report_sideblind.json`의 `criterion2_by_window[*].new_signal_always_long.no_gate.
reason_counts`(신규 스크립트 실행 없이 이미 있는 산출물만 재분석, 신규 학습/백테스트 없음)를
윈도우별로 열어 always_long-type 거래 중 `new_exit_signal`이 실제로 청산 사유가 된 비율(발동률)을
직접 확인했다.

### 8.2 발견 — 레인지②(03-10~05-05)의 "통과"는 신호가 한 번도 발동하지 않아서 나온 결과다

| 윈도우 | AL 거래수 | new_exit_signal 발동 | 발동률 | §7.6 조건2 |
|---|---:|---:|---:|:---:|
| VAL | 41 | 20 | 49% | X |
| OOS-Q1 | 32 | 12 | 38% | **O** |
| OOS-Q2 | 26 | 12 | 46% | X |
| 레인지①(05-12~07-07) | 15 | 3 | 20% | X |
| 레인지②(03-10~05-05) | 6 | **0** | **0%** | O(트리비얼) |
| 레인지③(02-09~04-06) | 24 | 11 | 46% | **O** |

레인지②는 §7.6 표에 이미 "동률(-14.90%→-14.90%)"로 기록돼 있었지만, 그 이유가 "신호가 개입했지만
결과가 우연히 같았다"가 아니라 **"신호가 그 6건 중 단 한 번도 임계값(0.5)을 넘지 않아 애초에
개입하지 않았다"**는 것을 이번에 확인했다. `mdd_improves_always_long = (new_mdd >= base_mdd)`가
등호를 포함하므로, 완전히 개입하지 않은 경우도 "개선"으로 카운트된다 — 이건 판정 로직의 버그는
아니지만(등호 포함이 관대한 쪽으로 설계된 것 자체는 정당), **"통과"라는 라벨이 "신호가 똑똑하게
작동했다"를 의미하지 않고 "신호가 아무것도 안 했다"를 의미할 수도 있다는 걸 헤드라인 숫자(3/6)만
봐서는 알 수 없다**.

### 8.3 재계산 — 실제로 개입한 창만 놓고 보면 2/5(40%), 헤드라인보다 나쁘다

레인지②를 트리비얼 통과로 제외하면, **실제로 신호가 개입한 5개 창 중 조건2를 통과한 건 2개
(OOS-Q1, 레인지③)뿐이고 3개(VAL, OOS-Q2, 레인지①)는 실패**했다 — 40% 성공률로, "3/6"이라는
헤드라인보다 나쁜 그림이다. §7.6이 이미 지적한 대로 하락추세/레인지 구분으로도 안 갈린다
(하락추세 3개 중 1승2패, 레인지 3개 중 진짜 개입 기준으로는 1승1패 — 레인지②를 빼면 사실상
2개뿐이라 통계적 의미 없음). **표본 크기(N=5개 창)로는 발동률·레짐·거래수 어느 것도 조건2
성패의 명확한 설명변수가 아니다** — 이 정도 표본에서 상관관계를 찾으려는 시도 자체가 노이즈를
쫓는 것과 구분이 안 된다는 게 정직한 결론이다.

### 8.4 정정된 판정

**성공조건 1(§7.5, 6/6 통과)의 결론은 바뀌지 않는다** — 이 절의 발견은 오직 성공조건 2의 해석에만
영향을 준다. **성공조건 2는 "3/6(50%)"이 아니라 "진짜 개입 기준 2/5(40%)"로 하향 수정한다** —
게다가 그 2/5조차 어떤 변수로 설명되지 않는 사실상 무작위에 가까운 패턴이다. **종합 판정을
"진짜 신호 확인(조건1) + 실전 이득으로의 전환은 미해결, 현재 데이터로는 설명 불가(조건2, N=5로
축소)"로 하향 정정한다.** 승격/배포 근거로는 여전히 부족하다는 §7.7의 결론은 유지되나, 그 부족의
정도가 원래 생각했던 것보다 크다.

다음 단계 후보(미실행): (a) 조건2 실패 창에서 신호가 정확히 어느 시점에 발동해 손실을 키웠는지
bar-level로 진단(예: VAL/OOS-Q2에서 거래수가 baseline 대비 크게 늘었다는 게 이미 관찰됨 — 조기청산
후 재진입을 반복하며 회전율 비용만 늘렸을 가능성), (b) 임계값 0.5가 아닌 다른 임계값에서 발동률이
달라지는지(사후 임계값 사냥이 아니라 사전에 임계값-발동률 곡선 자체를 새 진단으로 취급), (c) N을
윈도우가 아니라 개별 거래 단위로 늘리는 재설계(윈도우 6개로는 근본적으로 검정력 부족).

## 9. 레짐게이팅 하이브리드(원본 exit_head ↔ side-blind 신규신호) 테스트 (2026-08-17, 같은 날 4차 세션)

### 9.1 배경과 질문

`docs/experiments/eth_zig075_veto_ranging_misfire_fix_candidate_20260817.md`의 "추가" 절이 다른
세션에서 h48qual의 **기존 배포된** 레짐인지형 exit 가드(Odyssey3, L9-②,
`docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`)를 검증했다 — 탐지기
(`rolling(2016).mean(dual_momentum>0) > 0.8025793650793651`, 2025 Q1+Q2로만 캘리브레이션)가
ON이면 h48qual **원본**(재라벨 전) exit_head 가중치(threshold=0.95)로, OFF면 현재 라이브의
liveATR 재라벨 가중치(threshold=0.95, 가중치만 다름)로 전환하는 구조다. 그 검증 결과: real_g0
실거래 경로에서 **6개 창 전부 PnL이 마스크(NONE/V1/V3)와 무관하게 소수점까지 완전동일** — 즉 이
가드는 (기존 exit_head를 OFF-브랜치 대상으로 삼는 한) 인과적으로 관성적(causally inert)이었다.

이번 세션의 질문: 그 관성 결과는 *기존* exit_head를 OFF-브랜치로 삼았을 때의 것이다. 같은
탐지기로 게이팅하되 **OFF 브랜치를 §7의 side-blind 신규 exit 신호로 교체**하면 — 특히 side-blind
신호 단독이 실패했던 창(§8: VAL/OOS-Q2/레인지①)에서 — 여전히 관성적인지, 아니면 개선이 있는지를
검증한다.

### 9.2 설계 — 신규 자유변수 0개

- **ON 브랜치**(탐지기 활성, 지속상승장 감지 bar): h48qual **원본**(재라벨 전) exit_head 가중치,
  threshold=0.95 — 배포된 가드의 ON 브랜치와 완전히 동일, 무변경(`comp["guard_base_np"]`/
  `comp["guard_exit_runtime"]`/`comp["guard_pos_idx"]`/`comp["guard_exit_threshold"]`, 이미
  `research_eth_odyssey4_random_direction_risk_management_ablation_20260817.build_ablation_components`가
  h48qual 컴포넌트에 부착해 놓은 것을 그대로 재사용).
- **OFF 브랜치**(탐지기 비활성): §7의 side-blind 신규 exit 신호
  (`tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/new_exit_signal_bundle_sideblind.pkl`),
  threshold=0.5 — 기존엔 여기가 liveATR 재라벨 exit_head였던 자리를 대체.
- TP/SL은 항상 최우선(무변경), zig075 SHORT 진입베토는 완전히 무관여(이 게이팅은 h48qual의 보유-bar
  exit 분기에만 적용).
- 탐지기·임계값·양쪽 exit 모델·양쪽 결정 임계값(0.95/0.5) 전부 기존/이미 검증된 것을 그대로
  재사용했다 — 신규 학습, 신규 임계값 스윕 없음.

### 9.3 구현 및 검증

- 공유 replay 함수: `scripts/research_ilias_eth_adaptive_exit_signal_common_regime_gated_20260817.py`의
  `greedy_replay_new_exit_signal_regime_gated` — §7의
  `greedy_replay_new_exit_signal_sideblind`(자체가 `research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.greedy_replay_regime_aware_exit_guard`의 문서화된 복사본)을
  확장한 것으로, 보유-bar 분기 로직만 "탐지기 마스크를 먼저 확인 → ON이면 guard_*(원본), OFF면
  new_exit_model(side-blind)"으로 바꾼 게 유일한 변경이다. TP/SL 우선순위·zig075 베토·엔트리
  사이징·렛저 기록은 전부 바이트 동일.
- 평가: `scripts/research_ilias_eth_adaptive_exit_signal_arm_eval_regime_gated_20260817.py`.
- **G0 identity check(자체 검증 게이트, 산출물 신뢰 전 실행)**: 신규 replay 함수에서
  `new_exit_model`을 부착하지 않은 h48qual 컴포넌트(탐지기 마스크만 존재)로 always_long/
  always_short를 6개 창 전부 리플레이해, `research_eth_odyssey4_random_direction_risk_management_ablation_20260817.run_arm`(현재 배포된 가드 자체를 그대로 호출하는 기존 함수)의 결과와 대조 —
  **6개 창 전부 pnl/mdd/trades가 소수점 둘째자리까지 완전 일치**(예: VAL AL −14.46%/−36.56%/23건
  ref=hyb 완전동일). 신규 게이팅 함수가 기존 가드 로직을 정확히 재현함을 확인했다.
- 배경 탐지기 활성률(창별, 기존 다른 세션이 인용한 값과 일치 확인 — OOS-Q1 5.44%가 6개 중 최저):

  | 창 | 탐지기 활성률 |
  |---|---:|
  | VAL | 7.55% |
  | OOS-Q1 | 5.44% |
  | OOS-Q2 | 8.19% |
  | 레인지①(2025-05-12~07-07) | 15.98% |
  | 레인지②(2025-03-10~05-05) | 11.53% |
  | 레인지③(2026-02-09~04-06) | 8.66% |

- 성공조건1(6/6 통과, `arm_eval_report_sideblind.json`의 `criterion1_by_window`)은 **재사용, 재실행
  안 함** — 이 조건은 side-blind 분류기 자체의 성질(`simulate_private_barrier_trades`+
  `score_arm_trades`)만 측정하며 게이팅 여부와 무관하기 때문에(라이브 replay를 전혀 호출하지 않음),
  하이브리드에서도 수치가 동일할 수밖에 없다 — 재계산은 같은 계산의 중복일 뿐이다.
- Fresh-Forward 준수: `fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`,
  `saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false`.

### 9.4 결과 — 3-way 비교 (a=real_g0 원본 양쪽 / b=side-blind 단독, 무게이팅 / c=이번 하이브리드)

| 창 | 탐지기 활성률 | a AL pnl/mdd | b AL pnl/mdd | c AL pnl/mdd | a AS pnl | b AS pnl | c AS pnl | b 조건2 | c 조건2 |
|---|---:|---|---|---|---:|---:|---:|:---:|:---:|
| VAL | 7.55% | −14.46%/−36.56% | −31.80%/−41.65% | −31.80%/−41.65%(동일) | +58.48% | +37.03% | +38.30% | X | **X**(동일 실패) |
| OOS-Q1 | 5.44% | −41.61%/−50.83% | −37.39%/−47.27% | **−32.92%/−43.51%**(추가개선) | +82.13% | +75.86% | +75.67% | O | **O**(추가개선) |
| OOS-Q2 | 8.19% | −26.96%/−35.94% | −31.85%/−40.23% | −31.85%/−40.23%(동일) | +48.89% | +36.21% | +51.62%(개선) | X | **X**(동일 실패, AL 무변화) |
| 레인지①(05-12~07-07) | 15.98% | −10.70%/−31.44% | −10.70%/−31.44%(동일) | −10.70%/−31.44%(동일) | −6.18% | −17.93%(가드레일 위반, 190%악화) | **−8.27%**(가드레일 통과, 34%악화) | **X** | **O**(회복!) |
| 레인지②(03-10~05-05) | 11.53% | −3.47%/−14.90% | −3.47%/−14.90%(동일) | −3.47%/−14.90%(동일) | +4.09% | +2.77% | +2.77%(b와 완전동일) | O(트리비얼) | O(트리비얼, b와 동일) |
| 레인지③(02-09~04-06) | 8.66% | −15.49%/−25.45% | −10.20%/−20.82% | **−3.79%/−15.17%**(큰 폭 추가개선) | +4.58% | +5.46% | +5.35% | O | **O**(추가개선) |

트리비얼 통과 확인(`reason_counts`로 AL arm의 실제 발동 여부 직접 확인, §8과 동일 방법론):
**6개 창 중 레인지②만 트리비얼**(OFF/ON 브랜치 둘 다 0건 발동, b/c 완전 동일 재현) — 나머지
5개 창은 전부 실제 발동이 있었다(발동률 20.0~48.8%, 아래). VAL/OOS-Q2/레인지① 세 창은 AL arm의
with_gate pnl/mdd가 b와 c 사이에 변화가 없지만, 이는 트리비얼(무개입)이 아니라 **duration 게이트가
개입된 거래를 걸러낸 결과**다(§3.3이 이미 경고한 패턴과 동일 — "우연의 일치이지 신호가 무해했다는
뜻은 아니다"):

| 창 | AL 총거래(no_gate) | OFF브랜치(side-blind) 발동 | ON브랜치(원본) 발동 | 발동률 |
|---|---:|---:|---:|---:|
| VAL | 41 | 20 | 0 | 48.8% |
| OOS-Q1 | 31 | 11 | 0 | 35.5% |
| OOS-Q2 | 26 | 12 | 0 | 46.2% |
| 레인지① | 15 | 3 | 0 | 20.0% |
| 레인지② | 6 | **0** | **0** | **0%(트리비얼)** |
| 레인지③ | 23 | 10 | 0 | 43.5% |

(ON브랜치가 always_long arm에서 한 번도 발동하지 않은 것은 always_short arm에서는 다를 수 있음 —
레인지①의 AS 결과 변화가 그 증거다. 표는 always_long arm 기준.)

### 9.5 핵심 발견 — §8이 지적한 3개 실패 창 중 1개만 회복, 관성은 아니다

1. **VAL/OOS-Q2 — 회복 안 됨, side-blind 단독과 사실상 동일**: AL arm의 with_gate pnl/mdd가 b와
   c 사이에 소수점까지 동일하다(VAL −31.80%/−41.65% 둘 다, OOS-Q2 −31.85%/−40.23% 둘 다) — 이
   두 창은 탐지기 활성률이 6개 중 가장 낮은 축(7.55%/8.19%, 최저는 OOS-Q1 5.44%)에 속해, ON
   브랜치가 AL arm의 실제 청산 결과를 한 번도 바꾸지 못했다. 조건2는 두 창 모두 **여전히 실패**한다
   (mdd_improves_always_long=False). 다른 세션이 기존 exit_head 대상으로 발견한 "관성적" 패턴이
   이 두 창에서는 재현된다 — 단, OOS-Q2의 AS arm은 b(+36.21%)보다 c(+51.62%, real_g0의
   +48.89%보다도 높음)가 눈에 띄게 개선됐다 — 조건2 판정(AL mdd 기준)에는 영향 없지만 완전한
   무영향은 아니라는 뜻.
2. **레인지① — 유일하게 회복, 단 AS/가드레일 축에서만**: side-blind 단독은 이 창에서 AS arm의
   PnL이 가드레일(50% 상대악화 한도)을 크게 위반했다(−6.18%→−17.93%, 190% 악화) — 하이브리드는
   같은 창에서 −8.27%(34% 악화)로 가드레일 안에 들어와 **조건2가 X→O로 뒤집힌다**. 이 창은 6개 중
   탐지기 활성률이 가장 높다(15.98%) — ON 브랜치가 개입할 기회가 가장 많았던 창에서 회복이
   일어난 것은 방향상 일관적이다. 다만 **AL arm 자체는 b/c 사이에 전혀 변화가 없다**(둘 다
   −10.70%/−31.44%, duration 게이트가 3건의 실제 개입을 전부 걸러냄) — 즉 이 회복은 "잘못된
   방향(AL)의 손실을 하이브리드가 더 빨리 잘랐다"가 아니라, "옳은 방향(AS)이 side-blind 신호의
   과잉청산으로부터 부분적으로 보호됐다"는 뜻이다 — 원본 exit_head 가드의 원래 존재 이유(Q3에서
   지속추세 포지션의 조기 회전을 막는 것)와 정성적으로 같은 메커니즘이 다른 맥락(AS 보호)에서
   나타난 것으로 해석된다.
3. **OOS-Q1/레인지③ — 이미 통과했던 창이 추가로 더 좋아짐**: 두 창 모두 side-blind 단독으로도
   조건2를 통과했지만, 하이브리드는 그 개선폭을 더 키운다(OOS-Q1 mdd −47.27%→−43.51%, 레인지③
   mdd −20.82%→−15.17%, pnl −10.20%→−3.79%). 특히 OOS-Q1은 탐지기 활성률이 6개 중 가장 낮은데도
   (5.44%) 가장 뚜렷한 추가개선을 보였다 — 활성률만으로 회복/개선 정도를 예측할 수 없다는 뜻
   (N=6 창으로는 설명변수를 특정할 수 없음, 정직한 한계로 기록).
4. **레인지② — 완전 무개입, b/c 완전 동일**: OFF/ON 브랜치 둘 다 0건 발동(§8이 이미 확인한 것과
   동일한 트리비얼 통과가 하이브리드에서도 그대로 재현).

**종합**: 이 하이브리드는 다른 세션이 기존 exit_head 대상으로 발견한 것처럼 **완전히 관성적이지는
않다** — 6개 창 중 5개에서 실제 발동이 있었고(트리비얼은 레인지② 하나뿐), 4개 창(OOS-Q1/
레인지①/레인지③ + OOS-Q2의 AS arm)에서 side-blind 단독 대비 측정 가능한 변화가 있었다. 그러나
**§8이 지적한 3개 실패 창(VAL/OOS-Q2/레인지①) 중 완전히 회복된 것은 레인지① 하나뿐**이고, 그
회복도 AL(틀린 방향) arm 자체가 아니라 AS(옳은 방향) arm의 가드레일 위반을 막은 것이다 — VAL과
OOS-Q2는 여전히 실패로 남는다. 헤드라인 조건2 통과 수는 3/6(side-blind 단독)에서 4/6(하이브리드)로
늘었지만, 트리비얼(레인지②)을 제외한 "진짜 개입" 기준으로는 §8의 2/5(40%)에서 **3/5(60%)**로
개선됐다 — 통계적으로 얇은 표본(N=6 창, 진짜 개입 5개)이지만 정직하게 "무의미한 관성은 아니고
방향상 개선"으로 판정한다.

### 9.6 정직한 판정

**성공조건1**: 재실행하지 않고 §7.5의 결과(6/6 통과)를 그대로 인용 — 이 조건은 게이팅과 무관한
분류기 고유 성질이므로 하이브리드에서도 동일하다.

**성공조건2**: 4/6 창 통과(헤드라인), 트리비얼 제외 시 3/5(60%) — side-blind 단독의 2/5(40%,
§8 정정판)보다 개선됐다. **관성 가설은 기각된다**(기존 exit_head를 OFF-브랜치로 뒀을 때와 달리,
이번엔 OFF-브랜치 교체가 실제로 5/6 창에서 발동하고 그중 다수에서 측정 가능한 변화를 냈다). 그러나
**"레짐게이팅이 side-blind 신호의 레짐의존적 실패를 대체로 해결한다"고 주장할 근거는 아니다** —
§8이 지적한 3개 실패 창 중 2개(VAL, OOS-Q2)는 여전히 실패로 남고, 유일한 회복(레인지①)도 AL arm
자체가 아니라 AS arm의 부수적 보호 효과다.

**최종 판정: 부분 개선(Partial improvement), 관성 아님, 그러나 목표했던 "실패 창 회복"은 1/3만
성공.** 게이팅 자체가 "도움이 안 된다"는 결론은 데이터와 맞지 않는다(3/5 진짜개입 창에서 개선,
헤드라인 통과 수 증가) — 그러나 "게이팅이 side-blind 신호의 약점(하락추세 VAL/OOS-Q2)을
고쳐준다"는 주장도 데이터와 맞지 않는다(그 두 창은 정확히 동일하게 실패).

**증거 강도**: 단일 config(탐지기 1개, 임계값 2개 모두 기존 고정값, 새 하이퍼파라미터 스윕 없음)
결과다([[feedback_dl_needs_optimization_before_failure_verdict]] 원칙 재적용). N=6 창(진짜 개입
5개)으로는 "어느 조건에서 하이브리드가 회복시키는가"(탐지기 활성률만으로는 설명 안 됨, §9.5 항목3)를
설명할 통계적 검정력이 없다 — 이 축을 더 파려면 개별 거래 단위로 표본을 늘리는 재설계가 필요하다
(§8이 이미 제안한 다음 단계 (c)와 동일한 한계).

### 9.7 이 절의 산출물

- `scripts/research_ilias_eth_adaptive_exit_signal_common_regime_gated_20260817.py` —
  `greedy_replay_new_exit_signal_regime_gated`(§7의 side-blind replay 문서화된 복사본, 보유-bar
  분기만 탐지기 마스크 우선 확인으로 변경), `simulate_private_barrier_trades`/
  `score_new_exit_signal`/`POS_VALUE_NAMES`/`FEATURE_COLUMNS`는 §7 모듈에서 무수정 재수입.
- `scripts/research_ilias_eth_adaptive_exit_signal_arm_eval_regime_gated_20260817.py` — G0 identity
  check + 배경 탐지기 활성률 + 성공조건1(재사용)/조건2(재실행) + 3-way 비교 + 트리비얼 판정.
- `tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_regime_gated_20260817/arm_eval_report_regime_gated.json` —
  전체 산출물(G0 identity check 결과, 창별 활성률, criterion1/2, 3-way 비교표).
