# BTC RL Label Teacher Design - 2026-07-15

Status: `reviewed_research_design_draft_not_promotion_artifact`

이 문서는 현재 live-wired BTC v1을 강화학습 정책으로 교체하지 않고, Bellman value를
과거 경로에서 계산해 supervised label로 증류하는 후속 연구 설계다. Live BTC v1 bundle,
risk sidecar, runtime 설정은 변경하지 않는다.

## 1. 결정 요약

권장안은 **Cross-Fitted SMDP Advantage Label Teacher (CSALT)** 다.

```text
pre-holdout BTC 5m causal features
  -> dollar-activity event sampler
  -> exact fixed-execution action paths
  -> finite semi-Markov dynamic programming on teacher-train history
  -> cross-fitted quantile action-value teacher
  -> OOF CASH/LONG/SHORT + horizon + quality + weight label pack
  -> supervised HGB student
  -> fresh-forward validation
  -> frozen candidate waits for untouched future holdout
```

CSALT는 넓은 의미의 model-based offline RL label generator다. Neural actor, replay buffer,
entropy reward나 live exploration은 사용하지 않는다. RL에서 필요한 부분은 action duration과
다음 기회까지의 기회비용을 포함하는 semi-Markov Bellman target이다.

핵심 결정:

1. Teacher는 live action을 내지 않는다. OOF label artifact 생성 전용이다.
2. 모든 label row는 **flat, capacity available**인 canonical event state다. 따라서 timestamp당
   label이 하나로 결정된다.
3. CASH는 다음 eligible event로 이동하고, 거래 action은 고정 exit lifecycle과 cooldown이
   끝난 뒤 첫 eligible event로 이동한다.
4. Training tape의 Bellman target만 미래 경로를 사용한다. Label fold의 row는 frozen teacher가
   현재 state만 보고 채점하며 해당 fold realized path가 label을 바꾸지 않는다.
5. 첫 student는 HGB 하나뿐이다. TabM, dynamic risk, learned exit은 gate 통과 후에만 검토한다.
6. 첫 sampler는 앞선 label-family screen에서 validation과 OOS가 모두 양수였던 dollar-event다.

## 2. 문제와 기존 RL 실패의 차이

BTC v1은 direction=zigzag segment, quality=4시간 triple-barrier, execution=수일 hold라는 서로
다른 시간축을 사용한다. Omega4.7-RL은 이를 해결하려 했지만 5분마다 target position을
바꾸고 다음 한 bar return을 보상으로 사용해 lifecycle보다 noise와 turnover를 학습했다.
기존 기록상 OOS -88%~-99%였다.

CSALT가 묻는 질문은 다르다.

> 현재 event에서 CASH를 선택하거나, 특정 side/horizon 거래를 선택했을 때 그 거래의 순수익과
> 포지션 점유 때문에 놓치는 다음 기회까지 합친 14일 action advantage는 얼마인가?

각 event의 독립 forward return만 보면 reward regression이다. CSALT는 action별로 다음 decision
event가 달라지는 SMDP Bellman value를 사용한다. 반대로 portfolio drawdown이나 과거 정책처럼
student가 재현할 수 없는 path state는 첫 버전에서 제외한다.

## 3. 검토한 접근

### A. Immediate reward label

CSALT와 동일한 5/7개 action template, event, execution, q10 ensemble, gate, student를 사용하되
target을 각 action의 exact lifecycle net return으로 두고 continuation `V(next)=0`만 제거한다.

- Effort: S
- 장점: CSALT와의 차이가 skipped-opportunity Bellman continuation 하나뿐인 강한 negative
  control이다.
- 단점: capacity와 skipped opportunity가 없어 앞선 reward-shaping 실패와 본질적으로 같다.
- 결정: 반드시 실행하되 baseline으로만 사용한다.

### B. Direct sequential actor

RL actor가 live에서 entry/exit/position을 직접 결정한다.

- Effort: XL
- 장점: 이론상 lifecycle 전체를 한 목적에 맞출 수 있다.
- 단점: Omega4.7 실패를 반복할 위험, reward hacking, live 설명 불가능성, artifact 검증 비용이
  너무 크다.
- 결정: 제외한다.

### C. Cross-fitted SMDP advantage teacher

Training history에서 exact action transition graph와 finite-horizon Bellman target을 만들고,
quantile HGB teacher가 보지 못한 fold state의 action value를 예측한다.

- Effort: M
- 장점: 경로 의존 기회비용을 사용하면서도 finite graph, unique timestamp label, OOF provenance가
  명확하다.
- 단점: 한 번 관측된 시장 경로의 empirical target이므로 진짜 조건부 미래분포를 식별한다고
  주장할 수 없다.
- 결정: 권장안이다.

## 4. 데이터와 시간 계약

### 4.1 Holdout

- candidate, feature, label, threshold, execution config freeze deadline은
  `2026-11-30 23:59:59 UTC`다.
- deadline을 놓치면 이 holdout schedule은 무효다. 자동으로 window를 미루지 않고 새 미래 날짜를
  별도 design version에서 다시 사전 등록한다.
- 2026년 12월은 1개월 embargo이며 candidate 선택이나 재학습에 사용하지 않는다.
- `HOLDOUT_START = 2027-01-01 00:00:00 UTC`
- 고정 holdout window는 `2027-01-01..2027-09-30 23:59:59 UTC`다.
- `2027-10-01`에 전체 window를 정확히 한 번 읽고 평가한다. 그 전에는 row, event count, PnL을
  포함한 어떤 중간 집계도 보지 않는다.
- 그 한 번의 평가에서 closed trades < 50, LONG < 15 또는 SHORT < 15이면 결과는
  `inconclusive_fail`이다. 같은 window를 연장하거나 다시 읽지 않는다.
- Stage 0에서는 pre-holdout event/active rate만으로 예상 trade count를 기록한다. 이 추정치는
  holdout window나 평가일을 바꾸는 근거로 사용하지 않는다.

### 4.2 Development timeline

최대 경제적 holding 14일과 settlement 2 bar를 포함하므로 outcome purge는 15일이다.

| teacher fold | DP target decision range | outcome cutoff | label range |
|---|---|---|---|
| T1 | 2024-01-01..2024-03-16 | 2024-03-31 | 2024-04-16..2024-06-30 |
| T2 | 2024-01-01..2024-06-15 | 2024-06-30 | 2024-07-16..2024-09-30 |
| T3 | 2024-01-01..2024-09-15 | 2024-09-30 | 2024-10-16..2024-12-31 |
| T4 | 2024-01-01..2024-12-16 | 2024-12-31 | 2025-01-16..2025-03-31 |
| T5 | 2024-01-01..2025-03-16 | 2025-03-31 | 2025-04-16..2025-06-30 |
| T6 | 2024-01-01..2025-06-15 | 2025-06-30 | 2025-07-16..2025-08-31 |

Student model/hyperparameter/threshold 선택은 T1-T6 안의 outer expanding folds에서만 한다.

| student outer fold | train label folds | fresh-forward test fold |
|---|---|---|
| S1 | T1,T2 | T3 |
| S2 | T1..T3 | T4 |
| S3 | T1..T4 | T5 |
| S4 | T1..T5 | T6 |

각 S-fold의 test는 student 학습에 들어가지 않는다. S1-S4 aggregate로 student depth,
`quality_threshold`, `min_edge`를 한 번 선택한다.

최종 research windows:

- final student fit: T1-T6 OOF label rows
- seen-window Q4 research checkpoint: `2025-09-16..2025-12-31`
- diagnostic-only OOS: `2026-01-01..2026-03-31`

프로젝트 기본 validation 시작일 `2025-09-01`에서 15일 늦춘 이유는 14일 Bellman outcome
purge다. 이 경계 변경은 모든 report에 기록한다. Q1 2026은 이미 반복 관찰됐으므로 promotion
근거가 아니다.

### 4.3 Upstream feature provenance

- BTCUSDT 5분봉 OHLCV, funding, OI와 stationary BTC-native causal features만 사용한다.
- raw OHLC price level과 의미가 뒤집힌 cross-asset alias는 새 contract에서 제외한다.
- learned regime probability가 필요하면 각 teacher/student fold training cutoff 안에서 별도로
  fit한 OOF regime model만 허용한다.
- scaler, rolling expectation, activity threshold, regime bundle마다 `fit_end`와 SHA-256을
  fold manifest에 기록한다.
- 어떤 upstream fit_end도 해당 teacher label range 또는 student test range에 닿으면 실패한다.

## 5. Dollar-event sampler

Event는 5분봉 `close * volume` 누적 dollar activity로 정의한다.

1. 각 T-fold threshold는 DP target history에서만 계산한 hourly-equivalent median dollar
   activity로 고정한다.
2. DP target interval과 label interval은 accumulator를 0에서 시작한다.
3. 첫 threshold crossing은 boundary warmup event로 기록하고 label/decision에서 버린다.
4. 이후 crossing 완료 bar가 `timestamp`, 다음 5분봉 open이 `entry_available_timestamp`다.
5. 매 completed bar에서 `accumulator += close * volume`을 한 번 수행한다. Threshold 이상이면
   그 close에 event를 정확히 하나 emit하고 accumulator를 0으로 reset한다. Overshoot와 한 bar의
   추가 multiple crossing은 이 Phase 1에서 버린다.
6. Position/cooldown 중에도 같은 방식으로 crossing과 reset을 계속한다. 그 event는 queue하지 않고
   non-decision event로 버린다. `eligible event`는 flat이고 cooldown이 끝난 뒤 처음 새로 emit된
   event다.
7. Threshold, interval-start accumulator=0, overshoot discard 규칙을 fold manifest에 기록한다.
8. Final runtime candidate는 development 전체에서 freeze한 threshold를 쓰고 live accumulator를
   연속 유지한다. 재시작 시 persisted accumulator가 없으면 fail closed한다.

DC event는 ablation으로만 남긴다. 이전 screen에서 validation -17.42%, OOS +30.28%로
불안정했기 때문이다.

## 6. Phase-1 execution contract

Execution contract는 oracle이나 teacher 결과를 보기 전에 다음 값으로 freeze한다.

### 6.1 Position sizing

```text
margin_fraction = 0.30
leverage = 2.0
notional = margin_fraction * leverage = 0.60
side = +1 for LONG, -1 for SHORT
directional_price_move = side * (exit_fill / entry_fill - 1)
gross_account_return = directional_price_move * notional
```

TP/SL/trailing line은 price move다. Notional을 계산한 뒤 leverage를 다시 곱하지 않는다.
`notional`과 이후 fee/funding/PnL은 모두 trade 직전 account equity 대비 fraction이다. Phase 1은
cross-margin account simulation이며 trade 중 position quantity와 starting equity denominator를
고정하고 중간 compounding은 하지 않는다.

### 6.2 Entry and costs

- signal event 다음 5분봉 open에 market 진입
- `fee_rate = 0.0005`
- `slippage_rate = 0.0002`
- LONG entry fill = `next_open * (1 + slippage_rate)`
- SHORT entry fill = `next_open * (1 - slippage_rate)`
- entry와 exit 모두 taker fee를 적용한다.
- cost stress는 모든 fee/slippage를 각각 1.5x, 2.0x로 다시 계산
- recorded funding을 8시간 settlement timestamp마다
  `settlement_notional = notional * settlement_close / entry_fill`,
  `funding_cashflow = -side * funding_rate * settlement_notional`로 반영
- funding row가 누락되면 해당 action path를 보정하지 않고 artifact 생성을 실패시킴

Lifecycle reward는 다음 식 하나만 사용한다.

```text
exit_notional = notional * exit_fill / entry_fill
entry_fee = stressed_fee_rate * notional
exit_fee = stressed_fee_rate * exit_notional
funding_return = sum(funding_cashflow over lifecycle)
account_return = gross_account_return - entry_fee - exit_fee + funding_return
lifecycle_log_return = log1p(account_return)
```

Slippage는 stressed slippage rate로 entry/exit fill에 먼저 반영하고 fee notional은 그 fill로 계산한다.
`account_return <= -1` 또는 어느 중간값이 non-finite면 보정/cap하지 않고 fold artifact 생성을
실패시킨다.

### 6.3 ATR and exit

- entry ATR: completed 5분봉 true range, rolling 192, `min_periods=48`, `shift(1)`
- entry 후 ATR은 고정
- `entry_atr_pct = entry_atr / entry_fill`
- hard stop: `2.5 * entry_atr_pct` adverse price move; LONG stop price =
  `entry_fill * (1 - 2.5 * entry_atr_pct)`, SHORT은 `1 +`를 사용
- `favorable_close_move = side * (bar.close / entry_fill - 1)`
- trailing arm: `favorable_close_move >= 3.333333 * entry_atr_pct`
- armed peak: 지금까지의 최대 `favorable_close_move`
- trailing trigger: `armed_peak - favorable_close_move >= 8.333333 * entry_atr_pct`
- action horizon: 24h, 72h 또는 168h
- cooldown: exit 후 36개 5분봉
- exit 검사는 entry가 체결된 5분봉이 완전히 끝난 뒤부터 completed bar마다 한 번 수행한다.
- hard stop trigger: LONG은 `bar.low <= stop_price`, SHORT은 `bar.high >= stop_price`
- trailing arm/peak/giveback은 completed close만 사용한다. 먼저 기존 armed peak 대비 giveback을
  검사한 뒤, exit하지 않은 경우 current close로 arm/peak를 갱신한다.
- time trigger: entry fill timestamp부터 해당 24h/72h/168h가 지난 첫 completed bar
- 같은 bar에 여러 trigger가 있으면 stop, trailing, time 순으로 exit reason을 정한다.
- trigger bar 다음 bar open에 market exit
- LONG exit fill = `next_open * (1 - slippage_rate)`
- SHORT exit fill = `next_open * (1 + slippage_rate)`
- Phase 1에는 maker/limit fill을 사용하지 않는다.
- DP target/outcome interval의 5분봉이 하나라도 누락되어 next-bar fill을 확정할 수 없으면 해당
  fold artifact 생성을 실패시킨다.

위 수식과 trigger priority를 구현한 하나의 공유 execution function만 teacher와 fresh-forward
replay에서 사용한다. 두 경로가 별도 구현을 가지면 실패다.

### 6.4 Action set

| id | action | max hold |
|---:|---|---:|
| 0 | `CASH` | 다음 eligible event까지 |
| 1 | `LONG_H24` | 24h |
| 2 | `SHORT_H24` | 24h |
| 3 | `LONG_H72` | 72h |
| 4 | `SHORT_H72` | 72h |
| 5 | `LONG_H168` | 168h |
| 6 | `SHORT_H168` | 168h |

Stage 0 coverage gate가 action당 유효 training transition 300개 미만을 보이면 H168을 제거한
5-action contract로 **학습 전에** 축소한다. 성능을 본 뒤 action을 추가/제거하지 않는다.

## 7. Finite SMDP target

### 7.1 Canonical state

각 decision state는 다음만 포함한다.

```text
state = {
  event timestamp까지 확정된 stationary market features,
  event duration/activity features,
  position = 0,
  capacity_available = 1
}
```

Drawdown, prior action, bars since exit은 Phase 1 state에 없다. 따라서 같은 timestamp는 항상 같은
state/label을 가진다. 이 단순화 때문에 CSALT label은 실제 policy의 완전한 path-dependent
optimal action이 아니라 **canonical-flat SMDP advantage label**이다.

### 7.2 Transition

- CASH: reward=0, next state=다음 eligible event
- active action: reward=비용/funding 포함 lifecycle log-equity return,
  next state=exit+cooldown 뒤 첫 eligible event
- action duration `delta(a)`는 current event부터 next state까지 5분봉 수
- next state가 없거나 episode boundary 뒤면 terminal

### 7.3 Episode and Bellman equation

경제적 holding horizon은 최대 14일이고, planning boundary는 next-bar entry와 time-exit fill을
확정하기 위한 2개 settlement bar를 더한다. Discount는 없다(`gamma=1`)며, 이 finite episode가
far-future regime 지배를 막는다. `B = 4034`는 14일의 5분봉 4032개와 settlement bar 2개이고,
`h`는 남은 5분봉 budget이다. Outcome purge는 여전히 보수적으로 15일을 쓴다.

```text
Q*(s_t, a, h) = lifecycle_log_return(s_t, a)
                 + V*(next_event(s_t, a), h - delta(s_t, a))
V*(s_t, h)    = max_{a in A(h)} Q*(s_t, a, h)
A(h)          = {CASH} union {a: contract_max_duration(a) <= h}
V*(terminal, h) = 0
V*(s_t, h <= 0) = 0
```

각 teacher target은 `Q*(s_t, a, B)`다. Successor도 최초 start의 남은 `h`를 이어받으므로 매
transition마다 horizon이 14일로 다시 늘어나지 않는다. CASH의 next event가 `h` 밖이면
`delta=h`, reward 0, terminal value 0인 truncated transition으로 정의한다. Active action의
가능 여부는 realized stop/trailing 시점이 아니라 entry settlement 1 bar + 계약상 최대 hold bar +
exit settlement 1 bar인 `contract_max_duration`만으로 결정한다. 따라서 “미래에 일찍 stop될 것”을
보고 terminal 근처 action을 허용하지 않는다. 허용된 active action은 contract상 `h` 안에서 반드시
exit fill까지 끝난다. Data gap/funding 누락은 action availability가 아니라 fold-level data-contract
failure다. 모든 decision timestamp는 outcome cutoff보다 최소 14일과 settlement 2 bar 앞서야 한다.
Fold label row의 future path는 이 계산에 절대 들어가지 않는다.

### 7.4 What is and is not identified

Training tape의 DP는 그 시점에 실제 관측된 한 시장 경로에서 모든 discrete execution action을
재생한 empirical target이다. 이것은 동일 state에서 가능한 모든 미래 시장분포를 식별하지 않는다.
Teacher quantile과 seed dispersion은 cross-sectional conditional estimate다. 이 설계에서는 별도
확률 calibration을 하지 않으므로 calibrated probability나 true tail risk라고 부르지 않고 모두
`score`로 저장한다.

## 8. Cross-fitted teacher

### 8.1 Model

Neural FQI와 conservative-Q penalty를 사용하지 않는다. 전체 discrete action outcome을 simulator가
제공하므로 logged-behavior support 문제를 만들 필요가 없다.

Action별로 다음 HGB teacher를 학습한다.

- Bellman target quantile regressors: q10, q50, q90
- lifecycle positive-return binary classifier의 raw ensemble score
- seeds: 5개 bootstrap ensemble
- bootstrap unit: teacher fit range 안의 연속 24시간 event block
- input: canonical state features only

Teacher output `q10`은 empirical lower quantile estimate다. Seed dispersion은 ensemble instability다.
Positive-return output은 calibration되지 않은 ranking score이며 probability로 해석하지 않는다.

### 8.2 Genuine OOF rule

각 T-fold에서:

1. DP target range만으로 event threshold/upstream features/DP table을 만든다.
2. DP table만으로 teacher ensemble을 fit한다.
3. 15일 purge 후 label range current state를 frozen teacher가 채점한다.
4. label row의 realized future path는 label, q-value, vote, uncertainty, weight에 사용하지 않는다.
5. label range realized path는 별도 score-reliability diagnostic에만 저장한다.
6. 해당 diagnostic은 같은 fold teacher나 label threshold를 다시 fit하지 않는다.

`teacher_fit_end < label_start - 15 days`와 모든 upstream `fit_end`를 코드에서 assert한다.

## 9. RL-generated label contract

### 9.1 Fixed formulas

각 action `a`에 대해:

```text
q10_lcb(a) = median_seed(q10(a)) - std_seed(q10(a))
best        = argmax_a q10_lcb(a)
edge        = q10_lcb(best) - q10_lcb(CASH)
vote_ratio  = count_seed(argmax q10 == best) / 5
uncertainty = std_seed(q10(best))
```

Cost-stress teacher는 1.5x와 2.0x cost로 별도 DP target/teacher를 training range에서만 fit한다.
`cost_stress_q10`은 fold label realized outcome이 아니라 그 frozen teacher prediction이다.

고정 label gate:

```text
min_edge = 0.0010 log-equity
min_vote_ratio = 0.80
epsilon = 0.0001

active iff:
  best != CASH
  and edge > min_edge
  and vote_ratio >= min_vote_ratio
  and q10_lcb_1p5x(best) > q10_lcb_1p5x(CASH)
  and q10_lcb_2p0x(best) >= q10_lcb_2p0x(CASH)
```

S1-S4에서 `min_edge` negative-control grid `{0.0005, 0.0010, 0.0015}`를 비교할 수 있지만 final
값은 seen-window Q4 checkpoint를 재생하기 전에 freeze한다.

Soft target temperature는 teacher training DP q50의 action-wise IQR로만 계산한다.

```text
temperature = max(0.5 * IQR_train(q50), 0.0005)
active_soft  = softmax([q10_lcb(CASH), max_side_q10_lcb(LONG),
                        max_side_q10_lcb(SHORT)] / temperature)
soft_action  = active_soft   if active
               [1, 0, 0]     if CASH
active_margin = q10_lcb(best_active) - q10_lcb(CASH)
cash_margin   = q10_lcb(CASH) - max_active(q10_lcb)
label_margin  = active_margin                         if active
                max(cash_margin, 0.5 * min_edge)       if CASH
raw_weight    = label_margin / max(uncertainty, epsilon)
sample_weight = clip(raw_weight, 0.25, 10.0)
```

따라서 명확한 CASH row도 direction 학습에 참여하고, 불확실한 CASH row도 최소 weight 0.25를
갖는다. Gate에서 탈락한 row의 soft target도 CASH로 고정해 student가 teacher의 uncertainty/cost
gate를 우회하지 못하게 한다. Data-fitted percentile cap은 사용하지 않는다.

### 9.2 Targets

- `rl_action`: CASH=0, LONG=1, SHORT=2
- `rl_horizon_class`: none=0, H24=1, H72=2, H168=3
- `rl_quality_score`: selected **active action lifecycle** net return > 0 raw teacher score
- CASH row는 quality/horizon loss mask=0
- score는 calibration된 확률이 아니며 long-run Bellman Q도 아니다.

### 9.3 Label artifact

| column | meaning |
|---|---|
| `timestamp` | event confirmation close time |
| `entry_available_timestamp` | next 5m open |
| `rl_action` | CASH/LONG/SHORT hard label |
| `rl_horizon_class` | none/H24/H72/H168 |
| `rl_soft_cash/long/short` | action soft target |
| `rl_quality_score` | active lifecycle positive-return raw score |
| `rl_q10_edge` | empirical conservative edge over CASH |
| `rl_seed_vote_ratio` | teacher agreement |
| `rl_uncertainty` | seed instability |
| `rl_sample_weight` | bounded edge/uncertainty weight |
| `teacher_fold` | T1..T6 OOF provenance |
| `teacher_bundle_sha256` | exact teacher identity |

Realized return, MFE/MAE, exit timestamp와 future path는 label pack에 넣지 않는다. 별도
`teacher_diagnostics/` 경로에 두고 student loader allowlist 밖으로 분리한다.

## 10. Student and outer selection

### 10.1 Head contract

첫 student는 다음 독립 HGB regressor head의 5-seed ensemble이다. Soft label을 classifier의 hard
class API에 넣지 않는다.

| head | model/loss | training rows | target/output |
|---|---|---|---|
| direction | HGB squared-error regressor 3개 | 모든 row | `rl_soft_cash/long/short`; seed median을 [0,1] clip 후 합 1로 normalize |
| quality-long | HGB squared-error regressor | LONG active row | `rl_quality_score`; seed median을 [0,1] clip |
| quality-short | HGB squared-error regressor | SHORT active row | `rl_quality_score`; seed median을 [0,1] clip |
| horizon-long | HGB squared-error regressor 3개 | LONG active row | H24/H72/H168 one-hot; seed median을 [0,1] clip 후 normalize |
| horizon-short | HGB squared-error regressor 3개 | SHORT active row | H24/H72/H168 one-hot; seed median을 [0,1] clip 후 normalize |

- 모든 head는 `rl_sample_weight`를 사용한다.
- 각 seed는 해당 outer-fold training row를 24시간 event block 단위로 bootstrap한다.
- live input은 canonical causal state만 허용한다. Teacher Q/reward/label provenance는 feature로
  금지한다.
- direction normalize 전 합이 0이면 CASH, horizon normalize 전 합이 0이면 H24로 fail closed한다.

### 10.2 Runtime decoding

Frozen student의 bar-by-bar decision은 다음 순서를 바꾸지 않는다.

1. flat이고 cooldown이 끝났으며 새 dollar event가 확정된 bar에서만 평가한다.
2. direction head 세 값을 normalize하고 argmax를 구한다.
3. top-2 direction score 차이가 `<= 1e-6`이거나 argmax가 CASH면 CASH다.
4. LONG/SHORT argmax에 대응하는 side-specific `quality_score`를 선택한다.
5. score가 `contract.quality_threshold`보다 작으면 CASH다.
6. 해당 side horizon argmax를 선택한다. 동률이면 더 짧은 horizon을 선택한다.
7. 선택된 action은 고정 sizing과 Section 6의 next-bar market execution을 사용한다.

어느 seed든 non-finite output을 내거나 feature contract/hash가 다르면 해당 seed를 버리지 않고
그 decision 전체를 fail closed한다.

### 10.3 Outer selection

S1-S4에서만 다음을 선택한다.

- HGB max depth `{3,4}`
- minimum leaf `{40,80}`
- `min_edge` `{0.0005,0.0010,0.0015}`
- runtime `quality_threshold` `{0.50,0.55,0.60,0.65,0.70}`

Selection key는 aggregate Calmar이며 다음 constraint를 먼저 만족해야 한다.

- 모든 S-fold PnL > 0
- 각 S-fold MDD >= -12%
- cost 1.5x aggregate PnL > 0
- total trades >= 80, side별 >= 20

동률이면 더 얕은 depth, 더 큰 leaf, 더 높은 CASH rate 순으로 선택한다. Seen-window Q4
checkpoint/Q1 diagnostic을 본 뒤 다시 선택하지 않는다.

## 11. Leakage and fail-fast contract

필수 report flags:

```json
{
  "fresh_forward_bar_by_bar": true,
  "trade_ledgers_used_as_input": false,
  "saved_parent_exit_timestamps_used": false,
  "future_rows_used_for_entry": false,
  "teacher_predictions_are_purged_oof": true,
  "teacher_outputs_used_as_student_features": false,
  "label_fold_realized_outcomes_used_to_change_labels": false,
  "holdout_data_read_before_scheduled_evaluation": false,
  "holdout_evaluation_count": 0
}
```

Stage 6 최종 report에서만 `holdout_evaluation_count=1`로 바뀐다.

Fail-fast:

- same checkpoint가 fit한 row를 다시 label하면 실패
- contract-max duration 또는 DP bootstrap가 outcome cutoff를 넘는 decision row가 생성되면 fold 실패
- label row의 future outcome으로 active gate/weight를 바꾸면 실패
- event timestamp와 entry availability가 같은 5분봉이면 실패
- threshold/scaler/regime/teacher hash 또는 fit_end가 manifest와 다르면 실패
- student feature에 `rl_q`, `reward`, `future`, `exit`, `pnl`, `label` token이 있으면 실패
- funding row가 action lifecycle 중 누락되면 실패
- seed 5개, vote, uncertainty, teacher hash가 없으면 실패

## 12. Stage gates

### Stage 0: Feasibility and parity

필수 표:

- fold별 dollar events
- action/horizon별 valid DP targets
- teacher input feature 수
- effective sample size와 class/side balance
- estimated full runtime, peak RAM/GPU memory
- pre-holdout event/active rate 기반 fixed holdout end의 예상 trade count

Abort:

- any action valid targets < 300이면 H168을 제거하고 contract를 학습 전에 freeze
- 5-action에서도 any action < 300이면 RL teacher 중단
- deterministic fixture에서 reward/equity/exit가 수작업 계산과 다르면 중단

### Stage 1: Oracle ceiling

같은 execution contract로 계산:

1. 동일 action set의 `V(next)=0` immediate-lifecycle oracle
2. 14-day single-capacity SMDP DP oracle
3. causal dollar-event supervised baseline

SMDP oracle aggregate Calmar가 baseline보다 25% 이상 높지 않으면 teacher를 만들지 않는다.
Oracle은 ceiling 진단일 뿐 performance claim이나 label fold input이 아니다.

### Stage 2: Teacher negative controls

- N0 immediate-lifecycle q10 ensemble: CSALT와 동일한 action/gate/student, `V(next)=0`만 다름
- N1 DP q50 HGB
- N2 DP q10 ensemble without gate
- N3 CSALT full gate

CSALT가 S1-S4 aggregate와 label stability에서 N0를 이기지 못하면 중단한다.

### Stage 3: Student fresh-forward

- exact next-bar feature availability와 fill
- single position
- 5분봉 stop/trailing/time-exit/funding
- cost 1.0x/1.5x/2.0x
- closed-equity MDD와 MTM MDD
- month/side/regime/confidence quintile
- label chart: 각 T-fold price+label+teacher edge+student signal

각 label-family/fold 실행 직후, 다음 4-panel PNG를 저장해 사용자 확인 checkpoint로 삼는다.

1. BTC close와 event marker, CASH/LONG/SHORT background
2. action별 q10 LCB와 선택 action edge
3. vote ratio, uncertainty, sample weight
4. student direction/quality/horizon과 실제 fresh-forward entry/exit

Event confirmation과 next-open entry를 서로 다른 수직선으로 표시한다. Chart는 이미 freeze된 label을
설명하는 진단물이며, 눈으로 본 뒤 같은 fold gate/threshold를 조정하는 입력으로 사용하지 않는다.

### Stage 4: Seen-window Q4 checkpoint

`2025-09-16..2025-12-31`은 앞선 label-family screen에서 dollar-event sampler 선택에 이미 사용된
기간이다. 따라서 미관측 validation이 아니며 performance/promotion 근거로 쓰지 않는다. 현
candidate를 freeze한 뒤 compatibility checkpoint로 한 번 재생한다.

Pass:

- PnL > 0
- MDD >= -12%, MTM MDD >= -15%
- cost 1.5x PnL > 0
- trades >= 20, side별 >= 5
- top confidence quintile mean return > bottom quintile

Fail 시 Q1 OOS를 열어 salvage/tuning하지 않는다. Candidate를 폐기한다. Pass도 미래 holdout
성과의 증거라고 주장하지 않는다.

### Stage 5: Diagnostic-only Q1

Validation pass candidate만 `2026-01-01..2026-03-31`을 재생한다. 이미 본 기간이므로 숫자는
research diagnostic이며 selection/promotion 근거가 아니다.

### Stage 6: Future holdout

`2027-01-01..2027-09-30 23:59:59 UTC`의 frozen artifact를 `2027-10-01`에 정확히 한 번
평가한다. 평가 전에는 event/trade count도 읽지 않는다. 비교 대상 BTC v1도 holdout 시작 전에
bundle hash와 당시 production execution/risk contract를 freeze하며, 같은 window에서 변경 없이
bar-by-bar 재생한다.

Holdout pass:

- closed trades >= 50, LONG >= 15, SHORT >= 15
- PnL > 0
- MDD >= -12%, MTM MDD >= -15%
- cost 1.5x PnL > 0
- candidate Calmar > concurrent-window BTC v1 Calmar
- 어느 한 side가 PnL의 80% 이상을 만들지 않음
- 어느 한 calendar month가 양의 총 PnL의 60% 이상을 만들지 않음

최소 trade/side count를 못 채우면 `inconclusive_fail`, 그 외 하나라도 못 채우면 `fail`이다. 어느
경우든 같은 holdout의 연장, 재평가, threshold/model/reward 변경 candidate promotion은 불가다.

## 13. Promotion and Omega artifact contract

Stage 0-5 산출물은 non-Omega research artifact이며 promotion 불가다. 현재 Omega integrity
audit은 risk component와 exact per-bar prediction artifacts를 요구하므로 student label/event
ledger만으로 통과했다고 주장하지 않는다.

Promotion candidate는 별도 wrapper를 추가해야 한다.

- deterministic fixed-risk component: margin 0.30, leverage 2.0, notional 0.60
- report에 `baseline_bundle`과 `risk_model.precomputed_prediction_dir/tag` 기록
- `contract.quality_threshold` 고정
- `qXXX = round(quality_threshold * 100)` zero-padded
- 정확한 `train_predictions_qXXX.csv`
- 정확한 `validation_predictions_qXXX.csv`
- 정확한 `oos_predictions_qXXX.csv`
- 세 prediction file은 event row가 아니라 해당 split의 **모든 5분봉**을 포함
- non-event bar는 explicit CASH/event_available=false
- exact threshold files를 risk component가 사용
- `scripts/audit_omega_artifact_integrity_20260630.py` exit 0 및
  `promotion_pass=true`

Audit이 현 deterministic-risk wrapper를 지원하지 않으면 audit을 우회하지 않는다. Compatible
promotion wrapper/policy를 먼저 명시적으로 설계하고 다시 리뷰한다.

## 14. Artifact layout

```text
tmp/causal_regen_20260516/btc_csalt_YYYYMMDD/
  environment_contract.json
  fold_manifest.json
  dp_targets/
    fold_T*_targets.parquet
  teacher/
    fold_T*_action_*_seed_*.joblib
  oof_rl_label_pack.parquet
  teacher_diagnostics/
    fold_T*_realized_paths.parquet
    score_reliability_summary.csv
  student_bundle.joblib
  train_predictions_qXXX.csv
  validation_predictions_qXXX.csv
  oos_predictions_qXXX.csv
  validation_ledger.csv
  oos_diagnostic_ledger.csv
  label_charts/
    N*_fold_T*.png
    confidence_reliability.png
    action_q10_edges.png
  report.json
  manifest.sha256.json
```

Prediction artifacts는 research phase에서도 per-bar 형식으로 미리 생성하되, promotion-grade라고
표시하지 않는다.

## 15. 구현 순서

1. Shared execution function과 deterministic parity fixtures
2. Stage 0 event/action coverage table
3. Canonical-flat 14-day DP oracle
4. N0 immediate reward baseline
5. T1 one-fold q10/q50/q90 HGB teacher smoke test
6. T1 label chart와 leakage audit
7. T1-T6 cross-fitted label pack
8. S1-S4 HGB student selection
9. Frozen seen-window Q4 checkpoint

첫 구현에서 neural actor, replay buffer, direct live RL, dynamic leverage, learned exit,
multi-asset RL, TabM student는 범위 밖이다.

## 16. 성공과 중단 기준 요약

계속 진행:

- DP oracle ceiling >= baseline Calmar +25%
- action당 target >= 300
- S1-S4 모든 fold PnL > 0
- cost 1.5x aggregate > 0
- confidence/return separation 존재
- seen-window Q4 checkpoint pass(연구 지속 조건일 뿐 성능 근거 아님)

즉시 중단:

- N0 immediate reward가 CSALT와 같거나 더 좋음
- seed에 따라 active label 30% 이상 뒤집힘
- validation은 음수이고 이미 본 Q1에서만 양수
- cost 제거 시에만 양수
- event/action coverage 부족
- fold-test future outcome이 label에 영향을 준 흔적 발견

## 17. 다음 작업

다음 작업은 RL network 구현이 아니다. **Stage 0 coverage/parity와 Stage 1 canonical-flat DP
oracle**을 먼저 만든다. 이 두 단계가 RL advantage의 존재와 구현 가능성을 입증하지 못하면
dollar-event supervised baseline을 유지하고 CSALT를 종료한다.

## 18. Review history

독립 1차 리뷰 점수는 4/10이었다. 다음 문제를 반영해 설계를 축소·수정했다.

- path-dependent state와 timestamp당 단일 label의 모순 제거
- CASH/active action/terminal transition과 finite Bellman target 명시
- exact fold/date/purge/outer selection 소유권 명시
- 실행 비용, funding, ATR, gap, exit precedence freeze
- label-fold realized outcome가 label을 변경하지 못하도록 분리
- neural FQI/conservative penalty 제거, empirical quantile HGB로 축소
- per-bar exact-threshold Omega artifact와 deterministic risk wrapper 요구 추가
- future holdout freeze deadline, embargo, 단일 평가일, 최소 거래 수, PnL/MDD/cost gate 사전 등록

2차 리뷰는 7/10이었고 runtime decoding, soft-target head, remaining-horizon Bellman state, CASH
weight, market-only fill을 추가했다. 3차 공식 리뷰는 9/10이었으며 causal/OOF/SMDP/holdout/scope는
모두 pass했다. 마지막 지적은 execution-to-account-return-to-log-reward 수식 누락 한 건이었다.
이를 Section 6.1-6.3의 directional move, ATR denominator, 양방향 fee notional, funding, `log1p`,
non-finite fail-fast 식으로 반영했다. 다음 검증은 추가 문서 리뷰가 아니라 Stage 0 deterministic
parity fixture다.
