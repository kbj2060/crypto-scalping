# Omega1.2 Parent Quality Gate and RL Entry/Exit Problem Report

작성일: 2026-06-18

## 요약

현재 Omega1.2 parent의 핵심 병목은 단순히 모델 종류가 약하다는 문제가 아니다. 3-head TabM의 direction head는 long/short 후보를 많이 만들지만, 그 후보 풀 자체가 손절 쪽으로 강하게 기울어져 있다. quality gate 0.8은 거칠고 보수적이지만, 현재까지의 실험에서는 이 gate가 가장 강한 보호 장치로 작동했다.

따라서 `entry / quality / exit` head 출력만으로 DSAC 강화학습 모델을 학습해 진입과 청산을 맡기는 접근은 연구 실험으로는 가능하지만, live 개선 후보로 바로 기대하기 어렵다. 먼저 이 문제를 "후보 품질", "라벨 불안정성", "검증-OOS 일반화", "RL reward 설계" 관점에서 분해해야 한다.

## 현재 Parent 구조

현재 parent는 Regime3 router 기반 3-head TabM 구조다.

```text
Regime3 router
-> bull / bear / chop expert 선택
-> True 3-head TabM
   1. Direction head: cash / long / short
   2. Quality head: action quality
   3. Exit head: lifecycle / exit signal
-> quality threshold
-> final parent action
```

운영 기준 parent-only baseline은 `cash_alpha43 aggressive` decision builder를 적용한 결과다.

```text
Current quality gate parent
Validation PnL: 100.54
OOS PnL: 72.76
Validation WR: 0.636
OOS WR: 0.722
Validation trades: 33
OOS trades: 18
```

## 핵심 증거

### 1. Quality gate 제거는 실패했다

quality gate를 제거하고 direction head의 raw action을 그대로 쓰면 성능이 급격히 무너진다.

```text
Raw direction, no quality gate
Validation PnL: 2.82
OOS PnL: -31.74
```

이는 direction head가 방향 후보를 많이 내더라도, 그 자체만으로는 실행 가능한 trade policy가 아니라는 뜻이다.

### 2. Raw direction 후보 풀은 손절 쪽으로 기울어져 있다

raw direction 후보를 시뮬레이션해 만든 후보 라벨은 아래와 같다.

```text
candidate labels: 20,071
TP rate: 31.35%
SL rate: 55.78%
net mean: 0.00043
MAE mean: -0.02216

exit reason counts:
  take_profit: 6,293
  stop_loss: 11,196
  forced_end: 2,582
```

즉 후보의 절반 이상이 stop loss로 끝난다. 이 상황에서는 단순 classifier나 RL agent가 "좋은 진입을 조금 더 잘 고르는 문제"가 아니라, 매우 나쁜 후보 풀에서 극소수의 안전한 trade만 골라야 하는 문제를 풀어야 한다.

### 3. 별도 veto/meta classifier는 current quality gate를 못 이겼다

기존 3-head 출력과 market/trace feature를 사용해 별도 veto 모델을 붙였지만 실패했다.

```text
Current quality gate parent
Validation PnL: 100.54
OOS PnL: 72.76

Best validation veto
Validation PnL: 27.86
OOS PnL: 4.66

Best OOS diagnostic veto
Validation PnL: 16.15
OOS PnL: 48.12
```

OOS diagnostic에서 일부 개선처럼 보이는 조합도 validation 기준으로는 current quality gate보다 훨씬 낮다. OOS 기준 선택은 금지해야 하므로 live 후보가 될 수 없다.

### 4. Quality gate 대체안도 실패했다

다음 대체안들을 모두 같은 기준으로 테스트했다.

```text
Quality-Scaled Notional
Combined EV Score
Adaptive Threshold
Win Probability Meta
Multi-target supervised gate
```

family별 validation 최고 결과는 아래와 같다.

```text
Current quality gate parent
Validation PnL: 100.54
OOS PnL: 72.76

Combined EV
Validation PnL: 43.36
OOS PnL: 2.94

Quality-Scaled Notional
Validation PnL: 29.38
OOS PnL: 21.68

Win Probability Meta
Validation PnL: 25.97
OOS PnL: -0.35

Adaptive Threshold
Validation PnL: 12.86
OOS PnL: -3.08
```

결론은 명확하다. quality threshold 0.8은 단순하고 거칠지만, 현재 feature/label/후보 구조에서는 가장 강한 방어막이다.

### 5. Multi-target supervised gate도 너무 보수적으로 무너졌다

P(TP), P(SL), net lower-bound, MAE lower-bound를 따로 학습한 multi-target gate도 실패했다.

```text
Selected multi-target gate
Validation PnL: 5.21
OOS PnL: 0.00
Validation trades: 1
OOS trades: 0
```

중요한 원인은 net lower-bound calibration이다.

```text
final net residual q80: 0.04535
```

후보별 실현 net 변동성이 커서 보수적 lower-bound를 적용하면 대부분의 후보가 음수로 깎인다. 즉 "보수적인 모델"은 거의 모든 trade를 막고, "덜 보수적인 모델"은 stop loss를 많이 허용한다.

### 6. Exit head 단독 또는 exit head 제거도 실패했다

Exit head만으로 청산하면 실패했다.

```text
Exit-head-only selected
Validation PnL: 0.46
OOS PnL: -6.05
OOS WR: 0.266
```

Exit head를 제거하고 Direction+Quality 2-head로 재학습해도 실패했다.

```text
2-head no-exit parent-only
Validation PnL: -0.44
OOS PnL: 8.68
OOS WR: 0.50
```

이는 exit head가 직접 청산 모델로는 부족하지만, 3-head 멀티태스크 학습에서 representation regularizer 역할을 하고 있을 가능성을 보여준다.

## 왜 DSAC가 바로 답이 되기 어려운가

### 1. 상태가 너무 빈약할 수 있다

`entry / quality / exit` head 출력만 상태로 쓰면 시장 context가 빠진다.

예를 들어 같은 quality score 0.70이라도 의미는 아래 조건에 따라 달라진다.

```text
Regime: bull / bear / chop
Regime confidence
ATR / volatility
recent return path
range expansion
time-of-day
funding / liquidity context
```

3-head 출력만 쓰는 DSAC는 이 context를 간접적으로만 볼 수 있다. 이미 supervised gate가 3-head + market feature를 함께 써도 current gate를 못 이긴 점을 고려하면, 3-head 출력만 쓰는 RL은 더 불리할 수 있다.

### 2. Reward shaping에 매우 민감하다

RL reward를 단순 PnL로 두면 overtrading과 drawdown을 쉽게 허용한다.

```text
bad reward:
  reward = realized_pnl

better reward:
  reward = realized_pnl
           - transaction_cost
           - stop_loss_penalty
           - drawdown_penalty
           - turnover_penalty
           - unsupported_state_penalty
```

하지만 penalty를 조금만 잘못 잡아도 validation에만 맞는 policy가 나온다. 지금까지의 실험에서도 OOS diagnostic만 좋아 보이고 validation selection에서는 실패하는 패턴이 반복됐다.

### 3. Offline RL은 선택 편향에 취약하다

현재 후보 데이터는 parent가 만든 direction/quality 분포에 의해 강하게 필터링되어 있다. offline RL은 데이터에 없는 행동의 가치를 과대평가하기 쉽다.

특히 아래 상황이 위험하다.

```text
training data에는 거의 없는 aggressive reversal
낮은 quality 구간의 long/short flip
고변동성 cash 구간의 무리한 entry
```

이 때문에 DSAC를 쓰더라도 CQL류의 conservative penalty나 action mask가 필요하다.

### 4. 현재 문제는 "정책 최적화" 이전의 "후보 품질" 문제다

raw direction 후보의 stop-loss rate가 높다.

```text
TP: 31.35%
SL: 55.78%
```

이 상태에서 RL이 해결해야 하는 문제는 "좋은 행동을 강화"하는 것이 아니라, "대부분의 행동을 하지 않는 정책"을 배우는 것이다. 그 결과는 current quality gate처럼 매우 보수적인 정책으로 수렴할 가능성이 높다.

## 그래도 RL을 한다면 필요한 설계

DSAC 실험을 한다면 live 후보가 아니라 research-only로 분리해야 한다.

### State

3-head 출력만 쓰는 것은 부족하다. 최소한 아래를 포함해야 한다.

```text
Direction:
  dir_p_cash
  dir_p_long
  dir_p_short
  dir_confidence
  dir_side_edge
  dir_trade_prob

Quality:
  quality_p_cash
  quality_p_long
  quality_p_short
  quality_for_action

Exit:
  exit probability / lifecycle signal

Regime:
  router_expert
  router_confidence
  router_margin
  router_is_bull / bear / chop

Market context:
  atr14_pct
  ret_1 / 3 / 6 / 12 / 24
  ret_vol
  range_mean
  ema gap
  time-of-day

Position context:
  position side
  hold bars
  unrealized PnL
  MFE / MAE
  distance to TP / SL
```

### Action

완전 자유 action은 위험하다. action mask가 필요하다.

```text
Flat state:
  0 = stay cash
  1 = enter long
  2 = enter short

In-position state:
  0 = hold
  1 = exit
```

Long position에서 즉시 short 전환, short position에서 즉시 long 전환은 연구 초기에는 막아야 한다. 반전은 close 후 다음 bar에서만 허용하는 편이 안전하다.

### Reward

추천 reward:

```text
reward =
  realized_net
  - cost_penalty
  - stop_loss_penalty
  - max_drawdown_penalty
  - turnover_penalty
  - unsupported_state_penalty
```

단순 PnL reward는 금지하는 것이 좋다.

### Split

OOS 선택을 막기 위해 아래 split을 고정해야 한다.

```text
Train:
  pre-2025-10 data

Validation:
  2025-10-01 ~ 2025-12-31

OOS:
  2026-01-01 ~ 2026-02-28
```

checkpoint, threshold, reward weight 선택은 validation에서만 해야 한다.

## 권장 실험 순서

### Step 1. Offline Policy Classifier Smoke Test

DSAC보다 먼저 supervised oracle policy classifier를 테스트한다.

```text
Input:
  3-head outputs + market context + position context

Label:
  flat state:
    enter long / enter short / stay cash

  in-position state:
    hold / exit

Selection:
  validation-only
```

이 smoke test가 current quality gate 근처에도 못 가면 RL은 진행하지 않는 것이 맞다.

### Step 2. Conservative Offline RL

Smoke test가 최소 기준을 넘을 때만 DSAC/CQL 계열을 시도한다.

필수 조건:

```text
action mask
conservative penalty
OOD support gate
turnover penalty
validation-only checkpoint selection
```

### Step 3. Limited Action Scope

처음부터 entry와 exit를 모두 RL에 맡기지 말고 scope를 제한한다.

추천 순서:

```text
1. exit-only RL
2. entry-veto RL
3. entry + exit lifecycle RL
```

현재까지 exit-only head는 실패했지만, position context를 포함한 exit-only policy는 별도 실험 가치가 있다.

## 현재 결론

`entry / quality / exit` head 출력만으로 DSAC를 학습해 진입과 청산을 맡기는 접근은 연구적으로 가능하지만, 현재 evidence 기준으로 live 개선 가능성은 낮다.

핵심 이유:

```text
1. raw direction 후보 풀의 stop-loss 비율이 높다.
2. quality gate를 제거한 모든 대체안이 current gate를 못 이겼다.
3. supervised gate도 current quality gate를 못 이겼다.
4. exit head 단독 청산도 실패했다.
5. RL은 reward shaping과 OOS 과최적화 위험이 더 크다.
```

따라서 바로 DSAC로 가기보다 아래 순서를 권장한다.

```text
1. 현재 quality gate parent는 유지
2. supervised oracle policy classifier로 smoke test
3. 통과할 때만 conservative offline RL 진행
4. RL을 하더라도 entry/exit 전체가 아니라 exit-only 또는 entry-veto부터 시작
```

운영 판단:

```text
DSAC 3-head policy는 live 후보가 아니라 research-only 후보.
현재 live 기준선은 quality gate parent + 검증된 sleeve/risk 구조를 유지하는 것이 맞다.
```
