# PPO 모델 명세서 (PPO Model Specification)

## 📋 목차
1. [개요](#개요)
2. [데이터 흐름](#데이터-흐름)
3. [모델 아키텍처](#모델-아키텍처)
4. [액션 체계](#액션-체계)
5. [보상 체계](#보상-체계)
6. [학습 프로세스](#학습-프로세스)
7. [하이퍼파라미터](#하이퍼파라미터)

---

## 1. 개요

이 PPO 모델은 암호화폐 스캘핑 거래를 위한 강화학습 에이전트입니다. xLSTM 기반 Actor-Critic 네트워크를 사용하여 시계열 패턴을 학습하고, Dense Reward 체계를 통해 매 스텝마다 학습 신호를 제공합니다.

### 핵심 특징
- **xLSTM 기반**: Multi-Layer sLSTM + Pre-LN Residual Connection
- **Dense Reward**: 매 스텝 평가금액 변화에 보상 부여
- **AI 판단 청산**: Action 0으로 스스로 청산 시점 결정
- **State Retention**: LSTM 상태를 에피소드 내에서 유지

---

## 2. 데이터 흐름

### 2.1 전체 파이프라인

```
[원본 데이터] 
    ↓
[Feature Engineering] → training_features.csv (29개 피처)
    ↓
[Strategy Pre-calculation] → cached_strategies.csv (12개 전략 점수)
    ↓
[Data Loading] → training_features.csv + cached_strategies.csv 병합
    ↓
[Scaler Training] → Train Set 80%만 사용 (Data Leakage 방지)
    ↓
[Episode Loop]
    ├─ Observation (obs_seq, obs_info) 생성
    ├─ Action 선택 (0, 1, 2)
    ├─ Trading Logic 실행
    ├─ Reward 계산 (Dense Reward)
    └─ PPO Update (GAE + Clipped Surrogate)
    ↓
[Model Save] → ppo_model_best.pth / ppo_model_last.pth
```

### 2.2 데이터 분할

- **Train Set**: 70% (앞부분)
- **Validation Set**: 15% (중간)
- **Test Set**: 15% (뒷부분)

**중요**: 스케일러는 Train Set 80%만 사용하여 학습합니다.

### 2.3 피처 구성

#### 시계열 피처 (29개)
```
1. log_return              # 로그 수익률
2. roll_return_6           # 6봉 롤링 수익률
3. atr_ratio               # ATR 비율
4. bb_width                # 볼린저 밴드 폭
5. bb_pos                  # 볼린저 밴드 위치
6. rsi                     # RSI (14)
7. macd_hist               # MACD 히스토그램
8. hma_ratio               # HMA 비율
9. cci                     # CCI
10. rvol                    # 상대 변동성
11. taker_ratio             # 테이커 비율
12. cvd_change              # CVD 변화량
13. mfi                     # MFI
14. cmf                     # CMF
15. vwap_dist               # VWAP 거리
16. wick_upper              # 상단 심지
17. wick_lower              # 하단 심지
18. range_pos               # 레인지 위치
19. swing_break             # 스윙 브레이크
20. chop                    # Choppiness Index
21. btc_return              # BTC 수익률
22. btc_rsi                 # BTC RSI
23. btc_corr                # BTC 상관계수
24. btc_vol                 # BTC 변동성
25. eth_btc_ratio           # ETH/BTC 비율
26. rsi_15m                 # 15분봉 RSI
27. trend_15m               # 15분봉 추세
28. rsi_1h                  # 1시간봉 RSI
29. trend_1h                # 1시간봉 추세
```

#### 전략 점수 (12개)
```
strategy_0:  BTCEthCorrelationStrategy
strategy_1:  VolatilitySqueezeStrategy
strategy_2:  OrderblockFVGStrategy
strategy_3:  HMAMomentumStrategy
strategy_4:  MFIMomentumStrategy
strategy_5:  BollingerMeanReversionStrategy
strategy_6:  VWAPDeviationStrategy
strategy_7:  RangeTopBottomStrategy
strategy_8:  StochRSIMeanReversionStrategy
strategy_9:  CMFDivergenceStrategy
strategy_10: CCIReversalStrategy
strategy_11: WilliamsRStrategy
```

각 전략 점수는 `-confidence ~ +confidence` 범위로 정규화됩니다.

#### 포지션 정보 (3개)
```
[0]: position_value    # 1.0 (LONG), -1.0 (SHORT), 0.0 (None)
[1]: unrealized_pnl    # 평가손익 (×10 스케일링)
[2]: holding_time      # 보유 시간 (정규화: holding_time / max_steps)
```

### 2.4 관측값 (Observation) 구조

```python
obs_seq: torch.Tensor  # Shape: (1, LOOKBACK, 29)
    # 최근 40봉의 29개 시계열 피처 (Z-Score 정규화)

obs_info: torch.Tensor  # Shape: (1, 15)
    # [12개 전략 점수] + [3개 포지션 정보] = 15차원
```

**반환 형식**: `(obs_seq, obs_info)` 튜플

---

## 3. 모델 아키텍처

### 3.1 xLSTMActorCritic 네트워크

#### 입력 차원
- **obs_seq**: `(batch, LOOKBACK, 29)` → `(batch, 40, 29)`
- **obs_info**: `(batch, 15)` → `(batch, 12 + 3)`

#### 네트워크 구조

```
Input (obs_seq: 29차원)
    ↓
[Input Projection] → Linear(29 → 128) + LayerNorm
    ↓
[Multi-Layer xLSTM Stack] (2 layers)
    ├─ Layer 1: sLSTMCell(128 → 128) + Pre-LN Residual
    └─ Layer 2: sLSTMCell(128 → 128) + Pre-LN Residual
    ↓
[Multi-Head Attention] → Weighted Pooling (4 heads)
    ↓ (128차원)
[Late Fusion] → Concat([attention_output, info_encoded])
    ↓ (128 + 64 = 192차원)
[Info Encoder] → Linear(15 → 64) + LayerNorm + GELU
    ↓ (64차원)
[Shared Trunk]
    ├─ Linear(192 → 256) + LayerNorm + GELU + Dropout(0.1)
    └─ Linear(256 → 128) + LayerNorm + GELU
    ↓ (128차원)
[Separate Heads]
    ├─ Actor Head: Linear(128 → 64) → Linear(64 → 3) + Softmax
    └─ Critic Head: Linear(128 → 64) → Linear(64 → 1)
    ↓
Output: (action_probs: [3], value: [1])
```

#### 주요 컴포넌트

1. **sLSTMCell**: Exponential Gating을 통한 메모리 강화
   - Input Gate: `i = exp(clamp(i, -5, 5))`
   - Forget Gate: `f = exp(clamp(f, -5, 5))`
   - Cell State: `c_next = f * c + i * tanh(z)`
   - Normalizer: `n_next = f * n + i`
   - Hidden: `h_next = sigmoid(o) * (c_next / n_next)`

2. **Multi-Head Attention**: Weighted Pooling
   - 4개 헤드로 시퀀스 내 중요 시점 학습
   - 학습 가능한 가중치로 풀링

3. **Pre-LN Residual**: `Norm(x) → Layer → x + Output`
   - 깊은 네트워크에서 안정적인 학습

4. **State Retention**: LSTM 상태를 에피소드 내에서 유지
   - `states = (h, c, n)` 형태로 관리
   - 에피소드 시작 시 `reset_episode_states()` 호출

### 3.2 출력

- **Actor Output**: `action_probs` - Shape: `(batch, 3)`
  - `[P(Action 0), P(Action 1), P(Action 2)]`
  - Softmax로 정규화되어 확률 분포 형성

- **Critic Output**: `value` - Shape: `(batch, 1)`
  - 상태 가치 함수 V(s) 추정

---

## 4. 액션 체계

### 4.1 액션 공간

**Discrete Action Space**: 3개 액션

| Action | 의미 | 동작 |
|--------|------|------|
| **0** | **AI 판단 청산** | 포지션이 있을 때 → 즉시 청산 (익절/손절)<br>포지션이 없을 때 → 관망 (현금 보유) |
| **1** | **LONG** | 포지션이 없을 때 → LONG 진입<br>SHORT 포지션일 때 → 스위칭 (SHORT 청산 + LONG 진입)<br>이미 LONG일 때 → 유지 (Keep Holding) |
| **2** | **SHORT** | 포지션이 없을 때 → SHORT 진입<br>LONG 포지션일 때 → 스위칭 (LONG 청산 + SHORT 진입)<br>이미 SHORT일 때 → 유지 (Keep Holding) |

### 4.2 액션 실행 로직

```python
# 최소 보유 시간 잠금 (Churning 방지)
is_locked = (current_position is not None) and (holding_time < 3)

# A. 강제 안전장치 (Stop Loss -2%)
if unrealized_pnl < -0.02:
    → 즉시 청산 (잠금 무시)

# B. AI 판단 행동 (잠금 해제 시)
if action == 0 and current_position is not None:
    → AI 판단 청산 (익절/손절)
    
if action == 1:
    if current_position == 'SHORT': → 스위칭
    elif current_position is None: → LONG 진입
    else: → 유지
    
if action == 2:
    if current_position == 'LONG': → 스위칭
    elif current_position is None: → SHORT 진입
    else: → 유지
```

### 4.3 액션 선택 메커니즘

1. **확률 분포**: Categorical Distribution 사용
   - `action_probs`에서 샘플링
   - 탐험을 위해 엔트로피 보너스 적용

2. **엔트로피 스케줄링**:
   ```
   entropy_coef = max(0.02, 0.05 * (0.999 ^ episode))
   ```
   - 초기: 0.05 (높은 탐험)
   - 점진적 감소: 0.999^episode
   - 최소값: 0.02 (지속적 탐험 유지)

---

## 5. 보상 체계

### 5.1 Dense Reward 구조

**핵심 개념**: 매 스텝마다 평가금액 변화에 보상을 부여하여 학습 신호를 밀도 있게 제공합니다.

#### 보상 함수 시그니처
```python
calculate_reward(step_pnl, realized_pnl, trade_done, holding_time)
```

#### 보상 구성 요소

**1. 과정 보상 (Shaping Reward)**
```python
reward += step_pnl * 50.0
```
- `step_pnl = unrealized_pnl - prev_unrealized_pnl`
- 포지션을 들고 있는 동안 가격이 유리하게 가면 보상
- 불리하게 가면 벌점
- **목적**: 포지션 유지 중에도 학습 신호 제공

**2. 결과 보상 (Terminal Reward)**
```python
if trade_done:
    fee = 0.0015  # TRANSACTION_COST
    net_pnl = realized_pnl - fee
    
    if net_pnl > 0:
        reward += net_pnl * 100.0  # 수익은 크게 칭찬
        reward += 1.0              # 승리 보너스
    else:
        reward += net_pnl * 80.0   # 손실은 아프게
```
- 거래 종료 시 확정 손익에 대한 보상/페널티
- 수수료를 반영한 순수익 기준

**3. 홀딩 비용 (Holding Cost)**
```python
if not trade_done:
    reward -= 0.0005 * holding_time
```
- 포지션을 너무 오래 들고 있으면 미미한 페널티
- 빠른 익절을 유도

#### 보상 클리핑
```python
reward = clip(reward, -10, 10)
```
- 안정적인 학습을 위해 보상 범위 제한

### 5.2 보상 계산 예시

#### 시나리오 1: 수익성 있는 홀딩
```
Step 1: LONG 진입 (entry_price = $3000)
Step 2: 가격 $3003 (+0.1%) → step_pnl = +0.001
        reward = 0.001 * 50.0 = +0.05
Step 3: 가격 $3006 (+0.2%) → step_pnl = +0.001
        reward = 0.001 * 50.0 = +0.05
Step 4: AI 판단 청산 (Action 0) → realized_pnl = +0.002
        reward = 0.002 * 50.0 + (0.002 - 0.0015) * 100.0 + 1.0
               = 0.1 + 0.05 + 1.0 = +1.15
```

#### 시나리오 2: 손실성 홀딩
```
Step 1: LONG 진입 (entry_price = $3000)
Step 2: 가격 $2997 (-0.1%) → step_pnl = -0.001
        reward = -0.001 * 50.0 = -0.05
Step 3: 가격 $2994 (-0.2%) → step_pnl = -0.001
        reward = -0.001 * 50.0 = -0.05
Step 4: Stop Loss 발동 (-2%) → realized_pnl = -0.02
        reward = -0.02 * 50.0 + (-0.02 - 0.0015) * 80.0
               = -1.0 + (-1.72) = -2.72
```

### 5.3 보상 체계의 장점

1. **빠른 학습**: 매 스텝마다 학습 신호 제공
2. **포지션 유지 인센티브**: 수익성 있는 포지션을 유지하면 지속적 보상
3. **안정성**: 클리핑으로 보상 폭발 방지
4. **균형**: 과정 보상(50.0)과 결과 보상(100.0)의 적절한 비율

---

## 6. 학습 프로세스

### 6.1 에피소드 구조

```
Episode Start
    ↓
[1] LSTM 상태 초기화 (reset_episode_states)
    ↓
[2] 랜덤 시작점 선택 (Train Set 내)
    ↓
[3] For each step (max 480 steps):
    ├─ Observation 생성 (obs_seq, obs_info)
    ├─ Action 선택 (Categorical Sampling)
    ├─ Trading Logic 실행
    │   ├─ Stop Loss 체크 (-2%)
    │   ├─ Action 0: AI 판단 청산
    │   ├─ Action 1: LONG 진입/스위칭/유지
    │   └─ Action 2: SHORT 진입/스위칭/유지
    ├─ Reward 계산 (Dense Reward)
    ├─ Transition 저장 (state, action, reward, next_state, prob, done)
    └─ Next State 생성
    ↓
[4] PPO Update (GAE + Clipped Surrogate)
    ├─ GAE 계산 (Generalized Advantage Estimation)
    ├─ PPO Loss 계산 (10 epochs)
    └─ Gradient Update
    ↓
Episode End
```

### 6.2 PPO 업데이트 과정

#### 1. GAE (Generalized Advantage Estimation)
```python
# TD Target
td_target = r + gamma * V(next_state) * (1 - done)

# TD Error
delta = td_target - V(state)

# GAE (Backward Pass)
gae = delta + gamma * lambda * gae_prev

# Returns
returns = gae + V(state)
```

#### 2. PPO Loss
```python
# Policy Ratio
ratio = exp(log_prob_new - log_prob_old)

# Clipped Surrogate
surr1 = ratio * advantage
surr2 = clip(ratio, 1-eps, 1+eps) * advantage
actor_loss = -min(surr1, surr2).mean()

# Critic Loss
critic_loss = SmoothL1Loss(V(state), returns)

# Entropy Bonus
entropy_loss = dist.entropy().mean()

# Total Loss
loss = actor_loss + 1.0 * critic_loss - entropy_coef * entropy_loss
```

#### 3. 학습 파라미터
- **Learning Rate**: 3e-5 (Adam Optimizer)
- **Gradient Clipping**: 0.5
- **Update Epochs**: 10 (k_epochs)
- **Batch Size**: 메모리에 쌓인 모든 트랜지션

### 6.3 모델 저장

- **Best Model**: `ppo_model_best.pth` + `ppo_model_best_scaler.pkl`
  - 최고 점수 갱신 시 저장
  - 실전 투입용

- **Last Model**: `ppo_model_last.pth` + `ppo_model_last_scaler.pkl`
  - 10 에피소드마다 저장
  - 학습 재개용

---

## 7. 하이퍼파라미터

### 7.1 PPO 알고리즘

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `PPO_GAMMA` | 0.99 | 할인율 (Discount Factor) |
| `PPO_LAMBDA` | 0.95 | GAE 람다 파라미터 |
| `PPO_EPS_CLIP` | 0.2 | PPO 클리핑 범위 |
| `PPO_K_EPOCHS` | 10 | PPO 업데이트 반복 횟수 |
| `PPO_LEARNING_RATE` | 3e-5 | 학습률 |
| `PPO_ENTROPY_COEF` | 0.05 | 엔트로피 계수 (초기값) |
| `PPO_ENTROPY_DECAY` | 0.999 | 엔트로피 감소율 |
| `PPO_ENTROPY_MIN` | 0.02 | 엔트로피 최소값 |

### 7.2 보상 함수

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `REWARD_MULTIPLIER` | 300 | (사용 안 함, Dense Reward 사용) |
| `LOSS_PENALTY_MULTIPLIER` | 500 | (사용 안 함, Dense Reward 사용) |
| `TRANSACTION_COST` | 0.0015 | 거래 비용 (0.15%) |
| `TIME_COST` | 0.0005 | 시간 비용 |
| `STOP_LOSS_THRESHOLD` | -0.02 | 강제 손절 임계값 (-2%) |
| **Step PnL Multiplier** | **50.0** | 과정 보상 배율 |
| **Terminal PnL Multiplier** | **100.0** | 결과 보상 배율 (수익) |
| **Terminal Loss Multiplier** | **80.0** | 결과 보상 배율 (손실) |
| **Holding Cost** | **0.0005** | 홀딩 비용 (스텝당) |

### 7.3 네트워크 아키텍처

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `LOOKBACK` | 40 | 시계열 봉 개수 |
| `NETWORK_HIDDEN_DIM` | 128 | 은닉층 차원 |
| `NETWORK_NUM_LAYERS` | 2 | xLSTM 레이어 개수 |
| `NETWORK_DROPOUT` | 0.1 | Dropout 비율 |
| `NETWORK_ATTENTION_HEADS` | 4 | Multi-Head Attention 헤드 개수 |
| `NETWORK_INFO_ENCODER_DIM` | 64 | Info Encoder 출력 차원 |
| `NETWORK_SHARED_TRUNK_DIM1` | 256 | Shared Trunk 첫 번째 레이어 |
| `NETWORK_SHARED_TRUNK_DIM2` | 128 | Shared Trunk 두 번째 레이어 |
| `NETWORK_ACTOR_HEAD_DIM` | 64 | Actor Head 은닉층 |
| `NETWORK_CRITIC_HEAD_DIM` | 64 | Critic Head 은닉층 |

### 7.4 학습 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `TRAIN_SPLIT` | 0.7 | 학습 데이터 비율 (70%) |
| `VAL_SPLIT` | 0.85 | 검증 데이터 비율 (85%) |
| `TRAIN_MAX_STEPS_PER_EPISODE` | 480 | 에피소드당 최대 스텝 수 |
| `TRAIN_NUM_EPISODES` | 2000 | 총 에피소드 수 |
| `MIN_HOLDING_TIME` | 3 | 최소 보유 캔들 수 (Churning 방지) |
| `TRAIN_SAMPLE_SIZE` | 50000 | 스케일러 학습용 샘플 크기 |

---

## 8. 데이터 차원 요약

### 8.1 입력 차원

| 구성 요소 | 차원 | 설명 |
|---------|------|------|
| `obs_seq` | `(1, 40, 29)` | 시계열 피처 (40봉 × 29개 피처) |
| `obs_info` | `(1, 15)` | 전략 점수(12) + 포지션 정보(3) |
| **Total Input** | **1,175차원** | `40 × 29 + 15 = 1,175` |

### 8.2 출력 차원

| 구성 요소 | 차원 | 설명 |
|---------|------|------|
| `action_probs` | `(1, 3)` | 액션 확률 분포 [P(0), P(1), P(2)] |
| `value` | `(1, 1)` | 상태 가치 함수 V(s) |

### 8.3 네트워크 내부 차원

| 레이어 | 입력 차원 | 출력 차원 |
|--------|---------|---------|
| Input Projection | 29 | 128 |
| xLSTM Layer 1 | 128 | 128 |
| xLSTM Layer 2 | 128 | 128 |
| Attention Pooling | (40, 128) | 128 |
| Info Encoder | 15 | 64 |
| Late Fusion | - | 192 (128 + 64) |
| Shared Trunk | 192 | 128 |
| Actor Head | 128 | 3 |
| Critic Head | 128 | 1 |

---

## 9. 학습 루프 상세

### 9.1 단일 스텝 처리

```python
# 1. Observation 생성
state = env.get_observation(
    position_info=[pos_val, unrealized_pnl*10, holding_time/max_steps],
    current_index=current_idx
)
# Returns: (obs_seq: (1,40,29), obs_info: (1,15))

# 2. Action 선택
action, log_prob = agent.select_action(state)
# LSTM 상태 유지: self.current_states 업데이트

# 3. Trading Logic
if action == 0 and current_position is not None:
    → 청산 (realized_pnl 계산)
elif action == 1:
    → LONG 진입/스위칭/유지
elif action == 2:
    → SHORT 진입/스위칭/유지

# 4. Reward 계산
step_pnl = unrealized_pnl - prev_unrealized_pnl
reward = env.calculate_reward(step_pnl, realized_pnl, trade_done, holding_time)

# 5. Transition 저장
agent.put_data((state, action, reward, next_state, log_prob, done))
```

### 9.2 배치 업데이트

```python
# 메모리에 트랜지션이 쌓이면
if len(agent.memory) >= batch_size or episode_end:
    agent.train_net(episode=episode_num)
    # GAE 계산 → PPO Loss → 10 epochs 업데이트
```

---

## 10. 주요 특징 요약

### 10.1 Dense Reward의 효과

- **이전 (Sparse Reward)**: 거래 종료 시에만 보상 → 학습 신호 부족
- **현재 (Dense Reward)**: 매 스텝 평가금액 변화에 보상 → 빠른 학습

### 10.2 AI 판단 청산의 효과

- **이전 (Passive Hold)**: Action 0 = 유지 → 과잉 거래 방지
- **현재 (AI Exit)**: Action 0 = 청산 → AI가 스스로 청산 시점 결정

### 10.3 State Retention의 효과

- **이전**: 매 스텝마다 LSTM 상태 초기화 → 시계열 패턴 학습 불가
- **현재**: 에피소드 내 LSTM 상태 유지 → 장기 패턴 학습 가능

---

## 11. 파일 구조

```
model/
├── train_ppo.py          # 학습 스크립트
├── evaluate_ppo.py       # 평가 스크립트
├── ppo_agent.py          # PPO 알고리즘 구현
├── trading_env.py        # 트레이딩 환경 (보상 함수 포함)
├── xlstm_network.py      # xLSTM 네트워크 아키텍처
├── preprocess.py         # 데이터 전처리 (Z-Score)
├── feature_engineering.py  # 피처 생성
└── mtf_processor.py     # 멀티 타임프레임 처리

data/
├── training_features.csv      # 29개 피처 + 전략 점수
├── cached_strategies.csv      # 전략 점수 캐시
├── ppo_model_best.pth         # 최고 성능 모델
├── ppo_model_best_scaler.pkl  # Best 모델용 스케일러
├── ppo_model_last.pth         # 최신 모델
└── ppo_model_last_scaler.pkl  # Last 모델용 스케일러

config.py                 # 모든 하이퍼파라미터 중앙 관리
```

---

## 12. 성능 최적화

### 12.1 캐싱 시스템

1. **피처 캐싱**: `training_features.csv`
   - 29개 피처를 미리 계산하여 저장
   - 학습 시작 시 즉시 로드

2. **전략 캐싱**: `cached_strategies.csv`
   - 12개 전략 점수를 미리 계산하여 저장
   - 병렬 처리로 계산 속도 향상

### 12.2 데이터 누수 방지

- 스케일러는 Train Set 80%만 사용
- 에피소드는 Train Set 내에서만 실행
- Test Set은 평가 시에만 사용

---

## 13. 참고사항

### 13.1 액션 의미 변경 이력

1. **초기**: Action 0 = Hold (유지)
2. **중간**: Action 0 = Exit (즉시 청산) → 과잉 거래 발생
3. **현재**: Action 0 = AI 판단 청산 (최소 보유 시간 후 가능)

### 13.2 보상 체계 변경 이력

1. **초기**: Realized PnL만 보상 (Sparse)
2. **중간**: Unrealized PnL 변화도 보상 (Dense)
3. **현재**: Step PnL + Terminal PnL (Dense Reward)

---

**문서 버전**: 1.0  
**최종 업데이트**: 2026-01-23  
**작성자**: PPO Model Documentation
