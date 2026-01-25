# SAC 모델 데이터 흐름 및 명세서

## 📋 목차
1. [전체 데이터 흐름](#전체-데이터-흐름)
2. [모델 아키텍처 명세](#모델-아키텍처-명세)
3. [액션 체계](#액션-체계)
4. [보상 체계](#보상-체계)
5. [상태 관리](#상태-관리)
6. [학습 프로세스](#학습-프로세스)

---

## 🔄 전체 데이터 흐름

### 초기화 단계 (SACTrainer.__init__)

```
┌─────────────────────────────────────────────────────────┐
│ 1. 데이터 수집기 초기화                                  │
│    DataCollector(use_saved_data=True)                   │
│    → eth_data: DataFrame (원본 OHLCV + 피처)             │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 피처 파일 로드/생성                                   │
│    _load_or_create_features()                          │
│                                                          │
│    IF data/training_features.csv 존재:                  │
│      → CSV 로드 (1초 이내)                              │
│    ELSE:                                                 │
│      → FeatureEngineer.generate_features()              │
│        * 25개 기본 기술적 지표 생성                      │
│      → MTFProcessor.add_mtf_features()                  │
│        * 4개 멀티 타임프레임 지표 추가                   │
│      → CSV 저장                                          │
│                                                          │
│    결과: eth_data = DataFrame (175,202행 × 29+ 컬럼)   │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 3. 전략 신호 사전 계산 (Pre-calculation)                │
│    precalculate_strategies()                            │
│                                                          │
│    IF data/cached_strategies.csv 존재:                  │
│      → 캐시 로드 (빠름) ⚡                               │
│    ELSE:                                                 │
│      → 12개 전략 순회하며 계산                           │
│        * strategy_0, strategy_1, ..., strategy_11       │
│        * 각 전략의 confidence × signal (LONG=+, SHORT=-)│
│      → CSV 저장                                          │
│                                                          │
│    결과: eth_data에 strategy_0~11 컬럼 추가             │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 스케일러 학습                                         │
│    _fit_global_scaler()                                 │
│                                                          │
│    - 29개 타겟 컬럼 선택                                 │
│    - 50,000개 샘플 무작위 추출                           │
│    - DataPreprocessor.fit() 실행                         │
│      * Mean, Std 계산 (Z-Score 정규화용)                 │
│                                                          │
│    결과: 전역 스케일러 학습 완료                         │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 5. SAC Agent 초기화                                      │
│    SACAgent(state_dim=29, action_dim=1, info_dim=15)      │
│                                                          │
│    - SACActor: Gaussian Policy (μ, log σ)                │
│    - SACCritic: Twin Q-Network (Q1, Q2)                 │
│    - Replay Buffer: 100,000 capacity                    │
│    - Alpha (Entropy): 자동 튜닝                          │
└─────────────────────────────────────────────────────────┘
```

### 에피소드 학습 단계 (train_episode)

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: 상태 관측 (get_observation)                     │
│                                                          │
│ 입력:                                                    │
│   - current_index: 현재 캔들 인덱스                      │
│   - position_info: [pos_val, unrealized_pnl, hold_time] │
│                                                          │
│ 처리:                                                    │
│   1. 시계열 데이터 슬라이싱                              │
│      df.iloc[curr_idx - 60 : curr_idx]                  │
│      → 29개 피처 컬럼 추출                               │
│                                                          │
│   2. Z-Score 정규화                                      │
│      normalized = (data - mean) / std                    │
│      → (1, 60, 29) tensor                                │
│                                                          │
│   3. 전략 점수 수집                                      │
│      df['strategy_0'] ~ df['strategy_11']               │
│      → 12개 전략 점수                                    │
│                                                          │
│   4. 포지션 정보 결합                                    │
│      [12개 전략 점수] + [3개 포지션 정보]                │
│      → (1, 15) tensor                                    │
│                                                          │
│ 출력:                                                    │
│   state = (obs_seq, obs_info)                           │
│   - obs_seq: (1, 60, 29) 시계열 피처                    │
│   - obs_info: (1, 15) 전략+포지션 정보                  │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: 행동 선택 (select_action)                       │
│                                                          │
│ 입력: state = (obs_seq, obs_info)                        │
│                                                          │
│ 처리:                                                    │
│   1. SACActor.forward()                                  │
│      - Input Projection: (60, 29) → (60, 128)            │
│      - Multi-Layer xLSTM: 2층 처리                       │
│        * Layer 0: (60, 128) → (60, 128)                 │
│        * Layer 1: (60, 128) → (60, 128)                 │
│      - Attention: (60, 128) → (128)                     │
│      - Info Encoder: (15) → (64)                         │
│      - Backbone: (128+64) → (128)                       │
│      - Heads: (128) → μ, log σ                          │
│                                                          │
│   2. Reparameterization Trick                            │
│      z ~ N(μ, σ)                                         │
│      action = tanh(z)                                    │
│                                                          │
│ 출력:                                                    │
│   action: float [-1.0, 1.0]                             │
│   next_states: (h, c, n) 튜플 (상태 유지)                │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: 액션 해석 (interpret_action)                   │
│                                                          │
│ 입력: action_value ∈ [-1.0, 1.0]                        │
│                                                          │
│ 처리:                                                    │
│   threshold = 0.3                                        │
│                                                          │
│   IF action_value > 0.3:                                 │
│     → action_code = 1 (LONG 진입)                       │
│   ELIF action_value < -0.3:                             │
│     → action_code = 2 (SHORT 진입)                       │
│   ELSE:                                                  │
│     → action_code = 0 (NEUTRAL - 청산/관망)              │
│                                                          │
│ 출력:                                                    │
│   action_code: {0, 1, 2}                                │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: 트레이딩 로직 실행                               │
│                                                          │
│ 포지션 청산 조건:                                        │
│   - 신호 변경: LONG인데 Short/Neutral 신호               │
│   - 손절: unrealized_pnl < -2%                           │
│                                                          │
│ 신규 진입 조건:                                          │
│   - 포지션 없을 때만                                     │
│   - action_code == 1 → LONG                              │
│   - action_code == 2 → SHORT                             │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 5: 보상 계산 (calculate_reward)                    │
│                                                          │
│ IF trade_done (거래 확정):                              │
│   - Realized PnL 계산                                   │
│   - 수수료 차감 (0.1%)                                   │
│   - 익절: net_pnl × 500 + 1.0                           │
│   - 손절: net_pnl × 600                                  │
│                                                          │
│ ELSE (포지션 유지):                                      │
│   - 시간 페널티: -0.001 × holding_time                  │
│   - 극단 손실: pnl < -2% → pnl × 10                     │
│                                                          │
│ 출력:                                                    │
│   reward: float (클리핑: [-20, 20])                      │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 6: Replay Buffer 저장                              │
│                                                          │
│ 저장 내용:                                               │
│   - state: (obs_seq, obs_info)                          │
│   - action: float (연속형)                               │
│   - reward: float                                       │
│   - next_state: (next_obs_seq, next_obs_info)          │
│   - done: bool                                          │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 7: 학습 업데이트 (update)                          │
│                                                          │
│ IF len(memory) > batch_size:                            │
│   1. Critic Update                                       │
│      - Target Q 계산 (Soft Q-Learning)                   │
│      - MSE Loss (Q1, Q2)                                │
│                                                          │
│   2. Actor Update                                       │
│      - Policy Gradient (Entropy Regularized)             │
│      - Maximize: min(Q) - α × log_prob                  │
│                                                          │
│   3. Alpha Update                                        │
│      - Automatic Entropy Tuning                          │
│      - Target: -action_dim                              │
└─────────────────────────────────────────────────────────┘
```

---

## 🏗️ 모델 아키텍처 명세

### 1. SACActor (Gaussian Policy)

#### 입력 차원
- **obs_seq**: `(batch, 60, 29)` - 시계열 피처
- **obs_info**: `(batch, 15)` - 전략 점수(12) + 포지션 정보(3)
- **states**: `(h, c, n)` - 각각 `(num_layers, batch, 128)`

#### 네트워크 구조

```
Input: (batch, 60, 29)
  ↓
[Input Projection]
  Linear(29 → 128)
  LayerNorm(128)
  Dropout(0.1)
  ↓
(batch, 60, 128)
  ↓
[Multi-Layer xLSTM]
  Layer 0:
    sLSTMCell(128 → 128)
    Pre-LN + Residual
  Layer 1:
    sLSTMCell(128 → 128)
    Pre-LN + Residual
  ↓
(batch, 60, 128)
  ↓
[Multi-Head Attention]
  Weighted Pooling
  ↓
(batch, 128)
  ↓
[Info Encoder]
  Linear(15 → 64)
  LayerNorm + GELU
  Dropout
  Linear(64 → 64)
  LayerNorm + GELU
  ↓
(batch, 64)
  ↓
[Late Fusion]
  Concat([128, 64]) → (batch, 192)
  ↓
[Backbone]
  Linear(192 → 256)
  LayerNorm + GELU + Dropout
  Linear(256 → 128)
  LayerNorm + GELU
  ↓
(batch, 128)
  ↓
[Heads]
  μ_head: Linear(128 → 1)
  log_std_head: Linear(128 → 1)
  ↓
Output: μ, log_std
```

#### 출력
- **μ**: `(batch, 1)` - 행동의 평균
- **log_std**: `(batch, 1)` - 행동의 로그 표준편차 (클리핑: [-20, 2])
- **next_states**: `(h, c, n)` - 다음 LSTM 상태

#### 샘플링 (sample 메서드)
```python
std = exp(log_std)
z ~ N(μ, std)  # Reparameterization Trick
action = tanh(z)  # [-1, 1] 범위로 압축
log_prob = log_prob(z) - log(1 - tanh(z)^2)  # Tanh 보정
```

### 2. SACCritic (Twin Q-Network)

#### 입력 차원
- **obs_seq**: `(batch, 60, 29)` - 시계열 피처
- **action**: `(batch, 1)` - 연속형 액션
- **obs_info**: `(batch, 15)` - 전략+포지션 정보
- **states**: `(h, c, n)` - LSTM 상태

#### 네트워크 구조

```
Input: (batch, 60, 29)
  ↓
[Input Projection]
  Linear(29 → 128)
  LayerNorm + Dropout
  ↓
(batch, 60, 128)
  ↓
[Multi-Layer xLSTM] (Actor와 동일 구조)
  Layer 0 + Layer 1
  ↓
(batch, 60, 128)
  ↓
[Attention]
  Weighted Pooling
  ↓
(batch, 128)
  ↓
[Info Encoder]
  Linear(15 → 64)
  ↓
(batch, 64)
  ↓
[State-Action Encoder]
  Concat([128, 64, 1]) → (batch, 193)
  Linear(193 → 256)
  LayerNorm + GELU + Dropout
  Linear(256 → 128)
  LayerNorm + GELU
  ↓
(batch, 128)
  ↓
[Twin Heads]
  Q1: Linear(128 → 1)
  Q2: Linear(128 → 1)
  ↓
Output: Q1, Q2
```

#### 출력
- **Q1**: `(batch, 1)` - 첫 번째 Q값
- **Q2**: `(batch, 1)` - 두 번째 Q값
- **next_states**: `(h, c, n)` - 다음 LSTM 상태

---

## 🎮 액션 체계

### 연속형 액션 공간

SAC는 **연속형 액션 공간**을 사용합니다.

#### 액션 출력
- **범위**: `[-1.0, 1.0]` (tanh로 압축)
- **분포**: Gaussian Policy (μ, σ)
- **샘플링**: Reparameterization Trick

#### 액션 해석 (interpret_action)

```python
threshold = 0.3

IF action_value > 0.3:
    → action_code = 1 (LONG 진입)
    → 진입 강도: action_value (0.3 ~ 1.0)
    → 예: 0.8 → 강한 LONG 신호
    
ELIF action_value < -0.3:
    → action_code = 2 (SHORT 진입)
    → 진입 강도: abs(action_value) (0.3 ~ 1.0)
    → 예: -0.9 → 강한 SHORT 신호
    
ELSE:  # -0.3 ≤ action_value ≤ 0.3
    → action_code = 0 (NEUTRAL)
    → 의미: 청산 또는 관망
    → 예: 0.1, -0.2 → 확신 없음, 포지션 청산
```

#### 액션 값 예시

| action_value | action_code | 의미 | 동작 |
|-------------|-------------|------|------|
| 0.85 | 1 | 강한 LONG | LONG 진입 또는 유지 |
| 0.45 | 1 | 중간 LONG | LONG 진입 또는 유지 |
| 0.2 | 0 | 약한 신호 | 포지션 청산 또는 관망 |
| -0.1 | 0 | 약한 신호 | 포지션 청산 또는 관망 |
| -0.5 | 2 | 중간 SHORT | SHORT 진입 또는 유지 |
| -0.9 | 2 | 강한 SHORT | SHORT 진입 또는 유지 |

#### Dead Zone (-0.3 ~ 0.3)

**목적**: 무한 존버(무한 보유) 방지

- **확신이 낮을 때**: 포지션을 잡지 않거나 청산
- **효과**: 불확실한 상황에서 빠른 의사결정 유도

#### 트레이딩 로직

```
현재 포지션: LONG
  ↓
액션 해석:
  - action_code == 1: 포지션 유지
  - action_code == 0 or 2: 청산 (신호 변경)
  
현재 포지션: SHORT
  ↓
액션 해석:
  - action_code == 2: 포지션 유지
  - action_code == 0 or 1: 청산 (신호 변경)
  
현재 포지션: None
  ↓
액션 해석:
  - action_code == 1: LONG 진입
  - action_code == 2: SHORT 진입
  - action_code == 0: 관망
```

#### 강제 청산 조건

1. **신호 변경**: 현재 포지션과 반대 신호
2. **손절**: `unrealized_pnl < -2%`
3. **Neutral 신호**: Dead Zone 내 액션

---

## 💰 보상 체계

### 핵심 원칙

**Realized PnL 중심**: 확정 손익만 보상으로 사용
- **Unrealized PnL 제거**: Hold 시 변동성 보상 방지
- **명확한 신호**: 거래 확정 시에만 보상/페널티

### 보상 함수 상세

#### 1. 거래 확정 시 (trade_done = True)

```python
transaction_cost = 0.001  # 0.1% (진입+청산)
net_pnl = pnl - transaction_cost

IF net_pnl > 0:  # 익절
    reward = net_pnl × 500 + 1.0
    # 수익률에 비례한 보상 + 성공 보너스
    
ELSE:  # 손절
    reward = net_pnl × 600
    # 손실은 더 큰 페널티 (600 > 500)
```

**예시**:
- 익절 (+1%): `(0.01 - 0.001) × 500 + 1.0 = 5.5`
- 익절 (+2%): `(0.02 - 0.001) × 500 + 1.0 = 10.5`
- 손절 (-1%): `(-0.01 - 0.001) × 600 = -6.6`
- 손절 (-2%): `(-0.02 - 0.001) × 600 = -12.6`

#### 2. 포지션 유지 중 (trade_done = False)

```python
# 시간 경과 페널티
reward = -0.001 × holding_time

# 극단 손실 페널티 (손절 유도)
IF pnl < -0.02:  # -2% 초과
    reward += pnl × 10
```

**예시**:
- Hold 50 스텝: `-0.001 × 50 = -0.05`
- Hold 100 스텝: `-0.001 × 100 = -0.1`
- Hold 200 스텝 + 손실 -3%: `-0.001 × 200 + (-0.03 × 10) = -0.5`
- Hold 300 스텝 + 손실 -5%: `-0.001 × 300 + (-0.05 × 10) = -0.8`

#### 3. 보상 클리핑

```python
reward = clip(reward, -20, 20)
```

**이유**: 극단적인 보상으로 인한 학습 불안정 방지

### 보상 체계 비교

| 상황 | 이전 (Unrealized 포함) | 현재 (Realized 중심) |
|------|----------------------|---------------------|
| Hold 중 수익 | 즉시 보상 | 보상 없음 |
| Hold 중 손실 | 즉시 페널티 | 시간 페널티만 |
| 거래 확정 | 추가 보상 | **주 보상** |
| 효과 | 변동성에 민감 | **명확한 신호** |

---

## 🧠 상태 관리

### LSTM 상태 (Memory Retention)

#### 상태 구조
```python
states = (h, c, n)
- h: Hidden state (num_layers, batch, hidden_dim)
- c: Cell state (num_layers, batch, hidden_dim)
- n: Normalization state (num_layers, batch, hidden_dim)
```

#### 상태 흐름

```
에피소드 시작:
  reset_episode_states()
  → actor_state = None
  
Step 1:
  action, _, _, next_states = actor.sample(obs_seq, obs_info, None)
  → actor_state = next_states  # 상태 저장
  
Step 2:
  action, _, _, next_states = actor.sample(obs_seq, obs_info, actor_state)
  → actor_state = next_states  # 상태 갱신
  
... (에피소드 동안 계속 유지)
```

#### 학습 시 상태 처리

```python
# Replay Buffer 샘플링 시
# 랜덤 배치이므로 상태를 None으로 초기화
action_new, _, _, _ = actor.sample(obs_seq, obs_info, states=None)
```

**이유**: 
- Replay Buffer의 샘플들은 서로 다른 시점
- 상태 연속성이 없으므로 초기화된 상태에서 시작

---

## 📊 학습 프로세스

### SAC 알고리즘

#### 1. Critic 업데이트

```python
# Target Q 계산
with torch.no_grad():
    next_action, next_log_prob, _, _ = actor.sample(next_state)
    q1_next, q2_next = critic_target(next_state, next_action)
    min_q_next = min(q1_next, q2_next) - alpha × next_log_prob
    q_target = reward + (1 - done) × gamma × min_q_next

# Current Q 계산
q1, q2 = critic(state, action)
critic_loss = MSE(q1, q_target) + MSE(q2, q_target)
```

#### 2. Actor 업데이트

```python
# Policy Gradient
action_new, log_prob, _, _ = actor.sample(state)
q1_new, q2_new = critic(state, action_new)
min_q_new = min(q1_new, q2_new)

actor_loss = (alpha × log_prob - min_q_new).mean()
# Maximize: min_q - alpha × log_prob
```

#### 3. Alpha (Entropy) 업데이트

```python
target_entropy = -action_dim  # -1.0
alpha_loss = -(log_alpha × (log_prob + target_entropy)).mean()
alpha = exp(log_alpha)
```

### 하이퍼파라미터

#### SAC 알고리즘 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `SAC_GAMMA` | 0.99 | 할인율 (Discount Factor) |
| `SAC_TAU` | 0.005 | Soft Target Update 계수 |
| `SAC_ALPHA` | 0.2 (초기) | 엔트로피 계수 (자동 튜닝, target=-1.0) |
| `SAC_LEARNING_RATE` | 3e-4 | 학습률 (Actor, Critic, Alpha 모두 동일) |
| `SAC_BATCH_SIZE` | 256 | 배치 크기 |
| `SAC_REPLAY_BUFFER_SIZE` | 100,000 | Replay Buffer 크기 |
| `SAC_WARMUP_RATIO` | 0.05 | Warmup 구간 비율 (5%) |

#### 네트워크 아키텍처 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `NETWORK_HIDDEN_DIM` | 128 | xLSTM 은닉층 차원 |
| `NETWORK_NUM_LAYERS` | 2 | xLSTM 레이어 개수 |
| `NETWORK_DROPOUT` | 0.1 | Dropout 비율 |
| `NETWORK_ATTENTION_HEADS` | 4 | Multi-Head Attention 헤드 개수 |
| `NETWORK_INFO_ENCODER_DIM` | 64 | Info Encoder 출력 차원 |
| `NETWORK_SHARED_TRUNK_DIM1` | 256 | Backbone 첫 번째 레이어 차원 |
| `NETWORK_SHARED_TRUNK_DIM2` | 128 | Backbone 두 번째 레이어 차원 |

#### 학습 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `LOOKBACK` | 60 | 시계열 피처를 위한 봉 개수 |
| `TRAIN_MAX_STEPS_PER_EPISODE` | 480 | 에피소드당 최대 스텝 수 (약 24시간) |
| `TRAIN_SAVE_INTERVAL` | 50 | 모델 저장 간격 (에피소드) |
| `TRAIN_SPLIT` | 0.8 | 학습 데이터 비율 (80%) |

### 스케줄러

- **Warmup**: 전체 학습의 5% 동안 LR 0 → 1.0
- **Linear Decay**: 나머지 95% 동안 LR 1.0 → 0

---

## 📈 데이터 차원 요약

### 입력 차원

| 구성 요소 | 차원 | 설명 |
|---------|------|------|
| **obs_seq** | (1, 60, 29) | 시계열 피처 (60개 시점, 29개 피처) |
| **obs_info** | (1, 15) | 전략 점수(12) + 포지션 정보(3) |
| **전체 상태** | (obs_seq, obs_info) | 튜플 형태 |

### 출력 차원

| 구성 요소 | 차원 | 설명 |
|---------|------|------|
| **action** | (1,) | 연속형 액션 [-1.0, 1.0] |
| **log_prob** | (1, 1) | 액션의 로그 확률 |
| **next_states** | (h, c, n) | 각각 (2, 1, 128) |

### 네트워크 차원

| 레이어 | 입력 | 출력 | 설명 |
|--------|------|------|------|
| Input Projection | (60, 29) | (60, 128) | 차원 확장 |
| xLSTM Layer 0 | (60, 128) | (60, 128) | 시계열 처리 |
| xLSTM Layer 1 | (60, 128) | (60, 128) | 깊은 패턴 학습 |
| Attention | (60, 128) | (128) | 시퀀스 요약 |
| Info Encoder | (15) | (64) | 포지션 정보 인코딩 |
| Backbone | (192) | (128) | 특징 결합 |
| Actor Head | (128) | (1, 1) | μ, log_std |

---

## 🔑 핵심 설계 원칙

1. **전략 신호 Pre-calculation**: 학습 루프 내 계산 제거 (속도 향상)
2. **Realized PnL 중심**: 명확한 보상 신호
3. **Dead Zone**: 무한 존버 방지
4. **Stateful**: 에피소드 동안 기억 유지
5. **Multi-Layer xLSTM**: 깊은 패턴 학습
6. **Best/Last 분리**: 실전용과 재개용 모델 분리

---

## 📝 참고사항

- **학습 데이터**: 전체 데이터의 80% (Train)
- **에피소드 길이**: 최대 480 스텝 (약 24시간, 3분봉 기준)
- **모델 저장**: 
  - Best: 신기록 달성 시 (`sac_model_best.pth`)
  - Last: 주기적 저장 (`sac_model_last.pth`)
- **스케일러**: Best/Last 모델과 짝을 맞춰 저장 (`*_scaler.pkl`)
- **전략 신호**: `data/cached_strategies.csv`에 캐싱
- **피처 파일**: `data/training_features.csv`에 저장

## 🔍 주요 개선사항 (vs 이전 버전)

1. **전략 신호 Pre-calculation**: 학습 루프 내 계산 제거
2. **Input Projection**: 차원 불일치 해결 (29 → 128)
3. **Multi-Layer xLSTM**: 깊은 패턴 학습 (2층)
4. **Dead Zone**: 무한 존버 방지 (-0.3 ~ 0.3)
5. **Realized PnL 중심**: 명확한 보상 신호
6. **Stateful**: 에피소드 동안 기억 유지
7. **Best/Last 분리**: 실전용과 재개용 모델 분리
