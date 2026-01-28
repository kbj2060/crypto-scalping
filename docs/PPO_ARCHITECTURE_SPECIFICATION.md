# PPO 모델 아키텍처 및 데이터 흐름 명세서

## 목차
1. [전체 아키텍처 개요](#전체-아키텍처-개요)
2. [네트워크 구조 (XLSTMNetwork)](#네트워크-구조-xlstmnetwork)
3. [PPO 알고리즘 상세](#ppo-알고리즘-상세)
4. [환경 및 리워드 구조](#환경-및-리워드-구조)
5. [액션 체계 (3-Action Target Position)](#액션-체계-3-action-target-position)
6. [데이터 흐름](#데이터-흐름)
7. [학습 파이프라인](#학습-파이프라인)
8. [하이퍼파라미터](#하이퍼파라미터)

---

## 전체 아키텍처 개요

### 시스템 구성
```
┌─────────────────────────────────────────────────────────────┐
│                    PPO Trading System                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐  │
│  │   Market     │ ───> │  Trading     │ ───> │   PPO    │  │
│  │   Data      │      │  Environment │      │  Agent   │  │
│  └──────────────┘      └──────────────┘      └──────────┘  │
│         │                      │                   │        │
│         │                      │                   │        │
│         └──────────────────────┴───────────────────┘        │
│                            │                                │
│                            ▼                                │
│                    ┌──────────────┐                         │
│                    │   XLSTM      │                         │
│                    │   Network    │                         │
│                    │ (Actor-Critic)│                        │
│                    └──────────────┘                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 컴포넌트
1. **TradingEnvironment**: 시장 데이터 처리 및 리워드 계산
2. **PPOAgent**: PPO 알고리즘 구현 및 정책 업데이트
3. **XLSTMNetwork**: Actor-Critic 신경망 (LSTM + Attention)
4. **DataCollector**: 시장 데이터 수집 및 전처리
5. **Strategy System**: 12개 기술적 분석 전략

---

## 네트워크 구조 (XLSTMNetwork)

### 전체 아키텍처
```
Input: (Batch, Seq_len=60, Feature_dim=29)
  │
  ├─> [1] Input Normalization (LayerNorm)
  │      Output: (Batch, 60, 29)
  │
  ├─> [2] LSTM (Multi-Layer)
  │      Input:  (Batch, 60, 29)
  │      Output: (Batch, 60, Hidden=128)
  │      States: (h_n, c_n) - 에피소드 간 유지
  │
  ├─> [3] Multi-Head Attention (Residual)
  │      Query/Key/Value: LSTM Output
  │      Output: (Batch, 60, 128)
  │      Residual: LSTM_out + Attention_out
  │      Normalization: LayerNorm
  │
  ├─> [4] Conv1D (Temporal Refinement)
  │      Input:  (Batch, 60, 128) -> Permute -> (Batch, 128, 60)
  │      Kernel: 3, Padding: 1
  │      Output: (Batch, 128, 60) -> Permute -> (Batch, 60, 128)
  │      Activation: GELU
  │      Normalization: LayerNorm
  │
  ├─> [5] Pooling (Last Time Step)
  │      Input:  (Batch, 60, 128)
  │      Output: (Batch, 128)  [context_feature]
  │
  ├─> [6] Info Encoder (Auxiliary Features)
  │      Input:  (Batch, Info_dim=15)  [12 strategies + 3 position info]
  │      Layers: Linear(15→64) → GELU → Linear(64→64) → GELU
  │      Output: (Batch, 64)  [info_feature]
  │
  ├─> [7] Feature Fusion
  │      Input:  [context_feature (128), info_feature (64)]
  │      Concatenate: (Batch, 192)
  │
  ├─> [8] Shared Trunk
  │      Layers: Linear(192→256) → GELU → Dropout(0.1)
  │              → Linear(256→128) → GELU → Dropout(0.1)
  │      Output: (Batch, 128)
  │
  └─> [9] Heads
       ├─> Actor Head: Linear(128→3) → Softmax
       │    Output: (Batch, 3) [Action Probabilities]
       │
       └─> Critic Head: Linear(128→1)
            Output: (Batch, 1) [State Value]
```

### 레이어별 상세

#### 1. Input Normalization
- **목적**: 입력 피처의 분포 정규화
- **구조**: `nn.LayerNorm(input_dim=29)`
- **입력**: `(Batch, 60, 29)` - 29개 피처, 60개 시퀀스
- **출력**: `(Batch, 60, 29)`

#### 2. LSTM (Multi-Layer)
- **구조**: `nn.LSTM(input_size=29, hidden_size=128, num_layers=1, batch_first=True)`
- **입력**: `(Batch, 60, 29)`
- **출력**: 
  - `output`: `(Batch, 60, 128)` - 시퀀스 전체 출력
  - `(h_n, c_n)`: `(1, Batch, 128)` - 마지막 은닉 상태 (에피소드 간 유지)
- **역할**: 시계열 패턴 추출 및 장기 의존성 학습

#### 3. Multi-Head Attention
- **구조**: `nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)`
- **입력**: LSTM Output `(Batch, 60, 128)`
- **처리**: Self-Attention (Query=Key=Value=LSTM Output)
- **Residual**: `LayerNorm(LSTM_out + Attention_out)`
- **출력**: `(Batch, 60, 128)`
- **역할**: 중요한 시점 강조 및 장거리 의존성 포착

#### 4. Conv1D (Temporal Refinement)
- **구조**: `nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)`
- **입력**: `(Batch, 60, 128)` → Permute → `(Batch, 128, 60)`
- **출력**: `(Batch, 128, 60)` → Permute → `(Batch, 60, 128)`
- **Activation**: GELU
- **Normalization**: LayerNorm
- **역할**: 시간축 패턴 압축 및 정제

#### 5. Pooling (Last Time Step)
- **방법**: 마지막 타임스텝 선택 `[:, -1, :]`
- **입력**: `(Batch, 60, 128)`
- **출력**: `(Batch, 128)` - `context_feature`

#### 6. Info Encoder
- **입력**: `(Batch, 15)` - [12 strategy scores + 3 position info]
- **구조**:
  ```
  Linear(15 → 64) → GELU
  Linear(64 → 64) → GELU
  ```
- **출력**: `(Batch, 64)` - `info_feature`
- **역할**: 전략 신호 및 포지션 정보 인코딩

#### 7. Shared Trunk
- **입력**: `(Batch, 192)` - Concatenate(context_feature, info_feature)
- **구조**:
  ```
  Linear(192 → 256) → GELU → Dropout(0.1)
  Linear(256 → 128) → GELU → Dropout(0.1)
  ```
- **출력**: `(Batch, 128)`
- **역할**: 통합 피처 추출 (Actor와 Critic 공유)

#### 8. Actor Head
- **구조**: `Linear(128 → 3) → Softmax`
- **입력**: Shared Trunk Output `(Batch, 128)`
- **출력**: `(Batch, 3)` - Action Probabilities [P(0), P(1), P(2)]
- **역할**: 정책(Policy) 출력

#### 9. Critic Head
- **구조**: `Linear(128 → 1)`
- **입력**: Shared Trunk Output `(Batch, 128)`
- **출력**: `(Batch, 1)` - State Value V(s)
- **역할**: 상태 가치 추정

### 가중치 초기화
- **Linear/RNN 가중치**: Orthogonal Initialization (gain=√2)
- **Bias**: Constant(0.0)
- **목적**: 학습 안정성 및 수렴 속도 향상

---

## PPO 알고리즘 상세

### PPO (Proximal Policy Optimization) 개요
PPO는 Trust Region 기반 Policy Gradient 방법으로, 정책 업데이트를 안정적으로 수행합니다.

### 핵심 구성 요소

#### 1. GAE (Generalized Advantage Estimation)
```python
# TD Error 계산
td_target = r + γ * V(s_next) * done_mask
delta = td_target - V(s)

# GAE 역순 계산
gae = 0.0
for t in reversed(range(len(delta))):
    gae = delta[t] + γ * λ * done_mask[t] * gae
    advantage_lst.insert(0, gae)

advantage = stack(advantage_lst)
target_v = advantage + V(s)  # Value Target
```

**파라미터**:
- `γ (gamma)`: 0.99 - 할인율
- `λ (lambda)`: 0.95 - GAE 람다

#### 2. PPO Clipped Objective
```python
ratio = exp(log_prob_new - log_prob_old)
surr1 = ratio * advantage
surr2 = clamp(ratio, 1-ε, 1+ε) * advantage
actor_loss = -min(surr1, surr2).mean()
```

**파라미터**:
- `ε (eps_clip)`: 0.2 - 클리핑 범위

#### 3. Value Function Clipping
```python
if PPO_USE_VALUE_CLIP:
    v_clipped = old_v + clamp(curr_v - old_v, -ε_v, ε_v)
    v_loss_1 = (curr_v - target_v)²
    v_loss_2 = (v_clipped - target_v)²
    critic_loss = 0.5 * max(v_loss_1, v_loss_2).mean()
```

**파라미터**:
- `ε_v (value_clip_eps)`: 0.2

#### 4. KL Divergence Early Stopping
```python
approx_kl = (exp(log_ratio) - 1) - log_ratio
if approx_kl.mean() > KL_TARGET:
    break  # 조기 종료
```

**파라미터**:
- `KL_TARGET`: 0.02

#### 5. Entropy Bonus
```python
entropy_coef = max(ENTROPY_MIN, ENTROPY_COEF * (ENTROPY_DECAY ** episode))
entropy_loss = entropy_coef * dist.entropy().mean()
```

**파라미터**:
- `ENTROPY_COEF`: 0.02 (초기값)
- `ENTROPY_DECAY`: 0.9999
- `ENTROPY_MIN`: 0.005

#### 6. 최종 Loss
```python
loss = actor_loss + critic_loss - entropy_loss
```

### 학습률 스케줄러
- **타입**: Cosine Annealing Warm Restarts
- **T_0**: 100 에피소드 (첫 주기)
- **T_mult**: 2 (주기 2배씩 증가: 100 → 200 → 400...)
- **eta_min**: 1e-6 (최소 학습률)

---

## 환경 및 리워드 구조

### TradingEnvironment 구조

#### 관측 공간 (Observation Space)
```
obs_seq:  (1, 60, 29)  - 시계열 피처
obs_info: (1, 15)      - 전략 점수 + 포지션 정보
```

**시계열 피처 (29개)**:
- 기본: log_return, roll_return_6, atr_ratio, bb_width, bb_pos
- 기술지표: rsi, macd_hist, hma_ratio, cci, rvol, taker_ratio, cvd_change
- 추가: mfi, cmf, vwap_dist, wick_upper, wick_lower, range_pos, swing_break, chop
- BTC 상관: btc_return, btc_rsi, btc_corr, btc_vol, eth_btc_ratio
- MTF: rsi_15m, trend_15m, rsi_1h, trend_1h

**정보 벡터 (15개)**:
- 전략 점수: strategy_0 ~ strategy_11 (12개)
- 포지션 정보: [position_value, unrealized_pnl*10, holding_time/1000] (3개)

#### 리워드 함수 (Reward Function)

**구조**:
```python
reward = 0.0

# 1. Step Reward (방향성 강화)
reward += step_pnl * 200.0

# 2. Holding Penalty (최소화)
if current_position is not None:
    reward -= 0.0001

# 3. Terminal Reward (핵심 목표)
if trade_done:
    fee = 0.0005
    net_pnl = realized_pnl - fee
    reward += net_pnl * 800.0
    
    if net_pnl > 0:
        reward += 5.0   # 익절 보너스
    elif net_pnl < 0:
        reward -= 2.0   # 손실 패널티
```

**리워드 구성 요소**:
| 구성 요소 | 배율/값 | 설명 |
|---------|--------|------|
| Step Reward | `step_pnl * 200.0` | 매 스텝 평가 손익 변화 |
| Terminal Reward | `net_pnl * 800.0` | 거래 완료 시 실현 손익 |
| 익절 보너스 | `+5.0` | 수익 거래 시 추가 보상 |
| 손실 패널티 | `-2.0` | 손실 거래 시 추가 처벌 |
| Holding Penalty | `-0.0001` | 포지션 보유 시 매 스텝 비용 |
| 수수료 | `0.0005` (0.05%) | 거래 비용 |

**특징**:
- Clipping 없음: Raw Reward 사용 (PPO 내부 정규화 활용)
- RL 논문 기준 스케일: Step * 200, Terminal * 800
- 실현 수익 중심: Terminal Reward가 Step Reward보다 4배 큼

---

## 액션 체계 (3-Action Target Position)

### 액션 정의
| Action | 의미 | 동작 |
|--------|------|------|
| **0: Neutral** | 무포지션 목표 | 포지션이 있으면 청산, 없으면 관망 |
| **1: Long** | 롱 포지션 목표 | 포지션이 없으면 진입, SHORT면 스위칭, LONG면 유지 |
| **2: Short** | 숏 포지션 목표 | 포지션이 없으면 진입, LONG면 스위칭, SHORT면 유지 |

### 액션 처리 로직

#### Action 0: Neutral
```python
if current_position == 'LONG':
    # 롱 청산
    realized_pnl = unrealized_pnl
    trade_done = True
    current_position = None
    trade_count += 1
elif current_position == 'SHORT':
    # 숏 청산
    realized_pnl = unrealized_pnl
    trade_done = True
    current_position = None
    trade_count += 1
# 이미 None이면 관망 (Pass)
```

#### Action 1: Long
```python
if current_position is None:
    # 신규 롱 진입
    current_position = 'LONG'
    entry_price = curr_price
    entry_index = current_idx
    trade_count += 1
elif current_position == 'SHORT':
    # 스위칭: 숏 청산 + 롱 진입
    realized_pnl = unrealized_pnl  # Short 종료
    trade_done = True
    current_position = 'LONG'
    entry_price = curr_price
    entry_index = current_idx
    trade_count += 2  # 청산 1회 + 진입 1회
# 이미 LONG이면 유지 (Pass)
```

#### Action 2: Short
```python
if current_position is None:
    # 신규 숏 진입
    current_position = 'SHORT'
    entry_price = curr_price
    entry_index = current_idx
    trade_count += 1
elif current_position == 'LONG':
    # 스위칭: 롱 청산 + 숏 진입
    realized_pnl = unrealized_pnl  # Long 종료
    trade_done = True
    current_position = 'SHORT'
    entry_price = curr_price
    entry_index = current_idx
    trade_count += 2  # 청산 1회 + 진입 1회
# 이미 SHORT면 유지 (Pass)
```

### Action Masking
- **기본 정책**: 모든 액션 허용 `[1.0, 1.0, 1.0]`
- **이유**: Action 3는 "같은 액션 = HOLD"이므로 막으면 안 됨
- **예시**: LONG 포지션일 때 Action 1(Long) 선택 = HOLD (포지션 유지)

---

## 데이터 흐름

### 학습 시 데이터 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Data Flow                       │
└─────────────────────────────────────────────────────────────┘

1. 데이터 로드
   ┌─────────────┐
   │ Market Data │ → training_features.csv
   │ (ETHUSDT)   │ → cached_strategies.csv (12 strategies)
   └─────────────┘
         │
         ▼
   ┌─────────────────┐
   │ DataCollector   │
   │ - eth_data      │
   │ - current_index │
   └─────────────────┘
         │
         ▼
2. 피처 추출 및 전처리
   ┌─────────────────────────────────────┐
   │ Feature Engineering                 │
   │ - 29 Technical Indicators           │
   │ - MTF Features (15m, 1h)           │
   │ - Strategy Scores (12 strategies)  │
   └─────────────────────────────────────┘
         │
         ▼
   ┌─────────────────┐
   │ DataPreprocessor │ → Z-Score Normalization
   │ (Scaler)        │ → Fit on Train Set Only
   └─────────────────┘
         │
         ▼
3. 에피소드 시작
   ┌─────────────────────────────────────┐
   │ Episode Initialization              │
   │ - Random start_idx (Train Set)      │
   │ - Reset LSTM states                 │
   │ - Reset reward states               │
   └─────────────────────────────────────┘
         │
         ▼
4. 스텝 루프 (for step in range(max_steps))
   ┌─────────────────────────────────────┐
   │ Step Processing                    │
   │                                     │
   │ a) 관측 생성                        │
   │    ┌─────────────────────────────┐ │
   │    │ get_observation()           │ │
   │    │ - obs_seq: (1, 60, 29)      │ │
   │    │ - obs_info: (1, 15)         │ │
   │    └─────────────────────────────┘ │
   │              │                      │
   │              ▼                      │
   │    ┌─────────────────────────────┐ │
   │    │ select_action()             │ │
   │    │ - Forward pass (XLSTM)      │ │
   │    │ - Sample action             │ │
   │    │ - Return: (action, log_prob,│ │
   │    │            value)            │ │
   │    └─────────────────────────────┘ │
   │              │                      │
   │              ▼                      │
   │    ┌─────────────────────────────┐ │
   │    │ Action Execution            │ │
   │    │ - Update position           │ │
   │    │ - Calculate PnL             │ │
   │    └─────────────────────────────┘ │
   │              │                      │
   │              ▼                      │
   │    ┌─────────────────────────────┐ │
   │    │ calculate_reward()          │ │
   │    │ - Step reward               │ │
   │    │ - Terminal reward           │ │
   │    │ - Penalties                 │ │
   │    └─────────────────────────────┘ │
   │              │                      │
   │              ▼                      │
   │    ┌─────────────────────────────┐ │
   │    │ put_data()                  │ │
   │    │ - Transition 저장           │ │
   │    │   (state, action, reward,   │ │
   │    │    next_state, prob, done, │ │
   │    │    value)                   │ │
   │    └─────────────────────────────┘ │
   └─────────────────────────────────────┘
         │
         ▼
5. 에피소드 종료 후 학습
   ┌─────────────────────────────────────┐
   │ train_net()                         │
   │                                     │
   │ a) 배치 변환                        │
   │    - Transition → Tensor            │
   │                                     │
   │ b) GAE 계산                         │
   │    - TD Error                       │
   │    - Advantage Estimation           │
   │    - Value Target                   │
   │                                     │
   │ c) PPO Update (K epochs)           │
   │    - Policy Loss (Clipped)          │
   │    - Value Loss (Clipped)           │
   │    - Entropy Loss                   │
   │    - KL Early Stopping              │
   │    - Gradient Clipping              │
   │                                     │
   │ d) Scheduler Step                   │
   │    - Cosine Annealing               │
   └─────────────────────────────────────┘
```

### 평가 시 데이터 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Data Flow                      │
└─────────────────────────────────────────────────────────────┘

1. 모델 로드
   ┌─────────────────┐
   │ Load Model      │ → ppo_model_best.pth
   │ Load Scaler     │ → ppo_model_best_scaler.pkl
   └─────────────────┘
         │
         ▼
2. 데이터 구간 설정
   ┌─────────────────┐
   │ Mode Selection  │
   │ - 'val': 70~85% │
   │ - 'test': 85~100%│
   └─────────────────┘
         │
         ▼
3. 백테스트 루프
   ┌─────────────────────────────────────┐
   │ For each step in eval range         │
   │                                     │
   │ a) 관측 생성                        │
   │ b) Action 선택 (argmax)             │
   │ c) 포지션 업데이트                  │
   │ d) 거래 기록                        │
   │ e) 자산 곡선 업데이트               │
   └─────────────────────────────────────┘
         │
         ▼
4. 리포트 생성
   ┌─────────────────────────────────────┐
   │ Metrics Calculation                 │
   │ - Final Balance                      │
   │ - ROI                               │
   │ - Win Rate                          │
   │ - Profit Factor                     │
   │ - Equity Curve Plot                 │
   │ - PnL Distribution                  │
   └─────────────────────────────────────┘
```

### 데이터 분할

```
전체 데이터 (100%)
│
├─ Train Set (70%)
│  └─ 학습 및 스케일러 Fit
│
├─ Validation Set (15%)
│  └─ 하이퍼파라미터 튜닝 및 조기 종료
│
└─ Test Set (15%)
   └─ 최종 성능 평가
```

**중요**: 스케일러는 Train Set만 사용하여 학습 (Data Leakage 방지)

---

## 학습 파이프라인

### 전체 학습 프로세스

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
└─────────────────────────────────────────────────────────────┘

1. 초기화 (__init__)
   ├─ DataCollector 초기화
   ├─ 전략 리스트 생성 (12개)
   ├─ 피처 로드 (training_features.csv)
   ├─ 전략 사전 계산 (cached_strategies.csv)
   ├─ TradingEnvironment 생성
   ├─ 스케일러 Fit (Train Set만 사용)
   ├─ PPOAgent 초기화
   └─ 기존 모델 로드 (있는 경우)

2. 학습 루프 (train)
   For episode in range(1, TRAIN_NUM_EPISODES + 1):
       ├─ train_episode(episode)
       │   ├─ 랜덤 시작점 선택 (Train Set 내)
       │   ├─ LSTM 상태 리셋
       │   ├─ 리워드 상태 리셋
       │   │
       │   For step in range(max_steps):
       │       ├─ 관측 생성
       │       ├─ 액션 선택 (with LSTM state)
       │       ├─ 액션 실행
       │       ├─ 리워드 계산
       │       ├─ Transition 저장
       │       └─ Next State 생성
       │
       │   └─ train_net() 호출
       │       ├─ GAE 계산
       │       ├─ PPO Update (K epochs)
       │       └─ Scheduler Step
       │
       ├─ 에피소드 리워드 기록
       ├─ 평균 리워드 계산
       ├─ 실시간 그래프 업데이트
       │
       ├─ Best Model 저장 (리워드 갱신 시)
       └─ Last Model 저장 (주기적)

3. 종료
   └─ 그래프 저장 및 정리
```

### 학습 최적화 기법

#### 1. Value Function Clipping
- **목적**: Critic 학습 안정화
- **방법**: 과거 Value 예측값에서 크게 벗어나지 않도록 제한
- **효과**: 급격한 Value 변화 방지

#### 2. KL Divergence Early Stopping
- **목적**: 정책 급변 방지
- **방법**: KL > 0.02 시 조기 종료
- **효과**: 학습 안정성 향상

#### 3. Cosine Annealing Warm Restarts
- **목적**: Local Minimum 탈출
- **방법**: 주기적 학습률 리셋 (100 → 200 → 400...)
- **효과**: 탐색 재개 및 수렴 개선

#### 4. Gradient Clipping
- **목적**: Gradient 폭발 방지
- **방법**: `max_norm=10.0`
- **효과**: LSTM 기울기 안정화

#### 5. Advantage Normalization
- **목적**: Advantage 스케일 정규화
- **방법**: `(advantage - mean) / (std + 1e-5)`
- **효과**: 학습 안정성 향상

---

## 하이퍼파라미터

### 네트워크 아키텍처
| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `LOOKBACK` | 60 | 시계열 시퀀스 길이 |
| `NETWORK_HIDDEN_DIM` | 128 | LSTM 은닉 차원 |
| `NETWORK_NUM_LAYERS` | 1 | LSTM 레이어 수 |
| `NETWORK_DROPOUT` | 0.1 | Dropout 비율 |
| `NETWORK_ATTENTION_HEADS` | 4 | Multi-Head Attention 헤드 수 |
| `NETWORK_INFO_ENCODER_DIM` | 64 | Info Encoder 출력 차원 |
| `STATE_DIM` | 29 | 시계열 피처 차원 |
| `INFO_DIM` | 15 | 정보 벡터 차원 (12 strategies + 3 position) |
| `ACTION_DIM` | 3 | 액션 차원 (Neutral, Long, Short) |

### PPO 알고리즘
| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `PPO_GAMMA` | 0.99 | 할인율 (Discount Factor) |
| `PPO_LAMBDA` | 0.95 | GAE 람다 파라미터 |
| `PPO_EPS_CLIP` | 0.2 | PPO 클리핑 범위 |
| `PPO_K_EPOCHS` | 4 | PPO 업데이트 반복 횟수 |
| `PPO_LEARNING_RATE` | 0.0002 | 학습률 (2e-4) |
| `PPO_ENTROPY_COEF` | 0.02 | 엔트로피 계수 (초기값) |
| `PPO_ENTROPY_DECAY` | 0.9999 | 엔트로피 감소율 |
| `PPO_ENTROPY_MIN` | 0.005 | 엔트로피 최소값 |
| `PPO_USE_VALUE_CLIP` | True | Value Clipping 사용 여부 |
| `PPO_VALUE_CLIP_EPS` | 0.2 | Value Clipping 범위 |
| `PPO_KL_TARGET` | 0.02 | KL Divergence 조기 종료 임계값 |

### 학습 설정
| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `TRAIN_NUM_EPISODES` | 2000 | 총 에피소드 수 |
| `TRAIN_MAX_STEPS_PER_EPISODE` | 480 | 에피소드당 최대 스텝 |
| `TRAIN_BATCH_SIZE` | 128 | 배치 크기 |
| `TRAIN_SPLIT` | 0.7 | 학습 데이터 비율 |
| `VAL_SPLIT` | 0.15 | 검증 데이터 비율 |
| `TEST_SPLIT` | 0.15 | 테스트 데이터 비율 |
| `TRAIN_SAVE_INTERVAL` | 50 | 모델 저장 간격 |

### 리워드 파라미터
| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `Step Reward Multiplier` | 200.0 | Step PnL 배율 |
| `Terminal Reward Multiplier` | 800.0 | Realized PnL 배율 |
| `익절 보너스` | +5.0 | 수익 거래 시 추가 보상 |
| `손실 패널티` | -2.0 | 손실 거래 시 추가 처벌 |
| `Holding Penalty` | -0.0001 | 포지션 보유 시 매 스텝 비용 |
| `TRANSACTION_COST` | 0.0005 | 거래 수수료 (0.05%) |

---

## 파일 구조

```
model/
├── config.py              # 하이퍼파라미터 설정
├── trading_env.py         # 트레이딩 환경 및 리워드
├── ppo_agent.py           # PPO 에이전트
├── xlstm_network.py       # XLSTM 네트워크 아키텍처
├── train_ppo.py           # 학습 스크립트
├── evaluate_ppo.py        # 평가 스크립트
└── preprocess.py          # 데이터 전처리 (Scaler)

data/
├── training_features.csv   # 전처리된 피처 데이터
├── cached_strategies.csv  # 사전 계산된 전략 점수
├── ppo_model_best.pth     # 최고 성능 모델
├── ppo_model_last.pth     # 최신 모델
├── ppo_model_best_scaler.pkl  # 최고 성능 스케일러
└── ppo_model_last_scaler.pkl  # 최신 스케일러
```

---

## 요약

### 핵심 특징
1. **3-Action Target Position**: Neutral, Long, Short (목표 포지션 기반)
2. **XLSTM Network**: LSTM + Multi-Head Attention + Conv1D
3. **고급 PPO 기법**: Value Clipping, KL Early Stopping, Cosine Scheduler
4. **RL 논문 기준 리워드**: Step * 200, Terminal * 800
5. **데이터 누수 방지**: Train Set만으로 스케일러 Fit

### 데이터 차원
- **입력**: `(Batch, 60, 29)` - 시계열 피처
- **정보**: `(Batch, 15)` - 전략 점수 + 포지션 정보
- **출력**: `(Batch, 3)` - 액션 확률, `(Batch, 1)` - 상태 가치

### 학습 안정성
- Value Clipping으로 Critic 안정화
- KL Early Stopping으로 정책 급변 방지
- Cosine Scheduler로 Local Minimum 탈출
- Gradient Clipping으로 기울기 안정화

---

**문서 버전**: 1.0  
**최종 업데이트**: 2026-01-27
