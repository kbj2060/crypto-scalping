# PPO 모델 명세서 v2.0 (4-Action Strategy)

## 📋 목차
1. [시스템 개요](#1-시스템-개요)
2. [데이터 흐름 (Data Flow)](#2-데이터-흐름-data-flow)
3. [모델 아키텍처](#3-모델-아키텍처)
4. [액션 체계 (4-Action)](#4-액션-체계-4-action)
5. [보상 함수 (Reward Function)](#5-보상-함수-reward-function)
6. [학습 프로세스 (Training Process)](#6-학습-프로세스-training-process)
7. [하이퍼파라미터](#7-하이퍼파라미터)

---

## 1. 시스템 개요

### 1.1 모델 개요
이 PPO 모델은 암호화폐 스캘핑 거래를 위한 강화학습 에이전트입니다.

**핵심 특징:**
- **xLSTM 기반 Actor-Critic**: Multi-Layer sLSTM + Multi-Head Attention
- **4-Action Strategy**: HOLD, LONG, SHORT, EXIT (명시적 청산 분리)
- **Dense Reward**: 매 스텝 평가손익 변화에 실시간 피드백
- **State Retention**: LSTM 상태를 에피소드 내에서 유지하여 시계열 맥락 보존
- **Value Clipping**: Critic 학습 안정성을 위한 Clipped Loss 사용

### 1.2 기술 스택
- **알고리즘**: PPO (Proximal Policy Optimization)
- **네트워크**: xLSTM (sLSTMCell) + Multi-Head Attention
- **Action Space**: Discrete (4 actions)
- **State Space**: Tuple (Sequence Features + Info Features)

---

## 2. 데이터 흐름 (Data Flow)

### 2.1 전체 파이프라인

```
[원본 데이터]
    ↓
[Feature Engineering] → 25개 기본 피처 생성
    ├─ 가격 & 변동성 (9개)
    ├─ 거래량 & 오더플로우 (6개)
    ├─ 패턴 & 유동성 (5개)
    └─ 시장 상관관계 (5개)
    ↓
[MTF Processing] → 4개 멀티타임프레임 피처 추가
    ├─ RSI_15m, Trend_15m
    └─ RSI_1h, Trend_1h
    ↓
[전략 신호 계산] → 12개 전략 점수 (strategy_0 ~ strategy_11)
    ├─ 폭발장 전략 (6개)
    └─ 횡보장 전략 (6개)
    ↓
[데이터 저장] → training_features.csv + cached_strategies.csv
    ↓
[Scaler Training] → Train Set 70%만 사용 (Data Leakage 방지)
    └─ 저장: data/ppo_model_best_scaler.pkl
    ↓
[Episode Loop]
    ├─ Observation 생성
    │   ├─ obs_seq: (1, 60, 29) - 시계열 피처
    │   └─ obs_info: (1, 15) - 전략 점수(12) + 포지션 정보(3)
    ├─ Action 선택 (0, 1, 2, 3)
    ├─ Trading Logic 실행
    │   ├─ Action 1: LONG (진입/스위칭)
    │   ├─ Action 2: SHORT (진입/스위칭)
    │   ├─ Action 3: EXIT (명시적 청산)
    │   └─ Action 0: HOLD (관망/유지)
    ├─ Reward 계산 (4-Action Reward Function)
    ├─ Transition 저장
    └─ PPO Update (GAE + Clipped Surrogate + Value Clipping)
    ↓
[Model Save] → data/ppo_model_best.pth / data/ppo_model_last.pth
```

### 2.2 데이터 구조

#### 2.2.1 입력 데이터
- **ETH/USDT 3분봉 데이터**: `data/eth_3m_1year.csv`
  - 컬럼: `open`, `high`, `low`, `close`, `volume`, `taker_buy_volume`, `cvd`
- **BTC/USDT 3분봉 데이터**: `data/btc_3m_1year.csv` (선택적)
  - 컬럼: `close`, `volume`

#### 2.2.2 피처 엔지니어링 결과
**29개 시계열 피처** (`training_features.csv`):
1. `log_return` - 로그 수익률
2. `roll_return_6` - 6봉 롤링 수익률
3. `atr_ratio` - ATR 비율
4. `bb_width` - 볼린저 밴드 폭
5. `bb_pos` - 볼린저 밴드 위치
6. `rsi` - RSI
7. `macd_hist` - MACD 히스토그램
8. `hma_ratio` - HMA 비율
9. `cci` - CCI
10. `rvol` - 상대 거래량
11. `taker_ratio` - 테이커 매수 비율
12. `cvd_change` - CVD 변화량
13. `mfi` - MFI
14. `cmf` - CMF
15. `vwap_dist` - VWAP 거리
16. `wick_upper` - 상단 심지 비율
17. `wick_lower` - 하단 심지 비율
18. `range_pos` - 레인지 위치
19. `swing_break` - 스윙 브레이크 플래그
20. `chop` - 촙 인덱스
21. `btc_return` - BTC 수익률
22. `btc_rsi` - BTC RSI
23. `btc_corr` - BTC-ETH 상관관계
24. `btc_vol` - BTC 변동성
25. `eth_btc_ratio` - ETH/BTC 비율
26. `rsi_15m` - 15분봉 RSI
27. `trend_15m` - 15분봉 추세
28. `rsi_1h` - 1시간봉 RSI
29. `trend_1h` - 1시간봉 추세

**12개 전략 점수** (`cached_strategies.csv`):
- `strategy_0` ~ `strategy_11`: 각 전략의 신호 강도 (-1.0 ~ 1.0)

#### 2.2.3 관측 공간 (Observation Space)

**obs_seq (시계열 피처)**
- Shape: `(1, LOOKBACK, 29)`
- LOOKBACK: 60 (config.LOOKBACK)
- 29개 피처: 위의 29개 컬럼
- 정규화: Z-Score Normalization (DataPreprocessor)

**obs_info (정보 피처)**
- Shape: `(1, 15)`
- 구성:
  - `[0:12]`: 전략 점수 (12개)
  - `[12]`: 포지션 값 (1.0=LONG, -1.0=SHORT, 0.0=None)
  - `[13]`: 평가손익 (unrealized_pnl * 10)
  - `[14]`: 보유 시간 (holding_time / max_steps)

**최종 State**
- Type: Tuple `(obs_seq, obs_info)`
- obs_seq: `torch.FloatTensor` shape `(1, 60, 29)`
- obs_info: `torch.FloatTensor` shape `(1, 15)`

### 2.3 전처리 파이프라인

#### 2.3.1 Feature Engineering (`model/feature_engineering.py`)
```python
FeatureEngineer.generate_features()
  ├─ _add_price_volatility_features() → 9개 피처
  ├─ _add_volume_flow_features() → 6개 피처
  ├─ _add_pattern_liquidity_features() → 5개 피처
  └─ _add_market_correlation_features() → 5개 피처 (BTC 데이터 필요)
```

#### 2.3.2 MTF Processing (`model/mtf_processor.py`)
```python
MTFProcessor.add_mtf_features()
  ├─ 15분봉 리샘플링 → RSI_15m, Trend_15m 계산 → Shift(1)
  └─ 1시간봉 리샘플링 → RSI_1h, Trend_1h 계산 → Shift(1)
  → Look-ahead Bias 완벽 차단
```

#### 2.3.3 정규화 (`model/preprocess.py`)
```python
DataPreprocessor
  ├─ fit(): 전체 데이터셋에서 mean, std 계산
  ├─ transform(): Z-Score 정규화 (x - mean) / std
  └─ save()/load(): pickle로 저장/로드
```

---

## 3. 모델 아키텍처

### 3.1 네트워크 구조 (`model/xlstm_network.py`)

```
Input (obs_seq: [1, 60, 29], obs_info: [1, 15])
    ↓
[Input Projection]
    Linear(29 → 128) + LayerNorm + Dropout(0.1)
    ↓
[xLSTM Stack (2 Layers)]
    For each layer:
        sLSTMCell(128 → 128)
        ├─ Input Norm (LayerNorm)
        ├─ sLSTM Forward (h, c, n 상태 유지)
        └─ Residual Connection (input + h)
    ↓
[Multi-Head Attention]
    MultiheadAttention(128, heads=4)
    ├─ Self-Attention
    ├─ Weighted Pooling (Linear → Softmax → Weighted Sum)
    └─ Output: [1, 128]
    ↓
[Info Encoder]
    Linear(15 → 64) → LayerNorm → GELU → Dropout → Linear(64 → 64)
    Output: [1, 64]
    ↓
[Concatenate]
    Concat([context(128), info_encoded(64)]) → [1, 192]
    ↓
[Shared Trunk]
    Linear(192 → 256) → LayerNorm → GELU
    → Linear(256 → 128) → LayerNorm → GELU
    Output: [1, 128]
    ↓
[Separate Heads]
    ├─ Actor Head: Linear(128 → 64) → GELU → Dropout → Linear(64 → 4) → Softmax
    │   Output: [1, 4] (Action Probabilities)
    └─ Critic Head: Linear(128 → 32) → LayerNorm → GELU → Linear(32 → 1)
        Output: [1, 1] (State Value)
```

### 3.2 주요 컴포넌트

#### 3.2.1 sLSTMCell
- **입력**: `(x, h, c, n)`
- **출력**: `(h_next, c_next, n_next)`
- **특징**:
  - Gate Clamping: `[-5, 5]` 범위로 제한
  - State Clamping: `c_next [-1e6, 1e6]`, `n_next [1e-6, 1e6]`
  - NaN/Inf 방지: `nan_to_num` 처리

#### 3.2.2 MultiHeadAttention
- **입력**: `[batch, seq_len, hidden_dim]`
- **출력**: `[batch, hidden_dim]` (Weighted Pooling)
- **특징**:
  - Self-Attention으로 시퀀스 내 의존성 학습
  - Weighted Pooling으로 시퀀스를 단일 벡터로 압축

#### 3.2.3 Info Encoder
- **입력**: `[batch, 15]` (전략 점수 12 + 포지션 정보 3)
- **출력**: `[batch, 64]`
- **목적**: 전략 신호와 포지션 정보를 고차원으로 인코딩

#### 3.2.4 Shared Trunk
- **입력**: `[batch, 192]` (context 128 + info_encoded 64)
- **출력**: `[batch, 128]`
- **특징**: Actor와 Critic이 공유하는 특징 추출기

#### 3.2.5 Actor Head
- **입력**: `[batch, 128]`
- **출력**: `[batch, 4]` (Action Probabilities)
- **특징**: Dropout 유지 (탐험 유도)

#### 3.2.6 Critic Head
- **입력**: `[batch, 128]`
- **출력**: `[batch, 1]` (State Value)
- **특징**: LayerNorm 추가 (Value Function 안정화), Dropout 제거

### 3.3 상태 유지 (State Retention)

**LSTM 상태 구조:**
- `h`: Hidden state `[num_layers, batch, hidden_dim]`
- `c`: Cell state `[num_layers, batch, hidden_dim]`
- `n`: Normalization state `[num_layers, batch, hidden_dim]`

**에피소드 내 상태 유지:**
- `select_action()` 호출 시 `self.current_states` 유지
- `reset_episode_states()`로 에피소드 시작 시 초기화

---

## 4. 액션 체계 (4-Action)

### 4.1 액션 정의

| Action | 값 | 의미 | 동작 |
|--------|-----|------|------|
| HOLD | 0 | 관망/유지 | 현재 포지션 유지 (무포지션이면 계속 관망) |
| LONG | 1 | 롱 진입/스위칭 | SHORT → LONG: 스위칭 (기존 청산 + 새 진입)<br>None → LONG: 진입<br>이미 LONG: 유지 |
| SHORT | 2 | 숏 진입/스위칭 | LONG → SHORT: 스위칭 (기존 청산 + 새 진입)<br>None → SHORT: 진입<br>이미 SHORT: 유지 |
| EXIT | 3 | 명시적 청산 | 포지션 있으면 청산<br>포지션 없으면 HOLD와 동일 |

### 4.2 액션 처리 로직 (`train_ppo.py`)

```python
# Action 1: LONG
if action == 1:
    if current_position == 'SHORT':  # 스위칭
        realized_pnl = unrealized_pnl
        trade_done = True
        current_position = 'LONG'
        entry_price = curr_price
        entry_index = current_idx
    elif current_position is None:  # 진입
        current_position = 'LONG'
        entry_price = curr_price
        entry_index = current_idx
    # 이미 LONG이면 유지

# Action 2: SHORT
elif action == 2:
    if current_position == 'LONG':  # 스위칭
        realized_pnl = unrealized_pnl
        trade_done = True
        current_position = 'SHORT'
        entry_price = curr_price
        entry_index = current_idx
    elif current_position is None:  # 진입
        current_position = 'SHORT'
        entry_price = curr_price
        entry_index = current_idx
    # 이미 SHORT면 유지

# Action 3: EXIT
elif action == 3:
    if current_position is not None:
        realized_pnl = unrealized_pnl
        trade_done = True
        current_position = None
        entry_price = 0.0
        entry_index = 0

# Action 0: HOLD
# 아무것도 하지 않음 (Pass)
```

### 4.3 강제 안전장치

**Stop Loss Threshold:**
- 임계값: `config.STOP_LOSS_THRESHOLD = -0.05` (-5%)
- 동작: `unrealized_pnl < -0.05`일 때 강제 청산
- 목적: 극단적 손실 방지 (학습 가속)

---

## 5. 보상 함수 (Reward Function)

### 5.1 보상 함수 구조 (`model/trading_env.py`)

```python
def calculate_reward(step_pnl, realized_pnl, trade_done, action, prev_position):
    reward = 0.0
    
    # 1. HOLD Small Bonus
    if action == 0:
        reward += 0.0002
    
    # 2. Position Holding Reward (Trend Riding)
    if prev_position is not None:
        reward += step_pnl * 30.0
    
    # 3. Switching Penalty
    if trade_done and (action == 1 or action == 2):
        reward -= 0.5
    
    # 4. EXIT Rewards (Realized PnL)
    if trade_done and action == 3:
        fee = config.TRANSACTION_COST
        net_pnl = realized_pnl - fee
        
        if net_pnl > 0:
            reward += net_pnl * 250.0
            if net_pnl > 0.005:  # 0.5% 이상
                reward += 1.0  # 보너스
        else:
            reward += net_pnl * 300.0
            reward -= 0.2  # 고정 페널티
    
    return np.clip(reward, -10, 10)
```

### 5.2 보상 구성 요소

#### 5.2.1 HOLD 보너스
- **값**: `+0.0002`
- **목적**: 관망도 전략임을 인지시킴

#### 5.2.2 Position Holding Reward
- **공식**: `step_pnl * 30.0`
- **의미**: 포지션 보유 중 평가익 변화에 실시간 피드백
- **효과**: 추세를 길게 타도록 유도

#### 5.2.3 Switching Penalty
- **값**: `-0.5`
- **조건**: `trade_done=True` AND `action in [1, 2]`
- **목적**: 잦은 포지션 변경 방지

#### 5.2.4 EXIT Rewards
**익절 시:**
- 기본: `net_pnl * 250.0`
- 보너스: `net_pnl > 0.005`일 때 `+1.0`

**손절 시:**
- 기본: `net_pnl * 300.0` (더 큰 페널티)
- 고정 페널티: `-0.2`

**수수료:**
- `-config.TRANSACTION_COST` (약 -0.0015)

### 5.3 보상 스케일

**예시 계산:**

**시나리오 1: 익절 (+2%)**
```
EXIT 보상: (0.02 - 0.0015) * 250.0 = +4.625
보너스: +1.0 (0.02 > 0.005)
총 보상: +5.625
```

**시나리오 2: 손절 (-2%)**
```
EXIT 보상: (-0.02 - 0.0015) * 300.0 = -6.45
고정 페널티: -0.2
총 보상: -6.65
```

**시나리오 3: 스위칭 (SHORT → LONG)**
```
스위칭 페널티: -0.5
Position Holding: step_pnl * 30.0 (평가익 변화에 따라)
```

---

## 6. 학습 프로세스 (Training Process)

### 6.1 PPO 알고리즘

#### 6.1.1 GAE (Generalized Advantage Estimation)
```python
# TD Target
td_target = r_batch + gamma * v_next * done_batch
delta = td_target - v_s

# GAE 계산 (역방향)
gae = 0
for step in reversed(range(len(r_batch)):
    if done_batch[step] == 0:
        gae = delta[step] + gamma * lambda * gae
    else:
        gae = delta[step]
    advantages.insert(0, gae)

# 정규화
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
returns = advantages + v_s.squeeze()
```

#### 6.1.2 PPO Clipped Surrogate Loss
```python
# Importance Sampling Ratio
ratio = exp(log_prob_new - log_prob_old)

# Clipped Surrogate
surr1 = ratio * advantages
surr2 = clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages
actor_loss = -min(surr1, surr2).mean()
```

#### 6.1.3 Value Clipping (Critic)
```python
# Old Value 저장
old_values = v_s.clone()

# Unclipped Loss
loss_v1 = smooth_l1_loss(v_pred, v_target, reduction='none')

# Clipped Loss
v_pred_clipped = old_values + clamp(v_pred - old_values, -eps_clip, eps_clip)
loss_v2 = smooth_l1_loss(v_pred_clipped, v_target, reduction='none')

# 보수적 업데이트
critic_loss = max(loss_v1, loss_v2).mean()
```

#### 6.1.4 엔트로피 정책
```python
entropy_loss = dist.entropy().mean()
current_entropy_coef = max(
    PPO_ENTROPY_MIN,
    PPO_ENTROPY_COEF * (PPO_ENTROPY_DECAY ** episode)
)
```

#### 6.1.5 최종 Loss
```python
loss = actor_loss + 0.5 * critic_loss - current_entropy_coef * entropy_loss
```

### 6.2 학습 루프 (`train_ppo.py`)

#### 6.2.1 에피소드 구조
```python
for episode in range(1, num_episodes + 1):
    # 1. 랜덤 시작 인덱스 선택
    start_idx = random(start_min, start_max)
    
    # 2. 에피소드 실행
    episode_reward, trade_count = train_episode(episode)
    
    # 3. PPO 업데이트 (에피소드 종료 후)
    loss = agent.train_net(episode=episode)
    
    # 4. 모델 저장 (최고 성능 또는 주기적)
    if episode_reward > best_reward:
        save_model(best_model)
    elif episode % save_interval == 0:
        save_model(last_model)
```

#### 6.2.2 스텝 루프
```python
for step in range(max_steps):
    # 1. 평가손익 계산
    unrealized_pnl = calculate_unrealized_pnl()
    step_pnl = unrealized_pnl - prev_unrealized_pnl
    
    # 2. 관측 생성
    state = env.get_observation(position_info, current_index)
    
    # 3. 행동 선택
    action, prob = agent.select_action(state)
    
    # 4. 거래 로직 실행
    # - 강제 손절 체크
    # - AI 행동 처리 (4-Action)
    
    # 5. 보상 계산
    reward = env.calculate_reward(step_pnl, realized_pnl, trade_done, action, prev_position)
    
    # 6. Transition 저장
    agent.put_data((state, action, reward, next_state, prob, done))
    
    # 7. 다음 스텝 준비
    prev_unrealized_pnl = unrealized_pnl if not trade_done else 0.0
```

### 6.3 학습률 스케줄링

**LinearLR Scheduler:**
```python
scheduler = LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.01,  # config.PPO_LR_END_FACTOR
    total_iters=2000  # config.TRAIN_NUM_EPISODES
)
```

**동작:**
- 에피소드 0: `lr = 5e-5 * 1.0 = 5e-5`
- 에피소드 1000: `lr = 5e-5 * 0.505 ≈ 2.525e-5`
- 에피소드 2000: `lr = 5e-5 * 0.01 = 5e-7`

---

## 7. 하이퍼파라미터

### 7.1 PPO 하이퍼파라미터 (`config.py`)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `PPO_GAMMA` | 0.99 | 할인율 (미래 보상 가중치) |
| `PPO_LAMBDA` | 0.95 | GAE 람다 (bias-variance trade-off) |
| `PPO_EPS_CLIP` | 0.15 | PPO 클리핑 범위 |
| `PPO_K_EPOCHS` | 4 | 업데이트 반복 횟수 |
| `PPO_ENTROPY_COEF` | 0.01 | 엔트로피 계수 (탐험률) |
| `PPO_ENTROPY_DECAY` | 0.9996 | 엔트로피 감소율 |
| `PPO_ENTROPY_MIN` | 0.005 | 엔트로피 최소값 |
| `PPO_LEARNING_RATE` | 5e-5 | 학습률 |
| `PPO_LR_END_FACTOR` | 0.01 | 학습 종료 시 학습률 비율 |

### 7.2 네트워크 아키텍처 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `NETWORK_HIDDEN_DIM` | 128 | 은닉층 차원 |
| `NETWORK_NUM_LAYERS` | 2 | xLSTM 레이어 개수 |
| `NETWORK_DROPOUT` | 0.1 | Dropout 비율 |
| `NETWORK_ATTENTION_HEADS` | 4 | Multi-Head Attention 헤드 개수 |
| `NETWORK_INFO_ENCODER_DIM` | 64 | Info Encoder 출력 차원 |
| `NETWORK_SHARED_TRUNK_DIM1` | 256 | Shared Trunk 첫 번째 레이어 |
| `NETWORK_SHARED_TRUNK_DIM2` | 128 | Shared Trunk 두 번째 레이어 |
| `NETWORK_ACTOR_HEAD_DIM` | 64 | Actor Head 은닉층 |
| `NETWORK_CRITIC_HEAD_DIM` | 32 | Critic Head 은닉층 |

### 7.3 학습 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `TRAIN_ACTION_DIM` | 4 | 행동 차원 (HOLD, LONG, SHORT, EXIT) |
| `TRAIN_BATCH_SIZE` | 1024 | 배치 크기 |
| `TRAIN_SAMPLE_SIZE` | 50000 | 스케일러 학습용 샘플 크기 |
| `TRAIN_SPLIT` | 0.7 | 학습 데이터 비율 (70%) |
| `TRAIN_NUM_EPISODES` | 2000 | 총 에피소드 수 |
| `TRAIN_MAX_STEPS_PER_EPISODE` | 480 | 에피소드당 최대 스텝 |
| `TRAIN_SAVE_INTERVAL` | 50 | 모델 저장 간격 |

### 7.4 보상 함수 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `TRANSACTION_COST` | 0.0015 | 거래 비용 (0.15%) |
| `STOP_LOSS_THRESHOLD` | -0.05 | 강제 손절 임계값 (-5%) |

---

## 8. 데이터 저장 및 로드

### 8.1 저장 파일

**모델 파일:**
- `data/ppo_model_best.pth`: 최고 성능 모델
- `data/ppo_model_last.pth`: 최신 모델

**스케일러 파일:**
- `data/ppo_model_best_scaler.pkl`: 최고 성능 스케일러
- `data/ppo_model_last_scaler.pkl`: 최신 스케일러

**데이터 파일:**
- `data/training_features.csv`: 피처 엔지니어링 결과 (29개 피처)
- `data/cached_strategies.csv`: 전략 신호 캐시 (12개 전략)

### 8.2 모델 체크포인트 구조

```python
{
    'model_state_dict': {...},  # xLSTMActorCritic 가중치
    'optimizer_state_dict': {...}  # Adam optimizer 상태
}
```

---

## 9. 주요 개선 사항 (v2.0)

### 9.1 4-Action Strategy
- **기존**: 3-Action (HOLD, LONG, SHORT)
- **변경**: 4-Action (HOLD, LONG, SHORT, EXIT)
- **효과**: 명시적 청산과 관망을 분리하여 학습 명확성 향상

### 9.2 보상 함수 개선
- **HOLD 보너스**: 관망도 전략임을 인지
- **Position Holding Reward**: 추세를 길게 타도록 유도
- **Switching Penalty**: 잦은 포지션 변경 방지
- **EXIT Rewards**: Realized PnL 중심의 강력한 피드백

### 9.3 Value Clipping
- **Critic Loss Clipping**: 큰 보상 환경에서도 안정적 학습
- **보수적 업데이트**: `max(loss_v1, loss_v2)` 사용

### 9.4 물리적 제약 제거
- **쿨다운 제거**: AI가 자유롭게 행동
- **최소 보유 시간 제거**: 즉각적인 손절 가능
- **리워드 기반 학습**: 시스템 제약 대신 리워드로 자제 학습

---

## 10. 성능 최적화

### 10.1 피처 엔지니어링 최적화
- 전체 데이터에 대해 한 번만 피처 생성
- 스케일러 학습 시 샘플링으로 메모리 효율화

### 10.2 전략 신호 캐싱
- `cached_strategies.csv`로 전략 계산 결과 재사용
- 병렬 처리 지원 (joblib)

### 10.3 Gradient Checkpointing
- `NETWORK_USE_CHECKPOINTing`: 메모리 절약 (현재 False)

---

## 11. 참고사항

### 11.1 주의사항
- **모델 호환성**: 3-Action 모델은 4-Action과 호환되지 않음
- **학습 재시작**: 4-Action으로 변경 시 기존 모델 삭제 필요
- **데이터 준비**: `training_features.csv`와 `cached_strategies.csv` 필요

### 11.2 디버깅 팁
- 로그 파일: `logs/train_ppo.log`
- 스케일러 체크: `scaler_fitted` 플래그 확인
- 메모리 상태: `len(agent.memory)` 확인

---

**문서 버전**: v2.0 (4-Action Strategy)  
**최종 업데이트**: 2026-01-23
