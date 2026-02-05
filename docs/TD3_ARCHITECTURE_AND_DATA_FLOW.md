# TD3 모델 아키텍처 및 데이터 흐름 명세

## 1. 개요

TD3(Twin Delayed DDPG)는 **연속 행동 공간**을 사용하는 강화학습 알고리즘으로, 본 프로젝트에서는 스캘핑 환경의 **연속 신호(-1.0 ~ 1.0)**를 출력하고, 이를 이산 행동(0: Hold, 1: Buy, 2: Sell)으로 변환하여 기존 `TradingEnvironment`와 연동한다.

- **Actor**: Position-Aware Gating을 적용한 정책 네트워크 (포지션 보유 시 행동 억제)
- **Critic**: Twin Q-Network (Q1, Q2)로 불확실성 |Q1 - Q2|를 탐험 노이즈 조절에 활용
- **공통 백본**: MacroHFT의 `SharedBackbone`(시계열) + `StrategyInteractionLayer`(전략 12차원) 재사용

---

## 2. 디렉터리 및 파일 구조

```
TD3/
├── __init__.py          # PositionAwareActor, TD3Critic, TD3Agent, ReplayBuffer export
├── td3_network.py       # PositionAwareActor, TD3Critic 정의
├── td3_agent.py         # ReplayBuffer, TD3Agent (학습/추론/저장)
└── train_td3.py         # TD3Trainer: 데이터 로드, 환경 연동, 학습 루프

common/
├── config.py            # TD3_* 하이퍼파라미터, LOOKBACK 등
├── preprocess.py        # DataPreprocessor (TradingEnvironment에서 사용)
└── ...

macroHFT/
├── xlstm_network.py     # SharedBackbone, StrategyInteractionLayer (TD3에서 import)
└── trading_env.py       # TradingEnvironment (관측/보상)
```

---

## 3. 설정 (common/config.py)

### 3.1 TD3 전용 하이퍼파라미터

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `TD3_LEARNING_RATE` | 3e-4 | Actor/Critic 공통 학습률 |
| `TD3_GAMMA` | 0.99 | 할인율 |
| `TD3_TAU` | 0.005 | Target 네트워크 Soft Update 비율 |
| `TD3_POLICY_NOISE` | 0.2 | Target Policy Smoothing 노이즈 |
| `TD3_NOISE_CLIP` | 0.5 | 노이즈 클리핑 범위 |
| `TD3_EXPLORE_NOISE` | 0.1 | 탐험용 노이즈 (실전 행동 선택 시) |
| `TD3_POLICY_FREQ` | 2 | Actor 업데이트 빈도 (Critic 2회당 1회) |
| `TD3_BATCH_SIZE` | 256 | 미니배치 크기 |
| `TD3_BUFFER_SIZE` | 100000 | 리플레이 버퍼 최대 크기 |
| `TD3_WARMUP_STEPS` | 5000 | 랜덤 행동으로 버퍼 채우는 스텝 수 |

### 3.2 공통 네트워크 파라미터 (TD3에서 사용)

- `LOOKBACK`: 60 (시퀀스 길이)
- `NETWORK_HIDDEN_DIM`: 256
- `NETWORK_NUM_LAYERS`: 2
- `NETWORK_DROPOUT`: 0.1

---

## 4. 네트워크 아키텍처

### 4.1 공통 백본 (macroHFT.xlstm_network)

#### SharedBackbone

- **입력**: `x` (Batch, Lookback, input_dim)
- **구조**:
  - `input_proj`: Linear → LayerNorm → GELU → Dropout → (B, L, hidden_dim)
  - `layers`: ModuleList of `ResidualGRU` (GRU + residual + LayerNorm)
  - Temporal Attention: `context = softmax(scores) @ x` → (B, hidden_dim)
- **출력**: `context` (B, hidden_dim), `next_states` (GRU hidden states)

#### StrategyInteractionLayer

- **입력**: `strategies` (B, 12)
- **구조**: Linear(12 → 12×32) → reshape (B, 12, 32) → Self-Attention (Q,K,V) → out_proj → 64차원
- **출력**: (B, 64)

---

### 4.2 PositionAwareActor (TD3/td3_network.py)

**역할**: 포지션 상태에 따라 행동의 적극성을 조절하는 Gating Actor.

| 단계 | 연산 | 입력 | 출력 |
|------|------|------|------|
| 1 | Backbone | x (B, L, state_dim) | context (B, hidden_dim) |
| 2 | Info 분해 | info (B, 15) | pos_val (B,1), strategies (B,12), pos_meta (B,2) |
| 3 | Position Gate | pos_context = [pos_val, pos_meta] (B, 3) | gate (B, 1), Sigmoid |
| 4 | Strategy | strategies (B, 12) | strat_features (B, 64) |
| 5 | Fusion | [context, strat_features, pos_context] | fused (B, hidden_dim) |
| 6 | Head | fused | raw_action (B, action_dim), Tanh |
| 7 | Gating | raw_action, gate | scaled_action = raw_action * (0.3 + 0.7 * gate) |

- **fusion_dim** = hidden_dim + 64 + 3
- **출력**: `(scaled_action, next_states, gate.mean())`  
  - `scaled_action`: (B, 1), 범위 [-1, 1]  
  - gate가 낮으면(포지션 보유 등) 행동이 0 근처로 눌림.

---

### 4.3 TD3Critic (TD3/td3_network.py)

**역할**: Twin Critic (Q1, Q2). Actor와 동일한 백본·전략·pos_context로 state_repr 구성 후, (state_repr, action)으로 Q1, Q2 각각 계산.

| 단계 | 연산 | 비고 |
|------|------|------|
| 1 | Backbone | context (B, hidden_dim) |
| 2 | Info → pos_context, strat_features | Actor와 동일 |
| 3 | state_repr | [context, strat_features, pos_context] (B, fusion_dim) |
| 4 | Q1 | Linear(fusion_dim + action_dim → hidden_dim) → LayerNorm → GELU → Linear → 1 |
| 5 | Q2 | 동일 구조 (독립 파라미터) |

- **출력**: `(q1, q2)` 각 (B, 1)

---

## 5. ReplayBuffer (TD3/td3_agent.py)

### 5.1 저장 형식

| 필드 | shape | dtype |
|------|--------|--------|
| state_seq | (max_size, LOOKBACK, state_dim) | float32 |
| state_info | (max_size, info_dim) | float32 |
| action | (max_size, action_dim) | float32 |
| reward | (max_size, 1) | float32 |
| next_state_seq | (max_size, LOOKBACK, state_dim) | float32 |
| next_state_info | (max_size, info_dim) | float32 |
| not_done | (max_size, 1) | float32 |

- `add(state, action, reward, next_state, done)`: state/next_state는 `(obs_seq, obs_info)` 튜플. 배치 차원(1)이 있으면 squeeze 후 저장.
- `sample(batch_size)`: 7개 텐서를 device로 반환.

---

## 6. TD3Agent (TD3/td3_agent.py)

### 6.1 구성 요소

- **Actor / Actor Target**: PositionAwareActor
- **Critic / Critic Target**: TD3Critic
- **ReplayBuffer**: 위와 동일
- **Cooldown**: `position_cooldown`, `min_hold_steps`, `last_position` — 포지션 변경 시 일정 스텝 Hold 강제

### 6.2 select_action(state, noise, current_position)

1. **차원 정규화**: state[0] 3D → 2D, state[1] 2D → 1D 후 배치 차원 1 추가.
2. **Actor 추론**: `action, _, gate_mean = actor(obs_seq, obs_info)`.
3. **Cooldown**: `position_cooldown > 0`이면 `(np.array([0.0]), gate_mean.item())` 반환.
4. **리스크 기반 노이즈**: `noise > 0`이면 `_estimate_uncertainty(obs_seq, obs_info, action)`로 `|Q1-Q2|/|Q1|` 계산 후 `adaptive_noise = noise * (1 - min(risk, 0.8))` 적용.
5. **반환**: `(np.clip(action, -1, 1), gate_mean.item())`.

### 6.3 _estimate_uncertainty(obs_seq, obs_info, action_np)

- Critic으로 (obs_seq, obs_info, action) forward → Q1, Q2
- `uncertainty = |Q1 - Q2| / (|Q1| + 1e-6)` (스칼라 반환)

### 6.4 enforce_cooldown(action_val, current_position)

- `current_position != last_position`이면 `position_cooldown = min_hold_steps`, `last_position` 갱신.
- `position_cooldown > 0`이면 `0.0` 반환, 아니면 `action_val` 그대로 반환.

### 6.5 train(batch_size)

1. **샘플**: (s_seq, s_info, action, ns_seq, ns_info, reward, not_done)
2. **Target Policy Smoothing**: next_action = actor_target(ns) + clipped_noise
3. **Target Q**: target_Q = reward + γ * min(Q1_target(ns, next_action), Q2_target(ns, next_action)) * not_done
4. **Critic 업데이트**: MSE(current_Q1, target_Q) + MSE(current_Q2, target_Q)
5. **Delayed Actor** (total_it % policy_freq == 0): actor_loss = -Q1(s, π(s)).mean(), Soft Update (τ) for target networks

### 6.6 save / load

- **save(filename)**: `{base}_actor.pth`, `{base}_critic.pth` 저장.
- **load(filename)**: 위 두 개 로드 후 actor_target, critic_target을 deepcopy로 동기화.

---

## 7. 학습 루프 (TD3Trainer.train, train_td3.py)

### 7.1 데이터 및 환경

- **데이터**: `data/training_features.csv` → `DataCollector.eth_data` (ffill + dropna, bfill 미사용).
- **환경**: `TradingEnvironment(data_collector, strategies)` — MacroHFT와 동일한 12개 전략·전처리·캐시.
- **관측**: `env.get_observation(position_info, current_index)` → `(obs_seq, obs_info)`  
  - obs_seq: (1, LOOKBACK, state_dim), obs_info: (1, 15).

### 7.2 에피소드 루프

- **시작 인덱스**: `start_idx` = [LOOKBACK+100, train_end_idx - max_steps - 100) 구간에서 무작위.
- **에피소드당**: `prev_unrealized_pnl = 0.0`, `trade_count = 0`.

### 7.3 스텝별 흐름

| 순서 | 항목 | Warmup | 실전 |
|------|------|--------|------|
| 1 | 행동 | 랜덤 uniform(-1,1) | select_action(state, noise, current_position) |
| 2 | Cooldown | 미적용 | enforce_cooldown(action_val, current_position) |
| 3 | 이산 행동 | threshold 0.3으로 직접 매핑 (거래 횟수 제한 없음) | continuous_to_discrete(action_val, position, trade_count) (30회 제한) |

### 7.4 보상 (Delta PnL)

- **unrealized_pnl**: 현재 포지션 기준 총 미실현 수익률 (LONG/SHORT/None).
- **step_pnl_delta** (이번 스텝만의 변동분):
  - 포지션 없음: `trade_done`이면 `realized_pnl - prev_unrealized_pnl`, 아니면 0.0. 이후 `prev_unrealized_pnl = 0`.
  - 포지션 있음: `step_pnl_delta = unrealized_pnl - prev_unrealized_pnl`, `prev_unrealized_pnl = unrealized_pnl`.
- **reward** = `env.calculate_reward(step_pnl=step_pnl_delta, realized_pnl, trade_done, ...)`.

### 7.5 버퍼 및 학습

- 매 스텝: `replay_buffer.add(state, action_val, reward, next_state, done)` (연속 action_val 저장).
- Warmup 이후: `agent.train(batch_size=TD3_BATCH_SIZE)` 호출.

### 7.6 저장

- **매 에피소드**: `data/last_td3_model` (덮어쓰기).
- **최고 보상 갱신 시**: `data/best_td3_model` + 로그.
- **50 에피소드마다**: `data/td3_model_{ep}`.

---

## 8. continuous_to_discrete (train_td3.py)

- **입력**: action_val ∈ [-1, 1], current_position ∈ {None, 'LONG', 'SHORT'}, trade_count.
- **임계값**: 0.3.
- **규칙**:
  - action_val > 0.3 → Long 신호: 이미 LONG이면 0, SHORT/None이면 1 (단, None이고 trade_count ≥ 30이면 0).
  - action_val < -0.3 → Short 신호: 이미 SHORT이면 0, LONG/None이면 2 (단, None이고 trade_count ≥ 30이면 0).
  - 그 외 → 0 (Hold).

---

## 9. 데이터 흐름 요약

```
[데이터]
  data/training_features.csv
       ↓ ffill(), dropna()
  DataCollector.eth_data

[에피소드 시작]
  start_idx 랜덤 → env.get_observation(pos_info, start_idx)
       ↓
  state = (obs_seq [1,L,state_dim], obs_info [1,15])

[스텝 루프]
  state → (Warmup: 랜덤 / 실전: agent.select_action) → action_val ∈ [-1,1]
       ↓
  enforce_cooldown → continuous_to_discrete → env_action ∈ {0,1,2}
       ↓
  포지션/가격 시뮬레이션 → unrealized_pnl, realized_pnl, trade_done
       ↓
  Delta PnL: step_pnl_delta = Δ(unrealized/realized)
       ↓
  reward = env.calculate_reward(step_pnl_delta, ...)
       ↓
  next_state = env.get_observation(...)
       ↓
  replay_buffer.add(state, action_val, reward, next_state, done)
       ↓
  (Warmup 이후) replay_buffer.sample(B) → TD3Agent.train(B)
       ↓
  Critic: MSE(Q, target_Q); (2회당 1회) Actor: -Q1(s,π(s)); Soft Update target
```

---

## 10. 입출력 shape 정리

| 단계 | 변수 | Shape |
|------|------|--------|
| 관측 | obs_seq | (1, LOOKBACK, state_dim) |
| 관측 | obs_info | (1, 15) |
| Actor 출력 | action | (1,) → numpy [-1, 1] |
| 버퍼 샘플 | s_seq, ns_seq | (B, LOOKBACK, state_dim) |
| 버퍼 샘플 | s_info, ns_info | (B, 15) |
| 버퍼 샘플 | action | (B, 1) |
| Critic 출력 | Q1, Q2 | (B, 1) |

---

## 11. 참고

- **MacroHFT와의 차이**: TD3는 연속 행동 + Off-Policy (ReplayBuffer) + Twin Critic + Target Policy Smoothing; MacroHFT는 이산 행동 + On-Policy (PPO) + Router/Experts.
- **공통 사용**: `common.config`, `common.preprocess`, `macroHFT.trading_env`, `macroHFT.xlstm_network`(SharedBackbone, StrategyInteractionLayer).
