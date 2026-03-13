# TD3 모델 명세 및 데이터 흐름

## 1. 개요

- **목적**: 연속 포지션 크기(Continuous Position) 기반 스캘핑 에이전트.
- **알고리즘**: TD3 (Twin Delayed DDPG) + CQL (Conservative Q-Learning).
- **입력**: 시계열 피처(60봉) + 정보 벡터(16차원). **출력**: 포지션 크기 `[-1, 1]` (연속값).

---

## 2. 설정 (config)

| 항목 | 값 | 설명 |
|------|-----|------|
| `LOOKBACK` | 60 | 시계열 윈도우(봉 수) |
| `TD3_INFO_DIM` | 16 | 정보 벡터 차원 |
| `TD3_LEARNING_RATE` | 3e-5 | Actor/Critic 학습률 |
| `TD3_GAMMA` | 0.99 | 할인율 |
| `TD3_TAU` | 0.005 | Target 네트워크 soft update 비율 |
| `TD3_POLICY_NOISE` | 0.1 | Target Policy 스무딩 노이즈 |
| `TD3_NOISE_CLIP` | 0.5 | 노이즈 클리핑 |
| `TD3_POLICY_FREQ` | 2 | Actor 업데이트 주기(2회 Critic당 1회 Actor) |
| `TD3_BATCH_SIZE` | 256 | 미니배치 크기 |
| `TD3_BUFFER_SIZE` | 100000 | Replay Buffer 크기 |
| `TD3_WARMUP_STEPS` | 5000 | 랜덤 행동 웜업 스텝 |
| `REWARD_MULTIPLIER` | 100.0 | 보상 스케일 |
| `TRANSACTION_COST` | 0.0005 | 거래 비용(0.05%) |
| `TRAIN_SPLIT` / `VAL_SPLIT` / `TEST_SPLIT` | 0.7 / 0.15 / 0.15 | 데이터 분할 |

---

## 3. 상태(State) 명세

상태는 **튜플 `(obs_seq, obs_info)`** 로 구성된다.

### 3.1 obs_seq (시계열 피처)

- **Shape**: `(1, LOOKBACK, state_dim)` → 실질적으로 `(1, 60, 29)`.
- **state_dim**: `TradingEnvironment.get_state_dim()` 반환값. `cached_features`의 컬럼 수 = **29**.
- **생성**: `trading_env.precompute_data()` 에서 **Rolling Z-Score** 적용.
  - `rolling(window=LOOKBACK, min_periods=1)` 로 각 시점에서 **과거 60봉**만 사용해 mean/std 계산 후 `(x - mean) / std`.
  - 미래 데이터 참조 없음.

**피처 컬럼 (29개)**  
`log_return`, `roll_return_6`, `atr_ratio`, `bb_width`, `bb_pos`, `rsi`, `macd_hist`, `hma_ratio`, `cci`, `rvol`, `taker_ratio`, `cvd_change`, `mfi`, `cmf`, `vwap_dist`, `wick_upper`, `wick_lower`, `range_pos`, `swing_break`, `chop`, `btc_return`, `btc_rsi`, `btc_corr`, `btc_vol`, `eth_btc_ratio`, `rsi_15m`, `trend_15m`, `rsi_1h`, `trend_1h`.

### 3.2 obs_info (정보 벡터, 16차원)

- **Shape**: `(1, TD3_INFO_DIM)` = `(1, 16)`.
- **구성** (train_td3에서 `_augment_info`로 volatility 추가):
  - `[0:1]`: 포지션 값 (현재 포지션 크기, -1~1).
  - `[1:13]`: 전략 점수 12개 (`strategy_0` ~ `strategy_11`).
  - `[13:15]`: 포지션 메타 (미실현 PnL×10, hold_ratio 등).
  - `[15:16]`: **volatility_20tick** (추가 변동성 피처).

---

## 4. 행동(Action) 명세

- **차원**: 1 (연속 스칼라).
- **범위**: `[-1, 1]` (Tanh 출력 후 게이팅 적용).
- **의미**:
  - `> 0`: 롱 포지션 크기 (예: 0.7 → 70% 롱).
  - `< 0`: 숏 포지션 크기 (예: -0.5 → 50% 숏).
  - `0`: 무포지션.

### 4.1 학습 시 포지션 변경 규칙 (Flip-Lock)

- **Deadzone**: `|action| > 0.3` 일 때만 포지션 반영, 그 외는 0으로 해석.
- **허용 변경**:
  - **Opening**: 무포지션 → 롱/숏.
  - **Flipping**: 롱↔숏 반대 전환.
  - **Strength Change**: `|target_pos_size - current_pos_size| > 0.4`.
- 위 조건이 아니면 `target_pos_size = current_pos_size` (유지).

### 4.2 평가 시 (evaluate_td3.py)

- **Deadzone**: `DEADZONE = 0.5` → `|action| >= 0.5` 일 때만 포지션 반영.
- **Strength Change 임계값**: `STRENGTH_CHANGE_THRESHOLD = 0.4`.
- **거래 비용**: `TRANSACTION_COST` 동일 적용.

---

## 5. 보상(Reward) 명세

- **스텝 PnL**: `step_pnl = current_pos_size * price_return - trade_cost`.
- **Risk-Adjusted (학습)**:
  - 최근 20스텝 `step_pnl` 의 표준편차로 `risk_penalty` 계산.
  - `step_pnl > 0` 일 때만 `adjusted_pnl = step_pnl - risk_penalty` 적용.
- **최종 보상**: `reward = adjusted_pnl * REWARD_MULTIPLIER` (스칼라).

---

## 6. 네트워크 구조

### 6.1 PositionAwareActor (td3_network.py)

- **SharedBackbone**: 입력 `(B, 60, 29)` → Linear → ResidualGRU×2 → Temporal Attention → `context` (B, hidden_dim).
- **StrategyInteractionLayer**: Info의 전략 12차원 → Self-Attention → 64차원.
- **RiskAwareGate**: `pos_context`(포지션값, PnL 등) + `volatility`(info[:, 15:16]) → Sigmoid 게이트. 손실 -2% 미만 시 게이트 부드럽게 감소.
- **Fusion**: `context` + 전략 피처 + `pos_context` → fusion_proj → head(Tanh) → `raw_action`.
- **게이팅**: `magnitude = |raw_action|`, `direction = sign(raw_action)`. `scaled_magnitude = magnitude * (0.1 + 0.9 * gate)`. 출력 `direction * scaled_magnitude`.

### 6.2 TD3Critic

- **SharedBackbone** + **StrategyInteractionLayer** 동일.
- **volatility_embed**: info의 volatility(1차원) → 16 → hidden_dim, context에 가산.
- **Q1, Q2**: state_repr + action → 각각 별도 MLP → Q1, Q2. **클리핑**: `clamp(q1, -1, 1)`, `clamp(q2, -1, 1)`.

### 6.3 ReplayBuffer (td3_agent.py)

- **저장 단위**: (state, action, reward, next_state, done).
- **state / next_state**: 각각 `(state_seq, state_info)` 형태. `state_seq` (lookback, state_dim), `state_info` (info_dim).

---

## 7. 학습 데이터 흐름 (train_td3.py)

```
1. 데이터 로드
   data/training_features.csv → DataCollector.eth_data
   volatility_20tick 없으면 add_volatility_feature() 적용
   cached_strategies.csv 있으면 strategy_* 컬럼 병합

2. 환경
   TradingEnvironment(data_collector, strategies)
   env.precompute_data() → 전체 시리즈에 Rolling Z-Score → cached_features (T, 29), cached_strategies (T, 12)

3. 에피소드
   - start_idx: [LOOKBACK+100, train_end_idx - max_steps - 100] 균일 샘플
   - 초기 포지션: 50% 0, 25% 0.5, 25% -0.5 (편향 방지)
   - state = env.get_observation(pos_info, start_idx) → (obs_seq, obs_info)
   - state = (state[0], _augment_info(state[1], start_idx))  # info 15→16

4. 스텝 루프
   - Warmup: action = uniform(-1, 1)
   - 그 외: agent.select_action(state, noise=TD3_EXPLORE_NOISE) → (action, gate, risk)
   - target_pos_size = action (|action|>0.3만 반영), Flip-Lock 규칙 적용
   - trade_cost = |trade_amount| * TRANSACTION_COST
   - step_pnl = current_pos_size * price_return - trade_cost
   - pnl_history에 step_pnl 누적 → risk_penalty = std(pnl_history)*0.5 (step_pnl>0일 때만 차감)
   - reward = adjusted_pnl * REWARD_MULTIPLIER
   - next_state = env.get_observation(next_pos_info, next_idx); next_state도 _augment_info 적용
   - replay_buffer.add(state, action_for_buffer, reward, next_state, done)

5. 학습 (Warmup 이후, 매 스텝)
   - replay_buffer.sample(TD3_BATCH_SIZE)
   - Critic: target_Q = r + gamma * min(Q1_target(s',a'), Q2_target(s',a')); MSE + CQL Loss
   - CQL: 랜덤 액션에 대한 Q를 낮추는 항 (logsumexp(Q_rand) - Q_current)
   - Actor: policy_freq마다 -Q1.mean() 최대화, grad_clip 1.0
   - Target soft update: tau=0.005
```

---

## 8. 평가 데이터 흐름 (evaluate_td3.py)

```
1. 데이터 로드
   training_features.csv + volatility_20tick + cached_strategies (동일)

2. 구간
   mode='val' | 'test' | 'full' 에 따라 start_idx, end_idx 설정

3. Online Z-Score (공변량 이동 방지)
   _precompute_online_features_for_eval():
   - 평가 구간만 슬라이스: [start_idx - lookback, end_idx)
   - 해당 슬라이스에 대해 Rolling(window=60) Z-Score → _online_features, _online_base_idx

4. 스텝 루프
   - state = _get_observation_online(idx, pos_info)  # 캐시된 온라인 Z-Score로 (obs_seq, obs_info) 생성
   - state = (state[0], _augment_info(state[1], idx))
   - action = agent.select_action(state, noise=0)
   - target_pos_size: |action| >= DEADZONE(0.5)만 반영, Flip-Lock 및 STRENGTH_CHANGE_THRESHOLD(0.4) 적용
   - 거래 비용 적용 후 잔고·포지션 이력 누적
```

---

## 9. 모델 저장/로드

- **저장 경로**: `data/td3/<run_timestamp>/`
  - `last_td3_model_actor.pth`, `last_td3_model_critic.pth`
  - `best_td3_model_actor.pth`, `best_td3_model_critic.pth`
- **로드**: `agent.load(base_path)` → Actor/Critic state_dict 로드 후 target 복사.

---

## 10. 요약

| 구분 | 내용 |
|------|------|
| 상태 | (60×29 시계열 Z-Score, 16차원 정보) |
| 행동 | 연속 1차원 [-1, 1], Deadzone·Flip-Lock 적용 |
| 보상 | Risk-Adjusted step_pnl × REWARD_MULTIPLIER |
| 학습 | TD3 + CQL, ReplayBuffer, Target soft update |
| 평가 | 평가 구간만 Rolling Z-Score로 전처리 후 동일 규칙 적용 |
