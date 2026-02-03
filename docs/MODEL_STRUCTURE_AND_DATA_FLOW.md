# PPO 트레이딩 모델: 구조 및 데이터 흐름 명세

> **model** 폴더 코드 기준으로 정리한 **모델 구조**와 **데이터 흐름** 상세 명세입니다.  
> (Genius Version 2.0: Causal CNN, Shared Backbone, Strategy Gating, Logits + State Passing 반영)

---

## 1. 전체 구조 개요

```
[원시/피처 데이터] → [전처리·전략 점수] → [TradingEnvironment] → [PPOAgent (계층형)]
       │                     │                      │                        │
       │                     │                      │                        ├── 무포지션 → Entry Agent (XLSTM) → 0=Wait, 1=Long, 2=Short
       │                     │                      │                        └── 포지션 있음 → Exit Agent (XLSTM) → 0=Hold, 1→Global 3=Exit
       │                     │                      │
       │                     │                      ├── get_observation() → (obs_seq, obs_info)
       │                     │                      └── calculate_reward() [DSR + Action Dampening]
       │                     │
       └── training_features.csv, cached_strategies.csv
       └── DataPreprocessor (Rolling Norm), 12개 전략 점수
```

- **계층형 PPO**: 포지션 유무에 따라 **Entry 에이전트**(action_dim=3)와 **Exit 에이전트**(action_dim=2)가 분리되어 동작.
- **Global Action 4개**: `0=Wait/Hold`, `1=Long`, `2=Short`, `3=Exit`.
- **네트워크**: 단일 **SharedBackbone** (Causal CNN + sLSTM) + **StrategyGating** + Head 3개(actor, critic, aux). **Logits** 및 **next_states** 반환, 에이전트에서 temperature·Categorical 처리.

---

## 2. 데이터 소스 및 전처리

### 2.1 데이터 소스

| 구분 | 경로/모듈 | 설명 |
|------|------------|------|
| **시계열 피처** | `data/training_features.csv` | ETH/BTC OHLCV 기반 **29개 피처** (feature_engineering 등으로 생성) |
| **전략 점수** | `data/cached_strategies.csv` 또는 재계산 | 12개 전략의 `strategy_0` ~ `strategy_11` (봉별 LONG/SHORT/NEUTRAL → confidence 스칼라) |
| **학습 시 병합** | `train_ppo.py` `_load_features()` | `training_features.csv` 로드 후 `cached_strategies.csv`의 strategy 컬럼 병합 → `DataCollector.eth_data` |

### 2.2 전처리: DataPreprocessor (Rolling Normalization)

**파일**: `model/preprocess.py`

- **방식**: **Rolling (Instance) Normalization**. 전역 Z-Score가 아닌 **현재 윈도우(Lookback)** 내 통계로 정규화.
- **목적**: 시계열 비정상성 완화, 로컬 패턴에 집중.
- **`transform(data)`**:
  - 입력: `(seq_len, feature_dim)` (예: 60×29).
  - `mean = np.mean(data, axis=0)`, `std = np.std(data, axis=0)`.
  - `std[std < epsilon] = 1.0` (epsilon=1e-8) → NaN/제로 나눗셈 방지.
  - 반환: `(data - mean) / std`.
- **`fit()` / `save()` / `load()`**: 호환성만 유지, Rolling 방식이라 저장할 통계 없음.

### 2.3 Lookback 및 차원

- **config.LOOKBACK** = 60: 시계열 관측 구간(봉 개수).
- **state_dim** = 29: `trading_env.get_state_dim()` 및 시계열 피처 개수(`target_cols` 개수).

---

## 3. 관측(Observation) 구성

**위치**: `TradingEnvironment.get_observation(position_info, current_index)`  
**반환**: `(obs_seq, obs_info)` 튜플. 실패 시 `None`.

### 3.1 obs_seq (시계열 상태)

| 항목 | 값 |
|------|-----|
| **Shape** | `(1, lookback, state_dim)` = `(1, 60, 29)` |
| **생성 흐름** | `curr_idx - lookback ~ curr_idx` 구간의 `target_cols` 슬라이스 → `preprocessor.transform()` (Rolling Norm) → `torch.FloatTensor` → `unsqueeze(0)` |
| **target_cols (29개)** | log_return, roll_return_6, atr_ratio, bb_width, bb_pos, rsi, macd_hist, hma_ratio, cci, rvol, taker_ratio, cvd_change, mfi, cmf, vwap_dist, wick_upper, wick_lower, range_pos, swing_break, chop, btc_return, btc_rsi, btc_corr, btc_vol, eth_btc_ratio, rsi_15m, trend_15m, rsi_1h, trend_1h |

### 3.2 obs_info (정보 벡터, info_dim=15)

| 항목 | 값 |
|------|-----|
| **Shape** | `(1, 15)` |
| **앞 12개** | 전략 점수 `strategy_0` ~ `strategy_11` (현재 봉 기준 스칼라) |
| **뒤 3개** | **포지션 정보** (position_info): `[position_flag, unrealized_pnl*10, holding_time_norm]` |
| | - position_flag: 1.0=Long, -1.0=Short, 0.0=없음 |
| | - unrealized_pnl*10: 미실현 손익 스케일 |
| | - holding_time_norm: 예) holding_time / max_steps |

---

## 4. 네트워크 구조 (xlstm_network.py)

### 4.1 CausalConv1d

- **역할**: 미래 시점 참조 방지 (Look-ahead Bias 제거).
- **구현**: `nn.Conv1d(..., padding=(kernel_size-1)*dilation)` 적용 후 **오른쪽(미래) 패딩 제거** → `out[:, :, :-self.padding]`.
- **입력**: `[B, C_in, L]` → **출력**: `[B, C_out, L]` (시퀀스 길이 L 유지).

### 4.2 StabilizedSLSTMCell

- **역할**: xLSTM의 sLSTM 셀 (Scalar LSTM, exponential gating).
- **상태**: `(h, c, n, m)` 4-tuple.
- **안정화**: `i_pre`, `f_pre`를 clamp(-20, 20) 후 exp; `output = c_t / (n_t + 1e-6)`, `h_t = o_t * output`; NaN 시 `nan_to_num(..., nan=0.0)`.

### 4.3 SharedBackbone (CNN + xLSTM)

- **역할**: 시계열 → **단일 context 벡터** + **next_states**. Actor/Critic **공유**.
- **CNN 블록** (Conv 출력 [B,C,L]에 맞춤):
  - CausalConv1d(input_dim → hidden_dim, k=3) → **BatchNorm1d(hidden_dim)** → ELU
  - CausalConv1d(hidden_dim → hidden_dim, k=3) → **BatchNorm1d(hidden_dim)** → ELU
- **입력**: `x` [B, L, input_dim] → permute(0,2,1) → [B, D, L] → cnn_block → permute(0,2,1) → [B, L, hidden_dim].
- **input_proj**: Linear → LayerNorm → Dropout.
- **LSTM**: num_layers개의 StabilizedSLSTMCell, 층마다 Residual(inp + h_next) + LayerNorm.
- **출력**: `context_vector = current_input[:, -1, :]` [B, hidden_dim], `next_states` (list of (h,c,n,m) per layer).

### 4.4 StrategyGating

- **역할**: Attention 대신 경량 **Gating**. Context가 각 전략 점수를 얼마나 신뢰할지 가중치로 반영.
- **연산**: `gates = sigmoid(Linear(context))` [B, num_strategies] → `weighted = strategy_scores * gates` → `strat_proj(weighted)` → [B, hidden_dim].

### 4.5 XLSTMNetwork

- **구성**: `backbone`(SharedBackbone) 1개 + `strat_gating`(StrategyGating) + `pos_enc`(Linear(3, hidden_dim)) + **actor_head**, **critic_head**, **aux_head**.
- **forward(x, info, states=None, temperature=1.0)**:
  1. `context, next_states = backbone(x, states)` (states 전달 시 시퀀스 간 기억 유지).
  2. info가 3차원이면 squeeze(1). `strategy_scores = info[:, :12]`, `pos_info = info[:, 12:]`.
  3. `strat_feat = strat_gating(context, strategy_scores)`, `pos_feat = gelu(pos_enc(pos_info))`.
  4. `combined = concat(context, strat_feat, pos_feat)` → [B, hidden_dim*3].
  5. `logits = actor_head(combined)`, `value = critic_head(combined)`, `aux_value = aux_head(combined)`.
- **반환**: **(logits, value, aux_value, next_states)**.  
  - **logits**: softmax 전 raw logits (temperature는 에이전트에서 적용).  
  - **next_states**: 다음 스텝에서 `states=` 인자로 넘겨 State Passing.

---

## 5. 계층형 PPO 에이전트 (ppo_agent.py)

### 5.1 CorePPOAgent (Entry / Exit 각 1개)

| 구분 | Entry Agent | Exit Agent |
|------|-------------|------------|
| **action_dim** | 3 (Wait=0, Long=1, Short=2) | 2 (Hold=0, Exit=1) |
| **호출 시점** | 무포지션일 때만 | 포지션 있을 때만 |
| **모델** | XLSTMNetwork(state_dim, 3, info_dim=15, ...) | XLSTMNetwork(state_dim, 2, info_dim=15, ...) |
| **model_target** | soft update (tau=0.995) | 동일 |

- **current_states**: LSTM 상태를 에피소드 내에서 유지. `reset_episode_states()`로 None 초기화.
- **select_action(state, action_mask=None)**:
  - state = (obs_seq, obs_info) → device 변환.
  - `logits, value, _, self.current_states = self.model(obs_seq, obs_info, states=self.current_states)`.
  - action_mask 있으면 `logits = logits.masked_fill(mask==0, -1e10)`.
  - `logits = logits / self.temperature` → `Categorical(logits=logits)` → sample → 반환 (action, log_prob, value).
- **put_data(transition)**: transition을 리스트에 append (7개 또는 8개 요소 호환).
- **train_net(episode)**:
  - 버퍼에서 (s, a, r, next_s, prob_a, done, val[, aux_target]) 수집.
  - 8개일 때 aux_target 사용, 7개일 때 aux_target=0.0.
  - GAE: `model(s_seq, s_info, states=None)`, `model_target(next_s_seq, next_s_info, states=None)` 로 v, next_v 계산 → td_target, advantage.
  - PPO 루프: `curr_logits, curr_v, curr_aux, _ = model(s_seq, s_info, states=None)` → `Categorical(logits=curr_logits/self.temperature)` → policy loss, value loss, entropy, **aux_loss = MSE(curr_aux, aux_target)** (가중치 0.5).
  - 학습 시에는 **states=None** (Truncated BPTT).

### 5.2 PPOAgent (계층 라우팅)

- **select_action(state)**: obs_info의 포지션 플래그(abs(pos_flag)>0.1)로 포지션 여부 판단 → 무포지션이면 entry_agent, 있으면 exit_agent 호출. Exit의 action=1 → Global 3(Exit).
- **put_data(transition)**: 포지션 플래그로 Entry/Exit 구분. Entry는 global_a in {0,1,2}일 때만 entry_agent에 (s, global_a, ...). Exit은 global_a in {0,3}일 때 local_a(0/1)로 exit_agent에 저장.
- **train_net(episode)**: entry_agent.train_net(), exit_agent.train_net() 호출 후 두 loss 평균 반환.

---

## 6. 보상 함수 (TradingEnvironment.calculate_reward)

**위치**: `model/trading_env.py`

- **시그니처**: `calculate_reward(step_pnl, realized_pnl, trade_done, holding_time=0, action=0, prev_position=None, current_position=None, agent_type='ENTRY')`.
- **구성**:
  1. **Action Dampening**: `action in [1,2,3]` (진입·청산)일 때 **trade_penalty = -0.5**.
  2. **Differential Sharpe Ratio (DSR)**: `r_t = step_pnl`, `std_dev = sqrt(max(B - A², 1e-4))`, `dsr_reward = (r_t * 100) / std_dev`.  
     그다음 A, B를 EMA 갱신: `A = (1-eta)*A + eta*r_t`, `B = (1-eta)*B + eta*r_t²` (eta=0.01).
  3. **total_reward** = dsr_reward + trade_penalty.
  4. **MDD 방지**: `step_pnl < -0.02`이면 total_reward -= 5.0.
  5. **청산 보너스**: `trade_done`이면 total_reward += realized_pnl * 10.0, trade_count += 1.
- **reset_reward_states()**: trade_count, step_pnl_ema, A, B 초기화.

---

## 7. 학습 파이프라인 (train_ppo.py)

### 7.1 초기화

- DataCollector(use_saved_data=True), 12개 전략, `_load_features()` → eth_data에 피처·전략 병합.
- TradingEnvironment, `_fit_global_scaler_dummy()` → train_end_idx 설정, scaler_fitted=True.
- PPOAgent(state_dim=29, action_dim=4, info_dim=15).
- TensorBoard: `logs/tensorboard/<YYYYMMDD_HHMMSS>`.
- `_prepare_curriculum_indices()`: all_indices, trend_indices 설정.

### 7.2 커리큘럼 학습

- **all_indices**: `[LOOKBACK+100, train_end_idx-500)` 구간의 정수 인덱스.
- **trend_indices**: 위 구간 중 `chop < 50` 인 인덱스 (추세장, 상대적으로 쉬운 구간).
- **에피소드 번호 < 500** 이고 trend_indices가 비어 있지 않으면: start_idx를 **trend_indices**에서 샘플링 (EASY).
- 그 외: start_idx를 **all_indices**에서 샘플링 (HARD).

### 7.3 에피소드 스텝 루프

1. current_idx에서 close, next_candle(high, low, close) 수집.
2. **aux_target** = (next_high - next_low) / next_close * 100 (다음 봉 변동성, %).
3. unrealized_pnl, step_pnl, pos_info = [pos_val, unrealized_pnl*10, holding_time/max_steps].
4. **state** = env.get_observation(position_info=pos_info, current_index=current_idx).
5. **action, prob, val** = agent.select_action(state).
6. action에 따라 position, entry_price, entry_index, trade_done, realized_pnl, holding_time_norm 갱신. trade_done이면 episode_pnl += realized_pnl.
7. **reward** = env.calculate_reward(step_pnl, realized_pnl, trade_done, holding_time=holding_time_norm, action=action, ..., agent_type='ENTRY'/'EXIT').
8. next_state = get_observation(next_pos_info, next_idx). done 판정.
9. **agent.put_data((state, action, reward, next_state, prob, done, val, aux_target))** (8개 요소).
10. current_index += 1. 에피소드 끝이거나 next_state가 None이면 루프 종료.

### 7.4 에피소드 종료 후

- 미청산 포지션이 있으면 강제 청산: realized_pnl 계산, calculate_reward 호출, put_data(..., action=3, ..., aux_target=0.0).
- **agent.train_net(episode_num)** 호출.
- TensorBoard: Reward/Total, Metrics/PnL, Loss/Total 기록.
- best reward 갱신 시 best 모델·스케일러 저장, 주기적으로 last 저장.

---

## 8. 평가 파이프라인 (evaluate_ppo.py)

- 데이터·전략·스케일러 로드, val/test/full 구간 설정.
- PPOAgent 생성 후 `*_entry.pth` / `*_exit.pth` 계층형 모델 로드.
- 구간 내 각 인덱스에서 pos_info 계산 → get_observation → select_action → 진입/청산 처리, 수수료 반영, balance·trades 기록.
- 결과: 거래 내역, 잔고 곡선, 성과 지표 등.

---

## 9. 설정 요약 (config.py)

| 구분 | 항목 | 비고 |
|------|------|------|
| **Lookback** | LOOKBACK=60 | 시계열 봉 개수 |
| **보상** | REWARD_MULTIPLIER, LOSS_PENALTY_MULTIPLIER, STOP_LOSS_THRESHOLD | 보상·손실 한도 |
| **PPO** | PPO_GAMMA, PPO_LAMBDA, PPO_EPS_CLIP, PPO_K_EPOCHS, PPO_ENTROPY_COEF, PPO_LEARNING_RATE | GAE, 클리핑, 탐험, 학습률 |
| **고급 PPO** | PPO_USE_VALUE_CLIP, PPO_VALUE_CLIP_EPS, PPO_KL_TARGET | Value clip, KL 조기 종료 |
| **네트워크** | NETWORK_HIDDEN_DIM(64), NETWORK_NUM_LAYERS(1), NETWORK_DROPOUT(0.1) | SharedBackbone·Head |
| **학습** | TRAIN_BATCH_SIZE(128), TRAIN_MAX_STEPS_PER_EPISODE(480), TRAIN_SPLIT(0.7) | 배치, 에피소드 길이, 데이터 분할 |
| **데이터 분할** | TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT = 1.0 | 0.7 + 0.15 + 0.15 |

---

## 10. 차원 및 Shape 요약표

| 항목 | 차원/Shape | 비고 |
|------|------------|------|
| state_dim (시계열 피처 수) | 29 | get_state_dim(), target_cols 개수 |
| Lookback | 60 | config.LOOKBACK |
| obs_seq | (1, 60, 29) | 배치 1, 시퀀스 60, 피처 29 |
| 전략 수 | 12 | strategy_0 ~ strategy_11 |
| 포지션 정보 | 3 | position_flag, unrealized_pnl*10, holding_time_norm |
| info_dim (obs_info) | 15 | 12 + 3 |
| Entry action_dim | 3 | Wait, Long, Short |
| Exit action_dim | 2 | Hold, Exit |
| Global Action | 4 | 0=Wait, 1=Long, 2=Short, 3=Exit |
| CausalConv1d 출력 | (B, hidden_dim, L) | L=60 유지 |
| SharedBackbone context | (B, hidden_dim) | 마지막 시점 hidden |
| combined (Head 입력) | (B, hidden_dim*3) | context + strat_feat + pos_feat |
| XLSTMNetwork 반환 | logits (B, action_dim), value (B, 1), aux_value (B, 1), next_states | list of (h,c,n,m) per layer |
| Transition (put_data) | 8개: (s, a, r, next_s, prob_a, done, val, aux_target) | 7개 시 aux_target=0.0 호환 |

---

## 11. 파일별 역할 요약

| 파일 | 역할 |
|------|------|
| **xlstm_network.py** | CausalConv1d, StabilizedSLSTMCell, SharedBackbone(Causal CNN + BatchNorm1d + sLSTM), StrategyGating, XLSTMNetwork → (logits, value, aux_value, next_states) |
| **ppo_agent.py** | CorePPOAgent(학습·추론·버퍼·train_net, logits/Categorical, State Passing), PPOAgent(계층 라우팅, put_data 7/8 호환) |
| **trading_env.py** | get_observation(obs_seq 60×29, obs_info 15), calculate_reward(DSR, Action Dampening, MDD, trade_done), get_state_dim=29 |
| **preprocess.py** | DataPreprocessor: Rolling Normalization (transform 윈도우 통계, epsilon 처리) |
| **config.py** | LOOKBACK, PPO/네트워크/학습/데이터 분할 파라미터 |
| **train_ppo.py** | 데이터 로드·커리큘럼 인덱스·에피소드 루프·aux_target·put_data 8요소·train_net·TensorBoard·best/last 저장 |
| **evaluate_ppo.py** | 데이터·스케일러·계층형 모델 로드, val/test/full 백테스트, select_action으로 행동 결정 |

---

이 문서는 **model 폴더의 현재 코드**(Genius V2: Causal CNN, SharedBackbone, StrategyGating, Logits + State Passing, BatchNorm1d)를 기준으로 작성되었습니다.
