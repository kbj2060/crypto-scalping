# PPO 트레이딩 모델: 아키텍처 & 데이터 흐름

> model 폴더 코드 기준으로 정리한 **모델 아키텍처**와 **데이터 흐름** 문서입니다.

---

## 1. 전체 구조 개요

```
[원시/피처 데이터] → [전처리·전략 점수] → [TradingEnvironment] → [PPOAgent (계층형)]
       │                     │                      │                        │
       │                     │                      │                        ├── 무포지션 → Entry Agent (XLSTM) → 0=Wait, 1=Long, 2=Short
       │                     │                      │                        └── 포지션 있음 → Exit Agent (XLSTM)  → 0=Hold, 1=Exit → Global 3=Exit
       │                     │                      │
       │                     │                      └── get_observation() → (obs_seq, obs_info)
       │                     │                      └── calculate_reward()
       │                     │
       └── training_features.csv, cached_strategies.csv
       └── DataPreprocessor (Z-Score), 12개 전략 점수
```

- **계층형 PPO**: 포지션 유무에 따라 **Entry 에이전트**와 **Exit 에이전트**가 나뉘어 동작합니다.
- **Global Action 4개**: `0=Wait/Hold`, `1=Long`, `2=Short`, `3=Exit`
- **학습**: 각 에피소드에서 (state, action, reward, next_state, log_prob, done, value)를 버퍼에 쌓고, 포지션/액션에 따라 Entry 또는 Exit 버퍼로만 넣은 뒤 각각 `train_net()`으로 PPO 업데이트합니다.

---

## 2. 데이터 소스 & 전처리

| 단계 | 위치/모듈 | 설명 |
|------|-----------|------|
| **원시/피처 데이터** | `data/training_features.csv` | ETH/BTC OHLCV 기반 **29개 시계열 피처** (feature_engineering 등으로 생성) |
| **전략 점수** | `data/cached_strategies.csv` 또는 재계산 | 12개 전략의 `strategy_0` ~ `strategy_11` (봉별 LONG/SHORT/NEUTRAL 신호 → confidence 스칼라) |
| **스케일러** | `DataPreprocessor` (`model/preprocess.py`) | `fit()`: 학습 구간 샘플로 mean/std 계산 → `transform()`: Z-Score 정규화. `.pkl`로 저장/로드 |
| **Lookback** | `config.LOOKBACK = 60` | 과거 60봉 시퀀스로 시계열 상태 표현 |

**학습 시**: `train_ppo.py`에서 `_load_features()`로 `training_features.csv` + `cached_strategies.csv` 병합 후 `DataCollector.eth_data`에 넣고, `_fit_global_scaler()`에서 학습 구간(`TRAIN_SPLIT`) 일부 샘플로 `preprocessor.fit(sample)` 호출 후 `scaler_fitted = True`로 설정합니다.

**관련 파일**: `model/preprocess.py`, `model/feature_engineering.py`, `model/collect_training_data.py`, `core/data_collector.py`

---

## 3. 관측(Observation) 구성

`TradingEnvironment.get_observation(position_info, current_index)` 가 반환하는 **state** 는 `(obs_seq, obs_info)` 튜플입니다.

### 3.1 `obs_seq` (시계열 상태)

- **shape**: `(1, lookback, state_dim)` = `(1, 60, 29)`
- **state_dim = 29** (`get_state_dim()` 반환값): `trading_env.py`의 `target_cols` 29개
  - 예: log_return, roll_return_6, atr_ratio, bb_width, bb_pos, rsi, macd_hist, hma_ratio, cci, rvol, taker_ratio, cvd_change, mfi, cmf, vwap_dist, wick_upper, wick_lower, range_pos, swing_break, chop, btc_return, btc_rsi, btc_corr, btc_vol, eth_btc_ratio, rsi_15m, trend_15m, rsi_1h, trend_1h
- **흐름**: `curr_idx - lookback ~ curr_idx` 구간의 `target_cols` 슬라이스 → `preprocessor.transform()` (Z-Score) → `torch.FloatTensor` → `(1, 60, 29)`

### 3.2 `obs_info` (정보 벡터, Info Dim 15)

- **shape**: `(1, 15)`
- **구성** (순서 고정):
  - **앞 12개**: 전략 점수 `strategy_0` ~ `strategy_11` (현재 봉 기준 스칼라)
  - **뒤 3개**: 포지션 정보
    - `position_flag`: 1.0=Long, -1.0=Short, 0.0=없음
    - `unrealized_pnl * 10`: 미실현 손익 스케일
    - `holding_time` 정규화 (예: holding_time / max_steps)

**관련 파일**: `model/trading_env.py` (`get_observation`, `get_state_dim`)

---

## 4. XLSTM 네트워크 (현재 아키텍처)

파일: `model/xlstm_network.py`. **Actor와 Critic이 완전히 분리된 Dual Backbone** 구조이며, Attention 없이 **Last Hidden State**만 사용합니다.

### 4.1 StabilizedSLSTMCell (핵심 셀)

- **역할**: xLSTM의 sLSTM 셀 (Scalar LSTM with Exponential Gating)
- **입력**: `x`, `state = (h_prev, c_prev, n_prev, m_prev)`
- **게이트**: `weight_ih(x) + weight_hh(h_prev)` → chunk 4개 (z, i, f, o)
- **활성화**: z_t = tanh(z_pre), o_t = sigmoid(o_pre)
- **Log-Space Stabilization**: `m_t = max(f_pre + m_prev, i_pre)`, `i_prime = exp(i_pre - m_t)`, `f_prime = exp(f_pre + m_prev - m_t)` 로 exp 폭발 완화
- **상태 갱신**: c_t, n_t, 출력 `h_t = o_t * (c_t / (n_t + 1e-6))`

### 4.2 SLSTMBackbone (시계열 → 특징 벡터)

- **역할**: 시계열 `(B, L, input_dim)` 을 받아 **마지막 타임스텝의 hidden** 만 반환. Actor/Critic이 **각각 별도 인스턴스**로 가짐 (파라미터 공유 없음).
- **구조**:
  1. **Input Projection**: Linear(input_dim, hidden_dim) → LayerNorm → GELU → Dropout
  2. **Stacked sLSTM**: num_layers개의 StabilizedSLSTMCell, 층마다 Residual(inp + h_next) + LayerNorm
  3. **출력**: `current_input[:, -1, :]` → **last_hidden** `(B, hidden_dim)`, 그리고 next_states (현재 호출부에서는 사용하지 않음)

### 4.3 XLSTMNetwork (래퍼)

- **Actor Trunk** (Policy):
  - `actor_backbone`: SLSTMBackbone(input_dim, hidden_dim, num_layers, dropout) → `actor_feat` (B, hidden_dim)
  - `actor_info_enc`: Linear(info_dim, 64) → GELU → `actor_info` (B, 64)
  - `actor_input = concat(actor_feat, actor_info)` → (B, hidden_dim + 64)
  - `actor_head`: Linear → LayerNorm → GELU → Linear → **logits** (B, action_dim)
  - logits / temperature → softmax → **probs**

- **Critic Trunk** (Value, Actor와 완전 분리):
  - `critic_backbone`: SLSTMBackbone(동일 인자) → `critic_feat`
  - `critic_info_enc`: Linear(info_dim, 64) → GELU → `critic_info`
  - `critic_input = concat(critic_feat, critic_info)` → `critic_head` → **value** (B, 1)

- **forward(x, info, states=None, temperature=1.0)**:
  - info가 3차원이면 squeeze(1)
  - Actor 경로: actor_backbone(x, None) → last_hidden + info_enc → actor_head → logits/temperature → softmax → probs
  - Critic 경로: critic_backbone(x, None) → last_hidden + info_enc → critic_head → value
  - **반환**: `(probs, value, None)` (next_states는 미사용)

**관련 파일**: `model/xlstm_network.py`

---

## 5. 계층형 PPO 에이전트 (PPOAgent)

파일: `model/ppo_agent.py`

### 5.1 CorePPOAgent (Entry / Exit 각 1개)

- **Entry Agent**: action_dim=3 (Wait=0, Long=1, Short=2). 무포지션일 때만 호출.
- **Exit Agent**: action_dim=2 (Hold=0, Exit=1). 포지션 있을 때만 호출.
- 각 CorePPOAgent는 **XLSTMNetwork** 1개 + **model_target** (soft update용), optimizer, scheduler, 버퍼 `data` 를 가짐.

### 5.2 포지션 판단 및 라우팅

- **select_action(state, action_mask=None)**  
  - `state = (obs_seq, obs_info)`.  
  - **포지션 플래그**: `obs_info[0]` (또는 텐서일 때 `obs_info[0,0]`)으로 현재는 첫 번째 요소 사용. (설계상으로는 obs_info 마지막 3개가 포지션 정보이므로, 의도라면 index 12 등이어야 할 수 있음.)  
  - `is_position_open = (abs(pos_flag) > 0.1)`  
  - **무포지션** → Entry Agent만 호출 → 반환 action 0,1,2 그대로  
  - **포지션 있음** → Exit Agent만 호출 → Local 0→Global 0, Local 1→Global 3

### 5.3 Action Mask (4-dim → 3/2-dim)

- 외부에서 4-dim 마스크 `[Wait, Long, Short, Exit]` 를 넘기면:
  - Entry: `mask[0:3]`
  - Exit: `[mask[0], mask[3]]`
- 학습 시 `train_ppo.py`에서는 현재 `action_mask=None`으로 호출.

### 5.4 학습 시 데이터 라우팅 (put_data)

- Transition: `(s, global_a, r, next_s, prob_a, done, val)`
- `s[1]`(obs_info)으로 포지션 유무 판단:
  - **포지션 없음** & global_a in {0,1,2} → Entry Agent 버퍼에만 저장 (global_a 그대로)
  - **포지션 있음** & global_a in {0, 3} → Exit Agent 버퍼에만 저장 (3→local 1, 0→local 0)

### 5.5 train_net (GAE + PPO)

- 버퍼에서 (s_seq, s_info, a, r, next_s_seq, next_s_info, prob_a, done_mask, old_v) 등으로 배치 구성
- **GAE**: model(s_seq, s_info) → v, model_target(next_s_seq, next_s_info) → next_v, TD target = r + gamma*next_v*done_mask, delta → advantage 계산 후 정규화
- **PPO**: k_epochs 동안 curr_probs, curr_v 재계산, ratio = exp(curr_log_prob - prob_a), clip, entropy, value clip loss → 역전파, grad clip, optimizer step
- Target 네트워크 soft update (tau=0.995), scheduler.step()

**관련 파일**: `model/ppo_agent.py`

---

## 6. 보상 함수 (TradingEnvironment.calculate_reward)

- **No-Position Penalty**: 포지션 없고 trade_done도 아니면 -0.001
- **Step Reward (EMA)**: 포지션 있을 때 step_pnl로 EMA 갱신 후 reward += step_pnl_ema * 50.0
- **Directional Bonus**: 포지션 있고 step_pnl > 0 이면 +0.02
- **Terminal (trade_done)**: net_pnl = realized_pnl - fee(0.0005), reward += net_pnl * 150.0, 진입 비용 -0.02, holding_time < 0.005면 -0.05, net_pnl < -0.02면 -2.0, trade_count 증가, step_pnl_ema 초기화

**관련 파일**: `model/trading_env.py`

---

## 7. 학습 시 데이터 흐름 (train_ppo.py)

1. **초기화**  
   - DataCollector(use_saved_data=True), 12개 전략, `_load_features()` → training_features.csv + cached_strategies 병합  
   - TradingEnvironment, `_fit_global_scaler()` → 학습 구간 샘플로 Z-Score fit, scaler_fitted=True  
   - PPOAgent(state_dim=29, action_dim=4, info_dim=15)

2. **에피소드 시작**  
   - 학습 구간 내에서 start_idx 랜덤 샘플링, `data_collector.current_index = start_idx`  
   - `env.reset_reward_states()`, `agent.reset_episode_states()`

3. **스텝 루프** (최대 TRAIN_MAX_STEPS_PER_EPISODE)  
   - 현재 인덱스에서 close 가격, 포지션으로 unrealized_pnl, step_pnl, pos_info 계산  
   - `state = env.get_observation(position_info=pos_info, current_index=current_idx)` → (obs_seq, obs_info)  
   - `action, prob, val = agent.select_action(state, action_mask=None)`  
   - action에 따라 position/entry_price/entry_index 갱신, action=3이면 청산(realized_pnl, trade_done, episode_pnl 누적)  
   - `reward = env.calculate_reward(step_pnl, realized_pnl, trade_done, holding_time_norm, ...)`  
   - next_state = get_observation(next_idx), done 판정  
   - `agent.put_data((state, action, reward, next_state, prob, done, val))`  
   - data_collector.current_index += 1

4. **에피소드 종료 시**  
   - 미청산 포지션이 있으면 강제 청산: realized_pnl 계산, calculate_reward 호출, `put_data(..., action=3, ...)` 한 번 더, episode_pnl 누적

5. **PPO 업데이트**  
   - `agent.train_net(episode_num)` → Entry/Exit 각각 버퍼로 GAE·PPO 업데이트

6. **로그·저장**  
   - episode_reward, avg_reward(최근 10), trade_count, **episode_pnl** 로그  
   - best reward 갱신 시 best 모델·스케일러 저장, 주기적으로 last 모델·스케일러 저장

**관련 파일**: `model/train_ppo.py`, `model/trading_env.py`, `model/ppo_agent.py`

---

## 8. 평가 시 데이터 흐름 (evaluate_ppo.py)

1. **데이터·스케일러·모델 로드**  
   - training_features.csv + cached_strategies (또는 전략 재계산), env.preprocessor.load(scaler), PPOAgent, load_model(best/last 등)

2. **구간**  
   - mode에 따라 val/test/full 구간의 start_idx ~ end_idx 설정

3. **백테스트 루프**  
   - 매 인덱스마다 pos_info 계산 → `state = env.get_observation(position_info, current_index=idx)`  
   - `action, _, _ = agent.select_action(state, action_mask=None)`  
   - action에 따라 진입/청산 처리, 수수료 반영, balance·trades 기록

4. **결과**  
   - 거래 내역, 잔고 곡선, 성과 지표 등

**관련 파일**: `model/evaluate_ppo.py`

---

## 9. 차원 요약표

| 항목 | 차원/값 | 비고 |
|------|---------|------|
| 시계열 피처 수 (state_dim) | 29 | get_state_dim(), target_cols 개수 |
| Lookback | 60 | config.LOOKBACK |
| obs_seq | (1, 60, 29) | 배치 1, 시퀀스 60, 피처 29 |
| 전략 수 | 12 | strategy_0 ~ strategy_11 |
| 포지션 정보 | 3 | flag, unrealized_pnl*10, holding 정규화 |
| obs_info (info_dim) | 15 | 12 + 3 |
| Entry Agent action_dim | 3 | Wait, Long, Short |
| Exit Agent action_dim | 2 | Hold, Exit |
| Global Action | 4 | 0=Wait, 1=Long, 2=Short, 3=Exit |
| XLSTM hidden_dim | 128 | config.NETWORK_HIDDEN_DIM |
| Info Encoder 출력 | 64 | actor_info_enc, critic_info_enc |
| Fusion 차원 (Actor/Critic 각각) | 128 + 64 = 192 | last_hidden + info_emb |
| 네트워크 출력 | probs (B, action_dim), value (B, 1) | next_states=None |

---

## 10. 파일별 역할 요약

| 파일 | 역할 |
|------|------|
| `model/xlstm_network.py` | StabilizedSLSTMCell, SLSTMBackbone (Last Hidden만 사용), XLSTMNetwork (Actor/Critic Dual Backbone + Info Fusion), (probs, value, None) 반환 |
| `model/ppo_agent.py` | CorePPOAgent(학습·추론·버퍼·train_net), PPOAgent(계층 라우팅·put_data 분리·load/save _entry/_exit) |
| `model/trading_env.py` | get_observation(obs_seq 29차원×60, obs_info 15), calculate_reward, get_state_dim=29 |
| `model/preprocess.py` | DataPreprocessor: Z-Score fit/transform, save/load (.pkl) |
| `model/config.py` | LOOKBACK, PPO/네트워크 하이퍼파라미터, 데이터 분할, 경로 등 |
| `model/train_ppo.py` | 데이터 로드·스케일러 fit, 에피소드 루프·보상·put_data·train_net, PnL·Reward 로그, best/last 저장 |
| `model/evaluate_ppo.py` | 데이터·스케일러·모델 로드, val/test 구간 백테스트, select_action으로 행동 결정 |
| `model/feature_engineering.py` | 가격·거래량·패턴·매크로 등 29개 피처 생성 |
| `model/collect_training_data.py` | DataCollector로 1년치 수집·저장 래퍼 |

---

이 문서는 **model 폴더의 현재 코드**를 기준으로 작성되었으며, Dual Backbone XLSTM(Attention 제거, Last Hidden + Info Fusion) 및 계층형 PPO 데이터 흐름을 반영합니다.
