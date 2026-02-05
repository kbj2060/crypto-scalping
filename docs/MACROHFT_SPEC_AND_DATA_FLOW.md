# MacroHFT (PPO) 모델 명세 및 데이터 흐름

## 1. 개요

- **목적**: 3-Action(Hold/Buy/Sell) 이산 행동 기반 스캘핑. 시장 국면별 전문가 3명 + 라우터 앙상블.
- **알고리즘**: PPO (Proximal Policy Optimization) + GAE. Dynamic Entropy(변동성 기반 엔트로피 계수).
- **입력**: 시계열 피처(60봉) + 정보 벡터(15차원). **출력**: 행동 0(Hold), 1(Buy), 2(Sell).

---

## 2. 설정 (config)

| 항목 | 값 | 설명 |
|------|-----|------|
| `LOOKBACK` | 60 | 시계열 윈도우 |
| `TRAIN_ACTION_DIM` | 3 | Hold, Buy, Sell |
| `PPO_GAMMA` | 0.99 | 할인율 |
| `PPO_LAMBDA` | 0.95 | GAE Lambda |
| `PPO_EPS_CLIP` | 0.2 | PPO 클리핑 |
| `PPO_K_EPOCHS` | 4 | 에피소드당 업데이트 횟수 |
| `PPO_ENTROPY_COEF` | 0.02 | 기본 엔트로피 계수 (Dynamic 시 가변) |
| `PPO_LEARNING_RATE` | 5e-5 | 학습률 |
| `NETWORK_HIDDEN_DIM` | 256 | XLSTM hidden 차원 |
| `NETWORK_NUM_LAYERS` | 2 | ResidualGRU 레이어 수 |
| `NETWORK_DROPOUT` | 0.1 | 드롭아웃 |
| `TRAIN_MAX_STEPS_PER_EPISODE` | 480 | 에피소드당 최대 스텝 |
| `TRANSACTION_COST` | 0.0005 | 거래 비용 |
| `STOP_LOSS_THRESHOLD` | -0.02 | 강제 청산 손절 |
| `MACROHFT_EXPERT_PHASE_EPISODES` | 3000 | Phase1 전문가 학습 에피소드 수 |

---

## 3. 상태(State) 명세

상태는 **튜플 `(obs_seq, obs_info)`** 로 구성된다. (TD3와 동일 환경 사용)

### 3.1 obs_seq (시계열 피처)

- **Shape**: `(1, 60, state_dim)`, state_dim = **29**.
- **생성**: `TradingEnvironment.precompute_data()` 에서 **Rolling Z-Score** (window=60, min_periods=1).
- **피처 컬럼**: TD3와 동일 29개 (log_return, roll_return_6, atr_ratio, … , trend_1h).

### 3.2 obs_info (정보 벡터, 15차원)

- **Shape**: `(1, 15)` (MacroHFT는 평가 시 volatility를 info에 넣지 않아도 됨; 전문가 네트워크는 15차원 기대).
- **구성**:
  - `[0:1]`: 포지션 값 (1=Long, -1=Short, 0=None).
  - `[1:13]`: 전략 점수 12개 (`strategy_0` ~ `strategy_11`).
  - `[13:15]`: 미실현 PnL×10, holding_time 정규화 등.

---

## 4. 행동(Action) 명세

- **차원**: 1 (이산). **공간**: {0, 1, 2}.
- **의미**:
  - **0**: Hold (포지션 유지 또는 무포지션 유지).
  - **1**: Buy (무포지션→Long, Short→청산 후 무포지션).
  - **2**: Sell (무포지션→Short, Long→청산 후 무포지션).

### 4.1 Action Masking (get_action_mask)

- **LONG** 보유 시: Buy 불가 → mask[1]=0.
- **SHORT** 보유 시: Sell 불가 → mask[2]=0.
- **무포지션**:
  - 에피소드 종료 임박(step > max_steps - 10): Buy/Sell 진입 금지 (mask[1]=mask[2]=0).
- 마스크 적용: `logits + (mask - 1) * 1e10` 후 Categorical 샘플링.

---

## 5. 보상(Reward) 명세 (trading_env.calculate_reward)

- **A. 스텝 보상**: 포지션 보유 중 `step_pnl * 50.0`, 클리핑 ±1.
- **B. 실현 손익**: 거래 체결 시 `realized_pnl * 100.0`, 클리핑 ±5. (거래 페널티 없음)
- **D. 리스크 페널티**: 포지션 보유 중 `step_pnl < -0.01` 이면 -0.2.
- **E. 최종**: `reward` 클리핑 ±5.

---

## 6. 네트워크 구조

### 6.1 Router (ppo_agent.py)

- **입력**: obs_seq의 마지막 시점만 사용 `x[:, -1, :]` → (B, state_dim).
- **구조**: Linear(256)→LayerNorm→ReLU→Dropout→Linear(64)→ReLU→Linear(3)→Softmax.
- **출력**: (B, 3) 전문가 가중치 [w1, w2, w3], 합=1.

### 6.2 XLSTMNetwork (xlstm_network.py) × 3 (Trend, Volatility, Sideways)

- **SharedBackbone**: (B, 60, 29) → input_proj → ResidualGRU×2 → Temporal Attention → context (B, hidden_dim).
- **StrategyInteractionLayer**: info[:, 1:13] (12차원) → Self-Attention → 64차원.
- **Gated Fusion**: context + strat_features + pos_info(3) → gate(Sigmoid) → gated * input → fusion_proj → hidden.
- **Actor**: hidden → MLP → logits (3).
- **Critic**: hidden → critic_mean(1), critic_cvar(1), aux(1).
- **반환**: logits, val_mean, val_cvar, aux_val, next_states, gate_mean.

### 6.3 앙상블 (Router 모드)

- 각 전문가 logits → stack (B, 3, 3). Router weights (B, 3) → `weighted_logits = sum(weights.unsqueeze(-1) * logits, dim=1)`.
- 학습 시 Router 모드에서는 Critic Loss 없음 (curr_val=0).

---

## 7. 학습 데이터 흐름 (train_ppo.py)

### 7.1 데이터·환경

```
1. data/training_features.csv → DataCollector.eth_data
   data/cached_strategies.csv → strategy_* 컬럼 병합
2. TradingEnvironment.precompute_data() → Rolling Z-Score → cached_features (T, 29), cached_strategies (T, 12)
3. _prepare_curriculum_indices(): chop, atr_ratio 기준으로
   - indices_trend (chop < 45)
   - indices_vol (atr > 0.75 분위)
   - indices_chop (chop > 50, atr < mean)
   부족 시(<100) 전체 valid_indices 로 Fallback
```

### 7.2 에피소드 스케줄 (Interleaved)

- `episode_num % 4`:
  - 0,1,2 → **expert** 모드, expert_idx=0,1,2. 각각 idx_trend, idx_vol, idx_side에서 start_idx 샘플.
  - 3 → **router** 모드, idx_all에서 start_idx 샘플.

### 7.3 에피소드 루프

```
1. current_position=None, entry_price, entry_index 초기화
2. volatility_label: 과거 10봉 수익률의 std * 100 (Aux Target / Dynamic Entropy용)
3. pos_info = [pos_val, unrealized_pnl*10, holding_time/max_steps]
4. state = env.get_observation(pos_info, current_idx)
5. action_mask = get_action_mask(current_position, market_vol, step)
6. action, prob, val = agent.select_action(state, action_mask, mode, expert_idx)
7. Stop Loss / Take Profit / Time Stop 시 강제 청산(exit_action)
8. action 1/2에 따라 포지션 변경, realized_pnl, trade_done 계산
9. reward = env.calculate_reward(step_pnl, realized_pnl, trade_done, ...)
10. next_state = env.get_observation(next_pos_info, next_idx)
11. agent.put_data((state, action, reward, next_state, prob, done, val, volatility_label, action_mask))
    → transition: x[0]=state, x[1]=action, x[2]=reward, x[3]=next_state, x[4]=prob, x[5]=done, x[6]=val, x[7]=volatility_label, x[8]=action_mask
12. 에피소드 종료 시 미청산 포지션 강제 청산 → put_data(terminal transition, volatility_label=0)
13. agent.train_net(episode, mode, expert_idx)
```

### 7.4 train_net (ppo_agent.py)

```
1. 배치 추출: s_seq, s_info, a, r, prob_a, done_mask, val, vol_label, masks
2. GAE: deltas = r + gamma * next_val * done_mask - val; advantage 역방향 누적; target_val = advantage + val; advantage 정규화
3. Optimizer: expert 모드 → opt_experts[expert_idx], router 모드 → opt_router
4. Dynamic Entropy: avg_vol = mean(vol_label); dynamic_entropy_coef = base_entropy * (1 + 0.5 * avg_vol)
5. k_epochs 반복:
   - expert: logits, curr_val = network(s_seq, s_info)
   - router: 3명 logits 가중합, curr_val=0
   - logits += (masks - 1) * 1e10; Categorical; actor_loss = -min(surr1, surr2); critic_loss (expert만, 가중치 1.0)
   - entropy_loss = -dynamic_entropy_coef * entropy.mean()
   - loss = actor_loss + critic_loss + entropy_loss; clip_grad_norm_(0.5); step
6. 반환 메트릭: Loss/Total, Entropy/Coeff, Loss/Expert_i 또는 Loss/Router
```

---

## 8. 평가 데이터 흐름 (evaluate_ppo.py)

- **모델**: `data/macroHFT/<timestamp>/ppo_model_best.pth` 또는 `_last.pth`. 최신 run 폴더에서 선택.
- **상태**: `env.get_observation(pos_info, current_index)` (Rolling Z-Score는 env 캐시 그대로 사용).
- **행동**: `select_action(state, action_mask=get_action_mask(...), deterministic=True)` → argmax.
- **구간**: val / test / full 에 따라 start_idx, end_idx 설정.

---

## 9. 모델 저장/로드

- **저장 경로**: `data/macroHFT/<run_timestamp>/`
  - `ppo_model_best.pth`, `ppo_model_last.pth`
- **내용**: `experts`(3개 state_dict), `router` state_dict, 옵티마이저 state_dict 등.
- **로드**: `agent.load_model(path)` → experts/router state_dict 로드.

---

## 10. 요약

| 구분 | 내용 |
|------|------|
| 상태 | (60×29 시계열 Z-Score, 15차원 정보) |
| 행동 | 이산 3 (Hold/Buy/Sell), Action Mask 적용 |
| 보상 | 스텝 보상 + 실현 PnL + 리스크 페널티, 클리핑 ±5 |
| 학습 | PPO + GAE, Expert 3명 + Router, Curriculum(국면별 인덱스), Dynamic Entropy(vol_label) |
| 평가 | deterministic argmax, 동일 env 캐시·마스크 |
