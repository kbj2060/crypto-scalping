# PPO 트레이딩 모델: 구현 명세

> 대화에서 정리·반영한 **보상 함수**, **네트워크 아키텍처**, **학습·로깅** 관련 구현 내용을 문서화한 것입니다.

---

## 1. 보상 함수 (Differential Sharpe Ratio + Action Dampening)

**위치**: `model/trading_env.py` — `TradingEnvironment.calculate_reward()`

**참고**: Moody & Saffell (2001) — Differential Sharpe Ratio

### 1.1 시그니처

```python
calculate_reward(step_pnl, realized_pnl, trade_done, holding_time=0, action=0,
                prev_position=None, current_position=None, agent_type='ENTRY')
```

- **holding_time**: 보유 구간 정규화 (과도하게 짧은 매매 페널티용, 현재 DSR 식 내부에서는 미사용).
- **agent_type**: Entry/Exit 구분용 (학습 시 `train_ppo.py`에서 전달).

### 1.2 구성 요소

| 항목 | 설명 |
|------|------|
| **Action Dampening** | `action in [1,2,3]` (진입·청산)일 때 **거래 페널티 -0.5** 적용. |
| **Differential Sharpe Ratio** | `r_t = step_pnl` 기준으로 이동평균 A, B 갱신 후 `std_dev = sqrt(B - A²)`, **dsr_reward = (r_t * 100) / std_dev**. |
| **상태 (EMA)** | `A`, `B`: 수익률·수익률 제곱의 EMA. `eta = 0.01`로 갱신. `reset_reward_states()`에서 A, B 초기화. |
| **MDD 방지** | `step_pnl < -0.02`이면 **total_reward -= 5.0**. |
| **청산 보너스** | `trade_done`일 때 **total_reward += realized_pnl * 10.0**, `trade_count` 증가. |

### 1.3 config 연동

- `REWARD_MULTIPLIER`, `LOSS_PENALTY_MULTIPLIER`, `STOP_LOSS_THRESHOLD` 등은 `config.py`에 정의되어 있으며, 보상 식을 세밀 조정할 때 참고.

---

## 2. 네트워크 아키텍처 (CNN–sLSTM Hybrid + Strategy Attention)

**위치**: `model/xlstm_network.py`

### 2.1 StabilizedSLSTMCell

- **역할**: sLSTM 셀 (Scalar LSTM, exponential gating).
- **안정화**: `i_pre`, `f_pre`를 `clamp(-20, 20)` 후 exp 사용. NaN 발생 시 `nan_to_num`으로 0 대체.

### 2.2 HybridBackbone (CNN + sLSTM)

- **CNN 블록**: `Conv1d(input_dim → hidden_dim, k=3, padding=1)` → BatchNorm → ELU, 2층.
- **입력**: `(B, L, input_dim)` → permute로 `(B, input_dim, L)` → CNN → 다시 `(B, L, hidden_dim)`.
- **sLSTM**: CNN 출력에 `input_proj` 적용 후, `StabilizedSLSTMCell` 1층 (기본) 순차 적용. Residual + LayerNorm.
- **출력**: 마지막 타임스텝 hidden만 반환 → `(B, hidden_dim)`, `next_states`.

### 2.3 StrategyAttention

- **역할**: 전략 점수 `strategy_0` ~ `strategy_11`를 MultiheadAttention으로 시계열 컨텍스트와 결합.
- **구조**: `strategy_embedding(1→hidden_dim)` → MHA(query=context, key/value=embedding) → LayerNorm + residual → Linear → 출력 `(B, hidden_dim)`.

### 2.4 XLSTMNetwork (Actor + Critic, Shadow Mode)

- **Actor**: `actor_backbone`(HybridBackbone) → `actor_attention`(StrategyAttention) + `actor_pos_enc`(pos_info 3차원) → concat **hidden*3** → `actor_head` → logits / temperature → softmax → **probs**.
- **Critic**: 동일 구조로 **완전 분리** (`critic_backbone`, `critic_attention`, `critic_pos_enc`).
  - **Main**: `critic_head` → **value** (B, 1).
  - **Auxiliary (Shadow Mode)**: `critic_aux_head` → **aux_value** (다음 봉 변동성 예측용).
- **forward 반환**: `(probs, value, aux_value, None)`.

---

## 3. 학습 파이프라인 (train_ppo.py)

### 3.1 커리큘럼 학습

- **all_indices**: `LOOKBACK+100` ~ `train_end_idx-500` 구간 인덱스.
- **trend_indices**: `chop < 50` 인 구간만 필터 (추세장, 상대적으로 쉬운 구간).
- **에피소드 500 미만**: `start_idx`를 `trend_indices`에서만 샘플링 (EASY).
- **에피소드 500 이상**: `start_idx`를 `all_indices`에서 샘플링 (HARD).

### 3.2 Transition 및 보조 타겟

- **aux_target**: 다음 봉의 `(high - low) / close * 100` (변동성 스칼라).
- **put_data**: `(state, action, reward, next_state, prob, done, val, aux_target)` — 8개 요소.  
  - `ppo_agent.put_data`는 7개/8개 모두 호환.

### 3.3 보상 호출

- 스텝 보상: `holding_time_norm = (current_idx - entry_index) / max_steps` 계산 후  
  `env.calculate_reward(..., holding_time=holding_time_norm, action=action, ..., agent_type='ENTRY'/'EXIT')` 전달.
- 강제 청산 시: `holding_time=0.0`, `action=3`으로 `calculate_reward` 호출.

### 3.4 PPO 업데이트 (ppo_agent.py)

- **GAE** 후 기존 PPO loss (policy clip, value clip, entropy).
- **Auxiliary Loss**: `aux_value` vs `aux_target` MSE. 총 loss에 **0.5** 가중치로 가산.
- **train_net** 시 버퍼에서 `aux_target` 언패킹 후 위와 같이 사용.

---

## 4. 로깅 및 저장

### 4.1 TensorBoard

- **SummaryWriter**: `logs/tensorboard/ppo_<timestamp>` 형태의 하위 폴더 사용 (세션별 run 분리).
- **스칼라**: `Metrics/PnL` (episode_pnl), `Loss/Total` 등.

### 4.2 콘솔 로그

- 에피소드별 **episode_reward**, **trade_count**, **episode_pnl** 출력.
- best reward 갱신 시 best 모델·스케일러 저장. 주기적으로 last 모델·스케일러 저장.

---

## 5. 설정 요약 (config.py)

| 구분 | 항목 | 비고 |
|------|------|------|
| 보상 | REWARD_MULTIPLIER, LOSS_PENALTY_MULTIPLIER, STOP_LOSS_THRESHOLD | 보상 스케일·손실 한도 |
| PPO | PPO_ENTROPY_COEF, PPO_LEARNING_RATE, PPO_EPS_CLIP, PPO_K_EPOCHS | 탐험·학습률·클리핑 |
| 학습 | TRAIN_BATCH_SIZE (128), TRAIN_MAX_STEPS_PER_EPISODE (480) | 배치·에피소드 길이 |
| 네트워크 | NETWORK_HIDDEN_DIM (64), NETWORK_NUM_LAYERS (1), NETWORK_DROPOUT (0.1) | HybridBackbone·Attention |

---

## 6. 파일별 구현 요약

| 파일 | 구현 내용 |
|------|-----------|
| `trading_env.py` | DSR 보상, A/B/eta 상태, reset_reward_states, holding_time/agent_type 인자 |
| `xlstm_network.py` | StabilizedSLSTMCell, HybridBackbone, StrategyAttention, XLSTMNetwork (probs, value, aux_value, None) |
| `train_ppo.py` | 커리큘럼(trend_indices/all_indices), aux_target 계산, holding_time_norm·agent_type 전달, TensorBoard, episode_pnl |
| `ppo_agent.py` | put_data 7/8 요소 호환, train_net에서 aux_loss(MSE, 0.5) 반영, select_action 시 aux_value 수신 후 미사용 |
| `config.py` | REWARD_*, PPO_*, TRAIN_*, NETWORK_* 파라미터 |

---

이 문서는 **model 폴더의 현재 코드**를 기준으로 작성되었으며, DSR 보상·Hybrid CNN–sLSTM·Auxiliary Loss·커리큘럼·로깅 방식을 반영합니다.
