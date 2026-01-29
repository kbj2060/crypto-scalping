# PPO 모델 아키텍처 및 데이터 흐름

## 1. 전체 구조 개요

```
[원시/피처 데이터] → [TradingEnvironment] → (obs_seq, obs_info)
                                                    ↓
[PPOAgent] ←──────────────────────────── [XLSTMNetwork]
     │                                              │
     ├─ select_action(state) → action (0/1/2)       ├─ Actor Head → π(a|s)
     ├─ put_data(transition)                        └─ Critic Head → V(s)
     └─ train_net() [GAE + PPO Clip + Target Critic EMA]
```

- **Action 3**: `0` = Neutral(청산/대기), `1` = Long, `2` = Short (Target Position 방식)
- **입력**: 시계열 피처 29차원 × 60봉 + 정보 벡터 15차원 → 네트워크 하나로 통합

---

## 2. 입력 데이터 (Observation)

환경 `TradingEnvironment.get_observation()` 이 반환하는 것은 **두 개의 텐서**입니다.

### 2.1 시계열 `obs_seq` (state_dim = 29, lookback = 60)

- **Shape**: `(1, 60, 29)` — 배치 1, 시퀀스 길이 60, 피처 29개
- **출처**: `config.LOOKBACK` 구간만큼의 과거 봉에 대해 아래 29개 컬럼을 Z-Score 정규화한 값  
  (`DataPreprocessor.transform()`)

| 구분 | 컬럼 예시 |
|------|-----------|
| 수익/변동성 | `log_return`, `roll_return_6`, `atr_ratio`, `bb_width`, `bb_pos` |
| 모멘텀/과매수과매도 | `rsi`, `macd_hist`, `hma_ratio`, `cci` |
| 거래량/가격 | `rvol`, `taker_ratio`, `cvd_change`, `mfi`, `cmf`, `vwap_dist` |
| 가격 구조 | `wick_upper`, `wick_lower`, `range_pos`, `swing_break`, `chop` |
| BTC/연관 | `btc_return`, `btc_rsi`, `btc_corr`, `btc_vol`, `eth_btc_ratio` |
| 고 timeframe | `rsi_15m`, `trend_15m`, `rsi_1h`, `trend_1h` |

### 2.2 정보 벡터 `obs_info` (info_dim = 15)

- **Shape**: `(1, 15)`
- **구성**:
  - **전략 점수 12개**: `strategy_0` ~ `strategy_11` (각 전략의 LONG/SHORT/NEUTRAL 신호를 -1~1 스케일로)
  - **포지션 정보 3개**: `[pos_value, unrealized_pnl*10, holding_time/1000]`
    - `pos_value`: Long=1, Short=-1, 무포지션=0
    - 나머지는 스케일링된 미실현 손익·보유 시간

---

## 3. XLSTM 네트워크 아키텍처 (model/xlstm_network.py)

### 3.1 블록 구성

```
obs_seq (B, 60, 29)          obs_info (B, 15)
        │                              │
        ▼                              │
[Input Projection]                    │
   Linear(29 → 128)                   │
   Dropout                            │
        │                              │
        ▼                              │
[Stabilized sLSTM × num_layers]      │
   - StabilizedSLSTMCell (log-space)  │
   - LayerNorm per layer             │
   - 시퀀스 타임스텝별 순차 처리       │
        │                              │
        ▼                              │
[Multi-Head Attention]                │
   num_heads=4, batch_first           │
   Residual + LayerNorm               │
        │                              │
        ▼                              │
[Attention Pooling]                   │
   Linear(hidden→1) → softmax →      │
   weighted sum → context_feature     │
   (B, 128)                           │
        │                              │
        ├─────────────────────────────┤
        ▼                              ▼
        [Concat] → (B, 128+64)
                    │
                    ▼
            [Info Encoder]
               Linear(15→64), LayerNorm, GELU, Dropout
               Linear(64→64), GELU
               → (B, 64)
                    │
                    ▼
            [Shared Trunk]
               256 → LayerNorm → GELU → Dropout
               128 → LayerNorm → GELU → Dropout
               → (B, 128)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
[Actor Head]            [Critic Head]
 128→64→GELU→3           128→64→1
 → action_probs          → state_value
 (B, 3)                  (B, 1)
```

### 3.2 출력

- **action_probs**: `(batch, 3)` — Softmax( logits / temperature ), 각 스텝에서 Neutral/Long/Short 확률
- **state_value**: `(batch, 1)` — V(s)
- **next_states**: sLSTM 각 레이어의 (h, c, n, m) 상태; 다음 스텝에서 그대로 넣어 재귀 처리

### 3.3 설정값 (config 연동)

- `input_dim=state_dim` (29), `action_dim=3`, `info_dim=15`
- `hidden_dim=config.NETWORK_HIDDEN_DIM` (128)
- `num_layers=config.NETWORK_NUM_LAYERS` (1)

---

## 4. PPO Agent 데이터 흐름 (model/ppo_agent.py)

### 4.1 학습 시 한 스텝 (train_ppo / 환경과 상호작용)

1. **관측**: `state = env.get_observation(pos_info, current_index)` → `(obs_seq, obs_info)`
2. **액션 선택**: `action, log_prob, value = agent.select_action(state, action_mask=[1,1,1])`
   - `model(obs_seq, obs_info, current_states, temperature)` → `probs`, `value`, `next_states`
   - `Categorical(probs).sample()` → action (0/1/2)
   - `current_states` 갱신해 다음 스텝에 재사용
3. **환경 스텝**: 환경이 action에 따라 포지션·청산·진입 처리, `reward = env.calculate_reward(...)`
4. **버퍼 저장**: `agent.put_data((state, action, reward, next_state, log_prob, done, value))`

### 4.2 에피소드 종료 후 학습 (train_net)

1. **버퍼 → 텐서**:  
   `s_seq`, `s_info`, `a`, `r`, `next_s_seq`, `next_s_info`, `prob_a`, `done`, `old_v`
2. **GAE (Target Critic 사용)**  
   - 현재 V(s): Main model  
   - V(s'): **Target model** (EMA)  
   - TD: `delta = (r + γ * V_target(s')) - V(s)`  
   - GAE로 advantage 계산 후 정규화
3. **PPO 업데이트** (k_epochs 반복)  
   - Ratio = π(a|s) / π_old(a|s), clip(ratio, 1-ε, 1+ε)  
   - Actor: `-min(ratio*adv, clipped*adv)` + entropy 보너스  
   - Critic: Value clipping 적용 가능, MSE with target  
   - Gradient clip 10.0, optimizer step
4. **부가 갱신**  
   - Temperature decay  
   - Target model EMA: `θ_target = τ*θ_target + (1-τ)*θ`

---

## 5. 보상 구조 (TradingEnvironment.calculate_reward)

| 구분 | 조건 | 보상/페널티 |
|------|------|-------------|
| Dynamic Neutral Penalty | 무포지션 & 거래 없음 | `-max(0.002, min(0.001*ep/200, 0.005))` |
| Step Reward (EMA) | 매 스텝 | `+ step_pnl_ema * 50` (포지션 있을 때만 EMA 갱신) |
| Directional Bonus | 포지션 있음 & step_pnl > 0 | +0.02 |
| Terminal (청산) | trade_done | 수익: `+ net_pnl*200`, 손실: `+ net_pnl*80`, 진입비 -0.001, 초단기 -0.03, 대손 -1.0 |

---

## 6. 학습/평가 파이프라인 요약

| 단계 | 학습 (train_ppo) | 평가 (evaluate_ppo) |
|------|------------------|---------------------|
| 데이터 | training_features + cached_strategies (전략 강제 재계산 가능) | eval_test_data 우선, 없으면 training_features |
| 관측 | 동일: (obs_seq 60×29, obs_info 15) | 동일 |
| 액션 | Categorical(probs).sample() | **argmax(probs)** (deterministic) |
| LSTM 상태 | 에피소드 내 유지, 에피소드마다 reset | 에피소드 내 유지 |
| Scaler | 학습 구간으로 fit, pkl 저장 | `*_scaler.pkl` 로드 |

---

## 7. 주요 설정 (config.py)

| 항목 | 값 | 비고 |
|------|-----|------|
| LOOKBACK | 60 | 시퀀스 길이 |
| state_dim | 29 | 시계열 피처 수 |
| info_dim | 15 | 전략 12 + 포지션 3 |
| action_dim | 3 | Neutral / Long / Short |
| NETWORK_HIDDEN_DIM | 128 | LSTM/Attention hidden |
| NETWORK_NUM_LAYERS | 1 | sLSTM 레이어 수 |
| PPO_EPS_CLIP | 0.3 | 정책 업데이트 clip |
| PPO_GAMMA / LMBDA | 0.99 / 0.95 | GAE |

이 문서는 `model/` 내 실제 코드와 `config.py` 기준으로 정리한 PPO 모델의 아키텍처와 데이터 흐름입니다.
