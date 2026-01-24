# Model 폴더 구조 및 데이터 흐름 요약

## 📁 Model 폴더 구조

```
model/
├── __init__.py                    # 모듈 초기화
├── collect_training_data.py       # 학습 데이터 수집 (별도 실행)
├── feature_engineering.py         # 25개 기본 피처 생성
├── mtf_processor.py               # 멀티 타임프레임 피처 (4개 추가)
├── preprocess.py                  # Z-Score 정규화 스케일러
├── trading_env.py                 # 강화학습 환경 (상태 관측 + 보상 계산)
├── ppo_agent.py                   # PPO 알고리즘 구현
├── xlstm_network.py               # xLSTM Actor-Critic 신경망
└── train_ppo.py                   # 학습 메인 스크립트
```

---

## 🔄 전체 데이터 흐름 개요

```
[원본 데이터]
    ↓
[피처 캐싱 시스템] (train_ppo.py)
    ↓
[피처 엔지니어링] (feature_engineering.py + mtf_processor.py)
    ↓
[스케일러 학습] (preprocess.py)
    ↓
[에피소드 루프] (train_ppo.py)
    ↓
[상태 관측] (trading_env.py)
    ↓
[행동 선택] (ppo_agent.py → xlstm_network.py)
    ↓
[보상 계산] (trading_env.py)
    ↓
[학습 업데이트] (ppo_agent.py)
```

---

## 📊 단계별 데이터 흐름 상세

### 1️⃣ 초기화 단계 (train_ppo.py: PPOTrainer.__init__)

```
┌─────────────────────────────────────────────────────────┐
│ 1. 데이터 수집기 초기화                                  │
│    DataCollector(use_saved_data=True)                   │
│    → eth_data: DataFrame (원본 OHLCV)                    │
│    → btc_data: DataFrame (BTC 상관관계용)               │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 피처 캐싱 시스템 (_load_or_create_features)          │
│                                                          │
│    IF data/training_features.csv 존재:                  │
│      → CSV 로드 (1초 이내)                              │
│      → eth_data에 피처 포함 DataFrame 할당              │
│                                                          │
│    ELSE:                                                 │
│      → FeatureEngineer.generate_features()              │
│        * 25개 기본 기술적 지표 생성                      │
│      → MTFProcessor.add_mtf_features()                  │
│        * 4개 멀티 타임프레임 지표 추가                   │
│      → CSV 저장 (data/training_features.csv)            │
│                                                          │
│    결과: eth_data = DataFrame (175,202행 × 29+ 컬럼)   │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 3. 스케일러 학습 (_fit_global_scaler)                   │
│                                                          │
│    - 29개 타겟 컬럼 선택                                 │
│    - 50,000개 샘플 무작위 추출                           │
│    - DataPreprocessor.fit() 실행                         │
│      → mean, std 계산 (Z-Score 정규화용)                 │
│    - 스케일러 저장 (ppo_model_scaler.pkl)               │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 환경 및 에이전트 생성                                 │
│                                                          │
│    TradingEnvironment(data_collector, strategies)        │
│    PPOAgent(state_dim=29, info_dim=15, action_dim=3)   │
└─────────────────────────────────────────────────────────┘
```

### 2️⃣ 에피소드 실행 단계 (train_ppo.py: train_episode)

```
for episode in range(num_episodes):
    ┌─────────────────────────────────────────────────────┐
    │ 에피소드 시작                                        │
    │   - reset_index(random_start=True)                  │
    │   - 포지션 상태 초기화                               │
    └─────────────────────────────────────────────────────┘
                    ↓
    for step in range(max_steps):
        ┌─────────────────────────────────────────────────┐
        │ Step 1: 인덱스 증가                              │
        │   current_index += 1                            │
        └─────────────────────────────────────────────────┘
                    ↓
        ┌─────────────────────────────────────────────────┐
        │ Step 2: 상태 관측 (trading_env.py)              │
        │   → [상세 내용은 아래 참조]                    │
        └─────────────────────────────────────────────────┘
                    ↓
        ┌─────────────────────────────────────────────────┐
        │ Step 3: 행동 선택 (ppo_agent.py)                │
        │   → [상세 내용은 아래 참조]                    │
        └─────────────────────────────────────────────────┘
                    ↓
        ┌─────────────────────────────────────────────────┐
        │ Step 4: 보상 계산 및 포지션 업데이트            │
        │   → [상세 내용은 아래 참조]                    │
        └─────────────────────────────────────────────────┘
                    ↓
        ┌─────────────────────────────────────────────────┐
        │ Step 5: 트랜지션 저장                           │
        │   memory.append((state, action, log_prob,       │
        │                  reward, is_terminal))         │
        └─────────────────────────────────────────────────┘
                    ↓
        ┌─────────────────────────────────────────────────┐
        │ Step 6: 배치 업데이트 (1024개 트랜지션마다)     │
        │   IF len(memory) >= 1024:                      │
        │     → agent.update(next_state, episode)        │
        └─────────────────────────────────────────────────┘
```

### 3️⃣ 상태 관측 상세 (trading_env.py: get_observation)

```
┌─────────────────────────────────────────────────────────┐
│ 입력: position_info = [pos_val, pnl_val, hold_val]     │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 1. 원본 데이터 슬라이싱                                  │
│                                                          │
│    IF eth_data에 피처가 이미 있음 (rsi_1h 컬럼 존재):   │
│      candles = eth_data.iloc[current_idx - 40 : current_idx]│
│      → 피처 재계산 생략 (속도 향상)                     │
│                                                          │
│    ELSE:                                                 │
│      candles = get_candles('ETH', count=100)            │
│      → FeatureEngineer + MTFProcessor 실행               │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 29개 타겟 컬럼 선택                                   │
│                                                          │
│    target_cols = [                                       │
│      'log_return', 'roll_return_6', 'atr_ratio', ...   │
│      'rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h'      │
│    ]                                                     │
│                                                          │
│    recent_df = df[target_cols].iloc[-40:]              │
│    → (40, 29) numpy array                               │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Z-Score 정규화 (preprocess.py)                       │
│                                                          │
│    seq_features = recent_df.values.astype(np.float32)   │
│    normalized = (seq_features - mean) / std              │
│    → (40, 29) numpy array (정규화됨)                    │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 텐서 변환                                             │
│                                                          │
│    obs_seq = torch.FloatTensor(normalized).unsqueeze(0) │
│    → (1, 40, 29) tensor                                 │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 5. 전략 점수 수집 (12개 전략)                           │
│                                                          │
│    FOR each strategy in strategies:                      │
│      result = strategy.analyze(collector)               │
│      score = result['confidence']                        │
│      IF signal == 'SHORT': score = -score                │
│      strategy_scores.append(score)                       │
│                                                          │
│    → [12개 전략 점수] (float 리스트)                    │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 6. 포지션 정보 결합                                      │
│                                                          │
│    position_info = [                                     │
│      pos_val,      # 1.0 (LONG) / -1.0 (SHORT) / 0.0   │
│      pnl_val,      # prev_pnl * 10                      │
│      hold_val       # (current_idx - entry_idx) / max_steps│
│    ]                                                     │
│                                                          │
│    obs_info = strategy_scores + position_info           │
│    → [15개 값] (12개 전략 + 3개 포지션)                │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 7. 최종 반환                                             │
│                                                          │
│    obs_info_tensor = torch.FloatTensor(obs_info).unsqueeze(0)│
│    → (1, 15) tensor                                      │
│                                                          │
│    RETURN (obs_seq, obs_info_tensor)                    │
│    → obs_seq: (1, 40, 29)                               │
│    → obs_info: (1, 15)                                   │
└─────────────────────────────────────────────────────────┘
```

### 4️⃣ 행동 선택 상세 (ppo_agent.py → xlstm_network.py)

```
┌─────────────────────────────────────────────────────────┐
│ 입력                                                     │
│   obs_seq: (1, 40, 29) tensor                           │
│   obs_info: (1, 15) tensor                               │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 1. xLSTM 시퀀스 처리 (xlstm_network.py)                 │
│                                                          │
│    FOR t in range(40):                                  │
│      h, c, n = sLSTMCell(x[:, t, :], h, c, n)          │
│      all_h.append(h)                                    │
│                                                          │
│    seq_h = torch.cat(all_h, dim=1)                      │
│    → (1, 40, 128) tensor                                │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Multi-Head Attention                                 │
│                                                          │
│    context = attention(seq_h)                           │
│    → (1, 128) tensor (시퀀스 요약)                      │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Late Fusion                                           │
│                                                          │
│    combined = torch.cat([context, obs_info], dim=-1)   │
│    → (1, 143) tensor (128 + 15)                         │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Actor Head (정책 네트워크)                           │
│                                                          │
│    combined → Linear(143 → 128) → LayerNorm → GELU     │
│            → Linear(128 → 64) → GELU                    │
│            → Linear(64 → 3) → Softmax                   │
│                                                          │
│    action_probs = (1, 3) tensor                          │
│    → [P(Hold), P(Long), P(Short)]                       │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Critic Head (가치 네트워크)                          │
│                                                          │
│    combined → Linear(143 → 128) → LayerNorm → GELU     │
│            → Linear(128 → 64) → GELU                    │
│            → Linear(64 → 1)                             │
│                                                          │
│    value = (1, 1) tensor (상태 가치)                    │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 6. 행동 샘플링                                           │
│                                                          │
│    dist = Categorical(action_probs)                     │
│    action = dist.sample()  # 0, 1, 또는 2               │
│    log_prob = dist.log_prob(action)                      │
│                                                          │
│    RETURN action.item(), log_prob.item()                │
└─────────────────────────────────────────────────────────┘
```

### 5️⃣ 보상 계산 상세 (trading_env.py: calculate_reward)

```
┌─────────────────────────────────────────────────────────┐
│ 입력 파라미터                                             │
│   - pnl: 수익률 (float, 예: 0.001 = 0.1%)               │
│   - trade_done: 거래 완료 여부 (bool)                    │
│   - holding_time: 보유 시간 (int, 캔들 수)              │
│   - pnl_change: 수익률 변화량 (float)                    │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 1. 평가 수익 변화량 보상                                 │
│                                                          │
│    reward = pnl_change * 300                            │
│    → 포지션 보유 중 수익률 변화에 대한 즉각 보상         │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 거래 완료 시 추가 보상/페널티                        │
│                                                          │
│    IF trade_done:                                       │
│      IF pnl > 0:  # 수익                                │
│        reward += pnl * 300              # 선형 보상      │
│        reward += sqrt(pnl * 100) * 0.5  # 제곱근 보너스│
│        reward += tanh(pnl * 100) * 0.5  # 승리 보너스  │
│                                                          │
│      ELSE:  # 손실                                       │
│        reward += pnl * 350              # 손실 페널티   │
│                                                          │
│      reward -= 0.2                    # 수수료 페널티   │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 3. 시간 비용                                             │
│                                                          │
│    reward -= 0.0005  # 보유 시간에 따른 미세 페널티     │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 클리핑 및 반환                                        │
│                                                          │
│    reward = clip(reward, -100, 100)                     │
│    RETURN reward                                         │
└─────────────────────────────────────────────────────────┘
```

### 6️⃣ 학습 업데이트 상세 (ppo_agent.py: update)

```
┌─────────────────────────────────────────────────────────┐
│ 1. 메모리에서 데이터 추출                                │
│                                                          │
│    states_seq: (N, 40, 29) tensor                       │
│    states_info: (N, 15) tensor                          │
│    actions: (N, 1) tensor                               │
│    old_log_probs: (N, 1) tensor                         │
│    rewards: [N개 float]                                 │
│    is_terminals: [N개 bool]                             │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 2. GAE 계산 (Generalized Advantage Estimation)          │
│                                                          │
│    - 다음 상태 가치 계산 (Bootstrap)                    │
│    - 현재 상태들의 가치 계산                             │
│    - 역순으로 GAE 계산                                   │
│    - Advantage 정규화                                   │
│                                                          │
│    → advantages: (N, 1) tensor                         │
│    → returns: (N, 1) tensor                             │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ 3. PPO 업데이트 (k_epochs=10 반복)                      │
│                                                          │
│    FOR epoch in range(10):                              │
│      # Forward pass                                     │
│      probs, curr_values = model(states_seq, states_info)│
│      dist = Categorical(probs)                         │
│      curr_log_probs = dist.log_prob(actions.squeeze())  │
│      entropy = dist.entropy().mean()                    │
│                                                          │
│      # Importance sampling ratio                        │
│      ratio = exp(curr_log_probs - old_log_probs)       │
│                                                          │
│      # Clipped surrogate objective                      │
│      surr1 = ratio * advantages                         │
│      surr2 = clip(ratio, 0.8, 1.2) * advantages        │
│      actor_loss = -min(surr1, surr2).mean()            │
│                                                          │
│      # Critic loss                                      │
│      critic_loss = MSE(curr_values, returns)            │
│                                                          │
│      # Entropy bonus (탐험 유도)                        │
│      entropy_coef = max(0.02, 0.2 * (0.999 ** episode))│
│      entropy_bonus = entropy_coef * entropy            │
│                                                          │
│      # Total loss                                       │
│      loss = actor_loss + 0.5 * critic_loss - entropy_bonus│
│                                                          │
│      # Backward pass                                    │
│      optimizer.zero_grad()                              │
│      loss.backward()                                    │
│      clip_grad_norm_(parameters, 0.5)                   │
│      optimizer.step()                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 📐 주요 데이터 형태 요약

| 단계 | 변수명 | 형태 | 설명 |
|------|--------|------|------|
| **원본 데이터** | `eth_data` | `(175202, 29+)` DataFrame | 피처가 포함된 전체 데이터 |
| **슬라이싱** | `candles` | `(40, 29+)` DataFrame | 현재 인덱스 기준 40개 캔들 |
| **피처 선택** | `recent_df` | `(40, 29)` numpy array | 29개 타겟 컬럼만 선택 |
| **정규화** | `normalized` | `(40, 29)` numpy array | Z-Score 정규화됨 |
| **시계열 텐서** | `obs_seq` | `(1, 40, 29)` tensor | 시계열 피처 |
| **정보 텐서** | `obs_info` | `(1, 15)` tensor | 전략 점수(12) + 포지션 정보(3) |
| **xLSTM 출력** | `seq_h` | `(1, 40, 128)` tensor | 시퀀스 처리 결과 |
| **Attention 출력** | `context` | `(1, 128)` tensor | 시퀀스 요약 |
| **Late Fusion** | `combined` | `(1, 143)` tensor | 128 + 15 |
| **행동 확률** | `action_probs` | `(1, 3)` tensor | [P(Hold), P(Long), P(Short)] |
| **상태 가치** | `value` | `(1, 1)` tensor | V(s) |
| **배치 시퀀스** | `states_seq` | `(N, 40, 29)` tensor | N개 트랜지션 |
| **배치 정보** | `states_info` | `(N, 15)` tensor | N개 정보 벡터 |
| **Advantages** | `advantages` | `(N, 1)` tensor | GAE 계산 결과 |
| **Returns** | `returns` | `(N, 1)` tensor | 목표 가치 |

---

## 🔑 핵심 모듈 역할

### 1. **feature_engineering.py** - 기본 피처 생성
- **입력**: ETH/BTC 원본 OHLCV 데이터
- **출력**: 25개 기술적 지표가 포함된 DataFrame
- **주요 피처**: RSI, MACD, BB, ATR, CCI, MFI, CMF, VWAP 등

### 2. **mtf_processor.py** - 멀티 타임프레임 피처
- **입력**: 3분봉 피처 DataFrame
- **출력**: 15분봉, 1시간봉 지표 추가 (4개)
- **Look-ahead Bias 방지**: Shift 적용

### 3. **preprocess.py** - 데이터 정규화
- **역할**: Z-Score 정규화 스케일러
- **fit()**: 전역 통계량 계산 (mean, std)
- **transform()**: 정규화 적용
- **save_scaler()**: 스케일러 저장/로드

### 4. **trading_env.py** - 강화학습 환경
- **get_observation()**: 상태 관측 생성
  - 피처 슬라이싱 → 정규화 → 텐서 변환
  - 전략 점수 수집 (12개)
  - 포지션 정보 결합 (3개)
- **calculate_reward()**: 보상 계산
  - 수익률 기반 보상 (배율 300)
  - 수수료/시간 비용 반영

### 5. **xlstm_network.py** - 신경망 구조
- **sLSTMCell**: 시퀀스 처리 (Exponential Gating)
- **MultiHeadAttention**: 시퀀스 요약
- **Late Fusion**: 시계열 정보(128) + 포지션 정보(15) 결합
- **Actor/Critic Head**: 행동 확률 및 상태 가치 출력

### 6. **ppo_agent.py** - PPO 알고리즘
- **select_action()**: 행동 선택 (탐험)
- **store_transition()**: 트랜지션 저장
- **update()**: PPO 업데이트 (GAE + Clipped Surrogate)

### 7. **train_ppo.py** - 학습 메인 스크립트
- **피처 캐싱**: CSV 로드/저장으로 속도 최적화
- **스케일러 학습**: 전역 통계량 계산
- **에피소드 루프**: 학습 실행 및 모델 저장

---

## 🎯 핵심 설계 원칙

1. **피처 캐싱**: `training_features.csv`로 피처 재계산 방지
2. **직접 데이터 접근**: `get_candles` 대신 `eth_data` 직접 슬라이싱
3. **피처 존재 확인**: `rsi_1h` 컬럼으로 재계산 여부 판단
4. **동적 차원**: 전략 개수에 따라 `info_dim` 자동 조정
5. **Late Fusion**: 시계열 정보와 포지션 정보 분리 처리
6. **전역 정규화**: 전체 데이터셋의 통계량으로 정규화
7. **배치 업데이트**: 1024개 트랜지션마다 한 번만 업데이트
8. **엔트로피 탐험**: 초기 0.2 → 최소 0.02로 점진적 감소

---

## 📊 29개 피처 목록

### 가격/변동성 (9개)
- `log_return`, `roll_return_6`, `atr_ratio`, `bb_width`, `bb_pos`
- `rsi`, `macd_hist`, `hma_ratio`, `cci`

### 거래량 (6개)
- `rvol`, `taker_ratio`, `cvd_change`, `mfi`, `cmf`, `vwap_dist`

### 패턴 (5개)
- `wick_upper`, `wick_lower`, `range_pos`, `swing_break`, `chop`

### 상관관계 (5개)
- `btc_return`, `btc_rsi`, `btc_corr`, `btc_vol`, `eth_btc_ratio`

### 멀티 타임프레임 (4개)
- `rsi_15m`, `trend_15m`, `rsi_1h`, `trend_1h`

---

## 🔄 학습 루프 요약

```
에피소드 시작
  ↓
무작위 시작 인덱스 선택
  ↓
for step in range(max_steps):  # 최대 160 스텝
  ↓
  current_index += 1
  ↓
  상태 관측 (40봉 × 29개 피처 + 15개 정보)
  ↓
  행동 선택 (xLSTM → Actor → Categorical 샘플링)
  ↓
  보상 계산 (수익률 기반, 배율 300)
  ↓
  트랜지션 저장
  ↓
  if len(memory) >= 1024:
    → PPO 업데이트 (GAE + Clipped Surrogate)
    → 메모리 초기화
  ↓
에피소드 종료
  ↓
모델 저장 (최고 성능 또는 주기적)
```

---

## 💾 저장되는 파일

1. **data/training_features.csv**: 피처가 포함된 전체 데이터 (캐싱용)
2. **data/ppo_model.pth**: 학습된 PPO 모델 가중치
3. **data/ppo_model_scaler.pkl**: Z-Score 정규화 스케일러 (mean, std)

---

## 🚀 성능 최적화 포인트

1. **피처 캐싱**: 최초 1회만 계산, 이후 CSV 로드 (1초 이내)
2. **직접 데이터 접근**: `get_candles` 필터링 우회
3. **피처 존재 확인**: 재계산 생략으로 속도 향상
4. **배치 업데이트**: 1024개 트랜지션마다 한 번만 업데이트
5. **스케일러 사전 학습**: 학습 시작 전 전역 스케일러 학습

---

## 📝 참고

- **전략 개수**: 12개 (동적, config 기반)
- **Lookback**: 40봉 (시계열 윈도우)
- **State Dim**: 29개 (시계열 피처)
- **Info Dim**: 15개 (전략 12 + 포지션 3)
- **Action Dim**: 3개 (Hold, Long, Short)
- **Hidden Dim**: 128개 (xLSTM 히든 차원)
