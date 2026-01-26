# 데이터 흐름 명세서 (Data Flow Specification)

## 📋 목차
1. [전체 데이터 파이프라인](#1-전체-데이터-파이프라인)
2. [데이터 수집 단계](#2-데이터-수집-단계)
3. [피처 엔지니어링 단계](#3-피처-엔지니어링-단계)
4. [전략 신호 계산 단계](#4-전략-신호-계산-단계)
5. [전처리 및 정규화 단계](#5-전처리-및-정규화-단계)
6. [학습 시 데이터 흐름](#6-학습-시-데이터-흐름)
7. [추론 시 데이터 흐름](#7-추론-시-데이터-흐름)

---

## 1. 전체 데이터 파이프라인

### 1.1 파이프라인 개요

```
[원본 데이터]
    ├─ data/eth_3m_1year.csv (ETH/USDT 3분봉)
    └─ data/btc_3m_1year.csv (BTC/USDT 3분봉, 선택적)
    ↓
[데이터 로딩] (DataCollector)
    ├─ CSV 파일 읽기
    ├─ 인덱스 정렬 (DatetimeIndex)
    └─ 기본 컬럼 검증
    ↓
[피처 엔지니어링] (FeatureEngineer)
    ├─ 가격 & 변동성 피처 (9개)
    ├─ 거래량 & 오더플로우 피처 (6개)
    ├─ 패턴 & 유동성 피처 (5개)
    └─ 시장 상관관계 피처 (5개)
    → 총 25개 기본 피처
    ↓
[MTF 처리] (MTFProcessor)
    ├─ 15분봉 리샘플링 → RSI_15m, Trend_15m
    └─ 1시간봉 리샘플링 → RSI_1h, Trend_1h
    → 총 29개 피처 (25 + 4)
    ↓
[전략 신호 계산] (StrategyCalculator)
    ├─ 폭발장 전략 (6개)
    └─ 횡보장 전략 (6개)
    → 총 12개 전략 점수
    ↓
[데이터 저장]
    ├─ data/training_features.csv (29개 피처)
    └─ data/cached_strategies.csv (12개 전략)
    ↓
[스케일러 학습] (DataPreprocessor)
    ├─ Train Set 70%만 사용
    ├─ Mean, Std 계산
    └─ 저장: data/ppo_model_best_scaler.pkl
    ↓
[학습 루프] (train_ppo.py)
    ├─ 에피소드 시작
    ├─ 관측 생성 (get_observation)
    ├─ 행동 선택 (select_action)
    ├─ 보상 계산 (calculate_reward)
    └─ PPO 업데이트 (train_net)
```

### 1.2 데이터 흐름 단계별 상세

---

## 2. 데이터 수집 단계

### 2.1 입력 데이터 형식

**ETH/USDT 3분봉 데이터** (`data/eth_3m_1year.csv`)
```csv
timestamp,open,high,low,close,volume,taker_buy_volume,cvd
2024-01-01 00:00:00,2500.0,2505.0,2498.0,2502.0,1000.5,600.3,50000.0
2024-01-01 00:03:00,2502.0,2508.0,2500.0,2506.0,1200.2,700.5,52000.0
...
```

**필수 컬럼:**
- `timestamp`: DatetimeIndex (또는 인덱스로 설정)
- `open`, `high`, `low`, `close`: OHLC 가격 데이터
- `volume`: 거래량
- `taker_buy_volume`: 테이커 매수 거래량
- `cvd`: Cumulative Volume Delta

**BTC/USDT 3분봉 데이터** (`data/btc_3m_1year.csv`, 선택적)
```csv
timestamp,close,volume
2024-01-01 00:00:00,42000.0,5000.0
2024-01-01 00:03:00,42050.0,5500.0
...
```

### 2.2 데이터 로딩 프로세스

**DataCollector 초기화:**
```python
collector = DataCollector(
    eth_symbol='ETHUSDT',
    btc_symbol='BTCUSDT',
    timeframe='3m',
    lookback_period=1500
)

# 데이터 로드
collector.load_data()
# → self.eth_data: DataFrame
# → self.btc_data: DataFrame (선택적)
```

**데이터 검증:**
- 인덱스가 DatetimeIndex인지 확인
- 필수 컬럼 존재 여부 확인
- NaN 값 처리 (forward fill)

---

## 3. 피처 엔지니어링 단계

### 3.1 FeatureEngineer 프로세스

**초기화:**
```python
engineer = FeatureEngineer(
    eth_data=collector.eth_data,
    btc_data=collector.btc_data  # 선택적
)

# 피처 생성
df_features = engineer.generate_features()
```

### 3.2 피처 생성 상세

#### 3.2.1 가격 & 변동성 피처 (9개)

**1. log_return**
```python
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
```

**2. roll_return_6**
```python
df['roll_return_6'] = df['close'].pct_change(6)
```

**3. atr_ratio**
```python
atr = calculate_atr(df, period=14)
df['atr_ratio'] = atr / df['close']
```

**4. bb_width**
```python
bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df, period=20)
df['bb_width'] = (bb_upper - bb_lower) / bb_middle
```

**5. bb_pos**
```python
df['bb_pos'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
```

**6. rsi**
```python
df['rsi'] = calculate_rsi(df['close'], period=14)
```

**7. macd_hist**
```python
macd, signal, hist = calculate_macd(df['close'])
df['macd_hist'] = hist
```

**8. hma_ratio**
```python
hma = calculate_hma(df['close'], period=21)
df['hma_ratio'] = (df['close'] - hma) / hma
```

**9. cci**
```python
df['cci'] = calculate_cci(df, period=20)
```

#### 3.2.2 거래량 & 오더플로우 피처 (6개)

**10. rvol (상대 거래량)**
```python
df['rvol'] = df['volume'] / df['volume'].rolling(20).mean()
```

**11. taker_ratio**
```python
df['taker_ratio'] = df['taker_buy_volume'] / df['volume']
```

**12. cvd_change**
```python
df['cvd_change'] = df['cvd'].diff()
```

**13. mfi**
```python
df['mfi'] = calculate_mfi(df, period=14)
```

**14. cmf**
```python
df['cmf'] = calculate_cmf(df, period=20)
```

**15. vwap_dist**
```python
vwap = calculate_vwap(df)
df['vwap_dist'] = (df['close'] - vwap) / vwap
```

#### 3.2.3 패턴 & 유동성 피처 (5개)

**16. wick_upper**
```python
df['wick_upper'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'])
```

**17. wick_lower**
```python
df['wick_lower'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'])
```

**18. range_pos**
```python
df['range_pos'] = (df['close'] - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min())
```

**19. swing_break**
```python
# 스윙 하이/로우 브레이크 플래그
df['swing_break'] = calculate_swing_break(df)
```

**20. chop**
```python
df['chop'] = calculate_chop_index(df, period=14)
```

#### 3.2.4 시장 상관관계 피처 (5개)

**21. btc_return**
```python
df['btc_return'] = btc_df['close'].pct_change()
```

**22. btc_rsi**
```python
df['btc_rsi'] = calculate_rsi(btc_df['close'], period=14)
```

**23. btc_corr**
```python
df['btc_corr'] = df['close'].rolling(20).corr(btc_df['close'])
```

**24. btc_vol**
```python
df['btc_vol'] = btc_df['close'].pct_change().rolling(20).std()
```

**25. eth_btc_ratio**
```python
df['eth_btc_ratio'] = df['close'] / btc_df['close']
```

### 3.3 피처 저장

**결과 저장:**
```python
df_features.to_csv('data/training_features.csv', index=True)
```

**컬럼 목록:**
- 25개 기본 피처 (위의 1~25번)
- 인덱스: DatetimeIndex

---

## 4. MTF 처리 단계

### 4.1 MTFProcessor 프로세스

**초기화:**
```python
mtf = MTFProcessor(df_3m=df_features)
df_with_mtf = mtf.add_mtf_features()
```

### 4.2 MTF 피처 생성

#### 4.2.1 15분봉 처리

**리샘플링:**
```python
df_15m = df_3m.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})
```

**지표 계산:**
```python
df_15m['rsi_15m'] = calculate_rsi(df_15m['close'], period=14)
df_15m['trend_15m'] = calculate_trend(df_15m['close'])
```

**Look-ahead Bias 방지:**
```python
df_15m_shifted = df_15m.shift(1)  # 한 칸 밑으로 내림
```

**병합:**
```python
merged_15m = df_15m_shifted.reindex(df_3m.index, method='ffill')
df_3m['rsi_15m'] = merged_15m['rsi_15m']
df_3m['trend_15m'] = merged_15m['trend_15m']
```

#### 4.2.2 1시간봉 처리

**리샘플링:**
```python
df_1h = df_3m.resample('1h').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})
```

**지표 계산:**
```python
df_1h['rsi_1h'] = calculate_rsi(df_1h['close'], period=14)
df_1h['trend_1h'] = calculate_trend(df_1h['close'])
```

**Look-ahead Bias 방지:**
```python
df_1h_shifted = df_1h.shift(1)
```

**병합:**
```python
merged_1h = df_1h_shifted.reindex(df_3m.index, method='ffill')
df_3m['rsi_1h'] = merged_1h['rsi_1h']
df_3m['trend_1h'] = merged_1h['trend_1h']
```

### 4.3 최종 피처 개수

**총 29개 피처:**
- 기본 피처: 25개
- MTF 피처: 4개 (rsi_15m, trend_15m, rsi_1h, trend_1h)

---

## 5. 전략 신호 계산 단계

### 5.1 StrategyCalculator 프로세스

**전략 목록:**
```python
STRATEGIES = {
    # 폭발장 전략
    'btc_eth_correlation': True,
    'volatility_squeeze': True,
    'orderblock_fvg': True,
    'hma_momentum': True,
    'mfi_momentum': True,
    # 횡보장 전략
    'bollinger_mean_reversion': True,
    'vwap_deviation': True,
    'range_top_bottom': True,
    'stoch_rsi_mean_reversion': True,
    'cmf_divergence': True
}
```

### 5.2 전략 신호 계산

**각 전략별 신호 계산:**
```python
for i, (strategy_name, enabled) in enumerate(STRATEGIES.items()):
    if enabled:
        signal = calculate_strategy_signal(df, strategy_name)
        df[f'strategy_{i}'] = signal  # -1.0 ~ 1.0 범위
    else:
        df[f'strategy_{i}'] = 0.0
```

**전략 신호 범위:**
- `-1.0`: 강한 매도 신호
- `0.0`: 중립
- `1.0`: 강한 매수 신호

### 5.3 전략 신호 저장

**캐시 저장:**
```python
strategy_cols = [f'strategy_{i}' for i in range(len(STRATEGIES))]
df_strategies = df[strategy_cols]
df_strategies.to_csv('data/cached_strategies.csv', index=True)
```

**총 12개 전략 점수:**
- `strategy_0` ~ `strategy_11`

---

## 6. 전처리 및 정규화 단계

### 6.1 DataPreprocessor 프로세스

**초기화:**
```python
preprocessor = DataPreprocessor()
```

**스케일러 학습:**
```python
# Train Set 70%만 사용 (Data Leakage 방지)
train_size = int(len(df) * 0.7)
train_df = df.iloc[:train_size]

# 샘플링 (메모리 효율)
sample_size = min(50000, len(train_df))
train_sample = train_df.sample(n=sample_size, random_state=42)

# 피처 추출
features = train_sample[target_cols].values  # (sample_size, 29)

# 스케일러 학습
preprocessor.fit(features)
# → mean, std 계산
```

**스케일러 저장:**
```python
preprocessor.save('data/ppo_model_best_scaler.pkl')
```

### 6.2 정규화 공식

**Z-Score Normalization:**
```python
normalized = (x - mean) / std
```

**각 피처별 독립적 정규화:**
- Mean: `(sample_size, 29)` → `(29,)`
- Std: `(sample_size, 29)` → `(29,)`
- Normalized: `(batch, seq_len, 29)`

### 6.3 정규화 적용

**학습 시:**
```python
# 관측 생성 시
features = df[target_cols].iloc[start:end].values  # (seq_len, 29)
normalized = preprocessor.transform(features)  # (seq_len, 29)
obs_seq = torch.FloatTensor(normalized).unsqueeze(0)  # (1, seq_len, 29)
```

---

## 7. 학습 시 데이터 흐름

### 7.1 에피소드 시작

**랜덤 시작 인덱스:**
```python
start_min = LOOKBACK + 100  # 최소 160
start_max = train_end_idx - max_steps - 50
start_idx = np.random.randint(start_min, start_max)
collector.current_index = start_idx
```

### 7.2 스텝 루프 데이터 흐름

#### 7.2.1 관측 생성 (`get_observation`)

**입력:**
- `current_index`: 현재 인덱스
- `position_info`: `[pos_val, unrealized_pnl*10, holding_time/max_steps]`

**프로세스:**
```python
# 1. 시계열 피처 추출
recent_df = df[target_cols].iloc[current_idx - LOOKBACK : current_idx]
# → (60, 29)

# 2. 정규화
normalized_seq = preprocessor.transform(recent_df.values)
# → (60, 29)

# 3. 텐서 변환
obs_seq = torch.FloatTensor(normalized_seq).unsqueeze(0)
# → (1, 60, 29)

# 4. 전략 신호 추출
strategy_scores = [df[f'strategy_{i}'].iloc[current_idx] for i in range(12)]
# → (12,)

# 5. 정보 벡터 결합
obs_info = np.concatenate([strategy_scores, position_info])
# → (15,)

# 6. 텐서 변환
obs_info_tensor = torch.FloatTensor(obs_info).unsqueeze(0)
# → (1, 15)

# 7. 반환
return (obs_seq, obs_info_tensor)
```

#### 7.2.2 행동 선택 (`select_action`)

**입력:**
- `state`: `(obs_seq, obs_info)`

**프로세스:**
```python
# 1. LSTM 상태 유지
probs, value, new_states = model(
    obs_seq, obs_info,
    states=current_states,  # 에피소드 내 유지
    return_states=True
)
# → probs: (1, 4), value: (1, 1)

# 2. 행동 샘플링
dist = Categorical(probs)
action = dist.sample()  # 0, 1, 2, 3 중 하나

# 3. 반환
return action.item(), dist.log_prob(action).item()
```

#### 7.2.3 거래 로직 실행

**평가손익 계산:**
```python
if current_position == 'LONG':
    unrealized_pnl = (curr_price - entry_price) / entry_price
elif current_position == 'SHORT':
    unrealized_pnl = (entry_price - curr_price) / entry_price
else:
    unrealized_pnl = 0.0

step_pnl = unrealized_pnl - prev_unrealized_pnl
```

**행동 처리:**
```python
if action == 1:  # LONG
    if current_position == 'SHORT':
        realized_pnl = unrealized_pnl
        trade_done = True
        # 스위칭
    elif current_position is None:
        # 진입
        current_position = 'LONG'
        entry_price = curr_price
        entry_index = current_idx

elif action == 2:  # SHORT
    # 유사한 로직

elif action == 3:  # EXIT
    if current_position is not None:
        realized_pnl = unrealized_pnl
        trade_done = True
        current_position = None
```

#### 7.2.4 보상 계산 (`calculate_reward`)

**입력:**
- `step_pnl`: 평가손익 변화
- `realized_pnl`: 확정 손익 (Exit 시)
- `trade_done`: 거래 종료 여부
- `action`: 행동 (0, 1, 2, 3)
- `prev_position`: 이전 포지션

**프로세스:**
```python
reward = 0.0

# 1. HOLD 보너스
if action == 0:
    reward += 0.0002

# 2. Position Holding Reward
if prev_position is not None:
    reward += step_pnl * 30.0

# 3. Switching Penalty
if trade_done and (action == 1 or action == 2):
    reward -= 0.5

# 4. EXIT Rewards
if trade_done and action == 3:
    fee = 0.0015
    net_pnl = realized_pnl - fee
    if net_pnl > 0:
        reward += net_pnl * 250.0
        if net_pnl > 0.005:
            reward += 1.0
    else:
        reward += net_pnl * 300.0
        reward -= 0.2

# 5. Clipping
reward = np.clip(reward, -10, 10)
```

#### 7.2.5 Transition 저장

**저장:**
```python
transition = (
    state,           # (obs_seq, obs_info)
    action,        # int
    reward,         # float
    next_state,     # (obs_seq, obs_info)
    prob,           # float (log_prob)
    done            # bool
)
agent.put_data(transition)
```

### 7.3 PPO 업데이트 (`train_net`)

**배치 구성:**
```python
# 메모리에서 배치 추출
s_list, a_list, r_list, next_s_list, prob_list, done_list = [], [], [], [], [], []
for data in agent.memory:
    s, a, r, next_s, prob, done = data
    # 리스트에 추가

# 텐서 변환
s_seq_batch = torch.cat([item[0] for item in s_list], dim=0)  # (batch, 60, 29)
s_info_batch = torch.cat([item[1] for item in s_list], dim=0)  # (batch, 15)
a_batch = torch.tensor(a_list, dtype=torch.long)  # (batch,)
r_batch = torch.tensor(r_list, dtype=torch.float)  # (batch,)
```

**GAE 계산:**
```python
# Value 계산
_, v_s = model(s_seq_batch, s_info_batch)
_, v_next = model(ns_seq_batch, ns_info_batch)

# TD Target
td_target = r_batch + gamma * v_next * done_batch
delta = td_target - v_s

# GAE (역방향)
advantages = []
gae = 0
for step in reversed(range(len(r_batch))):
    if done_batch[step] == 0:
        gae = delta[step] + gamma * lambda * gae
    else:
        gae = delta[step]
    advantages.insert(0, gae)

# 정규화
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
returns = advantages + v_s.squeeze()
```

**PPO Loss 계산:**
```python
for epoch in range(k_epochs):
    # 현재 정책
    pi_probs, v_s = model(s_seq_batch, s_info_batch)
    dist = Categorical(pi_probs)
    pi_a = dist.log_prob(a_batch).unsqueeze(1)
    
    # Ratio
    ratio = torch.exp(pi_a - prob_a_batch)
    
    # Actor Loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    
    # Critic Loss (Value Clipping)
    old_values = v_s.clone()
    loss_v1 = F.smooth_l1_loss(v_s.squeeze(), returns, reduction='none')
    v_pred_clipped = old_values.squeeze() + torch.clamp(
        v_s.squeeze() - old_values.squeeze(), -eps_clip, eps_clip
    )
    loss_v2 = F.smooth_l1_loss(v_pred_clipped, returns, reduction='none')
    critic_loss = torch.max(loss_v1, loss_v2).mean()
    
    # Entropy
    entropy_loss = dist.entropy().mean()
    current_entropy_coef = max(ENTROPY_MIN, ENTROPY_COEF * (ENTROPY_DECAY ** episode))
    
    # Total Loss
    loss = actor_loss + 0.5 * critic_loss - current_entropy_coef * entropy_loss
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
```

---

## 8. 추론 시 데이터 흐름

### 8.1 모델 로드

```python
agent = PPOAgent(state_dim=29, action_dim=4, info_dim=15)
agent.load_model('data/ppo_model_best.pth')

preprocessor = DataPreprocessor()
preprocessor.load('data/ppo_model_best_scaler.pkl')
```

### 8.2 실시간 추론

**관측 생성:**
```python
# 현재 인덱스에서 관측 생성
state = env.get_observation(
    position_info=[pos_val, unrealized_pnl*10, holding_time/max_steps],
    current_index=current_idx
)
```

**행동 선택:**
```python
action, _ = agent.select_action(state)
```

**행동 실행:**
```python
if action == 1:
    # LONG 진입/스위칭
elif action == 2:
    # SHORT 진입/스위칭
elif action == 3:
    # EXIT
else:
    # HOLD
```

---

## 9. 데이터 검증 체크리스트

### 9.1 입력 데이터 검증
- [ ] DatetimeIndex 형식 확인
- [ ] 필수 컬럼 존재 확인
- [ ] NaN 값 처리 확인
- [ ] 데이터 범위 검증 (가격 > 0, 거래량 >= 0)

### 9.2 피처 검증
- [ ] 29개 피처 모두 생성 확인
- [ ] MTF 피처 Look-ahead Bias 방지 확인
- [ ] Inf/NaN 값 처리 확인

### 9.3 전략 신호 검증
- [ ] 12개 전략 점수 범위 확인 (-1.0 ~ 1.0)
- [ ] 캐시 파일 일관성 확인

### 9.4 정규화 검증
- [ ] 스케일러 학습 데이터 분리 확인 (Train 70%)
- [ ] 정규화 후 분포 확인 (Mean ≈ 0, Std ≈ 1)

### 9.5 학습 데이터 검증
- [ ] 관측 Shape 확인 ((1, 60, 29), (1, 15))
- [ ] 보상 범위 확인 ([-10, 10])
- [ ] Transition 저장 확인

---

## 10. 성능 최적화

### 10.1 데이터 캐싱
- 피처 엔지니어링 결과를 CSV로 저장
- 전략 신호를 캐시 파일로 저장
- 스케일러를 pickle로 저장

### 10.2 메모리 효율화
- 스케일러 학습 시 샘플링 (50,000개)
- 배치 처리로 메모리 사용량 최적화

### 10.3 계산 최적화
- 전략 신호 병렬 계산 (joblib)
- 벡터화 연산 사용 (NumPy, Pandas)

---

**문서 버전**: v1.0  
**최종 업데이트**: 2026-01-23
