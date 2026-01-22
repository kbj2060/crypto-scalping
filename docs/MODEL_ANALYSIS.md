# Model 구조 및 데이터 흐름 분석 리포트

## 📊 데이터 흐름 (Data Flow)

### 1. 피처 엔지니어링 파이프라인

```
원본 데이터 (ETH/USDT)
    ↓
FeatureEngineer.generate_features()
    ├─ 가격 & 변동성 (9개): log_return, roll_return_6, atr_ratio, bb_width, bb_pos, rsi, macd_hist, hma_ratio, cci
    ├─ 거래량 & 오더플로우 (6개): rvol, taker_ratio, cvd_change, mfi, cmf, vwap_dist
    ├─ 패턴 & 유동성 (5개): wick_upper, wick_lower, range_pos, swing_break, chop
    └─ 시장 상관관계 (5개): btc_return, btc_rsi, btc_corr, btc_vol, eth_btc_ratio
    ↓
MTFProcessor.add_mtf_features()
    └─ 상위 프레임 지표 (4개): rsi_15m, trend_15m, rsi_1h, trend_1h
    ↓
precalculate_strategy_scores()
    └─ 전략 점수 (12개): strat_btc_eth_corr, strat_vol_squeeze, ..., strat_cci_reversal, strat_williams_r
    ↓
XGBoost Feature Selection (선택적)
    └─ TOP_K_FEATURES (기본 25개) 선택
    ↓
최종 피처 리스트 (약 20~30개)
```

### 2. 학습 파이프라인

```
DDQNTrainer.__init__()
    ├─ 데이터 로드
    ├─ FeatureEngineer 적용
    ├─ MTFProcessor 적용
    ├─ 전략 점수 계산
    ├─ XGBoost 피처 선택 (선택적)
    ├─ 스케일러 학습 (_fit_global_scaler)
    └─ DDQNAgent 초기화
    ↓
train_episode()
    ├─ 랜덤 시작점 선택
    ├─ 각 스텝마다:
    │   ├─ get_observation() → (obs_seq, obs_info)
    │   ├─ agent.act() → action
    │   ├─ 매매 로직 실행
    │   ├─ calculate_reward() → reward
    │   ├─ agent.remember() → N-step 버퍼에 저장
    │   └─ agent.train_step() → 학습
    └─ 에피소드 종료
```

## 🔍 발견된 문제점

### ❌ **치명적 문제 1: obs_info가 모델에 전달되지 않음**

**현재 상황:**
- `trading_env.py`: `(obs_seq, obs_info)` 튜플 반환
- `dqn_agent.py`의 `act()`: `obs_seq`만 사용
- `DuelingGRU` 모델: `obs_seq`만 받음
- **결과**: 포지션 정보(3차원)가 모델에 전달되지 않음!

**영향:**
- 모델이 현재 포지션 상태, PnL, 보유 시간을 알 수 없음
- 동일한 차트 패턴이라도 포지션에 따라 다른 행동이 필요한데 구분 불가

**수정 필요:**
```python
# dqn_model.py의 DuelingGRU.forward() 수정
def forward(self, x, info=None):
    # x: (batch, seq, input_dim)
    # info: (batch, 3) - 포지션 정보
    
    # 기존 로직...
    context_vector = self.attention(gru_out)
    
    # [추가] info 통합
    if info is not None:
        context_vector = torch.cat([context_vector, info], dim=-1)
        # 또는 별도 FC 레이어로 통합
    
    value = self.value_stream(context_vector)
    advantage = self.advantage_stream(context_vector)
    ...
```

### ⚠️ **문제 2: 스케일러 차원 불일치 (부분 해결됨)**

**현재 상황:**
- `scaler_feature_order`로 순서 보장 시도
- 하지만 `trading_env.py`에서 피처 매핑 시 인덱스 불일치 가능성

**개선 필요:**
- 스케일러 저장 시 피처 이름도 함께 저장
- 로드 시 피처 이름으로 매핑 (인덱스 대신)

### ⚠️ **문제 3: N-step 버퍼 데이터 유실 가능성**

**현재 로직:**
```python
# dqn_agent.py의 remember()
while self.n_step_buffer:
    if len(self.n_step_buffer) < self.n_step and not done:
        break  # done이 아니면 N개 찰 때까지 대기
```

**문제:**
- 에피소드가 끝나지 않으면 버퍼에 남은 데이터가 처리되지 않을 수 있음
- 마지막 N-1개 경험이 손실될 수 있음

**개선 필요:**
- 에피소드 종료 시 버퍼 flush 로직 강화

### ⚠️ **문제 4: NoisyNet 사용법**

**현재:**
- 매 스텝마다 `reset_noise()` 호출
- 이는 올바른 사용법이지만, 학습 효율에 영향을 줄 수 있음

**권장:**
- 에피소드 시작 시 한 번만 리셋 (현재는 매 스텝)
- 또는 배치 학습 전에만 리셋

### ⚠️ **문제 5: 데이터 누수 방지 (부분 해결됨)**

**현재:**
- 스케일러 학습 시 80% 구간만 사용 (좋음)
- 하지만 전략 점수 계산은 전체 데이터 사용

**개선 필요:**
- 전략 점수도 학습 구간만 사용하도록 제한

## ✅ 잘 구현된 부분

1. **피처 엔지니어링**: FeatureEngineer로 체계적 관리
2. **MTF 처리**: Look-ahead bias 방지 (shift 적용)
3. **스케일러 순서 보장**: `scaler_feature_order` 사용
4. **N-step Learning**: 구현 완료
5. **PER**: 구현 완료
6. **보상 함수**: 현실적인 버전으로 개선됨

## 🛠️ 개선 제안

### 1. **obs_info 통합 (최우선)**

```python
# dqn_model.py 수정
class DuelingGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, action_dim=3, 
                 info_dim=3, noisy=True):  # info_dim 추가
        ...
        # Info 통합 레이어 추가
        self.info_proj = nn.Linear(info_dim, hidden_dim // 4)
        self.final_dim = hidden_dim + hidden_dim // 4
        
        self.value_stream = nn.Sequential(
            LinearLayer(self.final_dim, 128),  # hidden_dim -> final_dim
            ...
        )
    
    def forward(self, x, info=None):
        # 기존 로직...
        context_vector = self.attention(gru_out)
        
        # Info 통합
        if info is not None:
            info_proj = self.info_proj(info)
            context_vector = torch.cat([context_vector, info_proj], dim=-1)
        else:
            # Info가 없으면 0으로 채움
            info_proj = torch.zeros(context_vector.size(0), hidden_dim // 4).to(context_vector.device)
            context_vector = torch.cat([context_vector, info_proj], dim=-1)
        
        value = self.value_stream(context_vector)
        ...
```

### 2. **스케일러 피처 이름 저장**

```python
# preprocess.py 수정
def save_scaler(self, path='saved_models/scaler.pkl', feature_names=None):
    data = {
        'mean': self.mean,
        'std': self.std,
        'feature_names': feature_names  # 추가
    }
    pickle.dump(data, f)
```

### 3. **N-step 버퍼 완전 flush**

```python
# dqn_agent.py의 remember() 수정
def remember(self, state, action, reward, next_state, done):
    self.n_step_buffer.append((state, action, reward, next_state, done))
    
    # Flush 로직 개선
    while len(self.n_step_buffer) >= self.n_step or (done and len(self.n_step_buffer) > 0):
        current_n = min(self.n_step, len(self.n_step_buffer))
        # ... 기존 로직
```

### 4. **NoisyNet 리셋 최적화**

```python
# train_dqn.py 수정
# 에피소드 시작 시 한 번만 리셋
if hasattr(self.agent, 'reset_noise'):
    self.agent.reset_noise()

# 스텝 내부에서는 리셋하지 않음
```

## 📈 성능 최적화 제안

1. **배치 크기 조정**: 현재 64 → 128로 증가 고려
2. **Target Update 주기**: 현재 1000 → 500으로 단축 고려
3. **Learning Rate 스케줄링**: 고정 LR 대신 Cosine Annealing 고려
4. **그래디언트 클리핑**: 현재 1.0 → 0.5로 조정 고려

## 🔒 안정성 개선

1. **NaN/Inf 체크 강화**: 모든 텐서 연산 후 체크
2. **에러 핸들링**: try-except 블록 추가
3. **로깅 강화**: 중요한 단계마다 로그 출력
