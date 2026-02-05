# Config.py 튜닝 가이드 (3070Ti + 100억 미션)

## 📋 튜닝 요약

| 카테고리 | 파라미터 | 기존값 | 튜닝값 | 근거 |
|---------|---------|--------|--------|------|
| **거래 설정** | `LEVERAGE` | 1 | **10** | TD3 Risk Gate가 리스크 관리 |
| | `MAX_POSITION_SIZE` | 100 | **1e9** | 사실상 무제한 (자산 대비 % 로직) |
| | `STOP_LOSS_PERCENT` | 0.2 | **0.5** | 3분봉 노이즈 고려 |
| **학습 파라미터** | `TRAIN_BATCH_SIZE` | 256 | **1024** | 3070Ti 8GB + AMP 활용 |
| | `TRAIN_NUM_EPISODES` | 5000 | **10000** | 충분한 탐험 |
| | `MACROHFT_EXPERT_PHASE` | 3000 | **5000** | Expert 정밀 학습 |
| **PPO** | `PPO_LEARNING_RATE` | 5e-5 | **1e-4** | 배치 증가에 따라 |
| | `PPO_ENTROPY_COEF` | 0.02 | **0.05** | 강력한 초기 탐험 |
| | `PPO_K_EPOCHS` | 4 | **10** | 안정성 향상 |
| **네트워크** | `NETWORK_HIDDEN_DIM` | 256 | **512** | 표현력 증대 |
| | `NETWORK_ATTENTION_HEADS` | 4 | **8** | Hidden=512에 맞춤 |
| | `NETWORK_INFO_ENCODER_DIM` | 128 | **256** | 전략 인코딩 강화 |
| **TD3** | `TD3_LEARNING_RATE` | 3e-5 | **1e-4** | 배치 증가 |
| | `TD3_BATCH_SIZE` | 256 | **512** | 3070Ti 활용 |
| | `TD3_BUFFER_SIZE` | 100k | **200k** | 더 다양한 경험 |
| | `TD3_EXPLORE_NOISE` | 0.1 | **0.15** | 초기 탐험 강화 |

---

## 🎯 핵심 변경사항

### 1. 거래 설정 (Aggressive Trading)

#### LEVERAGE: 1 → 10
```python
# 기존: 보수적 레버리지
LEVERAGE = 1

# 튜닝: TD3 Risk Gate 신뢰
LEVERAGE = 10  # Position-Aware Actor가 손실 시 게이트 닫음
```

**근거**:
- TD3의 `RiskAwareGate`가 PnL < -2% 시 자동으로 포지션 축소
- 100억 목표는 레버리지 없이는 불가능
- 3070Ti가 충분한 리스크 모델링 가능

---

#### MAX_POSITION_SIZE: 100 → 1e9 (사실상 무제한)
```python
# 기존: 100 USDT 고정 제한
MAX_POSITION_SIZE = 100

# 튜닝: 제한 해제 (자산 대비 % 관리)
MAX_POSITION_SIZE = 1e9
```

**근거**:
- 100 USDT는 초기 자본 10,000 USDT 대비 1%에 불과
- 실제 리스크 관리는 `trading_env.py`의 보상 함수에서 처리
- Kelly Criterion 기반 동적 포지션 사이징 필요

**TODO**: `trading_env.py`에서 자산 대비 % 로직 추가
```python
# 예시
current_equity = self.initial_balance * (1 + total_pnl)
max_position = current_equity * 0.5  # 자산의 50%까지
```

---

#### STOP_LOSS_PERCENT: 0.2 → 0.5
```python
# 기존: 0.2% 손절 (너무 타이트)
STOP_LOSS_PERCENT = 0.2

# 튜닝: 0.5% 손절
STOP_LOSS_PERCENT = 0.5
```

**근거**:
- 3분봉 ETH 평균 변동성: ~0.3%
- 0.2%는 정상적인 노이즈에도 손절
- Sortino Ratio 보상이 하방 리스크 관리

---

### 2. 학습 파라미터 (3070Ti 최대 활용)

#### TRAIN_BATCH_SIZE: 256 → 1024
```python
# 기존: 보수적 배치 사이즈
TRAIN_BATCH_SIZE = 256

# 튜닝: 3070Ti 8GB + AMP 활용
TRAIN_BATCH_SIZE = 1024
```

**메모리 계산**:
```
Batch=1024, Seq=60, Hidden=512:
- Input: 1024 × 60 × 44 × 2 bytes (FP16) = 5.4 MB
- Hidden: 1024 × 512 × 2 bytes = 1.0 MB
- Gradient: ~2x = 12.8 MB
- Model: ~30 MB (PPO 3 Experts)
- Total: ~50 MB (3070Ti 8GB의 0.6%)
```

**효과**:
- 학습 안정성 향상 (큰 배치 = 낮은 분산)
- 속도 향상 (GPU 병렬화 극대화)

---

#### TRAIN_NUM_EPISODES: 5000 → 10000
```python
# 기존: 5000 에피소드
TRAIN_NUM_EPISODES = 5000

# 튜닝: 10000 에피소드 (Early Stopping 신뢰)
TRAIN_NUM_EPISODES = 10000
```

**근거**:
- Elite 8 전략 + 44 피처 = 복잡한 공간
- Early Stopping으로 과적합 방지
- TensorBoard로 수렴 모니터링

---

### 3. PPO 하이퍼파라미터

#### PPO_ENTROPY_COEF: 0.02 → 0.05
```python
# 기존: 낮은 엔트로피
PPO_ENTROPY_COEF = 0.02

# 튜닝: 강력한 초기 탐험
PPO_ENTROPY_COEF = 0.05
PPO_ENTROPY_DECAY = 0.9995  # 천천히 감소
```

**효과**:
- 초반 100 에피소드: 매우 다양한 행동 시도
- Episode 1000: Entropy ≈ 0.025 (중간)
- Episode 5000: Entropy ≈ 0.01 (수렴)

---

#### PPO_LEARNING_RATE: 5e-5 → 1e-4
```python
# 기존: 작은 LR
PPO_LEARNING_RATE = 5e-5

# 튜닝: 배치 증가에 따라
PPO_LEARNING_RATE = 1e-4
```

**근거**:
- Batch Size ∝ Learning Rate (선형 스케일링 법칙)
- Batch 256 → 1024 (4배) → LR 5e-5 → 2e-4 권장
- 1e-4는 안전한 중간값

---

#### PPO_K_EPOCHS: 4 → 10
```python
# 기존: 4 에폭
PPO_K_EPOCHS = 4

# 튜닝: 10 에폭
PPO_K_EPOCHS = 10
```

**근거**:
- 큰 배치 = 더 많은 데이터 → 더 많이 학습 가능
- PPO Clipping이 과적합 방지
- Entropy Decay로 점진적 수렴

---

### 4. 네트워크 아키텍처

#### NETWORK_HIDDEN_DIM: 256 → 512
```python
# 기존: 작은 모델
NETWORK_HIDDEN_DIM = 256

# 튜닝: 표현력 증대
NETWORK_HIDDEN_DIM = 512
```

**근거**:
- Elite 8 전략 (8-dim) + 44 피처 = 복잡한 입력
- StrategyInteractionLayer: 8 → 64
- CrossAttentionFusion: Query 67-dim
- Hidden=512로 충분한 표현 공간 확보

**파라미터 증가**:
- 기존: ~2.5M params
- 튜닝: ~10M params (4배)
- 메모리: ~40 MB (FP16)

---

#### NETWORK_ATTENTION_HEADS: 4 → 8
```python
# 기존: 4 헤드
NETWORK_ATTENTION_HEADS = 4

# 튜닝: 8 헤드 (Hidden=512에 맞춤)
NETWORK_ATTENTION_HEADS = 8
```

**근거**:
- Hidden=512 → Head Dim=64 (512/8)
- 더 다양한 Attention 패턴 학습

---

### 5. TD3 설정

#### TD3_BATCH_SIZE: 256 → 512
```python
# 기존: 256
TD3_BATCH_SIZE = 256

# 튜닝: 512
TD3_BATCH_SIZE = 512
```

**근거**:
- Off-Policy 알고리즘 = 큰 배치 선호
- CQL Loss 안정성 향상

---

#### TD3_BUFFER_SIZE: 100k → 200k
```python
# 기존: 100,000
TD3_BUFFER_SIZE = 100000

# 튜닝: 200,000
TD3_BUFFER_SIZE = 200000
```

**메모리**:
- 200k transitions × 60 × 44 × 4 bytes = 2.1 GB
- 메인 RAM 16GB 대비 13%

---

#### TD3_EXPLORE_NOISE: 0.1 → 0.15
```python
# 기존: 낮은 탐험
TD3_EXPLORE_NOISE = 0.1

# 튜닝: 강화
TD3_EXPLORE_NOISE = 0.15
```

**근거**:
- 연속 행동 공간 (-1 ~ 1)
- Noise=0.15 → 행동 범위: ±0.15 추가 변동
- 초기 10k steps에서 충분한 탐험

---

## 🚀 최적화 설정

### PyTorch 자동 최적화
```python
# config.py에 추가
USE_AMP = True                      # FP16 연산
USE_TORCH_COMPILE = True            # 그래프 최적화
USE_CUDNN_BENCHMARK = True          # cuDNN 자동 튜닝
USE_HIGH_MATMUL_PRECISION = True    # TF32 (Ampere)
```

### train_ppo.py 적용
```python
# 자동으로 config에서 읽어서 적용
if config.USE_CUDNN_BENCHMARK:
    torch.backends.cudnn.benchmark = True

if config.USE_HIGH_MATMUL_PRECISION:
    torch.set_float32_matmul_precision('high')
```

---

## 📊 예상 성능

### 학습 속도
| 항목 | 기존 (Batch=256, Hidden=256) | 튜닝 (Batch=1024, Hidden=512) |
|------|----------------------------|----------------------------|
| Episode 시간 | ~40초 | ~60초 (+50%) |
| GPU 사용률 | ~60% | **~95%** |
| GPU 메모리 | ~2GB | **~4GB** |
| Throughput | ~3 eps/min | **~2 eps/min** |

**Trade-off**:
- 속도는 약간 느려짐 (모델 크기 4배)
- **학습 품질 대폭 향상** (표현력, 안정성)
- 100억 목표에는 품질 > 속도

---

### 최종 성능 목표
| 지표 | 보수적 목표 | 공격적 목표 (100억) |
|------|------------|-------------------|
| Sharpe Ratio | 1.5 | **3.0+** |
| Max Drawdown | -15% | **-10%** |
| Win Rate | 55% | **65%** |
| 연간 수익률 | 50% | **1000%+** |

---

## ⚠️ 주의사항

### 1. 메모리 부족 시
```python
# config.py 조정
TRAIN_BATCH_SIZE = 512      # 1024 → 512
NETWORK_HIDDEN_DIM = 384    # 512 → 384
```

### 2. 학습 불안정 시
```python
# Learning Rate 낮추기
PPO_LEARNING_RATE = 5e-5    # 1e-4 → 5e-5
TD3_LEARNING_RATE = 5e-5

# Gradient Clipping 강화
# ppo_agent.py에서
nn.utils.clip_grad_norm_(params, max_norm=0.5)  # 기존 1.0
```

### 3. 과적합 의심 시
```python
# Dropout 증가
NETWORK_DROPOUT = 0.2       # 0.1 → 0.2

# Entropy 최소값 올리기
PPO_ENTROPY_MIN = 0.01      # 0.005 → 0.01
```

---

## 🎮 실행 방법

### 기존 학습 중단 (필수!)
```bash
# Ctrl+C로 중단
# config.py 변경으로 모델 구조 변경됨 → 기존 체크포인트 호환 불가
```

### 새 학습 시작
```bash
# PPO (처음부터)
python .\macroHFT\train_ppo.py

# TD3 (처음부터)
python .\TD3\train_td3.py
```

### TensorBoard 모니터링
```bash
tensorboard --logdir=logs/tensorboard
# http://localhost:6006

# 주요 지표:
# - Loss/Actor: 감소 추세
# - Episode/Reward: 증가 추세
# - Episode/Trades: 안정화
```

---

**작성일**: 2026-02-06  
**적용 대상**: 3070Ti 8GB VRAM  
**목표**: 100억 미션  
**Breaking Change**: ✅ Yes (모델 구조 변경, 재학습 필요)
