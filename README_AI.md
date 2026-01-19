# AI 강화학습 모델 학습 및 사용 가이드

## 전체 시스템 흐름도

```
1. Data Fetch (DataCollector)
   ↓
2. Denoising (Wavelet) - 가격 데이터의 노이즈 제거
   ↓
3. Feature Engineering - 10개 전략의 Score를 Feature 벡터에 결합
   ↓
4. Scaling (Min-Max) - 모든 입력값을 [-1, 1] 범위로 정규화
   ↓
5. xLSTM Input - 전처리된 텐서를 모델에 입력
   ↓
6. PPO Action - 에이전트가 행동(Action)을 결정하고 보상(Reward)을 통해 학습
```

## 학습 방법

### 1. 학습 데이터 수집 (먼저 실행)

1년치 과거 데이터를 수집하여 `data` 폴더에 저장합니다:

```bash
python model/collect_training_data.py
```

이 스크립트는:
- 바이낸스에서 1년치 과거 데이터를 가져옵니다
- `data/eth_3m_1year.csv`와 `data/btc_3m_1year.csv`에 저장합니다
- 약 175,200개 캔들 (3분봉 기준)을 수집합니다

**참고**: 데이터 수집에는 시간이 걸릴 수 있습니다 (약 10-30분).

### 2. 모델 학습

수집된 데이터로 모델을 학습합니다:

```bash
python model/train_ppo.py --episodes 100 --steps 100 --save-interval 10
```

**파라미터:**
- `--episodes`: 학습 에피소드 수 (기본값: 100)
- `--steps`: 에피소드당 최대 스텝 수 (기본값: 100)
- `--save-interval`: 모델 저장 간격, 에피소드 단위 (기본값: 10)

**예시:**
```bash
# 200 에피소드 학습, 에피소드당 50 스텝, 20 에피소드마다 저장
python train_ppo.py --episodes 200 --steps 50 --save-interval 20
```

### 2. 학습된 모델 사용

`config.py`에서 AI 모드를 활성화:

```python
ENABLE_AI = True  # AI 모드 활성화
AI_MODEL_PATH = 'model/ppo_model.pth'  # 학습된 모델 경로
```

그리고 `trading_bot.py`를 실행:

```bash
python trading_bot.py
```

## 전처리 파이프라인 상세

### 1. Data Fetch
- `DataCollector.get_candles()`: 실시간 캔들 데이터 수집
- 최소 40봉 필요 (웨이블릿 변환용)

### 2. Denoising (Wavelet)
- `DataPreprocessor.wavelet_denoising()`: Daubechies 4 웨이블릿 사용
- 고주파 노이즈 제거로 추세 선명화
- Whipsaw(미세 변동) 제거

### 3. Feature Engineering
- 노이즈 제거된 가격 데이터
- 거래량 데이터
- 10개 전략의 신뢰도 점수 (confidence)
- 최종 피처: `(20, 2 + num_strategies)` 형태

### 4. Scaling (Min-Max)
- 범위: `[-1, 1]`
- xLSTM의 exp() 연산 시 값 발산 방지
- `DataPreprocessor.fit_transform()` 사용

### 5. xLSTM Input
- 입력 형태: `(1, 20, state_dim)` 텐서
- 시퀀스 길이: 20봉
- 배치 크기: 1

### 6. PPO Action
- 행동 공간: 3개 (0: Hold, 1: Long, 2: Short)
- 보상 함수: 수익률 기반 + 거래 비용 페널티
- 학습: GAE(Generalized Advantage Estimation) + PPO Clipping

## 파일 구조

```
crypto-scalping/
├── trading_bot.py         # 트레이딩 봇 (추론만 수행)
├── model/
│   ├── train_ppo.py       # 모델 학습 스크립트 (별도 실행)
│   ├── collect_training_data.py  # 학습 데이터 수집 스크립트
│   ├── ppo_agent.py       # PPO 에이전트
│   ├── trading_env.py     # 트레이딩 환경 (전처리 파이프라인 포함)
│   ├── preprocess.py      # 데이터 전처리 (Wavelet, Scaling)
│   ├── xlstm_network.py   # xLSTM 네트워크
│   └── ppo_model.pth      # 학습된 모델 (학습 후 생성)
└── data/                   # 학습 데이터 저장 폴더
    ├── eth_3m_1year.csv
    └── btc_3m_1year.csv
```

## 주의사항

1. **학습과 추론 분리**: 
   - 학습: `train_ppo.py`에서 별도로 수행
   - 추론: `trading_bot.py`에서 학습된 모델만 로드하여 사용

2. **모델 파일 필수**: 
   - `trading_bot.py` 실행 전에 반드시 `model/train_ppo.py`로 모델을 학습해야 함
   - 모델 파일이 없으면 AI 모드가 비활성화됨

3. **전처리 파이프라인**:
   - 모든 단계가 `TradingEnvironment.get_observation()`에서 자동으로 수행됨
   - 수동 조작 불필요

4. **성능 최적화**:
   - GPU 사용 시 자동 감지 (`cuda` 사용)
   - CPU만 사용 가능한 경우 자동으로 `cpu` 모드로 전환

## 학습 모니터링

학습 중 로그는 `logs/train_ppo.log`에 저장됩니다.

주요 메트릭:
- 에피소드별 총 보상
- 최근 10개 에피소드 평균 보상
- 최고 성능 모델 자동 저장
