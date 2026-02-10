"""
설정 파일 - 3070Ti 최적화 + 100억 미션
- Elite 8 전략 반영
- Info Dimension 최적화 (11 for PPO, 12 for TD3)
- 3070Ti VRAM 8GB 최대 활용
- Aggressive Trading for 100억 목표
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 바이낸스 API 설정
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
BINANCE_TESTNET = False

# =============================================================================
# [거래 설정] - 100억 미션 최적화
# =============================================================================
ETH_SYMBOL = os.getenv('ETH_SYMBOL', 'ETHUSDT')
BTC_SYMBOL = os.getenv('BTC_SYMBOL', 'BTCUSDT')

# [야수 모드] 레버리지 20배 (바이낸스 알트코인 Max 수준)
# TD3 Action이 -1~1 범위이므로, 실제 레버리지는 |action| * LEVERAGE
# 예: action=0.8 → 0.8 * 20 = 16배 레버리지
LEVERAGE = 20  # 최대 레버리지 (TD3가 동적 조절)

# [수정 불가] 포지션 크기 제한 (레버리지로 대체)
MAX_POSITION_SIZE = 1e9  # 사실상 무제한

# [레버리지 대응] 손절 기준 강화
# 레버리지 20배 시: 가격 0.5% 변동 = 자산 10% 변동
# 자산 기준 -10% 손실 시 칼같이 손절
STOP_LOSS_PERCENT = 0.10  # 자산 기준 10% 손실 = 손절
STOP_LOSS_THRESHOLD = -0.20  # ROE 기준 -20% 손실 = 손절

# [Elite 8 전략 설정]
# 이 딕셔너리의 순서가 model input의 순서와 일치해야 함
STRATEGIES = {
    # Alpha Strategies (New & Powerful)
    'whale_sentiment': True,
    'liquidation_squeeze': True,

    # Order Flow & Structure
    'orderblock_fvg': True,
    'net_taker_flow': True,

    # Standard Technicals
    'btc_eth_correlation': True,
    'volatility_squeeze': True,
    'vwap_deviation': True,
    'hma_momentum': True
}

# 시간프레임 설정
TIMEFRAME = '3m'
LOOKBACK_PERIOD = 1500

# 거래 실행 설정
ENABLE_TRADING = False
ENABLE_AI = True
AI_MODEL_PATH = 'data/ppo_model.pth'

# AI 모델 하이퍼파라미터
# [유지] 시계열 윈도우는 60으로 증가 권장 (현재 train에서 60 사용 중)
LOOKBACK = 120

# 보상 함수 파라미터
REWARD_MULTIPLIER = 100.0
LOSS_PENALTY_MULTIPLIER = 50.0
TRANSACTION_COST = 0.0005

# [레버리지 시스템] 청산 및 손절 임계값
LIQUIDATION_THRESHOLD = -0.80  # ROE -80% = 강제 청산 (게임 오버)
TAKE_PROFIT_THRESHOLD = 0.50   # ROE +50% = 자동 익절

# =============================================================================
# [PPO 하이퍼파라미터] - Aggressive Exploration
# =============================================================================
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
PPO_EPS_CLIP = 0.2

# [튜닝] K_EPOCHS 증가 (배치 사이즈 커지면 더 많이 학습)
PPO_K_EPOCHS = 10  # 4 → 10 (안정성 향상)

# [튜닝] Entropy 감소 (뇌동매매 억제)
PPO_ENTROPY_COEF = 0.02 # 0.05 → 0.005 (불필요한 랜덤 행동 억제)
PPO_ENTROPY_DECAY = 0.9995  # 0.999 → 0.9995 (천천히 감소)
PPO_ENTROPY_MIN = 0.001  # 0.005 → 0.001 (최소값 낮춤)

# [튜닝] Learning Rate 증가 (배치 사이즈 증가에 따라)
PPO_LEARNING_RATE = 1e-4  # 5e-5 → 1e-4

PPO_VALUE_CLIP_EPS = 0.3
PPO_TEMP_INIT = 0.8
PPO_TEMP_MIN = 0.3
PPO_TEMP_DECAY = 0.9995

# =============================================================================
# [네트워크 아키텍처] - 3070Ti 최대 활용
# =============================================================================
# [튜닝] Hidden Dim 증가 (모델 표현력 향상)
NETWORK_HIDDEN_DIM = 1024  # 256 → 512 (Elite 8 + 44 features 소화)

# [유지] Layers는 유지 (속도 고려)
NETWORK_NUM_LAYERS = 2  # TD3는 3 (strategic mode에서 자동)
TD_NETWORK_NUM_LAYERS = 3

NETWORK_DROPOUT = 0.1

# [튜닝] Attention Heads 증가
NETWORK_ATTENTION_HEADS = 8  # 4 → 8 (Hidden=512에 맞춰)

# [튜닝] Encoder/Trunk Dim 증가
NETWORK_INFO_ENCODER_DIM = 256  # 128 → 256
NETWORK_SHARED_TRUNK_DIM1 = 512  # 256 → 512
NETWORK_SHARED_TRUNK_DIM2 = 256  # 128 → 256
NETWORK_ACTOR_HEAD_DIM = 128  # 64 → 128
NETWORK_CRITIC_HEAD_DIM = 64  # 32 → 64

# =============================================================================
# [학습 파라미터] - 3070Ti 8GB VRAM 최대 활용
# =============================================================================
# [튜닝] Batch Size 대폭 증가 (AMP 활용)
TRAIN_BATCH_SIZE = 1024  # 256 → 1024 (3070Ti + AMP로 충분)

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# [튜닝] Episode 증가 (Early Stopping 신뢰)
TRAIN_NUM_EPISODES = 3000  # 5000 → 10000

TRAIN_MAX_STEPS_PER_EPISODE = 240
MAX_TRADES_PER_EPISODE = 50
TRAIN_SAVE_INTERVAL = 200

EVAL_INITIAL_CAPITAL = 10000
EVAL_VERBOSE_INTERVAL = 200

# =============================================================================
# [TD3 설정] - 긴급 처방 적용 (야수 모드)
# =============================================================================
# PPO/MacroHFT: Info = 11 (pos_val + 8 strategies + pos_meta 2)
# TD3: 변동성 1차원 추가 시 12 사용 (train_td3 _augment_info)
TD3_INFO_DIM = 12

# [긴급 처방 3] Learning Rate 3배 가속
TD3_LEARNING_RATE = 3e-4  # 1e-4 → 3e-4 (Q값 정체 타파)

TD3_GAMMA = 0.99
TD3_TAU = 0.005

# [유지] Policy Noise (Target Smoothing)
TD3_POLICY_NOISE = 0.2

TD3_NOISE_CLIP = 0.5

# [긴급 처방 3] Explore Noise 3배 증폭
TD3_EXPLORE_NOISE = 0.3  # 0.15 → 0.3 (겁먹은 놈 등떠밀기)

TD3_POLICY_FREQ = 2

# [유지] Batch Size 증가 (안정성)
TD3_BATCH_SIZE = 192
TD3_LAMBDA_ANNEAL_EPISODES = 2000

# [유지] Buffer Size 증가
TD3_BUFFER_SIZE = 100000

# [긴급 처방 3] Warmup 증가 (충분한 데이터 수집)
TD3_WARMUP_STEPS = 10000  # 충분히 데이터를 모으고 학습 시작

# [긴급 처방 3] Deadzone 증가 (미세 진입 방지)
TD3_DEADZONE = 0.3
TD3_MIN_TRADE_SIZE = 0.3
# =============================================================================
# [성능 최적화] - PyTorch Settings
# =============================================================================
# 이 설정들은 train_*.py에서 자동으로 적용됨

# AMP (Automatic Mixed Precision)
USE_AMP = True  # FP16 연산으로 메모리 50% 절감, 속도 2배

# Torch Compile (PyTorch 2.0+)
USE_TORCH_COMPILE = False  # 그래프 최적화

# cuDNN Benchmark (3070Ti Ampere 최적화)
USE_CUDNN_BENCHMARK = True  # 입력 크기 고정 시 최적 알고리즘 탐색

# TensorCore Precision
USE_HIGH_MATMUL_PRECISION = True  # TF32 사용 (Ampere)
