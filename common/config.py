"""
설정 파일 - MacroHFT v3.5 SOTA (Mamba + D-PPO + Reward v4)
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 바이낸스 API 설정
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
BINANCE_TESTNET = False

# =============================================================================
# [기본 환경 설정]
# =============================================================================
ETH_SYMBOL = os.getenv('ETH_SYMBOL', 'ETHUSDT')
BTC_SYMBOL = os.getenv('BTC_SYMBOL', 'BTCUSDT')
TIMEFRAME = '3m'
LOOKBACK_PERIOD = 1500

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# [야수 모드] 레버리지 20배
LEVERAGE = 20
MAX_POSITION_SIZE = 1e9

# [레버리지 대응] 손절 기준
STOP_LOSS_PERCENT = 0.10
STOP_LOSS_THRESHOLD = -0.20

# [Elite 8 전략 설정]
STRATEGIES = {
    'whale_sentiment': True,
    'liquidation_squeeze': True,
    'orderblock_fvg': True,
    'net_taker_flow': True,
    'btc_eth_correlation': True,
    'volatility_squeeze': True,
    'vwap_deviation': True,
    'hma_momentum': True
}

# AI 모델 설정
ENABLE_TRADING = False
ENABLE_AI = True
AI_MODEL_PATH = 'data/ppo_model.pth'
LOOKBACK = 120

# =============================================================================
# [MacroHFT v3.5 SOTA Settings] - 핵심 업그레이드
# =============================================================================
# 1. Mamba Architecture (속도 & 장기 기억)
USE_MAMBA = True               # True: Mamba, False: Transformer
MAMBA_D_STATE = 16             # SSM State Dimension
MAMBA_D_CONV = 4               # Local Conv Kernel Size
MAMBA_EXPAND = 2               # Block Expansion Factor

# 2. Distributional RL (D-PPO) - 리스크 관리
# Critic이 평균값이 아닌 수익률 분포(Quantiles)를 예측
NUM_QUANTILES = 32             # 분위수 개수
QUANTILES_EMBED_DIM = 64       # 임베딩 차원

# =============================================================================
# [MacroHFT Reward v4 Settings] - 보상 체계 고도화
# =============================================================================
# 1. Kahneman-Tversky Asymmetry (손실 회피)
REWARD_LOSS_AVERSION = 2.25       # 손실 고통 계수 (λ)
REWARD_BASE_MULT = 50.0           # 기본 PnL 배수

# 2. Risk Controls
REWARD_DOWNSIDE_PENALTY = 0.5     # 하방 변동성 페널티
REWARD_MDD_PENALTY_COEF = 20.0    # MDD 페널티 강도

# 3. Expert Specifics
REWARD_TREND_LOG_RETURN_SCALE = 100.0   # Trend: 로그 수익률
REWARD_VOLATILITY_SHARPE_SCALE = 2.0    # Volatility: 샤프 지수
REWARD_SIDEWAYS_MDD_THRESHOLD = 0.02    # Sideways: MDD 허용치 (2%)
REWARD_SIDEWAYS_DECAY_START = 30        # Sideways: 시간 감점 시작 틱

# 4. Soft Clipping
REWARD_CLIP_SCALE = 10.0          # 보상 클리핑 (Tanh)

# =============================================================================
# [PPO 학습 파라미터]
# =============================================================================
PPO_LEARNING_RATE = 1e-4
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
PPO_EPS_CLIP = 0.2
PPO_K_EPOCHS = 10
PPO_ENTROPY_COEF = 0.01

# Expert Gamma (시야 차별화)
EXPERT_GAMMAS = {
    0: 0.995,  # Trend: Long-term
    1: 0.99,   # Volatility: Mid-term
    2: 0.90    # Sideways: Short-term
}

# =============================================================================
# [공통 학습 설정]
# =============================================================================
TRAIN_BATCH_SIZE = 1024
TRAIN_NUM_EPISODES = 3000
TRAIN_MAX_STEPS_PER_EPISODE = 480
MAX_TRADES_PER_EPISODE = 50
TRAIN_SAVE_INTERVAL = 200

EVAL_INITIAL_CAPITAL = 10000
EVAL_VERBOSE_INTERVAL = 200

# [거래 비용]
TRANSACTION_COST = 0.0005 # 0.05%

# =============================================================================
# [성능 최적화]
# =============================================================================
USE_AMP = True
USE_TORCH_COMPILE = False
USE_CUDNN_BENCHMARK = True
USE_HIGH_MATMUL_PRECISION = True

TRAIN_ACTION_DIM = 3

# =============================================================================
# [TD3 설정]
# =============================================================================
# PPO/MacroHFT: Info = 11 (pos_val + 8 strategies + pos_meta 2)
# TD3: 변동성 1차원 추가 시 12 사용 (train_td3 _augment_info)
TD3_INFO_DIM = 12
TD3_LEARNING_RATE = 1e-4
TD3_ACTOR_LR = 1e-4
TD3_CRITIC_LR = 1e-3
TD3_GAMMA = 0.99
TD3_TAU = 0.005
TD3_POLICY_NOISE = 0.2
TD3_NOISE_CLIP = 0.5
TD3_POLICY_DELAY = 2
TD3_BUFFER_SIZE = 1000000
TD3_WARMUP_STEPS = 5000
TD3_BATCH_SIZE = 512
TD3_DEADZONE = 0.6
TD3_MIN_TRADE_SIZE = 0.6
