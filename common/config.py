"""
설정 파일 - MacroHFT v3 최적화
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

# 보상 함수 파라미터
REWARD_MULTIPLIER = 100.0
LOSS_PENALTY_MULTIPLIER = 50.0
TRANSACTION_COST = 0.005
TIME_COST = 0.0

# [레버리지 시스템] 청산 및 손절 임계값
LIQUIDATION_THRESHOLD = -0.80
TAKE_PROFIT_THRESHOLD = 0.50

# =============================================================================
# [PPO & Expert 설정] - MacroHFT v3 Core
# =============================================================================
PPO_LEARNING_RATE = 1e-4
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
PPO_EPS_CLIP = 0.2
PPO_K_EPOCHS = 10
PPO_ENTROPY_COEF = 0.01  # 안정적인 탐색 (0.05 -> 0.01)

# [Expert Gamma] - 전문가별 시야 차별화 (학습 시 자동 적용)
EXPERT_GAMMAS = {
    0: 0.995,  # Trend: 장기 (Lookahead High)
    1: 0.99,   # Volatility: 중기
    2: 0.90    # Sideways: 단기 (Instant Gratification)
}

# =============================================================================
# [리워드 설정] - 보상 가중치 (macrohft_reward.py에서 사용)
# =============================================================================
REWARD_STEP_PNL_MULT = 100.0
REWARD_REALIZED_PNL_MULT = 100.0

REWARD_TREND_EXIT_MULT = 150.0
REWARD_VOLATILITY_EXIT_MULT = 200.0
REWARD_SIDEWAYS_EXIT_MULT = 200.0

REWARD_HOLDING_BONUS_BASE = 0.1
REWARD_HOLDING_BONUS_COEF = 0.2
PENALTY_CHURNING = 5.0             # 단타 방지 (강함)
PENALTY_LOSS_VOLATILITY = 50.0
PENALTY_LOSS_SIDEWAYS = 200.0

REWARD_CLIP_MIN = -10.0
REWARD_CLIP_MAX = 10.0

# =============================================================================
# [네트워크 설정]
# =============================================================================
# 주의: MacroHFT v3는 Expert별로 최적화된 d_model을 내부적으로 사용하므로
# 아래 NETWORK_HIDDEN_DIM은 Router나 기본 네트워크 초기화 시에만 참조될 수 있음.
NETWORK_HIDDEN_DIM = 512
NETWORK_NUM_LAYERS = 2
NETWORK_DROPOUT = 0.1
NETWORK_ATTENTION_HEADS = 8
NETWORK_INFO_ENCODER_DIM = 256
NETWORK_SHARED_TRUNK_DIM1 = 512
NETWORK_SHARED_TRUNK_DIM2 = 256
NETWORK_ACTOR_HEAD_DIM = 128
NETWORK_CRITIC_HEAD_DIM = 64
NETWORK_USE_CHECKPOINTING = False

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

# =============================================================================
# [학습 파라미터]
# =============================================================================
TRAIN_ACTION_DIM = 3
TRAIN_BATCH_SIZE = 1024
TRAIN_SAMPLE_SIZE = 4096
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

TRAIN_NUM_EPISODES = 3000
TRAIN_MAX_STEPS_PER_EPISODE = 480
MAX_TRADES_PER_EPISODE = 50
TRAIN_SAVE_INTERVAL = 200

EVAL_INITIAL_CAPITAL = 10000
EVAL_VERBOSE_INTERVAL = 200

# =============================================================================
# [성능 최적화] - PyTorch Settings
# =============================================================================
USE_AMP = True
USE_TORCH_COMPILE = False
USE_CUDNN_BENCHMARK = True

# =============================================================================
# [MacroHFT Reward v4 Settings] - 2026 SOTA Research-Aligned
# =============================================================================
# 1. Kahneman-Tversky Asymmetry (Prospect Theory)
REWARD_LOSS_AVERSION = 2.25       # 손실 고통 계수 (λ)
REWARD_BASE_MULT = 50.0           # 기본 PnL 배수 (Base Scale)

# 2. Risk Controls
REWARD_DOWNSIDE_PENALTY = 0.5     # 하방 변동성 페널티 가중치
REWARD_MDD_PENALTY_COEF = 20.0    # MDD 발생 시 페널티 강도

# 3. Expert Specifics
REWARD_TREND_LOG_RETURN_SCALE = 100.0   # 추세 전문가: 로그 수익률 스케일
REWARD_VOLATILITY_SHARPE_SCALE = 2.0    # 변동성 전문가: 샤프 지수 보너스
REWARD_SIDEWAYS_MDD_THRESHOLD = 0.02    # 횡보 전문가: MDD 허용치 (2%)
REWARD_SIDEWAYS_DECAY_START = 30        # 횡보 전문가: 시간 감점 시작 틱

# 4. Soft Clipping
REWARD_CLIP_SCALE = 10.0          # Soft Clip (tanh) 스케일
