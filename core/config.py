"""
설정 파일 - MacroHFT v4.0 PRODUCTION READY
=============================================
- 연속 레버리지 포지셔닝
- Kahneman-Tversky 리워드
- Transformer Projection Layer 복원
- Out-of-time 검증 분리
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 바이낸스 API (실전 미사용)
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
BINANCE_TESTNET = False

# =============================================================================
# 기본 환경 설정
# =============================================================================
ETH_SYMBOL = 'ETHUSDT'
BTC_SYMBOL = 'BTCUSDT'
TIMEFRAME = '5m'
LOOKBACK = 120                # Transformer 시퀀스 길이
LOOKBACK_PERIOD = 1500        # 데이터 준비용

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# =============================================================================
# [모델 아키텍처] - 대규모 확장
# =============================================================================
D_MODEL = 256                 # 128 → 256 (표현력 2배)
N_HEAD = 8                   # 4 → 8
N_LAYERS = 6                 # 4 → 6
PROJ_DIM = 256               # Projection Layer 복원
USE_MAMBA = False           # Transformer 고정

# =============================================================================
# [행동 공간] - 연속 레버리지
# =============================================================================
MAX_LEVERAGE = 20           # 최대 레버리지 20배
MIN_LEVERAGE = 1.0         # 최소 실행 레버리지
TRANSACTION_COST = 0.0005  # 0.05% (테이커 기준)

# 청산/손절 기준 (ROE 기준)
STOP_LOSS_THRESHOLD = -0.20    # -20% ROE
TAKE_PROFIT_THRESHOLD = 0.50   # +50% ROE
LIQUIDATION_THRESHOLD = -0.80  # -80% ROE (강제청산)
TIME_STOP_STEPS = 100          # 100스텝 무수익 시 청산

# =============================================================================
# [MacroHFT Reward v4] - 행동경제학 + 리스크 페널티
# =============================================================================
# 1. Kahneman-Tversky 손실 회피
REWARD_LOSS_AVERSION = 2.25    # λ = 2.25 (손실 고통 2.25배)

# 2. 전문가별 계수
REWARD_TREND_LOG_SCALE = 100.0
REWARD_VOLATILITY_SCALE = 50.0
REWARD_SIDEWAYS_BASE = 0.01

# 3. 리스크 페널티 (에피소드 종료 시 부과)
REWARD_DOWNSIDE_PENALTY = 0.5      # 하방 변동성 페널티 계수
REWARD_MDD_PENALTY_COEF = 20.0     # 최대낙폭(MDD) 페널티 계수

# 4. 소프트 클리핑
REWARD_STEP_SCALE = 100.0

# =============================================================================
# [PPO 하이퍼파라미터] - 전문가/라우터 통합
# =============================================================================
PPO_LEARNING_RATE = 5e-5
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
PPO_EPS_CLIP = 0.15
PPO_K_EPOCHS = 7
PPO_ENTROPY_COEF = 0.2

# 전문가별 감마 (시야 차별화)
EXPERT_GAMMAS = {
    0: 0.995,   # Trend: 장기
    1: 0.99,    # Volatility: 중기
    2: 0.90     # Sideways: 단기
}

# =============================================================================
# [라우터 PPO] - Neural EXP3.P 기반 (Regret Matching 대체)
# =============================================================================
ROUTER_LR = 1e-5
ROUTER_ENTROPY_COEF = 0.05
ROUTER_EPS_CLIP = 0.2
ROUTER_GAMMA = 0.99
ROUTER_EXP3_ETA = 0.05       # EXP3.P 학습률

# =============================================================================
# [정규화 및 손실 계수]
# =============================================================================
SHARPE_COEF = 0.001          # 샤프 비율 손실 계수
ORTHO_COEF = 0.01           # 직교 정규화 계수 (0.0005 → 0.01)
ENTROPY_DECAY = 0.998

# =============================================================================
# [학습 환경]
# =============================================================================
TRAIN_NUM_EPISODES = 3000
TRAIN_MAX_STEPS_PER_EPISODE = 480
TRAIN_BATCH_SIZE = 1024
TRAIN_SAVE_INTERVAL = 200

EVAL_INITIAL_CAPITAL = 10000
EVAL_VERBOSE_INTERVAL = 200

# =============================================================================
# [성능 최적화]
# =============================================================================
USE_AMP = True
USE_TORCH_COMPILE = True
USE_CUDNN_BENCHMARK = True
USE_HIGH_MATMUL_PRECISION = True

# =============================================================================
# [동적 포지션 사이징] – 리스크 기반 포지션 크기 제어
# =============================================================================
RISK_TARGET_VOL = 0.15           # 연간 목표 변동성 (15%)
RISK_LOOKBACK = 20              # 변동성 측정 기간 (틱)
RISK_MAX_LEVERAGE = 20          # 최대 허용 레버리지 (브로커 제한)
RISK_MAX_POSITION_RATIO = 0.5   # 자본금 대비 최대 포지션 비율 (안전장치)
RISK_VOL_ADJUSTMENT_MIN = 0.5   # 변동성 조정 하한 (50%)
RISK_VOL_ADJUSTMENT_MAX = 2.0   # 변동성 조정 상한 (200%)
VOLATILITY_ALREADY_ANNUALIZED = False

# =============================================================================
# [MacroHFT Reward v5] – PnL 비례 리워드 + 경량 패널티
# =============================================================================
REWARD_PNL_SCALE = 100.0          # 🔥 1% = 100점
REWARD_LEVERAGE_PENALTY = -0.0002
REWARD_TRADE_PENALTY = -0.001
REWARD_LOSS_AVERSION = 1.0        # 🔥 손실 회피 제거 (1.0 = 없음)
REWARD_CLIP_SCALE = 50.0         # 🔥 tanh 클리핑 거의 제거 (실질적 -100~100)

# 전문가 보너스 (PnL 대비 10% 미만)F
REWARD_TREND_HOLDING_BONUS = 0.001
REWARD_VOLATILITY_BONUS = 0.01
REWARD_SIDEWAYS_WIN_BONUS = 0.05
REWARD_SIDEWAYS_LOSS_PENALTY = -0.005
REWARD_SIDEWAYS_SMALL_BONUS = 0.05

# =============================================================================
# [CVaR Risk] – 하위 분위수 집중 학습
# =============================================================================
CVAR_COEF = 0.5
CVAR_ALPHA = 0.05

# =============================================================================
# [레버리지 이산화] – 학습 안정성 향상
# =============================================================================
LEVERAGE_CANDIDATES = [1,5,10]          # 사용할 레버리지 배수
LEVERAGE_DISCRETE = len(LEVERAGE_CANDIDATES)     # 5
ACTION_DIM = 3 * LEVERAGE_DISCRETE               # 3방향 × 5레버리지 = 15