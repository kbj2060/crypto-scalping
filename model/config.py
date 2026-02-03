"""
설정 파일
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 바이낸스 API 설정
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
BINANCE_TESTNET = False  # 테스트넷 비활성화 (실제 거래소 사용)
# BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'False').lower() == 'true'  # .env 파일로 제어하려면 주석 해제

# 거래 설정
ETH_SYMBOL = os.getenv('ETH_SYMBOL', 'ETHUSDT')
BTC_SYMBOL = os.getenv('BTC_SYMBOL', 'BTCUSDT')
# 레버리지 로직 제거됨 (model 내 PnL은 순수 가격 수익률). 외부 연동용으로 1 고정.
LEVERAGE = 1
MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '100'))
STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', '0.2'))

# 전략 활성화 설정
STRATEGIES = {
    # 폭발장 전략
    'btc_eth_correlation': True,
    'volatility_squeeze': True,
    'orderblock_fvg': True,
    'hma_momentum': True,
    'mfi_momentum': True,
    # 횡보장 전략 (Top 5 Mean-Reversion)
    'bollinger_mean_reversion': True,
    'vwap_deviation': True,
    'range_top_bottom': True,
    'stoch_rsi_mean_reversion': True,
    'cmf_divergence': True,
    'cci_reversal': True,
    'williams_r': True,
}

# 시간프레임 설정
TIMEFRAME = '3m'  # 3분봉
LOOKBACK_PERIOD = 1500  # 과거 데이터 조회 기간 (1500봉)

# 거래 실행 설정
ENABLE_TRADING = False  # True: 거래 실행, False: 분석만 수행 (거래 비활성화)

# AI 강화학습 설정
ENABLE_AI = True  # True: AI 기반 결정, False: 기존 전략 조합 방식
AI_MODEL_PATH = 'data/ppo_model.pth'  # AI 모델 저장 경로 (data 폴더에 저장)

# AI 모델 하이퍼파라미터
LOOKBACK = 60  # 시계열 피처를 위한 봉 개수

# 보상 함수 파라미터 (수정됨)
REWARD_MULTIPLIER = 50.0       # 유지
LOSS_PENALTY_MULTIPLIER = 50.0 # 75.0 -> 50.0 (손실/수익 1:1, 손실 공포 완화)
# 거래 수수료 (0.0005 = 0.05%). 0이면 에이전트가 무의미한 잦은 매매를 반복할 수 있음
TRANSACTION_COST = 0.0005
TIME_COST = 0.0001  # 시간 비용
STOP_LOSS_THRESHOLD = -0.02    # -5%는 너무 널널함, -2%로 타이트하게

# ---------------------------------------------------------
# [논문 기반 최적화] PPO 하이퍼파라미터 (스캘핑 특화)
# ---------------------------------------------------------
# 논문 근거: Gort et al. (2022) - 변동성 장세에서의 빠른 적응력 확보
PPO_GAMMA = 0.99             # 0.995 -> 0.99 (스캘핑은 단기 흐름이 더 중요)
PPO_LAMBDA = 0.95            # 유지 (GAE 표준)
PPO_EPS_CLIP = 0.2           # 0.15 -> 0.2 (Gate가 노이즈를 거르므로 학습폭 확대 가능)
PPO_K_EPOCHS = 4             # 유지

# [Dynamic Entropy 보조]
# [수정] 탐험(Entropy) 수치 안정화 — 0.05 (산만함) -> 0.01 (차분함)
PPO_ENTROPY_COEF = 0.01
PPO_ENTROPY_DECAY = 0.999    # 감쇠 속도는 유지
PPO_ENTROPY_MIN = 0.01
PPO_LEARNING_RATE = 2e-4     # 1e-4 -> 2e-4 (배치 사이즈 축소에 따른 LR 미세 상향)

# 웜업(Warm-up) 설정 (즉시 학습 시작)
PPO_LR_WARMUP_EPISODES = 30   # 50 (즉시 학습)
PPO_TEMP_WARMUP_EPISODES = 10 # 10 (거의 즉시 감소)

# 고급 PPO 설정
PPO_USE_VALUE_CLIP = True     # Value Function Clipping 사용 여부
PPO_VALUE_CLIP_EPS = 0.3      # Value Clipping 범위 (0.2 → 0.3 학습 안정화)
PPO_KL_TARGET = 0.05          # 0.02 -> 0.05 상향 (초기 학습 시 조기 종료 방지, 안정성 확보)
# Temperature (탐험 억제: 차분하게)
PPO_TEMP_INIT = 0.8           # 1.0 → 0.8 (초반부터 확신 있는 행동)
PPO_TEMP_MIN = 0.3            # 0.5 → 0.3
PPO_TEMP_DECAY = 0.9995       # 더 느리게 감쇠

# ---------------------------------------------------------
# [논문 기반 최적화] 아키텍처 파라미터 (표현력 강화)
# ---------------------------------------------------------
# 논문 근거: Yang et al. (2020) - 복잡한 전략 융합을 위한 용량 증설
# Wang et al. (KDD 2023) - 12개 전략·15차원 Info 처리 시 Information Bottleneck 방지
NETWORK_HIDDEN_DIM = 256     # 128 -> 256 (전략 상호작용 정보 수용량 증대)
NETWORK_NUM_LAYERS = 2       # 1 -> 2 (추상화 깊이 확보, 단순 패턴 -> 복합 패턴 인식)
NETWORK_DROPOUT = 0.1        # 유지 (과적합 방지)
# Causal Conv 파라미터 (Look-ahead Bias 방지, 수용 범위 확장)
CONV_KERNEL_SIZE = 5       # 수용 범위 확장
CONV_DILATION = 2          # 수용 범위 (5-1)*2+1 = 9스텝
NETWORK_ATTENTION_HEADS = 4  # Multi-Head Attention 헤드 개수
NETWORK_INFO_ENCODER_DIM = 128  # Info Encoder 출력 차원
NETWORK_SHARED_TRUNK_DIM1 = 256  # Shared Trunk 첫 번째 레이어 차c원
NETWORK_SHARED_TRUNK_DIM2 = 128  # Shared Trunk 두 번째 레이어 차원
NETWORK_ACTOR_HEAD_DIM = 64  # Actor Head 은닉층 차원
NETWORK_CRITIC_HEAD_DIM = 32  # Critic Head 은닉층 차원
NETWORK_USE_CHECKPOINTING = False  # Gradient Checkpointing 사용 여부

# ---------------------------------------------------------
# [논문 기반 최적화] 학습 파라미터 (노이즈 강건성)
# ---------------------------------------------------------
# 논문 근거: FinRL (2021) - 금융 데이터의 노이즈를 고려한 배치 사이즈 축소
TRAIN_ACTION_DIM = 3  # 행동 차원 (0:HOLD, 1:BUY, 2:SELL) - 3-Action 구조
TRAIN_BATCH_SIZE = 256       # 512 -> 256 (업데이트 빈도 2배 증가로 적응력 향상)
TRAIN_SAMPLE_SIZE = 50000    # 유지

# 데이터 분할 비율 (명확화)
TRAIN_SPLIT = 0.7  # 학습용 (0% ~ 70%)
VAL_SPLIT = 0.15   # 검증용 (70% ~ 85%)
TEST_SPLIT = 0.15  # 테스트용 (85% ~ 100%)
# 합계가 1.0이 되어야 함 (0.7 + 0.15 + 0.15 = 1.0)

TRAIN_NUM_EPISODES = 5000  # 에피소드 수
TRAIN_MAX_STEPS_PER_EPISODE = 480  # 에피소드당 최대 스텝 수
MAX_TRADES_PER_EPISODE = 50  # 과도한 거래 방지 (480스텝 기준)
TRAIN_SAVE_INTERVAL = 50  # 모델 저장 간격 (에피소드)

# 평가 파라미터
EVAL_INITIAL_CAPITAL = 10000  # 평가 시작 자본금
EVAL_VERBOSE_INTERVAL = 100  # 진행 상황 출력 간격 (스텝)
