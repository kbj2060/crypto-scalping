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
LEVERAGE = int(os.getenv('LEVERAGE', '10'))
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
    'cmf_divergence': True
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
MIN_HOLDING_TIME = 5  # 최소 보유 캔들 수 (Action Masking용)

# 보상 함수 파라미터
REWARD_MULTIPLIER = 300  # 수익 보상 배율
LOSS_PENALTY_MULTIPLIER = 500  # 손실 페널티 배율
TRANSACTION_COST = 0.0015  # 거래 비용 (0.15%, 스프레드 + 슬리피지 포함)
TIME_COST = 0.0005  # 시간 비용
STOP_LOSS_THRESHOLD = -0.02  # 강제 손절 임계값 (-2%)

# PPO 알고리즘 하이퍼파라미터
PPO_GAMMA = 0.99  # 할인율 (Discount Factor)
PPO_LAMBDA = 0.95  # GAE 람다 파라미터
PPO_EPS_CLIP = 0.2  # PPO 클리핑 범위
PPO_K_EPOCHS = 10  # PPO 업데이트 반복 횟수
PPO_ENTROPY_COEF = 0.05  # 엔트로피 계수 (초기값)
PPO_ENTROPY_DECAY = 0.999  # 엔트로피 감소율 (에피소드마다)
PPO_ENTROPY_MIN = 0.02  # 엔트로피 최소값
PPO_LEARNING_RATE = 0.00003  # 학습률
PPO_SCHEDULER_FACTOR = 0.5  # 학습률 스케줄러 감소율
PPO_SCHEDULER_PATIENCE = 200  # 학습률 스케줄러 인내심
PPO_SCHEDULER_MIN_LR = 1e-6  # 학습률 최소값

# 네트워크 아키텍처 파라미터
NETWORK_HIDDEN_DIM = 128  # 은닉층 차원
NETWORK_NUM_LAYERS = 2  # xLSTM 레이어 개수
NETWORK_DROPOUT = 0.1  # Dropout 비율
NETWORK_ATTENTION_HEADS = 4  # Multi-Head Attention 헤드 개수
NETWORK_INFO_ENCODER_DIM = 64  # Info Encoder 출력 차원
NETWORK_SHARED_TRUNK_DIM1 = 256  # Shared Trunk 첫 번째 레이어 차원
NETWORK_SHARED_TRUNK_DIM2 = 128  # Shared Trunk 두 번째 레이어 차원
NETWORK_ACTOR_HEAD_DIM = 64  # Actor Head 은닉층 차원
NETWORK_CRITIC_HEAD_DIM = 64  # Critic Head 은닉층 차원
NETWORK_USE_CHECKPOINTING = False  # Gradient Checkpointing 사용 여부

# 학습 파라미터
TRAIN_ACTION_DIM = 3  # 행동 차원 (HOLD, LONG, SHORT)
TRAIN_BATCH_SIZE = 1024  # 배치 크기 (메모리에서 업데이트할 최소 스텝 수)
TRAIN_SAMPLE_SIZE = 50000  # 스케일러 학습용 샘플 크기
TRAIN_SPLIT = 0.7  # 학습 데이터 비율 (70%)
VAL_SPLIT = 0.85  # 검증 데이터 비율 (85%, Train + Val)
TRAIN_NUM_EPISODES = 2000  # 에피소드 수
TRAIN_MAX_STEPS_PER_EPISODE = 480  # 에피소드당 최대 스텝 수
TRAIN_SAVE_INTERVAL = 50  # 모델 저장 간격 (에피소드)

# 평가 파라미터
EVAL_INITIAL_CAPITAL = 10000  # 평가 시작 자본금
EVAL_VERBOSE_INTERVAL = 100  # 진행 상황 출력 간격 (스텝)
