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
AI_MODEL_PATH = 'model/ppo_model.pth'  # AI 모델 저장 경로 (PPO)
DDQN_MODEL_PATH = 'model/ddqn_model.pth'  # DDQN 모델 저장 경로
SELECTED_FEATURES_PATH = 'model/selected_features.json'  # 호환성 유지용

# 1. 기술적 지표 피처 (17개)
TECHNICAL_FEATURES = [
    'log_return', 'log_volume',         # PPO 기본 (2)
    'high_ratio', 'low_ratio',          # 캔들 모양/꼬리 (2)
    'taker_ratio',                      # 수급 (1)
    'rsi', 'macd_hist',                 # 모멘텀 (2)
    'bb_width', 'bb_position',          # 변동성/위치 (2)
    'stoch_rsi', 'mfi', 'cmf',          # 오실레이터/자금 (3)
    'hma_ratio', 'vwap_dist', 'atr_ratio', # 이격도 (3)
    'adx', 'chop'                       # 추세/횡보 판별 (2) - 신규 추가
]

# 2. 전략 기반 피처 (10개) - 새로 추가됨
STRATEGY_FEATURES = [
    'strat_btc_eth_corr',    # BTC 연동
    'strat_vol_squeeze',     # 변동성 스퀴즈
    'strat_ob_fvg',          # 오더블록+FVG
    'strat_hma',             # HMA 모멘텀
    'strat_mfi',             # MFI 모멘텀
    'strat_bb_reversion',    # 볼린저 역추세
    'strat_vwap',            # VWAP 이격
    'strat_range',           # 박스권 반전
    'strat_stoch',           # StochRSI
    'strat_cmf'              # CMF 다이버전스
]

# 최종 사용할 모든 피처 합치기
FEATURE_COLUMNS = TECHNICAL_FEATURES + STRATEGY_FEATURES

# 환경 설정
LOOKBACK_WINDOW = 60  # 3분봉 * 60 = 180분 (3시간)

# 3. DDQN 하이퍼파라미터
DDQN_CONFIG = {
    'input_dim': len(FEATURE_COLUMNS),  # 17 + 10 = 27개 (자동 계산)
    'hidden_dim': 128,  # GRU 및 FC 레이어 노드 수 (입력 증가에 따라 64 -> 128)
    'num_layers': 2,  # GRU 레이어 수
    'action_dim': 3,  # 행동 개수 (0: Hold, 1: Long, 2: Short)
    'batch_size': 64,  # 한 번 학습 시 사용할 샘플 수
    'learning_rate': 0.0001,  # 학습률 (1e-4)
    'gamma': 0.99,  # 미래 보상 할인율
    'buffer_size': 50000,  # 리플레이 버퍼 크기
    'epsilon_start': 1.0,  # 초기 탐험 확률
    'epsilon_end': 0.01,  # 최소 탐험 확률
    'epsilon_decay': 0.999999,  # 탐험 감소 비율 (매우 천천히 감소하여 탐험 기간 연장)
    'target_update': 1000,  # 타겟 네트워크 동기화 주기 (스텝)
}
