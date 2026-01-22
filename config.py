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
    'cmf_divergence': True,
    # 신규 추가 전략
    'cci_reversal': True,
    'williams_r': True
}

# 시간프레임 설정
TIMEFRAME = '3m'  # 3분봉
LOOKBACK_PERIOD = 1500  # 과거 데이터 조회 기간 (1500봉)

# 거래 실행 설정
ENABLE_TRADING = False  # True: 거래 실행, False: 분석만 수행 (거래 비활성화)

# AI 강화학습 설정
ENABLE_AI = True  # True: AI 기반 결정, False: 기존 전략 조합 방식
AI_MODEL_PATH = 'model/ppo_model.pth'  # AI 모델 저장 경로 (PPO)
DDQN_MODEL_PATH = 'saved_models/ddqn_model.pth'  # DDQN 모델 저장 경로
SELECTED_FEATURES_PATH = 'model/selected_features.json'  # 호환성 유지용

# 1. 기술적 지표 피처 (FeatureEngineer 기반 - 25개 기본 + 4개 MTF)
TECHNICAL_FEATURES = [
    # 가격 & 변동성 (9개)
    'log_return', 'roll_return_6',      # 수익률 (1봉, 6봉)
    'atr_ratio',                        # 변동성 확장 비율
    'bb_width', 'bb_pos',               # 볼린저 밴드 (너비, 위치)
    'rsi', 'macd_hist',                  # 모멘텀 지표
    'hma_ratio',                        # 추세 괴리율
    'cci',                              # 고빈도 스캘핑용 CCI
    
    # 거래량 & 오더플로우 (6개)
    'rvol',                             # 상대 거래량 (평소 대비)
    'taker_ratio',                      # 공격적 매수세
    'cvd_change',                       # 순매수 거래량 변화 (세력 흔적) 🔥
    'mfi', 'cmf',                       # 자금 흐름 지표
    'vwap_dist',                        # VWAP 이격도
    
    # 패턴 & 유동성 (5개)
    'wick_upper', 'wick_lower',         # 캔들 꼬리 (윗꼬리, 아랫꼬리)
    'range_pos',                        # 박스권 위치 (0=바닥, 1=천장)
    'swing_break',                      # 구조물 돌파 (1, 0, -1)
    'chop',                             # 추세 vs 횡보 판별
    
    # BTC 커플링 데이터 (5개)
    'btc_return',                      # 비트코인 수익률
    'btc_rsi',                          # 비트코인 과열도
    'btc_corr',                         # BTC-ETH 상관계수 🔥
    'btc_vol',                          # 비트코인 변동성
    'eth_btc_ratio',                    # ETH/BTC 비율 (알트장 여부)
    
    # 멀티 타임프레임 (MTF) (4개)
    'trend_1h', 'rsi_1h',               # 1시간봉 추세/RSI
    'trend_15m', 'rsi_15m'              # 15분봉 추세/RSI
]

# 2. 전략 기반 피처 (12개) - CCI Reversal, Williams %R 추가
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
    'strat_cmf',             # CMF 다이버전스
    'strat_cci_reversal',    # CCI 반전 전략 (신규)
    'strat_williams_r'       # Williams %R 전략 (신규)
]

# 최종 사용할 모든 피처 합치기
FEATURE_COLUMNS = TECHNICAL_FEATURES + STRATEGY_FEATURES

# 환경 설정
LOOKBACK_WINDOW = 60  # 3분봉 * 60 = 180분 (3시간)

# [신규] 피처 선택 설정
USE_XGBOOST_SELECTION = True  # 활성화 여부
TOP_K_FEATURES = 20            # 선택할 피처 개수 (DDQN 입력 차원)

# [신규] Prioritized Experience Replay (PER) 설정
USE_PER = True  # PER 사용 여부 (True: 우선순위 기반 샘플링, False: 일반 랜덤 샘플링)

# [신규] N-step Learning 설정
N_STEP = 3  # Multi-step Learning의 스텝 수 (기본 3, 1~5 권장)

# 3. DDQN 하이퍼파라미터 (성능 최적화 적용)
DDQN_CONFIG = {
    'input_dim': len(FEATURE_COLUMNS),  # 17 + 10 = 27개 (자동 계산)
    'hidden_dim': 128,  # GRU 및 FC 레이어 노드 수 (입력 증가에 따라 64 -> 128)
    'num_layers': 2,  # GRU 레이어 수
    'action_dim': 3,  # 행동 개수 (0: Hold, 1: Long, 2: Short)
    'batch_size': 128,  # [최적화] 한 번 학습 시 사용할 샘플 수 (64 -> 128)
    'learning_rate': 0.0001,  # 학습률 (Cosine Annealing 적용 예정)
    'gamma': 0.99,  # 미래 보상 할인율
    'buffer_size': 50000,  # 리플레이 버퍼 크기
    'epsilon_start': 0.0,  # 초기 탐험 확률
    'epsilon_end': 0.00,  # 최소 탐험 확률
    'epsilon_decay': 0.0,  # 탐험 감소 비율 (매우 천천히 감소하여 탐험 기간 연장)
    'target_update': 500,  # [최적화] 타겟 네트워크 동기화 주기 (1000 -> 500)
    'grad_clip': 0.5,  # [최적화] 그래디언트 클리핑 값 (1.0 -> 0.5)
    'use_lr_scheduler': True,  # [최적화] Learning Rate 스케줄링 사용 여부
}
