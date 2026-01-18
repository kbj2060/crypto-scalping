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
    'liquidity_sweep': True,
    'btc_eth_correlation': True,
    'cvd_delta': True,
    'volatility_squeeze': True,
    'funding_rate': True,
    'orderblock_fvg': True,
    'liquidation_spike': True
}

# 시간프레임 설정
TIMEFRAME = '3m'  # 3분봉
LOOKBACK_PERIOD = 1500  # 과거 데이터 조회 기간 (1500봉)

# 거래 실행 설정
ENABLE_TRADING = False  # True: 거래 실행, False: 분석만 수행 (거래 비활성화)
