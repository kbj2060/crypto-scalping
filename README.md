# 암호화폐 스캘핑 자동매매 봇

바이낸스 선물 거래를 위한 이더리움 자동매매 프로그램입니다. 7가지 고급 매매 전략을 통합하여 실시간으로 거래를 실행합니다.

## 주요 기능

### 7가지 매매 전략

#### 폭발장 전략 (Breakout/Trend)

1. **유동성 스윕(Liquidity Sweep) 전략**
   - 고점/저점의 유동성을 터치하고 반전하는 현상 탐지
   - Swing High/Low 기반 스윕 탐지
   - 손절: 스윕 고점/저점 바깥 0.2%

2. **BTC/ETH 강도 상관 전략**
   - BTC의 방향성을 ETH 거래의 선행 신호로 활용
   - BTC RSI, 모멘텀, 단기 트렌드 분석
   - 1~3캔들 내 ETH 방향 확인 후 진입

3. **CVD + Aggressive Delta 전략**
   - 가격과 CVD의 다이버전스 탐지
   - 스마트머니의 흡수 패턴 분석
   - Bullish/Bearish Divergence 기반 진입

4. **변동성 수축 → 폭발 전략 (Squeeze)**
   - 볼린저 밴드 수축 탐지
   - 거래량 수축 후 폭발 시점 포착
   - FVG 되돌림 진입 권장

6. **청산 스파이크 전략**
   - 대량 청산 발생 시점 탐지
   - 청산 후 반대 방향 움직임 포착
   - 강한 추세 전환 신호 활용

#### 횡보장 전략 (Range Trading)

7. **오더블록 + FVG 전략 (ICT/SMC)**
   - Fair Value Gap 탐지 및 회귀 진입
   - FVG 중간값 ~ 60% 구간 터치 시 진입
   - 손절: FVG 반대쪽 바깥

8. **펀딩비 극단 전략**
   - 펀딩비 극단값 활용 (0.02% 이상)
   - CVD 다이버전스와 결합하여 신뢰도 향상
   - BTC 방향성 확인으로 확률 상승

## 설치 방법

### 1. 가상환경 활성화

```bash
source venv-mac/bin/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

**참고**: `ta-lib` 패키지는 시스템에 TA-Lib 라이브러리가 설치되어 있어야 합니다.

macOS에서 TA-Lib 설치:
```bash
brew install ta-lib
```

### 3. 환경 변수 설정

`env.example` 파일을 참고하여 `.env` 파일을 생성하고 API 키를 입력하세요:

```bash
cp env.example .env
```

그 다음 `.env` 파일을 열어서 실제 API 키를 입력하세요:

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here
BINANCE_TESTNET=True
ETH_SYMBOL=ETHUSDT
BTC_SYMBOL=BTCUSDT
LEVERAGE=10
MAX_POSITION_SIZE=100
STOP_LOSS_PERCENT=0.2
```

**중요 설정 설명:**
- `BINANCE_TESTNET`: `True`로 설정하면 테스트넷에서 거래합니다. 실제 자금 사용 전 필수!
- `LEVERAGE`: 레버리지 배수 (1~125)
- `MAX_POSITION_SIZE`: 최대 포지션 크기 (USDT)
- `STOP_LOSS_PERCENT`: 손절 비율 (%)

**중요**: 실제 거래 전에 반드시 `BINANCE_TESTNET=True`로 테스트넷에서 테스트하세요!

## 사용 방법

### 봇 실행

```bash
python trading_bot.py
```

### 로그 확인

봇 실행 중 생성되는 `logs/trading_bot.log` 파일에서 상세한 로그를 확인할 수 있습니다.

## 프로젝트 구조

```
crypto-scalping/
├── core/                  # 핵심 모듈
│   ├── binance_client.py      # 바이낸스 API 클라이언트
│   ├── data_collector.py      # 실시간 데이터 수집
│   ├── indicators.py          # 기술적 지표 계산
│   └── risk_manager.py         # 리스크 관리
├── strategies/            # 매매 전략 모듈
│   ├── breakout/              # 폭발장 전략
│   │   ├── liquidity_sweep.py
│   │   ├── btc_eth_correlation.py
│   │   ├── cvd_delta.py
│   │   ├── volatility_squeeze.py
│   │   └── liquidation_spike.py
│   └── range/                 # 횡보장 전략
│       ├── funding_rate.py
│       └── orderblock_fvg.py
├── utils/                 # 유틸리티
│   └── check_api.py           # API 키 검증 스크립트
├── logs/                  # 로그 파일
│   └── trading_bot.log
├── trading_bot.py         # 메인 트레이딩 봇
├── config.py              # 설정 파일
├── requirements.txt       # 패키지 목록
└── README.md
```

## 리스크 관리

- **포지션 크기**: 리스크 기반 자동 계산
- **손절**: 기본 0.2% (설정 가능)
- **익절**: 2% 이상 수익 시 고려
- **최대 포지션**: 설정 파일에서 제한 가능

## 주의사항

⚠️ **이 프로그램은 교육 및 연구 목적으로 제작되었습니다.**

- 실제 자금으로 거래하기 전에 반드시 테스트넷에서 충분히 테스트하세요.
- 암호화폐 거래는 높은 리스크를 수반합니다.
- 과거 성과가 미래 수익을 보장하지 않습니다.
- 자신의 책임 하에 사용하시기 바랍니다.

## 라이선스

이 프로젝트는 개인 사용 목적으로 제작되었습니다.
