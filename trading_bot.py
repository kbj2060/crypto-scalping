"""
메인 트레이딩 봇 (DDQN Agent 적용 버전)
"""
import logging
import time
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config
from core.data_collector import DataCollector
from core.risk_manager import RiskManager
from core.binance_client import BinanceClient
from strategies import (
    BTCEthCorrelationStrategy,
    VolatilitySqueezeStrategy,
    OrderblockFVGStrategy,
    HMAMomentumStrategy,
    MFIMomentumStrategy,
    BollingerMeanReversionStrategy,
    VWAPDeviationStrategy,
    RangeTopBottomStrategy,
    StochRSIMeanReversionStrategy,
    CMFDivergenceStrategy
)

# AI 강화학습 모듈
TORCH_AVAILABLE = False
if config.ENABLE_AI:
    try:
        import torch
        from model.trading_env import TradingEnvironment
        from model.dqn_agent import DDQNAgent  # [변경] DDQN 에이전트
        from model.preprocess import DataPreprocessor
        from model.mtf_processor import MTFProcessor
        from model.train_dqn import calculate_technical_features # 학습 코드에서 지표 계산 함수 재사용
        TORCH_AVAILABLE = True
    except ImportError as e:
        TORCH_AVAILABLE = False
        print(f"⚠️ AI 모듈 로드 실패: {e}")

# 로깅 설정
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingBot:
    def __init__(self):
        self.data_collector = DataCollector()
        self.risk_manager = RiskManager()
        self.client = BinanceClient()
        
        # 전략 초기화
        self.breakout_strategies = []
        self.range_strategies = []
        self._init_strategies()
        
        # 전체 전략 리스트
        self.strategies = self.breakout_strategies + self.range_strategies
        
        # AI 강화학습 초기화
        self.use_ai = config.ENABLE_AI and TORCH_AVAILABLE
        self.env = None
        self.agent = None
        self.current_position = None 
        self.entry_price = None
        self.entry_time = None
        self.selected_features = None  # 학습 시 선택된 피처 저장
        
        if self.use_ai:
            self._init_ai_agent()
        
        logger.info(f"트레이딩 봇 초기화 완료 - 전략: {len(self.strategies)}개")
        if self.use_ai:
            logger.info("🤖 AI(DDQN) 기반 결정 모드 활성화")

    def _init_strategies(self):
        """전략 객체 초기화"""
        if config.STRATEGIES['btc_eth_correlation']: 
            self.breakout_strategies.append(BTCEthCorrelationStrategy())
        if config.STRATEGIES.get('volatility_squeeze', False): 
            self.breakout_strategies.append(VolatilitySqueezeStrategy())
        if config.STRATEGIES.get('orderblock_fvg', False): 
            self.breakout_strategies.append(OrderblockFVGStrategy())
        if config.STRATEGIES.get('hma_momentum', False): 
            self.breakout_strategies.append(HMAMomentumStrategy())
        if config.STRATEGIES.get('mfi_momentum', False): 
            self.breakout_strategies.append(MFIMomentumStrategy())
        
        if config.STRATEGIES.get('bollinger_mean_reversion', False): 
            self.range_strategies.append(BollingerMeanReversionStrategy())
        if config.STRATEGIES.get('vwap_deviation', False): 
            self.range_strategies.append(VWAPDeviationStrategy())
        if config.STRATEGIES.get('range_top_bottom', False): 
            self.range_strategies.append(RangeTopBottomStrategy())
        if config.STRATEGIES.get('stoch_rsi_mean_reversion', False): 
            self.range_strategies.append(StochRSIMeanReversionStrategy())
        if config.STRATEGIES.get('cmf_divergence', False): 
            self.range_strategies.append(CMFDivergenceStrategy())

    def _init_ai_agent(self):
        """DDQN 에이전트 및 환경 초기화"""
        try:
            logger.info("🧠 AI 에이전트 초기화 중...")
            
            # ---------------------------------------------------------------------
            # [수정] 저장된 피처 목록 파일(json)이 있으면 그걸 우선 사용!
            # ---------------------------------------------------------------------
            features_path = 'saved_models/selected_features.json'
            
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    # config 설정을 덮어씌움 (이게 가장 확실함)
                    config.FEATURE_COLUMNS = json.load(f)
                logger.info(f"📂 학습된 피처 목록 로드 완료: {len(config.FEATURE_COLUMNS)}개")
                logger.info(f"📋 피처 목록: {config.FEATURE_COLUMNS}")
            else:
                logger.warning(f"⚠️ {features_path} 파일이 없습니다. config.FEATURE_COLUMNS를 그대로 사용합니다.")
                # 파일이 없으면 config에 의존 (위험할 수 있음)
            
            # 이후 로직은 config.FEATURE_COLUMNS를 사용하므로 자연스럽게 연결됨
            self.selected_features = config.FEATURE_COLUMNS
            
            # 2. 트레이딩 환경 생성
            self.env = TradingEnvironment(
                self.data_collector, 
                strategies=[], 
                lookback=config.LOOKBACK_WINDOW,
                selected_features=self.selected_features
            )
            
            # 3. 학습된 스케일러 로드
            scaler_path = 'saved_models/scaler.pkl'
            if self.env.preprocessor.load_scaler(scaler_path):
                self.env.scaler_fitted = True
                logger.info(f"✅ 스케일러 로드 완료: {scaler_path}")
            else:
                logger.error(f"❌ 스케일러 파일({scaler_path})이 없습니다.")
                self.use_ai = False
                return
            
            # 4. DDQN 에이전트 생성
            ddqn_config = config.DDQN_CONFIG
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.agent = DDQNAgent(
                input_dim=len(self.selected_features),
                hidden_dim=ddqn_config['hidden_dim'],
                num_layers=ddqn_config['num_layers'],
                action_dim=ddqn_config['action_dim'],
                lr=ddqn_config['learning_rate'],
                device=device,
                # 실전에서는 탐험 불필요
                epsilon_start=0.0,
                epsilon_end=0.0
            )
            
            # 5. 모델 가중치 로드
            model_path = 'saved_models/best_ddqn_model.pth'
            if os.path.exists(model_path):
                self.agent.load_model(model_path)
                self.agent.policy_net.eval()  # [중요] 평가 모드
                logger.info(f"✅ 학습된 모델 로드 완료: {model_path}")
            else:
                logger.error(f"❌ 모델 파일이 없습니다: {model_path}")
                self.use_ai = False
                return
                
        except Exception as e:
            logger.error(f"AI 초기화 실패: {e}", exc_info=True)
            self.use_ai = False

    def update_data(self):
        """데이터 업데이트"""
        return self.data_collector.update_data()
    
    def _prepare_ai_features(self):
        """실시간 데이터에 대해 학습과 동일한 피처 계산 및 주입"""
        try:
            # 1. 기술적 지표 계산 (TA)
            tech_df = calculate_technical_features(self.data_collector.eth_data)
            if tech_df is None: 
                return False
            
            # 데이터 병합
            for col in tech_df.columns:
                self.data_collector.eth_data[col] = tech_df[col]
            
            # 2. MTF (Multi-Timeframe) 피처 계산
            try:
                # 인덱스가 DatetimeIndex인지 확인
                if not isinstance(self.data_collector.eth_data.index, pd.DatetimeIndex):
                    try:
                        self.data_collector.eth_data.index = pd.to_datetime(self.data_collector.eth_data.index)
                    except:
                        logger.warning("인덱스를 DatetimeIndex로 변환할 수 없습니다. MTF 프로세서를 건너뜁니다.")
                        return True  # MTF 없이 계속 진행
                
                mtf_processor = MTFProcessor(self.data_collector.eth_data)
                self.data_collector.eth_data = mtf_processor.add_mtf_features()
            except Exception as e:
                logger.warning(f"MTF 계산 오류: {e}")

            # 3. 전략 점수(Strategy Scores) 실시간 계산 (12개)
            strat_map = {
                'BTCEthCorrelationStrategy': 'strat_btc_eth_corr',
                'VolatilitySqueezeStrategy': 'strat_vol_squeeze',
                'OrderblockFVGStrategy': 'strat_ob_fvg',
                'HMAMomentumStrategy': 'strat_hma',
                'MFIMomentumStrategy': 'strat_mfi',
                'BollingerMeanReversionStrategy': 'strat_bb_reversion',
                'VWAPDeviationStrategy': 'strat_vwap',
                'RangeTopBottomStrategy': 'strat_range',
                'StochRSIMeanReversionStrategy': 'strat_stoch',
                'CMFDivergenceStrategy': 'strat_cmf',
                'CCIReversalStrategy': 'strat_cci_reversal',  # [신규] CCI 반전 전략
                'WilliamsRStrategy': 'strat_williams_r'       # [신규] Williams %R 전략
            }
            
            # 전략 점수 컬럼 초기화 (없으면 생성)
            for strat_col in strat_map.values():
                if strat_col not in self.data_collector.eth_data.columns:
                    self.data_collector.eth_data[strat_col] = 0.0

            # 현재 시점(마지막 캔들)의 전략 점수만 업데이트
            if len(self.data_collector.eth_data) > 0:
                for strategy in self.strategies:
                    strat_col = strat_map.get(type(strategy).__name__)
                    if not strat_col or strat_col not in self.data_collector.eth_data.columns: 
                        continue
                    
                    try:
                        # 전략 분석 실행
                        result = strategy.analyze(self.data_collector)
                        score = 0.0
                        if result:
                            conf = float(result.get('confidence', 0.5))
                            if result['signal'] == 'LONG': 
                                score = conf
                            elif result['signal'] == 'SHORT': 
                                score = -conf
                        
                        # 데이터프레임에 값 할당 (마지막 행) - 더 안전한 방법
                        self.data_collector.eth_data.at[self.data_collector.eth_data.index[-1], strat_col] = score
                        
                    except Exception as e:
                        logger.debug(f"전략 {strat_col} 계산 오류: {e}")

            # 4. 누락된 피처 0으로 채우기
            for col in self.selected_features:
                if col not in self.data_collector.eth_data.columns:
                    self.data_collector.eth_data[col] = 0.0
            
            return True

        except Exception as e:
            logger.error(f"피처 준비 중 오류: {e}", exc_info=True)
            return False

    def _run_ai_mode(self):
        """AI(DDQN) 기반 실시간 결정 및 실행"""
        try:
            # 1. 최신 데이터로 피처 업데이트
            if not self._prepare_ai_features():
                return

            # [🚨 긴급 수정] 인덱스를 데이터의 가장 마지막(최신)으로 설정해야 함!
            # 이걸 안 하면 기본값 0이 되어 'start=-60' 에러가 남
            self.data_collector.current_index = len(self.data_collector.eth_data)

            # 2. 현재 가격 및 포지션 정보
            current_price = float(self.data_collector.eth_data.iloc[-1]['close'])
            
            pos_val = 1.0 if self.current_position == 'LONG' else (-1.0 if self.current_position == 'SHORT' else 0.0)
            pnl_val = 0.0
            hold_val = 0.0
            
            if self.current_position and self.entry_price:
                if self.current_position == 'LONG':
                    pnl_val = (current_price - self.entry_price) / self.entry_price
                else:
                    pnl_val = (self.entry_price - current_price) / self.entry_price
                
                if self.entry_time:
                    hold_minutes = (datetime.now() - self.entry_time).total_seconds() / 60
                    hold_val = min(1.0, hold_minutes / 160.0)  # 정규화

            # 3. 관측(Observation) 생성
            state = self.env.get_observation(position_info=[pos_val, pnl_val * 10, hold_val])
            
            if state is None:
                logger.warning("AI 관측 생성 실패 (데이터 부족 등)")
                return

            # 4. 모델 추론 (Action 결정)
            action = self.agent.act(state, training=False)
            
            # Q-값 확인 (로그용)
            with torch.no_grad():
                obs_seq, _ = state
                q_values = self.agent.policy_net(obs_seq.to(self.agent.device))
            
            action_names = {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}
            logger.info(f"🤖 AI 결정: {action_names[action]} | Q-Values: {q_values.cpu().numpy()[0]}")

            # 5. 거래 실행 로직
            signal = None
            if action == 1:  # LONG
                if self.current_position != 'LONG':
                    signal = {'signal': 'LONG', 'entry_price': current_price, 'strategy': 'DDQN Agent'}
            elif action == 2:  # SHORT
                if self.current_position != 'SHORT':
                    signal = {'signal': 'SHORT', 'entry_price': current_price, 'strategy': 'DDQN Agent'}
            
            # 신호가 있고 포지션 변경이 필요한 경우 실행
            if signal:
                logger.info(f"✨ AI 매매 신호 발생: {signal['signal']}")
                if config.ENABLE_TRADING:
                    if self.execute_trade(signal):
                        self.current_position = signal['signal']
                        self.entry_price = current_price
                        self.entry_time = datetime.now()
                else:
                    logger.info(f"   (모의 투자) {signal['signal']} 진입 @ {current_price}")
                    self.current_position = signal['signal']  # 모의 상태 업데이트

        except Exception as e:
            logger.error(f"AI 추론 루프 오류: {e}", exc_info=True)

    def execute_trade(self, signal):
        """실제 거래 실행"""
        try:
            if not config.ENABLE_TRADING:
                return True  # 모의 투자 모드
            
            # RiskManager를 통한 검증
            if not self.risk_manager.validate_signal(signal):
                logger.warning("거래 신호가 리스크 관리 규칙에 의해 거부되었습니다.")
                return False
            
            # BinanceClient를 통한 실제 거래 실행
            # (실제 구현은 BinanceClient에 따라 다름)
            logger.info(f"거래 실행: {signal}")
            return True
            
        except Exception as e:
            logger.error(f"거래 실행 오류: {e}", exc_info=True)
            return False

    def monitor_positions(self):
        """포지션 모니터링"""
        if self.current_position and self.entry_price:
            current_price = float(self.data_collector.eth_data.iloc[-1]['close'])
            if self.current_position == 'LONG':
                pnl = (current_price - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - current_price) / self.entry_price
            
            logger.info(f"📊 현재 포지션: {self.current_position} | 진입가: ${self.entry_price:.2f} | 현재가: ${current_price:.2f} | PnL: {pnl:.2%}")

    def _wait_for_next_candle(self):
        """다음 캔들까지 대기"""
        time.sleep(180)  # 3분봉이므로 180초 대기

    def run(self):
        """메인 루프"""
        logger.info("🚀 트레이딩 봇 시작")
        if not self.update_data(): 
            return
        
        iteration = 0
        while True:
            try:
                iteration += 1
                logger.info(f"\n=== Iteration {iteration} ({datetime.now().strftime('%H:%M:%S')}) ===")
                
                # 데이터 업데이트
                if not self.update_data():
                    time.sleep(5)
                    continue
                
                # 모니터링
                self.monitor_positions()

                # AI 모드 실행
                if self.use_ai:
                    self._run_ai_mode()
                else:
                    logger.warning("AI 모드가 비활성화되어 있습니다.")
                
                # 대기
                self._wait_for_next_candle()
                
            except KeyboardInterrupt:
                logger.info("트레이딩 봇 종료 요청")
                break
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                time.sleep(10)

if __name__ == '__main__':
    bot = TradingBot()
    bot.run()
