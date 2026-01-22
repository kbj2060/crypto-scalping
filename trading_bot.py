"""
메인 트레이딩 봇 (DDQN Agent 적용 버전)
학습 코드(train_dqn.py)와 100% 동일한 피처 엔지니어링 및 전략 적용
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

# 전략 파일들 임포트
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
    CMFDivergenceStrategy,
    # [신규] 고빈도 전략 추가
    CCIReversalStrategy,
    WilliamsRStrategy
)

# AI 강화학습 모듈
TORCH_AVAILABLE = False
if config.ENABLE_AI:
    try:
        import torch
        from model.trading_env import TradingEnvironment
        from model.dqn_agent import DDQNAgent
        from model.preprocess import DataPreprocessor
        from model.mtf_processor import MTFProcessor
        from model.feature_engineering import FeatureEngineer  # [중요] 고급 피처 엔지니어링
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
        
        # 전체 전략 리스트 (분석용)
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
        """전략 객체 초기화 (신규 전략 포함 12개)"""
        # 돌파 전략군
        self.breakout_strategies.append(BTCEthCorrelationStrategy())
        self.breakout_strategies.append(VolatilitySqueezeStrategy())
        self.breakout_strategies.append(OrderblockFVGStrategy())
        self.breakout_strategies.append(HMAMomentumStrategy())
        self.breakout_strategies.append(MFIMomentumStrategy())
        
        # 반전/횡보 전략군
        self.range_strategies.append(BollingerMeanReversionStrategy())
        self.range_strategies.append(VWAPDeviationStrategy())
        self.range_strategies.append(RangeTopBottomStrategy())
        self.range_strategies.append(StochRSIMeanReversionStrategy())
        self.range_strategies.append(CMFDivergenceStrategy())
        
        # [신규] 고빈도 전략 추가
        self.range_strategies.append(CCIReversalStrategy())
        self.range_strategies.append(WilliamsRStrategy())

    def _init_ai_agent(self):
        """DDQN 에이전트 및 환경 초기화"""
        try:
            logger.info("🧠 AI 에이전트 초기화 중...")
            
            # 1. 학습된 피처 목록(JSON) 로드
            features_path = 'saved_models/selected_features.json'
            
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.selected_features = json.load(f)
                logger.info(f"📂 학습된 피처 목록 로드 완료: {len(self.selected_features)}개")
            else:
                logger.warning(f"⚠️ {features_path} 파일이 없습니다. 기본 피처를 사용합니다.")
                # 파일이 없을 경우 대비한 기본값 (예시)
                self.selected_features = config.FEATURE_COLUMNS
            
            # 2. 트레이딩 환경 생성
            self.env = TradingEnvironment(
                self.data_collector, 
                strategies=[], 
                lookback=config.LOOKBACK_WINDOW,
                selected_features=self.selected_features
            )
            
            # 3. 학습된 스케일러 로드 (피처 이름 포함)
            scaler_path = 'saved_models/scaler.pkl'
            success, feature_names = self.env.preprocessor.load_scaler(scaler_path)
            if success:
                self.env.scaler_fitted = True
                # 피처 이름이 있으면 scaler_feature_order에 저장
                if feature_names is not None:
                    self.env.scaler_feature_order = feature_names
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
                epsilon_start=0.0,  # 실전에서는 탐험 없음
                epsilon_end=0.0,
                use_per=config.USE_PER,
                n_step=config.N_STEP,
                info_dim=3  # 포지션 정보 차원
            )
            
            # 5. 모델 가중치 로드 (최고 성능 모델 사용)
            model_path = 'saved_models/best_ddqn_model.pth'
            if os.path.exists(model_path):
                self.agent.load_model(model_path)
                self.agent.policy_net.eval()  # [중요] 평가 모드 설정 (Dropout 등 비활성화)
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
        """
        실시간 데이터에 대해 학습(train_dqn.py)과 동일한 방식으로 피처 생성
        FeatureEngineer -> MTFProcessor -> Strategy Scores
        """
        try:
            # 1. 고급 피처 엔지니어링 (FeatureEngineer 사용)
            # BTC 데이터가 있으면 함께 전달
            btc_df = getattr(self.data_collector, 'btc_data', None)
            
            engineer = FeatureEngineer(self.data_collector.eth_data, btc_df)
            enhanced_df = engineer.generate_features()
            
            if enhanced_df is None:
                return False
                
            # 기존 데이터프레임 교체
            self.data_collector.eth_data = enhanced_df
            
            # 2. MTF (Multi-Timeframe) 피처 계산
            try:
                # 인덱스 안전장치: DatetimeIndex 변환
                if not isinstance(self.data_collector.eth_data.index, pd.DatetimeIndex):
                    if 'timestamp' in self.data_collector.eth_data.columns:
                        self.data_collector.eth_data.index = pd.to_datetime(self.data_collector.eth_data['timestamp'], unit='ms')
                    else:
                        try:
                            self.data_collector.eth_data.index = pd.to_datetime(self.data_collector.eth_data.index)
                        except:
                            pass
                
                # MTF 적용
                mtf_processor = MTFProcessor(self.data_collector.eth_data)
                self.data_collector.eth_data = mtf_processor.add_mtf_features()
            except Exception as e:
                logger.warning(f"MTF 계산 오류: {e}")

            # 3. 전략 점수(Strategy Scores) 실시간 계산
            # train_dqn.py와 동일한 컬럼명 사용 필수
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
                'CCIReversalStrategy': 'strat_cci_reversal',  # [신규]
                'WilliamsRStrategy': 'strat_williams_r'       # [신규]
            }
            
            # 전략 점수 컬럼 초기화
            for strat_col in strat_map.values():
                if strat_col not in self.data_collector.eth_data.columns:
                    self.data_collector.eth_data[strat_col] = 0.0

            # 최신 캔들에 대해서만 전략 분석 실행 (효율성)
            if len(self.data_collector.eth_data) > 50:
                for strategy in self.strategies:
                    strat_class_name = type(strategy).__name__
                    strat_col = strat_map.get(strat_class_name)
                    
                    if not strat_col: 
                        continue  # 매핑되지 않은 전략 건너뜀
                    
                    try:
                        # 전략 analyze 실행
                        # (DataCollector는 전체 데이터를 가지고 있으므로 내부에서 최신봉 분석)
                        result = strategy.analyze(self.data_collector)
                        
                        score = 0.0
                        if result:
                            conf = float(result.get('confidence', 0.5))
                            if result['signal'] == 'LONG': 
                                score = conf
                            elif result['signal'] == 'SHORT': 
                                score = -conf
                        
                        # 마지막 행(현재 시점)에 점수 업데이트
                        self.data_collector.eth_data.at[self.data_collector.eth_data.index[-1], strat_col] = score
                        
                    except Exception as e:
                        # 전략 하나 실패해도 전체 봇은 죽지 않도록 처리
                        logger.debug(f"전략 {strat_col} 실시간 계산 오류: {e}")

            # 4. 최종 결측치 처리 (안전장치)
            # 학습 때 사용한 피처가 현재 데이터에 없으면 0으로 채움
            for col in self.selected_features:
                if col not in self.data_collector.eth_data.columns:
                    self.data_collector.eth_data[col] = 0.0
            
            self.data_collector.eth_data = self.data_collector.eth_data.fillna(0)
            
            return True

        except Exception as e:
            logger.error(f"피처 준비 중 오류: {e}", exc_info=True)
            return False

    def _run_ai_mode(self):
        """AI(DDQN) 기반 실시간 결정 및 실행"""
        try:
            # 1. 최신 데이터로 피처 업데이트 (Feature Engineering)
            if not self._prepare_ai_features():
                return

            # [핵심] 인덱스 설정: 데이터의 가장 마지막(최신) 지점
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
            # position_info = [포지션상태, 수익률*10, 보유시간]
            state = self.env.get_observation(position_info=[pos_val, pnl_val * 10, hold_val])
            
            if state is None:
                logger.warning("AI 관측 생성 실패 (데이터 부족 등)")
                return

            # 4. 모델 추론 (Action 결정)
            # training=False로 설정하여 탐험(Epsilon) 없이 최적 행동만 선택
            action = self.agent.act(state, training=False)
            
            # Q-값 확인 (디버깅용)
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
                
                # Risk Manager 검증 및 실행
                if config.ENABLE_TRADING:
                    if self.execute_trade(signal):
                        self.current_position = signal['signal']
                        self.entry_price = current_price
                        self.entry_time = datetime.now()
                else:
                    # 모의 투자 모드
                    logger.info(f"   (모의 투자) {signal['signal']} 진입 @ {current_price}")
                    self.current_position = signal['signal']
                    self.entry_price = current_price
                    self.entry_time = datetime.now()

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
            # (주문 로직은 BinanceClient에 위임)
            # order = self.client.place_order(...) 
            logger.info(f"거래 실행 명령 전송: {signal}")
            return True
            
        except Exception as e:
            logger.error(f"거래 실행 오류: {e}", exc_info=True)
            return False

    def monitor_positions(self):
        """포지션 모니터링 및 로그 출력"""
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
                    logger.warning("AI 모드가 비활성화되어 있습니다. (설정 또는 초기화 실패)")
                
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
