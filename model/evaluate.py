"""
DDQN 모델 평가(Backtest) 스크립트
학습된 모델(best_ddqn_model.pth)을 로드하여 테스트 데이터 구간에서 성능 측정
train_dqn.py와 동일한 Feature Engineering 파이프라인 적용
"""
import sys
import os
import json
import torch
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_collector import DataCollector
from model.dqn_agent import DDQNAgent
from model.trading_env import TradingEnvironment
from model.feature_engineering import FeatureEngineer
from model.mtf_processor import MTFProcessor
from model.train_dqn import precalculate_strategy_scores  # 학습 코드의 전략 계산 함수 재사용
import config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path='saved_models/best_ddqn_model.pth'):
        self.model_path = model_path
        self.data_collector = DataCollector(use_saved_data=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 데이터 로드
        if not self.data_collector.load_saved_data():
            raise ValueError("데이터 로드 실패")
            
        # ---------------------------------------------------------------------
        # 1. 피처 엔지니어링 (학습과 동일한 파이프라인)
        # ---------------------------------------------------------------------
        logger.info("🛠️ 데이터 전처리 및 피처 엔지니어링 시작...")
        
        # 1-1. 고급 피처 생성
        btc_df = getattr(self.data_collector, 'btc_data', None)
        engineer = FeatureEngineer(self.data_collector.eth_data, btc_df)
        enhanced_df = engineer.generate_features()
        self.data_collector.eth_data = enhanced_df
        
        # 1-2. MTF 피처 생성
        if not isinstance(self.data_collector.eth_data.index, pd.DatetimeIndex):
            if 'timestamp' in self.data_collector.eth_data.columns:
                self.data_collector.eth_data.index = pd.to_datetime(self.data_collector.eth_data['timestamp'], unit='ms')
            else:
                self.data_collector.eth_data.index = pd.to_datetime(self.data_collector.eth_data.index)
        
        mtf_processor = MTFProcessor(self.data_collector.eth_data)
        self.data_collector.eth_data = mtf_processor.add_mtf_features()
        
        # 1-3. 전략 점수 계산 (train_dqn의 함수 재사용)
        # 주의: 전략이 추가되었으므로 train_dqn.py의 precalculate 함수가 최신이어야 함
        logger.info("🧠 전략 신호 계산 중...")
        strat_df = precalculate_strategy_scores(self.data_collector, force_recalculate=False)
        
        # 인덱스 정렬 및 병합
        if not strat_df.index.equals(self.data_collector.eth_data.index):
            strat_df = strat_df.reindex(self.data_collector.eth_data.index).fillna(0)
            
        for col in strat_df.columns:
            self.data_collector.eth_data[col] = strat_df[col]
            
        # ---------------------------------------------------------------------
        # 2. 학습된 피처 목록 로드
        # ---------------------------------------------------------------------
        features_json_path = 'saved_models/selected_features.json'
        if os.path.exists(features_json_path):
            with open(features_json_path, 'r') as f:
                self.feature_columns = json.load(f)
            logger.info(f"📂 학습된 피처 목록 로드: {len(self.feature_columns)}개")
        else:
            logger.warning("⚠️ 피처 목록 파일이 없습니다. config.FEATURE_COLUMNS 사용")
            self.feature_columns = config.FEATURE_COLUMNS
            
        # 누락된 컬럼 0으로 채우기
        for col in self.feature_columns:
            if col not in self.data_collector.eth_data.columns:
                self.data_collector.eth_data[col] = 0.0
                
        # ---------------------------------------------------------------------
        # 3. 환경 및 에이전트 설정
        # ---------------------------------------------------------------------
        self.env = TradingEnvironment(
            self.data_collector,
            strategies=[],
            lookback=config.LOOKBACK_WINDOW,
            selected_features=self.feature_columns
        )
        
        # 스케일러 로드
        scaler_path = 'saved_models/scaler.pkl'
        if self.env.preprocessor.load_scaler(scaler_path):
            self.env.scaler_fitted = True
            logger.info("✅ 스케일러 로드 완료")
        else:
            logger.warning("⚠️ 스케일러 파일이 없습니다. 결과가 부정확할 수 있습니다.")
            
        # 에이전트 초기화
        ddqn_config = config.DDQN_CONFIG
        self.agent = DDQNAgent(
            input_dim=len(self.feature_columns),
            hidden_dim=ddqn_config['hidden_dim'],
            num_layers=ddqn_config['num_layers'],
            action_dim=ddqn_config['action_dim'],
            device=self.device,
            epsilon_start=0.0,  # 평가 모드: 탐험 없음
            epsilon_end=0.0,
            use_per=config.USE_PER,
            n_step=config.N_STEP
        )
        
        # 모델 가중치 로드
        if os.path.exists(self.model_path):
            self.agent.load_model(self.model_path)
            self.agent.policy_net.eval()
            logger.info(f"✅ 모델 로드 완료: {self.model_path}")
        else:
            raise ValueError(f"모델 파일 없음: {self.model_path}")

    def run_backtest(self, start_index=None, steps=2000):
        """백테스트 실행"""
        logger.info(f"🚀 백테스트 시작 (Steps: {steps})")
        
        # 테스트 구간 설정 (데이터의 마지막 부분 사용)
        total_len = len(self.data_collector.eth_data)
        if start_index is None:
            start_index = total_len - steps - 100
            if start_index < config.LOOKBACK_WINDOW:
                start_index = config.LOOKBACK_WINDOW
        
        self.data_collector.current_index = start_index
        
        balance = 1000.0  # 초기 자본 $1000
        initial_balance = balance
        position = None  # 'LONG', 'SHORT', None
        entry_price = 0.0
        entry_idx = 0
        
        history = []
        equity_curve = [balance]
        
        # 시뮬레이션 루프
        for i in range(steps):
            if self.data_collector.current_index >= total_len - 1:
                break
                
            # 1. 관측
            # 포지션 정보 구성
            pos_val = 1.0 if position == 'LONG' else (-1.0 if position == 'SHORT' else 0.0)
            pnl_val = 0.0
            hold_val = 0.0
            
            current_price = float(self.data_collector.eth_data.iloc[self.data_collector.current_index]['close'])
            
            if position:
                if position == 'LONG':
                    pnl_val = (current_price - entry_price) / entry_price
                else:
                    pnl_val = (entry_price - current_price) / entry_price
                hold_val = min(1.0, (self.data_collector.current_index - entry_idx) / 160.0)
            
            state = self.env.get_observation(position_info=[pos_val, pnl_val * 10, hold_val])
            
            if state is None:
                self.data_collector.current_index += 1
                continue
                
            # 2. 행동 결정 (Greedy)
            action = self.agent.act(state, training=False)
            
            # 3. 매매 로직
            # 0: HOLD, 1: LONG, 2: SHORT
            new_position = position
            trade_pnl = 0.0
            fee_rate = 0.0005  # 0.05%
            
            if action == 1:  # LONG 신호
                if position == 'SHORT':  # 숏 청산 후 롱
                    # 청산
                    pnl = (entry_price - current_price) / entry_price
                    realized_pnl = pnl - fee_rate
                    balance *= (1 + realized_pnl)
                    history.append({'type': 'CLOSE_SHORT', 'price': current_price, 'pnl': realized_pnl, 'balance': balance})
                    
                    # 진입
                    balance *= (1 - fee_rate)  # 진입 수수료
                    entry_price = current_price
                    entry_idx = self.data_collector.current_index
                    new_position = 'LONG'
                    history.append({'type': 'OPEN_LONG', 'price': current_price, 'balance': balance})
                    
                elif position is None:  # 신규 롱
                    balance *= (1 - fee_rate)
                    entry_price = current_price
                    entry_idx = self.data_collector.current_index
                    new_position = 'LONG'
                    history.append({'type': 'OPEN_LONG', 'price': current_price, 'balance': balance})
                    
            elif action == 2:  # SHORT 신호
                if position == 'LONG':  # 롱 청산 후 숏
                    # 청산
                    pnl = (current_price - entry_price) / entry_price
                    realized_pnl = pnl - fee_rate
                    balance *= (1 + realized_pnl)
                    history.append({'type': 'CLOSE_LONG', 'price': current_price, 'pnl': realized_pnl, 'balance': balance})
                    
                    # 진입
                    balance *= (1 - fee_rate)
                    entry_price = current_price
                    entry_idx = self.data_collector.current_index
                    new_position = 'SHORT'
                    history.append({'type': 'OPEN_SHORT', 'price': current_price, 'balance': balance})
                    
                elif position is None:  # 신규 숏
                    balance *= (1 - fee_rate)
                    entry_price = current_price
                    entry_idx = self.data_collector.current_index
                    new_position = 'SHORT'
                    history.append({'type': 'OPEN_SHORT', 'price': current_price, 'balance': balance})
            
            position = new_position
            equity_curve.append(balance)
            self.data_collector.current_index += 1
            
        # 결과 분석
        self._print_stats(initial_balance, balance, history, equity_curve)
        
    def _print_stats(self, initial, final, history, equity):
        """성과 분석 출력"""
        trades = [h for h in history if 'pnl' in h]
        wins = [t for t in trades if t['pnl'] > 0]
        
        total_return = (final - initial) / initial * 100
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        
        # MDD 계산
        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        mdd = drawdown.max() * 100
        
        print("\n" + "="*50)
        print(f"📊 백테스트 결과 (구간: {len(equity)} 캔들)")
        print("="*50)
        print(f"💰 초기 자본: ${initial:.2f}")
        print(f"💰 최종 자본: ${final:.2f}")
        print(f"📈 총 수익률: {total_return:.2f}%")
        print(f"📉 MDD (최대 낙폭): {mdd:.2f}%")
        print(f"🎲 총 거래 횟수: {len(trades)}회")
        print(f"🎯 승률: {win_rate:.2f}%")
        
        if trades:
            avg_pnl = np.mean([t['pnl'] for t in trades]) * 100
            print(f"⚖️ 평균 손익: {avg_pnl:.4f}%")
        print("="*50 + "\n")
        
        # 수익 곡선 그래프
        plt.figure(figsize=(12, 6))
        plt.plot(equity, label='Equity Curve')
        plt.title(f'Backtest Result (Return: {total_return:.2f}%)')
        plt.xlabel('Steps')
        plt.ylabel('Balance ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == '__main__':
    try:
        evaluator = ModelEvaluator()
        # 최근 2000개 데이터(약 4일치)로 테스트
        evaluator.run_backtest(steps=2000)
    except Exception as e:
        logger.error(f"평가 중 오류: {e}", exc_info=True)
