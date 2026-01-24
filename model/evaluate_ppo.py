"""
PPO 모델 평가 스크립트
학습된 모델을 테스트 데이터셋으로 평가하고 성능 지표를 출력합니다.
"""
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# 상위 폴더를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core import DataCollector
from strategies import (
    BTCEthCorrelationStrategy, VolatilitySqueezeStrategy, OrderblockFVGStrategy,
    HMAMomentumStrategy, MFIMomentumStrategy, BollingerMeanReversionStrategy,
    VWAPDeviationStrategy, RangeTopBottomStrategy, StochRSIMeanReversionStrategy,
    CMFDivergenceStrategy,
    CCIReversalStrategy, WilliamsRStrategy
)

from model.trading_env import TradingEnvironment
from model.ppo_agent import PPOAgent
from model.preprocess import DataPreprocessor

# 로깅 설정
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/evaluate_ppo.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 피처 엔지니어링 로그 끄기
logging.getLogger('model.feature_engineering').setLevel(logging.WARNING)
logging.getLogger('model.mtf_processor').setLevel(logging.WARNING)


class PPOModelEvaluator:
    """PPO 모델 평가 클래스"""
    
    def __init__(self, model_path=None, scaler_path=None):
        """
        Args:
            model_path: 모델 파일 경로 (None이면 config에서 가져옴)
            scaler_path: 스케일러 파일 경로 (None이면 config에서 가져옴)
            Note: 데이터 분할은 train_ppo.py와 동일하게 70:15:15 기준을 따름
        """
        self.model_path = model_path or config.AI_MODEL_PATH
        self.scaler_path = scaler_path or config.AI_MODEL_PATH.replace('.pth', '_scaler.pkl')
        
        # 1. 데이터 수집기 초기화
        self.data_collector = DataCollector(use_saved_data=True)
        
        # 2. 전략 초기화 (train_ppo.py와 동일)
        self.breakout_strategies = []
        self.range_strategies = []
        
        # 폭발장 전략
        if config.STRATEGIES.get('btc_eth_correlation', False):
            self.breakout_strategies.append(BTCEthCorrelationStrategy())
        if config.STRATEGIES.get('volatility_squeeze', False):
            self.breakout_strategies.append(VolatilitySqueezeStrategy())
        if config.STRATEGIES.get('orderblock_fvg', False):
            self.breakout_strategies.append(OrderblockFVGStrategy())
        if config.STRATEGIES.get('hma_momentum', False):
            self.breakout_strategies.append(HMAMomentumStrategy())
        if config.STRATEGIES.get('mfi_momentum', False):
            self.breakout_strategies.append(MFIMomentumStrategy())
        
        self.breakout_strategies.append(CCIReversalStrategy())
        
        # 횡보장 전략
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
        
        self.range_strategies.append(WilliamsRStrategy())
        
        self.strategies = self.breakout_strategies + self.range_strategies
        logger.info(f"✅ 전략 초기화 완료: 총 {len(self.strategies)}개")
        
        # 3. 환경 초기화
        self.env = TradingEnvironment(
            data_collector=self.data_collector,
            strategies=self.strategies
            # lookback과 min_holding_time은 config에서 자동으로 가져옴
        )
        
        # 4. 스케일러 로드
        self._load_scaler()
        
        # 5. 에이전트 초기화
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 디바이스: {self.device}")
        
        state_dim = self.env.get_state_dim()
        action_dim = 3  # HOLD, LONG, SHORT
        info_dim = len(self.strategies) + 3  # 전략 점수 + 포지션 정보
        
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.NETWORK_HIDDEN_DIM,
            device=self.device,
            info_dim=info_dim
        )
        
        # 6. 모델 로드
        self._load_model()
        
        # 평가 결과 저장
        self.trades = []  # 거래 내역
        self.equity_curve = []  # 자산 곡선
        self.actions_taken = {'HOLD': 0, 'LONG': 0, 'SHORT': 0}
        
    def _load_scaler(self):
        """스케일러 로드"""
        try:
            if os.path.exists(self.scaler_path):
                self.env.preprocessor.load(self.scaler_path)
                self.env.scaler_fitted = True
                logger.info(f"✅ 스케일러 로드 완료: {self.scaler_path}")
            else:
                logger.warning(f"⚠️ 스케일러 파일이 없습니다: {self.scaler_path}")
        except Exception as e:
            logger.error(f"❌ 스케일러 로드 실패: {e}", exc_info=True)
    
    def _load_model(self):
        """모델 로드"""
        try:
            if os.path.exists(self.model_path):
                self.agent.load_model(self.model_path)
                self.agent.model.eval()  # 평가 모드
                logger.info(f"✅ 모델 로드 완료: {self.model_path}")
            else:
                logger.error(f"❌ 모델 파일이 없습니다: {self.model_path}")
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}", exc_info=True)
            raise
    
    def _prepare_test_data(self):
        """
        테스트 데이터 준비 
        Train(70%) / Val(15%) / Test(15%) 기준을 따름 (train_ppo.py와 동일)
        """
        try:
            # 1. 데이터 로드 (피처 파일 우선)
            feature_file = 'data/training_features.csv'
            if os.path.exists(feature_file):
                logger.info(f"📂 피처 데이터 로드: {feature_file}")
                df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
                self.data_collector.eth_data = df
            else:
                logger.warning("⚠️ 피처 파일이 없어 원본 데이터를 사용합니다.")
                if self.data_collector.eth_data is None:
                    raise ValueError("데이터가 없습니다.")
                df = self.data_collector.eth_data

            total_len = len(df)
            
            # 2. 데이터 분할 (train_ppo.py와 동일한 기준 적용)
            train_end = int(total_len * config.TRAIN_SPLIT)
            val_end = int(total_len * config.VAL_SPLIT)
            
            # 평가 구간 설정 (기본적으로 Test Set인 마지막 15% 사용)
            # 필요에 따라 Validation Set(70%~85%)을 평가할 수도 있음
            test_start_idx = val_end
            
            logger.info(f"📊 전체 데이터: {total_len}개")
            logger.info(f"📊 데이터 분할: Train(0~{train_end}), Val({train_end}~{val_end}), Test({val_end}~{total_len})")
            logger.info(f"📊 평가 구간(Test Set): {test_start_idx} ~ {total_len} ({total_len - test_start_idx}개)")
            
            return test_start_idx, total_len
            
        except Exception as e:
            logger.error(f"❌ 테스트 데이터 준비 실패: {e}", exc_info=True)
            raise
    
    def evaluate(self, initial_capital=10000, max_steps=None, verbose=True):
        """
        모델 평가 실행
        
        Args:
            initial_capital: 초기 자본금
            max_steps: 최대 평가 스텝 수 (None이면 전체 테스트 데이터 사용)
            verbose: 상세 로그 출력 여부
        
        Returns:
            dict: 평가 결과 (성능 지표 포함)
        """
        logger.info("=" * 80)
        logger.info("🚀 PPO 모델 평가 시작")
        logger.info("=" * 80)
        
        # 테스트 데이터 준비
        test_start_idx, total_len = self._prepare_test_data()
        
        # 평가 범위 설정
        if max_steps is None:
            max_steps = total_len - test_start_idx
        
        end_idx = min(test_start_idx + max_steps, total_len)
        actual_steps = end_idx - test_start_idx
        
        logger.info(f"📊 평가 범위: 인덱스 {test_start_idx} ~ {end_idx} ({actual_steps} 스텝)")
        
        # 초기화
        capital = initial_capital
        current_position = None  # {'type': 'LONG' or 'SHORT', 'entry_price': float, 'entry_idx': int}
        entry_price = 0.0
        entry_idx = 0
        
        # [개선] 평가 시작 시 LSTM 상태 초기화
        self.agent.reset_episode_states()
        
        self.trades = []
        self.equity_curve = [initial_capital]
        self.actions_taken = {'HOLD': 0, 'LONG': 0, 'SHORT': 0}
        
        total_pnl = 0.0
        winning_trades = 0
        losing_trades = 0
        
        # 평가 루프
        for step in range(actual_steps):
            current_idx = test_start_idx + step
            
            # 현재 인덱스 설정
            self.data_collector.current_index = current_idx
            
            # 현재 가격 가져오기
            if current_idx >= len(self.data_collector.eth_data):
                break
            
            current_candle = self.data_collector.eth_data.iloc[current_idx]
            current_price = current_candle['close']
            
            # 포지션 정보 생성
            if current_position is None:
                position_info = [0.0, 0.0, 0.0]  # [포지션, 미실현PnL, 보유시간]
            else:
                # 미실현 PnL 계산
                if current_position['type'] == 'LONG':
                    unrealized_pnl = (current_price - entry_price) / entry_price
                else:  # SHORT
                    unrealized_pnl = (entry_price - current_price) / entry_price
                
                # 보유 시간 정규화 (0~1)
                holding_time = (step - entry_idx) / max(actual_steps, 1)
                holding_time = min(holding_time, 1.0)
                
                position_val = 1.0 if current_position['type'] == 'LONG' else -1.0
                position_info = [position_val, unrealized_pnl, holding_time]
            
            # 관측 생성
            obs = self.env.get_observation(position_info)
            if obs is None:
                logger.warning(f"⚠️ 관측 생성 실패 (인덱스 {current_idx})")
                continue
            
            obs_seq, obs_info = obs
            
            # 행동 선택 (평가 모드: 탐험 없이 결정론적)
            with torch.no_grad():
                action_probs, value = self.agent.model(obs_seq.to(self.device), info=obs_info.to(self.device))
                action = torch.argmax(action_probs, dim=-1).item()
            
            action_names = ['HOLD', 'LONG', 'SHORT']
            action_name = action_names[action]
            self.actions_taken[action_name] += 1
            
            # 거래 실행
            trade_done = False
            pnl = 0.0
            
            if action == 1:  # LONG
                if current_position is None:
                    # 진입
                    current_position = {'type': 'LONG', 'entry_price': current_price, 'entry_idx': step}
                    entry_price = current_price
                    entry_idx = step
                    if verbose:
                        logger.info(f"📈 LONG 진입 | 가격: ${current_price:.2f} | 인덱스: {current_idx}")
                elif current_position['type'] == 'SHORT':
                    # 반대 포지션 전환: SHORT 청산 후 LONG 진입
                    # SHORT 청산
                    exit_pnl = (entry_price - current_price) / entry_price
                    pnl = exit_pnl
                    trade_done = True
                    total_pnl += pnl
                    
                    if pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                    
                    self.trades.append({
                        'entry_idx': current_position['entry_idx'],
                        'exit_idx': step,
                        'type': 'SHORT',
                        'entry_price': current_position['entry_price'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl * 100
                    })
                    
                    if verbose:
                        logger.info(f"📉 SHORT 청산 | 진입: ${current_position['entry_price']:.2f} | 청산: ${current_price:.2f} | PnL: {pnl*100:.2f}%")
                    
                    # LONG 진입
                    current_position = {'type': 'LONG', 'entry_price': current_price, 'entry_idx': step}
                    entry_price = current_price
                    entry_idx = step
                    if verbose:
                        logger.info(f"📈 LONG 진입 | 가격: ${current_price:.2f} | 인덱스: {current_idx}")
            
            elif action == 2:  # SHORT
                if current_position is None:
                    # 진입
                    current_position = {'type': 'SHORT', 'entry_price': current_price, 'entry_idx': step}
                    entry_price = current_price
                    entry_idx = step
                    if verbose:
                        logger.info(f"📉 SHORT 진입 | 가격: ${current_price:.2f} | 인덱스: {current_idx}")
                elif current_position['type'] == 'LONG':
                    # 반대 포지션 전환: LONG 청산 후 SHORT 진입
                    # LONG 청산
                    exit_pnl = (current_price - entry_price) / entry_price
                    pnl = exit_pnl
                    trade_done = True
                    total_pnl += pnl
                    
                    if pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                    
                    self.trades.append({
                        'entry_idx': current_position['entry_idx'],
                        'exit_idx': step,
                        'type': 'LONG',
                        'entry_price': current_position['entry_price'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl * 100
                    })
                    
                    if verbose:
                        logger.info(f"📈 LONG 청산 | 진입: ${current_position['entry_price']:.2f} | 청산: ${current_price:.2f} | PnL: {pnl*100:.2f}%")
                    
                    # SHORT 진입
                    current_position = {'type': 'SHORT', 'entry_price': current_price, 'entry_idx': step}
                    entry_price = current_price
                    entry_idx = step
                    if verbose:
                        logger.info(f"📉 SHORT 진입 | 가격: ${current_price:.2f} | 인덱스: {current_idx}")
            
            elif action == 0:  # HOLD
                # 포지션이 있으면 유지, 없으면 대기
                pass
            
            # 자산 곡선 업데이트
            if current_position is not None:
                if current_position['type'] == 'LONG':
                    unrealized_pnl = (current_price - entry_price) / entry_price
                else:
                    unrealized_pnl = (entry_price - current_price) / entry_price
                current_equity = initial_capital * (1 + total_pnl + unrealized_pnl)
            else:
                current_equity = initial_capital * (1 + total_pnl)
            
            self.equity_curve.append(current_equity)
            
            # 진행 상황 출력 (config에서 설정한 간격)
            if (step + 1) % config.EVAL_VERBOSE_INTERVAL == 0:
                logger.info(f"진행: {step + 1}/{actual_steps} | 자산: ${current_equity:.2f} | 거래: {len(self.trades)}회")
        
        # 마지막 포지션 청산
        if current_position is not None:
            final_candle = self.data_collector.eth_data.iloc[min(end_idx - 1, len(self.data_collector.eth_data) - 1)]
            final_price = final_candle['close']
            
            if current_position['type'] == 'LONG':
                exit_pnl = (final_price - entry_price) / entry_price
            else:
                exit_pnl = (entry_price - final_price) / entry_price
            
            pnl = exit_pnl
            total_pnl += pnl
            
            if pnl > 0:
                winning_trades += 1
            else:
                losing_trades += 1
            
            self.trades.append({
                'entry_idx': current_position['entry_idx'],
                'exit_idx': actual_steps - 1,
                'type': current_position['type'],
                'entry_price': current_position['entry_price'],
                'exit_price': final_price,
                'pnl': pnl,
                'pnl_pct': pnl * 100
            })
            
            if verbose:
                logger.info(f"🔚 최종 포지션 청산 | {current_position['type']} | 진입: ${entry_price:.2f} | 청산: ${final_price:.2f} | PnL: {pnl*100:.2f}%")
        
        # 성능 지표 계산
        final_equity = initial_capital * (1 + total_pnl)
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        # 최대 낙폭 계산
        equity_array = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak * 100
        max_drawdown = np.min(drawdown)
        
        # 승률 계산
        total_trades = len(self.trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        # 평균 수익/손실
        if total_trades > 0:
            pnls = [t['pnl'] for t in self.trades]
            avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0.0
            avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0.0
            profit_factor = abs(sum([p for p in pnls if p > 0]) / sum([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else float('inf')
        else:
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
        
        # 샤프 비율 (간단 버전: 수익률 / 변동성)
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 288)  # 연율화 (3분봉 기준)
        
        # 결과 정리
        results = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win * 100,
            'avg_loss': avg_loss * 100,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_pnl': total_pnl * 100,
            'actions_taken': self.actions_taken,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
        
        # 결과 출력
        self._print_results(results)
        
        return results
    
    def _print_results(self, results):
        """평가 결과 출력"""
        logger.info("=" * 80)
        logger.info("📊 평가 결과")
        logger.info("=" * 80)
        logger.info(f"💰 초기 자본금: ${results['initial_capital']:,.2f}")
        logger.info(f"💰 최종 자산: ${results['final_equity']:,.2f}")
        logger.info(f"📈 총 수익률: {results['total_return']:.2f}%")
        logger.info(f"📉 최대 낙폭: {results['max_drawdown']:.2f}%")
        logger.info(f"📊 샤프 비율: {results['sharpe_ratio']:.2f}")
        logger.info("")
        logger.info(f"🔄 총 거래 횟수: {results['total_trades']}회")
        logger.info(f"✅ 승리 거래: {results['winning_trades']}회")
        logger.info(f"❌ 손실 거래: {results['losing_trades']}회")
        logger.info(f"🎯 승률: {results['win_rate']:.2f}%")
        logger.info(f"📊 평균 수익: {results['avg_win']:.2f}%")
        logger.info(f"📊 평균 손실: {results['avg_loss']:.2f}%")
        logger.info(f"💎 수익 팩터: {results['profit_factor']:.2f}")
        logger.info("")
        logger.info("🎲 행동 분포:")
        for action, count in results['actions_taken'].items():
            logger.info(f"   {action}: {count}회 ({count/sum(results['actions_taken'].values())*100:.1f}%)")
        logger.info("=" * 80)
        
        # 상위/하위 거래 출력
        if len(results['trades']) > 0:
            pnls = [(i, t['pnl_pct']) for i, t in enumerate(results['trades'])]
            pnls.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("🏆 상위 5개 거래:")
            for i, (idx, pnl) in enumerate(pnls[:5]):
                trade = results['trades'][idx]
                logger.info(f"   {i+1}. {trade['type']} | 진입: ${trade['entry_price']:.2f} | 청산: ${trade['exit_price']:.2f} | PnL: {pnl:.2f}%")
            
            logger.info("")
            logger.info("📉 하위 5개 거래:")
            for i, (idx, pnl) in enumerate(pnls[-5:]):
                trade = results['trades'][idx]
                logger.info(f"   {i+1}. {trade['type']} | 진입: ${trade['entry_price']:.2f} | 청산: ${trade['exit_price']:.2f} | PnL: {pnl:.2f}%")


def main():
    """메인 함수"""
    try:
        # 평가기 초기화
        evaluator = PPOModelEvaluator(test_split=0.2)
        
        # 평가 실행
        results = evaluator.evaluate(
            initial_capital=10000,
            max_steps=None,  # 전체 테스트 데이터 사용
            verbose=True
        )
        
        logger.info("✅ 평가 완료!")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 평가 중 오류 발생: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    main()
