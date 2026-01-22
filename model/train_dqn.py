"""
DDQN 학습 스크립트 (Final Optimized)
전략 지표 10개 + PPO 기본 데이터 5개 = 총 15개 피처 사용
메모리 안전 연산(.values) 적용 완료
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import logging
import time

from core.data_collector import DataCollector
from core.indicators import Indicators
from model.dqn_agent import DDQNAgent
from model.dqn_model import DuelingGRU
from model.trading_env import TradingEnvironment
from model.feature_selection import FeatureSelector
from model.mtf_processor import MTFProcessor
from model.feature_engineering import FeatureEngineer
import config

# 전략 파일들 임포트
from strategies import (
    BTCEthCorrelationStrategy, VolatilitySqueezeStrategy, OrderblockFVGStrategy,
    HMAMomentumStrategy, MFIMomentumStrategy, BollingerMeanReversionStrategy,
    VWAPDeviationStrategy, RangeTopBottomStrategy, StochRSIMeanReversionStrategy,
    CMFDivergenceStrategy, CCIReversalStrategy, WilliamsRStrategy
)

# 진행률 표시용 (선택적)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, desc=""):
        return iterable

# 로깅 설정
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_dqn.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 시각화 (선택적)
try:
    import matplotlib.pyplot as plt
    from collections import deque
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("matplotlib이 설치되지 않아 시각화 기능을 사용할 수 없습니다.")


class LiveVisualizer:
    """학습 리워드를 실시간으로 그래프화하는 클래스"""
    def __init__(self, window_size=10, enable=True):
        if not enable or not VISUALIZATION_AVAILABLE:
            self.enabled = False
            return
        
        self.enabled = True
        plt.ion()  # 대화형 모드 활성화
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.rewards = []
        self.moving_avg = []
        self.window_size = window_size
        self.ax.set_title("DDQN Training Performance", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("Episode", fontsize=12)
        self.ax.set_ylabel("Total Reward", fontsize=12)
        self.line1, = self.ax.plot([], [], label='Episode Reward', alpha=0.3, color='blue', linewidth=1)
        self.line2, = self.ax.plot([], [], label=f'Moving Avg ({window_size})', color='red', linewidth=2)
        self.ax.legend(loc='upper left')
        self.ax.grid(True, alpha=0.3)
        plt.tight_layout()
    
    def update(self, reward):
        """보상 업데이트 및 그래프 갱신"""
        if not self.enabled:
            return
        
        try:
            self.rewards.append(reward)
            
            # 이동 평균 계산
            if len(self.rewards) >= self.window_size:
                avg = np.mean(self.rewards[-self.window_size:])
            else:
                avg = np.mean(self.rewards) if self.rewards else 0
            self.moving_avg.append(avg)
            
            # 데이터 업데이트
            x = np.arange(len(self.rewards))
            self.line1.set_data(x, self.rewards)
            self.line2.set_data(x, self.moving_avg)
            
            # 화면 범위 자동 조절
            self.ax.relim()
            self.ax.autoscale_view()
            
            # Y축 범위를 적절하게 설정 (이상치 제외)
            if len(self.rewards) > 0:
                y_min = min(min(self.rewards), min(self.moving_avg))
                y_max = max(max(self.rewards), max(self.moving_avg))
                margin = (y_max - y_min) * 0.1
                self.ax.set_ylim(y_min - margin, y_max + margin)
            
            plt.draw()
            plt.pause(0.01)  # 짧은 휴식으로 그래프 갱신 보장
            
        except Exception as e:
            logger.debug(f"시각화 업데이트 실패: {e}")
    
    def close(self):
        """그래프 창 닫기"""
        if self.enabled:
            plt.close(self.fig)


def calculate_technical_features(data):
    """
    기술적 지표 15개 계산 (기존 함수)
    안전한 Numpy 연산으로 메모리 오류 방지
    """
    try:
        # 1. 데이터 추출 (Numpy Array)
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        open_val = data['open'].values
        volume = data['volume'].values
        
        # DataFrame 준비
        df = pd.DataFrame(index=data.index)
        
        # --- Group A: PPO 모델 기본 데이터 (5개) ---
        
        # 1. Log Return (로그 수익률)
        df['log_return'] = np.concatenate([[0], np.diff(np.log(close + 1e-8))])
        
        # 2. Log Volume (거래량 로그)
        df['log_volume'] = np.log1p(volume)
        
        # 3. High-Close Ratio (윗꼬리)
        df['high_ratio'] = (high - close) / (close + 1e-8)
        
        # 4. Low-Close Ratio (아랫꼬리)
        df['low_ratio'] = (low - close) / (close + 1e-8)
        
        # 5. Taker Buy Ratio (없으면 Trades로 대체)
        if 'taker_buy_base' in data.columns:
            df['taker_ratio'] = data['taker_buy_base'].values / (volume + 1e-8)
        elif 'taker_buy_base_volume' in data.columns:
            df['taker_ratio'] = data['taker_buy_base_volume'].values / (volume + 1e-8)
        else:
            df['taker_ratio'] = np.log1p(data['trades'].values) if 'trades' in data.columns else np.zeros_like(close)

        # --- Group B: 전략 파일 기반 핵심 지표 (10개) ---
        
        # 6. RSI (14)
        rsi = Indicators.calculate_rsi(data, period=14)
        if rsi is not None:
            if isinstance(rsi, pd.DataFrame):
                df['rsi'] = rsi.iloc[:, 0].values
            else:
                df['rsi'] = rsi.values
        else:
            df['rsi'] = np.zeros_like(close)

        # 7. MACD Histogram
        macd = Indicators.calculate_macd(data)
        if macd is not None and 'histogram' in macd:
            df['macd_hist'] = macd['histogram'].values
        else:
            df['macd_hist'] = np.zeros_like(close)

        # 8, 9. Bollinger Bands (Width, Position)
        bb = Indicators.calculate_bollinger_bands(data, period=20)
        if bb is not None:
            u = bb['upper'].values if isinstance(bb['upper'], pd.Series) else bb['upper']
            l = bb['lower'].values if isinstance(bb['lower'], pd.Series) else bb['lower']
            m = bb['middle'].values if isinstance(bb['middle'], pd.Series) else bb['middle']
            df['bb_width'] = (u - l) / (m + 1e-8)
            df['bb_position'] = (close - l) / (u - l + 1e-8)
        else:
            df['bb_width'] = np.zeros_like(close)
            df['bb_position'] = np.zeros_like(close)
        
        # 10. Stoch RSI K
        stoch = Indicators.calculate_stoch_rsi(data)
        if stoch is not None and 'k' in stoch:
            df['stoch_rsi'] = stoch['k'].values
        else:
            df['stoch_rsi'] = np.zeros_like(close)
        
        # 11. MFI (자금 흐름) - Indicators에 없으면 계산
        try:
            # MFI는 Typical Price와 Money Flow 기반
            tp = (high + low + close) / 3
            money_flow = tp * volume
            positive_mf = np.where(tp > np.roll(tp, 1), money_flow, 0)
            negative_mf = np.where(tp < np.roll(tp, 1), money_flow, 0)
            positive_mf[0] = 0
            negative_mf[0] = 0
            
            # 14기간 롤링 합
            period = 14
            pos_sum = pd.Series(positive_mf).rolling(period).sum().values
            neg_sum = pd.Series(negative_mf).rolling(period).sum().values
            money_ratio = pos_sum / (neg_sum + 1e-8)
            df['mfi'] = 100 - (100 / (1 + money_ratio))
        except:
            df['mfi'] = np.zeros_like(close)
        
        # 12. CMF (매집/분산) - Indicators에 없으면 계산
        try:
            # CMF = ((Close - Low) - (High - Close)) / (High - Low) * Volume
            mf_mult = ((close - low) - (high - close)) / ((high - low) + 1e-8)
            mf_vol = mf_mult * volume
            period = 20
            cmf = pd.Series(mf_vol).rolling(period).sum().values / (pd.Series(volume).rolling(period).sum().values + 1e-8)
            df['cmf'] = cmf
        except:
            df['cmf'] = np.zeros_like(close)
        
        # 13. HMA Ratio (괴리율)
        hma = Indicators.calculate_hma(data, period=14)
        if hma is not None:
            hma_val = hma.iloc[:, 0].values if isinstance(hma, pd.DataFrame) else hma.values
            df['hma_ratio'] = (close - hma_val) / (hma_val + 1e-8)
        else:
            df['hma_ratio'] = np.zeros_like(close)
            
        # 14. VWAP Deviation (이격도)
        vwap = Indicators.calculate_vwap(data)
        if vwap is not None:
            vwap_val = vwap.iloc[:, 0].values if isinstance(vwap, pd.DataFrame) else vwap.values
            df['vwap_dist'] = (close - vwap_val) / (vwap_val + 1e-8)
        else:
            df['vwap_dist'] = np.zeros_like(close)
            
        # 15. ATR Ratio (변동성 비율)
        atr = Indicators.calculate_atr(data, period=14)
        if atr is not None:
            atr_val = atr.iloc[:, 0].values if isinstance(atr, pd.DataFrame) else atr.values
            df['atr_ratio'] = atr_val / (close + 1e-8)
        else:
            df['atr_ratio'] = np.zeros_like(close)
        
        # 16. ADX (추세 강도 지표) - 시장의 성격을 규정하는 핵심 지표
        try:
            # TR (True Range) 계산
            tr1 = np.abs(high - low)
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # DM (Directional Movement) 계산
            up_move = high - np.roll(high, 1)
            down_move = np.roll(low, 1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            
            # Smoothing (14 period)
            alpha = 1/14
            
            # Pandas Series로 변환하여 ewm 사용 (구현 편의성)
            tr_s = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
            plus_dm_s = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
            minus_dm_s = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
            
            plus_di = 100 * (plus_dm_s / (tr_s + 1e-8))
            minus_di = 100 * (minus_dm_s / (tr_s + 1e-8))
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
            adx = dx.ewm(alpha=alpha, adjust=False).mean().values
            
            # 데이터프레임에 추가
            df['adx'] = np.nan_to_num(adx, nan=0.0, posinf=0.0, neginf=0.0)
            
            # [중요] Choppiness Index (횡보장 판별기)도 추가
            # 0에 가까우면 추세, 100에 가까우면 횡보
            high_14 = pd.Series(high).rolling(14).max()
            low_14 = pd.Series(low).rolling(14).min()
            atr_14 = pd.Series(tr).rolling(14).sum()
            chop = 100 * np.log10(atr_14 / (high_14 - low_14 + 1e-8)) / np.log10(14)
            df['chop'] = np.nan_to_num(chop.values, nan=50.0, posinf=50.0, neginf=50.0)  # NaN은 중간값으로 대체
            
        except Exception as e:
            logger.error(f"ADX/Chop 계산 실패: {e}")
            df['adx'] = np.zeros_like(close)
            df['chop'] = np.full_like(close, 50.0)

        # NaN/Inf 처리 (0으로 채움)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # config에 정의된 기술적 피처만 반환
        final_cols = [c for c in config.TECHNICAL_FEATURES if c in df.columns]
        if len(final_cols) != len(config.TECHNICAL_FEATURES):
            missing = set(config.TECHNICAL_FEATURES) - set(final_cols)
            logger.warning(f"누락된 기술적 피처: {missing}")
        
        return df[final_cols] if final_cols else df

    except Exception as e:
        logger.error(f"기술적 지표 계산 중 오류 발생: {e}", exc_info=True)
        return None


def precalculate_strategy_scores(collector, force_recalculate=False):
    """
    모든 전략을 전체 데이터에 대해 미리 실행하여 점수(Score)를 계산
    Long 신호: +Confidence, Short 신호: -Confidence, None: 0
    
    Args:
        collector: DataCollector 인스턴스
        force_recalculate: True면 저장된 파일을 무시하고 다시 계산
    """
    strategy_scores_path = 'data/strategy_scores.csv'
    
    # 저장된 파일이 있고 재계산이 아니면 로드
    if not force_recalculate and os.path.exists(strategy_scores_path):
        try:
            logger.info(f"📂 저장된 전략 점수 로드 중: {strategy_scores_path}")
            scores_df = pd.read_csv(strategy_scores_path, index_col=0, parse_dates=True)
            
            # 필수 전략 컬럼 정의 (12개)
            required_strategy_cols = [
                'strat_btc_eth_corr', 'strat_vol_squeeze', 'strat_ob_fvg',
                'strat_hma', 'strat_mfi', 'strat_bb_reversion',
                'strat_vwap', 'strat_range', 'strat_stoch', 'strat_cmf',
                'strat_cci_reversal', 'strat_williams_r'  # [신규] 2개
            ]
            
            # 데이터 길이 확인
            if len(scores_df) != len(collector.eth_data):
                logger.warning(f"저장된 파일 길이 불일치 ({len(scores_df)} vs {len(collector.eth_data)}), 재계산합니다.")
            # 필수 컬럼 확인 (새로 추가된 전략 2개 포함)
            elif not all(col in scores_df.columns for col in required_strategy_cols):
                missing_cols = [col for col in required_strategy_cols if col not in scores_df.columns]
                logger.warning(f"저장된 파일에 필수 컬럼 누락: {missing_cols}, 재계산합니다.")
            else:
                logger.info(f"✅ 저장된 전략 점수 로드 완료: {len(scores_df)}개 캔들, {len(scores_df.columns)}개 전략")
                return scores_df
        except Exception as e:
            logger.warning(f"저장된 파일 로드 실패: {e}, 재계산합니다.")
    
    # 전략 점수 계산
    logger.info("🧠 내 전략들의 신호 미리 계산 중 (시간이 조금 걸릴 수 있습니다)...")
    
    # 전략 초기화 (12개)
    strategies = [
        (BTCEthCorrelationStrategy(), 'strat_btc_eth_corr'),
        (VolatilitySqueezeStrategy(), 'strat_vol_squeeze'),
        (OrderblockFVGStrategy(), 'strat_ob_fvg'),
        (HMAMomentumStrategy(), 'strat_hma'),
        (MFIMomentumStrategy(), 'strat_mfi'),
        (BollingerMeanReversionStrategy(), 'strat_bb_reversion'),
        (VWAPDeviationStrategy(), 'strat_vwap'),
        (RangeTopBottomStrategy(), 'strat_range'),
        (StochRSIMeanReversionStrategy(), 'strat_stoch'),
        (CMFDivergenceStrategy(), 'strat_cmf'),
        (CCIReversalStrategy(), 'strat_cci_reversal'),  # [신규] CCI 반전 전략
        (WilliamsRStrategy(), 'strat_williams_r')      # [신규] Williams %R 전략
    ]
    
    # 결과를 담을 DataFrame 생성 (0으로 초기화)
    total_len = len(collector.eth_data)
    scores_df = pd.DataFrame(0.0, index=collector.eth_data.index, columns=[s[1] for s in strategies])
    
    # 효율성을 위해 인덱스 루프를 돌며 시뮬레이션
    # 전략의 analyze는 '현재 시점'을 기준으로 과거를 봄.
    # 따라서 과거부터 미래로 순회하며 collector의 index를 변경해줘야 함.
    
    # 기술적 지표 계산 시 필요한 최소 데이터:
    # - RSI(14): 최소 15개
    # - MACD(12,26,9): 최소 26+9=35개
    # - Bollinger(20): 최소 20개
    # - Stochastic RSI: 최소 14+14+3=31개
    # - HMA(14): 최소 14*2=28개
    # - VWAP: 세션 기준이므로 1개부터 가능하지만 안정성을 위해 20개
    # - ATR(14): 최소 15개
    # 가장 큰 값인 MACD 기준으로 여유분 포함: 100개
    start_idx = 100  # 기술적 지표 계산용 여유분 (MACD 등 최대 기간 고려)
    
    # 진행률 표시와 함께 루프 실행
    for i in tqdm(range(start_idx, total_len), desc="전략 신호 계산"):
        collector.current_index = i
        
        for strategy, col_name in strategies:
            try:
                # 전략 실행
                result = strategy.analyze(collector)
                
                if result:
                    # 신호 파싱
                    signal = result.get('signal')
                    confidence = float(result.get('confidence', 0.5))
                    
                    # 점수 변환 (Long: +, Short: -)
                    if signal == 'LONG':
                        scores_df.iloc[i, scores_df.columns.get_loc(col_name)] = confidence
                    elif signal == 'SHORT':
                        scores_df.iloc[i, scores_df.columns.get_loc(col_name)] = -confidence
            except Exception as e:
                logger.debug(f"전략 {col_name} 실행 실패 (인덱스 {i}): {e}")
                pass  # 에러 나도 진행
                
    logger.info(f"✅ 전략 신호 계산 완료: {len(scores_df)}개 캔들")
    
    # 파일로 저장
    try:
        scores_df.to_csv(strategy_scores_path)
        logger.info(f"💾 전략 점수 저장 완료: {strategy_scores_path}")
    except Exception as e:
        logger.warning(f"전략 점수 저장 실패: {e}")
    
    return scores_df


class DDQNTrainer:
    def __init__(self, force_recalculate_strategies=False):
        """
        Args:
            force_recalculate_strategies: True면 저장된 전략 점수를 무시하고 재계산
        """
        # 1. 데이터 로드
        self.data_collector = DataCollector(use_saved_data=True)
        if not self.data_collector.load_saved_data():
            raise ValueError("데이터 로드 실패: collect_training_data.py를 먼저 실행하세요.")
        
        # -----------------------------------------------------
        # [수정] FeatureEngineer를 사용하여 고급 피처 25개 생성
        # -----------------------------------------------------
        logger.info("🛠️ 고급 피처 엔지니어링 수행 중...")
        
        # BTC 데이터 확보 (DataCollector에 btc_data가 있다고 가정)
        btc_df = getattr(self.data_collector, 'btc_data', None)
        
        engineer = FeatureEngineer(self.data_collector.eth_data, btc_df)
        enhanced_df = engineer.generate_features()
        
        if enhanced_df is None:
            raise ValueError("피처 생성 실패")
            
        # 생성된 데이터로 교체
        self.data_collector.eth_data = enhanced_df
        
        # 1.5. MTF 프로세서 적용 (15분봉, 1시간봉 지표 추가)
        # [수정됨] 인덱스가 DatetimeIndex인지 확인하고, 아니면 'timestamp' 컬럼으로 변환
        if not isinstance(self.data_collector.eth_data.index, pd.DatetimeIndex):
            if 'timestamp' in self.data_collector.eth_data.columns:
                # 'timestamp' 컬럼(밀리초)을 사용하여 인덱스 설정 (가장 정확)
                self.data_collector.eth_data.index = pd.to_datetime(self.data_collector.eth_data['timestamp'], unit='ms')
            else:
                try:
                    # timestamp 컬럼이 없으면 기존 인덱스 변환 시도
                    self.data_collector.eth_data.index = pd.to_datetime(self.data_collector.eth_data.index)
                except:
                    logger.warning("인덱스를 DatetimeIndex로 변환할 수 없습니다. MTF 프로세서를 건너뜁니다.")
        
        # 인덱스 변환 후 MTF 적용
        if isinstance(self.data_collector.eth_data.index, pd.DatetimeIndex):
            try:
                mtf_processor = MTFProcessor(self.data_collector.eth_data)
                self.data_collector.eth_data = mtf_processor.add_mtf_features()
                
                # [확인] 데이터가 잘 들어갔는지 통계 출력 (head 말고 describe 확인)
                mtf_check = ['rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h']
                existing_mtf = [c for c in mtf_check if c in self.data_collector.eth_data.columns]
                
                if existing_mtf:
                    logger.info(f"📊 MTF 데이터 통계 (0이 아닌지 확인):\n{self.data_collector.eth_data[existing_mtf].describe().loc[['mean', 'max', 'min']]}")
            except Exception as e:
                logger.warning(f"MTF 프로세서 적용 실패: {e}. 계속 진행합니다.")
        
        # 3. 전략 점수 피처 계산 (10개)
        logger.info("2. 전략 신호 계산 중...")
        strat_df = precalculate_strategy_scores(self.data_collector, force_recalculate=force_recalculate_strategies)
        
        # 인덱스 일치 확인 및 병합
        if len(strat_df) != len(self.data_collector.eth_data):
            raise ValueError(f"전략 점수 길이 불일치: 원본={len(self.data_collector.eth_data)}, 전략={len(strat_df)}")
        
        if not strat_df.index.equals(self.data_collector.eth_data.index):
            logger.warning("전략 점수 인덱스 불일치, 재인덱싱합니다.")
            strat_df.index = self.data_collector.eth_data.index
        
        # 전략 점수 병합
        for col in strat_df.columns:
            self.data_collector.eth_data[col] = strat_df[col]
        
        # 피처 컬럼 정의 (25개 기본 피처)
        initial_features = [
            'log_return', 'roll_return_6', 'atr_ratio', 'bb_width', 'bb_pos',
            'rsi', 'macd_hist', 'hma_ratio', 'cci',
            'rvol', 'taker_ratio', 'cvd_change', 'mfi', 'cmf', 'vwap_dist',
            'wick_upper', 'wick_lower', 'range_pos', 'swing_break', 'chop',
            'btc_return', 'btc_rsi', 'btc_corr', 'btc_vol', 'eth_btc_ratio'
        ]
        
        # 전략 점수 피처 추가
        strat_cols = [c for c in self.data_collector.eth_data.columns if c.startswith('strat_')]
        initial_features.extend(strat_cols)
        
        # [추가] MTF 피처 자동 감지 및 추가 (rsi_15m, trend_15m, rsi_1h, trend_1h)
        mtf_features = ['rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h']
        for mtf_feat in mtf_features:
            if mtf_feat in self.data_collector.eth_data.columns and mtf_feat not in initial_features:
                initial_features.append(mtf_feat)
                logger.info(f"✅ MTF 피처 자동 추가: {mtf_feat}")
        
        # 누락된 컬럼 0으로 채우기 (XGBoost 에러 방지)
        for col in initial_features:
            if col not in self.data_collector.eth_data.columns:
                logger.warning(f"누락된 피처 {col}를 0으로 채웁니다.")
                self.data_collector.eth_data[col] = 0.0
        
        # [수정 후 코드: XGBoost 적용] -----------------------------------------
        # MTF 피처 확인 로깅
        mtf_features_in_data = [f for f in ['rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h'] 
                                if f in self.data_collector.eth_data.columns]
        if mtf_features_in_data:
            logger.info(f"📊 MTF 피처 확인: {mtf_features_in_data} (총 {len(mtf_features_in_data)}개)")
            logger.info(f"📊 MTF 피처 샘플 값: {self.data_collector.eth_data[mtf_features_in_data[0]].head(5).tolist()}")
        else:
            logger.warning("⚠️ MTF 피처가 데이터에 없습니다!")
        
        if config.USE_XGBOOST_SELECTION:
            logger.info("🤖 XGBoost 피처 선택 프로세스 가동...")
            logger.info(f"📋 후보 피처 개수: {len(initial_features)}개 (MTF 포함 여부 확인)")
            
            selector = FeatureSelector(top_k=config.TOP_K_FEATURES)
            
            # 미래 20봉(1시간) 뒤의 변동성을 가장 잘 설명하는 피처 선정
            selected_features = selector.select_features(
                self.data_collector.eth_data, 
                initial_features, 
                target_horizon=10 
            )
            
            # MTF 피처 선택 여부 확인
            selected_mtf = [f for f in selected_features if f in mtf_features_in_data]
            if selected_mtf:
                logger.info(f"✅ XGBoost가 선택한 MTF 피처: {selected_mtf}")
            else:
                logger.info(f"ℹ️ XGBoost가 MTF 피처를 선택하지 않았습니다. (선택된 피처: {selected_features})")
            
            # [안전장치] 만약 선택된 피처가 너무 적으면 기본값 사용
            if len(selected_features) < 3:
                logger.warning("XGBoost가 선택한 피처가 너무 적습니다. 기본 설정으로 복귀합니다.")
                self.feature_columns = initial_features
            else:
                self.feature_columns = selected_features
        else:
            self.feature_columns = initial_features
        
        # [핵심] 방향성 필수 지표 강제 포함 (Whitelist)
        # RSI(과매수/과매도), MACD(추세), BB Position(현재 위치), Choppiness(횡보/추세 판별)
        # FeatureEngineer가 생성한 피처 이름 사용: bb_pos (bb_position 아님)
        must_include = ['rsi', 'macd_hist', 'bb_pos', 'chop']
        
        # MTF 피처도 강제 포함 (상위 프레임 정보는 중요)
        mtf_must_include = [f for f in ['rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h'] 
                           if f in self.data_collector.eth_data.columns]
        must_include.extend(mtf_must_include)
        
        # 필수 지표가 데이터에 있는지 확인 후 추가
        for f in must_include:
            if f in self.data_collector.eth_data.columns and f not in self.feature_columns:
                self.feature_columns.append(f)
                logger.info(f"✅ 필수 지표 강제 포함: {f}")
        # ---------------------------------------------------------------------
        
        # ---------------------------------------------------------------------
        # [추가] 확정된 피처 리스트를 JSON 파일로 저장
        # ---------------------------------------------------------------------
        import json
        save_path = 'saved_models/selected_features.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            with open(save_path, 'w') as f:
                json.dump(self.feature_columns, f, indent=2)
            logger.info(f"💾 피처 목록 저장 완료: {save_path} (총 {len(self.feature_columns)}개)")
        except Exception as e:
            logger.error(f"피처 목록 저장 실패: {e}")
        # ---------------------------------------------------------------------
                
        logger.info(f"✅ 최종 입력 피처 ({len(self.feature_columns)}개): {self.feature_columns}")
        
        # 3. 환경 설정 (feature_columns 전달)
        self.env = TradingEnvironment(
            self.data_collector, 
            strategies=[], 
            lookback=config.LOOKBACK_WINDOW,  # [수정] 20 -> 60 (3시간의 흐름을 보게 함)
            selected_features=self.feature_columns
        )
        
        # 4. 전역 스케일러 학습
        self._fit_global_scaler()
        
        # 5. 에이전트 설정
        ddqn_config = config.DDQN_CONFIG.copy()
        ddqn_config['input_dim'] = len(self.feature_columns)  # 차원 동기화
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"학습 장치: {device}")
        
        self.agent = DDQNAgent(
            input_dim=ddqn_config['input_dim'],
            hidden_dim=ddqn_config['hidden_dim'],
            num_layers=ddqn_config['num_layers'],
            action_dim=ddqn_config['action_dim'],
            lr=ddqn_config['learning_rate'],
            gamma=ddqn_config['gamma'],
            epsilon_start=ddqn_config['epsilon_start'],
            epsilon_end=ddqn_config['epsilon_end'],
            epsilon_decay=ddqn_config['epsilon_decay'],
            buffer_size=ddqn_config['buffer_size'],
            batch_size=ddqn_config['batch_size'],
            target_update=ddqn_config['target_update'],
            device=device,
            use_per=config.USE_PER,  # PER 사용 여부
            n_step=config.N_STEP,  # N-step Learning 스텝 수
            info_dim=3  # 포지션 정보 차원
        )
        
        self.episode_rewards = []
        self.total_steps = 0
        self.current_position = None
        self.entry_price = None
        self.entry_index = None
        self.env.entry_index = None  # 환경의 entry_index도 초기화
        self.prev_pnl = 0.0
        
        # 시각화 초기화 (기본적으로 비활성화)
        self.visualizer = None
    
    def _fit_global_scaler(self):
        """전역 스케일러 학습 (기술적 지표만 학습!)
        
        [수정] 차원 불일치 및 데이터 누수 방지:
        1. selected_features 순서 보장
        2. 학습 구간만 사용 (전체 데이터 사용 금지)
        3. 기술적 피처만 선택하여 fit
        """
        try:
            logger.info("전역 스케일러 학습 시작 (전략 점수 제외)...")
            
            # [핵심 수정] selected_features 순서대로 기술적 지표만 필터링
            # self.feature_columns는 이미 확정된 피처 리스트 (순서 보장)
            tech_cols = [f for f in self.feature_columns if not f.startswith('strat_')]
            
            # MTF 피처 확인
            mtf_in_scaler = [f for f in tech_cols if '_15m' in f or '_1h' in f]
            if mtf_in_scaler:
                logger.info(f"✅ 스케일러에 포함된 MTF 피처: {mtf_in_scaler}")
            else:
                logger.warning(f"⚠️ 스케일러에 MTF 피처가 없습니다. (기술적 지표: {tech_cols})")
            
            if not tech_cols:
                logger.warning("기술적 지표가 없어 스케일러 학습을 건너뜁니다.")
                return
            
            # [데이터 누수 방지] 학습 구간만 사용 (전체 데이터의 80%만 사용)
            start_idx = 100  # 초기 NaN/0 데이터 배제
            data_len = len(self.data_collector.eth_data)
            train_split_idx = int(data_len * 0.8)  # 80%를 학습 구간으로 사용
            
            if data_len <= start_idx:
                logger.warning("데이터가 너무 적어 스케일러를 학습할 수 없습니다.")
                return
            
            # [수정] 학습 구간만 사용 (데이터 누수 방지)
            train_data = self.data_collector.eth_data.iloc[start_idx:train_split_idx]
            
            # 기술적 지표 데이터만 선택 (순서 보장)
            tech_df = train_data[tech_cols].copy()
            
            # 누락된 컬럼 확인
            missing_cols = [col for col in tech_cols if col not in tech_df.columns]
            if missing_cols:
                logger.warning(f"누락된 피처를 0으로 채웁니다: {missing_cols}")
                for col in missing_cols:
                    tech_df[col] = 0.0
            
            # 샘플링 (너무 많으면 5만개만)
            if len(tech_df) > 50000:
                tech_df = tech_df.sample(n=50000, random_state=42)
            
            # Numpy 변환 및 0/Inf 처리
            tech_data = tech_df.values.astype(np.float32)
            tech_data = np.nan_to_num(tech_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # [핵심] 스케일러 학습 (tech_cols 순서대로)
            self.env.preprocessor.fit(tech_data)
            self.env.scaler_fitted = True
            
            # 스케일러에 사용된 피처 순서 저장 (trading_env.py에서 동일 순서 사용 보장)
            self.env.scaler_feature_order = tech_cols
            
            # 학습 완료된 스케일러를 파일로 저장 (피처 이름 포함)
            self.env.preprocessor.save_scaler('saved_models/scaler.pkl', feature_names=tech_cols)
            
            logger.info(f"✅ 스케일러 학습 및 저장 완료:")
            logger.info(f"   - 학습 구간: Index {start_idx} ~ {train_split_idx} (전체 {data_len}개 중 {train_split_idx - start_idx}개 사용)")
            logger.info(f"   - 샘플 수: {len(tech_data)}개")
            logger.info(f"   - 피처 수: {len(tech_cols)}개 기술적 피처 (순서 보장)")
            logger.info(f"   - 전략 점수: {len(self.feature_columns) - len(tech_cols)}개 제외")
            
        except Exception as e:
            logger.error(f"스케일러 학습 실패: {e}", exc_info=True)

    def train_episode(self, episode_num, max_steps=1000):
        """한 에피소드 학습"""
        # [수정] 에피소드 시작 시 한 번만 노이즈 리셋 (성능 최적화)
        if hasattr(self.agent, 'reset_noise'):
            self.agent.reset_noise()
        
        episode_reward = 0.0
        steps = 0
        
        # 랜덤 시작 (과적합 방지)
        self.data_collector.reset_index(max_steps=max_steps, random_start=True)
        
        # 상태 초기화
        self.current_position = None
        self.entry_price = None
        self.entry_index = None
        self.env.entry_index = None  # 환경의 entry_index도 초기화
        self.prev_pnl = 0.0
        
        available_steps = len(self.data_collector.eth_data) - self.data_collector.current_index
        actual_steps = min(max_steps, available_steps)
        
        if actual_steps <= 50:
            return None  # 데이터 부족 시 스킵
        
        logger.info(f"에피소드 {episode_num} 시작: {actual_steps}개 스텝 (인덱스: {self.data_collector.current_index}부터)")
        
        for step in range(actual_steps):
            try:
                # [수정] 매 스텝마다 노이즈 리셋 제거 (에피소드 시작 시만)
                
                # 1. 인덱스 증가
                self.data_collector.current_index += 1
                if self.data_collector.current_index >= len(self.data_collector.eth_data):
                    break
                
                # 2. 관측 (15개 피처)
                pos_val = 1.0 if self.current_position == 'LONG' else (-1.0 if self.current_position == 'SHORT' else 0.0)
                pnl_val = self.prev_pnl * 10
                hold_val = 0.0
                if self.entry_index:
                    hold_val = min(1.0, (self.data_collector.current_index - self.entry_index) / 160.0)
                
                pos_info = [pos_val, pnl_val, hold_val]
                state = self.env.get_observation(position_info=pos_info)
                if state is None:
                    continue
                
                # 3. 행동 선택
                action = self.agent.act(state, training=True)
                
                # 4. 가격 확인
                current_price = float(self.data_collector.eth_data.iloc[self.data_collector.current_index - 1]['close'])
                
                # 5. 보상 계산 로직
                reward = 0.0
                trade_done = False
                current_pnl = 0.0
                pnl_change = 0.0
                
                # --- 매매 로직 ---
                if action == 1:  # LONG
                    if self.current_position == 'SHORT':  # 스위칭
                        pnl = (self.entry_price - current_price) / self.entry_price
                        pnl_change = pnl - self.prev_pnl
                        reward = self.env.calculate_reward(pnl, True, 0, pnl_change)
                        trade_done = True
                        self.prev_pnl = 0.0
                        self.env.entry_index = None  # 거래 종료 시 환경의 entry_index 초기화
                    
                    if self.current_position != 'LONG':
                        self.current_position = 'LONG'
                        self.entry_price = current_price
                        self.entry_index = self.data_collector.current_index
                        self.env.entry_index = self.entry_index  # 환경에도 설정
                        self.prev_pnl = 0.0
                
                elif action == 2:  # SHORT
                    if self.current_position == 'LONG':  # 스위칭
                        pnl = (current_price - self.entry_price) / self.entry_price
                        pnl_change = pnl - self.prev_pnl
                        reward = self.env.calculate_reward(pnl, True, 0, pnl_change)
                        trade_done = True
                        self.prev_pnl = 0.0
                        self.env.entry_index = None  # 거래 종료 시 환경의 entry_index 초기화
                    
                    if self.current_position != 'SHORT':
                        self.current_position = 'SHORT'
                        self.entry_price = current_price
                        self.entry_index = self.data_collector.current_index
                        self.env.entry_index = self.entry_index  # 환경에도 설정
                        self.prev_pnl = 0.0
                        
                else:  # HOLD
                    if self.current_position:
                        if self.current_position == 'LONG':
                            current_pnl = (current_price - self.entry_price) / self.entry_price
                        else:
                            current_pnl = (self.entry_price - current_price) / self.entry_price
                        pnl_change = current_pnl - self.prev_pnl
                        holding_time = self.data_collector.current_index - self.entry_index
                        reward = self.env.calculate_reward(current_pnl, False, holding_time, pnl_change)
                        self.prev_pnl = current_pnl

                # 6. 다음 상태
                next_state = None
                if not trade_done and step < actual_steps - 1:
                    # 임시 인덱스 증가
                    self.data_collector.current_index += 1
                    next_state = self.env.get_observation(position_info=pos_info)
                    self.data_collector.current_index -= 1  # 복구
                
                done = (step == actual_steps - 1)
                
                # 7. 저장 및 학습
                self.agent.remember(state, action, reward, next_state, done)
                loss = self.agent.train_step()
                
                episode_reward += reward
                steps += 1
                self.total_steps += 1
                
                if done:
                    break
                
            except Exception as e:
                logger.error(f"Step Error: {e}", exc_info=True)
                continue
                
        return episode_reward, steps

    def train(self, num_episodes=1000, max_steps_per_episode=1000, save_interval=100, enable_visualization=False):
        """학습 메인 루프"""
        logger.info("=" * 60)
        logger.info(f"🚀 DDQN 학습 시작: {num_episodes} 에피소드")
        logger.info(f"피처: {len(self.feature_columns)}개 ({', '.join(self.feature_columns)})")
        logger.info(f"시각화: {'활성화' if enable_visualization and VISUALIZATION_AVAILABLE else '비활성화'}")
        logger.info("=" * 60)
        
        # 시각화 초기화
        if enable_visualization and VISUALIZATION_AVAILABLE:
            self.visualizer = LiveVisualizer(window_size=10, enable=True)
        else:
            self.visualizer = LiveVisualizer(window_size=10, enable=False)
        
        best_reward = float('-inf')
        
        try:
            for episode in range(1, num_episodes + 1):
                result = self.train_episode(episode, max_steps_per_episode)
                if result:
                    rw, st = result
                    self.episode_rewards.append(rw)
                    avg_rw = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else rw
                    logger.info(f"Ep {episode}: Reward {rw:.2f} | Avg {avg_rw:.2f} | Steps {st} | Eps {self.agent.epsilon:.4f} | Buffer {len(self.agent.memory)}")
                    
                    # 시각화 업데이트
                    if self.visualizer:
                        self.visualizer.update(rw)
                    
                    # 최고 성능 모델 저장
                    if rw > best_reward:
                        best_reward = rw
                        # 최고점 모델은 이름을 다르게 저장!
                        best_model_path = 'saved_models/best_ddqn_model.pth'
                        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                        self.agent.save_model(best_model_path)
                        
                        # [추가] 선택된 피처도 함께 저장
                        import json
                        features_path = 'saved_models/selected_features.json'
                        with open(features_path, 'w') as f:
                            json.dump(self.feature_columns, f, indent=2)
                        logger.info(f"🏆 최고 기록 갱신! best_ddqn_model.pth 저장 (Reward: {rw:.2f})")
                        logger.info(f"📋 선택된 피처 저장: {features_path}")
                    
                    # 주기적 저장
                    if episode % save_interval == 0:
                        self.agent.save_model(config.DDQN_MODEL_PATH)
                        logger.info(f"💾 모델 저장 완료 (에피소드 {episode})")
        except KeyboardInterrupt:
            logger.info("학습 중단 요청")
        finally:
            # 시각화 창 닫기
            if self.visualizer:
                self.visualizer.close()
        
        # 최종 모델 저장
        self.agent.save_model(config.DDQN_MODEL_PATH)
        logger.info("=" * 60)
        logger.info("✅ 학습 완료")
        logger.info(f"평균 보상: {np.mean(self.episode_rewards) if self.episode_rewards else 0:.4f}")
        logger.info("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DDQN 모델 학습')
    parser.add_argument('--episodes', type=int, default=1000, help='학습 에피소드 수')
    parser.add_argument('--steps', type=int, default=1000, help='에피소드당 최대 스텝 수')
    parser.add_argument('--save-interval', type=int, default=100, help='모델 저장 간격 (에피소드)')
    parser.add_argument('--visualize', action='store_true', help='보상 그래프 시각화 활성화')
    parser.add_argument('--no-visualize', action='store_true', help='보상 그래프 시각화 비활성화 (기본값)')
    parser.add_argument('--recalculate-strategies', action='store_true', help='저장된 전략 점수를 무시하고 재계산')
    
    args = parser.parse_args()
    
    # 시각화 옵션 결정
    enable_viz = args.visualize and not args.no_visualize
    
    try:
        trainer = DDQNTrainer(force_recalculate_strategies=args.recalculate_strategies)
        trainer.train(
            num_episodes=args.episodes,
            max_steps_per_episode=args.steps,
            save_interval=args.save_interval,
            enable_visualization=enable_viz
        )
    except KeyboardInterrupt:
        logger.info("학습 중단")
    except Exception as e:
        logger.error(f"치명적 오류: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
