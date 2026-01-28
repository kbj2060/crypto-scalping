"""
백테스팅 모듈
1년치 데이터를 사용하여 전략 성과를 시뮬레이션합니다.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from model import config
from core import DataCollector, RiskManager
from core.indicators import Indicators
from trading_bot import TradingBot

# 로깅 설정
os.makedirs('logs', exist_ok=True)

# Windows에서 UTF-8 인코딩 설정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class Backtester:
    def __init__(self, initial_capital=10000):
        """백테스터 초기화"""
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []  # 거래 내역
        self.current_position = None  # 현재 포지션
        self.equity_curve = []  # 자산 곡선
        self.data_collector = DataCollector()
        self.risk_manager = RiskManager()
        
        # 통계
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.max_drawdown = 0
        self.peak_equity = initial_capital
        
        # data 폴더 생성
        os.makedirs('data', exist_ok=True)
        
    def get_data_filepath(self, symbol, interval, start_date, end_date):
        """데이터 파일 경로 생성"""
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        filename = f"{symbol.lower()}_{interval}_{start_str}_{end_str}.csv"
        return os.path.join('data', filename)
    
    def load_data(self, symbol, interval, start_date, end_date):
        """저장된 데이터 로드"""
        filepath = self.get_data_filepath(symbol, interval, start_date, end_date)
        
        if os.path.exists(filepath):
            try:
                logger.info(f"📂 저장된 데이터 로드: {filepath}")
                df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
                logger.info(f"✅ {symbol} 데이터 로드 완료: {len(df)}개 캔들")
                logger.info(f"   기간: {df.index[0]} ~ {df.index[-1]}")
                return df
            except Exception as e:
                logger.warning(f"데이터 로드 실패 ({filepath}): {e}")
                return None
        return None
    
    def save_data(self, df, symbol, interval, start_date, end_date):
        """데이터 저장"""
        filepath = self.get_data_filepath(symbol, interval, start_date, end_date)
        try:
            df.to_csv(filepath, encoding='utf-8-sig')
            logger.info(f"💾 데이터 저장 완료: {filepath}")
        except Exception as e:
            logger.error(f"데이터 저장 실패 ({filepath}): {e}")
    
    def fetch_1year_data(self, symbol, interval='3m', use_cache=True):
        """1년치 데이터 수집 또는 로드 (3분봉 기준 약 175,200개)"""
        # 1년 전 날짜 계산
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # 캐시된 데이터가 있으면 로드
        if use_cache:
            cached_data = self.load_data(symbol, interval, start_date, end_date)
            if cached_data is not None:
                return cached_data
        
        # 데이터 수집
        logger.info(f"📥 {symbol} 1년치 데이터 수집 시작...")
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        logger.info(f"   기간: {start_str} ~ {end_str}")
        
        all_data = []
        limit = 1000  # 바이낸스 API 최대 제한
        
        try:
            # 바이낸스 API는 한 번에 최대 1000개만 가져올 수 있으므로
            # 여러 번 나눠서 요청
            current_start = start_date
            
            while current_start < end_date:
                try:
                    current_end = min(current_start + timedelta(days=7), end_date)  # 7일씩 나눠서
                    current_start_str = current_start.strftime('%Y-%m-%d')
                    current_end_str = current_end.strftime('%Y-%m-%d')
                    
                    # 데이터 수집을 위한 별도 클라이언트 (백테스팅 모드 아님)
                    from core.binance_client import BinanceClient
                    data_client = BinanceClient(backtest_mode=False)
                    
                    if data_client.use_futures:
                        klines = data_client.client.futures_historical_klines(
                            symbol=symbol,
                            interval=interval,
                            start_str=current_start_str,
                            end_str=current_end_str
                        )
                    else:
                        klines = data_client.client.get_historical_klines(
                            symbol=symbol,
                            interval=interval,
                            start_str=current_start_str,
                            end_str=current_end_str
                        )
                    
                    if klines and len(klines) > 0:
                        # 중복 제거를 위해 기존 데이터와 비교
                        if all_data:
                            last_timestamp = all_data[-1][0]
                            klines = [k for k in klines if k[0] > last_timestamp]
                        
                        all_data.extend(klines)
                        logger.info(f"  수집 진행: {len(all_data)}개 캔들 ({current_start_str} ~ {current_end_str})")
                    
                    # 다음 구간으로 이동
                    current_start = current_end
                    
                    # API 제한 방지
                    import time
                    time.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"데이터 수집 중 오류 ({current_start_str}): {e}")
                    # 오류 발생 시 다음 구간으로 이동
                    current_start = current_start + timedelta(days=7)
                    continue
            
            if not all_data:
                logger.error(f"{symbol} 데이터 수집 실패")
                return None
            
            # DataFrame 변환
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # 데이터 타입 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                            'quote_volume', 'taker_buy_base', 'taker_buy_quote']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()  # 시간순 정렬
            
            logger.info(f"✅ {symbol} 데이터 수집 완료: {len(df)}개 캔들")
            logger.info(f"   기간: {df.index[0]} ~ {df.index[-1]}")
            
            # 데이터 저장
            self.save_data(df, symbol, interval, start_date, end_date)
            
            return df
            
        except Exception as e:
            logger.error(f"1년치 데이터 수집 실패 ({symbol}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def simulate_trade(self, signal, entry_price, current_price, timestamp):
        """거래 시뮬레이션"""
        if signal is None:
            return
        
        # 현재 포지션이 있으면 신호가 반대 방향일 때만 청산
        if self.current_position:
            pos_type = self.current_position['type']
            signal_type = signal['signal']
            
            # 반대 신호가 오면 청산 후 진입
            if (pos_type == 'LONG' and signal_type == 'SHORT') or \
               (pos_type == 'SHORT' and signal_type == 'LONG'):
                self.close_position(current_price, timestamp)
            else:
                # 같은 방향 신호면 무시 (이미 포지션 있음)
                return
        
        # 새 포지션 진입
        if signal['signal'] == 'LONG':
            # 포지션 크기 계산 (초기 자본의 10% 사용)
            position_size_usd = self.capital * 0.1
            position_size = position_size_usd / entry_price
            
            # 손절가 계산 (기본 0.2% 또는 신호에서 제공된 값)
            stop_loss = signal.get('stop_loss')
            if stop_loss is None:
                stop_loss = entry_price * (1 - config.STOP_LOSS_PERCENT / 100)
            
            self.current_position = {
                'type': 'LONG',
                'entry_price': entry_price,
                'size': position_size,
                'entry_time': timestamp,
                'stop_loss': stop_loss,
                'take_profit': entry_price * (1 + config.STOP_LOSS_PERCENT * 2 / 100),  # 익절: 손절의 2배
                'entry_capital': self.capital
            }
            # 백테스팅 중에는 진입 로그를 출력하지 않음 (결과 보고서에서 확인)
        
        elif signal['signal'] == 'SHORT':
            # 포지션 크기 계산
            position_size_usd = self.capital * 0.1
            position_size = position_size_usd / entry_price
            
            # 손절가 계산
            stop_loss = signal.get('stop_loss')
            if stop_loss is None:
                stop_loss = entry_price * (1 + config.STOP_LOSS_PERCENT / 100)
            
            self.current_position = {
                'type': 'SHORT',
                'entry_price': entry_price,
                'size': position_size,
                'entry_time': timestamp,
                'stop_loss': stop_loss,
                'take_profit': entry_price * (1 - config.STOP_LOSS_PERCENT * 2 / 100),  # 익절: 손절의 2배
                'entry_capital': self.capital
            }
            # 백테스팅 중에는 진입 로그를 출력하지 않음 (결과 보고서에서 확인)
    
    def close_position(self, exit_price, timestamp):
        """포지션 청산"""
        if not self.current_position:
            return
        
        pos = self.current_position
        entry_price = pos['entry_price']
        size = pos['size']
        
        if pos['type'] == 'LONG':
            pnl = (exit_price - entry_price) * size
        else:  # SHORT
            pnl = (entry_price - exit_price) * size
        
        # 수수료 차감 (0.04% = 매수 0.02% + 매도 0.02%)
        fee = (entry_price * size * 0.0002) + (exit_price * size * 0.0002)
        pnl -= fee
        
        # 자본 업데이트
        self.capital += pnl
        
        # 통계 업데이트
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        self.total_profit += pnl
        
        # 최대 낙폭 계산
        if self.capital > self.peak_equity:
            self.peak_equity = self.capital
        drawdown = (self.peak_equity - self.capital) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # 거래 기록
        trade_record = {
            'entry_time': pos['entry_time'],
            'exit_time': timestamp,
            'type': pos['type'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'pnl_pct': (pnl / pos['entry_capital']) * 100,
            'duration': (timestamp - pos['entry_time']).total_seconds() / 60  # 분 단위
        }
        self.positions.append(trade_record)
        
        # 백테스팅 중에는 청산 로그를 출력하지 않음 (결과 보고서에서 확인)
        
        self.current_position = None
    
    def check_stop_loss_take_profit(self, current_price, timestamp):
        """손절/익절 확인"""
        if not self.current_position:
            return False
        
        pos = self.current_position
        stop_loss = pos.get('stop_loss')
        take_profit = pos.get('take_profit')
        
        # 손절 확인
        if stop_loss:
            if pos['type'] == 'LONG' and current_price <= stop_loss:
                # 백테스팅 중에는 손절 로그를 출력하지 않음
                self.close_position(stop_loss, timestamp)
                return True
            elif pos['type'] == 'SHORT' and current_price >= stop_loss:
                # 백테스팅 중에는 손절 로그를 출력하지 않음
                self.close_position(stop_loss, timestamp)
                return True
        
        # 익절 확인
        if take_profit:
            if pos['type'] == 'LONG' and current_price >= take_profit:
                # 백테스팅 중에는 익절 로그를 출력하지 않음
                self.close_position(take_profit, timestamp)
                return True
            elif pos['type'] == 'SHORT' and current_price <= take_profit:
                # 백테스팅 중에는 익절 로그를 출력하지 않음
                self.close_position(take_profit, timestamp)
                return True
        
        return False
    
    def run_backtest(self, start_date=None, end_date=None):
        """백테스팅 실행"""
        logger.info("=" * 80)
        logger.info("🚀 백테스팅 시작")
        logger.info("=" * 80)
        
        # 1년치 데이터 로드 또는 수집
        eth_data = self.fetch_1year_data(config.ETH_SYMBOL, config.TIMEFRAME, use_cache=True)
        btc_data = self.fetch_1year_data(config.BTC_SYMBOL, config.TIMEFRAME, use_cache=True)
        
        if eth_data is None or btc_data is None:
            logger.error("데이터 수집 실패")
            return
        
        # 데이터 기간 필터링
        if start_date:
            eth_data = eth_data[eth_data.index >= start_date]
            btc_data = btc_data[btc_data.index >= start_date]
        if end_date:
            eth_data = eth_data[eth_data.index <= end_date]
            btc_data = btc_data[btc_data.index <= end_date]
        
        logger.info(f"백테스팅 기간: {eth_data.index[0]} ~ {eth_data.index[-1]}")
        logger.info(f"총 캔들 수: {len(eth_data)}개")
        
        # TradingBot 초기화 (백테스팅 모드: API 호출 없이 가상 거래)
        bot = TradingBot(backtest_mode=True)
        
        # 백테스팅 중에는 trading_bot의 로깅을 억제 (WARNING 이상만)
        trading_bot_logger = logging.getLogger('trading_bot')
        strategies_logger = logging.getLogger('strategies')
        original_trading_bot_level = trading_bot_logger.level
        original_strategies_level = strategies_logger.level
        trading_bot_logger.setLevel(logging.WARNING)
        strategies_logger.setLevel(logging.WARNING)
        
        # 시뮬레이션 시작 (3분봉 단위로 진행)
        logger.info("")
        logger.info("📊 백테스팅 시뮬레이션 시작...")
        logger.info("-" * 80)
        
        # 슬라이딩 윈도우로 데이터 업데이트하며 진행
        window_size = 1500  # lookback period
        total_candles = len(eth_data)
        last_progress = -1  # 진행률 추적
        
        for i in range(window_size, total_candles):
            try:
                # 현재 시점의 데이터 슬라이딩 윈도우
                current_eth = eth_data.iloc[i-window_size:i+1]
                current_btc = btc_data.iloc[i-window_size:i+1]
                
                # DataCollector에 데이터 설정
                bot.data_collector.eth_data = current_eth
                bot.data_collector.btc_data = current_btc
                
                current_timestamp = eth_data.index[i]
                current_price = float(eth_data.iloc[i]['close'])
                
                # 손절/익절 확인
                if self.check_stop_loss_take_profit(current_price, current_timestamp):
                    # 자산 곡선 업데이트
                    equity = self.capital
                    self.equity_curve.append({
                        'timestamp': current_timestamp,
                        'equity': equity,
                        'position': None
                    })
                    continue
                
                # 시장 모드 판단 및 전략 분석
                market_mode = bot.detect_market_mode()
                bot.current_market_mode = market_mode
                
                if market_mode == 'NEUTRAL':
                    # Neutral 모드는 거래하지 않음
                    self.equity_curve.append({
                        'timestamp': current_timestamp,
                        'equity': self.capital,
                        'position': None
                    })
                    continue
                
                # 전략 분석
                if market_mode == 'TREND':
                    signals = bot._analyze_trend_mode()
                elif market_mode == 'RANGE':
                    signals = bot._analyze_range_mode()
                else:
                    signals = []
                
                # 신호 조합
                if signals:
                    final_signal = bot.combine_signals(signals)
                    if final_signal:
                        self.simulate_trade(final_signal, current_price, current_price, current_timestamp)
                
                # 자산 곡선 업데이트
                equity = self.capital
                if self.current_position:
                    pos = self.current_position
                    if pos['type'] == 'LONG':
                        unrealized_pnl = (current_price - pos['entry_price']) * pos['size']
                    else:
                        unrealized_pnl = (pos['entry_price'] - current_price) * pos['size']
                    equity += unrealized_pnl
                
                self.equity_curve.append({
                    'timestamp': current_timestamp,
                    'equity': equity,
                    'position': self.current_position['type'] if self.current_position else None
                })
                
                # 진행 상황 출력 (1% 단위, 같은 줄에서 업데이트)
                progress = int(((i - window_size) / (total_candles - window_size)) * 100)
                if progress != last_progress:
                    print(f"\r⏳ 진행률: {progress}% | 자본: ${self.capital:,.2f} | 거래 수: {self.total_trades}건", end='', flush=True)
                    last_progress = progress
                
            except Exception as e:
                logger.error(f"백테스팅 중 오류 (인덱스 {i}): {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # 마지막 포지션 청산
        if self.current_position:
            final_price = float(eth_data.iloc[-1]['close'])
            final_timestamp = eth_data.index[-1]
            self.close_position(final_price, final_timestamp)
        
        # 로깅 레벨 복원
        trading_bot_logger.setLevel(original_trading_bot_level)
        strategies_logger.setLevel(original_strategies_level)
        
        # 진행률 100% 표시
        print(f"\r✅ 진행률: 100% | 자본: ${self.capital:,.2f} | 거래 수: {self.total_trades}건")
        logger.info("")
        logger.info("백테스팅 시뮬레이션 완료!")
        
        # 결과 출력
        self.print_results()
    
    def print_results(self):
        """백테스팅 결과 보고서 출력"""
        print("\n" + "=" * 100)
        print("📊 백테스팅 결과 보고서")
        print("=" * 100)
        
        # 기본 통계
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_profit = self.total_profit / self.total_trades if self.total_trades > 0 else 0
        
        # 수익률 색상 표시
        return_color = "🟢" if total_return > 0 else "🔴" if total_return < 0 else "⚪"
        
        print(f"\n💰 자본 현황")
        print(f"   초기 자본: ${self.initial_capital:,.2f}")
        print(f"   최종 자본: ${self.capital:,.2f}")
        print(f"   {return_color} 총 수익률: {total_return:+.2f}%")
        print(f"   총 손익: ${self.total_profit:+,.2f}")
        
        print(f"\n📈 거래 통계")
        print(f"   총 거래 수: {self.total_trades}건")
        print(f"   승리 거래: {self.winning_trades}건")
        print(f"   손실 거래: {self.losing_trades}건")
        print(f"   승률: {win_rate:.2f}%")
        print(f"   평균 손익: ${avg_profit:+,.2f}")
        print(f"   최대 낙폭: {self.max_drawdown*100:.2f}%")
        
        # 거래별 상세 통계
        if self.positions:
            winning_pnls = [p['pnl'] for p in self.positions if p['pnl'] > 0]
            losing_pnls = [p['pnl'] for p in self.positions if p['pnl'] <= 0]
            
            print(f"\n📊 상세 통계")
            if winning_pnls:
                avg_win = np.mean(winning_pnls)
                max_win = max(winning_pnls)
                print(f"   평균 승리: ${avg_win:+,.2f}")
                print(f"   최대 승리: ${max_win:+,.2f}")
            
            if losing_pnls:
                avg_loss = np.mean(losing_pnls)
                max_loss = min(losing_pnls)
                print(f"   평균 손실: ${avg_loss:+,.2f}")
                print(f"   최대 손실: ${max_loss:+,.2f}")
            
            if winning_pnls and losing_pnls:
                profit_factor = abs(sum(winning_pnls) / sum(losing_pnls)) if sum(losing_pnls) != 0 else float('inf')
                print(f"   Profit Factor: {profit_factor:.2f}")
            
            # 거래 기간 통계
            durations = [p['duration'] for p in self.positions]
            if durations:
                avg_duration = np.mean(durations)
                print(f"   평균 보유 기간: {avg_duration:.1f}분")
            
            # 월별 수익률 분석
            if self.equity_curve and len(self.equity_curve) > 0:
                try:
                    df_equity = pd.DataFrame(self.equity_curve)
                    df_equity['timestamp'] = pd.to_datetime(df_equity['timestamp'])
                    df_equity['month'] = df_equity['timestamp'].dt.to_period('M')
                    monthly_returns = df_equity.groupby('month')['equity'].agg(['first', 'last'])
                    monthly_returns['return'] = ((monthly_returns['last'] - monthly_returns['first']) / monthly_returns['first']) * 100
                    
                    print(f"\n📅 월별 수익률")
                    for month, row in monthly_returns.iterrows():
                        month_return = row['return']
                        month_color = "🟢" if month_return > 0 else "🔴" if month_return < 0 else "⚪"
                        print(f"   {month}: {month_color} {month_return:+.2f}%")
                except Exception as e:
                    # 월별 분석 실패 시 무시
                    pass
        
        print("\n" + "=" * 100)
        
        # 거래 내역 CSV 저장
        if self.positions:
            df_trades = pd.DataFrame(self.positions)
            df_trades.to_csv('logs/backtest_trades.csv', index=False, encoding='utf-8-sig')
            print(f"💾 거래 내역 저장: logs/backtest_trades.csv")
        
        # 자산 곡선 CSV 저장
        if self.equity_curve:
            df_equity = pd.DataFrame(self.equity_curve)
            df_equity.to_csv('logs/backtest_equity.csv', index=False, encoding='utf-8-sig')
            print(f"💾 자산 곡선 저장: logs/backtest_equity.csv")
        
        print("=" * 100 + "\n")
        
        # 로그에도 기록
        logger.info("=" * 80)
        logger.info("📊 백테스팅 결과")
        logger.info("=" * 80)
        logger.info(f"초기 자본: ${self.initial_capital:,.2f}")
        logger.info(f"최종 자본: ${self.capital:,.2f}")
        logger.info(f"총 수익률: {total_return:.2f}%")
        logger.info(f"총 거래 수: {self.total_trades}건")
        logger.info(f"승률: {win_rate:.2f}%")
        logger.info(f"최대 낙폭: {self.max_drawdown*100:.2f}%")


if __name__ == '__main__':
    backtester = Backtester(initial_capital=10000)
    backtester.run_backtest()

