import os
import sys
import asyncio
import time
import logging
import gc
import json
import re
import unicodedata
import numpy as np
import pandas as pd
import torch
import ccxt.async_support as ccxt
from datetime import datetime

# 💡 [1. 경로 설정]
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_PATHS = [
    _THIS_DIR,
    os.path.join(_THIS_DIR, "models"),
    os.path.join(_THIS_DIR, "timesfm"),
    os.path.join(_THIS_DIR, "uni2ts", "src"),
    os.path.join(_THIS_DIR, "strategies"),
    os.path.join(_THIS_DIR, "ensemble"), 
]
for p in TARGET_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from core.feature_engineering import FeatureEngineer
from ensemble.ensemble_router import (
    TFTForecaster, MacroHFTForecaster, ChronosForecaster, 
    KronosForecaster, TimesFMForecaster, MoiraiForecaster
)

from ensemble.train_rl_agent import IQNTrader
from strategies.elite_builder import EliteSignals, row_to_market_row


class Colors:
    GREEN, RED, YELLOW, CYAN, RESET, BOLD = '\033[92m', '\033[91m', '\033[93m', '\033[96m', '\033[0m', '\033[1m'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("LiveBot")

# ════════════════════════════════════════════════════════════════
# 1. 데이터 수집기
# ════════════════════════════════════════════════════════════════
class BinanceLiveFetcher:
    def __init__(self, symbol='ETHUSDT', timeframe='5m', limit=2500):
        self.symbol = symbol.replace('/', '') 
        self.timeframe = timeframe
        self.limit = limit
        self.exchange = ccxt.binance({'options': {'defaultType': 'future'}})

    def load_local_data(self):
        try:
            eth_df = pd.read_csv('data/test/eth_test_data.csv')
            btc_df = pd.read_csv('data/test/btc_test_data.csv')
            for df in [eth_df, btc_df]:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                cols = df.columns.drop('timestamp')
                df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            return eth_df, btc_df
        except Exception as e:
            logger.error(f"로컬 데이터 로드 실패: {e}")
            return None, None

    async def fetch_klines_raw(self, symbol, target_limit):
        all_klines = []
        last_end_time = None
        while len(all_klines) < target_limit:
            params = {'symbol': symbol, 'interval': self.timeframe, 'limit': 1000}
            if last_end_time: params['endTime'] = last_end_time - 1
            klines = await self.exchange.fapiPublicGetKlines(params)
            if not klines: break
            all_klines = klines + all_klines
            last_end_time = klines[0][0]
            if len(klines) < 1000: break
        return all_klines[-target_limit:]

    async def fetch_ancillary_data(self, limit=500):
        tasks = [
            self.exchange.fapiDataGetOpenInterestHist({'symbol': self.symbol, 'period': self.timeframe, 'limit': limit}),
            self.exchange.fapiDataGetTopLongShortAccountRatio({'symbol': self.symbol, 'period': self.timeframe, 'limit': limit}),
            self.exchange.fapiDataGetGlobalLongShortAccountRatio({'symbol': self.symbol, 'period': self.timeframe, 'limit': limit}),
            self.exchange.fapiPublicGetFundingRate({'symbol': self.symbol, 'limit': limit})
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _process_to_df(self, eth_klines, btc_klines, ancillary_results):
        eth_df = pd.DataFrame(eth_klines).iloc[:, :11]
        eth_df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
        eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'], unit='ms')
        num_cols = eth_df.columns.drop('timestamp')
        eth_df[num_cols] = eth_df[num_cols].apply(pd.to_numeric, errors='coerce')

        btc_df = pd.DataFrame(btc_klines).iloc[:, [0, 4, 5, 7]]
        btc_df.columns = ['timestamp', 'close_btc', 'volume_btc', 'quote_volume_btc']
        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
        btc_df[btc_df.columns.drop('timestamp')] = btc_df[btc_df.columns.drop('timestamp')].apply(pd.to_numeric, errors='coerce')

        if ancillary_results:
            mappings = [(0, 'sumOpenInterestValue', 'sum_open_interest_value'), (1, 'longShortRatio', 'sum_toptrader_long_short_ratio'), (2, 'longShortRatio', 'count_long_short_ratio'), (3, 'fundingRate', 'last_funding_rate')]
            for idx, key, new_name in mappings:
                res = ancillary_results[idx]
                if isinstance(res, list) and len(res) > 0:
                    try:
                        temp_df = pd.DataFrame(res)
                        t_col = next((c for c in ['timestamp', 'fundingTime', 'time'] if c in temp_df.columns), None)
                        if t_col and key in temp_df.columns:
                            subset = temp_df[[t_col, key]].rename(columns={t_col: 'timestamp', key: new_name})
                            subset['timestamp'] = pd.to_datetime(subset['timestamp'], unit='ms')
                            subset[new_name] = pd.to_numeric(subset[new_name], errors='coerce')
                            eth_df = pd.merge_asof(eth_df.sort_values('timestamp'), subset.sort_values('timestamp'), on='timestamp', direction='nearest')
                    except Exception: pass
        return eth_df.ffill().bfill(), btc_df

    async def fetch_initial_data(self):
        eth_klines = await self.fetch_klines_raw(self.symbol, self.limit)
        btc_klines = await self.fetch_klines_raw('BTCUSDT', self.limit)
        ancillary = await self.fetch_ancillary_data(500)
        return self._process_to_df(eth_klines, btc_klines, ancillary)

    async def fetch_latest_patch(self):
        eth_klines = await self.exchange.fapiPublicGetKlines({'symbol': self.symbol, 'interval': self.timeframe, 'limit': 5})
        btc_klines = await self.exchange.fapiPublicGetKlines({'symbol': 'BTCUSDT', 'interval': self.timeframe, 'limit': 5})
        ancillary = await self.fetch_ancillary_data(5)
        return self._process_to_df(eth_klines, btc_klines, ancillary)


# ════════════════════════════════════════════════════════════════
# 2. 6대 앙상블 모델 예측기
# ════════════════════════════════════════════════════════════════
class EnsemblePredictor:
    def __init__(self):
        self.models = {
            'TFT': TFTForecaster(),
            'MacroHFT': MacroHFTForecaster(),
            'Chronos': ChronosForecaster(),
            'Kronos': KronosForecaster(),
            'TimesFM': TimesFMForecaster(),
            'Moirai': MoiraiForecaster()
        }
        self.model_order = ['TFT', 'MacroHFT', 'Chronos', 'Kronos', 'TimesFM', 'Moirai']

    async def predict_all_async(self, df: pd.DataFrame):
        loop = asyncio.get_event_loop()
        def _run_inference(m):
            if not getattr(m, 'available', False): return None
            try: return m.predict(df, horizon=6) 
            except Exception: return None

        tasks = [loop.run_in_executor(None, _run_inference, self.models[name]) for name in self.model_order]
        results = await asyncio.gather(*tasks)
        
        preds, confs = [], []
        for i, res in enumerate(results):
            p_val, c_val = 0.0, 0.0
            if res is not None and getattr(res, 'median', None) is not None:
                trajectory = np.array(res.median[-1], dtype=np.float32) 
                if len(trajectory) > 1:
                    slope = float(np.polyfit(np.arange(len(trajectory)), trajectory, 1)[0])
                    delta = trajectory[-1] - trajectory[0]
                    if slope > 0 and delta > 0: p_val = 1.0  
                    elif slope < 0 and delta < 0: p_val = -1.0 
                else: p_val = float(trajectory.mean())
                c_val = float(res.confidence[-1].mean())
            preds.append(p_val)
            confs.append(c_val)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return np.array(preds), np.array(confs)


# ════════════════════════════════════════════════════════════════
# 3. 실전용 강화학습 AI 라우터 (31-Dim IQN Agent)
# ════════════════════════════════════════════════════════════════
class IQNLiveRouter:
    def __init__(self, model_path='data/router/hybrid_iqn_elite_best.pt'):
        self.model_names = ['TFT(5m)', 'Macro(30m)', 'Chronos', 'Kronos', 'TimesFM', 'Moirai']
        self.iqn = IQNTrader(model_path=model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.elite_extractor = EliteSignals()
        self.pos, self.entry_price, self.hold_count = 0.0, 0.0, 0
        self.peak_equity, self.current_equity = 1.0, 1.0
        logger.info(f"{Colors.GREEN}✅ IQN 강화학습 라우터 로드 완료 (31-Dim State){Colors.RESET}")

    def get_signal(self, df_buffer, preds, confs):
        last_row = df_buffer.iloc[-1]
        prev_row = df_buffer.iloc[-2]
        smf_std = df_buffer['smart_money_flow'].std() if 'smart_money_flow' in df_buffer.columns else 1.0
        
        cur_mr = row_to_market_row(last_row)
        prev_mr = row_to_market_row(prev_row)
        elite_sigs = self.elite_extractor.compute_all(current=cur_mr, prev=prev_mr, smf_std=smf_std)
        
        features = {}
        model_keys = ['tft', 'macro', 'chronos', 'kronos', 'timesfm', 'moirai']
        for i, mk in enumerate(model_keys):
            features[f'pred_{mk}'] = preds[i]
            features[f'conf_{mk}'] = confs[i]
        features.update(elite_sigs)
        
        unr = 0.0
        if self.pos != 0.0:
            cp = last_row['close']
            unr = (cp - self.entry_price) / self.entry_price if self.pos > 0 else (self.entry_price - cp) / self.entry_price
            self.current_equity = 1.0 + unr
            if self.current_equity > self.peak_equity: self.peak_equity = self.current_equity
        else: self.current_equity, self.peak_equity = 1.0, 1.0
            
        mdd = min((self.current_equity / self.peak_equity) - 1.0, 0.0) if self.peak_equity > 0 else 0.0
        pos_info = {'position': self.pos, 'leverage': 1.0 if self.pos != 0 else 0.0, 'unrealized': unr, 'mdd': mdd, 'hold_norm': min(self.hold_count / 100.0, 1.0)}
        
        action, q_vals = self.iqn.decide(features, pos_info)
        direction = [0, 1, -1][action]
        
        if direction != 0:
            if self.pos == 0.0:
                self.pos, self.entry_price, self.hold_count = float(direction), last_row['close'], 0
            elif (direction == 1 and self.pos == -1.0) or (direction == -1 and self.pos == 1.0):
                self.pos, self.entry_price, self.hold_count = 0.0, 0.0, 0
        else:
            if self.pos != 0.0: self.hold_count += 1
        return direction, q_vals, elite_sigs, action

    def print_dashboard(self, current_price, preds, confs, direction, q_vals, elite_sigs, timestamp, last_features, action):
        BOX_W = 84 
        def strip_ansi(s): return re.sub(r'\x1b\[[0-9;]*m', '', s)
        def vw(s):
            w = 0
            for c in strip_ansi(s):
                if unicodedata.east_asian_width(c) in ('F', 'W') or ord(c) > 0x2E7F: w += 2
                else: w += 1
            return w
        def pad_line(content):
            w = vw(content)
            return f"{Colors.BOLD}║{Colors.RESET} {content}{' ' * max(0, BOX_W - w)} {Colors.BOLD}║{Colors.RESET}"
        def align_cols(col1, col2, c1_vw=40):
            w1 = vw(col1)
            return f"{col1}{' ' * max(0, c1_vw - w1)} │ {col2}"
        def draw_bar(val, max_val=1.0, width=8, color=Colors.CYAN):
            filled = int(round(min(max(abs(val) / max_val, 0.0), 1.0) * width))
            return f"{color}{'█' * filled}{Colors.RESET}{'░' * (width - filled)}"
        def sep(): print(f"{Colors.BOLD}╠{'═'*86}╣{Colors.RESET}")

        ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S KST')
        regime_str = f"{Colors.GREEN}[추세장]{Colors.RESET}" if last_features.get('regime_trending', 0) > 0 else f"{Colors.YELLOW}[횡보장]{Colors.RESET}"
        chop_val = last_features.get('chop_index', 50.0)
        chop_bar = draw_bar(chop_val, 100, 8, Colors.RED if chop_val > 61.8 else Colors.GREEN)
        vol_z = last_features.get('volatility_z', 0.0)
        vol_bar = draw_bar(abs(vol_z), 3.0, 6, Colors.RED if vol_z > 1.0 else Colors.CYAN)

        sm_str = f"{Colors.GREEN}매집(+){Colors.RESET}" if last_features.get('smart_money_flow', 0) > 0.5 else (f"{Colors.RED}분배(-){Colors.RESET}" if last_features.get('smart_money_flow', 0) < -0.5 else "관망(=)")
        taker_str = f"{Colors.GREEN}순매수(+){Colors.RESET}" if last_features.get('net_taker_ratio', 0) > 0 else (f"{Colors.RED}순매도(-){Colors.RESET}" if last_features.get('net_taker_ratio', 0) < 0 else "중립(=)")
        whale_str = f"{Colors.GREEN}고래 주도{Colors.RESET}" if last_features.get('whale_retail_ratio', 1) > 0.5 else f"{Colors.RED}개미 주도{Colors.RESET}"
        fund_str = f"{Colors.RED}롱과열{Colors.RESET}" if last_features.get('funding_z_score', 0) > 1.0 else (f"{Colors.GREEN}숏과열{Colors.RESET}" if last_features.get('funding_z_score', 0) < -1.0 else "정상")

        print(f"\n{Colors.BOLD}╔{'═'*86}╗{Colors.RESET}")
        print(pad_line(f"TIME {Colors.CYAN}{ts_str}{Colors.RESET}  |  ETH {Colors.YELLOW}${current_price:,.2f}{Colors.RESET}"))
        sep()
        print(pad_line(f"** 시장 환경 및 미시구조 (Market Intelligence) **"))
        print(pad_line(align_cols(f" 시장체제 : {regime_str}", f" 휩소지수 : [{chop_bar}] {chop_val:4.1f}")))
        print(pad_line(align_cols(f" 변 동 성 : [{vol_bar}] Z:{vol_z:>+4.1f}", f" 펀딩비율 : {fund_str}")))
        print(pad_line(f" 오더플로 : 🧠스마트머니[{sm_str}]  ⚡시장가[{taker_str}]  🐋주도세력[{whale_str}]"))
        sep()
        print(pad_line("** 6대 파운데이션 모델 예측 (Sub-Brains) **"))
        for i in range(0, len(self.model_names), 2):
            def fmt_m(idx):
                if idx >= len(preds): return ""
                p, c = preds[idx], confs[idx]
                d = f"{Colors.GREEN}UP  {Colors.RESET}" if p > 0 else (f"{Colors.RED}DOWN{Colors.RESET}" if p < 0 else f"{Colors.YELLOW}FLAT{Colors.RESET}")
                return f" {self.model_names[idx]:<10s} [{d}] {draw_bar(c, 1.0, 6, Colors.GREEN if c >= 0.7 else Colors.YELLOW)} {c*100:4.1f}%"
            print(pad_line(align_cols(fmt_m(i), fmt_m(i+1), 42)))
        sep()
        print(pad_line("** 11대 엘리트 퀀트 전략 (RL Decision Matrix) **"))
        sig_list = list(elite_sigs.items())
        for i in range(0, len(sig_list), 2):
            def fmt_s(k, v):
                c = Colors.GREEN if v > 0 else (Colors.RED if v < 0 else Colors.RESET)
                return f" {k.replace('sig_',''):<11s} [{draw_bar(abs(v),1.0,5,c)}] {c}{'상승' if v>0 else ('하락' if v<0 else '중립')}({v:>+4.2f}){Colors.RESET}"
            print(pad_line(align_cols(fmt_s(*sig_list[i]), fmt_s(*sig_list[i+1]) if i+1 < len(sig_list) else "", 42)))
        sep()
        print(pad_line("** IQN 강화학습 에이전트 (Master Brain) **"))
        def f_q(idx, cur, lbl, val):
            if idx == cur: return f"{Colors.CYAN}{Colors.BOLD}▶[{lbl}:{val:>+6.3f}]{Colors.RESET}"
            return f"  [{lbl}:{val:>+6.3f}] "
        print(pad_line(f" 기대수익(Q) : {f_q(0,action,'HOLD',q_vals[0])} {f_q(1,action,'LONG',q_vals[1])} {f_q(2,action,'SHRT',q_vals[2])}"))
        pnl_pct = (self.current_equity - 1.0) * 100
        pc = Colors.GREEN if pnl_pct > 0 else (Colors.RED if pnl_pct < 0 else Colors.RESET)
        st = f"{Colors.YELLOW}무포지션{Colors.RESET}" if self.pos == 0.0 else (f"{Colors.GREEN}롱(LONG) 보유{Colors.RESET}" if self.pos > 0 else f"{Colors.RED}숏(SHORT)보유{Colors.RESET}")
        print(pad_line(align_cols(f" 에이전트상태 : {st}", f" 보유:{self.hold_count:3d}봉  수익:{pc}{pnl_pct:>+5.2f}%{Colors.RESET}", 40)))
        dec = f"{Colors.GREEN}🟩 LONG 진입/유지{Colors.RESET}" if direction == 1 else (f"{Colors.RED}🟥 SHORT 진입/유지{Colors.RESET}" if direction == -1 else f"{Colors.YELLOW}🟨 HOLD 관망/대기{Colors.RESET}")
        print(pad_line(f" 최종결단 : {dec}"))
        print(f"{Colors.BOLD}╚{'═'*86}╝{Colors.RESET}\n")

# ════════════════════════════════════════════════════════════════
# 4. 메인 루프
# ════════════════════════════════════════════════════════════════
async def main():
    fetcher, fe_engine, ensemble = BinanceLiveFetcher(), FeatureEngineer(), EnsemblePredictor()
    router = IQNLiveRouter(model_path='data/router/hybrid_iqn_elite_best.pt')
    try:
        logger.info("⏳ 거래소 초기 데이터 수집 중..."); eth_buffer, btc_buffer = await fetcher.fetch_initial_data()
        if eth_buffer is None: return
        first_run = True
        while True:
            if not first_run:
                now = time.time(); wait_sec = int(max(0, (now - (now % 300) + 300 + 2) - now))
                for r in range(wait_sec, 0, -1):
                    sys.stdout.write(f"\r{Colors.CYAN}⏳ 다음 5분봉까지 대기 중... ({r}초 남음)   {Colors.RESET}"); sys.stdout.flush(); await asyncio.sleep(1)
                print(); logger.info("🔄 최신 데이터를 갱신합니다.")
                new_eth, new_btc = await fetcher.fetch_latest_patch()
                eth_buffer = pd.concat([eth_buffer, new_eth]).drop_duplicates('timestamp').tail(2500)
                btc_buffer = pd.concat([btc_buffer, new_btc]).drop_duplicates('timestamp').tail(2500)
            else: logger.info(f"{Colors.GREEN}🚀 봇 가동 시작!{Colors.RESET}"); first_run = False
            
            processed_df = fe_engine.process(eth_buffer, btc_buffer)
            preds, confs = await ensemble.predict_all_async(processed_df)
            direction, q_vals, elite_sigs, action = router.get_signal(processed_df, preds, confs)            
            current_time_kst = eth_buffer['timestamp'].iloc[-1] + pd.Timedelta(hours=9)
            router.print_dashboard(eth_buffer['close'].iloc[-1], preds, confs, direction, q_vals, elite_sigs, current_time_kst, processed_df.iloc[-1].to_dict(), action)
    finally: await fetcher.exchange.close()

if __name__ == "__main__": asyncio.run(main())